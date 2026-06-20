"""CSV trace loading and preprocessing."""

import os
from typing import TYPE_CHECKING

import numpy as np

from .trace import RadioTrace

if TYPE_CHECKING:
    from ho_optim_drl.config import Config


def load_radio_trace(
    config: "Config",
    data_dir: str,
    rsrp_file: str,
    sinr_file: str,
) -> RadioTrace:
    """
    Load one trajectory into three explicit protocol streams.

    RSRP is L3-filtered for A3. SINR is not resampled or L3-filtered:
    the raw stream is used independently for estimated rate and the RLM proxy.
    """
    trace_id = _validate_trace_pair_names(rsrp_file, sinr_file)
    rsrp_raw, rsrp_time_ms = _load_csv_measurements(
        os.path.join(data_dir, rsrp_file)
    )
    sinr_raw, sinr_time_ms = _load_csv_measurements(
        os.path.join(data_dir, sinr_file)
    )
    if rsrp_raw.shape != sinr_raw.shape:
        raise ValueError(
            f"RSRP/SINR shape mismatch: {rsrp_file} {rsrp_raw.shape}, "
            f"{sinr_file} {sinr_raw.shape}"
        )
    if not np.array_equal(rsrp_time_ms, sinr_time_ms):
        raise ValueError("RSRP and SINR time columns do not match.")
    _validate_sampling_interval(rsrp_time_ms, config.delta_t_ms, rsrp_file)

    rsrp_a3 = l3_filtering(rsrp_raw, config.l3_filter_w)
    return RadioTrace(
        rsrp_a3_db=rsrp_a3.T,
        sinr_rate_db=sinr_raw.T,
        sinr_rlm_db=sinr_raw.T,
        delta_t_ms=config.delta_t_ms,
        trace_id=trace_id,
    )


def _validate_trace_pair_names(rsrp_file: str, sinr_file: str) -> str:
    rsrp_stem = os.path.splitext(os.path.basename(rsrp_file))[0]
    sinr_stem = os.path.splitext(os.path.basename(sinr_file))[0]
    if not rsrp_stem.startswith("rsrp_") or not sinr_stem.startswith("sinr_"):
        raise ValueError("Trace filenames must start with 'rsrp_' and 'sinr_'.")

    rsrp_trace_id = rsrp_stem.removeprefix("rsrp_")
    sinr_trace_id = sinr_stem.removeprefix("sinr_")
    if rsrp_trace_id != sinr_trace_id:
        raise ValueError(
            f"RSRP/SINR trajectory mismatch: {rsrp_trace_id!r} and "
            f"{sinr_trace_id!r}."
        )
    return rsrp_trace_id


def _load_csv_measurements(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load timestep-by-BS measurements and their time column."""
    with open(path, encoding="utf-8") as csv_file:
        columns = csv_file.readline().strip().split(",")

    if len(columns) < 3 or columns[:2] != ["sample", "time_ms"]:
        raise ValueError(
            f"{path} must start with columns 'sample,time_ms' and contain "
            "at least one BS column."
        )
    expected_bs_columns = [f"bs_{index}" for index in range(len(columns) - 2)]
    if columns[2:] != expected_bs_columns:
        raise ValueError(
            f"{path} has invalid BS columns {columns[2:]}; "
            f"expected {expected_bs_columns}."
        )

    data = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    if not np.all(np.isfinite(data)):
        raise ValueError(f"{path} contains NaN or infinite values.")
    sample = data[:, 0]
    time_ms = data[:, 1]
    if not np.array_equal(sample, np.arange(len(data))):
        raise ValueError(f"{path} has a non-contiguous sample column.")
    if len(data) > 1 and not np.allclose(np.diff(time_ms), np.diff(time_ms)[0]):
        raise ValueError(f"{path} has a non-uniform time_ms column.")
    return data[:, 2:], time_ms


def _validate_sampling_interval(
    time_ms: np.ndarray, expected_delta_t_ms: int, filename: str
) -> None:
    expected_time_ms = np.arange(len(time_ms)) * expected_delta_t_ms
    if not np.allclose(time_ms, expected_time_ms):
        actual_delta_t_ms = (
            float(time_ms[1] - time_ms[0]) if len(time_ms) > 1 else None
        )
        raise ValueError(
            f"{filename} must start at 0 ms and use {expected_delta_t_ms} ms "
            f"samples; observed interval is {actual_delta_t_ms}."
        )


def l3_filtering(measurements: np.ndarray, w: float = 0.1) -> np.ndarray:
    """Apply exponential L3 smoothing along the first, time, dimension."""
    measurements = np.asarray(measurements, dtype=np.float64)
    if measurements.ndim != 2 or len(measurements) == 0:
        raise ValueError("measurements must be a non-empty timestep-by-BS array.")
    if not 0 < w <= 1:
        raise ValueError("L3 filter weight must be in (0, 1].")

    filtered_values = np.empty_like(measurements)
    filtered_values[0] = measurements[0]
    for tic in range(1, len(measurements)):
        filtered_values[tic] = (
            (1 - w) * filtered_values[tic - 1] + w * measurements[tic]
        )
    return filtered_values
