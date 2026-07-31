"""Normalize UAV simulation CSV traces for the handover data loader.

Input files have names such as ``ue0_rsrp_30kmh_0.csv`` and columns such as
``Step,bs0,...``.  This utility writes new files using the repository format:
``rsrp_30kmh_0.csv`` / ``sinr_30kmh_0.csv`` with
``sample,time_ms,bs_0,...`` columns.  Input files are never modified.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np


_UAV_FILE_PATTERN = re.compile(
    r"ue(?P<ue>\d+)_(?P<kind>rsrp|sinr)_(?P<speed>\d+)kmh_(?P<trace>\d+)\.csv"
)


@dataclass(frozen=True, order=True)
class _SourcePair:
    ue: int
    speed_kmh: int
    source_trace_id: int
    rsrp_path: Path
    sinr_path: Path


def _discover_source_pairs(input_dir: Path) -> list[_SourcePair]:
    discovered: dict[tuple[int, int, int], dict[str, Path]] = {}
    for path in input_dir.glob("*.csv"):
        match = _UAV_FILE_PATTERN.fullmatch(path.name)
        if match is None:
            continue
        groups = match.groupdict()
        key = (int(groups["ue"]), int(groups["speed"]), int(groups["trace"]))
        discovered.setdefault(key, {})[groups["kind"]] = path

    if not discovered:
        raise FileNotFoundError(
            f"No UAV CSV files matching {_UAV_FILE_PATTERN.pattern!r} in {input_dir}"
        )

    incomplete = {
        key: sorted({"rsrp", "sinr"} - set(paths))
        for key, paths in discovered.items()
        if set(paths) != {"rsrp", "sinr"}
    }
    if incomplete:
        raise ValueError(f"Incomplete UAV RSRP/SINR trace pairs: {incomplete}")

    return [
        _SourcePair(
            ue=ue,
            speed_kmh=speed_kmh,
            source_trace_id=source_trace_id,
            rsrp_path=paths["rsrp"],
            sinr_path=paths["sinr"],
        )
        for (ue, speed_kmh, source_trace_id), paths in sorted(discovered.items())
    ]


def _load_uav_csv(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    # ``utf-8-sig`` accepts both ordinary UTF-8 and a UTF-8 BOM.  The UAV
    # exports use the latter, which would otherwise make the first column
    # appear as ``\\ufeffStep``.
    with path.open(encoding="utf-8-sig") as csv_file:
        columns = [column.strip() for column in csv_file.readline().split(",")]

    if len(columns) < 2 or columns[0] != "Step":
        raise ValueError(f"{path} must start with 'Step' followed by BS columns.")
    expected_bs_columns = [f"bs{index}" for index in range(len(columns) - 1)]
    if columns[1:] != expected_bs_columns:
        raise ValueError(
            f"{path} has invalid BS columns {columns[1:]}; "
            f"expected {expected_bs_columns}."
        )

    data = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    if not np.all(np.isfinite(data)):
        raise ValueError(f"{path} contains NaN or infinite values.")
    steps = data[:, 0]
    if not np.array_equal(steps, np.arange(len(data))):
        raise ValueError(f"{path} has a non-contiguous Step column.")
    return steps.astype(np.int64), data[:, 1:], columns[1:]


def _write_normalized_csv(
    output_path: Path,
    steps: np.ndarray,
    measurements: np.ndarray,
    delta_t_ms: int,
    overwrite: bool,
) -> None:
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"{output_path} already exists. Pass --overwrite to replace it."
        )
    time_ms = steps * delta_t_ms
    csv_data = np.column_stack((steps, time_ms, measurements))
    header = ",".join(
        ["sample", "time_ms"]
        + [f"bs_{index}" for index in range(measurements.shape[1])]
    )
    formats = ["%d", "%d"] + ["%.17g"] * measurements.shape[1]
    np.savetxt(
        output_path,
        csv_data,
        delimiter=",",
        header=header,
        comments="",
        fmt=formats,
    )


def convert_directory(
    input_dir: Path,
    output_dir: Path,
    delta_t_ms: int,
    overwrite: bool,
) -> int:
    if delta_t_ms <= 0:
        raise ValueError("delta_t_ms must be positive.")

    pairs = _discover_source_pairs(input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target_ids_by_speed: dict[int, int] = {}

    for index, pair in enumerate(pairs, start=1):
        rsrp_steps, rsrp_values, rsrp_columns = _load_uav_csv(pair.rsrp_path)
        sinr_steps, sinr_values, sinr_columns = _load_uav_csv(pair.sinr_path)
        if rsrp_values.shape != sinr_values.shape:
            raise ValueError(
                f"RSRP/SINR shape mismatch: {pair.rsrp_path.name} "
                f"{rsrp_values.shape}, {pair.sinr_path.name} {sinr_values.shape}."
            )
        if not np.array_equal(rsrp_steps, sinr_steps):
            raise ValueError(
                f"RSRP/SINR Step columns differ: {pair.rsrp_path.name} and "
                f"{pair.sinr_path.name}."
            )
        if rsrp_columns != sinr_columns:
            raise ValueError(
                f"RSRP/SINR BS columns differ: {pair.rsrp_path.name} and "
                f"{pair.sinr_path.name}."
            )

        target_trace_id = target_ids_by_speed.get(pair.speed_kmh, 0)
        target_ids_by_speed[pair.speed_kmh] = target_trace_id + 1
        rsrp_output = output_dir / f"rsrp_{pair.speed_kmh}kmh_{target_trace_id}.csv"
        sinr_output = output_dir / f"sinr_{pair.speed_kmh}kmh_{target_trace_id}.csv"
        _write_normalized_csv(
            rsrp_output, rsrp_steps, rsrp_values, delta_t_ms, overwrite
        )
        _write_normalized_csv(
            sinr_output, sinr_steps, sinr_values, delta_t_ms, overwrite
        )
        print(
            f"[{index:3d}/{len(pairs)}] UE {pair.ue}, "
            f"source trace {pair.source_trace_id} -> "
            f"{rsrp_output.name}, {sinr_output.name}"
        )

    return len(pairs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("data/UAV_sim"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/uav_processed")
    )
    parser.add_argument("--delta-t-ms", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    num_pairs = convert_directory(
        args.input_dir, args.output_dir, args.delta_t_ms, args.overwrite
    )
    print(f"Converted {num_pairs} UAV trace pairs into {args.output_dir}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
