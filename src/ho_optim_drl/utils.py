"""Utility functions for the project."""

import csv
import os
from typing import Any

import numpy as np


def get_sync_state(sinr_db: float, q_in_db: float, q_out_db: float):
    """
    Check if the SINR is within the QoS thresholds.

    Parameters
    ----------
    sinr_db : float
        SINR in dB.
    q_in_db : float
        Upper QoS threshold in dB. Indicates in-sync.
    q_out_db : float
        Lower QoS threshold in dB. Indicates out-of-sync.

    Returns
    -------
    int
        State of the SINR. 1: out-of-sync, 2: in hysteresis, 3: in-sync.
    """
    if sinr_db < q_out_db:
        return 1
    if sinr_db < q_in_db:
        return 2
    return 3


def clipnorm(ins: np.ndarray, clip_l: float, clip_h: float) -> np.ndarray:
    """
    Clip and normalize the input values.

    Parameters
    ----------
    ins : np.ndarray
        Input values.
    clip_l : float
        Lower clipping value.
    clip_h : float
        Upper clipping value.

    Returns
    -------
    np.ndarray
        Normalized values between 0 and 1.
    """
    outs = np.clip(ins, clip_l, clip_h)
    return (outs - clip_l) / (clip_h - clip_l)


def mwatt_to_dbm(mwatt: float | np.ndarray) -> float | np.ndarray:
    """
    Convert mW to dBm.

    Parameters
    ----------
    mwatt : float | np.ndarray
        Power in mW.

    Returns
    -------
    float | np.ndarray
        Power in dBm.
    """
    return 10 * np.log10(mwatt)


def dbm_to_mwatt(dbm: float | np.ndarray) -> float | np.ndarray:
    """
    Convert dBm to mW.

    Parameters
    ----------
    dbm : float | np.ndarray
        Power in dBm.

    Returns
    -------
    float | np.ndarray
        Power in mW.
    """
    return np.power(10, dbm / 10)


def get_result_container(speeds: list[int]) -> dict:
    """
    Get the result container for additional statistics.

    Parameters
    ----------
    speeds : list[int]
        List of speeds.

    Returns
    -------
    dict
        Result container.
    """
    bs_idxs = {}
    ho_idxs = {}
    rlf_idxs = {}
    sinr_connected = {}
    sinr_max = {}
    n_ho = {}
    n_ho_success = {}
    n_hof = {}
    n_pp = {}
    n_rlf = {}
    duration_s = {}
    sinr_at_ho_exe_pcell = []
    sinr_after_ho_exe_tcell = []
    for speed in speeds:
        bs_idxs[speed] = []
        ho_idxs[speed] = []
        rlf_idxs[speed] = []
        sinr_connected[speed] = []
        sinr_max[speed] = []
        n_ho[speed] = []
        n_ho_success[speed] = []
        n_hof[speed] = []
        n_pp[speed] = []
        n_rlf[speed] = []
        duration_s[speed] = []

    res_dict = {
        "bs_idxs": bs_idxs,
        "ho_idxs": ho_idxs,
        "rlf_idxs": rlf_idxs,
        "sinr_connected": sinr_connected,
        "sinr_max": sinr_max,
        "n_ho": n_ho,
        "n_ho_success": n_ho_success,
        "n_hof": n_hof,
        "n_pp": n_pp,
        "n_rlf": n_rlf,
        "duration_s": duration_s,
        "sinr_at_ho_exe_pcell": sinr_at_ho_exe_pcell,
        "sinr_after_ho_exe_tcell": sinr_after_ho_exe_tcell,
    }
    return res_dict


def print_statistics(stat_dict: dict[str, int | float]) -> None:
    """
    Print statistics.

    Parameters
    ----------
    stat_dict : dict[str, int | float]
        Dictionary containing the statistics.
    """
    params = [
        "Parameter",
        "Num BS",
        "Num samples",
        "",
        "Mean SINR",
        "Variance SINR",
        "Median SINR",
        "Q1 SINR",
        "Q3 SINR",
        "",
        "Max capacity",
        "Achieved capacity",
        "Total time",
        "Connected time",
        "",
        "# HO prep started",
        "# HO prep terminated",
        "# HO prep completed",
        "# HO prep aborted",
        "# HO prep failed",
        "# HO prep error",
        "",
        "# HO exe started",
        "# HO exe terminated",
        "# HO exe completed",
        "# HO exe aborted",
        "# HO exe failed",
        "# HO exe error",
        "",
        "# HOF",
        "HOF rate",
        "# RLF",
        "RLF recovery time",
        "# PP",
        "PP rate",
    ]
    values = [
        "Value",
        f"{stat_dict['num_base_stations']}",
        f"{stat_dict['num_samples']}",
        "",
        f"{stat_dict['mean_sinr']:.3f} dB",
        f"{stat_dict['variance_sinr']:.3f} dB^2",
        f"{stat_dict['median_sinr']:.3f} dB",
        f"{stat_dict['q1_sinr']:.3f} dB",
        f"{stat_dict['q3_sinr']:.3f} dB",
        "",
        f"{stat_dict['max_spectral_eff']:.3f} bits/s/Hz",
        f"{stat_dict['spectral_eff']:.3f} bits/s/Hz |"
        + f" {100 * stat_dict['spectral_eff'] / stat_dict['max_spectral_eff']:.3f}%",
        f"{stat_dict['total_time']} samples",
        f"{stat_dict['connected_time']} samples |"
        + f" {100 * stat_dict['connected_time'] / stat_dict['total_time']:.3f}%",
        "",
        f"{stat_dict['num_ho_prep_started']}",
        f"{stat_dict['num_ho_prep_terminated']}",
        f"{stat_dict['num_ho_prep_completed']}",
        f"{stat_dict['num_ho_prep_aborted']}",
        f"{stat_dict['num_ho_prep_failed']}",
        f"{stat_dict['num_ho_prep_error']}",
        "",
        f"{stat_dict['num_ho_exe_started']}",
        f"{stat_dict['num_ho_exe_terminated']}",
        f"{stat_dict['num_ho_exe_completed']}",
        f"{stat_dict['num_ho_exe_aborted']}",
        f"{stat_dict['num_ho_exe_failed']}",
        f"{stat_dict['num_ho_exe_error']}",
        "",
        f"{stat_dict['num_hof']}",
        f"{stat_dict['hof_percent']:.3f}%",
        f"{stat_dict['num_rlf']}",
        f"{stat_dict['rlf_time_percent']:.3f}%",
        f"{stat_dict['num_pp']}",
        f"{stat_dict['pp_percent']:.3f}%",
    ]
    tab = list(zip(params, values))
    max_param_len = max(len(row[0]) for row in tab)

    print("\nResults\n---------------------------------")
    for row in tab:
        print(f"{row[0]:<{max_param_len}}: {row[1]}")


def print_aggregated_stats(stats: dict):
    """
    Print statistics.

    Parameters
    ----------
    stats : dict
        Dictionary containing the aggregated statistics.
    """
    params = [
        "PARAMETERS",
        "Num datasets",
        "",
        "Avg relative rate (total)",
        "",
        "Speed",
        "Avg relative rate",
        "PP rate",
        "HOF rate",
        "RLF recovery time",
        "",
        "Avg # HO prep started",
        "Avg # HO prep terminated",
        "Avg # HO prep completed",
        "Avg # HO prep aborted",
        "Avg # HO prep failed",
        "Avg # HO prep error",
        "",
        "Avg # HO exe started",
        "Avg # HO exe terminated",
        "Avg # HO exe completed",
        "Avg # HO exe aborted",
        "Avg # HO exe failed",
        "Avg # HO exe error",
    ]
    values = [
        "VALUES",
        f"{len(stats['spectral_eff'])}",
        "",
        f"{np.mean(stats['relative_rate_percent']):6.3f} %",
        "",
        (
            f"[{"".join([f"{speed:6d}, " for speed in stats['speed']])}] km/h"
            if len(stats["speed"]) > 1
            else f"{stats["speed"][0]:6d} km/h"
        ),
        (
            f"[{"".join([f"{value:6.3f}, " for value in stats['relative_rate_percent']])}] %"
            if len(stats["relative_rate_percent"]) > 1
            else f"{stats['relative_rate_percent'][0]:6.3f} %"
        ),
        (
            f"[{"".join([f"{value:6.3f}, " for value in stats['pp_percent']])}] %"
            if len(stats["pp_percent"]) > 1
            else f"{stats['pp_percent'][0]:6.3f} %"
        ),
        (
            f"[{"".join([f"{value:6.3f}, " for value in stats['hof_percent']])}] %"
            if len(stats["hof_percent"]) > 1
            else f"{stats['hof_percent'][0]:6.3f} %"
        ),
        (
            f"[{"".join([f"{value:6.3f}, " for value in stats['rlf_time_percent']])}] %"
            if len(stats["rlf_time_percent"]) > 1
            else f"{stats['rlf_time_percent'][0]:6.3f} %"
        ),
        "",
        f"{np.mean(stats['num_ho_prep_started']):5.2f}",
        f"{np.mean(stats['num_ho_prep_terminated']):5.2f}",
        f"{np.mean(stats['num_ho_prep_completed']):5.2f}",
        f"{np.mean(stats['num_ho_prep_aborted']):5.2f}",
        f"{np.mean(stats['num_ho_prep_failed']):5.2f}",
        f"{np.mean(stats['num_ho_prep_error']):5.2f}",
        "",
        f"{np.mean(stats['num_ho_exe_started']):5.2f}",
        f"{np.mean(stats['num_ho_exe_terminated']):5.2f}",
        f"{np.mean(stats['num_ho_exe_completed']):5.2f}",
        f"{np.mean(stats['num_ho_exe_aborted']):5.2f}",
        f"{np.mean(stats['num_ho_exe_failed']):5.2f}",
        f"{np.mean(stats['num_ho_exe_error']):5.2f}",
    ]
    tab = list(zip(params, values))
    max_param_len = max(len(row[0]) for row in tab)

    print("\nAggregated Results\n---------------------------------")
    for row in tab:
        print(f"{row[0]:<{max_param_len}}: {row[1]}")


def write_to_csv(
    file_path: str, data_dict: dict[str, list[Any]], delimiter: str = ";"
) -> None:
    """Write data to a CSV file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    header = data_dict.keys()
    with open(file_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header, delimiter=delimiter)
        writer.writeheader()

        lens = [len(v) for v in data_dict.values()]
        if len(set(lens)) != 1:
            raise ValueError("Dictionary contains lists of different lengths.")
        for i in range(lens[0]):
            row = {key: data_dict[key][i] for key in header}
            writer.writerow(row)
