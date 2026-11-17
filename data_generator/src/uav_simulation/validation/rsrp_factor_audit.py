"""Replay a manifested P0 dataset and quantify what drives its RSRP variation."""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd

from uav_simulation import config
from uav_simulation.manifest import _source_fingerprints
from uav_simulation.simulation import CellularNetworkReceivedPower


EXPECTED_PROFILE = "p0_canonical_v1"
SCHEMA_NAME = "p0_rsrp_factor_audit"
SCHEMA_VERSION = 3
FACTOR_NAMES = (
    "antenna_gain_db",
    "fspl_db",
    "excess_loss_db",
    "shadow_loss_db",
    "fast_fading_gain_db",
)


@contextmanager
def _manifest_config(manifest: dict[str, Any]) -> Iterator[None]:
    """Temporarily replay the exact uppercase config captured at generation."""
    recorded = manifest.get("simulation_config")
    if not isinstance(recorded, dict):
        raise ValueError("Generation manifest has no simulation_config mapping.")
    unknown = sorted(name for name in recorded if not hasattr(config, name))
    if unknown:
        raise ValueError(f"Current generator lacks manifest config fields: {unknown}")
    original = {name: getattr(config, name) for name in recorded}
    try:
        for name, value in recorded.items():
            setattr(config, name, value)
        yield
    finally:
        for name, value in original.items():
            setattr(config, name, value)


def _percentiles(values: list[float] | np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("Expected non-empty finite values for summary statistics.")
    return {
        f"p{percentile}": float(np.percentile(array, percentile))
        for percentile in (5, 50, 95, 99)
    }


def _same_sector_step_delta_metrics(values: np.ndarray) -> dict[str, list[float]]:
    """Summarize absolute 10 ms changes separately for every sector."""
    absolute_delta = np.abs(np.diff(values, axis=0))
    if absolute_delta.ndim != 2 or absolute_delta.shape[0] == 0:
        raise ValueError("Expected at least two samples for every sector.")
    return {
        "mean_abs_by_sector": np.mean(absolute_delta, axis=0).tolist(),
        "p95_abs_by_sector": np.percentile(absolute_delta, 95, axis=0).tolist(),
        "p99_abs_by_sector": np.percentile(absolute_delta, 99, axis=0).tolist(),
        "max_abs_by_sector": np.max(absolute_delta, axis=0).tolist(),
    }


class _FactorAccumulator:
    def __init__(self) -> None:
        self.count = Counter()
        self.sum = Counter()
        self.sum_sq = Counter()
        self.delta_count = Counter()
        self.abs_delta_sum = Counter()
        self.sector_route_ranges: dict[str, list[float]] = {
            name: [] for name in (*FACTOR_NAMES, "rsrp_dbm")
        }
        self.dominant_step_count = Counter()

    def add_route(self, components: dict[str, np.ndarray]) -> None:
        impacts = {
            "antenna_gain_db": components["antenna_gain_db"],
            "fspl_db": -components["fspl_db"],
            "excess_loss_db": -components["excess_loss_db"],
            "shadow_loss_db": -components["shadow_loss_db"],
            "fast_fading_gain_db": components["fast_fading_gain_db"],
        }
        for name, values in impacts.items():
            flat = values.reshape(-1)
            self.count[name] += int(flat.size)
            self.sum[name] += float(np.sum(flat))
            self.sum_sq[name] += float(np.sum(flat * flat))
            self.sector_route_ranges[name].extend(np.ptp(values, axis=0).tolist())
            if values.shape[0] > 1:
                delta = np.diff(values, axis=0)
                self.delta_count[name] += int(delta.size)
                self.abs_delta_sum[name] += float(np.sum(np.abs(delta)))

        self.sector_route_ranges["rsrp_dbm"].extend(
            np.ptp(components["rsrp_dbm"], axis=0).tolist()
        )

        deltas = np.stack([np.abs(np.diff(impacts[name], axis=0)) for name in FACTOR_NAMES])
        winner_idx = np.argmax(deltas, axis=0)
        winners, counts = np.unique(winner_idx, return_counts=True)
        for index, count in zip(winners, counts):
            self.dominant_step_count[FACTOR_NAMES[int(index)]] += int(count)

    def summary(self) -> dict[str, Any]:
        total_dominant = sum(self.dominant_step_count.values())
        factors: dict[str, Any] = {}
        for name in FACTOR_NAMES:
            count = self.count[name]
            mean = self.sum[name] / count
            variance = max(0.0, self.sum_sq[name] / count - mean * mean)
            factors[name] = {
                "rsrp_signed_effect": {
                    "antenna_gain_db": "+Gtx",
                    "fspl_db": "-FSPL",
                    "excess_loss_db": "-LoS/NLoS excess loss",
                    "shadow_loss_db": "-shadow loss",
                    "fast_fading_gain_db": "+fast fading gain",
                }[name],
                "global_standard_deviation_db": math.sqrt(variance),
                "mean_absolute_delta_per_step_db": (
                    self.abs_delta_sum[name] / self.delta_count[name]
                    if self.delta_count[name] else 0.0
                ),
                "same_sector_route_range_db": {
                    **_percentiles(self.sector_route_ranges[name]),
                    "max": float(max(self.sector_route_ranges[name])),
                },
                "dominant_per_step_share_percent": (
                    100.0 * self.dominant_step_count[name] / total_dominant
                    if total_dominant else 0.0
                ),
            }
        return factors

    def rsrp_summary(self) -> dict[str, float]:
        return {
            **_percentiles(self.sector_route_ranges["rsrp_dbm"]),
            "max": float(max(self.sector_route_ranges["rsrp_dbm"])),
        }


def _selected_trajectories(
    metadata: dict[str, Any], speeds: set[int] | None, route_ids: set[int] | None,
) -> list[dict[str, Any]]:
    trajectories = metadata.get("trajectories")
    if not isinstance(trajectories, list) or not trajectories:
        raise ValueError("dataset_metadata.json has no trajectories.")
    selected = []
    for record in trajectories:
        speed = int(record["speed_kmh"])
        route_id = int(record["trajectory_id"])
        if speeds is not None and speed not in speeds:
            continue
        if route_ids is not None and route_id not in route_ids:
            continue
        selected.append(record)
    if not selected:
        raise ValueError("No trajectories match the requested filters.")
    return sorted(selected, key=lambda row: (int(row["speed_kmh"]), int(row["trajectory_id"])))


def _replay_route(parameters: dict[str, Any], speed_kmh: int, route_id: int) -> dict[str, np.ndarray]:
    sim = CellularNetworkReceivedPower(
        num_ues=1,
        rect_len_m=float(parameters["width_m"]),
        rect_wid_m=float(parameters["height_m"]),
        grid_spacing_m=float(parameters["grid_spacing_m"]),
        uav_ue_altitude_m=float(parameters["altitude_m"]),
        uav_ue_speed_kmh=float(speed_kmh),
        seed=speed_kmh * 1000 + route_id,
        headless=True,
        train_ready_output=True,
    )
    steps = int(parameters["steps"])
    collected: dict[str, list[np.ndarray]] = {
        name: [] for name in (*FACTOR_NAMES, "is_los", "rsrp_dbm", "distance_3d_m", "position_xy_m")
    }
    positions = np.asarray(sim.bs_positions, dtype=float)
    heights = np.asarray(sim.bs_heights, dtype=float)
    initial_position = np.asarray(sim.ue_positions[0], dtype=float).copy()
    for frame in range(steps):
        sim.current_frame = frame
        distance_m = float(sim.ue_speeds_ms[0] * sim.time_per_step)
        sim.mobility.step_manhattan_on_grid(0, distance_m)
        sim.radio_model.calculate_all_bs_measurements(0)
        current = sim.radio_model.last_measurement_components
        for name in collected:
            if name == "position_xy_m":
                collected[name].append(np.asarray(sim.ue_positions[0], dtype=float).copy())
                continue
            if name == "distance_3d_m":
                delta = positions - np.asarray(sim.ue_positions[0], dtype=float)
                collected[name].append(np.sqrt(np.sum(delta * delta, axis=1) + (heights - sim.h_ut) ** 2))
                continue
            collected[name].append(np.asarray(current[name], dtype=float).copy())
        if (frame + 1) % 1000 == 0 or frame + 1 == steps:
            print(
                f"[RSRP factor audit] replay speed={speed_kmh} route={route_id} "
                f"sample={frame + 1}/{steps}",
                flush=True,
            )
    result = {name: np.stack(values) for name, values in collected.items()}
    result["initial_position_xy_m"] = initial_position
    return result


def _verify_replay(data_dir: Path, record: dict[str, Any], replay_rsrp: np.ndarray) -> float:
    source = data_dir / str(record["rsrp_file"])
    frame = pd.read_csv(source)
    columns = [f"bs_{index}" for index in range(replay_rsrp.shape[1])]
    if list(frame.columns[:2]) != ["sample", "time_ms"] or list(frame.columns[2:]) != columns:
        raise ValueError(f"Unexpected RSRP schema: {source}")
    observed = frame[columns].to_numpy(dtype=float)
    if observed.shape != replay_rsrp.shape:
        raise ValueError(f"Replay shape differs from stored RSRP: {source}")
    return float(np.max(np.abs(observed - np.round(replay_rsrp, 2))))


def _summarize_by_speed(route_rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for speed in sorted({int(row["speed_kmh"]) for row in route_rows}):
        rows = [row for row in route_rows if int(row["speed_kmh"]) == speed]
        factor_ranges = {}
        for name in (*FACTOR_NAMES, "rsrp_dbm"):
            values = np.concatenate([
                np.asarray(row["same_sector_route_range_db"][name], dtype=float)
                for row in rows
            ])
            factor_ranges[name] = {**_percentiles(values), "max": float(np.max(values))}
        distance_ranges = np.concatenate([
            np.asarray(row["same_sector_distance_route_range_m"], dtype=float)
            for row in rows
        ])
        result[str(speed)] = {
            "routes": len(rows),
            "sector_route_traces": int(distance_ranges.size),
            "travel_distance_m": _percentiles([row["travel_distance_m"] for row in rows]),
            "same_sector_distance_route_range_m": {
                **_percentiles(distance_ranges), "max": float(np.max(distance_ranges))
            },
            "same_sector_route_range_db": factor_ranges,
        }
    return result


def run_audit(
    data_dir: Path,
    output_dir: Path,
    *,
    speeds: set[int] | None = None,
    route_ids: set[int] | None = None,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite audit output: {output_dir}")
    manifest_path = data_dir / "simulation_manifest.json"
    metadata_path = data_dir / "dataset_metadata.json"
    if not manifest_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError("Dataset must contain simulation_manifest.json and dataset_metadata.json.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    parameters = manifest.get("parameters")
    if not isinstance(parameters, dict) or parameters.get("profile") != EXPECTED_PROFILE:
        raise ValueError(f"This tool accepts only profile {EXPECTED_PROFILE!r}.")
    selected = _selected_trajectories(metadata, speeds, route_ids)
    output_dir.mkdir(parents=True)
    accumulator = _FactorAccumulator()
    route_rows: list[dict[str, Any]] = []
    all_los_transitions = 0
    max_replay_error_db = 0.0

    with _manifest_config(manifest):
        for index, record in enumerate(selected, 1):
            speed = int(record["speed_kmh"])
            route_id = int(record["trajectory_id"])
            components = _replay_route(parameters, speed, route_id)
            accumulator.add_route(components)
            is_los = components["is_los"].astype(bool)
            transition_count = int(np.count_nonzero(np.diff(is_los.astype(np.int8), axis=0)))
            all_los_transitions += transition_count
            replay_error = _verify_replay(data_dir, record, components["rsrp_dbm"])
            max_replay_error_db = max(max_replay_error_db, replay_error)
            route_rows.append({
                "speed_kmh": speed,
                "trajectory_id": route_id,
                "cells": int(components["rsrp_dbm"].shape[1]),
                "los_nlos_transitions": transition_count,
                "replay_max_abs_error_db": replay_error,
                "travel_distance_m": float(np.sum(np.linalg.norm(np.diff(
                    np.vstack([components["initial_position_xy_m"], components["position_xy_m"]]), axis=0
                ), axis=1))),
                "same_sector_route_range_db": {
                    name: np.ptp(components[name], axis=0).tolist()
                    for name in (*FACTOR_NAMES, "rsrp_dbm")
                },
                "same_sector_step_abs_delta_db": {
                    name: _same_sector_step_delta_metrics(components[name])
                    for name in (*FACTOR_NAMES, "rsrp_dbm")
                },
                "same_sector_distance_route_range_m": np.ptp(
                    components["distance_3d_m"], axis=0
                ).tolist(),
                "distance_3d_m": {
                    "minimum_by_sector": np.min(components["distance_3d_m"], axis=0).tolist(),
                    "maximum_by_sector": np.max(components["distance_3d_m"], axis=0).tolist(),
                },
                "factor_mean_abs_delta_per_step_db": {
                    name: float(np.mean(np.abs(np.diff(components[name], axis=0))))
                    for name in FACTOR_NAMES
                },
            })
            print(f"[RSRP factor audit] {index}/{len(selected)} speed={speed} route={route_id}", flush=True)

    current_tree_hash, _ = _source_fingerprints()
    recorded_tree_hash = manifest.get("generator_provenance", {}).get("source_tree_sha256")
    summary = accumulator.summary()
    ranking_by_route_range = sorted(
        summary, key=lambda name: summary[name]["same_sector_route_range_db"]["p50"], reverse=True
    )
    ranking_by_step_delta = sorted(
        summary, key=lambda name: summary[name]["mean_absolute_delta_per_step_db"], reverse=True
    )
    report = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "status": "COMPLETE",
        "dataset": {
            "path": str(data_dir.resolve()),
            "profile": parameters["profile"],
            "selected_routes": len(selected),
            "speeds": sorted({int(row["speed_kmh"]) for row in selected}),
            "route_ids": sorted({int(row["trajectory_id"]) for row in selected}),
            "recorded_source_tree_sha256": recorded_tree_hash,
            "current_source_tree_sha256": current_tree_hash,
            "source_tree_matches_manifest": current_tree_hash == recorded_tree_hash,
        },
        "replay_validation": {
            "max_abs_rsrp_error_db_after_csv_rounding": max_replay_error_db,
            "passes_exact_csv_replay": max_replay_error_db <= 1e-9,
        },
        "factor_summary": summary,
        "same_sector_rsrp_route_range_db": accumulator.rsrp_summary(),
        "by_speed": _summarize_by_speed(route_rows),
        "rankings": {
            "largest_median_same_sector_route_range": ranking_by_route_range,
            "largest_mean_absolute_delta_per_step": ranking_by_step_delta,
        },
        "los_nlos_transitions": {
            "total_across_all_sector_traces": all_los_transitions,
            "mean_per_selected_route": all_los_transitions / len(selected),
        },
        "interpretation": {
            "same_sector_route_range": "For each route and sector separately, max over time minus min over time; percentiles pool sector-route pairs.",
            "same_sector_step_abs_delta": "For each route and sector separately, absolute change between adjacent 10 ms samples; reports mean, p95, p99, and maximum.",
            "per_step_delta": "Mean absolute 10 ms change in the signed RSRP contribution.",
            "dominant_per_step_share": "Share of cell-step transitions where this component has the largest absolute contribution change.",
            "sign_convention": "Gtx and fast fading add to RSRP; FSPL, excess loss, and shadow loss subtract from RSRP.",
        },
    }
    (output_dir / "factor_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    with (output_dir / "route_factor_summary.jsonl").open("w", encoding="utf-8") as stream:
        for row in route_rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")

    markdown = [
        "# P0: Biên độ RSRP theo từng sector trên mỗi route",
        "",
        "Mỗi biên độ là `max - min` theo 6.000 bước của **cùng một sector** trên **một route**. Các phân vị gộp những cặp route–sector, không gộp công suất của 21 sector trước khi lấy biên độ.",
        "",
        f"- Số route phát lại: **{len(selected)}**.",
        f"- Khớp CSV sau làm tròn: **{report['replay_validation']['passes_exact_csv_replay']}**; sai số lớn nhất: `{max_replay_error_db:.6g} dB`.",
        f"- Thành phần có trung vị biên độ theo route lớn nhất: **{ranking_by_route_range[0]}**.",
        f"- Thành phần biến thiên trung bình mỗi bước lớn nhất: **{ranking_by_step_delta[0]}**.",
        f"- Số lần chuyển LoS/NLoS trên các vệt sector: **{all_los_transitions}**.",
        "",
        "| Thành phần | Trung vị biên độ cùng sector (dB) | Biên độ lớn nhất cùng sector (dB) | Trung bình độ lớn Δ mỗi 10 ms (dB) |",
        "|---|---:|---:|---:|",
    ]
    for name in FACTOR_NAMES:
        item = summary[name]
        markdown.append(
            f"| `{name}` | {item['same_sector_route_range_db']['p50']:.3f} | "
            f"{item['same_sector_route_range_db']['max']:.3f} | "
            f"{item['mean_absolute_delta_per_step_db']:.4f} |"
        )
    markdown.extend([
        "",
        "| Tốc độ (km/h) | Quãng đường (m) | Gtx max (dB) | FSPL max (dB) | LoS/NLoS max (dB) | Shadow max (dB) | Fast max (dB) | RSRP max (dB) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for speed, item in report["by_speed"].items():
        ranges = item["same_sector_route_range_db"]
        markdown.append(
            f"| {speed} | {item['travel_distance_m']['p50']:.1f} | "
            f"{ranges['antenna_gain_db']['max']:.2f} | {ranges['fspl_db']['max']:.2f} | "
            f"{ranges['excess_loss_db']['max']:.2f} | {ranges['shadow_loss_db']['max']:.2f} | "
            f"{ranges['fast_fading_gain_db']['max']:.2f} | {ranges['rsrp_dbm']['max']:.2f} |"
        )
    (output_dir / "report_vi.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--speeds", type=int, nargs="+", help="Optional speed filter.")
    parser.add_argument("--route-ids", type=int, nargs="+", help="Optional route ID filter.")
    args = parser.parse_args(argv)
    report = run_audit(
        args.data_dir.resolve(),
        args.output_dir.resolve(),
        speeds=None if args.speeds is None else set(args.speeds),
        route_ids=None if args.route_ids is None else set(args.route_ids),
    )
    print(json.dumps({"status": report["status"], "rankings": report["rankings"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
