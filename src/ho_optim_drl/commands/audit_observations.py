"""Audit P2 observation saturation and causal +90 ms prediction error."""

from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from ho_optim_drl.data import (
    TracePair,
    discover_trace_pairs,
    filter_trace_pairs_by_speed,
    load_cell_topology,
    load_radio_trace,
    stratified_trajectory_split,
)
from ho_optim_drl.data.project import load_protocol_config
from ho_optim_drl.protocol import HandoverEngine
from ho_optim_drl.smdp import TopologyCandidateObservationBuilder


def _sample_routes(
    pairs: tuple[TracePair, ...], *, routes_per_speed: int, seed: int
) -> tuple[TracePair, ...]:
    grouped: dict[int, list[TracePair]] = defaultdict(list)
    for pair in pairs:
        grouped[pair.speed_kmh].append(pair)
    rng = np.random.default_rng(seed)
    selected: list[TracePair] = []
    for speed in sorted(grouped):
        group = sorted(grouped[speed])
        order = rng.permutation(len(group))
        selected.extend(group[int(index)] for index in order[:routes_per_speed])
    return tuple(sorted(selected))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _prediction_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"samples": 0, "mae_db": None, "rmse_db": None, "bias_db": None}
    errors = np.asarray([row["error_db"] for row in rows], dtype=np.float64)
    return {
        "samples": len(rows),
        "mae_db": float(np.mean(np.abs(errors))),
        "rmse_db": float(np.sqrt(np.mean(errors ** 2))),
        "bias_db": float(np.mean(errors)),
    }


def audit_observations(
    *,
    data_dir: str,
    split: str,
    speeds: list[int] | None,
    split_seed: int,
    sample_seed: int,
    routes_per_speed: int,
    max_decisions_per_route: int,
    candidate_count: int,
    candidate_order: str,
    observation_schema: str,
    output_dir: str,
) -> dict[str, Any]:
    if split not in {"train", "validation"}:
        raise ValueError("Observation audit may only use train or validation.")
    if routes_per_speed <= 0 or max_decisions_per_route <= 0:
        raise ValueError("Route and decision limits must be positive.")

    protocol_config = load_protocol_config(data_dir)
    pairs = filter_trace_pairs_by_speed(discover_trace_pairs(data_dir), speeds)
    splits = stratified_trajectory_split(pairs, seed=split_seed)
    split_pairs = tuple(getattr(splits, split))
    selected = _sample_routes(
        split_pairs, routes_per_speed=routes_per_speed, seed=sample_seed
    )
    topology = load_cell_topology(data_dir)
    prediction_rows: list[dict[str, Any]] = []
    feature_counts: dict[str, dict[str, int]] = {}
    decision_count = 0
    horizon_ticks = int(
        round(90.0 / float(protocol_config.delta_t_ms))
    )

    for pair in selected:
        trace = load_radio_trace(
            protocol_config,
            data_dir,
            pair.rsrp_file,
            pair.sinr_file,
            speed_kmh=pair.speed_kmh,
        )
        engine = HandoverEngine(protocol_config)
        builder = TopologyCandidateObservationBuilder(
            protocol_config,
            topology=topology,
            candidate_count=candidate_count,
            candidate_order=candidate_order,
            observation_schema=observation_schema,
        )
        engine.reset(trace)
        builder.reset()
        builder.start_episode(trace)
        route_decisions = 0
        while route_decisions < max_decisions_per_route:
            tick = engine.advance_one_tick()
            builder.observe(tick)
            if tick.terminated:
                break
            if engine.decision_point is None:
                continue

            observation = builder.build(engine.snapshot)
            for name, value, low, high in zip(
                builder.feature_names,
                observation,
                builder.low,
                builder.high,
            ):
                counts = feature_counts.setdefault(
                    name, {"samples": 0, "near_low": 0, "near_high": 0}
                )
                counts["samples"] += 1
                tolerance = 0.01 * float(high - low)
                counts["near_low"] += int(value <= low + tolerance)
                counts["near_high"] += int(value >= high - tolerance)

            actual_tic = engine.decision_point.tic + horizon_ticks
            if actual_tic < trace.num_samples:
                for diagnostic in builder.prediction_diagnostics(engine.snapshot):
                    cell = int(diagnostic["cell"])
                    actual = float(trace.sinr_rlm_db[cell, actual_tic])
                    predicted = float(diagnostic["predicted_sinr_90ms_db"])
                    prediction_rows.append(
                        {
                            "split": split,
                            "speed_kmh": pair.speed_kmh,
                            "trajectory_id": pair.trajectory_id,
                            "decision_tic": engine.decision_point.tic,
                            "actual_tic": actual_tic,
                            **diagnostic,
                            "actual_sinr_90ms_db": actual,
                            "error_db": predicted - actual,
                        }
                    )
            decision_count += 1
            route_decisions += 1
            engine.start_handover()

    saturation_rows = []
    for name in (
        TopologyCandidateObservationBuilder(
            protocol_config,
            topology=topology,
            candidate_count=candidate_count,
            candidate_order=candidate_order,
            observation_schema=observation_schema,
        ).feature_names
    ):
        counts = feature_counts.get(
            name, {"samples": 0, "near_low": 0, "near_high": 0}
        )
        samples = counts["samples"]
        saturation_rows.append(
            {
                "feature": name,
                **counts,
                "near_low_fraction": (
                    counts["near_low"] / samples if samples else 0.0
                ),
                "near_high_fraction": (
                    counts["near_high"] / samples if samples else 0.0
                ),
            }
        )

    by_speed = {
        str(speed): _prediction_summary(
            [row for row in prediction_rows if row["speed_kmh"] == speed]
        )
        for speed in sorted({row["speed_kmh"] for row in prediction_rows})
    }
    by_slot = {
        str(slot): _prediction_summary(
            [row for row in prediction_rows if row["slot"] == slot]
        )
        for slot in range(1, candidate_count + 1)
    }
    summary = {
        "schema_name": "p2_observation_audit",
        "schema_version": 1,
        "split": split,
        "test_loaded": False,
        "speeds_kmh": sorted({pair.speed_kmh for pair in selected}),
        "routes": len(selected),
        "routes_per_speed_limit": routes_per_speed,
        "decisions": decision_count,
        "max_decisions_per_route": max_decisions_per_route,
        "split_seed": split_seed,
        "sample_seed": sample_seed,
        "candidate_count": candidate_count,
        "candidate_order": candidate_order,
        "observation_schema": observation_schema,
        "observation_size": len(saturation_rows),
        "projection_horizon_ms": 90,
        "prediction": {
            "aggregate": _prediction_summary(prediction_rows),
            "by_speed": by_speed,
            "by_slot": by_slot,
        },
        "feature_saturation": saturation_rows,
    }
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "observation_audit_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _write_csv(destination / "prediction_calibration.csv", prediction_rows)
    _write_csv(destination / "feature_saturation.csv", saturation_rows)
    return summary


def main(root_path: str, args: Any) -> int:
    summary = audit_observations(
        data_dir=os.path.abspath(os.path.join(root_path, args.data_dir)),
        split=args.split,
        speeds=args.speeds,
        split_seed=args.split_seed,
        sample_seed=args.sample_seed,
        routes_per_speed=args.routes_per_speed,
        max_decisions_per_route=args.max_decisions_per_route,
        candidate_count=args.candidate_count,
        candidate_order=args.candidate_order,
        observation_schema=args.observation_schema,
        output_dir=os.path.abspath(os.path.join(root_path, args.output_dir)),
    )
    prediction = summary["prediction"]["aggregate"]
    print(
        f"Audited {summary['decisions']} decisions on {summary['routes']} "
        f"{summary['split']} routes; projection MAE={prediction['mae_db']} dB."
    )
    return 0
