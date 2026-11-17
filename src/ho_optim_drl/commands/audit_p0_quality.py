"""Audit integrity, radio variation, and 3GPP viability of a P0 dataset."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import time
from typing import Any, Iterable

import numpy as np

from ho_optim_drl.data import discover_trace_pairs, load_radio_trace
from ho_optim_drl.data.project import load_protocol_config
from ho_optim_drl.protocol import A3ReferenceSimulator


SCHEMA_NAME = "p0_canonical_quality_audit"
SCHEMA_VERSION = 1
EXPECTED_PROFILE = "p0_canonical_v1"
EXPECTED_SPEEDS = (30, 50, 70, 90, 120, 150)


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _percentiles(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("Percentile input must be non-empty and finite.")
    return {
        f"p{percentile}": float(np.percentile(array, percentile))
        for percentile in (1, 5, 10, 50, 90, 95, 99)
    }


def _normalized_entropy(counts: Iterable[int]) -> float:
    values = np.asarray(list(counts), dtype=np.float64)
    values = values[values > 0]
    if values.size <= 1:
        return 0.0
    probabilities = values / np.sum(values)
    entropy = -float(np.sum(probabilities * np.log(probabilities)))
    return entropy / math.log(values.size)


def _protocol_metrics(simulator: A3ReferenceSimulator, trace: Any) -> dict[str, Any]:
    statistics = simulator.get_statistics(trace)
    maximum = float(statistics["max_estimated_rate_mbps"])
    estimated = float(statistics["estimated_rate_mbps"])
    return {
        "relative_rate_percent": 100.0 * estimated / maximum if maximum > 0 else 0.0,
        "estimated_rate_mbps": estimated,
        "max_estimated_rate_mbps": maximum,
        "handover_attempts": int(statistics["num_ho_exe_started"]),
        "handover_successes": int(statistics["num_ho_exe_completed"]),
        "handover_failures": int(statistics["num_hof"]),
        "ping_pong_count": int(statistics["num_pp"]),
        "rlf_starts": int(statistics["num_rlf"]),
        "rlf_recovery_samples": int(statistics["rlf_recovery_samples"]),
        "ttt_starts": len(simulator.cntr["tttc"].start_idxs),
        "ttt_aborts": len(simulator.cntr["tttc"].aborted_idxs),
        "a3_maturities": len(simulator.cntr["ho_prep"].start_idxs),
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Cannot aggregate an empty route group.")
    samples = sum(int(row["samples"]) for row in rows)
    duration_minutes = samples * float(rows[0]["delta_t_ms"]) / 60_000.0
    attempts = sum(int(row["protocol"]["handover_attempts"]) for row in rows)
    successes = sum(int(row["protocol"]["handover_successes"]) for row in rows)
    failures = sum(int(row["protocol"]["handover_failures"]) for row in rows)
    ping_pongs = sum(int(row["protocol"]["ping_pong_count"]) for row in rows)
    recovery = sum(int(row["protocol"]["rlf_recovery_samples"]) for row in rows)
    ttt_starts = sum(int(row["protocol"]["ttt_starts"]) for row in rows)
    ttt_aborts = sum(int(row["protocol"]["ttt_aborts"]) for row in rows)
    maturities = sum(int(row["protocol"]["a3_maturities"]) for row in rows)
    switches = sum(int(row["radio"]["rank_switches"]) for row in rows)
    winner_counts: Counter[int] = Counter()
    for row in rows:
        winner_counts.update({int(k): int(v) for k, v in row["winner_counts"].items()})

    best_rsrp = np.concatenate([np.asarray(row["best_rsrp_dbm"]) for row in rows])
    best_sinr = np.concatenate([np.asarray(row["best_sinr_db"]) for row in rows])
    top_gap = np.concatenate([np.asarray(row["top_gap_db"]) for row in rows])
    return {
        "routes": len(rows),
        "samples": samples,
        "duration_minutes": duration_minutes,
        "radio": {
            "best_rsrp_dbm": _percentiles(best_rsrp),
            "best_sinr_db": _percentiles(best_sinr),
            "top1_minus_top2_rsrp_db": _percentiles(top_gap),
            "top_gap_le_2db_percent": 100.0 * float(np.mean(top_gap <= 2.0)),
            "top_gap_le_5db_percent": 100.0 * float(np.mean(top_gap <= 5.0)),
            "rank_switches": switches,
            "rank_switches_per_minute": switches / duration_minutes,
            "unique_winner_cells": len(winner_counts),
            "winner_entropy_normalized": _normalized_entropy(winner_counts.values()),
            "max_winner_cell_share_percent": (
                100.0 * max(winner_counts.values()) / samples if samples else 0.0
            ),
        },
        "protocol": {
            "mean_route_relative_rate_percent": float(np.mean([
                row["protocol"]["relative_rate_percent"] for row in rows
            ])),
            "p10_route_relative_rate_percent": float(np.percentile([
                row["protocol"]["relative_rate_percent"] for row in rows
            ], 10)),
            "mean_route_estimated_rate_mbps": float(np.mean([
                row["protocol"]["estimated_rate_mbps"] for row in rows
            ])),
            "handover_attempts": attempts,
            "handover_successes": successes,
            "handover_failures": failures,
            "handover_failure_percent": 100.0 * failures / attempts if attempts else 0.0,
            "ping_pong_count": ping_pongs,
            "ping_pong_percent": 100.0 * ping_pongs / successes if successes else 0.0,
            "rlf_starts": sum(int(row["protocol"]["rlf_starts"]) for row in rows),
            "rlf_time_percent": 100.0 * recovery / samples if samples else 0.0,
            "ttt_starts": ttt_starts,
            "ttt_aborts": ttt_aborts,
            "a3_maturities": maturities,
            "ttt_abort_percent": 100.0 * ttt_aborts / ttt_starts if ttt_starts else 0.0,
            "a3_maturity_percent": 100.0 * maturities / ttt_starts if ttt_starts else 0.0,
        },
    }


def _quality_gates(aggregate: dict[str, Any], per_speed: dict[str, Any]) -> dict[str, Any]:
    all_speeds_present = set(map(int, per_speed)) == set(EXPECTED_SPEEDS)
    variation = aggregate["radio"]
    dynamic_range_pass = bool(
        variation["best_rsrp_dbm"]["p99"] - variation["best_rsrp_dbm"]["p1"] >= 10.0
        and variation["best_sinr_db"]["p99"] - variation["best_sinr_db"]["p1"] >= 10.0
    )
    decision_opportunity_pass = bool(all(
        group["radio"]["rank_switches"] > 0
        and group["radio"]["unique_winner_cells"] >= 3
        and group["protocol"]["ttt_starts"] > 0
        and group["protocol"]["a3_maturities"] > 0
        for group in per_speed.values()
    ))
    protocol_viability_pass = bool(all(
        group["protocol"]["mean_route_relative_rate_percent"] >= 70.0
        and group["protocol"]["handover_failure_percent"] <= 30.0
        and group["protocol"]["rlf_time_percent"] <= 10.0
        for group in per_speed.values()
    ))
    gates = {
        "G_integrity": {
            "passed": all_speeds_present and aggregate["routes"] == 120,
            "criterion": "120 routes; exactly 20 at each expected speed",
        },
        "G_radio_dynamic_range": {
            "passed": dynamic_range_pass,
            "criterion": "p99-p1 >= 10 dB for best-cell RSRP and SINR",
        },
        "G_decision_opportunity": {
            "passed": decision_opportunity_pass,
            "criterion": "every speed has rank switches, >=3 winners, TTT starts and maturities",
        },
        "G_protocol_viability": {
            "passed": protocol_viability_pass,
            "criterion": "every speed RR>=70%, HOF<=30%, RLF-time<=10%",
        },
    }
    gates["ready_for_main_or_training_pilot"] = all(
        value["passed"] for value in gates.values()
    )
    return gates


def run_audit(data_dir: Path, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite P0 audit: {output_dir}")
    metadata_path = data_dir / "dataset_metadata.json"
    manifest_path = data_dir / "simulation_manifest.json"
    if not metadata_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("P0 dataset metadata/manifest is incomplete.")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    parameters = metadata.get("generation", {}).get("parameters", {})
    if parameters.get("profile") != EXPECTED_PROFILE:
        raise ValueError(f"Expected profile {EXPECTED_PROFILE!r}.")

    pairs = discover_trace_pairs(str(data_dir))
    counts = Counter(int(pair.speed_kmh) for pair in pairs)
    if len(pairs) != 120 or counts != Counter({speed: 20 for speed in EXPECTED_SPEEDS}):
        raise ValueError(f"Unexpected P0 route inventory: {dict(counts)}")

    output_dir.mkdir(parents=True)
    config = load_protocol_config(data_dir)
    rows: list[dict[str, Any]] = []
    all_rsrp: list[np.ndarray] = []
    all_sinr: list[np.ndarray] = []
    started = time.perf_counter()
    rows_path = output_dir / "route_quality.jsonl"
    with rows_path.open("w", encoding="utf-8") as stream:
        for index, pair in enumerate(pairs, 1):
            trace = load_radio_trace(
                config, str(data_dir), pair.rsrp_file, pair.sinr_file,
                speed_kmh=pair.speed_kmh,
            )
            if trace.num_samples != int(parameters["steps"]):
                raise ValueError(f"Unexpected sample count in {trace.trace_id}.")
            rsrp = np.asarray(trace.rsrp_a3_db, dtype=np.float64)
            sinr = np.asarray(trace.sinr_rate_db, dtype=np.float64)
            if rsrp.shape != sinr.shape or not np.all(np.isfinite(rsrp)) or not np.all(np.isfinite(sinr)):
                raise ValueError(f"Invalid radio arrays in {trace.trace_id}.")

            top_two = np.partition(rsrp, -2, axis=0)[-2:]
            top_gap = top_two[1] - top_two[0]
            winners = np.argmax(rsrp, axis=0)
            best_rsrp = np.max(rsrp, axis=0)
            best_sinr = np.max(sinr, axis=0)
            simulator = A3ReferenceSimulator(config)
            simulator.predict(trace)
            row = {
                "trace_id": trace.trace_id,
                "speed_kmh": int(pair.speed_kmh),
                "trajectory_id": int(pair.trajectory_id),
                "samples": int(trace.num_samples),
                "cells": int(trace.num_base_stations),
                "delta_t_ms": int(trace.delta_t_ms),
                "radio": {
                    "rank_switches": int(np.count_nonzero(np.diff(winners))),
                    "unique_winner_cells": int(np.unique(winners).size),
                    "best_rsrp_dbm": _percentiles(best_rsrp),
                    "best_sinr_db": _percentiles(best_sinr),
                    "top_gap_db": _percentiles(top_gap),
                },
                "winner_counts": {
                    str(int(cell)): int(count)
                    for cell, count in zip(*np.unique(winners, return_counts=True))
                },
                "protocol": _protocol_metrics(simulator, trace),
                # Arrays are retained only in memory for exact grouped statistics.
                "best_rsrp_dbm": best_rsrp.tolist(),
                "best_sinr_db": best_sinr.tolist(),
                "top_gap_db": top_gap.tolist(),
            }
            rows.append(row)
            all_rsrp.append(rsrp.reshape(-1))
            all_sinr.append(sinr.reshape(-1))
            persisted = {key: value for key, value in row.items() if key not in {
                "best_rsrp_dbm", "best_sinr_db", "top_gap_db"
            }}
            stream.write(json.dumps(persisted, ensure_ascii=False, allow_nan=False) + "\n")
            if index % 10 == 0 or index == len(pairs):
                print(f"[P0 quality] {index}/{len(pairs)} {trace.trace_id}", flush=True)

    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["speed_kmh"])].append(row)
    aggregate = _aggregate(rows)
    aggregate["radio"]["all_cell_l3_rsrp_dbm"] = _percentiles(np.concatenate(all_rsrp))
    aggregate["radio"]["all_cell_sinr_db"] = _percentiles(np.concatenate(all_sinr))
    per_speed = {str(speed): _aggregate(selected) for speed, selected in sorted(grouped.items())}
    gates = _quality_gates(aggregate, per_speed)
    report = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "status": "COMPLETE",
        "scope": "P0 data quality; no PPO training; no historical validation/test data",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "path": str(data_dir.resolve()),
            "profile": parameters["profile"],
            "metadata_sha256": _sha256(metadata_path),
            "manifest_sha256": _sha256(manifest_path),
            "routes": len(pairs),
            "route_counts_by_speed": {str(k): v for k, v in sorted(counts.items())},
        },
        "protocol_config": asdict(config),
        "aggregate": aggregate,
        "per_speed": per_speed,
        "gates": gates,
        "runtime_seconds": time.perf_counter() - started,
        "interpretation": (
            "Quality gates detect malformed or degenerate data; they are not a claim "
            "that PPO improves on 3GPP. Stress profiles must be evaluated separately."
        ),
    }
    report_path = output_dir / "quality_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    summary_lines = [
        "# P0 Canonical pilot — quality audit",
        "",
        f"- Status: `{report['status']}`; routes: {len(pairs)}; profile: `{parameters['profile']}`.",
        f"- Ready gate: **{gates['ready_for_main_or_training_pilot']}**.",
        f"- Best-cell RSRP p5/p50/p95: {aggregate['radio']['best_rsrp_dbm']['p5']:.2f} / {aggregate['radio']['best_rsrp_dbm']['p50']:.2f} / {aggregate['radio']['best_rsrp_dbm']['p95']:.2f} dBm.",
        f"- Best-cell SINR p5/p50/p95: {aggregate['radio']['best_sinr_db']['p5']:.2f} / {aggregate['radio']['best_sinr_db']['p50']:.2f} / {aggregate['radio']['best_sinr_db']['p95']:.2f} dB.",
        f"- Top-gap <=2 dB: {aggregate['radio']['top_gap_le_2db_percent']:.2f}%; rank switches/min: {aggregate['radio']['rank_switches_per_minute']:.2f}.",
        f"- 3GPP RR: {aggregate['protocol']['mean_route_relative_rate_percent']:.2f}%; HOF: {aggregate['protocol']['handover_failure_percent']:.2f}%; PP: {aggregate['protocol']['ping_pong_percent']:.2f}%; RLF-time: {aggregate['protocol']['rlf_time_percent']:.2f}%.",
        f"- A3/TTT starts: {aggregate['protocol']['ttt_starts']}; aborts: {aggregate['protocol']['ttt_aborts']}; maturities: {aggregate['protocol']['a3_maturities']}.",
        "",
        "Các gate chỉ xác nhận dữ liệu hợp lệ, không suy ra PPO tốt hơn 3GPP.",
    ]
    (output_dir / "report_vi.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    run_manifest = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "quality_report_sha256": _sha256(report_path),
        "route_quality_sha256": _sha256(rows_path),
        "dataset_metadata_sha256": _sha256(metadata_path),
        "dataset_manifest_sha256": _sha256(manifest_path),
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    report = run_audit(args.data_dir.resolve(), args.output_dir.resolve())
    print(json.dumps({"status": report["status"], "gates": report["gates"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

