"""Pilot B1 audit: lower A3 offset while holding all other protocol rules fixed."""

from __future__ import annotations

from argparse import Namespace
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any, Sequence

import numpy as np

from ho_optim_drl.commands.audit_baseline_parity import _git_state, _sha256
from ho_optim_drl.config import Config
from ho_optim_drl.data import TracePair, load_radio_trace
from ho_optim_drl.data.project import load_protocol_config
from ho_optim_drl.protocol import A3ReferenceSimulator


SCHEMA_NAME = "early_ho_b1_lower_threshold_pilot"
SCHEMA_VERSION = 1


def b1_config(b0: Config) -> Config:
    """Return B1 with only A3 offset changed from 1 dB to 0 dB."""
    values = asdict(b0)
    if not np.isclose(float(values["a3_off"]), 1.0):
        raise ValueError(f"B0 a3_off must be 1 dB, got {values['a3_off']}.")
    if not np.isclose(float(values["a3_hys"]), 1.0):
        raise ValueError(f"B0 a3_hys must be 1 dB, got {values['a3_hys']}.")
    if int(values["a3_ttt_ms"]) != 40:
        raise ValueError(f"B0 A3 TTT must be 40 ms, got {values['a3_ttt_ms']}.")
    values["a3_off"] = 0.0
    return Config(**values)


def _relative_rate(stats: dict[str, Any]) -> float:
    maximum = float(stats["max_estimated_rate_mbps"])
    return 100.0 * float(stats["estimated_rate_mbps"]) / maximum if maximum > 0 else 0.0


def _route_metrics(simulator: A3ReferenceSimulator, trace: Any) -> dict[str, Any]:
    stats = simulator.get_statistics(trace)
    return {
        "relative_rate_percent": _relative_rate(stats),
        "estimated_rate_mbps": float(stats["estimated_rate_mbps"]),
        "max_estimated_rate_mbps": float(stats["max_estimated_rate_mbps"]),
        "total_samples": int(stats["total_time"]),
        "rlf_recovery_samples": int(stats["rlf_recovery_samples"]),
        "handover_attempts": int(stats["num_ho_exe_started"]),
        "handover_successes": int(stats["num_ho_exe_completed"]),
        "handover_failures": int(stats["num_hof"]),
        "ping_pong_count": int(stats["num_pp"]),
        "rlf_starts": int(stats["num_rlf"]),
        "ttt_starts": len(simulator.cntr["tttc"].start_idxs),
        "ttt_aborts": len(simulator.cntr["tttc"].aborted_idxs),
        "ho_prep_start_tics": list(simulator.cntr["ho_prep"].start_idxs),
    }


def _aggregate(rows: Sequence[dict[str, Any]], policy: str) -> dict[str, Any]:
    metrics = [row[policy] for row in rows]
    attempts = sum(item["handover_attempts"] for item in metrics)
    successes = sum(item["handover_successes"] for item in metrics)
    failures = sum(item["handover_failures"] for item in metrics)
    ping_pongs = sum(item["ping_pong_count"] for item in metrics)
    recovery = sum(item["rlf_recovery_samples"] for item in metrics)
    total = sum(item["total_samples"] for item in metrics)
    return {
        "routes": len(metrics),
        "mean_route_relative_rate_percent": float(
            np.mean([item["relative_rate_percent"] for item in metrics])
        ),
        "mean_route_estimated_rate_mbps": float(
            np.mean([item["estimated_rate_mbps"] for item in metrics])
        ),
        "handover_attempts": attempts,
        "handover_successes": successes,
        "handover_failures": failures,
        "handover_failure_percent": 100.0 * failures / attempts if attempts else None,
        "ping_pong_count": ping_pongs,
        "ping_pong_percent": 100.0 * ping_pongs / successes if successes else None,
        "rlf_starts": sum(item["rlf_starts"] for item in metrics),
        "rlf_recovery_samples": recovery,
        "rlf_time_percent": 100.0 * recovery / total if total else None,
        "ttt_starts": sum(item["ttt_starts"] for item in metrics),
        "ttt_aborts": sum(item["ttt_aborts"] for item in metrics),
    }


def _delta(b0: dict[str, Any], b1: dict[str, Any]) -> dict[str, Any]:
    names = (
        "mean_route_relative_rate_percent",
        "mean_route_estimated_rate_mbps",
        "handover_failure_percent",
        "ping_pong_percent",
        "rlf_time_percent",
    )
    return {
        name: (
            None if b0[name] is None or b1[name] is None
            else float(b1[name]) - float(b0[name])
        )
        for name in names
    }


def _load_pairs(manifest_path: Path, data_dir: Path) -> tuple[list[TracePair], dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_name") != "early_ho_b0_b0a_parity":
        raise ValueError("B1 requires a B0/B0A parity manifest.")
    if manifest.get("scope") != "train_only" or int(
        manifest.get("validation_or_test_rows_read", -1)
    ) != 0:
        raise ValueError("B1 accepts only a verified TRAIN-only B0 manifest.")
    pairs: list[TracePair] = []
    for record in manifest["selected_routes"]:
        pair = TracePair(
            int(record["speed_kmh"]), int(record["trajectory_id"]),
            str(record["rsrp_file"]), str(record["sinr_file"]),
        )
        for kind in ("rsrp", "sinr"):
            path = data_dir / str(record[f"{kind}_file"])
            if _sha256(path) != record[f"{kind}_sha256"]:
                raise ValueError(f"B1 input hash mismatch: {path}")
        pairs.append(pair)
    if not pairs:
        raise ValueError("B0 manifest contains no selected routes.")
    return pairs, manifest


def run_b1_audit(
    *, root: Path, data_dir: Path, baseline_manifest: Path, output_dir: Path
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing B1 pilot: {output_dir}")
    output_dir.mkdir(parents=True)
    pairs, source_manifest = _load_pairs(baseline_manifest, data_dir)
    b0_config = load_protocol_config(str(data_dir))
    b1_protocol_config = b1_config(b0_config)
    rows: list[dict[str, Any]] = []
    rows_path = output_dir / "route_outcomes.jsonl"
    started = time.perf_counter()
    with rows_path.open("w", encoding="utf-8") as stream:
        for index, pair in enumerate(pairs, 1):
            trace = load_radio_trace(
                b0_config, str(data_dir), pair.rsrp_file, pair.sinr_file,
                speed_kmh=pair.speed_kmh,
            )
            simulators = {
                "b0": A3ReferenceSimulator(b0_config),
                "b1": A3ReferenceSimulator(b1_protocol_config),
            }
            outcomes: dict[str, Any] = {}
            for label, simulator in simulators.items():
                simulator.predict(trace)
                outcomes[label] = _route_metrics(simulator, trace)
            row = {
                "trace_id": f"{pair.speed_kmh}:{pair.trajectory_id}",
                "speed_kmh": pair.speed_kmh,
                "trajectory_id": pair.trajectory_id,
                **outcomes,
                "paired_delta_b1_minus_b0": {
                    key: outcomes["b1"][key] - outcomes["b0"][key]
                    for key in (
                        "relative_rate_percent", "estimated_rate_mbps",
                        "handover_attempts", "handover_successes",
                        "handover_failures", "ping_pong_count",
                        "rlf_recovery_samples", "rlf_starts", "ttt_starts",
                        "ttt_aborts",
                    )
                },
            }
            rows.append(row)
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            print(f"[B1] {index}/{len(pairs)} {row['trace_id']}")

    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["speed_kmh"])].append(row)
    aggregate = {label: _aggregate(rows, label) for label in ("b0", "b1")}
    per_speed = {}
    for speed, selected in sorted(grouped.items()):
        values = {label: _aggregate(selected, label) for label in ("b0", "b1")}
        per_speed[str(speed)] = {**values, "delta_b1_minus_b0": _delta(values["b0"], values["b1"])}
    report = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "status": "PILOT_COMPLETE",
        "scope": "train_only_paired_pilot",
        "routes": len(rows),
        "samples": sum(int(row["b0"]["total_samples"]) for row in rows),
        "policy_contract": {
            "b0": {"a3_off_db": 1.0, "a3_hys_db": 1.0, "a3_ttt_ms": 40},
            "b1": {"a3_off_db": 0.0, "a3_hys_db": 1.0, "a3_ttt_ms": 40},
            "changed_factor": "a3_off_db_1_to_0_only",
            "b0_entering_delta_db": ">2",
            "b1_entering_delta_db": ">1",
            "b1_leaving_delta_db": "<-1",
        },
        "aggregate": {**aggregate, "delta_b1_minus_b0": _delta(aggregate["b0"], aggregate["b1"])},
        "per_speed": per_speed,
        "runtime_seconds": time.perf_counter() - started,
        "inference_warning": (
            "Eight TRAIN routes are a correctness/mechanism pilot; no bootstrap "
            "CI or research-success gate is evaluated."
        ),
    }
    (output_dir / "pilot_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    provenance_path = data_dir / "dataset_provenance.json"
    manifest = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "validation_or_test_rows_read": 0,
        "data_dir": str(data_dir.resolve()),
        "baseline_manifest": {
            "path": str(baseline_manifest.resolve()),
            "sha256": _sha256(baseline_manifest),
        },
        "dataset_provenance": (
            {"path": str(provenance_path.resolve()), "sha256": _sha256(provenance_path)}
            if provenance_path.is_file() else None
        ),
        "source_split": source_manifest["split"],
        "source_sampling": source_manifest["sampling"],
        "selected_routes": source_manifest["selected_routes"],
        "b0_protocol_config": asdict(b0_config),
        "b1_protocol_config": asdict(b1_protocol_config),
        "git": _git_state(root),
        "source_files": [
            {"path": str(path.relative_to(root)), "sha256": _sha256(path)}
            for path in (
                Path(__file__).resolve(),
                root / "src/ho_optim_drl/protocol/reference.py",
                root / "src/ho_optim_drl/config.py",
            )
        ],
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def main(root_path: str, args: Namespace) -> int:
    root = Path(root_path).resolve()
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = root / data_dir
    baseline_manifest = Path(args.baseline_manifest)
    if not baseline_manifest.is_absolute():
        baseline_manifest = root / baseline_manifest
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    report = run_b1_audit(
        root=root, data_dir=data_dir.resolve(),
        baseline_manifest=baseline_manifest.resolve(), output_dir=output_dir.resolve(),
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0
