"""Paired TRAIN-only B2/B3 early-window control pilot."""

from __future__ import annotations

from argparse import Namespace
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any

from ho_optim_drl.commands.audit_b1_lower_threshold import (
    _aggregate,
    _delta,
    _load_pairs,
    _route_metrics,
    b1_config,
)
from ho_optim_drl.commands.audit_baseline_parity import _git_state, _sha256
from ho_optim_drl.data import load_radio_trace
from ho_optim_drl.data.project import load_protocol_config
from ho_optim_drl.protocol import (
    A3ReferenceSimulator,
    EarlyTimingDecisionPoint,
    EarlyTimingHandoverEngine,
)


SCHEMA_NAME = "early_ho_b2_b3_controls_pilot"
SCHEMA_VERSION = 1


def run_early_control(
    engine: EarlyTimingHandoverEngine, *, delay_ticks: int
) -> dict[str, Any]:
    """Run immediate B2 (0) or cancellable fixed-delay B3 (>0)."""
    while True:
        decision = engine.decision_point
        if decision is not None:
            if not isinstance(decision, EarlyTimingDecisionPoint):
                raise RuntimeError("B2/B3 received a non-early decision.")
            if delay_ticks == 0 or decision.stage == "scheduled":
                engine.start_handover()
            elif decision.stage == "entry":
                engine.defer_early_handover(delay_ticks=delay_ticks)
            else:
                raise RuntimeError(f"Unknown early decision stage: {decision.stage}")
        tick = engine.advance_one_tick()
        if tick.terminated:
            return engine.early_audit_state()


def _early_sums(rows: list[dict[str, Any]], policy: str) -> dict[str, Any]:
    states = [row[f"{policy}_early"] for row in rows]
    delays = [
        int(delay)
        for state in states
        for delay in state["execution_delays_ticks"]
    ]
    return {
        "entries": sum(int(state["entries"]) for state in states),
        "schedules": sum(int(state["schedules"]) for state in states),
        "schedule_cancellations": sum(
            int(state["schedule_cancellations"]) for state in states
        ),
        "target_changes": sum(int(state["target_changes"]) for state in states),
        "executions": sum(int(state["executions"]) for state in states),
        "execution_delay_ticks": {
            str(delay): delays.count(delay) for delay in sorted(set(delays))
        },
    }


def run_audit(
    *, root: Path, data_dir: Path, baseline_manifest: Path, output_dir: Path
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing B2/B3 pilot: {output_dir}")
    output_dir.mkdir(parents=True)
    pairs, source_manifest = _load_pairs(baseline_manifest, data_dir)
    b0_config = load_protocol_config(str(data_dir))
    b1_protocol_config = b1_config(b0_config)
    delay_ticks = 40 // b0_config.delta_t_ms
    if delay_ticks * b0_config.delta_t_ms != 40:
        raise ValueError("B3 delay 40 ms must align with dataset dt.")

    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    with (output_dir / "route_outcomes.jsonl").open("w", encoding="utf-8") as stream:
        for index, pair in enumerate(pairs, 1):
            trace = load_radio_trace(
                b0_config, str(data_dir), pair.rsrp_file, pair.sinr_file,
                speed_kmh=pair.speed_kmh,
            )
            b0 = A3ReferenceSimulator(b0_config)
            b0.predict(trace)
            b1 = A3ReferenceSimulator(b1_protocol_config)
            b1.predict(trace)
            b2 = EarlyTimingHandoverEngine(
                b0_config, enter_delta_db=1.0, exit_delta_db=0.0
            )
            b2.reset(trace)
            b2_state = run_early_control(b2, delay_ticks=0)
            b3 = EarlyTimingHandoverEngine(
                b0_config, enter_delta_db=1.0, exit_delta_db=0.0
            )
            b3.reset(trace)
            b3_state = run_early_control(b3, delay_ticks=delay_ticks)
            outcomes = {
                "b0": _route_metrics(b0, trace),
                "b1": _route_metrics(b1, trace),
                "b2": _route_metrics(b2, trace),
                "b3": _route_metrics(b3, trace),
            }
            row = {
                "trace_id": f"{pair.speed_kmh}:{pair.trajectory_id}",
                "speed_kmh": pair.speed_kmh,
                "trajectory_id": pair.trajectory_id,
                **outcomes,
                "b2_early": b2_state,
                "b3_early": b3_state,
            }
            rows.append(row)
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            print(f"[B2/B3] {index}/{len(pairs)} {row['trace_id']}")

    policies = ("b0", "b1", "b2", "b3")
    aggregate = {policy: _aggregate(rows, policy) for policy in policies}
    aggregate["delta_b1_minus_b0"] = _delta(aggregate["b0"], aggregate["b1"])
    aggregate["delta_b2_minus_b0"] = _delta(aggregate["b0"], aggregate["b2"])
    aggregate["delta_b3_minus_b0"] = _delta(aggregate["b0"], aggregate["b3"])
    aggregate["b2_early"] = _early_sums(rows, "b2")
    aggregate["b3_early"] = _early_sums(rows, "b3")

    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["speed_kmh"])].append(row)
    per_speed: dict[str, Any] = {}
    for speed, selected in sorted(grouped.items()):
        values = {policy: _aggregate(selected, policy) for policy in policies}
        per_speed[str(speed)] = {
            **values,
            "delta_b1_minus_b0": _delta(values["b0"], values["b1"]),
            "delta_b2_minus_b0": _delta(values["b0"], values["b2"]),
            "delta_b3_minus_b0": _delta(values["b0"], values["b3"]),
            "b2_early": _early_sums(selected, "b2"),
            "b3_early": _early_sums(selected, "b3"),
        }

    report = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "status": "PILOT_COMPLETE",
        "scope": "train_only_paired_pilot",
        "routes": len(rows),
        "samples": sum(int(row["b0"]["total_samples"]) for row in rows),
        "policy_contract": {
            "b0": "A3 off=1 dB, hys=1 dB, TTT=40 ms",
            "b1": "A3 off=0 dB, hys=1 dB, TTT=40 ms",
            "b2": "early window delta>1 dB; immediate HO to current best neighbor",
            "b3": (
                "early window delta>1 dB; fixed 40 ms delay; cancel at delta<=0 "
                "or target/serving change"
            ),
            "early_decision_type": "EarlyTimingDecisionPoint(trigger=early_window)",
            "unchanged_physical_protocol": (
                "RLM, HO preparation/execution, RLF recovery and rate accounting"
            ),
        },
        "aggregate": aggregate,
        "per_speed": per_speed,
        "runtime_seconds": time.perf_counter() - started,
        "inference_warning": (
            "Eight TRAIN routes are a mechanism pilot; no bootstrap CI or "
            "research-success gate is evaluated."
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
        "early_window": {
            "enter_delta_db_strict": 1.0,
            "exit_delta_db_inclusive": 0.0,
            "b2_delay_ms": 0,
            "b3_delay_ms": 40,
        },
        "git": _git_state(root),
        "source_files": [
            {"path": str(path.relative_to(root)), "sha256": _sha256(path)}
            for path in (
                Path(__file__).resolve(),
                root / "src/ho_optim_drl/protocol/early_timing.py",
                root / "src/ho_optim_drl/protocol/engine.py",
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
    report = run_audit(
        root=root, data_dir=data_dir.resolve(),
        baseline_manifest=baseline_manifest.resolve(), output_dir=output_dir.resolve(),
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0
