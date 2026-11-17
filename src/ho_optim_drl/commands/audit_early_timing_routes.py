"""T3 TRAIN-only full-route opportunity audit with paired bootstrap."""

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

from ho_optim_drl.commands.audit_b1_lower_threshold import (
    _aggregate, _delta, _route_metrics, b1_config,
)
from ho_optim_drl.commands.audit_b2_b3_early_controls import run_early_control
from ho_optim_drl.commands.audit_baseline_parity import _git_state, _sha256
from ho_optim_drl.commands.audit_early_timing_references import (
    _metrics, _shadow_b0_parity, run_reference,
)
from ho_optim_drl.data import TracePair, load_radio_trace
from ho_optim_drl.data.project import load_protocol_config
from ho_optim_drl.protocol import A3ReferenceSimulator, EarlyTimingHandoverEngine


SCHEMA_NAME = "early_ho_timing_t3_full_route"
SCHEMA_VERSION = 1
POLICIES = ("b0", "b0a", "b1", "b2", "b3", "rpost", "rearly")
METRICS = (
    "mean_route_relative_rate_percent",
    "mean_route_estimated_rate_mbps",
    "handover_failure_percent",
    "ping_pong_percent",
    "rlf_time_percent",
)


def _load_route_split(path: Path, data_dir: Path) -> tuple[list[TracePair], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_name") != "early_ho_timing_train_route_split":
        raise ValueError("T3 requires an early-HO T2 route split.")
    if payload.get("split") != "train" or payload.get("validation_or_test_rows_read") != 0:
        raise ValueError("T3 accepts TRAIN-only route splits.")
    pairs = []
    for row in payload["selected_routes"]:
        for kind in ("rsrp", "sinr"):
            source = data_dir / row[f"{kind}_file"]
            if _sha256(source) != row[f"{kind}_sha256"]:
                raise ValueError(f"T3 data hash mismatch: {source}")
        pairs.append(TracePair(
            int(row["speed_kmh"]), int(row["trajectory_id"]),
            str(row["rsrp_file"]), str(row["sinr_file"]),
        ))
    if not pairs:
        raise ValueError("T3 route split is empty.")
    return pairs, payload


def _extended_aggregate(rows: Sequence[dict[str, Any]], policy: str, dt_ms: int) -> dict[str, Any]:
    base = _aggregate(rows, policy)
    metrics = [row[policy] for row in rows]
    minutes = sum(item["total_samples"] for item in metrics) * dt_ms / 60000.0
    rates = [float(item["estimated_rate_mbps"]) for item in metrics]
    base.update({
        "p10_route_estimated_rate_mbps": float(np.percentile(rates, 10)),
        "handover_attempts_per_minute": base["handover_attempts"] / minutes,
        "handover_failures_per_minute": base["handover_failures"] / minutes,
        "ping_pongs_per_minute": base["ping_pong_count"] / minutes,
        "rlf_starts_per_minute": base["rlf_starts"] / minutes,
        "route_minutes": minutes,
    })
    return base


def _sample_rows(
    rows: list[dict[str, Any]], rng: np.random.Generator
) -> list[dict[str, Any]]:
    by_speed: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_speed[int(row["speed_kmh"])].append(row)
    sampled: list[dict[str, Any]] = []
    for speed in sorted(by_speed):
        group = by_speed[speed]
        indices = rng.integers(0, len(group), size=len(group))
        sampled.extend(group[int(index)] for index in indices)
    return sampled


def paired_bootstrap(
    rows: list[dict[str, Any]], *, left: str, right: str,
    dt_ms: int, resamples: int, seed: int,
) -> dict[str, Any]:
    """Bootstrap route clusters within speed and recompute pooled ratios."""
    rng = np.random.default_rng(seed)
    values = {name: [] for name in METRICS}
    for _ in range(resamples):
        sampled = _sample_rows(rows, rng)
        a = _aggregate(sampled, left)
        b = _aggregate(sampled, right)
        delta = _delta(a, b)
        for name in METRICS:
            value = delta[name]
            if value is not None:
                values[name].append(float(value))
    point_left = _extended_aggregate(rows, left, dt_ms)
    point_right = _extended_aggregate(rows, right, dt_ms)
    point = _delta(point_left, point_right)
    return {
        "left": left,
        "right": right,
        "delta_right_minus_left": point,
        "ci95": {
            name: (
                [float(np.percentile(items, 2.5)), float(np.percentile(items, 97.5))]
                if items else None
            )
            for name, items in values.items()
        },
        "resamples": resamples,
        "seed": seed,
        "stratified_by_speed": True,
    }


def evaluate_gates(
    aggregate: dict[str, Any], comparisons: dict[str, Any],
    per_speed: dict[str, Any],
) -> dict[str, Any]:
    physical = comparisons["rearly_minus_b0"]
    early = comparisons["rearly_minus_rpost"]
    delta = physical["delta_right_minus_left"]
    ci = physical["ci95"]
    b0 = aggregate["b0"]
    rearly = aggregate["rearly"]
    hof_reduction = (
        (b0["handover_failure_percent"] - rearly["handover_failure_percent"])
        / b0["handover_failure_percent"]
        if b0["handover_failure_percent"] else None
    )
    pp_reduction = (
        (b0["ping_pong_percent"] - rearly["ping_pong_percent"])
        / b0["ping_pong_percent"]
        if b0["ping_pong_percent"] else None
    )
    strong = bool(
        delta["mean_route_relative_rate_percent"] >= 0.5
        and ci["mean_route_relative_rate_percent"][0] > 0
        and hof_reduction is not None and hof_reduction >= 0.10
        and pp_reduction is not None and pp_reduction >= 0.10
        and ci["handover_failure_percent"][1] <= 0
        and ci["ping_pong_percent"][1] <= 0
        and ci["rlf_time_percent"][1] <= 0
    )
    rate_retention = 100.0 * rearly["mean_route_estimated_rate_mbps"] / b0["mean_route_estimated_rate_mbps"]
    relative_rate_retention = (
        100.0 * rearly["mean_route_relative_rate_percent"]
        / b0["mean_route_relative_rate_percent"]
    )
    per_speed_rate_retention = {
        speed: 100.0 * item["rearly"]["mean_route_estimated_rate_mbps"]
        / item["b0"]["mean_route_estimated_rate_mbps"]
        for speed, item in per_speed.items()
    }
    per_speed_relative_rate_retention = {
        speed: 100.0 * item["rearly"]["mean_route_relative_rate_percent"]
        / item["b0"]["mean_route_relative_rate_percent"]
        for speed, item in per_speed.items()
    }
    qos_guard = bool(
        rate_retention >= 98.0
        and relative_rate_retention >= 98.0
        and all(value >= 98.0 for value in per_speed_rate_retention.values())
        and all(value >= 98.0 for value in per_speed_relative_rate_retention.values())
    )
    early_delta = early["delta_right_minus_left"]
    early_ci = early["ci95"]
    early_pass = bool(
        early_ci["mean_route_relative_rate_percent"][0] > 0
        and early_ci["handover_failure_percent"][1] <= 0
        and early_ci["ping_pong_percent"][1] <= 0
        and early_ci["rlf_time_percent"][1] <= 0
    )
    return {
        "g_physical_strong": {
            "passed": strong,
            "hof_relative_reduction": hof_reduction,
            "pp_relative_reduction": pp_reduction,
        },
        "g_qos_guard": {
            "passed": qos_guard,
            "definition_version": "qos_retention_vs_b0_v2",
            "threshold_percent": 98.0,
            "aggregate_rate_retention_percent": rate_retention,
            "aggregate_relative_rate_retention_percent": relative_rate_retention,
            "per_speed_rate_retention_percent": per_speed_rate_retention,
            "per_speed_relative_rate_retention_percent": per_speed_relative_rate_retention,
            "interpretation": (
                "requires both mean-rate and relative-rate retention versus B0 "
                ">=98% aggregate and at every audited speed"
            ),
        },
        "g_early": {
            "passed": early_pass,
            "delta": early_delta,
        },
        "overall_research_gate": "PASS" if strong and qos_guard else "NOT_PASSED",
    }


def run_audit(
    *, root: Path, data_dir: Path, route_split_path: Path, output_dir: Path,
    delays_ms: tuple[int, ...], horizon_seconds: float,
    bootstrap_resamples: int, bootstrap_seed: int,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite T3 output: {output_dir}")
    output_dir.mkdir(parents=True)
    pairs, route_split = _load_route_split(route_split_path, data_dir)
    config = load_protocol_config(str(data_dir))
    if any(ms < 0 or ms % config.delta_t_ms for ms in delays_ms):
        raise ValueError("T3 delays must align with dt.")
    delay_ticks = tuple(ms // config.delta_t_ms for ms in delays_ms)
    horizon_ticks = round(horizon_seconds * 1000 / config.delta_t_ms)
    lowered = b1_config(config)
    rows: list[dict[str, Any]] = []
    flattened_path = output_dir / "route_outcomes.jsonl"
    started = time.perf_counter()
    with flattened_path.open("w", encoding="utf-8") as stream:
        for index, pair in enumerate(pairs, 1):
            trace = load_radio_trace(
                config, str(data_dir), pair.rsrp_file, pair.sinr_file,
                speed_kmh=pair.speed_kmh,
            )
            b0 = A3ReferenceSimulator(config); b0.predict(trace)
            b1 = A3ReferenceSimulator(lowered); b1.predict(trace)
            b0a = EarlyTimingHandoverEngine(config, enable_early_decisions=False, enable_shadow_decisions=True)
            b0a.reset(trace)
            while True:
                if b0a.decision_point is not None: b0a.start_handover()
                if b0a.advance_one_tick().terminated: break
            parity = _shadow_b0_parity(b0, b0a)
            if not parity["passed"]:
                raise RuntimeError(f"T3 B0A parity failed on {pair}: {parity}")
            b2 = EarlyTimingHandoverEngine(config); b2.reset(trace)
            run_early_control(b2, delay_ticks=0)
            b3 = EarlyTimingHandoverEngine(config); b3.reset(trace)
            run_early_control(b3, delay_ticks=40 // config.delta_t_ms)
            rpost = EarlyTimingHandoverEngine(config, enable_early_decisions=False, enable_shadow_decisions=True)
            rpost.reset(trace)
            rpost_audit = run_reference(rpost, mode="post", delay_ticks=delay_ticks, horizon_ticks=horizon_ticks)
            rearly = EarlyTimingHandoverEngine(config, enable_early_decisions=True, enable_shadow_decisions=True)
            rearly.reset(trace)
            rearly_audit = run_reference(rearly, mode="early", delay_ticks=delay_ticks, horizon_ticks=horizon_ticks)
            row = {
                "trace_id": f"{pair.speed_kmh}:{pair.trajectory_id}",
                "speed_kmh": int(pair.speed_kmh),
                "trajectory_id": int(pair.trajectory_id),
                "b0": _route_metrics(b0, trace), "b0a": _metrics(b0a, trace),
                "b1": _route_metrics(b1, trace), "b2": _metrics(b2, trace),
                "b3": _metrics(b3, trace), "rpost": _metrics(rpost, trace),
                "rearly": _metrics(rearly, trace),
                "rpost_audit": rpost_audit, "rearly_audit": rearly_audit,
            }
            rows.append(row)
            for policy in POLICIES:
                stream.write(json.dumps({
                    "trace_id": row["trace_id"], "speed_kmh": row["speed_kmh"],
                    "trajectory_id": row["trajectory_id"], "policy": policy,
                    "metrics": row[policy],
                    "audit": row.get(f"{policy}_audit"),
                }, allow_nan=False) + "\n")
            print(f"[T3] {index}/{len(pairs)} {row['trace_id']}", flush=True)

    aggregate = {policy: _extended_aggregate(rows, policy, config.delta_t_ms) for policy in POLICIES}
    for policy in POLICIES[1:]:
        aggregate[f"delta_{policy}_minus_b0"] = _delta(aggregate["b0"], aggregate[policy])
    aggregate["delta_rearly_minus_rpost"] = _delta(aggregate["rpost"], aggregate["rearly"])
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows: grouped[int(row["speed_kmh"])].append(row)
    per_speed = {
        str(speed): {policy: _extended_aggregate(group, policy, config.delta_t_ms) for policy in POLICIES}
        for speed, group in sorted(grouped.items())
    }
    comparisons = {
        "rpost_minus_b0": paired_bootstrap(rows, left="b0", right="rpost", dt_ms=config.delta_t_ms, resamples=bootstrap_resamples, seed=bootstrap_seed),
        "rearly_minus_b0": paired_bootstrap(rows, left="b0", right="rearly", dt_ms=config.delta_t_ms, resamples=bootstrap_resamples, seed=bootstrap_seed + 1),
        "rearly_minus_rpost": paired_bootstrap(rows, left="rpost", right="rearly", dt_ms=config.delta_t_ms, resamples=bootstrap_resamples, seed=bootstrap_seed + 2),
        "rearly_minus_b1": paired_bootstrap(rows, left="b1", right="rearly", dt_ms=config.delta_t_ms, resamples=bootstrap_resamples, seed=bootstrap_seed + 3),
        "rearly_minus_b2": paired_bootstrap(rows, left="b2", right="rearly", dt_ms=config.delta_t_ms, resamples=bootstrap_resamples, seed=bootstrap_seed + 4),
        "rearly_minus_b3": paired_bootstrap(rows, left="b3", right="rearly", dt_ms=config.delta_t_ms, resamples=bootstrap_resamples, seed=bootstrap_seed + 5),
    }
    gates = evaluate_gates(aggregate, comparisons, per_speed)
    report = {
        "schema_name": SCHEMA_NAME, "schema_version": SCHEMA_VERSION,
        "status": "COMPLETE", "scope": "train_only", "routes": len(rows),
        "policy_route_rows": len(rows) * len(POLICIES),
        "aggregate": aggregate, "per_speed": per_speed,
        "paired_bootstrap": comparisons, "gates": gates,
        "reference_contract": {
            "delays_ms": list(delays_ms), "horizon_seconds": horizon_seconds,
            "target": "current best L3-RSRP neighbor", "receding_horizon": True,
            "future_is_reference_only": True,
        },
        "runtime_seconds": time.perf_counter() - started,
        "inference_warning": "TRAIN development audit; not validation/test/zero-shot or an online policy result.",
    }
    (output_dir / "main_report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output_dir / "decision_gate.json").write_text(json.dumps(gates, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    provenance = data_dir / "dataset_provenance.json"
    manifest = {
        "schema_name": SCHEMA_NAME, "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "validation_or_test_rows_read": 0, "data_dir": str(data_dir.resolve()),
        "route_split": {"path": str(route_split_path.resolve()), "sha256": _sha256(route_split_path)},
        "dataset_provenance": ({"path": str(provenance.resolve()), "sha256": _sha256(provenance)} if provenance.is_file() else None),
        "selected_routes": route_split["selected_routes"],
        "protocol_config": asdict(config), "bootstrap_resamples": bootstrap_resamples,
        "bootstrap_seed": bootstrap_seed, "git": _git_state(root),
        "source_files": [
            {"path": str(path.relative_to(root)), "sha256": _sha256(path)}
            for path in (Path(__file__).resolve(), root / "src/ho_optim_drl/commands/audit_early_timing_references.py", root / "src/ho_optim_drl/protocol/early_timing.py")
        ],
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main(root_path: str, args: Namespace) -> int:
    root = Path(root_path).resolve(); data_dir = Path(args.data_dir); split = Path(args.route_split); output = Path(args.output_dir)
    if not data_dir.is_absolute(): data_dir = root / data_dir
    if not split.is_absolute(): split = root / split
    if not output.is_absolute(): output = root / output
    report = run_audit(
        root=root, data_dir=data_dir.resolve(), route_split_path=split.resolve(),
        output_dir=output.resolve(), delays_ms=tuple(args.delays_ms),
        horizon_seconds=float(args.reference_horizon),
        bootstrap_resamples=int(args.bootstrap_resamples), bootstrap_seed=int(args.bootstrap_seed),
    )
    print(json.dumps({"status": report["status"], "routes": report["routes"], "gates": report["gates"]}, indent=2), flush=True)
    return 0
