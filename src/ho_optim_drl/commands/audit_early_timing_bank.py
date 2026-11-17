"""Build the T2 TRAIN state bank and local timing counterfactuals."""

from __future__ import annotations

from argparse import Namespace
import copy
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from ho_optim_drl.commands.audit_baseline_parity import _git_state, _sha256
from ho_optim_drl.commands.audit_early_timing_references import simulate_schedule
from ho_optim_drl.commands.audit_policy_learning import _selected_routes
from ho_optim_drl.data import RadioTrace, TracePair, load_radio_trace
from ho_optim_drl.data.project import load_protocol_config
from ho_optim_drl.protocol import (
    EarlyTimingDecisionPoint,
    EarlyTimingHandoverEngine,
    ShadowA3DecisionPoint,
)


SCHEMA_NAME = "early_ho_timing_t2_bank"
SCHEMA_VERSION = 1
SOURCES = ("b0", "b2", "b3")
CATEGORIES = ("early_pre_a3", "a3_pending_ttt", "post_ttt")


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _category(engine: EarlyTimingHandoverEngine) -> str:
    decision = engine.decision_point
    if isinstance(decision, ShadowA3DecisionPoint):
        return "post_ttt"
    if not isinstance(decision, EarlyTimingDecisionPoint):
        raise RuntimeError("T2 source produced a non-timing decision.")
    return "a3_pending_ttt" if engine._shadow_ttt.pending else "early_pre_a3"


def _counter_state(engine: EarlyTimingHandoverEngine) -> dict[str, Any]:
    return {
        name: {
            "cnt": int(counter.cnt),
            "active": bool(counter.active),
            "reached_max": bool(counter.reached_max),
        }
        for name, counter in engine.cntr.items()
    }


def _anchor_record(
    engine: EarlyTimingHandoverEngine, trace: RadioTrace, pair: TracePair,
    *, source: str, category: str,
) -> dict[str, Any]:
    decision = engine.decision_point
    if decision is None:
        raise RuntimeError("Cannot record an anchor without a decision.")
    tic = int(decision.tic)
    serving = int(decision.serving_cell)
    target = int(decision.target_cell)
    radio = {
        "serving_rsrp_db": float(trace.rsrp_a3_db[serving, tic]),
        "target_rsrp_db": float(trace.rsrp_a3_db[target, tic]),
        "serving_sinr_rate_db": float(trace.sinr_rate_db[serving, tic]),
        "target_sinr_rate_db": float(trace.sinr_rate_db[target, tic]),
        "serving_sinr_rlm_db": float(trace.sinr_rlm_db[serving, tic]),
        "target_sinr_rlm_db": float(trace.sinr_rlm_db[target, tic]),
    }
    physical_state = {
        "speed_kmh": int(pair.speed_kmh),
        "trajectory_id": int(pair.trajectory_id),
        "tic": tic,
        "phase": engine.snapshot.phase.value,
        "serving_cell": serving,
        "target_cell": target,
        "decision_trigger": str(getattr(decision, "trigger", "unknown")),
        "decision_stage": str(getattr(decision, "stage", "unknown")),
        "early_episode": engine._early_episode,
        "shadow_matured": engine._shadow_matured,
        "shadow_ttt": {
            "cnt": int(engine._shadow_ttt.cnt),
            "active": bool(engine._shadow_ttt.active),
            "reached_max": bool(engine._shadow_ttt.reached_max),
        },
        "counters": _counter_state(engine),
        "radio": radio,
    }
    fingerprint = _canonical_hash(physical_state)
    return {
        "state_id": f"{source}:{pair.speed_kmh}:{pair.trajectory_id}:{tic}:{fingerprint[:12]}",
        "source": source,
        "category": category,
        "speed_kmh": int(pair.speed_kmh),
        "trajectory_id": int(pair.trajectory_id),
        "tic": tic,
        "time_ms": tic * int(engine.config.delta_t_ms),
        "serving_cell": serving,
        "target_cell": target,
        "decision": {
            "trigger": str(getattr(decision, "trigger", "unknown")),
            "stage": str(getattr(decision, "stage", "unknown")),
            "a3_start_tic": int(decision.a3_start_tic),
            "margin_db": float(decision.a3_margin_db),
        },
        "radio": radio,
        "physical_state_fingerprint": fingerprint,
        "causal_contract": "values at or before anchor tic only",
    }


def _score(seed: int, record: dict[str, Any]) -> int:
    key = (
        f"{seed}|{record['source']}|{record['speed_kmh']}|"
        f"{record['trajectory_id']}|{record['category']}|{record['tic']}"
    )
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest(), 16)


def _retain_candidate(
    pools: dict[str, list[dict[str, Any]]], candidate: dict[str, Any],
    *, seed: int, per_category_cap: int,
) -> None:
    category = str(candidate["record"]["category"])
    candidate["score"] = _score(seed, candidate["record"])
    selected = pools.setdefault(category, [])
    selected.append(candidate)
    selected.sort(key=lambda item: int(item["score"]))
    if len(selected) > per_category_cap:
        selected.pop()


def _balanced_selection(
    pools: dict[str, list[dict[str, Any]]], max_anchors: int
) -> list[dict[str, Any]]:
    chosen: list[dict[str, Any]] = []
    remaining: list[dict[str, Any]] = []
    for category in CATEGORIES:
        values = sorted(pools.get(category, []), key=lambda item: int(item["score"]))
        if values:
            chosen.append(values[0])
            remaining.extend(values[1:])
    remaining.sort(key=lambda item: int(item["score"]))
    chosen.extend(remaining[: max(0, max_anchors - len(chosen))])
    return sorted(chosen[:max_anchors], key=lambda item: int(item["record"]["tic"]))


def _apply_source_action(engine: EarlyTimingHandoverEngine, source: str) -> None:
    decision = engine.decision_point
    if decision is None:
        raise RuntimeError("Source action requires a pending decision.")
    if source == "b0":
        if isinstance(decision, ShadowA3DecisionPoint):
            engine.start_handover()
        else:
            engine.wait_for_redecision(delay_ticks=20 // engine.config.delta_t_ms)
    elif source == "b2":
        engine.start_handover()
    elif source == "b3":
        if isinstance(decision, ShadowA3DecisionPoint) or (
            isinstance(decision, EarlyTimingDecisionPoint)
            and decision.stage == "scheduled"
        ):
            engine.start_handover()
        elif isinstance(decision, EarlyTimingDecisionPoint):
            engine.defer_early_handover(delay_ticks=40 // engine.config.delta_t_ms)
        else:
            raise RuntimeError("B3 received a non-timing decision.")
    else:
        raise ValueError(f"Unknown T2 source: {source}")


def collect_source_anchors(
    config: Any, trace: RadioTrace, pair: TracePair, *, source: str,
    sample_seed: int, max_anchors: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    engine = EarlyTimingHandoverEngine(
        config, enable_early_decisions=True, enable_shadow_decisions=True
    )
    engine.reset(trace)
    pools: dict[str, list[dict[str, Any]]] = {}
    available = {category: 0 for category in CATEGORIES}
    while True:
        if engine.decision_point is not None:
            category = _category(engine)
            available[category] += 1
            record = _anchor_record(engine, trace, pair, source=source, category=category)
            clone = copy.deepcopy(engine, {id(trace): trace})
            _retain_candidate(
                pools, {"record": record, "engine": clone}, seed=sample_seed,
                per_category_cap=max_anchors,
            )
            _apply_source_action(engine, source)
        if engine.advance_one_tick().terminated:
            break
    return _balanced_selection(pools, max_anchors), available


def _selected_route_records(data_dir: Path, pairs: list[TracePair]) -> list[dict[str, Any]]:
    return [
        {
            "speed_kmh": int(pair.speed_kmh),
            "trajectory_id": int(pair.trajectory_id),
            "rsrp_file": pair.rsrp_file,
            "sinr_file": pair.sinr_file,
            "rsrp_sha256": _sha256(data_dir / pair.rsrp_file),
            "sinr_sha256": _sha256(data_dir / pair.sinr_file),
        }
        for pair in pairs
    ]


def run_audit(
    *, root: Path, data_dir: Path, output_dir: Path, speeds: list[int],
    split_seed: int, sample_seed: int, routes_per_speed: int,
    max_anchors_per_source_route: int, delays_ms: tuple[int, ...],
    horizons_seconds: tuple[float, ...],
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite T2 bank: {output_dir}")
    output_dir.mkdir(parents=True)
    pairs = _selected_routes(
        data_dir=str(data_dir), speeds=speeds, split_seed=split_seed,
        sample_seed=sample_seed, routes_per_speed=routes_per_speed,
        train_fraction=0.70, validation_fraction=0.15, test_fraction=0.15,
    )
    config = load_protocol_config(str(data_dir))
    if any(ms < 0 or ms % config.delta_t_ms for ms in delays_ms):
        raise ValueError("T2 delays must be non-negative and align with dt.")
    horizon_ticks = tuple(
        round(seconds * 1000 / config.delta_t_ms) for seconds in horizons_seconds
    )
    if not horizon_ticks or min(horizon_ticks) <= 0:
        raise ValueError("T2 horizons must be positive.")
    delay_ticks = tuple(ms // config.delta_t_ms for ms in delays_ms)

    route_records = _selected_route_records(data_dir, pairs)
    route_split = {
        "schema_name": "early_ho_timing_train_route_split",
        "schema_version": 1,
        "split": "train",
        "validation_or_test_rows_read": 0,
        "split_seed": split_seed,
        "sample_seed": sample_seed,
        "speeds": speeds,
        "routes_per_speed": routes_per_speed,
        "selected_routes": route_records,
    }
    (output_dir / "route_split.json").write_text(
        json.dumps(route_split, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )

    started = time.perf_counter()
    anchor_count = 0
    outcome_count = 0
    category_counts: dict[str, int] = {category: 0 for category in CATEGORIES}
    source_counts: dict[str, int] = {source: 0 for source in SOURCES}
    shortage_rows: list[dict[str, Any]] = []
    fingerprints: dict[str, int] = {}
    simulated_ticks = 0
    with (
        (output_dir / "anchors.jsonl").open("w", encoding="utf-8") as anchor_stream,
        (output_dir / "local_outcomes.jsonl").open("w", encoding="utf-8") as outcome_stream,
    ):
        for route_index, pair in enumerate(pairs, 1):
            trace = load_radio_trace(
                config, str(data_dir), pair.rsrp_file, pair.sinr_file,
                speed_kmh=pair.speed_kmh,
            )
            for source in SOURCES:
                anchors, available = collect_source_anchors(
                    config, trace, pair, source=source, sample_seed=sample_seed,
                    max_anchors=max_anchors_per_source_route,
                )
                shortage_rows.append({
                    "trace_id": f"{pair.speed_kmh}:{pair.trajectory_id}",
                    "source": source,
                    "available_by_category": available,
                    "selected": len(anchors),
                    "shortage": max_anchors_per_source_route - len(anchors),
                })
                for item in anchors:
                    record = item["record"]
                    engine = item["engine"]
                    anchor_stream.write(json.dumps(record, allow_nan=False) + "\n")
                    anchor_count += 1
                    source_counts[source] += 1
                    category_counts[str(record["category"])] += 1
                    fingerprint = str(record["physical_state_fingerprint"])
                    fingerprints[fingerprint] = fingerprints.get(fingerprint, 0) + 1
                    for horizon, ticks in zip(horizons_seconds, horizon_ticks):
                        control, used = simulate_schedule(
                            engine, requested_delay_ticks=None,
                            horizon_ticks=ticks, mode="early",
                        )
                        simulated_ticks += used
                        control_row = {
                            "state_id": record["state_id"],
                            "source": source,
                            "category": record["category"],
                            "speed_kmh": record["speed_kmh"],
                            "trajectory_id": record["trajectory_id"],
                            "horizon_seconds": horizon,
                            "branch": "baseline_continuation",
                            "requested_delay_ms": None,
                            "physical": control,
                        }
                        outcome_stream.write(json.dumps(control_row, allow_nan=False) + "\n")
                        outcome_count += 1
                        for delay_ms, delay in zip(delays_ms, delay_ticks):
                            outcome, used = simulate_schedule(
                                engine, requested_delay_ticks=delay,
                                horizon_ticks=ticks, mode="early",
                            )
                            simulated_ticks += used
                            row = {
                                **{key: control_row[key] for key in (
                                    "state_id", "source", "category", "speed_kmh",
                                    "trajectory_id", "horizon_seconds",
                                )},
                                "branch": "timing_schedule",
                                "requested_delay_ms": delay_ms,
                                "physical": outcome,
                                "delta_vs_control": {
                                    name: float(outcome[name]) - float(control[name])
                                    for name in (
                                        "delivered_mbit", "relative_rate_percent",
                                        "handover_attempts", "handover_failures",
                                        "ping_pong_count", "rlf_starts",
                                        "recovery_seconds", "interruption_seconds",
                                    )
                                },
                            }
                            outcome_stream.write(json.dumps(row, allow_nan=False) + "\n")
                            outcome_count += 1
            print(f"[T2] {route_index}/{len(pairs)} {pair.speed_kmh}:{pair.trajectory_id}", flush=True)

    report = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "status": "COMPLETE",
        "scope": "train_only",
        "routes": len(pairs),
        "sources": list(SOURCES),
        "anchors": anchor_count,
        "local_outcomes": outcome_count,
        "source_counts": source_counts,
        "category_counts": category_counts,
        "same_state_duplicate_fingerprints": sum(
            1 for count in fingerprints.values() if count > 1
        ),
        "selection": {
            "max_anchors_per_source_route": max_anchors_per_source_route,
            "method": "seeded SHA-256 reservoir per category; one/category first, then global hash fill",
            "outcome_blind": True,
            "shortages": shortage_rows,
        },
        "counterfactual": {
            "delays_ms": list(delays_ms),
            "horizons_seconds": list(horizons_seconds),
            "primary_horizon_seconds": 1.0,
            "continuation": "B0 shadow A3+TTT with WAIT20 redecision",
            "simulated_ticks": simulated_ticks,
        },
        "runtime_seconds": time.perf_counter() - started,
    }
    (output_dir / "bank_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    provenance = data_dir / "dataset_provenance.json"
    manifest = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "validation_or_test_rows_read": 0,
        "data_dir": str(data_dir.resolve()),
        "dataset_provenance": ({"path": str(provenance.resolve()), "sha256": _sha256(provenance)} if provenance.is_file() else None),
        "route_split_sha256": _sha256(output_dir / "route_split.json"),
        "protocol_config": asdict(config),
        "git": _git_state(root),
        "source_files": [
            {"path": str(path.relative_to(root)), "sha256": _sha256(path)}
            for path in (
                Path(__file__).resolve(),
                root / "src/ho_optim_drl/commands/audit_early_timing_references.py",
                root / "src/ho_optim_drl/protocol/early_timing.py",
            )
        ],
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return report


def main(root_path: str, args: Namespace) -> int:
    root = Path(root_path).resolve()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    if not data_dir.is_absolute():
        data_dir = root / data_dir
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    report = run_audit(
        root=root, data_dir=data_dir.resolve(), output_dir=output_dir.resolve(),
        speeds=list(args.speeds), split_seed=int(args.split_seed),
        sample_seed=int(args.sample_seed), routes_per_speed=int(args.routes_per_speed),
        max_anchors_per_source_route=int(args.max_anchors_per_source_route),
        delays_ms=tuple(args.delays_ms), horizons_seconds=tuple(args.horizons),
    )
    print(json.dumps(report, indent=2), flush=True)
    return 0
