"""TRAIN-only pilot for the post-A3 and early-HO timing references."""

from __future__ import annotations

from argparse import Namespace
from collections import Counter, defaultdict
import copy
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
from typing import Any, Literal

import numpy as np

from ho_optim_drl.commands.audit_b1_lower_threshold import (
    _aggregate,
    _delta,
    _load_pairs,
    _route_metrics,
    b1_config,
)
from ho_optim_drl.commands.audit_b2_b3_early_controls import run_early_control
from ho_optim_drl.commands.audit_baseline_parity import _git_state, _sha256
from ho_optim_drl.data import RadioTrace, load_radio_trace
from ho_optim_drl.data.project import load_protocol_config
from ho_optim_drl.protocol import (
    A3ReferenceSimulator,
    EarlyTimingDecisionPoint,
    EarlyTimingHandoverEngine,
    ProtocolPhase,
    ShadowA3DecisionPoint,
)


SCHEMA_NAME = "early_ho_timing_references_pilot"
SCHEMA_VERSION = 1
ReferenceMode = Literal["post", "early"]
RISK_FIELDS = ("handover_failures", "ping_pong_count", "recovery_seconds")


def _physical_snapshot(
    totals: dict[str, float | int], *, elapsed_ticks: int, dt_seconds: float,
    terminated: bool,
) -> dict[str, Any]:
    best = float(totals["best_mbit"])
    delivered = float(totals["delivered_mbit"])
    return {
        "delivered_mbit": delivered,
        "best_mbit": best,
        "relative_rate_percent": 100.0 * delivered / best if best > 0 else 0.0,
        "handover_attempts": int(totals["handover_attempts"]),
        "handover_successes": int(totals["handover_successes"]),
        "handover_failures": int(totals["handover_failures"]),
        "ping_pong_count": int(totals["ping_pong_count"]),
        "rlf_starts": int(totals["rlf_starts"]),
        "recovery_seconds": int(totals["recovery_ticks"]) * dt_seconds,
        "interruption_seconds": int(totals["interruption_ticks"]) * dt_seconds,
        "elapsed_seconds": elapsed_ticks * dt_seconds,
        "trace_terminated": bool(terminated),
    }


def _baseline_action(engine: EarlyTimingHandoverEngine, mode: ReferenceMode) -> str:
    """Apply the registered continuation at the pending decision."""
    decision = engine.decision_point
    if decision is None:
        raise RuntimeError("Baseline continuation requires a pending decision.")
    if isinstance(decision, ShadowA3DecisionPoint):
        engine.start_handover()
        return "HO_NOW"
    if mode == "early" and isinstance(decision, EarlyTimingDecisionPoint):
        engine.wait_for_redecision(delay_ticks=20 // engine.config.delta_t_ms)
        return "WAIT_20"
    raise RuntimeError(f"Unexpected {mode} reference decision: {type(decision).__name__}")


def _plan_is_valid(
    engine: EarlyTimingHandoverEngine,
    *,
    trigger: str,
    episode: tuple[int, ...] | None,
) -> bool:
    if episode is None:
        return False
    if trigger == "early_window":
        return engine._early_episode == episode
    if trigger == "shadow_a3_maturity":
        return engine._shadow_matured == episode
    raise ValueError(f"Unknown timing trigger: {trigger}")


def simulate_schedule(
    engine: EarlyTimingHandoverEngine,
    *,
    requested_delay_ticks: int | None,
    horizon_ticks: int,
    mode: ReferenceMode,
) -> tuple[dict[str, Any], int]:
    """Replay one plan, then follow the registered baseline continuation.

    ``None`` denotes baseline continuation itself.  A non-negative delay is
    relative to the current decision; a delayed request is cancelled when its
    original timing episode/target is no longer valid.
    """
    if engine.decision_point is None or engine.trace is None:
        raise RuntimeError("Reference replay requires a pending timing decision.")
    if requested_delay_ticks is not None and requested_delay_ticks < 0:
        raise ValueError("Requested delay must be non-negative or None.")
    if horizon_ticks <= 0:
        raise ValueError("Reference horizon must be positive.")

    trace = engine.trace
    branch = copy.deepcopy(engine, {id(trace): trace})
    anchor = branch.decision_point
    if anchor is None:
        raise RuntimeError("Cloned branch lost its pending decision.")
    anchor_tic = int(anchor.tic)
    trigger = str(getattr(anchor, "trigger", ""))
    episode: tuple[int, ...] | None
    if isinstance(anchor, EarlyTimingDecisionPoint):
        episode = branch._early_episode
    elif isinstance(anchor, ShadowA3DecisionPoint):
        episode = branch._shadow_matured
    else:
        raise RuntimeError("Reference received a non-timing decision.")

    executed_tic: int | None = None
    cancelled = False
    if requested_delay_ticks is None:
        first_action = _baseline_action(branch, mode)
        plan_pending = False
    elif requested_delay_ticks == 0:
        branch.start_handover()
        first_action = "HO_NOW"
        executed_tic = anchor_tic
        plan_pending = False
    else:
        branch.wait_for_redecision(delay_ticks=20 // branch.config.delta_t_ms)
        first_action = "WAIT_20"
        plan_pending = True

    dt_seconds = branch.config.delta_t_ms / 1000.0
    totals: dict[str, float | int] = {
        "delivered_mbit": 0.0,
        "best_mbit": 0.0,
        "handover_attempts": 0,
        "handover_successes": 0,
        "handover_failures": 0,
        "ping_pong_count": 0,
        "rlf_starts": 0,
        "recovery_ticks": 0,
        "interruption_ticks": 0,
    }
    elapsed = 0
    terminated = False
    while elapsed < horizon_ticks and not terminated:
        tick = branch.advance_one_tick()
        elapsed += 1
        totals["delivered_mbit"] += tick.delivered_rate_mbps * dt_seconds
        totals["best_mbit"] += tick.best_rate_mbps * dt_seconds
        totals["handover_attempts"] += int(tick.handover_started)
        totals["handover_successes"] += int(tick.handover_succeeded)
        totals["handover_failures"] += int(tick.handover_failed)
        totals["ping_pong_count"] += int(tick.ping_pong_detected)
        totals["rlf_starts"] += int(tick.rlf_started)
        totals["recovery_ticks"] += int(
            tick.snapshot.phase is ProtocolPhase.RLF_RECOVERY
        )
        totals["interruption_ticks"] += int(
            tick.snapshot.phase
            in (ProtocolPhase.HO_PREPARATION, ProtocolPhase.HO_EXECUTION)
        )
        terminated = tick.terminated
        if plan_pending and not _plan_is_valid(
            branch, trigger=trigger, episode=episode
        ):
            cancelled = True
            plan_pending = False
        if terminated or branch.decision_point is None:
            continue

        if plan_pending:
            deadline_met = branch.decision_point.tic >= (
                anchor_tic + int(requested_delay_ticks)
            )
            if deadline_met:
                executed_tic = int(branch.decision_point.tic)
                branch.start_handover()
                plan_pending = False
            else:
                branch.wait_for_redecision(
                    delay_ticks=20 // branch.config.delta_t_ms
                )
        else:
            _baseline_action(branch, mode)

    physical = _physical_snapshot(
        totals, elapsed_ticks=elapsed, dt_seconds=dt_seconds,
        terminated=terminated,
    )
    physical["requested_delay_ms"] = (
        None if requested_delay_ticks is None
        else requested_delay_ticks * branch.config.delta_t_ms
    )
    physical["executed_tic"] = executed_tic
    physical["actual_delay_ms"] = (
        None if executed_tic is None else (executed_tic - anchor_tic) * branch.config.delta_t_ms
    )
    physical["request_cancelled"] = cancelled
    physical["first_action"] = first_action
    physical["horizon_censored"] = elapsed < horizon_ticks
    return physical, elapsed


def _no_worse(candidate: dict[str, Any], control: dict[str, Any]) -> bool:
    return all(
        float(candidate[name]) <= float(control[name]) + 1e-12
        for name in RISK_FIELDS
    )


def choose_schedule(
    control: dict[str, Any], candidates: list[dict[str, Any]]
) -> dict[str, Any]:
    """Choose safe data gain; exact/tied outcomes retain baseline."""
    feasible = [row for row in candidates if _no_worse(row, control)]
    improving = [
        row for row in feasible
        if float(row["delivered_mbit"]) > float(control["delivered_mbit"]) + 1e-12
    ]
    if not improving:
        return {"kind": "baseline", "delay_ticks": None, "physical": control}
    selected = max(
        improving,
        key=lambda row: (
            float(row["delivered_mbit"]),
            -int(row["requested_delay_ticks"]),
        ),
    )
    return {
        "kind": "schedule",
        "delay_ticks": int(selected["requested_delay_ticks"]),
        "physical": selected,
    }


def run_reference(
    engine: EarlyTimingHandoverEngine,
    *,
    mode: ReferenceMode,
    delay_ticks: tuple[int, ...],
    horizon_ticks: int,
) -> dict[str, Any]:
    """Run a full route with finite-lookahead receding-horizon decisions."""
    decisions = 0
    simulated_branches = 0
    simulated_ticks = 0
    selected = Counter()
    decisions_by_trigger = Counter()
    executed_by_trigger = Counter()
    while True:
        decision = engine.decision_point
        if decision is not None:
            decisions += 1
            trigger = str(getattr(decision, "trigger", "unknown"))
            decisions_by_trigger[trigger] += 1
            control, ticks = simulate_schedule(
                engine, requested_delay_ticks=None, horizon_ticks=horizon_ticks,
                mode=mode,
            )
            simulated_branches += 1
            simulated_ticks += ticks
            candidates: list[dict[str, Any]] = []
            for delay in delay_ticks:
                outcome, ticks = simulate_schedule(
                    engine, requested_delay_ticks=delay,
                    horizon_ticks=horizon_ticks, mode=mode,
                )
                outcome["requested_delay_ticks"] = delay
                candidates.append(outcome)
                simulated_branches += 1
                simulated_ticks += ticks
            choice = choose_schedule(control, candidates)
            if choice["kind"] == "baseline":
                action = _baseline_action(engine, mode)
                selected["baseline"] += 1
            elif int(choice["delay_ticks"]) == 0:
                engine.start_handover()
                action = "HO_NOW"
                selected["delay_0_ms"] += 1
            else:
                engine.wait_for_redecision(
                    delay_ticks=20 // engine.config.delta_t_ms
                )
                action = "WAIT_20"
                selected[f"delay_{int(choice['delay_ticks']) * engine.config.delta_t_ms}_ms"] += 1
            if action == "HO_NOW":
                executed_by_trigger[trigger] += 1
        tick = engine.advance_one_tick()
        if tick.terminated:
            break
    return {
        "decisions": decisions,
        "simulated_branches": simulated_branches,
        "simulated_ticks": simulated_ticks,
        "selected_schedules": dict(selected),
        "decisions_by_trigger": dict(decisions_by_trigger),
        "executed_ho_by_trigger": dict(executed_by_trigger),
        "shadow": engine.early_audit_state(),
    }


def _metrics(engine: EarlyTimingHandoverEngine, trace: RadioTrace) -> dict[str, Any]:
    metrics = _route_metrics(engine, trace)
    metrics["ttt_starts"] = len(engine._shadow_ttt.start_idxs)
    metrics["ttt_aborts"] = len(engine._shadow_ttt.aborted_idxs)
    return metrics


def _shadow_b0_parity(
    b0: A3ReferenceSimulator, b0a: EarlyTimingHandoverEngine
) -> dict[str, Any]:
    names = ("bs_idxs", "ho_idxs", "rlfr_idxs", "successful_ho_history", "ping_pong_idxs")
    mismatches = []
    for name in names:
        left = np.asarray(getattr(b0, name))
        right = np.asarray(getattr(b0a, name))
        if not np.array_equal(left, right, equal_nan=True):
            mismatches.append(name)
    for name in ("n310", "n311", "t310", "rlfr", "ho_prep", "ho_exec"):
        for field in ("start_idxs", "done_idxs", "error_idxs"):
            if list(getattr(b0.cntr[name], field)) != list(getattr(b0a.cntr[name], field)):
                mismatches.append(f"{name}.{field}")
    if b0.cntr["tttc"].start_idxs != b0a._shadow_ttt.start_idxs:
        mismatches.append("ttt.start_idxs")
    if b0.cntr["tttc"].terminated_idxs != b0a._shadow_ttt.terminated_idxs:
        mismatches.append("ttt.terminated_idxs")
    return {"passed": not mismatches, "mismatches": mismatches}


def run_audit(
    *, root: Path, data_dir: Path, baseline_manifest: Path, output_dir: Path,
    delays_ms: tuple[int, ...] = (0, 20, 40, 80, 160, 320),
    horizon_seconds: float = 1.0,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite timing reference pilot: {output_dir}")
    output_dir.mkdir(parents=True)
    pairs, source_manifest = _load_pairs(baseline_manifest, data_dir)
    config = load_protocol_config(str(data_dir))
    lowered = b1_config(config)
    if any(ms < 0 or ms % config.delta_t_ms for ms in delays_ms):
        raise ValueError("Every delay must be non-negative and align with dt.")
    horizon_ticks = round(horizon_seconds * 1000 / config.delta_t_ms)
    if horizon_ticks <= 0:
        raise ValueError("Reference horizon must contain at least one tick.")
    delay_ticks = tuple(ms // config.delta_t_ms for ms in delays_ms)

    rows: list[dict[str, Any]] = []
    route_costs: list[dict[str, Any]] = []
    started = time.perf_counter()
    try:
        import psutil  # type: ignore[import-not-found]
        process = psutil.Process(os.getpid())
    except (ImportError, OSError):
        process = None
    observed_peak_rss_bytes = 0
    rows_path = output_dir / "route_outcomes.jsonl"
    with rows_path.open("w", encoding="utf-8") as stream:
        for index, pair in enumerate(pairs, 1):
            route_started = time.perf_counter()
            trace = load_radio_trace(
                config, str(data_dir), pair.rsrp_file, pair.sinr_file,
                speed_kmh=pair.speed_kmh,
            )
            b0 = A3ReferenceSimulator(config)
            b0.predict(trace)
            b1 = A3ReferenceSimulator(lowered)
            b1.predict(trace)
            b0a = EarlyTimingHandoverEngine(
                config, enable_early_decisions=False, enable_shadow_decisions=True
            )
            b0a.reset(trace)
            while True:
                if b0a.decision_point is not None:
                    b0a.start_handover()
                if b0a.advance_one_tick().terminated:
                    break
            parity = _shadow_b0_parity(b0, b0a)
            if not parity["passed"]:
                raise RuntimeError(f"Shadow B0 parity failed for {pair}: {parity}")

            b2 = EarlyTimingHandoverEngine(config)
            b2.reset(trace)
            b2_state = run_early_control(b2, delay_ticks=0)
            b3 = EarlyTimingHandoverEngine(config)
            b3.reset(trace)
            b3_state = run_early_control(b3, delay_ticks=40 // config.delta_t_ms)
            rpost = EarlyTimingHandoverEngine(
                config, enable_early_decisions=False, enable_shadow_decisions=True
            )
            rpost.reset(trace)
            rpost_state = run_reference(
                rpost, mode="post", delay_ticks=delay_ticks,
                horizon_ticks=horizon_ticks,
            )
            rearly = EarlyTimingHandoverEngine(
                config, enable_early_decisions=True, enable_shadow_decisions=True
            )
            rearly.reset(trace)
            rearly_state = run_reference(
                rearly, mode="early", delay_ticks=delay_ticks,
                horizon_ticks=horizon_ticks,
            )
            outcomes = {
                "b0": _route_metrics(b0, trace),
                "b0a": _metrics(b0a, trace),
                "b1": _route_metrics(b1, trace),
                "b2": _metrics(b2, trace),
                "b3": _metrics(b3, trace),
                "rpost": _metrics(rpost, trace),
                "rearly": _metrics(rearly, trace),
            }
            route_seconds = time.perf_counter() - route_started
            row = {
                "trace_id": f"{pair.speed_kmh}:{pair.trajectory_id}",
                "speed_kmh": pair.speed_kmh,
                "trajectory_id": pair.trajectory_id,
                **outcomes,
                "b0a_parity": parity,
                "b2_audit": b2_state,
                "b3_audit": b3_state,
                "rpost_audit": rpost_state,
                "rearly_audit": rearly_state,
                "runtime_seconds": route_seconds,
            }
            rows.append(row)
            route_costs.append({
                "trace_id": row["trace_id"],
                "samples": trace.num_samples,
                "runtime_seconds": route_seconds,
                "rpost_decisions": rpost_state["decisions"],
                "rearly_decisions": rearly_state["decisions"],
                "simulated_branches": rpost_state["simulated_branches"] + rearly_state["simulated_branches"],
                "simulated_ticks": rpost_state["simulated_ticks"] + rearly_state["simulated_ticks"],
            })
            if process is not None:
                observed_peak_rss_bytes = max(
                    observed_peak_rss_bytes, int(process.memory_info().rss)
                )
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            print(f"[Rpost/Rearly] {index}/{len(pairs)} {row['trace_id']}", flush=True)

    policies = ("b0", "b0a", "b1", "b2", "b3", "rpost", "rearly")
    aggregate = {policy: _aggregate(rows, policy) for policy in policies}
    for policy in policies[1:]:
        aggregate[f"delta_{policy}_minus_b0"] = _delta(aggregate["b0"], aggregate[policy])
    aggregate["delta_rearly_minus_rpost"] = _delta(aggregate["rpost"], aggregate["rearly"])
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["speed_kmh"])].append(row)
    per_speed: dict[str, Any] = {}
    for speed, selected_rows in sorted(grouped.items()):
        values = {policy: _aggregate(selected_rows, policy) for policy in policies}
        values["delta_rearly_minus_rpost"] = _delta(values["rpost"], values["rearly"])
        values["delta_rearly_minus_b0"] = _delta(values["b0"], values["rearly"])
        per_speed[str(speed)] = values

    total_runtime = time.perf_counter() - started
    report = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "status": "PILOT_COMPLETE",
        "scope": "train_only_paired_pilot",
        "routes": len(rows),
        "samples": sum(int(row["b0"]["total_samples"]) for row in rows),
        "shadow_b0_parity_passed": all(row["b0a_parity"]["passed"] for row in rows),
        "reference_contract": {
            "rpost": "finite lookahead only at shadow A3+TTT decisions",
            "rearly": "same lookahead at early-window and shadow A3+TTT decisions",
            "target": "current best L3-RSRP neighbor only",
            "delays_ms": list(delays_ms),
            "decision_cadence_ms": 20,
            "horizon_seconds": horizon_seconds,
            "selection": "no worse HOF/PP/recovery than baseline continuation; maximize delivered Mbit; ties retain baseline then shortest delay",
            "execution": "receding horizon; execute only HO_NOW or WAIT_20 and replan",
            "future_access": "reference branches only; no observation or policy input",
        },
        "aggregate": aggregate,
        "per_speed": per_speed,
        "inference_warning": "Eight TRAIN routes are a correctness/cost pilot without bootstrap CI.",
    }
    (output_dir / "pilot_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    runtime_values = [row["runtime_seconds"] for row in route_costs]
    cost_report = {
        "routes": route_costs,
        "total_runtime_seconds": total_runtime,
        "median_route_runtime_seconds": float(np.median(runtime_values)),
        "worst_route_runtime_seconds": max(runtime_values),
        "simulated_branches": sum(row["simulated_branches"] for row in route_costs),
        "simulated_ticks": sum(row["simulated_ticks"] for row in route_costs),
        "observed_peak_rss_bytes": (
            observed_peak_rss_bytes if process is not None else None
        ),
        "memory_measurement": (
            "psutil RSS sampled after each route" if process is not None
            else "unavailable: psutil not installed"
        ),
        "route_outcomes_bytes": rows_path.stat().st_size,
        "linear_80_route_runtime_projection_seconds": total_runtime * 10.0,
    }
    (output_dir / "cost_report.json").write_text(
        json.dumps(cost_report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    provenance_path = data_dir / "dataset_provenance.json"
    manifest = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "validation_or_test_rows_read": 0,
        "data_dir": str(data_dir.resolve()),
        "baseline_manifest": {"path": str(baseline_manifest.resolve()), "sha256": _sha256(baseline_manifest)},
        "dataset_provenance": ({"path": str(provenance_path.resolve()), "sha256": _sha256(provenance_path)} if provenance_path.is_file() else None),
        "source_split": source_manifest["split"],
        "source_sampling": source_manifest["sampling"],
        "selected_routes": source_manifest["selected_routes"],
        "protocol_config": asdict(config),
        "reference": report["reference_contract"],
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
    baseline = Path(args.baseline_manifest)
    output = Path(args.output_dir)
    if not data_dir.is_absolute():
        data_dir = root / data_dir
    if not baseline.is_absolute():
        baseline = root / baseline
    if not output.is_absolute():
        output = root / output
    report = run_audit(
        root=root, data_dir=data_dir.resolve(), baseline_manifest=baseline.resolve(),
        output_dir=output.resolve(), delays_ms=tuple(args.delays_ms),
        horizon_seconds=float(args.reference_horizon),
    )
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    return 0
