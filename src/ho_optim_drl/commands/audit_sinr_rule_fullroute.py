"""Full-route TRAIN audit of the post-TTT target-SINR-worse rule."""

from __future__ import annotations

from argparse import Namespace
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any

from ho_optim_drl.commands.audit_baseline_parity import _git_state, _sha256
from ho_optim_drl.commands.audit_b1_lower_threshold import _route_metrics
from ho_optim_drl.commands.audit_early_timing_learnability import (
    _rule_wait,
    _run_online,
    feature_names,
)
from ho_optim_drl.commands.audit_early_timing_references import (
    _metrics,
    _shadow_b0_parity,
)
from ho_optim_drl.commands.audit_early_timing_routes import (
    _extended_aggregate,
    paired_bootstrap,
)
from ho_optim_drl.data import TracePair, load_radio_trace
from ho_optim_drl.data.project import load_protocol_config
from ho_optim_drl.protocol import A3ReferenceSimulator, EarlyTimingHandoverEngine


SCHEMA_NAME = "early_ho_sinr_rule_fullroute_confirmation"
SCHEMA_VERSION = 1
POLICIES = ("b0", "b0a", "sinr_rule")
FEATURE_SCHEMA = "t4b_protocol_history_v1"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _confirmation_pairs(
    route_split_path: Path,
    data_dir: Path,
    *,
    expected_routes_per_speed: int = 10,
) -> tuple[list[TracePair], dict[str, Any], list[dict[str, Any]]]:
    payload = _read_json(route_split_path)
    if (
        payload.get("schema_name") != "early_ho_post_ttt_train_route_split"
        or payload.get("split") != "train"
        or payload.get("validation_or_test_rows_read") != 0
    ):
        raise ValueError("SINR full-route audit requires the post-TTT TRAIN route split.")
    selected = [
        row for row in payload.get("selected_routes", [])
        if row.get("fold") == "confirmation"
    ]
    if not selected:
        raise ValueError("Post-TTT split has no confirmation routes.")
    counts: dict[int, int] = defaultdict(int)
    pairs = []
    seen: set[tuple[int, int]] = set()
    for row in selected:
        route = (int(row["speed_kmh"]), int(row["trajectory_id"]))
        if route in seen:
            raise ValueError(f"Duplicate confirmation route: {route}")
        seen.add(route)
        counts[route[0]] += 1
        for kind in ("rsrp", "sinr"):
            source = data_dir / str(row[f"{kind}_file"])
            if not source.is_file() or _sha256(source) != row[f"{kind}_sha256"]:
                raise ValueError(f"Confirmation route hash mismatch: {source}")
        pairs.append(TracePair(
            route[0], route[1], str(row["rsrp_file"]), str(row["sinr_file"])
        ))
    expected = {speed: expected_routes_per_speed for speed in sorted(counts)}
    if dict(sorted(counts.items())) != expected:
        raise ValueError(
            f"Expected {expected_routes_per_speed} confirmation routes/speed, got {dict(counts)}"
        )
    return sorted(pairs, key=lambda pair: (pair.speed_kmh, pair.trajectory_id)), payload, selected


def _run_b0a(
    config: Any, trace: Any,
) -> tuple[EarlyTimingHandoverEngine, dict[str, int]]:
    engine = EarlyTimingHandoverEngine(
        config, enable_early_decisions=False, enable_shadow_decisions=True
    )
    engine.reset(trace)
    decisions = 0
    while True:
        if engine.decision_point is not None:
            decisions += 1
            engine.start_handover()
        if engine.advance_one_tick().terminated:
            break
    return engine, {"decisions": decisions, "handover_actions": decisions, "wait_actions": 0}


def _fullroute_parity(
    b0: A3ReferenceSimulator,
    b0a: EarlyTimingHandoverEngine,
    trace: Any,
) -> dict[str, Any]:
    """Accept only the known terminal-censored preparation bookkeeping delta.

    The vectorized B0 can register an A3-matured preparation on the final trace
    sample. The online engine terminates on that sample before exposing a new
    decision. It cannot start HO execution or affect any study metric. Every
    other mismatch remains fatal.
    """
    raw = _shadow_b0_parity(b0, b0a)
    if raw["passed"]:
        return {"passed": True, "mode": "exact", "raw": raw}
    left_starts = list(b0.cntr["ho_prep"].start_idxs)
    right_starts = list(b0a.cntr["ho_prep"].start_idxs)
    terminal_only = bool(
        raw["mismatches"] == ["ho_prep.start_idxs"]
        and left_starts == [*right_starts, trace.num_samples - 1]
    )
    left_metrics = _route_metrics(b0, trace)
    right_metrics = _metrics(b0a, trace)
    study_fields = (
        "relative_rate_percent", "estimated_rate_mbps", "max_estimated_rate_mbps",
        "total_samples", "rlf_recovery_samples", "handover_attempts",
        "handover_successes", "handover_failures", "ping_pong_count", "rlf_starts",
    )
    metrics_equal = all(left_metrics[name] == right_metrics[name] for name in study_fields)
    if terminal_only and metrics_equal:
        return {
            "passed": True,
            "mode": "terminal_censored_preparation",
            "terminal_tic": trace.num_samples - 1,
            "study_metrics_equal": True,
            "raw": raw,
        }
    return {
        "passed": False,
        "mode": "mismatch",
        "terminal_only": terminal_only,
        "study_metrics_equal": metrics_equal,
        "raw": raw,
    }


def _evaluate_gates(
    aggregate: dict[str, Any],
    per_speed: dict[str, Any],
    comparison: dict[str, Any],
    *,
    parity_passed: bool,
) -> dict[str, Any]:
    b0 = aggregate["b0"]
    rule = aggregate["sinr_rule"]
    delta = comparison["delta_right_minus_left"]
    ci = comparison["ci95"]
    rate_retention = 100.0 * rule["mean_route_estimated_rate_mbps"] / b0["mean_route_estimated_rate_mbps"]
    relative_rate_retention = 100.0 * rule["mean_route_relative_rate_percent"] / b0["mean_route_relative_rate_percent"]
    per_speed_rate_retention = {
        speed: 100.0 * values["sinr_rule"]["mean_route_estimated_rate_mbps"]
        / values["b0"]["mean_route_estimated_rate_mbps"]
        for speed, values in per_speed.items()
    }
    per_speed_relative_rate_retention = {
        speed: 100.0 * values["sinr_rule"]["mean_route_relative_rate_percent"]
        / values["b0"]["mean_route_relative_rate_percent"]
        for speed, values in per_speed.items()
    }
    qos_98 = bool(
        rate_retention >= 98.0
        and relative_rate_retention >= 98.0
        and all(value >= 98.0 for value in per_speed_rate_retention.values())
        and all(value >= 98.0 for value in per_speed_relative_rate_retention.values())
    )
    qos_gain = bool(ci["mean_route_relative_rate_percent"][0] > 0)
    hof_nonworse = bool(ci["handover_failure_percent"][1] <= 0)
    pp_nonworse = bool(ci["ping_pong_percent"][1] <= 0)
    rlf_nonworse = bool(ci["rlf_time_percent"][1] <= 0)
    return {
        "g_b0_parity": {"passed": parity_passed},
        "g_qos_98": {
            "passed": qos_98,
            "rate_retention_percent": rate_retention,
            "relative_rate_retention_percent": relative_rate_retention,
            "per_speed_rate_retention_percent": per_speed_rate_retention,
            "per_speed_relative_rate_retention_percent": per_speed_relative_rate_retention,
        },
        "g_qos_gain": {"passed": qos_gain, "delta": delta["mean_route_relative_rate_percent"], "ci95": ci["mean_route_relative_rate_percent"]},
        "g_hof_nonworse": {"passed": hof_nonworse, "delta": delta["handover_failure_percent"], "ci95": ci["handover_failure_percent"]},
        "g_ping_pong_nonworse": {"passed": pp_nonworse, "delta": delta["ping_pong_percent"], "ci95": ci["ping_pong_percent"]},
        "g_rlf_time_nonworse": {"passed": rlf_nonworse, "delta": delta["rlf_time_percent"], "ci95": ci["rlf_time_percent"]},
        "strict_all": "PASSED" if all((parity_passed, qos_98, qos_gain, hof_nonworse, pp_nonworse, rlf_nonworse)) else "NOT_PASSED",
        "ppo_status": "CLOSED",
    }


def run_audit(
    *,
    root: Path,
    data_dir: Path,
    route_split_path: Path,
    output_dir: Path,
    bootstrap_resamples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite SINR full-route audit: {output_dir}")
    if bootstrap_resamples <= 0:
        raise ValueError("bootstrap_resamples must be positive.")
    pairs, route_split, selected_rows = _confirmation_pairs(
        route_split_path, data_dir
    )
    config = load_protocol_config(str(data_dir))
    names = feature_names(FEATURE_SCHEMA)
    rows: list[dict[str, Any]] = []
    output_dir.mkdir(parents=True)
    route_path = output_dir / "route_outcomes.jsonl"
    started = time.perf_counter()
    with route_path.open("w", encoding="utf-8") as stream:
        for index, pair in enumerate(pairs, 1):
            trace = load_radio_trace(
                config, str(data_dir), pair.rsrp_file, pair.sinr_file,
                speed_kmh=pair.speed_kmh,
            )
            b0 = A3ReferenceSimulator(config)
            b0.predict(trace)
            b0a, b0a_actions = _run_b0a(config, trace)
            parity = _fullroute_parity(b0, b0a, trace)
            if not parity["passed"]:
                raise RuntimeError(f"B0/B0A parity failed on {pair}: {parity}")
            rule_metrics, rule_actions = _run_online(
                trace,
                config,
                int(pair.speed_kmh),
                lambda feature: _rule_wait(
                    feature, "target_sinr_worse", names=names
                ),
                feature_schema=FEATURE_SCHEMA,
            )
            row = {
                "trace_id": f"{pair.speed_kmh}:{pair.trajectory_id}",
                "speed_kmh": int(pair.speed_kmh),
                "trajectory_id": int(pair.trajectory_id),
                "fold": "confirmation",
                "b0": _route_metrics(b0, trace),
                "b0a": _metrics(b0a, trace),
                "sinr_rule": rule_metrics,
                "b0a_actions": b0a_actions,
                "sinr_rule_actions": rule_actions,
                "parity": parity,
            }
            rows.append(row)
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            print(
                f"[SINR full-route] {index}/{len(pairs)} {row['trace_id']} "
                f"wait={rule_actions['wait_actions']}",
                flush=True,
            )

    aggregate = {
        policy: _extended_aggregate(rows, policy, config.delta_t_ms)
        for policy in POLICIES
    }
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["speed_kmh"])].append(row)
    per_speed = {
        str(speed): {
            policy: _extended_aggregate(group, policy, config.delta_t_ms)
            for policy in POLICIES
        }
        for speed, group in sorted(grouped.items())
    }
    comparison = paired_bootstrap(
        rows, left="b0", right="sinr_rule", dt_ms=config.delta_t_ms,
        resamples=bootstrap_resamples, seed=bootstrap_seed,
    )
    gates = _evaluate_gates(
        aggregate, per_speed, comparison,
        parity_passed=all(row["parity"]["passed"] for row in rows),
    )
    total_waits = sum(int(row["sinr_rule_actions"]["wait_actions"]) for row in rows)
    total_decisions = sum(int(row["sinr_rule_actions"]["decisions"]) for row in rows)
    frozen_outcome_blind = bool(route_split.get("selection_contract"))
    inference_warning = (
        "Fresh outcome-blind TRAIN confirmation split, frozen before this "
        "full-route audit and disjoint from the declared prior route splits. "
        "It is still TRAIN-only evidence, not validation/test/zero-shot."
        if frozen_outcome_blind else
        "TRAIN development confirmation reused after T4C/T4D; not unseen "
        "validation/test/zero-shot. Full-route dynamics are valid, but this "
        "is not a fresh confirmatory dataset."
    )
    report = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "status": "COMPLETE",
        "scope": "post_ttt_train_confirmation_full_routes",
        "validation_or_test_rows_read": 0,
        "routes": len(rows),
        "routes_per_speed": {
            str(speed): len(group) for speed, group in sorted(grouped.items())
        },
        "policy_route_rows": len(rows) * len(POLICIES),
        "policy_contract": {
            "b0": "3GPP A3 reference",
            "b0a": "shadow-A3 immediate HO parity control",
            "sinr_rule": "WAIT_20 while target SINR < serving SINR; otherwise HO_NOW",
            "feature_schema": FEATURE_SCHEMA,
            "target": "current shadow-A3 target",
            "future_features": [],
            "threshold_or_model_fit": False,
        },
        "actions": {
            "decisions": total_decisions,
            "wait_actions": total_waits,
            "wait_fraction": total_waits / total_decisions if total_decisions else 0.0,
        },
        "aggregate": aggregate,
        "per_speed": per_speed,
        "paired_bootstrap": {"sinr_rule_minus_b0": comparison},
        "gates": gates,
        "runtime_seconds": time.perf_counter() - started,
        "route_split_frozen_outcome_blind": frozen_outcome_blind,
        "inference_warning": inference_warning,
    }
    _write_json(output_dir / "main_report.json", report)
    _write_json(output_dir / "decision_gate.json", gates)
    provenance = data_dir / "dataset_provenance.json"
    manifest = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "validation_or_test_rows_read": 0,
        "data_dir": str(data_dir.resolve()),
        "route_split": {
            "path": str(route_split_path.resolve()),
            "sha256": _sha256(route_split_path),
        },
        "dataset_provenance": (
            {"path": str(provenance.resolve()), "sha256": _sha256(provenance)}
            if provenance.is_file() else None
        ),
        "selected_fold": "confirmation",
        "selected_routes": selected_rows,
        "protocol_config": asdict(config),
        "bootstrap_resamples": bootstrap_resamples,
        "bootstrap_seed": bootstrap_seed,
        "outputs": {
            name: {"path": str((output_dir / name).resolve()), "sha256": _sha256(output_dir / name)}
            for name in ("route_outcomes.jsonl", "main_report.json", "decision_gate.json")
        },
        "source_files": [
            {"path": str(path.relative_to(root)), "sha256": _sha256(path)}
            for path in (
                Path(__file__).resolve(),
                root / "src/ho_optim_drl/commands/audit_early_timing_learnability.py",
                root / "src/ho_optim_drl/protocol/early_timing.py",
            )
        ],
        "git": _git_state(root),
    }
    _write_json(output_dir / "run_manifest.json", manifest)
    delta = comparison["delta_right_minus_left"]
    ci = comparison["ci95"]
    lines = [
        "# SINR rule full-route audit — post-TTT confirmation routes",
        "",
        f"- Routes: **{len(rows)}** (10/speed); validation/test rows: **0**.",
        f"- Rule actions: **{total_waits}/{total_decisions} WAIT** ({100.0 * report['actions']['wait_fraction']:.2f}%).",
        f"- RR delta: **{delta['mean_route_relative_rate_percent']:+.4f} pp**, CI95% **{ci['mean_route_relative_rate_percent']}**.",
        f"- HOF delta: **{delta['handover_failure_percent']:+.4f} pp**, CI95% **{ci['handover_failure_percent']}**.",
        f"- PP delta: **{delta['ping_pong_percent']:+.4f} pp**, CI95% **{ci['ping_pong_percent']}**.",
        f"- RLF-time delta: **{delta['rlf_time_percent']:+.4f} pp**, CI95% **{ci['rlf_time_percent']}**.",
        f"- Strict gate: **{gates['strict_all']}**; PPO **CLOSED**.",
        "",
        report["inference_warning"],
    ]
    (output_dir / "report_vi.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main(root_path: str, args: Namespace) -> int:
    root = Path(root_path).resolve()
    resolve = lambda value: (
        Path(value) if Path(value).is_absolute() else root / value
    ).resolve()
    report = run_audit(
        root=root,
        data_dir=resolve(args.data_dir),
        route_split_path=resolve(args.route_split),
        output_dir=resolve(args.output_dir),
        bootstrap_resamples=int(args.bootstrap_resamples),
        bootstrap_seed=int(args.bootstrap_seed),
    )
    print(json.dumps({
        "status": report["status"],
        "routes": report["routes"],
        "actions": report["actions"],
        "gates": report["gates"],
    }, indent=2), flush=True)
    return 0
