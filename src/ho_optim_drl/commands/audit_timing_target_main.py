"""D63 main TRAIN-only analysis and finite-lookahead route reference."""

from __future__ import annotations

from argparse import Namespace
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any, Iterable, Sequence

import numpy as np

from ho_optim_drl.commands.audit_policy_learning import _make_builders, _sha256_file
from ho_optim_drl.commands.audit_timing_target import (
    CANDIDATE_COUNT,
    CANDIDATE_ORDER,
    LOCAL_METRICS,
    MAIN_SCHEMA,
    OBSERVATION_SCHEMA,
    REWARD_SCHEMA,
    STAY_MODE,
    STAY_REDECISION_MS,
    _apply_source_action,
    _branch_key,
    _execute_requested_slot,
    _json_write,
    _no_worse,
    _prepare_destination,
    _source_specs,
    simulate_timing_target,
)
from ho_optim_drl.data import TracePair, load_cell_topology, load_radio_trace
from ho_optim_drl.data.project import load_protocol_config
from ho_optim_drl.protocol import HandoverEngine, ProtocolPhase
from ho_optim_drl.smdp import RewardBreakdown, build_reward_model, trajectory_reward_config


POLICIES = (
    "always_ho_c1",
    "d61",
    "timing_only_reference",
    "joint_reference",
)
REFERENCE_CONTROL = "always_ho_c1"
RELIABILITY_METRICS = (
    "handover_failure_percent",
    "rlf_time_percent",
    "ping_pong_percent",
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected object at {path}:{line_number}.")
            rows.append(value)
    return rows


def _validate_local_artifact(local_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    required = (
        "registry.json",
        "state_bank_manifest.json",
        "state_bank.npz",
        "timing_target_outcomes.jsonl",
        "counterfactual_summary.json",
        "runtime.json",
        "pilot_report.json",
    )
    missing = [name for name in required if not (local_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete D63 local artifact; missing {missing}.")
    manifest = _load_json(local_dir / "state_bank_manifest.json")
    registry = _load_json(local_dir / "registry.json")
    if manifest.get("split") != "train" or int(
        manifest.get("validation_or_test_rows_read", -1)
    ) != 0:
        raise ValueError("D63 main analysis accepts a TRAIN-only state bank.")
    contract = registry.get("contract", {})
    expected = {
        "candidate_count": CANDIDATE_COUNT,
        "candidate_order": CANDIDATE_ORDER,
        "observation_schema": OBSERVATION_SCHEMA,
        "reward_schema": REWARD_SCHEMA,
        "stay_mode": STAY_MODE,
        "stay_redecision_ms": STAY_REDECISION_MS,
        "trigger": "completed_a3_ttt",
    }
    mismatch = {
        key: {"expected": value, "actual": contract.get(key)}
        for key, value in expected.items()
        if contract.get(key) != value
    }
    if mismatch:
        raise ValueError(f"D63 local artifact contract mismatch: {mismatch}")
    bank_path = local_dir / str(manifest["npz_file"])
    if _sha256_file(bank_path) != manifest["npz_sha256"]:
        raise ValueError("D63 state bank hash mismatch.")
    return manifest, registry


def _category_rows(
    state_rows: Sequence[dict[str, Any]], category: str
) -> list[dict[str, Any]]:
    if category == "timing_only":
        return [row for row in state_rows if int(row["requested_slot"]) == 1]
    if category == "target_only":
        return [row for row in state_rows if int(row["delay_ms"]) == 0]
    if category == "joint":
        return list(state_rows)
    raise ValueError(f"Unknown D63 category: {category}")


def build_local_paired_records(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Reduce every state to control-vs-best paired records for three scopes."""
    grouped: dict[tuple[str, float, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["source"]), float(row["horizon_seconds"]), str(row["state_id"]))].append(row)
    records: list[dict[str, Any]] = []
    for (source, horizon, state_id), state_rows in sorted(grouped.items()):
        control = next(
            (
                row
                for row in state_rows
                if int(row["delay_ms"]) == 0 and int(row["requested_slot"]) == 1
            ),
            None,
        )
        if control is None:
            raise ValueError(f"State {state_id} has no delay=0/C1 control.")
        for category in ("timing_only", "target_only", "joint"):
            candidates = _category_rows(state_rows, category)
            best = max(
                candidates,
                key=lambda row: (
                    *_branch_key(row),
                    -int(row["delay_ms"]),
                    -int(row["requested_slot"]),
                ),
            )
            deltas = {
                metric: float(best["physical"][metric])
                - float(control["physical"][metric])
                for metric in LOCAL_METRICS
            }
            strict = (
                deltas["delivered_mbit"] > 1e-9
                and _no_worse(best["physical"], control["physical"])
            )
            records.append(
                {
                    "state_id": state_id,
                    "source": source,
                    "speed_kmh": int(control["speed_kmh"]),
                    "trajectory_id": int(control["trajectory_id"]),
                    "horizon_seconds": horizon,
                    "category": category,
                    "best_delay_ms": int(best["delay_ms"]),
                    "best_requested_slot": int(best["requested_slot"]),
                    "best_executed_slot": best["execution"].get("executed_slot"),
                    "strict_dominant": bool(strict),
                    "fallback": best["execution"].get("fallback"),
                    "horizon_censored": bool(best.get("horizon_censored", False)),
                    "delta": deltas,
                }
            )
    return records


def _cluster_bootstrap_means(
    records: Sequence[dict[str, Any]],
    *,
    metric_getters: dict[str, Any],
    resamples: int,
    seed: int,
) -> dict[str, tuple[float, float, float]]:
    if resamples <= 0:
        raise ValueError("bootstrap_resamples must be positive.")
    by_speed_route: dict[int, dict[int, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for record in records:
        by_speed_route[int(record["speed_kmh"])][int(record["trajectory_id"])].append(record)
    if not by_speed_route:
        raise ValueError("Cannot bootstrap an empty route collection.")
    names = tuple(metric_getters)
    route_stats: dict[int, list[tuple[np.ndarray, int]]] = defaultdict(list)
    for speed in sorted(by_speed_route):
        for route in sorted(by_speed_route[speed]):
            cluster = by_speed_route[speed][route]
            values = np.asarray(
                [[metric_getters[name](row) for name in names] for row in cluster],
                dtype=np.float64,
            )
            route_stats[speed].append((np.sum(values, axis=0), len(cluster)))
    rng = np.random.default_rng(seed)
    estimates = np.empty((resamples, len(names)), dtype=np.float64)
    for index in range(resamples):
        total = np.zeros(len(names), dtype=np.float64)
        count = 0
        for speed in sorted(route_stats):
            clusters = route_stats[speed]
            picks = rng.integers(0, len(clusters), size=len(clusters))
            for pick in picks:
                cluster_sum, cluster_count = clusters[int(pick)]
                total += cluster_sum
                count += cluster_count
        estimates[index] = total / count
    return {
        name: (
            float(np.mean([metric_getters[name](row) for row in records])),
            float(np.percentile(estimates[:, column], 2.5)),
            float(np.percentile(estimates[:, column], 97.5)),
        )
        for column, name in enumerate(names)
    }


def summarize_local_main(
    rows: Sequence[dict[str, Any]], *, bootstrap_resamples: int, bootstrap_seed: int
) -> dict[str, Any]:
    paired = build_local_paired_records(rows)
    grouped: dict[tuple[str, float, str], list[dict[str, Any]]] = defaultdict(list)
    for record in paired:
        grouped[(record["source"], record["horizon_seconds"], record["category"])].append(record)
    summaries: list[dict[str, Any]] = []
    rng_offset = 0
    for (source, horizon, category), selected in sorted(grouped.items()):
        scopes: list[tuple[str, list[dict[str, Any]]]] = [("all", selected)]
        for speed in sorted({int(row["speed_kmh"]) for row in selected}):
            scopes.append((str(speed), [row for row in selected if int(row["speed_kmh"]) == speed]))
        for speed_scope, scoped in scopes:
            bootstrap = _cluster_bootstrap_means(
                scoped,
                metric_getters={
                    **{
                        metric: (
                            lambda row, name=metric: float(row["delta"][name])
                        )
                        for metric in LOCAL_METRICS
                    },
                    "strict_dominant": lambda row: float(row["strict_dominant"]),
                },
                resamples=bootstrap_resamples,
                seed=bootstrap_seed + rng_offset,
            )
            rng_offset += 1
            intervals = {
                metric: {
                    "mean_delta": bootstrap[metric][0],
                    "ci95": [bootstrap[metric][1], bootstrap[metric][2]],
                }
                for metric in LOCAL_METRICS
            }
            strict_mean, strict_low, strict_high = bootstrap["strict_dominant"]
            summaries.append(
                {
                    "source": source,
                    "horizon_seconds": horizon,
                    "category": category,
                    "speed_kmh": speed_scope,
                    "anchors": len(scoped),
                    "routes": len({(row["speed_kmh"], row["trajectory_id"]) for row in scoped}),
                    "strict_dominant_count": sum(bool(row["strict_dominant"]) for row in scoped),
                    "strict_dominant_fraction": strict_mean,
                    "strict_dominant_fraction_ci95": [strict_low, strict_high],
                    "fallback_count": sum(row["fallback"] is not None for row in scoped),
                    "censored_count": sum(bool(row["horizon_censored"]) for row in scoped),
                    "best_delay_counts": dict(Counter(str(row["best_delay_ms"]) for row in scoped)),
                    "best_slot_counts": dict(Counter(str(row["best_requested_slot"]) for row in scoped)),
                    "paired_deltas": intervals,
                }
            )
    return {
        "schema_name": "d63_local_main_analysis",
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "bootstrap": {
            "unit": "route_cluster_stratified_by_speed",
            "resamples": bootstrap_resamples,
            "seed": bootstrap_seed,
            "ci": "percentile_95",
        },
        "raw_outcome_rows": len(rows),
        "paired_records": len(paired),
        "summaries": summaries,
    }


def _route_pairs(manifest: dict[str, Any]) -> list[TracePair]:
    return [
        TracePair(
            int(item["speed_kmh"]),
            int(item["trajectory_id"]),
            str(item["rsrp_file"]),
            str(item["sinr_file"]),
        )
        for item in manifest["selected_routes"]
    ]


def _lookahead_choice(
    engine: HandoverEngine,
    builder: Any,
    *,
    policy: str,
    delay_ticks: tuple[int, ...],
    horizon_ticks: int,
) -> tuple[int, int, int]:
    slots = (1,) if policy == "timing_only_reference" else tuple(
        range(1, CANDIDATE_COUNT + 1)
    )
    choices: list[tuple[dict[str, Any], int, int]] = []
    simulated_ticks = 0
    for delay in delay_ticks:
        for slot in slots:
            outcome, ticks = simulate_timing_target(
                engine,
                builder,
                delay_ticks=delay,
                requested_slot=slot,
                horizon_ticks=(horizon_ticks,),
            )
            simulated_ticks += ticks
            choices.append((outcome[horizon_ticks], delay, slot))
    best, delay, slot = max(
        choices,
        key=lambda item: (*_branch_key(item[0]), -item[1], -item[2]),
    )
    del best
    return delay, slot, simulated_ticks


def _run_route_policy(
    *,
    pair: TracePair,
    data_dir: str,
    policy: str,
    source_specs: dict[str, dict[str, Any]],
    delay_ticks: tuple[int, ...],
    horizon_ticks: int,
) -> dict[str, Any]:
    protocol_config = load_protocol_config(data_dir)
    topology = load_cell_topology(data_dir)
    trace = load_radio_trace(
        protocol_config,
        data_dir,
        pair.rsrp_file,
        pair.sinr_file,
        speed_kmh=pair.speed_kmh,
    )
    engine = HandoverEngine(protocol_config)
    engine.reset(trace)
    builder = _make_builders(
        protocol_config,
        topology,
        candidate_count=CANDIDATE_COUNT,
        candidate_order=CANDIDATE_ORDER,
        schemas=(OBSERVATION_SCHEMA,),
    )[OBSERVATION_SCHEMA]
    builder.reset()
    builder.start_episode(trace)
    reward_model = build_reward_model(
        protocol_config, REWARD_SCHEMA, trajectory_reward_config(REWARD_SCHEMA)
    )
    reward_model.reset(trace)
    reward_model.begin_rollout()
    reward = RewardBreakdown()
    decisions = stays = handovers = pure_rlf = 0
    lookahead_ticks = lookahead_branches = 0
    plan_counts: Counter[str] = Counter()
    while True:
        tick = engine.advance_one_tick()
        builder.observe(tick)
        reward = reward + reward_model.compute(tick)
        pure_rlf += int(tick.rlf_started and not tick.handover_failed)
        if tick.terminated:
            break
        if engine.decision_point is None:
            continue
        decisions += 1
        observation = builder.build(engine.snapshot)
        if policy == "always_ho_c1":
            target = builder.candidate_cells[0]
            if target is None:
                engine.stay(redecision_delay_ticks=STAY_REDECISION_MS // protocol_config.delta_t_ms)
                stays += 1
            else:
                engine.start_handover(int(target))
                handovers += 1
        elif policy == "d61":
            action, _ = _apply_source_action(engine, builder, observation, source_specs["d61"])
            stays += int(action == 0)
            handovers += int(action != 0)
        else:
            delay, slot, simulated = _lookahead_choice(
                engine,
                builder,
                policy=policy,
                delay_ticks=delay_ticks,
                horizon_ticks=horizon_ticks,
            )
            branch_count = len(delay_ticks) * (1 if policy == "timing_only_reference" else CANDIDATE_COUNT)
            lookahead_ticks += simulated
            lookahead_branches += branch_count
            plan_counts[f"delay_{delay * protocol_config.delta_t_ms}_c{slot}"] += 1
            if delay > 0:
                engine.stay(redecision_delay_ticks=STAY_REDECISION_MS // protocol_config.delta_t_ms)
                stays += 1
            else:
                execution = _execute_requested_slot(
                    engine,
                    builder,
                    requested_slot=slot,
                    requested_delay_ticks=0,
                    anchor_tic=int(engine.snapshot.tic),
                )
                stays += int(execution["executed_action"] == "STAY")
                handovers += int(execution["executed_action"] != "STAY")
    stats = engine.get_statistics(trace)
    max_rate = float(stats["max_estimated_rate_mbps"])
    return {
        "policy": policy,
        "trace_id": f"{pair.speed_kmh}:{pair.trajectory_id}",
        "speed_kmh": pair.speed_kmh,
        "trajectory_id": pair.trajectory_id,
        "rsrp_file": pair.rsrp_file,
        "sinr_file": pair.sinr_file,
        "estimated_rate_mbps": float(stats["estimated_rate_mbps"]),
        "max_estimated_rate_mbps": max_rate,
        "relative_rate_percent": 100.0 * float(stats["estimated_rate_mbps"]) / max_rate if max_rate > 0 else 0.0,
        "total_time_s": float(stats["total_time_s"]),
        "connected_time_s": int(stats["connected_time"]) * protocol_config.delta_t_ms / 1000.0,
        "handover_interruption_seconds": float(np.count_nonzero(engine.ho_idxs)) * protocol_config.delta_t_ms / 1000.0,
        "recovery_seconds": float(stats["rlf_recovery_time_s"]),
        "handover_attempts": int(stats["num_ho_exe_started"]),
        "handover_successes": int(stats["num_ho_exe_completed"]),
        "handover_failures": int(stats["num_hof"]),
        "rlf_starts": int(stats["num_rlf"]),
        "pure_rlf_starts": pure_rlf,
        "ping_pong_count": int(stats["num_pp"]),
        "decision_count": decisions,
        "stay_actions": stays,
        "handover_actions": handovers,
        "lookahead_branch_simulations": lookahead_branches,
        "lookahead_simulated_ticks": lookahead_ticks,
        "selected_plan_counts": dict(plan_counts),
        "reward_raw": reward.as_dict(),
    }


def run_full_route_reference(
    *,
    manifest: dict[str, Any],
    registry: dict[str, Any],
    output_dir: Path,
    delays_ms: Sequence[int],
    horizon_seconds: float,
    progress: Any = print,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    data_dir = str(manifest["data_dir"])
    protocol_config = load_protocol_config(data_dir)
    if any(delay < 0 or delay % protocol_config.delta_t_ms for delay in delays_ms):
        raise ValueError("Reference delays must align with delta_t_ms.")
    horizon_ticks = int(round(horizon_seconds * 1000 / protocol_config.delta_t_ms))
    if horizon_ticks <= 0 or not np.isclose(
        horizon_ticks * protocol_config.delta_t_ms / 1000.0, horizon_seconds
    ):
        raise ValueError("Reference horizon must be positive and align with delta_t_ms.")
    checkpoints = {item["label"]: Path(item["checkpoint"]) for item in registry["sources"]}
    for item in registry["sources"]:
        checkpoint = Path(item["checkpoint"])
        if _sha256_file(checkpoint) != item["checkpoint_sha256"]:
            raise ValueError(f"Frozen checkpoint changed: {checkpoint}")
    source_specs = _source_specs(checkpoints["d61"], checkpoints["d62"])
    pairs = _route_pairs(manifest)
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    path = output_dir / "route_reference_outcomes.jsonl"
    total = len(pairs) * len(POLICIES)
    completed = 0
    with path.open("w", encoding="utf-8") as target:
        for pair in pairs:
            for policy in POLICIES:
                row = _run_route_policy(
                    pair=pair,
                    data_dir=data_dir,
                    policy=policy,
                    source_specs=source_specs,
                    delay_ticks=tuple(sorted({delay // protocol_config.delta_t_ms for delay in delays_ms})),
                    horizon_ticks=horizon_ticks,
                )
                rows.append(row)
                target.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                target.flush()
                completed += 1
                if progress is not None:
                    progress(
                        f"[D63 full-route] {completed}/{total} "
                        f"{policy} speed={pair.speed_kmh} route={pair.trajectory_id}"
                    )
    runtime = {
        "elapsed_seconds": time.perf_counter() - started,
        "routes": len(pairs),
        "policy_route_runs": len(rows),
        "lookahead_branch_simulations": sum(int(row["lookahead_branch_simulations"]) for row in rows),
        "lookahead_simulated_ticks": sum(int(row["lookahead_simulated_ticks"]) for row in rows),
    }
    return rows, runtime


def _aggregate_route_rows(rows: Sequence[dict[str, Any]]) -> dict[str, float | int]:
    if not rows:
        raise ValueError("Cannot aggregate empty route rows.")
    attempts = sum(int(row["handover_attempts"]) for row in rows)
    successes = sum(int(row["handover_successes"]) for row in rows)
    failures = sum(int(row["handover_failures"]) for row in rows)
    duration = sum(float(row["total_time_s"]) for row in rows)
    recovery = sum(float(row["recovery_seconds"]) for row in rows)
    ping_pong = sum(int(row["ping_pong_count"]) for row in rows)
    return {
        "routes": len(rows),
        "mean_rate_mbps": float(np.mean([row["estimated_rate_mbps"] for row in rows])),
        "mean_relative_rate_percent": float(np.mean([row["relative_rate_percent"] for row in rows])),
        "p10_relative_rate_percent": float(np.percentile([row["relative_rate_percent"] for row in rows], 10)),
        "handover_attempts": attempts,
        "handover_successes": successes,
        "handover_failures": failures,
        "handover_failure_percent": 100.0 * failures / attempts if attempts else 0.0,
        "rlf_starts": sum(int(row["rlf_starts"]) for row in rows),
        "pure_rlf_starts": sum(int(row["pure_rlf_starts"]) for row in rows),
        "recovery_seconds": recovery,
        "rlf_time_percent": 100.0 * recovery / duration if duration else 0.0,
        "ping_pong_count": ping_pong,
        "ping_pong_percent": 100.0 * ping_pong / successes if successes else 0.0,
        "handover_interruption_seconds": sum(float(row["handover_interruption_seconds"]) for row in rows),
        "decision_count": sum(int(row["decision_count"]) for row in rows),
        "stay_actions": sum(int(row["stay_actions"]) for row in rows),
        "handover_actions": sum(int(row["handover_actions"]) for row in rows),
    }


def _paired_bootstrap_route_metrics(
    treatment: Sequence[dict[str, Any]],
    control: Sequence[dict[str, Any]],
    *,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    treatment_by_id = {(row["speed_kmh"], row["trajectory_id"]): row for row in treatment}
    control_by_id = {(row["speed_kmh"], row["trajectory_id"]): row for row in control}
    if set(treatment_by_id) != set(control_by_id):
        raise ValueError("Full-route policies do not cover identical TRAIN routes.")
    by_speed: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for key in sorted(treatment_by_id):
        by_speed[int(key[0])].append(key)
    fields = ("mean_rate_mbps", "mean_relative_rate_percent", *RELIABILITY_METRICS)
    samples = {field: np.empty(resamples, dtype=np.float64) for field in fields}
    rng = np.random.default_rng(seed)
    for index in range(resamples):
        picked: list[tuple[int, int]] = []
        for speed in sorted(by_speed):
            keys = by_speed[speed]
            for item in rng.integers(0, len(keys), size=len(keys)):
                picked.append(keys[int(item)])
        t_agg = _aggregate_route_rows([treatment_by_id[key] for key in picked])
        c_agg = _aggregate_route_rows([control_by_id[key] for key in picked])
        for field in fields:
            samples[field][index] = float(t_agg[field]) - float(c_agg[field])
    point_t = _aggregate_route_rows(list(treatment_by_id.values()))
    point_c = _aggregate_route_rows(list(control_by_id.values()))
    return {
        field: {
            "delta": float(point_t[field]) - float(point_c[field]),
            "ci95": [float(np.percentile(samples[field], 2.5)), float(np.percentile(samples[field], 97.5))],
        }
        for field in fields
    }


def summarize_full_route(
    rows: Sequence[dict[str, Any]], *, bootstrap_resamples: int, bootstrap_seed: int
) -> dict[str, Any]:
    by_policy: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_policy[str(row["policy"])].append(row)
    missing = [policy for policy in POLICIES if policy not in by_policy]
    if missing:
        raise ValueError(f"Missing full-route policies: {missing}")
    aggregate = {policy: _aggregate_route_rows(by_policy[policy]) for policy in POLICIES}
    by_speed: dict[str, dict[str, Any]] = {}
    speeds = sorted({int(row["speed_kmh"]) for row in rows})
    for policy in POLICIES:
        by_speed[policy] = {
            str(speed): _aggregate_route_rows(
                [row for row in by_policy[policy] if int(row["speed_kmh"]) == speed]
            )
            for speed in speeds
        }
    comparisons: dict[str, Any] = {}
    control = by_policy[REFERENCE_CONTROL]
    for index, policy in enumerate(POLICIES):
        if policy == REFERENCE_CONTROL:
            continue
        comparisons[policy] = {
            "control": REFERENCE_CONTROL,
            "aggregate": _paired_bootstrap_route_metrics(
                by_policy[policy],
                control,
                resamples=bootstrap_resamples,
                seed=bootstrap_seed + index,
            ),
            "qos_retention": {
                "mean_rate_ratio": aggregate[policy]["mean_rate_mbps"] / aggregate[REFERENCE_CONTROL]["mean_rate_mbps"],
                "mean_relative_rate_ratio": aggregate[policy]["mean_relative_rate_percent"] / aggregate[REFERENCE_CONTROL]["mean_relative_rate_percent"],
                "relative_rate_ratio_by_speed": {
                    str(speed): by_speed[policy][str(speed)]["mean_relative_rate_percent"]
                    / by_speed[REFERENCE_CONTROL][str(speed)]["mean_relative_rate_percent"]
                    for speed in speeds
                },
            },
        }
    return {
        "schema_name": "d63_full_route_reference_summary",
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "split": "train",
        "policies": list(POLICIES),
        "aggregate": aggregate,
        "by_speed": by_speed,
        "paired_comparisons": comparisons,
        "bootstrap": {
            "unit": "paired_route_stratified_by_speed",
            "resamples": bootstrap_resamples,
            "seed": bootstrap_seed,
            "ci": "percentile_95",
        },
    }


def decide_gate_g1(full_summary: dict[str, Any]) -> dict[str, Any]:
    """Apply the preregistered practical G1 thresholds to route references."""
    candidates: list[dict[str, Any]] = []
    aggregate = full_summary["aggregate"]
    comparisons = full_summary["paired_comparisons"]
    for policy in ("timing_only_reference", "joint_reference"):
        metrics = aggregate[policy]
        control = aggregate[REFERENCE_CONTROL]
        paired = comparisons[policy]["aggregate"]
        retention = comparisons[policy]["qos_retention"]
        reliability_no_worse = all(
            float(metrics[name]) <= float(control[name]) + 1e-12
            for name in RELIABILITY_METRICS
        )
        rr_delta = float(paired["mean_relative_rate_percent"]["delta"])
        rr_ci = paired["mean_relative_rate_percent"]["ci95"]
        rate_path = rr_delta >= 0.5 and reliability_no_worse
        reductions = {
            name: (
                (float(control[name]) - float(metrics[name])) / float(control[name])
                if float(control[name]) > 0
                else 0.0
            )
            for name in RELIABILITY_METRICS
        }
        retention_path = (
            float(retention["mean_rate_ratio"]) >= 0.98
            and float(retention["mean_relative_rate_ratio"]) >= 0.98
            and min(float(value) for value in retention["relative_rate_ratio_by_speed"].values()) >= 0.98
            and reliability_no_worse
            and max(reductions.values()) >= 0.10
        )
        primary_interval_supports_direction = (
            float(rr_ci[0]) >= 0.0
            if rate_path
            else any(
                reductions[name] >= 0.10
                and float(paired[name]["ci95"][1]) <= 0.0
                for name in RELIABILITY_METRICS
            )
        )
        candidates.append(
            {
                "policy": policy,
                "rr_delta_percentage_points": rr_delta,
                "rr_delta_ci95": rr_ci,
                "reliability_no_worse_point_estimate": reliability_no_worse,
                "relative_reliability_reductions": reductions,
                "qos_retention": retention,
                "rate_path_point_pass": rate_path,
                "retention_reliability_path_point_pass": retention_path,
                "primary_interval_supports_direction": primary_interval_supports_direction,
            }
        )
    if any(
        (row["rate_path_point_pass"] or row["retention_reliability_path_point_pass"])
        and row["primary_interval_supports_direction"]
        for row in candidates
    ):
        status = "PASS"
        reason = "At least one finite-lookahead reference meets a preregistered point threshold and its primary bootstrap interval supports the benefit direction."
    elif any(
        row["rate_path_point_pass"] or row["retention_reliability_path_point_pass"]
        for row in candidates
    ):
        status = "INCONCLUSIVE"
        reason = "A practical point threshold is met, but route-bootstrap uncertainty does not support the primary benefit direction."
    else:
        status = "FAIL"
        reason = "Neither finite-lookahead reference meets the preregistered practical G1 point thresholds on the TRAIN routes."
    return {
        "schema_name": "d63_gate_g1",
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "scope": "TRAIN_only_20_routes_per_speed_finite_lookahead_reference",
        "reason": reason,
        "thresholds": {
            "rate_path": "RR delta >= 0.5 percentage points and reliability point estimates no worse",
            "retention_path": "aggregate/per-speed QoS retention >= 98%, >=10% relative improvement in one reliability metric, others no worse",
            "uncertainty": "primary 95% paired route-bootstrap interval must support the benefit direction for PASS",
        },
        "candidates": candidates,
        "does_not_evaluate_g2": True,
    }


def run_main_analysis(
    *,
    local_dir: Path,
    output_dir: Path,
    bootstrap_resamples: int,
    bootstrap_seed: int,
    delays_ms: tuple[int, ...],
    reference_horizon_seconds: float,
    progress: Any = print,
) -> dict[str, Any]:
    _prepare_destination(output_dir)
    manifest, registry = _validate_local_artifact(local_dir)
    if int(manifest["routes_per_speed_requested"]) != 20:
        raise ValueError("D63 main audit requires exactly 20 TRAIN routes per speed.")
    input_record = {
        "schema_name": "d63_main_input_registry",
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "local_artifact_dir": str(local_dir.resolve()),
        "files": {
            name: {"path": str((local_dir / name).resolve()), "sha256": _sha256_file(local_dir / name)}
            for name in (
                "registry.json",
                "state_bank_manifest.json",
                "state_bank.npz",
                "timing_target_outcomes.jsonl",
                "counterfactual_summary.json",
                "runtime.json",
                "pilot_report.json",
            )
        },
        "split": "train",
        "validation_or_test_rows_read": 0,
        "bootstrap_resamples": bootstrap_resamples,
        "bootstrap_seed": bootstrap_seed,
        "reference_horizon_seconds": reference_horizon_seconds,
        "delays_ms": list(delays_ms),
    }
    _json_write(output_dir / "input_registry.json", input_record)
    if progress is not None:
        progress("[D63 main] loading local counterfactual outcomes...")
    local_rows = _load_jsonl(local_dir / "timing_target_outcomes.jsonl")
    local_summary = summarize_local_main(
        local_rows,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
    )
    _json_write(output_dir / "local_main_analysis.json", local_summary)
    if progress is not None:
        progress("[D63 main] local paired/bootstrap analysis complete; starting full-route reference...")
    route_rows, route_runtime = run_full_route_reference(
        manifest=manifest,
        registry=registry,
        output_dir=output_dir,
        delays_ms=delays_ms,
        horizon_seconds=reference_horizon_seconds,
        progress=progress,
    )
    route_summary = summarize_full_route(
        route_rows,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
    )
    _json_write(output_dir / "route_reference_summary.json", route_summary)
    gate = decide_gate_g1(route_summary)
    _json_write(output_dir / "decision_gate.json", gate)
    report = {
        "schema_name": MAIN_SCHEMA,
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "split": "train",
        "validation_or_test_rows_read": 0,
        "state_bank": {
            "states": int(manifest["state_count"]),
            "routes_per_speed": int(manifest["routes_per_speed_requested"]),
            "max_anchors_per_source_route": int(manifest["max_anchors_per_source_route"]),
        },
        "local_analysis": {
            "raw_rows": local_summary["raw_outcome_rows"],
            "paired_records": local_summary["paired_records"],
        },
        "full_route": {"rows": len(route_rows), "runtime": route_runtime},
        "gate_g1": gate,
        "gate_g2_evaluated": False,
        "training_runs_started": 0,
        "d63_2_implemented": False,
    }
    _json_write(output_dir / "main_report.json", report)
    return report


def main(root_path: str, args: Namespace) -> int:
    root = Path(root_path)
    resolve = lambda value: Path(value) if Path(value).is_absolute() else (root / value).resolve()
    report = run_main_analysis(
        local_dir=resolve(args.local_audit_dir),
        output_dir=resolve(args.output_dir),
        bootstrap_resamples=int(args.bootstrap_resamples),
        bootstrap_seed=int(args.bootstrap_seed),
        delays_ms=tuple(int(value) for value in args.delays_ms),
        reference_horizon_seconds=float(args.reference_horizon),
    )
    runtime = report["full_route"]["runtime"]["elapsed_seconds"]
    print(
        f"D63 main complete: {report['full_route']['rows']} policy-route rows, "
        f"{runtime:.1f}s, G1={report['gate_g1']['status']}."
    )
    return 0
