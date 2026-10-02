"""Train-only counterfactual audit for the P1 candidate shortlist."""

from __future__ import annotations

from argparse import Namespace
from collections import defaultdict
import copy
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ho_optim_drl.agents.ppo import SMDPPPO
from ho_optim_drl.data import (
    TracePair,
    discover_trace_pairs,
    filter_trace_pairs_by_speed,
    load_cell_topology,
    load_radio_trace,
    stratified_trajectory_split,
)
from ho_optim_drl.data.project import load_protocol_config
from ho_optim_drl.protocol import HandoverEngine, ProtocolPhase
from ho_optim_drl.smdp import TopologyCandidateObservationBuilder


_SHORTLIST_SIZES = (3, 5, 10)


@dataclass(frozen=True)
class CounterfactualOutcome:
    """Cumulative physical outcome from one decision to one horizon."""

    delivered_mbit: float
    handover_failures: int
    rlf_count: int
    ping_pong_count: int
    recovery_seconds: float
    elapsed_seconds: float
    trace_terminated: bool


def _sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _select_train_routes(
    train_pairs: Iterable[TracePair],
    *,
    routes_per_speed: int,
    seed: int,
) -> tuple[TracePair, ...]:
    if routes_per_speed <= 0:
        raise ValueError("routes_per_speed must be positive.")
    rng = np.random.default_rng(seed)
    by_speed: dict[int, list[TracePair]] = defaultdict(list)
    for pair in train_pairs:
        by_speed[pair.speed_kmh].append(pair)
    selected: list[TracePair] = []
    for speed in sorted(by_speed):
        group = sorted(by_speed[speed])
        order = rng.permutation(len(group))
        selected.extend(group[index] for index in order[:routes_per_speed])
    return tuple(sorted(selected))


def _snapshot_outcome(
    totals: dict[str, float | int],
    *,
    elapsed_ticks: int,
    delta_t_seconds: float,
    terminated: bool,
) -> CounterfactualOutcome:
    return CounterfactualOutcome(
        delivered_mbit=float(totals["delivered_mbit"]),
        handover_failures=int(totals["handover_failures"]),
        rlf_count=int(totals["rlf_count"]),
        ping_pong_count=int(totals["ping_pong_count"]),
        recovery_seconds=float(totals["recovery_ticks"]) * delta_t_seconds,
        elapsed_seconds=elapsed_ticks * delta_t_seconds,
        trace_terminated=terminated,
    )


def simulate_counterfactual(
    engine: HandoverEngine,
    *,
    initial_target: int | None,
    horizon_ticks: tuple[int, ...],
) -> dict[int, CounterfactualOutcome]:
    """Branch a pending decision and continue with Always-HO-C1.

    ``initial_target=None`` means STAY. At all later decisions the branch uses
    the A3 target. The trace arrays are immutable, so they are deliberately
    shared across the deep copy instead of copied once per action.
    """
    if not horizon_ticks or min(horizon_ticks) <= 0:
        raise ValueError("Every audit horizon must contain at least one tick.")
    if engine.decision_point is None or engine.trace is None:
        raise RuntimeError("Counterfactual audit requires a pending decision.")
    trace = engine.trace
    branch = copy.deepcopy(engine, {id(trace): trace})
    if initial_target is None:
        branch.stay()
    else:
        branch.start_handover(initial_target)

    delta_t_seconds = branch.config.delta_t_ms / 1000.0
    totals: dict[str, float | int] = {
        "delivered_mbit": 0.0,
        "handover_failures": 0,
        "rlf_count": 0,
        "ping_pong_count": 0,
        "recovery_ticks": 0,
    }
    results: dict[int, CounterfactualOutcome] = {}
    max_ticks = max(horizon_ticks)
    elapsed_ticks = 0
    terminated = False
    while elapsed_ticks < max_ticks and not terminated:
        if branch.decision_point is not None:
            branch.start_handover()
        tick = branch.advance_one_tick()
        elapsed_ticks += 1
        totals["delivered_mbit"] += (
            tick.delivered_rate_mbps * delta_t_seconds
        )
        totals["handover_failures"] += int(tick.handover_failed)
        totals["rlf_count"] += int(tick.rlf_started)
        totals["ping_pong_count"] += int(tick.ping_pong_detected)
        totals["recovery_ticks"] += int(
            tick.snapshot.phase is ProtocolPhase.RLF_RECOVERY
        )
        terminated = tick.terminated
        if elapsed_ticks in horizon_ticks:
            results[elapsed_ticks] = _snapshot_outcome(
                totals,
                elapsed_ticks=elapsed_ticks,
                delta_t_seconds=delta_t_seconds,
                terminated=terminated,
            )

    # A short trace has no additional delivered data or protocol events.
    for horizon in horizon_ticks:
        if horizon not in results:
            results[horizon] = _snapshot_outcome(
                totals,
                elapsed_ticks=elapsed_ticks,
                delta_t_seconds=delta_t_seconds,
                terminated=True,
            )
    return results


def _no_worse_reliability(
    candidate: CounterfactualOutcome,
    reference: CounterfactualOutcome,
) -> bool:
    return (
        candidate.handover_failures <= reference.handover_failures
        and candidate.rlf_count <= reference.rlf_count
        and candidate.ping_pong_count <= reference.ping_pong_count
        and candidate.recovery_seconds <= reference.recovery_seconds + 1e-12
    )


def _is_useful(
    candidate: CounterfactualOutcome,
    reference: CounterfactualOutcome,
) -> bool:
    return (
        candidate.delivered_mbit > reference.delivered_mbit + 1e-9
        and _no_worse_reliability(candidate, reference)
    )


def _best_stay_or_c1(
    stay: CounterfactualOutcome,
    c1: CounterfactualOutcome,
) -> CounterfactualOutcome:
    """Deterministic comparison reference: data first, reliability second."""
    key = lambda item: (
        item.delivered_mbit,
        -item.handover_failures,
        -item.rlf_count,
        -item.ping_pong_count,
        -item.recovery_seconds,
    )
    return max((stay, c1), key=key)


def _policy_candidate_configuration(
    agent: SMDPPPO,
    checkpoint_extra: dict[str, Any],
) -> tuple[int, str]:
    objective = checkpoint_extra.get("objective", {})
    count = int(objective.get("candidate_count", agent.model.action_size - 1))
    order = str(objective.get("candidate_order_key", "legacy"))
    if count not in _SHORTLIST_SIZES:
        raise ValueError(f"Unsupported policy candidate count: {count}.")
    return count, order


def _apply_behavior_action(
    engine: HandoverEngine,
    builder: TopologyCandidateObservationBuilder,
    *,
    agent: SMDPPPO | None,
    observation: np.ndarray | None,
) -> int:
    if agent is None:
        engine.start_handover()
        return 1
    if observation is None:
        raise RuntimeError("Policy behavior requires an observation.")
    action = int(
        agent.predict(
            observation,
            deterministic=True,
            action_mask=builder.action_mask(),
        )
    )
    if action == 0:
        engine.stay()
    else:
        target = builder.candidate_cells[action - 1]
        if target is None:
            raise RuntimeError("Policy selected an invalid candidate slot.")
        engine.start_handover(target)
    return action


def _audit_state(
    engine: HandoverEngine,
    audit_builder: TopologyCandidateObservationBuilder,
    *,
    horizons_ticks: tuple[int, ...],
    source: str,
    pair: TracePair,
    decision_index: int,
) -> dict[str, Any]:
    snapshot = engine.snapshot
    audit_builder.build(snapshot)
    ranked = tuple(audit_builder.candidate_pool_cells)
    legacy = tuple(audit_builder.legacy_candidate_pool_cells)
    if ranked[0] != legacy[0] or {
        cell for cell in ranked if cell is not None
    } != {cell for cell in legacy if cell is not None}:
        raise RuntimeError("ranked_v1 changed the legacy physical pool.")

    action_targets: list[int | None] = [None, *ranked]
    outcomes = {
        action_index: simulate_counterfactual(
            engine,
            initial_target=target,
            horizon_ticks=horizons_ticks,
        )
        for action_index, target in enumerate(action_targets)
        if action_index == 0 or target is not None
    }
    delta_t_seconds = engine.config.delta_t_ms / 1000.0
    horizon_rows = []
    for horizon_ticks_value in horizons_ticks:
        stay = outcomes[0][horizon_ticks_value]
        c1 = outcomes[1][horizon_ticks_value]
        best_basic = _best_stay_or_c1(stay, c1)
        alternative_slots = range(2, len(ranked) + 1)
        useful_vs_c1 = [
            slot
            for slot in alternative_slots
            if slot in outcomes
            and _is_useful(outcomes[slot][horizon_ticks_value], c1)
        ]
        useful_vs_best = [
            slot
            for slot in alternative_slots
            if slot in outcomes
            and _is_useful(
                outcomes[slot][horizon_ticks_value], best_basic
            )
        ]
        horizon_rows.append(
            {
                "horizon_seconds": horizon_ticks_value * delta_t_seconds,
                "useful_slots_vs_c1": useful_vs_c1,
                "useful_slots_vs_best_stay_c1": useful_vs_best,
                "coverage_vs_c1": {
                    str(k): any(slot <= k for slot in useful_vs_c1)
                    for k in _SHORTLIST_SIZES
                },
                "coverage_vs_best_stay_c1": {
                    str(k): any(slot <= k for slot in useful_vs_best)
                    for k in _SHORTLIST_SIZES
                },
                "outcomes": {
                    ("STAY" if action == 0 else f"C{action}"): asdict(
                        action_outcomes[horizon_ticks_value]
                    )
                    for action, action_outcomes in outcomes.items()
                },
            }
        )
    decision = snapshot.decision_point
    if decision is None:
        raise RuntimeError("Candidate audit lost the pending decision.")
    return {
        "source": source,
        "trace_id": f"{pair.speed_kmh}kmh_{pair.trajectory_id}",
        "speed_kmh": pair.speed_kmh,
        "trajectory_id": pair.trajectory_id,
        "decision_index": decision_index,
        "tic": snapshot.tic,
        "serving_cell": decision.serving_cell,
        "a3_target_cell": decision.target_cell,
        "ranked_candidates": list(ranked),
        "legacy_candidates": list(legacy),
        "legacy_slot_by_cell": {
            str(cell): index + 1
            for index, cell in enumerate(legacy)
            if cell is not None
        },
        "horizons": horizon_rows,
    }


def _collect_source_states(
    *,
    source: str,
    pairs: tuple[TracePair, ...],
    data_dir: str,
    protocol_config,
    topology,
    horizons_ticks: tuple[int, ...],
    max_decisions_per_route: int,
    decision_stride: int,
    agent: SMDPPPO | None,
    policy_candidate_count: int,
    policy_candidate_order: str,
) -> list[dict[str, Any]]:
    states: list[dict[str, Any]] = []
    for route_number, pair in enumerate(pairs, start=1):
        trace = load_radio_trace(
            protocol_config,
            data_dir,
            pair.rsrp_file,
            pair.sinr_file,
        )
        engine = HandoverEngine(protocol_config)
        engine.reset(trace)
        audit_builder = TopologyCandidateObservationBuilder(
            protocol_config,
            topology=topology,
            candidate_count=10,
            candidate_order="ranked_v1",
        )
        policy_builder = TopologyCandidateObservationBuilder(
            protocol_config,
            topology=topology,
            candidate_count=policy_candidate_count,
            candidate_order=policy_candidate_order,
        )
        audit_builder.reset()
        policy_builder.reset()
        decision_index = 0
        sampled = 0
        while True:
            tick = engine.advance_one_tick()
            audit_builder.observe(tick)
            policy_builder.observe(tick)
            if tick.terminated:
                break
            if engine.decision_point is None:
                continue
            policy_observation = (
                policy_builder.build(engine.snapshot)
                if agent is not None
                else None
            )
            should_sample = (
                decision_index % decision_stride == 0
                and sampled < max_decisions_per_route
            )
            if should_sample:
                states.append(
                    _audit_state(
                        engine,
                        audit_builder,
                        horizons_ticks=horizons_ticks,
                        source=source,
                        pair=pair,
                        decision_index=decision_index,
                    )
                )
                sampled += 1
            _apply_behavior_action(
                engine,
                policy_builder,
                agent=agent,
                observation=policy_observation,
            )
            decision_index += 1
            if sampled >= max_decisions_per_route:
                break
        print(
            f"[candidate-audit:{source}] {route_number}/{len(pairs)} "
            f"{pair.speed_kmh}kmh_{pair.trajectory_id}: {sampled} states"
        )
    return states


def _flatten_state_rows(states: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for state in states:
        common = {
            key: state[key]
            for key in (
                "source",
                "trace_id",
                "speed_kmh",
                "trajectory_id",
                "decision_index",
                "tic",
                "serving_cell",
                "a3_target_cell",
            )
        }
        for horizon in state["horizons"]:
            rows.append(
                {
                    **common,
                    "horizon_seconds": horizon["horizon_seconds"],
                    "ranked_candidates": json.dumps(
                        state["ranked_candidates"], separators=(",", ":")
                    ),
                    "legacy_candidates": json.dumps(
                        state["legacy_candidates"], separators=(",", ":")
                    ),
                    "useful_slots_vs_c1": json.dumps(
                        horizon["useful_slots_vs_c1"], separators=(",", ":")
                    ),
                    "useful_slots_vs_best_stay_c1": json.dumps(
                        horizon["useful_slots_vs_best_stay_c1"],
                        separators=(",", ":"),
                    ),
                    **{
                        f"coverage_k{k}_vs_c1": horizon["coverage_vs_c1"][str(k)]
                        for k in _SHORTLIST_SIZES
                    },
                    **{
                        f"coverage_k{k}_vs_best_stay_c1": horizon[
                            "coverage_vs_best_stay_c1"
                        ][str(k)]
                        for k in _SHORTLIST_SIZES
                    },
                }
            )
    return rows


def summarize_coverage(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate positive-state counts and conditional prefix coverage."""
    groups: dict[tuple[str, str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        horizon = float(row["horizon_seconds"])
        groups[("overall", "all", horizon)].append(row)
        groups[("source", str(row["source"]), horizon)].append(row)
        groups[("speed_kmh", str(row["speed_kmh"]), horizon)].append(row)
    summary = []
    for (group_by, group, horizon), group_rows in sorted(groups.items()):
        record: dict[str, Any] = {
            "group_by": group_by,
            "group": group,
            "horizon_seconds": horizon,
            "state_count": len(group_rows),
        }
        for comparator, field in (
            ("c1", "useful_slots_vs_c1"),
            ("best_stay_c1", "useful_slots_vs_best_stay_c1"),
        ):
            positives = [row for row in group_rows if json.loads(row[field])]
            record[f"positive_states_vs_{comparator}"] = len(positives)
            for k in _SHORTLIST_SIZES:
                retained = sum(
                    bool(row[f"coverage_k{k}_vs_{comparator}"])
                    for row in positives
                )
                record[f"retained_k{k}_vs_{comparator}"] = retained
                record[f"coverage_k{k}_vs_{comparator}"] = (
                    retained / len(positives) if positives else None
                )
        summary.append(record)
    return summary


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(root_path: str, args: Namespace) -> int:
    """Run the P1 audit without touching validation or test trajectories."""
    data_dir = os.path.abspath(os.path.join(root_path, args.data_dir))
    output_dir = Path(root_path, args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.max_decisions_per_route <= 0 or args.decision_stride <= 0:
        raise ValueError("Decision sampling arguments must be positive.")
    if any(value <= 0 for value in args.horizons):
        raise ValueError("Audit horizons must be positive seconds.")

    checkpoint_path = None
    agent = None
    checkpoint_extra: dict[str, Any] = {}
    if args.checkpoint:
        checkpoint_path = os.path.abspath(
            os.path.join(root_path, args.checkpoint)
        )
        agent, checkpoint_extra = SMDPPPO.from_checkpoint(
            checkpoint_path,
            device="cpu",
        )
    protocol_config = load_protocol_config(data_dir, checkpoint_extra)
    topology = load_cell_topology(data_dir)
    pairs = filter_trace_pairs_by_speed(
        discover_trace_pairs(data_dir), args.speeds
    )
    splits = stratified_trajectory_split(
        pairs,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
        seed=args.split_seed,
    )
    selected = _select_train_routes(
        splits.train,
        routes_per_speed=args.routes_per_speed,
        seed=args.sample_seed,
    )
    if not selected:
        raise ValueError("The selected training audit set is empty.")

    horizon_ticks = tuple(
        sorted(
            {
                int(round(seconds * 1000 / protocol_config.delta_t_ms))
                for seconds in args.horizons
            }
        )
    )
    if any(
        not np.isclose(
            ticks * protocol_config.delta_t_ms / 1000.0,
            seconds,
        )
        for ticks, seconds in zip(horizon_ticks, sorted(set(args.horizons)))
    ):
        raise ValueError("Every horizon must align with delta_t_ms.")

    states = _collect_source_states(
        source="always_ho_c1",
        pairs=selected,
        data_dir=data_dir,
        protocol_config=protocol_config,
        topology=topology,
        horizons_ticks=horizon_ticks,
        max_decisions_per_route=args.max_decisions_per_route,
        decision_stride=args.decision_stride,
        agent=None,
        policy_candidate_count=10,
        policy_candidate_order="legacy",
    )
    policy_configuration = None
    if agent is not None:
        policy_count, policy_order = _policy_candidate_configuration(
            agent, checkpoint_extra
        )
        policy_configuration = {
            "candidate_count": policy_count,
            "candidate_order": policy_order,
            "observation_size": agent.model.observation_size,
            "action_size": agent.model.action_size,
        }
        states.extend(
            _collect_source_states(
                source="existing_policy",
                pairs=selected,
                data_dir=data_dir,
                protocol_config=protocol_config,
                topology=topology,
                horizons_ticks=horizon_ticks,
                max_decisions_per_route=args.max_decisions_per_route,
                decision_stride=args.decision_stride,
                agent=agent,
                policy_candidate_count=policy_count,
                policy_candidate_order=policy_order,
            )
        )

    rows = _flatten_state_rows(states)
    coverage = summarize_coverage(rows)
    selected_files = []
    for pair in selected:
        selected_files.append(
            {
                "speed_kmh": pair.speed_kmh,
                "trajectory_id": pair.trajectory_id,
                "rsrp_file": pair.rsrp_file,
                "sinr_file": pair.sinr_file,
                "rsrp_sha256": _sha256_file(Path(data_dir, pair.rsrp_file)),
                "sinr_sha256": _sha256_file(Path(data_dir, pair.sinr_file)),
            }
        )
    provenance_path = Path(data_dir, "dataset_provenance.json")
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "P1 train-only candidate shortlist counterfactual audit",
        "data_dir": data_dir,
        "split": "train_only",
        "split_seed": args.split_seed,
        "split_fractions": {
            "train": args.train_fraction,
            "validation": args.validation_fraction,
            "test": args.test_fraction,
        },
        "selected_speeds_kmh": sorted({pair.speed_kmh for pair in selected}),
        "routes_per_speed_requested": args.routes_per_speed,
        "selected_route_count": len(selected),
        "selected_routes": selected_files,
        "dataset_provenance": (
            {
                "file": provenance_path.name,
                "sha256": _sha256_file(provenance_path),
            }
            if provenance_path.is_file()
            else None
        ),
        "state_sources": ["always_ho_c1"]
        + (["existing_policy"] if agent is not None else []),
        "checkpoint": checkpoint_path,
        "policy_configuration": policy_configuration,
        "sampling": {
            "sample_seed": args.sample_seed,
            "max_decisions_per_route": args.max_decisions_per_route,
            "decision_stride": args.decision_stride,
            "state_count": len(states),
        },
        "counterfactual": {
            "horizons_seconds": [
                ticks * protocol_config.delta_t_ms / 1000.0
                for ticks in horizon_ticks
            ],
            "continuation_policy": "always_ho_c1_after_initial_action",
            "useful_definition": (
                "strictly_more_delivered_mbit_and_no_more_HOF_RLF_PP_or_"
                "recovery_time"
            ),
        },
        "selector": {
            "candidate_pool": (
                "same_legacy_10_cell_topology_pool_no_Qin_quality_mask"
            ),
            "order": (
                "C1_A3_target_then_best_current_SINR_then_distinct_best_"
                "causal_projected_SINR_90ms_then_legacy_order"
            ),
            "causality": (
                "future_samples_are_used_only_as_counterfactual_labels"
            ),
            "prefixes": list(_SHORTLIST_SIZES),
        },
        "coverage": coverage,
        "artifacts": {
            "states_json": "candidate_audit_states.json",
            "states_csv": "candidate_audit_states.csv",
            "coverage_csv": "candidate_audit_coverage.csv",
        },
    }
    (output_dir / "candidate_audit_states.json").write_text(
        json.dumps(states, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_csv(output_dir / "candidate_audit_states.csv", rows)
    _write_csv(output_dir / "candidate_audit_coverage.csv", coverage)
    (output_dir / "candidate_audit_summary.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output_dir": str(output_dir),
        "selected_route_count": len(selected),
        "state_count": len(states),
        "coverage": [
            row for row in coverage if row["group_by"] == "overall"
        ],
    }, indent=2, allow_nan=False))
    return 0
