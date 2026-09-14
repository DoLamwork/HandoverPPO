"""Fair trajectory-level benchmarking for handover policies."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
import csv
from dataclasses import dataclass
import json
import math
import os
from typing import Any

import numpy as np

from ho_optim_drl.envs import HandoverSMDPEnv
from ho_optim_drl.smdp import HandoverAction, RewardBreakdown


@dataclass(frozen=True)
class BenchmarkPolicy:
    """Named deterministic action selector."""

    name: str
    select_action: Callable[[np.ndarray], int | HandoverAction]


@dataclass(frozen=True)
class TrajectoryDescriptor:
    """Metadata aligned with one environment trajectory index."""

    trace_id: str
    speed_kmh: int
    trajectory_id: int


def benchmark_policy(
    env: HandoverSMDPEnv,
    policy: BenchmarkPolicy,
    trajectories: Sequence[TrajectoryDescriptor],
    *,
    progress_callback: Callable[[str], None] | None = None,
) -> list[dict[str, Any]]:
    """Evaluate one policy once on every trajectory in stable order."""
    if len(trajectories) != env.num_trajectories:
        raise ValueError("Trajectory metadata does not match the environment.")

    rows: list[dict[str, Any]] = []
    for index, descriptor in enumerate(trajectories):
        if progress_callback is not None:
            progress_callback(
                f"[{policy.name}] {index + 1:2d}/{len(trajectories)} "
                f"{descriptor.trace_id}"
            )

        actionable = True
        controlled_reward = RewardBreakdown()
        uncontrolled_reward = RewardBreakdown()
        macro_steps = 0
        handover_actions = 0
        stay_actions = 0
        wait_actions = 0
        controlled_duration_seconds = 0.0

        try:
            observation, reset_info = env.reset(
                options={"trace_index": index}
            )
            uncontrolled_reward = _reward_from_dict(
                reset_info["uncontrolled_reward"]
            )
            uncontrolled_duration_seconds = float(
                reset_info["uncontrolled_duration_seconds"]
            )
        except RuntimeError:
            actionable = False
            uncontrolled_reward = env.transition_builder.uncontrolled_reward
            uncontrolled_duration_seconds = (
                env.transition_builder.uncontrolled_duration_ticks
                * env.config.delta_t_ms
                / 1000.0
            )
            observation = None

        if actionable:
            while True:
                action = HandoverAction(
                    int(policy.select_action(observation))
                )
                (
                    observation,
                    _,
                    terminated,
                    truncated,
                    info,
                ) = env.step(int(action))
                controlled_reward = (
                    controlled_reward
                    + _reward_from_dict(info["reward"])
                )
                controlled_duration_seconds += float(info["duration_seconds"])
                macro_steps += 1
                handover_actions += int(action is HandoverAction.HANDOVER)
                stay_actions += int(action is HandoverAction.STAY)
                wait_actions += int(action is HandoverAction.WAIT_50)
                if terminated or truncated:
                    break

        trace = env.engine.trace
        if trace is None:
            raise RuntimeError("Benchmark environment did not retain its trace.")
        statistics = env.engine.get_statistics(trace)
        full_reward = uncontrolled_reward + controlled_reward
        total_actions = handover_actions + stay_actions + wait_actions
        total_time_s = float(statistics["total_time_s"])
        connected_percent = (
            100
            * float(statistics["connected_time"])
            / int(statistics["total_time"])
            if int(statistics["total_time"]) > 0
            else 0.0
        )

        rows.append(
            {
                "policy": policy.name,
                "trace_id": descriptor.trace_id,
                "speed_kmh": descriptor.speed_kmh,
                "trajectory_id": descriptor.trajectory_id,
                "actionable": actionable,
                "num_samples": int(statistics["num_samples"]),
                "total_time_s": total_time_s,
                "connected_percent": connected_percent,
                "uncontrolled_duration_seconds": uncontrolled_duration_seconds,
                "controlled_duration_seconds": controlled_duration_seconds,
                "macro_steps": macro_steps,
                "handover_actions": handover_actions,
                "stay_actions": stay_actions,
                "wait_actions": wait_actions,
                "handover_action_percent": (
                    100 * handover_actions / total_actions
                    if total_actions
                    else 0.0
                ),
                "stay_action_percent": (
                    100 * stay_actions / total_actions
                    if total_actions
                    else 0.0
                ),
                "wait_action_percent": (
                    100 * wait_actions / total_actions
                    if total_actions
                    else 0.0
                ),
                "controlled_reward_rate": controlled_reward.rate,
                "controlled_reward_rlf": controlled_reward.rlf,
                "controlled_reward_ping_pong": controlled_reward.ping_pong,
                "controlled_return": controlled_reward.total,
                "full_trace_reward_rate": full_reward.rate,
                "full_trace_reward_rlf": full_reward.rlf,
                "full_trace_reward_ping_pong": full_reward.ping_pong,
                "full_trace_return": full_reward.total,
                "estimated_rate_mbps": float(
                    statistics["estimated_rate_mbps"]
                ),
                "max_estimated_rate_mbps": float(
                    statistics["max_estimated_rate_mbps"]
                ),
                "relative_rate_percent": (
                    100
                    * float(statistics["estimated_rate_mbps"])
                    / float(statistics["max_estimated_rate_mbps"])
                    if float(statistics["max_estimated_rate_mbps"]) > 0
                    else 0.0
                ),
                "ho_attempts": int(statistics["num_ho_exe_started"]),
                "ho_successes": int(statistics["num_ho_exe_completed"]),
                "handover_failures": int(statistics["num_hof"]),
                "handover_failure_percent": float(
                    statistics["hof_percent"]
                ),
                "rlf_events": int(statistics["num_rlf"]),
                "rlf_recovery_time_s": float(
                    statistics["rlf_recovery_time_s"]
                ),
                "rlf_time_percent": float(
                    statistics["rlf_time_percent"]
                ),
                "ping_pong_events": int(statistics["num_pp"]),
                "ping_pong_percent": float(statistics["pp_percent"]),
            }
        )

    return rows


def aggregate_benchmark_rows(
    rows: Sequence[dict[str, Any]],
    *,
    by_speed: bool,
) -> list[dict[str, Any]]:
    """Aggregate rows by policy, optionally within each speed stratum."""
    if not rows:
        raise ValueError("At least one benchmark row is required.")

    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            (row["policy"], row["speed_kmh"])
            if by_speed
            else (row["policy"],)
        )
        groups[key].append(row)

    aggregated = []
    for key in sorted(groups):
        group = groups[key]
        duration_s = sum(float(row["total_time_s"]) for row in group)
        attempts = sum(int(row["ho_attempts"]) for row in group)
        successes = sum(int(row["ho_successes"]) for row in group)
        failures = sum(int(row["handover_failures"]) for row in group)
        rlf_events = sum(int(row["rlf_events"]) for row in group)
        rlf_recovery_time_s = sum(
            float(row["rlf_recovery_time_s"]) for row in group
        )
        ping_pong_events = sum(
            int(row["ping_pong_events"]) for row in group
        )
        handover_actions = sum(
            int(row["handover_actions"]) for row in group
        )
        stay_actions = sum(int(row["stay_actions"]) for row in group)
        wait_actions = sum(int(row.get("wait_actions", 0)) for row in group)
        total_actions = handover_actions + stay_actions + wait_actions

        result = {
            "policy": key[0],
            "trajectories": len(group),
            "actionable_trajectories": sum(
                bool(row["actionable"]) for row in group
            ),
            "total_time_s": duration_s,
            "mean_controlled_return": _mean(group, "controlled_return"),
            "std_controlled_return": _std(group, "controlled_return"),
            "mean_full_trace_return": _mean(group, "full_trace_return"),
            "mean_reward_rate": _mean(group, "controlled_reward_rate"),
            "mean_rlf_penalty": _mean(group, "controlled_reward_rlf"),
            "mean_ping_pong_penalty": _mean(
                group,
                "controlled_reward_ping_pong",
            ),
            "mean_estimated_rate_mbps": _mean(
                group,
                "estimated_rate_mbps",
            ),
            "duration_weighted_rate_mbps": (
                sum(
                    float(row["estimated_rate_mbps"])
                    * float(row["total_time_s"])
                    for row in group
                )
                / duration_s
                if duration_s > 0
                else 0.0
            ),
            "mean_relative_rate_percent": _mean(
                group,
                "relative_rate_percent",
            ),
            "mean_connected_percent": _mean(
                group,
                "connected_percent",
            ),
            "mean_macro_steps": _mean(group, "macro_steps"),
            "handover_actions": handover_actions,
            "stay_actions": stay_actions,
            "wait_actions": wait_actions,
            "handover_action_percent": (
                100 * handover_actions / total_actions
                if total_actions
                else 0.0
            ),
            "stay_action_percent": (
                100 * stay_actions / total_actions
                if total_actions
                else 0.0
            ),
            "wait_action_percent": (
                100 * wait_actions / total_actions
                if total_actions
                else 0.0
            ),
            "ho_attempts": attempts,
            "ho_successes": successes,
            "handover_failures": failures,
            "handover_failure_percent": (
                100 * failures / attempts if attempts else 0.0
            ),
            "rlf_events": rlf_events,
            "rlf_recovery_time_s": rlf_recovery_time_s,
            "rlf_time_percent": (
                100 * rlf_recovery_time_s / duration_s
                if duration_s
                else 0.0
            ),
            "ping_pong_events": ping_pong_events,
            "ping_pong_percent": (
                100 * ping_pong_events / successes if successes else 0.0
            ),
        }
        if by_speed:
            result["speed_kmh"] = key[1]
        aggregated.append(result)

    return aggregated


def build_paired_comparisons(
    rows: Sequence[dict[str, Any]],
    *,
    reference_policy: str,
) -> list[dict[str, Any]]:
    """Compare each policy with the reference on matched trajectories."""
    by_policy: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_policy[str(row["policy"])][str(row["trace_id"])] = row
    if reference_policy not in by_policy:
        raise ValueError(f"Reference policy {reference_policy!r} is missing.")

    reference_rows = by_policy[reference_policy]
    comparisons = []
    for policy in sorted(by_policy):
        if policy == reference_policy:
            continue
        policy_rows = by_policy[policy]
        common_trace_ids = sorted(set(reference_rows) & set(policy_rows))
        if not common_trace_ids:
            continue

        rate_deltas = np.array(
            [
                float(policy_rows[trace_id]["estimated_rate_mbps"])
                - float(reference_rows[trace_id]["estimated_rate_mbps"])
                for trace_id in common_trace_ids
            ]
        )
        return_deltas = np.array(
            [
                float(policy_rows[trace_id]["controlled_return"])
                - float(reference_rows[trace_id]["controlled_return"])
                for trace_id in common_trace_ids
            ]
        )
        rlf_deltas = np.array(
            [
                int(policy_rows[trace_id]["rlf_events"])
                - int(reference_rows[trace_id]["rlf_events"])
                for trace_id in common_trace_ids
            ]
        )
        pp_deltas = np.array(
            [
                int(policy_rows[trace_id]["ping_pong_events"])
                - int(reference_rows[trace_id]["ping_pong_events"])
                for trace_id in common_trace_ids
            ]
        )
        comparisons.append(
            {
                "policy": policy,
                "reference_policy": reference_policy,
                "paired_trajectories": len(common_trace_ids),
                "mean_rate_delta_mbps": float(np.mean(rate_deltas)),
                "rate_win_fraction": float(np.mean(rate_deltas > 0)),
                "mean_return_delta": float(np.mean(return_deltas)),
                "return_win_fraction": float(np.mean(return_deltas > 0)),
                "total_rlf_delta": int(np.sum(rlf_deltas)),
                "rlf_better_fraction": float(np.mean(rlf_deltas < 0)),
                "total_ping_pong_delta": int(np.sum(pp_deltas)),
                "ping_pong_better_fraction": float(np.mean(pp_deltas < 0)),
            }
        )
    return comparisons


def write_benchmark_outputs(
    output_dir: str,
    rows: Sequence[dict[str, Any]],
    aggregate_rows: Sequence[dict[str, Any]],
    speed_rows: Sequence[dict[str, Any]],
    comparisons: Sequence[dict[str, Any]],
    *,
    metadata: dict[str, Any],
) -> dict[str, str]:
    """Write the benchmark tables and machine-readable summary."""
    os.makedirs(output_dir, exist_ok=True)
    paths = {
        "per_trajectory": os.path.join(output_dir, "per_trajectory.csv"),
        "aggregate": os.path.join(output_dir, "aggregate.csv"),
        "by_speed": os.path.join(output_dir, "by_speed.csv"),
        "paired_comparison": os.path.join(
            output_dir,
            "paired_comparison.csv",
        ),
        "summary": os.path.join(output_dir, "summary.json"),
    }
    _write_csv(paths["per_trajectory"], rows)
    _write_csv(paths["aggregate"], aggregate_rows)
    _write_csv(paths["by_speed"], speed_rows)
    _write_csv(paths["paired_comparison"], comparisons)
    with open(paths["summary"], "w", encoding="utf-8") as summary_file:
        json.dump(
            {
                "metadata": metadata,
                "aggregate": list(aggregate_rows),
                "by_speed": list(speed_rows),
                "paired_comparison": list(comparisons),
            },
            summary_file,
            indent=2,
            allow_nan=False,
        )
    return paths


def _write_csv(path: str, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV table: {path}")
    with open(path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _reward_from_dict(values: dict[str, Any]) -> RewardBreakdown:
    return RewardBreakdown(
        rate=float(values["rate"]),
        rlf=float(values["rlf"]),
        ping_pong=float(values["ping_pong"]),
    )


def _mean(rows: Sequence[dict[str, Any]], field: str) -> float:
    values = np.asarray([float(row[field]) for row in rows], dtype=np.float64)
    result = float(np.mean(values))
    if not math.isfinite(result):
        raise RuntimeError(f"Non-finite aggregate for {field}.")
    return result


def _std(rows: Sequence[dict[str, Any]], field: str) -> float:
    values = np.asarray([float(row[field]) for row in rows], dtype=np.float64)
    result = float(np.std(values))
    if not math.isfinite(result):
        raise RuntimeError(f"Non-finite aggregate for {field}.")
    return result
