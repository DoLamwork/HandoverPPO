"""Benchmark PPO against 3GPP Always-HO on one dataset split."""

from __future__ import annotations

from argparse import Namespace
import os
from collections.abc import Sequence

from ho_optim_drl import HandoverSMDPEnv
from dataclasses import asdict
from ho_optim_drl.data.project import checkpoint_pairs, load_protocol_config
from ho_optim_drl.data import filter_trace_pairs_by_speed
from ho_optim_drl.agents.ppo import SMDPPPO
from ho_optim_drl.data import (
    TracePair,
    discover_trace_pairs,
    stratified_trajectory_split,
)
from ho_optim_drl.evaluation import (
    BenchmarkPolicy,
    TrajectoryDescriptor,
    aggregate_benchmark_rows,
    benchmark_policy,
    build_paired_comparisons,
    write_benchmark_outputs,
)
from ho_optim_drl.smdp import (
    HandoverAction,
    LEGACY_REWARD_SCHEMA,
    reward_config_from_dict,
)


def build_benchmark_policies(agent: SMDPPPO) -> tuple[BenchmarkPolicy, ...]:
    """Return only the PPO policy and its 3GPP Always-HO reference."""
    return (
        BenchmarkPolicy(
            "ppo",
            lambda observation, action_mask: agent.predict(
                observation,
                deterministic=True,
                action_mask=action_mask,
            ),
            uses_action_mask=True,
        ),
        BenchmarkPolicy(
            "3gpp_always_ho",
            lambda observation: HandoverAction.HO_C1,
        ),
    )


def select_benchmark_pairs(
    pairs: Sequence[TracePair],
    *,
    split: str,
    split_seed: int,
    speeds: Sequence[int] | None,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
) -> tuple[TracePair, ...]:
    """Select a stable split and optionally restrict it to UE speeds."""
    if split not in {"validation", "test", "all"}:
        raise ValueError(f"Unsupported benchmark split: {split!r}.")

    requested_speeds = set(speeds) if speeds is not None else None
    if requested_speeds is not None:
        available_speeds = {pair.speed_kmh for pair in pairs}
        missing_speeds = requested_speeds - available_speeds
        if missing_speeds:
            missing = ", ".join(str(speed) for speed in sorted(missing_speeds))
            raise ValueError(f"No trajectories found for speed(s): {missing} km/h.")

    if split == "all":
        selected_pairs = tuple(sorted(pairs))
    else:
        splits = stratified_trajectory_split(
            list(pairs),
            train_fraction=train_fraction,
            validation_fraction=validation_fraction,
            test_fraction=test_fraction,
            seed=split_seed,
        )
        selected_pairs = splits.validation if split == "validation" else splits.test

    if requested_speeds is not None:
        selected_pairs = tuple(
            pair
            for pair in selected_pairs
            if pair.speed_kmh in requested_speeds
        )
    if not selected_pairs:
        raise ValueError("The requested benchmark selection contains no trajectories.")
    return selected_pairs


def select_mixed_benchmark_pairs(
    pairs: Sequence[TracePair],
    *,
    split: str,
    split_seed: int,
    speeds: Sequence[int] | None,
    zero_shot_speeds: Sequence[int] | None,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
) -> tuple[TracePair, ...]:
    """Combine a held-out split at seen speeds with all zero-shot traces."""
    if zero_shot_speeds is None:
        return select_benchmark_pairs(
            pairs,
            split=split,
            split_seed=split_seed,
            speeds=speeds,
            train_fraction=train_fraction,
            validation_fraction=validation_fraction,
            test_fraction=test_fraction,
        )
    if speeds is None:
        raise ValueError(
            "--speeds is required when --zero-shot-speeds is specified."
        )

    seen_speeds = {int(speed) for speed in speeds}
    unseen_speeds = {int(speed) for speed in zero_shot_speeds}
    overlap = sorted(seen_speeds & unseen_speeds)
    if overlap:
        raise ValueError(
            "Seen and zero-shot speed selections overlap: "
            f"{overlap} km/h."
        )

    held_out_pairs = select_benchmark_pairs(
        pairs,
        split=split,
        split_seed=split_seed,
        speeds=sorted(seen_speeds),
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
    )
    zero_shot_pairs = select_benchmark_pairs(
        pairs,
        split="all",
        split_seed=split_seed,
        speeds=sorted(unseen_speeds),
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
    )
    selected_pairs = tuple(sorted((*held_out_pairs, *zero_shot_pairs)))
    if len({pair.key for pair in selected_pairs}) != len(selected_pairs):
        raise ValueError("Mixed benchmark trajectory selections overlap.")
    return selected_pairs


def main(root_path: str, args: Namespace) -> int:
    """Run the complete benchmark and write CSV/JSON outputs."""
    data_dir = args.data_dir
    if not os.path.isabs(data_dir):
        data_dir = os.path.join(root_path, data_dir)
    checkpoint_path = os.path.abspath(
        os.path.join(root_path, args.checkpoint)
    )
    agent, checkpoint_extra = SMDPPPO.from_checkpoint(
        checkpoint_path,
        device=args.device,
    )
    objective = checkpoint_extra.get("objective", {})
    candidate_count = int(
        objective.get("candidate_count", agent.model.action_size - 1)
    )
    candidate_order = str(
        objective.get("candidate_order_key", "legacy")
    )
    observation_schema = str(
        objective.get("observation_schema_key", "legacy_v1")
    )
    reward_schema = str(objective.get("reward", LEGACY_REWARD_SCHEMA))
    reward_config = reward_config_from_dict(
        reward_schema,
        objective.get("reward_config"),
    )
    protocol_config = load_protocol_config(data_dir, checkpoint_extra)
    if args.split_seed is None:
        args.split_seed = checkpoint_extra.get("split_seed", 17)
    checkpoint_fractions = checkpoint_extra.get("split_fractions", {})
    split_fractions = {
        "train": (
            args.train_fraction
            if args.train_fraction is not None
            else checkpoint_fractions.get("train", 0.70)
        ),
        "validation": (
            args.validation_fraction
            if args.validation_fraction is not None
            else checkpoint_fractions.get("validation", 0.15)
        ),
        "test": (
            args.test_fraction
            if args.test_fraction is not None
            else checkpoint_fractions.get("test", 0.15)
        ),
    }
    pairs = discover_trace_pairs(data_dir)
    if args.split == "all":
        selected_pairs = select_mixed_benchmark_pairs(
            pairs, split="all", split_seed=args.split_seed, speeds=args.speeds,
            zero_shot_speeds=args.zero_shot_speeds,
        )
    else:
        selected_pairs = checkpoint_pairs(
            data_dir, checkpoint_path, checkpoint_extra, split=args.split,
            speeds=args.speeds, split_seed=args.split_seed,
            fractions={"train": args.train_fraction, "validation": args.validation_fraction, "test": args.test_fraction},
        )
        if args.zero_shot_speeds:
            trained_speeds = set(checkpoint_extra.get("selected_speeds_kmh", args.speeds or []))
            if trained_speeds & set(args.zero_shot_speeds):
                raise ValueError("Zero-shot speeds overlap the checkpoint's training speeds.")
            zero_shot_pairs = filter_trace_pairs_by_speed(pairs, args.zero_shot_speeds)
            selected_pairs = tuple(sorted((*selected_pairs, *zero_shot_pairs)))
    if len({pair.key for pair in selected_pairs}) != len(selected_pairs):
        raise ValueError("Benchmark selections contain overlapping trajectories.")
    descriptors = [
        TrajectoryDescriptor(
            trace_id=f"{pair.speed_kmh}kmh_{pair.trajectory_id}",
            speed_kmh=pair.speed_kmh,
            trajectory_id=pair.trajectory_id,
        )
        for pair in selected_pairs
    ]
    policies = build_benchmark_policies(agent)

    all_rows = []
    for policy in policies:
        env = HandoverSMDPEnv(
            protocol_config,
            data_dir=data_dir,
            trace_pairs=selected_pairs,
            gamma_per_second=agent.config.gamma_per_second,
            candidate_count=candidate_count,
            candidate_order=candidate_order,
            observation_schema=observation_schema,
            reward_schema=reward_schema,
            reward_config=reward_config,
        )
        env.load_reward_state(checkpoint_extra.get("reward_state"))
        env.validate_checkpoint_objective(checkpoint_extra)
        all_rows.extend(
            benchmark_policy(
                env,
                policy,
                descriptors,
                progress_callback=lambda message: print(
                    message,
                    flush=True,
                ),
            )
        )

    aggregate_rows = aggregate_benchmark_rows(all_rows, by_speed=False)
    speed_rows = aggregate_benchmark_rows(all_rows, by_speed=True)
    comparisons = build_paired_comparisons(
        all_rows,
        reference_policy="3gpp_always_ho",
    )
    output_dir = os.path.abspath(
        os.path.join(root_path, args.output_dir)
    )
    paths = write_benchmark_outputs(
        output_dir,
        all_rows,
        aggregate_rows,
        speed_rows,
        comparisons,
        metadata={
            "checkpoint": checkpoint_path,
            "protocol_config": asdict(protocol_config),
            "checkpoint_extra": checkpoint_extra,
            "data_dir": data_dir,
            "split": args.split,
            "split_seed": args.split_seed,
            "split_fractions": split_fractions,
            "speeds": sorted(set(args.speeds)) if args.speeds else None,
            "zero_shot_speeds": (
                sorted(set(args.zero_shot_speeds))
                if args.zero_shot_speeds
                else None
            ),
            "trajectories": len(selected_pairs),
            "reference_policy": "3gpp_always_ho",
        },
    )

    print(
        f"Benchmark selection: split={args.split}, "
        f"trajectories={len(selected_pairs)}, "
        f"fractions={split_fractions}, "
        f"zero-shot speeds={args.zero_shot_speeds}",
        flush=True,
    )
    print("\nAggregate benchmark")
    print(
        "policy             relative(%)  HOF(%)  RLF-time(%)  PP(%)  "
        "HO-action(%)  STAY(%)  return"
    )
    for row in aggregate_rows:
        print(
            f"{row['policy']:<18}"
            f"{row['mean_relative_rate_percent']:>13.3f}"
            f"{row['handover_failure_percent']:>8.3f}"
            f"{row['rlf_time_percent']:>13.3f}"
            f"{row['ping_pong_percent']:>7.3f}"
            f"{row['handover_action_percent']:>13.1f}"
            f"{row['stay_action_percent']:>9.1f}"
            f"{row['mean_controlled_return']:>10.3f}"
        )
    print("\nPaired comparison versus 3gpp_always_ho")
    for row in comparisons:
        print(
            f"{row['policy']}: "
            f"rate {row['mean_rate_delta_mbps']:+.3f} Mbps, "
            f"RLF {row['total_rlf_delta']:+d}, "
            f"PP {row['total_ping_pong_delta']:+d}, "
            f"return {row['mean_return_delta']:+.3f}"
        )
    print("\nOutputs")
    for name, path in paths.items():
        print(f"{name}: {path}")
    return 0
