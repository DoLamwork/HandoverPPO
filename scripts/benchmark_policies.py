"""Benchmark PPO, 3GPP Always-HO, and Always-Stay on one dataset split."""

from __future__ import annotations

from argparse import Namespace
import os

from ho_optim_drl import Config, HandoverSMDPEnv
from ho_optim_drl.agents.ppo import SMDPPPO
from ho_optim_drl.data import discover_trace_pairs, stratified_trajectory_split
from ho_optim_drl.evaluation import (
    BenchmarkPolicy,
    TrajectoryDescriptor,
    aggregate_benchmark_rows,
    benchmark_policy,
    build_paired_comparisons,
    write_benchmark_outputs,
)
from ho_optim_drl.smdp import HandoverAction


def main(root_path: str, args: Namespace) -> int:
    """Run the complete benchmark and write CSV/JSON outputs."""
    data_dir = os.path.join(root_path, "data", "processed")
    pairs = discover_trace_pairs(data_dir)
    splits = stratified_trajectory_split(pairs, seed=args.split_seed)
    selected_pairs = (
        splits.validation if args.split == "validation" else splits.test
    )
    descriptors = [
        TrajectoryDescriptor(
            trace_id=f"{pair.speed_kmh}kmh_{pair.trajectory_id}",
            speed_kmh=pair.speed_kmh,
            trajectory_id=pair.trajectory_id,
        )
        for pair in selected_pairs
    ]
    checkpoint_path = os.path.abspath(
        os.path.join(root_path, args.checkpoint)
    )
    agent, checkpoint_extra = SMDPPPO.from_checkpoint(
        checkpoint_path,
        observation_size=19,
        action_size=2,
        device=args.device,
    )
    policies = [
        BenchmarkPolicy(
            "ppo",
            lambda observation: agent.predict(
                observation,
                deterministic=True,
            ),
        ),
        BenchmarkPolicy(
            "3gpp_always_ho",
            lambda observation: HandoverAction.HANDOVER,
        ),
        BenchmarkPolicy(
            "always_stay",
            lambda observation: HandoverAction.STAY,
        ),
    ]

    all_rows = []
    for policy in policies:
        env = HandoverSMDPEnv(
            Config(),
            data_dir=data_dir,
            trace_pairs=selected_pairs,
            gamma_per_second=agent.config.gamma_per_second,
        )
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
            "checkpoint_extra": checkpoint_extra,
            "split": args.split,
            "split_seed": args.split_seed,
            "trajectories": len(selected_pairs),
            "reference_policy": "3gpp_always_ho",
        },
    )

    print("\nAggregate benchmark")
    print(
        "policy             rate(Mbps)  relative(%)  RLF  PP  HOF  "
        "HO-action(%)  return"
    )
    for row in aggregate_rows:
        print(
            f"{row['policy']:<18}"
            f"{row['duration_weighted_rate_mbps']:>11.3f}"
            f"{row['mean_relative_rate_percent']:>13.3f}"
            f"{row['rlf_events']:>5d}"
            f"{row['ping_pong_events']:>4d}"
            f"{row['handover_failures']:>5d}"
            f"{100 * row['handover_fraction']:>13.1f}"
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
