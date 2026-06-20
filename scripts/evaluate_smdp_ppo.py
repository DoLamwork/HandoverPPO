"""Evaluate a saved duration-aware PPO checkpoint."""

from __future__ import annotations

from argparse import Namespace
from dataclasses import asdict
import json
import os

from ho_optim_drl import Config, HandoverSMDPEnv
from ho_optim_drl.agents.ppo import SMDPPPO
from ho_optim_drl.data import discover_trace_pairs, stratified_trajectory_split


def main(root_path: str, args: Namespace) -> int:
    """Evaluate one checkpoint on validation or test trajectories."""
    data_dir = os.path.join(root_path, "data", "processed")
    pairs = discover_trace_pairs(data_dir)
    splits = stratified_trajectory_split(pairs, seed=args.split_seed)
    selected_pairs = (
        splits.validation if args.split == "validation" else splits.test
    )
    checkpoint_path = os.path.abspath(
        os.path.join(root_path, args.checkpoint)
    )

    agent, checkpoint_extra = SMDPPPO.from_checkpoint(
        checkpoint_path,
        observation_size=19,
        action_size=2,
        device=args.device,
    )
    env = HandoverSMDPEnv(
        Config(),
        data_dir=data_dir,
        trace_pairs=selected_pairs,
        gamma_per_second=agent.config.gamma_per_second,
    )
    metrics = agent.evaluate(env)
    result = {
        "checkpoint": checkpoint_path,
        "checkpoint_extra": checkpoint_extra,
        "split": args.split,
        "split_seed": args.split_seed,
        "metrics": asdict(metrics),
    }
    print(json.dumps(result, indent=2))
    return 0
