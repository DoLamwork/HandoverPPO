"""Evaluate a saved duration-aware PPO checkpoint."""

from __future__ import annotations

from argparse import Namespace
from dataclasses import asdict
import json
import os

from ho_optim_drl import HandoverSMDPEnv
from ho_optim_drl.agents.ppo import SMDPPPO
from ho_optim_drl.data.project import checkpoint_pairs, load_protocol_config
from ho_optim_drl.smdp import (
    LEGACY_REWARD_SCHEMA,
    reward_config_from_dict,
)


def main(root_path: str, args: Namespace) -> int:
    """Evaluate one checkpoint on validation or test trajectories."""
    data_dir = os.path.abspath(os.path.join(root_path, args.data_dir))
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
    decision_trigger = str(objective.get("trigger", "completed_a3_ttt"))
    reward_schema = str(objective.get("reward", LEGACY_REWARD_SCHEMA))
    reward_config = reward_config_from_dict(
        reward_schema,
        objective.get("reward_config"),
    )
    stay_mode = str(objective.get("stay_mode", "reject_event_v1"))
    stay_redecision_ms = int(objective.get("stay_redecision_ms", 100))
    selected_pairs = checkpoint_pairs(
        data_dir, checkpoint_path, checkpoint_extra, split=args.split,
        speeds=args.speeds, split_seed=args.split_seed,
    )
    protocol_config = load_protocol_config(data_dir, checkpoint_extra)
    env = HandoverSMDPEnv(
        protocol_config,
        data_dir=data_dir,
        trace_pairs=selected_pairs,
        gamma_per_second=agent.config.gamma_per_second,
        candidate_count=candidate_count,
        candidate_order=candidate_order,
        observation_schema=observation_schema,
        decision_trigger=decision_trigger,
        reward_schema=reward_schema,
        reward_config=reward_config,
        stay_mode=stay_mode,
        stay_redecision_ms=stay_redecision_ms,
    )
    env.load_reward_state(checkpoint_extra.get("reward_state"))
    env.validate_checkpoint_objective(checkpoint_extra)
    metrics = agent.evaluate(env)
    result = {
        "checkpoint": checkpoint_path,
        "checkpoint_extra": checkpoint_extra,
        "data_dir": data_dir,
        "protocol_config": asdict(protocol_config),
        "trace_ids": [f"{pair.speed_kmh}kmh_{pair.trajectory_id}" for pair in selected_pairs],
        "split": args.split,
        "split_seed": checkpoint_extra.get("split_seed", 17) if args.split_seed is None else args.split_seed,
        "metrics": asdict(metrics),
    }
    if args.output:
        from pathlib import Path
        output = Path(root_path) / args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0
