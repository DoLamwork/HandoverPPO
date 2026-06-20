"""Train duration-aware PPO on trajectory-level dataset splits."""

from __future__ import annotations

from argparse import Namespace
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
import time
from typing import Any

from ho_optim_drl import Config, HandoverSMDPEnv
from ho_optim_drl.agents.ppo import SMDPPPO, SMDPPPOConfig
from ho_optim_drl.data import discover_trace_pairs, stratified_trajectory_split


class TrainingProgressLogger:
    """Print concise progress and persist every event as JSON Lines."""

    def __init__(self, path: str, target_macro_steps: int) -> None:
        self.path = path
        self.target_macro_steps = target_macro_steps
        self.started_at = time.perf_counter()
        self.last_progress_at = self.started_at
        self.last_progress_steps = 0
        with open(self.path, "w", encoding="utf-8"):
            pass

    def __call__(self, event: dict[str, Any]) -> None:
        now = time.perf_counter()
        record = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": now - self.started_at,
            **event,
        }
        with open(self.path, "a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(record, allow_nan=False) + "\n")

        event_name = event["event"]
        if event_name == "training_started":
            self.last_progress_steps = int(event["total_macro_steps"])
            self.last_progress_at = now
            print(
                f"[train] target={self.target_macro_steps} macro-steps | "
                f"rollout={event['rollout_steps']} | device={event['device']}",
                flush=True,
            )
        elif event_name == "rollout_started":
            print(
                f"[collect {event['update']:04d}] "
                f"starting {event['rollout_steps']} macro-steps",
                flush=True,
            )
        elif event_name == "rollout_progress":
            total_steps = int(event["total_macro_steps"])
            delta_steps = total_steps - self.last_progress_steps
            delta_time = max(now - self.last_progress_at, 1e-9)
            steps_per_second = delta_steps / delta_time
            self.last_progress_steps = total_steps
            self.last_progress_at = now
            print(
                f"[collect {event['update']:04d}] "
                f"{event['collected_steps']:4d}/{event['rollout_steps']} | "
                f"total {total_steps:7d}/{self.target_macro_steps} | "
                f"episodes {event['completed_episodes']:3d} | "
                f"{steps_per_second:6.2f} step/s",
                flush=True,
            )
        elif event_name == "update_complete":
            episode_return = event["mean_training_episode_return"]
            return_text = (
                f"{episode_return:8.3f}"
                if episode_return is not None
                else "     n/a"
            )
            print(
                f"[update {event['update']:04d}] "
                f"steps {event['macro_steps']:7d}/{self.target_macro_steps} | "
                f"return {return_text} | "
                f"pi {event['policy_loss']:+.4f} | "
                f"vf {event['value_loss']:.4f} | "
                f"ent {event['entropy']:.4f} | "
                f"kl {event['approximate_kl']:.5f} | "
                f"clip {event['clip_fraction']:.3f} | "
                f"ev {event['explained_variance']:+.3f}",
                flush=True,
            )
        elif event_name == "validation_started":
            print(
                f"[validation] evaluating "
                f"{event['trajectories']} trajectories...",
                flush=True,
            )
        elif event_name == "validation_complete":
            marker = " | best saved" if event["is_best"] else ""
            print(
                f"[validation] return {event['mean_return']:.3f} +/- "
                f"{event['std_return']:.3f} | "
                f"rate {event['mean_estimated_rate_mbps']:.3f} Mbps | "
                f"RLF {event['total_rlf']} | "
                f"PP {event['total_ping_pong']} | "
                f"HO {100 * event['handover_fraction']:.1f}%"
                f"{marker}",
                flush=True,
            )
        elif event_name == "training_complete":
            print(
                f"[train] completed {event['total_macro_steps']} "
                f"macro-steps in {now - self.started_at:.1f}s",
                flush=True,
            )


def main(root_path: str, args: Namespace) -> int:
    """Train PPO, save checkpoints, and report validation/test metrics."""
    data_dir = os.path.join(root_path, "data", "processed")
    pairs = discover_trace_pairs(data_dir)
    splits = stratified_trajectory_split(pairs, seed=args.split_seed)
    protocol_config = Config()
    ppo_config = SMDPPPOConfig(
        hidden_sizes=tuple(args.hidden_sizes),
        learning_rate=args.learning_rate,
        rollout_steps=args.rollout_steps,
        batch_size=args.batch_size,
        update_epochs=args.update_epochs,
        clip_range=args.clip_range,
        value_coefficient=args.value_coefficient,
        entropy_coefficient=args.entropy_coefficient,
        max_grad_norm=args.max_grad_norm,
        gamma_per_second=args.gamma_per_second,
        lambda_per_second=args.lambda_per_second,
        target_kl=args.target_kl,
        seed=args.seed,
        device=args.device,
    )
    ppo_config.validate()

    train_env = HandoverSMDPEnv(
        protocol_config,
        data_dir=data_dir,
        trace_pairs=splits.train,
        gamma_per_second=ppo_config.gamma_per_second,
    )
    validation_env = HandoverSMDPEnv(
        protocol_config,
        data_dir=data_dir,
        trace_pairs=splits.validation,
        gamma_per_second=ppo_config.gamma_per_second,
    )
    test_env = HandoverSMDPEnv(
        protocol_config,
        data_dir=data_dir,
        trace_pairs=splits.test,
        gamma_per_second=ppo_config.gamma_per_second,
    )
    agent = SMDPPPO(
        train_env.observation_space.shape[0],
        train_env.action_space.n,
        ppo_config,
    )

    output_dir = os.path.abspath(
        os.path.join(root_path, args.output_dir)
    )
    os.makedirs(output_dir, exist_ok=True)
    best_path = os.path.join(output_dir, "smdp_ppo_best.pt")
    latest_path = os.path.join(output_dir, "smdp_ppo_latest.pt")
    final_path = os.path.join(output_dir, "smdp_ppo_final.pt")
    metrics_path = os.path.join(output_dir, "training_metrics.jsonl")
    progress_logger = TrainingProgressLogger(
        metrics_path,
        args.total_macro_steps,
    )
    print(
        f"Dataset split: train={len(splits.train)}, "
        f"validation={len(splits.validation)}, test={len(splits.test)}",
        flush=True,
    )
    print(f"Live metrics: {metrics_path}", flush=True)
    history = agent.train(
        train_env,
        total_macro_steps=args.total_macro_steps,
        validation_env=validation_env,
        evaluation_interval=args.evaluation_interval,
        checkpoint_path=best_path,
        progress_callback=progress_logger,
        progress_interval=args.progress_interval,
    )
    agent.save_checkpoint(latest_path)

    selected_path = best_path if os.path.exists(best_path) else latest_path
    print(f"Selecting checkpoint: {selected_path}", flush=True)
    selected_agent, selection_metadata = SMDPPPO.from_checkpoint(
        selected_path,
        observation_size=train_env.observation_space.shape[0],
        action_size=train_env.action_space.n,
        device=ppo_config.device,
        load_optimizer=True,
    )
    print("Running final validation evaluation...", flush=True)
    validation_metrics = selected_agent.evaluate(validation_env)
    print("Running held-out test evaluation...", flush=True)
    test_metrics = selected_agent.evaluate(test_env)
    selected_agent.save_checkpoint(
        final_path,
        extra={
            "selected_from": selected_path,
            "selection_metadata": selection_metadata,
            "validation": asdict(validation_metrics),
            "test": asdict(test_metrics),
        },
    )

    report = {
        "ppo_config": asdict(ppo_config),
        "split_seed": args.split_seed,
        "split_sizes": {
            "train": len(splits.train),
            "validation": len(splits.validation),
            "test": len(splits.test),
        },
        "history": history,
        "validation": asdict(validation_metrics),
        "test": asdict(test_metrics),
        "checkpoints": {
            "best": best_path if os.path.exists(best_path) else None,
            "latest": latest_path,
            "selected_from": selected_path,
            "final": final_path,
            "metrics": metrics_path,
        },
    }
    report_path = os.path.join(output_dir, "training_report.json")
    with open(report_path, "w", encoding="utf-8") as report_file:
        json.dump(report, report_file, indent=2)

    print(f"Training complete: {agent.total_macro_steps} macro-steps")
    print(f"Selected checkpoint: {selected_path}")
    print(f"Validation mean return: {validation_metrics.mean_return:.6f}")
    print(f"Test mean return: {test_metrics.mean_return:.6f}")
    print(f"Final checkpoint: {final_path}")
    print(f"Training report: {report_path}")
    return 0
