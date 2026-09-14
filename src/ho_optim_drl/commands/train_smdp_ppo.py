"""Train duration-aware PPO on trajectory-level dataset splits."""

from __future__ import annotations

from argparse import Namespace
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import csv
import hashlib
import json
import os
import time
from typing import Any

from ho_optim_drl import HandoverSMDPEnv
from ho_optim_drl.data.project import load_protocol_config
from ho_optim_drl.agents.ppo import (
    SMDPPPO,
    SMDPPPOConfig,
    ValidationCheckpointManager,
)
from ho_optim_drl.data import (
    TracePair,
    TraceSplits,
    discover_trace_pairs,
    filter_trace_pairs_by_speed,
    stratified_trajectory_split,
)
from ho_optim_drl.evaluation import (
    BenchmarkPolicy,
    TrajectoryDescriptor,
    aggregate_benchmark_rows,
    benchmark_policy,
)
from ho_optim_drl.smdp import HandoverAction


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source_file:
        for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_record(
    data_dir: str,
    split_name: str,
    pair: TracePair,
) -> dict[str, Any]:
    rsrp_path = os.path.join(data_dir, pair.rsrp_file)
    sinr_path = os.path.join(data_dir, pair.sinr_file)
    return {
        "split": split_name,
        "trace_id": f"{pair.speed_kmh}kmh_{pair.trajectory_id}",
        "speed_kmh": pair.speed_kmh,
        "trajectory_id": pair.trajectory_id,
        "rsrp_file": pair.rsrp_file,
        "sinr_file": pair.sinr_file,
        "rsrp_size_bytes": os.path.getsize(rsrp_path),
        "sinr_size_bytes": os.path.getsize(sinr_path),
        "rsrp_sha256": _sha256_file(rsrp_path),
        "sinr_sha256": _sha256_file(sinr_path),
    }


def build_dataset_manifest(
    data_dir: str,
    splits: TraceSplits,
    *,
    selected_speeds: list[int],
    split_seed: int,
    split_fractions: dict[str, float],
) -> dict[str, Any]:
    """Build an immutable split/file fingerprint for training auditability."""
    split_pairs = {
        "train": splits.train,
        "validation": splits.validation,
        "test": splits.test,
    }
    records = {
        split_name: [
            _manifest_record(data_dir, split_name, pair) for pair in pairs
        ]
        for split_name, pairs in split_pairs.items()
    }
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_dir": os.path.abspath(data_dir),
        "selected_speeds_kmh": selected_speeds,
        "split_seed": split_seed,
        "split_fractions": split_fractions,
        "split_sizes": {
            split_name: len(split_records)
            for split_name, split_records in records.items()
        },
        "splits": records,
    }


def write_dataset_manifest(
    output_dir: str,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    """Persist the audit manifest in JSON and flat CSV forms."""
    json_path = os.path.join(output_dir, "dataset_split_manifest.json")
    csv_path = os.path.join(output_dir, "dataset_split_manifest.csv")
    with open(json_path, "w", encoding="utf-8") as manifest_file:
        json.dump(
            manifest,
            manifest_file,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )

    fieldnames = [
        "split",
        "trace_id",
        "speed_kmh",
        "trajectory_id",
        "rsrp_file",
        "sinr_file",
        "rsrp_size_bytes",
        "sinr_size_bytes",
        "rsrp_sha256",
        "sinr_sha256",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as manifest_file:
        writer = csv.DictWriter(manifest_file, fieldnames=fieldnames)
        writer.writeheader()
        for split_name in ("train", "validation", "test"):
            writer.writerows(manifest["splits"][split_name])

    return {
        "json": json_path,
        "csv": csv_path,
        "sha256": _sha256_file(json_path),
    }


def build_3gpp_validation_reference(
    protocol_config: Any,
    data_dir: str,
    trace_pairs: tuple[TracePair, ...],
    *,
    gamma_per_second: float,
) -> dict[str, Any]:
    """Evaluate Always-HO once as the QoS reference for checkpoint selection."""
    descriptors = [
        TrajectoryDescriptor(
            trace_id=f"{pair.speed_kmh}kmh_{pair.trajectory_id}",
            speed_kmh=pair.speed_kmh,
            trajectory_id=pair.trajectory_id,
        )
        for pair in trace_pairs
    ]
    reference_env = HandoverSMDPEnv(
        protocol_config,
        data_dir=data_dir,
        trace_pairs=trace_pairs,
        gamma_per_second=gamma_per_second,
    )
    rows = benchmark_policy(
        reference_env,
        BenchmarkPolicy(
            "3gpp_always_ho",
            lambda observation: HandoverAction.HANDOVER,
        ),
        descriptors,
    )
    actionable_rows = [row for row in rows if row["actionable"]]
    if not actionable_rows:
        raise RuntimeError(
            "Validation split contains no actionable A3 trajectory."
        )
    aggregate = aggregate_benchmark_rows(
        actionable_rows,
        by_speed=False,
    )[0]
    by_speed_rows = aggregate_benchmark_rows(
        actionable_rows,
        by_speed=True,
    )
    return {
        "label": "3gpp_always_ho",
        "trajectories": len(rows),
        "actionable_trajectories": len(actionable_rows),
        "skipped_no_decision": len(rows) - len(actionable_rows),
        "mean_estimated_rate_mbps": aggregate[
            "mean_estimated_rate_mbps"
        ],
        "mean_relative_rate_percent": aggregate[
            "mean_relative_rate_percent"
        ],
        "mean_estimated_rate_mbps_by_speed": {
            str(row["speed_kmh"]): row["mean_estimated_rate_mbps"]
            for row in by_speed_rows
        },
        "aggregate_metrics": aggregate,
        "by_speed_metrics": by_speed_rows,
    }


def aggregate_training_trace_usage(
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize exact macro-step and completed-episode usage by trace."""
    macro_steps: Counter[str] = Counter()
    completed_episodes: Counter[str] = Counter()
    for update in history:
        macro_steps.update(update.get("training_trace_macro_steps", {}))
        completed_episodes.update(update.get("completed_episode_trace_ids", []))
    return {
        "macro_steps_by_trace": dict(sorted(macro_steps.items())),
        "completed_episodes_by_trace": dict(sorted(completed_episodes.items())),
        "total_logged_macro_steps": sum(macro_steps.values()),
        "total_completed_episodes": sum(completed_episodes.values()),
    }


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
            labels = []
            if event.get("is_best_return"):
                labels.append("best-return")
            if event.get("is_best_qos"):
                labels.append("best-QoS")
            if event.get("is_best_rlf"):
                labels.append("best-RLF")
            if event.get("is_pareto"):
                labels.append("Pareto")
            if not labels and event["is_best"]:
                labels.append("best")
            marker = f" | saved: {', '.join(labels)}" if labels else ""
            qos_text = ""
            if "aggregate_rate_ratio_to_3gpp" in event:
                qos_state = "pass" if event["qos_feasible"] else "fail"
                qos_text = (
                    f" | QoS {100 * event['aggregate_rate_ratio_to_3gpp']:.1f}%"
                    f" ({qos_state})"
                )
            print(
                f"[validation] return {event['mean_return']:.3f} +/- "
                f"{event['std_return']:.3f} | "
                f"rate {event['mean_estimated_rate_mbps']:.3f} Mbps | "
                f"RLF-time {event['rlf_time_percent']:.3f}% | "
                f"HOF {event['handover_failure_percent']:.3f}% | "
                f"PP {event['ping_pong_percent']:.3f}% | "
                f"HO {100 * event['handover_fraction']:.1f}% | "
                f"WAIT_50 {100 * event['wait_fraction']:.1f}%"
                f"{qos_text}"
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
    data_dir = args.data_dir
    if not os.path.isabs(data_dir):
        data_dir = os.path.join(root_path, data_dir)
    pairs = filter_trace_pairs_by_speed(
        discover_trace_pairs(data_dir),
        args.speeds,
    )
    selected_speeds = sorted({pair.speed_kmh for pair in pairs})
    split_fractions = {
        "train": args.train_fraction,
        "validation": args.validation_fraction,
        "test": args.test_fraction,
    }
    splits = stratified_trajectory_split(
        pairs,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
        seed=args.split_seed,
    )
    protocol_config = load_protocol_config(data_dir)
    if any(not group for group in (splits.train, splits.validation, splits.test)):
        raise ValueError("Train, validation and test must each contain trajectories; provide more routes or adjust split fractions.")
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
    if os.path.isdir(output_dir) and any(name.endswith(".pt") or name == "training_report.json" for name in os.listdir(output_dir)):
        raise FileExistsError("Training output contains a previous run; choose a new --output-dir.")
    os.makedirs(output_dir, exist_ok=True)
    manifest = build_dataset_manifest(
        data_dir,
        splits,
        selected_speeds=selected_speeds,
        split_seed=args.split_seed,
        split_fractions=split_fractions,
    )
    manifest_paths = write_dataset_manifest(output_dir, manifest)
    print("Building validation 3GPP Always-HO QoS reference...", flush=True)
    validation_reference = build_3gpp_validation_reference(
        protocol_config,
        data_dir,
        splits.validation,
        gamma_per_second=ppo_config.gamma_per_second,
    )
    validation_reference_path = os.path.join(
        output_dir,
        "validation_3gpp_reference.json",
    )
    with open(
        validation_reference_path,
        "w",
        encoding="utf-8",
    ) as reference_file:
        json.dump(
            validation_reference,
            reference_file,
            indent=2,
            allow_nan=False,
        )
    audit_metadata = {
        "protocol_config": asdict(protocol_config),
        "dataset_manifest_file": os.path.basename(manifest_paths["json"]),
        "dataset_manifest_sha256": manifest_paths["sha256"],
        "validation_3gpp_reference_file": os.path.basename(
            validation_reference_path
        ),
        "selected_speeds_kmh": selected_speeds,
        "split_seed": args.split_seed,
        "split_fractions": split_fractions,
    }
    latest_path = os.path.join(output_dir, "smdp_ppo_latest.pt")
    final_path = os.path.join(output_dir, "smdp_ppo_final.pt")
    metrics_path = os.path.join(output_dir, "training_metrics.jsonl")
    checkpoint_manager = ValidationCheckpointManager(
        output_dir,
        reference_rate_mbps=validation_reference[
            "mean_estimated_rate_mbps"
        ],
        reference_rates_by_speed=validation_reference[
            "mean_estimated_rate_mbps_by_speed"
        ],
        qos_rate_ratio=args.qos_rate_ratio,
        qos_rate_ratio_per_speed=args.qos_rate_ratio_per_speed,
        checkpoint_extra=audit_metadata,
    )
    progress_logger = TrainingProgressLogger(
        metrics_path,
        args.total_macro_steps,
    )
    print(
        f"Dataset split: train={len(splits.train)}, "
        f"validation={len(splits.validation)}, test={len(splits.test)}",
        flush=True,
    )
    print(
        "Split fractions: "
        f"train={args.train_fraction:.3f}, "
        f"validation={args.validation_fraction:.3f}, "
        f"test={args.test_fraction:.3f}",
        flush=True,
    )
    print(f"Selected speeds: {selected_speeds} km/h", flush=True)
    print(f"Data directory: {data_dir}", flush=True)
    print(f"Dataset manifest: {manifest_paths['json']}", flush=True)
    print(f"Manifest SHA-256: {manifest_paths['sha256']}", flush=True)
    print(
        "Validation 3GPP rate: "
        f"{validation_reference['mean_estimated_rate_mbps']:.3f} Mbps",
        flush=True,
    )
    print(
        "QoS floors: aggregate="
        f"{100 * args.qos_rate_ratio:.1f}% of 3GPP, per-speed="
        f"{100 * args.qos_rate_ratio_per_speed:.1f}% of 3GPP",
        flush=True,
    )
    print(f"Live metrics: {metrics_path}", flush=True)
    history = agent.train(
        train_env,
        total_macro_steps=args.total_macro_steps,
        validation_env=validation_env,
        evaluation_interval=args.evaluation_interval,
        checkpoint_extra=audit_metadata,
        checkpoint_manager=checkpoint_manager,
        progress_callback=progress_logger,
        progress_interval=args.progress_interval,
    )
    checkpoint_summary = checkpoint_manager.summary()
    agent.save_checkpoint(
        latest_path,
        extra={
            **audit_metadata,
            "checkpoint_selection_summary": checkpoint_summary,
        },
    )

    selected_path = (
        checkpoint_summary["best_qos"]
        or checkpoint_summary["best_return"]
        or latest_path
    )
    selection_reason = (
        "best_qos"
        if checkpoint_summary["best_qos"]
        else "best_return_fallback"
        if checkpoint_summary["best_return"]
        else "latest_fallback"
    )
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
            **audit_metadata,
            "selected_from": selected_path,
            "selection_reason": selection_reason,
            "selection_metadata": selection_metadata,
            "checkpoint_selection_summary": checkpoint_summary,
            "validation": asdict(validation_metrics),
            "test": asdict(test_metrics),
        },
    )

    report = {
        "data_dir": data_dir,
        "selected_speeds_kmh": selected_speeds,
        "dataset_manifest": manifest_paths,
        "ppo_config": asdict(ppo_config),
        "protocol_config": asdict(protocol_config),
        "split_seed": args.split_seed,
        "split_fractions": split_fractions,
        "split_sizes": {
            "train": len(splits.train),
            "validation": len(splits.validation),
            "test": len(splits.test),
        },
        "history": history,
        "training_trace_usage": aggregate_training_trace_usage(history),
        "validation_3gpp_reference": validation_reference,
        "checkpoint_selection": checkpoint_summary,
        "selection_reason": selection_reason,
        "validation": asdict(validation_metrics),
        "test": asdict(test_metrics),
        "checkpoints": {
            "best": checkpoint_summary["best_return"],
            "best_return": checkpoint_summary["best_return"],
            "best_qos": checkpoint_summary["best_qos"],
            "best_rlf": checkpoint_summary["best_rlf"],
            "all": checkpoint_summary["all_dir"],
            "pareto": checkpoint_summary["pareto_dir"],
            "latest": latest_path,
            "selected_from": selected_path,
            "final": final_path,
            "metrics": metrics_path,
            "validation_metrics_csv": checkpoint_summary["metrics_csv"],
            "validation_metrics_json": checkpoint_summary["metrics_json"],
            "pareto_front_json": checkpoint_summary["pareto_json"],
            "pareto_front_csv": checkpoint_summary["pareto_csv"],
            "validation_3gpp_reference": validation_reference_path,
        },
    }
    report_path = os.path.join(output_dir, "training_report.json")
    with open(report_path, "w", encoding="utf-8") as report_file:
        json.dump(report, report_file, indent=2)

    print(f"Training complete: {agent.total_macro_steps} macro-steps")
    print(f"Selected checkpoint: {selected_path}")
    print(f"Selection reason: {selection_reason}")
    print(f"Validation mean return: {validation_metrics.mean_return:.6f}")
    print(f"Test mean return: {test_metrics.mean_return:.6f}")
    print(f"Final checkpoint: {final_path}")
    print(f"Dataset manifest: {manifest_paths['json']}")
    print(f"Training report: {report_path}")
    return 0
