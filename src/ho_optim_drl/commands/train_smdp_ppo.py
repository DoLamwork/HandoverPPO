"""Train duration-aware PPO on trajectory-level dataset splits."""

from __future__ import annotations

from argparse import Namespace
from collections import Counter, defaultdict
from dataclasses import asdict, replace
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
import os
import subprocess
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
from ho_optim_drl.smdp import (
    HandoverAction,
    LEGACY_REWARD_SCHEMA,
    trajectory_reward_config,
)


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source_file:
        for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return _sha256_bytes(payload)


def capture_source_state(root_path: str, output_dir: str) -> dict[str, Any]:
    """Persist the Git state and a hash manifest of executable sources."""
    def git(*arguments: str) -> subprocess.CompletedProcess[bytes]:
        return subprocess.run(
            ["git", *arguments],
            cwd=root_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

    revision_result = git("rev-parse", "HEAD")
    branch_result = git("branch", "--show-current")
    status_result = git("status", "--porcelain=v1", "--untracked-files=all")
    patch_result = git(
        "diff",
        "--binary",
        "HEAD",
        "--",
        "run.py",
        "src",
        "scripts",
        "tests",
        "pyproject.toml",
        "setup.py",
    )
    patch = patch_result.stdout
    patch_path = os.path.join(output_dir, "source_worktree.patch")
    with open(patch_path, "wb") as patch_file:
        patch_file.write(patch)

    roots = ["src", "scripts", "tests"]
    candidates = ["run.py", "pyproject.toml", "setup.py"]
    source_paths: list[str] = []
    for relative in candidates:
        if os.path.isfile(os.path.join(root_path, relative)):
            source_paths.append(relative)
    for relative_root in roots:
        absolute_root = os.path.join(root_path, relative_root)
        if not os.path.isdir(absolute_root):
            continue
        for directory, _, names in os.walk(absolute_root):
            for name in names:
                if name.endswith((".py", ".toml", ".yaml", ".yml", ".json")):
                    absolute = os.path.join(directory, name)
                    source_paths.append(os.path.relpath(absolute, root_path))
    files = [
        {
            "path": path.replace("\\", "/"),
            "size_bytes": os.path.getsize(os.path.join(root_path, path)),
            "sha256": _sha256_file(os.path.join(root_path, path)),
        }
        for path in sorted(set(source_paths))
    ]
    source_manifest_sha256 = _canonical_sha256(files)
    state = {
        "schema_version": 1,
        "git_revision": revision_result.stdout.decode().strip() or None,
        "git_branch": branch_result.stdout.decode().strip() or None,
        "git_dirty": bool(status_result.stdout.strip()),
        "git_status_porcelain": status_result.stdout.decode(
            errors="replace"
        ).splitlines(),
        "tracked_source_patch_file": os.path.basename(patch_path),
        "tracked_source_patch_sha256": _sha256_bytes(patch),
        "source_manifest_sha256": source_manifest_sha256,
        "source_files": files,
        "git_errors": [
            result.stderr.decode(errors="replace").strip()
            for result in (
                revision_result,
                branch_result,
                status_result,
                patch_result,
            )
            if result.returncode != 0
        ],
    }
    path = os.path.join(output_dir, "source_state.json")
    with open(path, "w", encoding="utf-8") as source_file:
        json.dump(state, source_file, indent=2, allow_nan=False)
    return {
        **state,
        "file": os.path.basename(path),
        "file_sha256": _sha256_file(path),
    }


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
    provenance_path = os.path.join(data_dir, "dataset_provenance.json")
    provenance = (
        {
            "file": os.path.basename(provenance_path),
            "size_bytes": os.path.getsize(provenance_path),
            "sha256": _sha256_file(provenance_path),
        }
        if os.path.isfile(provenance_path)
        else None
    )
    topology_sha256 = None
    if provenance is not None:
        with open(provenance_path, encoding="utf-8") as source:
            provenance_payload = json.load(source)
        if "topology" in provenance_payload:
            topology_sha256 = _canonical_sha256(
                provenance_payload["topology"]
            )
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
    manifest = {
        "schema_version": 2,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_dir": os.path.abspath(data_dir),
        "dataset_provenance": provenance,
        "topology_sha256": topology_sha256,
        "selected_speeds_kmh": selected_speeds,
        "split_seed": split_seed,
        "split_fractions": split_fractions,
        "split_sizes": {
            split_name: len(split_records)
            for split_name, split_records in records.items()
        },
        "splits": records,
    }
    manifest["dataset_content_sha256"] = _canonical_sha256(
        {
            "dataset_provenance": provenance,
            "topology_sha256": topology_sha256,
            "selected_speeds_kmh": selected_speeds,
            "split_seed": split_seed,
            "split_fractions": split_fractions,
            "splits": records,
        }
    )
    return manifest


def select_value_scale_calibration_pairs(
    train_pairs: tuple[TracePair, ...],
    *,
    routes_per_speed: int,
) -> tuple[TracePair, ...]:
    """Freeze a deterministic, speed-stratified TRAIN calibration subset."""
    if routes_per_speed <= 0:
        raise ValueError("routes_per_speed must be positive.")
    grouped: dict[int, list[TracePair]] = defaultdict(list)
    for pair in train_pairs:
        grouped[int(pair.speed_kmh)].append(pair)
    selected: list[TracePair] = []
    for speed in sorted(grouped):
        selected.extend(
            sorted(
                grouped[speed],
                key=lambda item: (str(item.trajectory_id), item.rsrp_file),
            )[:routes_per_speed]
        )
    return tuple(selected)


def fit_fixed_value_target_scale(
    env: HandoverSMDPEnv,
    *,
    trace_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Fit RMS discounted return-to-go under Always-HO-C1 on TRAIN."""
    returns: list[float] = []
    routes_used: list[str] = []
    routes_without_decision: list[str] = []
    env.transition_builder.reward_model.begin_rollout()
    for trace_index, record in enumerate(trace_records):
        trace_id = str(record["trace_id"])
        try:
            _, _ = env.reset(options={"trace_index": trace_index})
        except RuntimeError:
            routes_without_decision.append(trace_id)
            continue
        rewards: list[float] = []
        discounts: list[float] = []
        terminated = truncated = False
        while not (terminated or truncated):
            _, reward, terminated, truncated, info = env.step(
                int(HandoverAction.HO_C1)
            )
            rewards.append(float(reward))
            discounts.append(float(info["discount"]))
        return_to_go = 0.0
        episode_returns: list[float] = []
        for reward, discount in zip(reversed(rewards), reversed(discounts)):
            return_to_go = reward + discount * return_to_go
            episode_returns.append(return_to_go)
        returns.extend(reversed(episode_returns))
        routes_used.append(trace_id)
    if not returns:
        raise RuntimeError("Value-scale calibration produced no macro returns.")
    rms = math.sqrt(sum(value * value for value in returns) / len(returns))
    scale = max(1.0, rms)
    return {
        "schema_name": "ppo_value_scale_calibration",
        "schema_version": 1,
        "split": "train",
        "policy": "always_ho_c1",
        "estimator": "max(1,sqrt(mean(discounted_return_to_go**2)))",
        "value_target_scale": scale,
        "return_count": len(returns),
        "return_summary": {
            "mean": sum(returns) / len(returns),
            "rms": rms,
            "min": min(returns),
            "max": max(returns),
        },
        "routes_used": routes_used,
        "routes_without_completed_a3": routes_without_decision,
        "trace_records": trace_records,
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
        "dataset_content_sha256": manifest["dataset_content_sha256"],
    }


def build_3gpp_validation_reference(
    protocol_config: Any,
    data_dir: str,
    trace_pairs: tuple[TracePair, ...],
    *,
    gamma_per_second: float,
    candidate_count: int = 10,
    candidate_order: str = "legacy",
    observation_schema: str = "legacy_v1",
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
        candidate_count=candidate_count,
        candidate_order=candidate_order,
        observation_schema=observation_schema,
    )
    rows = benchmark_policy(
        reference_env,
        BenchmarkPolicy(
            "3gpp_always_ho",
            lambda observation: HandoverAction.HO_C1,
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
        "mean_relative_rate_percent_by_speed": {
            str(row["speed_kmh"]): row["mean_relative_rate_percent"]
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


def write_experiment_config(
    output_dir: str,
    *,
    args: Namespace,
    train_env: HandoverSMDPEnv,
    agent: SMDPPPO,
    ppo_config: SMDPPPOConfig,
    protocol_config: Any,
    manifest_paths: dict[str, Any],
    manifest: dict[str, Any],
    source_state: dict[str, Any],
    selected_speeds: list[int],
    value_scale_calibration: dict[str, Any] | None,
) -> dict[str, Any]:
    """Write the immutable contract that identifies one ablation arm."""
    builder = train_env.transition_builder.observation_builder
    reward_config = train_env.transition_builder.reward_model.config
    observation_config = asdict(builder.config)
    dt_ms = float(protocol_config.delta_t_ms)
    experiment_id = args.experiment_id or os.path.basename(
        os.path.normpath(output_dir)
    )
    config = {
        "schema_name": "handover_ppo_experiment_config",
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_id": experiment_id,
        "parent_id": args.parent_id,
        "changed_factor": args.changed_factor,
        "candidate": {
            "K": builder.candidate_count,
            "candidate_pool": builder.CANDIDATE_POLICY,
            "candidate_order": builder.candidate_order_spec,
            "candidate_order_key": builder.candidate_order,
            "mask_mode": "structural_slot_availability_only",
        },
        "action_schema": [
            action.name
            for action in list(HandoverAction)[
                : builder.candidate_count + 1
            ]
        ],
        "observation_schema": {
            "name": train_env.objective_spec["version"],
            "key": builder.observation_schema,
            "size": int(builder.size),
            "features": list(builder.feature_names),
        },
        "history_windows": {
            "trend_ticks": observation_config["trend_window_ticks"],
            "trend_ms": observation_config["trend_window_ticks"] * dt_ms,
            "short_lookback_ticks": observation_config[
                "short_window_ticks"
            ],
            "short_lookback_ms": observation_config[
                "short_window_ticks"
            ] * dt_ms,
            "short_inclusive_samples": observation_config[
                "short_window_ticks"
            ] + 1,
            "long_lookback_ticks": observation_config[
                "long_window_ticks"
            ],
            "long_lookback_ms": observation_config[
                "long_window_ticks"
            ] * dt_ms,
            "long_inclusive_samples": observation_config[
                "long_window_ticks"
            ] + 1,
            "history_ticks": observation_config["history_window_ticks"],
            "history_ms": observation_config["history_window_ticks"] * dt_ms,
        },
        "normalization": observation_config,
        "reward_schema": train_env.objective_spec["reward"],
        "reward_weights": asdict(reward_config),
        "reward_normalization": {
            "mode": (
                "fixed_rate_scale_mbps"
                if train_env.objective_spec["reward"]
                == LEGACY_REWARD_SCHEMA
                else "privileged_full_trace_best_rate_integral"
            ),
            "actor_critic_input": False,
            "trajectory_utility_sum_at_oracle": (
                None
                if train_env.objective_spec["reward"]
                == LEGACY_REWARD_SCHEMA
                else 60.0
            ),
        },
        "dual_update_contract": {
            "enabled": bool(getattr(reward_config, "dual_enabled", False)),
            "boundary": "after_ppo_update_before_next_rollout",
            "source": "training_completed_episodes_only",
            "frozen_within_rollout": True,
            "grouping": "speed_kmh",
        },
        "discount_mode": train_env.objective_spec["discount_mode"],
        "stay_mode": train_env.objective_spec["stay_mode"],
        "stay_redecision_ms": train_env.objective_spec[
            "stay_redecision_ms"
        ],
        "model": {
            "architecture": ppo_config.network_architecture,
            "hidden_sizes": list(ppo_config.hidden_sizes),
            "activation": ppo_config.activation,
            "observation_size": int(train_env.observation_space.shape[0]),
            "action_size": int(train_env.action_space.n),
            "parameter_counts": {
                "total": sum(
                    parameter.numel()
                    for parameter in agent.model.parameters()
                ),
                "actor": sum(
                    parameter.numel()
                    for name, parameter in agent.model.named_parameters()
                    if name.startswith((
                        "encoder.", "actor_encoder.", "policy_head.",
                        "stay_encoder.", "candidate_encoder.",
                        "stay_head.", "candidate_head.",
                    ))
                ),
                "critic": sum(
                    parameter.numel()
                    for name, parameter in agent.model.named_parameters()
                    if name.startswith(("encoder.", "critic_encoder.", "value_head."))
                ),
            },
            "value_normalization": ppo_config.value_normalization,
            "value_target_scale": ppo_config.value_target_scale,
        },
        "value_scale_calibration": value_scale_calibration,
        "ppo": asdict(ppo_config),
        "seeds": {
            "training": args.seed,
            "split": args.split_seed,
        },
        "budget": {
            "total_macro_steps": args.total_macro_steps,
            "rollout_macro_steps": args.rollout_steps,
            "validation_interval_macro_steps": args.evaluation_interval,
        },
        "evaluation": {
            "selected_speeds_kmh": selected_speeds,
            "qos_rate_ratio": args.qos_rate_ratio,
            "qos_rate_ratio_per_speed": args.qos_rate_ratio_per_speed,
            "qos_relative_rate_ratio": args.qos_relative_rate_ratio,
            "qos_relative_rate_ratio_per_speed": (
                args.qos_relative_rate_ratio_per_speed
            ),
            "final_test_enabled": not args.skip_final_test,
            "test_used_for_selection": False,
        },
        "protocol_config": asdict(protocol_config),
        "dataset": {
            "manifest_file": os.path.basename(manifest_paths["json"]),
            "manifest_sha256": manifest_paths["sha256"],
            "content_sha256": manifest_paths["dataset_content_sha256"],
            "topology_sha256": manifest.get("topology_sha256"),
        },
        "source": {
            "git_revision": source_state["git_revision"],
            "git_dirty": source_state["git_dirty"],
            "source_manifest_sha256": source_state[
                "source_manifest_sha256"
            ],
            "tracked_source_patch_sha256": source_state[
                "tracked_source_patch_sha256"
            ],
            "source_state_file": source_state["file"],
        },
        "checkpoint_contract": {
            "save_boundary": "ppo_update_boundary",
            "exact_resume": False,
            "resume_mode": "optimizer_and_rng_with_episode_restart",
        },
    }
    path = os.path.join(output_dir, "experiment_config.json")
    with open(path, "w", encoding="utf-8") as config_file:
        json.dump(config, config_file, indent=2, allow_nan=False)
    return {
        "config": config,
        "path": path,
        "sha256": _sha256_file(path),
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
                f"nent {event['normalized_entropy']:.3f} | "
                f"kl {event['approximate_kl']:.5f} | "
                f"clip {event['clip_fraction']:.3f} | "
                f"ev {event['explained_variance']:+.3f}",
                flush=True,
            )
            dual_update = event.get("dual_update")
            if dual_update is not None:
                lambdas = ", ".join(
                    f"{speed}:{value:.3f}"
                    for speed, value in dual_update[
                        "lambda_after_by_speed"
                    ].items()
                ) or "none"
                shortfalls = ", ".join(
                    f"{speed}:{value:+.3f}"
                    for speed, value in dual_update[
                        "shortfall_by_speed"
                    ].items()
                ) or "none"
                counts = ", ".join(
                    f"{speed}:{value}"
                    for speed, value in dual_update[
                        "sample_count_by_speed"
                    ].items()
                ) or "none"
                updated = ",".join(dual_update["updated_speeds"]) or "none"
                print(
                    f"[dual] lambda {lambdas} | "
                    f"shortfall {shortfalls} | n {counts} | "
                    f"updated {updated}",
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
                    f" | QoS Mbps "
                    f"{100 * event['aggregate_rate_ratio_to_3gpp']:.1f}%"
                    f" / RR "
                    f"{100 * event['aggregate_relative_rate_ratio_to_3gpp']:.1f}%"
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
                f"STAY {100 * event['stay_fraction']:.1f}%"
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
    if args.value_scale_calibration_routes_per_speed <= 0:
        raise ValueError(
            "--value-scale-calibration-routes-per-speed must be positive."
        )
    if args.value_normalization == "none" and args.value_target_scale is not None:
        raise ValueError(
            "--value-target-scale requires --value-normalization fixed_scale_v1."
        )
    if (
        args.network_architecture == "candidate_scorer_v1"
        and args.observation_schema != "qos_full_v1"
    ):
        raise ValueError(
            "candidate_scorer_v1 requires --observation-schema qos_full_v1."
        )
    initial_value_scale = (
        float(args.value_target_scale)
        if args.value_target_scale is not None
        else 1.0
    )
    ppo_config = SMDPPPOConfig(
        hidden_sizes=tuple(args.hidden_sizes),
        network_architecture=args.network_architecture,
        value_normalization=args.value_normalization,
        value_target_scale=initial_value_scale,
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
        learning_diagnostics=args.learning_diagnostics,
        seed=args.seed,
        device=args.device,
    )
    ppo_config.validate()

    reward_config = (
        None
        if args.reward_schema == LEGACY_REWARD_SCHEMA
        else trajectory_reward_config(
            args.reward_schema,
            dual_target_ratio=args.dual_target_ratio,
            dual_learning_rate=args.dual_learning_rate,
            dual_max_coefficient=args.dual_max_coefficient,
            dual_min_episodes_per_speed=(
                args.dual_min_episodes_per_speed
            ),
        )
    )

    value_scale_calibration: dict[str, Any] | None = None
    if args.value_normalization == "fixed_scale_v1":
        if args.value_target_scale is None:
            calibration_pairs = select_value_scale_calibration_pairs(
                tuple(splits.train),
                routes_per_speed=(
                    args.value_scale_calibration_routes_per_speed
                ),
            )
            trace_records = [
                _manifest_record(data_dir, "train", pair)
                for pair in calibration_pairs
            ]
            calibration_env = HandoverSMDPEnv(
                protocol_config,
                data_dir=data_dir,
                trace_pairs=calibration_pairs,
                gamma_per_second=ppo_config.gamma_per_second,
                candidate_count=args.candidate_count,
                candidate_order=args.candidate_order,
                observation_schema=args.observation_schema,
                reward_schema=args.reward_schema,
                reward_config=reward_config,
                stay_mode=args.stay_mode,
                stay_redecision_ms=args.stay_redecision_ms,
            )
            value_scale_calibration = fit_fixed_value_target_scale(
                calibration_env,
                trace_records=trace_records,
            )
            value_scale_calibration["trace_records_sha256"] = (
                _canonical_sha256(trace_records)
            )
            ppo_config = replace(
                ppo_config,
                value_target_scale=float(
                    value_scale_calibration["value_target_scale"]
                ),
            )
        else:
            value_scale_calibration = {
                "schema_name": "ppo_value_scale_calibration",
                "schema_version": 1,
                "split": "train",
                "policy": None,
                "estimator": "explicit_cli_value",
                "value_target_scale": initial_value_scale,
                "trace_records": [],
                "trace_records_sha256": _canonical_sha256([]),
            }
        ppo_config.validate()

    train_env = HandoverSMDPEnv(
        protocol_config,
        data_dir=data_dir,
        trace_pairs=splits.train,
        gamma_per_second=ppo_config.gamma_per_second,
        candidate_count=args.candidate_count,
        candidate_order=args.candidate_order,
        observation_schema=args.observation_schema,
        reward_schema=args.reward_schema,
        reward_config=reward_config,
        stay_mode=args.stay_mode,
        stay_redecision_ms=args.stay_redecision_ms,
    )
    validation_env = HandoverSMDPEnv(
        protocol_config,
        data_dir=data_dir,
        trace_pairs=splits.validation,
        gamma_per_second=ppo_config.gamma_per_second,
        candidate_count=args.candidate_count,
        candidate_order=args.candidate_order,
        observation_schema=args.observation_schema,
        reward_schema=args.reward_schema,
        reward_config=reward_config,
        stay_mode=args.stay_mode,
        stay_redecision_ms=args.stay_redecision_ms,
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
    source_state = capture_source_state(root_path, output_dir)
    if value_scale_calibration is not None:
        calibration_path = os.path.join(
            output_dir, "value_scale_calibration.json"
        )
        with open(calibration_path, "w", encoding="utf-8") as calibration_file:
            json.dump(
                value_scale_calibration,
                calibration_file,
                indent=2,
                allow_nan=False,
            )
        value_scale_calibration = {
            **value_scale_calibration,
            "file": os.path.basename(calibration_path),
            "file_sha256": _sha256_file(calibration_path),
        }
    experiment = write_experiment_config(
        output_dir,
        args=args,
        train_env=train_env,
        agent=agent,
        ppo_config=ppo_config,
        protocol_config=protocol_config,
        manifest_paths=manifest_paths,
        manifest=manifest,
        source_state=source_state,
        selected_speeds=selected_speeds,
        value_scale_calibration=value_scale_calibration,
    )
    print("Building validation 3GPP Always-HO QoS reference...", flush=True)
    validation_reference = build_3gpp_validation_reference(
        protocol_config,
        data_dir,
        splits.validation,
        gamma_per_second=ppo_config.gamma_per_second,
        candidate_count=args.candidate_count,
        candidate_order=args.candidate_order,
        observation_schema=args.observation_schema,
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
        "objective": train_env.objective_spec,
        "dataset_manifest_file": os.path.basename(manifest_paths["json"]),
        "dataset_manifest_sha256": manifest_paths["sha256"],
        "dataset_content_sha256": manifest_paths[
            "dataset_content_sha256"
        ],
        "topology_sha256": manifest["topology_sha256"],
        "experiment_config_file": os.path.basename(experiment["path"]),
        "experiment_config_sha256": experiment["sha256"],
        "experiment_config": experiment["config"],
        "source_state_file": source_state["file"],
        "source_manifest_sha256": source_state["source_manifest_sha256"],
        "source_patch_sha256": source_state[
            "tracked_source_patch_sha256"
        ],
        "validation_3gpp_reference_file": os.path.basename(
            validation_reference_path
        ),
        "selected_speeds_kmh": selected_speeds,
        "split_seed": args.split_seed,
        "split_fractions": split_fractions,
    }
    latest_path = os.path.join(output_dir, "smdp_ppo_latest.pt")
    selected_export_path = os.path.join(
        output_dir, "smdp_ppo_selected.pt"
    )
    metrics_path = os.path.join(output_dir, "training_metrics.jsonl")
    checkpoint_manager = ValidationCheckpointManager(
        output_dir,
        reference_rate_mbps=validation_reference[
            "mean_estimated_rate_mbps"
        ],
        reference_rates_by_speed=validation_reference[
            "mean_estimated_rate_mbps_by_speed"
        ],
        reference_relative_rate_percent=validation_reference[
            "mean_relative_rate_percent"
        ],
        reference_relative_rates_by_speed=validation_reference[
            "mean_relative_rate_percent_by_speed"
        ],
        qos_rate_ratio=args.qos_rate_ratio,
        qos_rate_ratio_per_speed=args.qos_rate_ratio_per_speed,
        qos_relative_rate_ratio=args.qos_relative_rate_ratio,
        qos_relative_rate_ratio_per_speed=(
            args.qos_relative_rate_ratio_per_speed
        ),
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
        "Dataset content SHA-256: "
        f"{manifest_paths['dataset_content_sha256']}",
        flush=True,
    )
    print(f"Experiment config: {experiment['path']}", flush=True)
    print(
        f"Source manifest SHA-256: {source_state['source_manifest_sha256']}",
        flush=True,
    )
    print(
        "Validation 3GPP rate: "
        f"{validation_reference['mean_estimated_rate_mbps']:.3f} Mbps",
        flush=True,
    )
    print(
        "QoS floors (Mbps / relative rate): aggregate="
        f"{100 * args.qos_rate_ratio:.1f}% / "
        f"{100 * args.qos_relative_rate_ratio:.1f}%, per-speed="
        f"{100 * args.qos_rate_ratio_per_speed:.1f}% / "
        f"{100 * args.qos_relative_rate_ratio_per_speed:.1f}%",
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
    training_reward_state = train_env.reward_state
    checkpoint_summary = checkpoint_manager.summary()
    agent.save_checkpoint(
        latest_path,
        extra={
            **audit_metadata,
            "reward_state": training_reward_state,
            "checkpoint_selection_summary": checkpoint_summary,
        },
    )

    selected_path = (
        checkpoint_summary["best_qos"]
        or checkpoint_summary["best_return"]
        or latest_path
    )
    qos_feasible = checkpoint_summary["best_qos"] is not None
    selection_reason = (
        "best_qos"
        if qos_feasible
        else "infeasible_best_return_best_effort"
        if checkpoint_summary["best_return"]
        else "infeasible_latest_best_effort"
    )
    experiment_status = "QOS_FEASIBLE" if qos_feasible else "QOS_INFEASIBLE"
    print(f"Selecting checkpoint: {selected_path}", flush=True)
    selected_agent, selection_metadata = SMDPPPO.from_checkpoint(
        selected_path,
        observation_size=train_env.observation_space.shape[0],
        action_size=train_env.action_space.n,
        device=ppo_config.device,
        load_optimizer=True,
    )
    selected_reward_state = selection_metadata.get("reward_state")
    validation_env.load_reward_state(selected_reward_state)
    print("Running final validation evaluation...", flush=True)
    validation_metrics = selected_agent.evaluate(validation_env)
    test_metrics = None
    if not args.skip_final_test:
        test_env = HandoverSMDPEnv(
            protocol_config,
            data_dir=data_dir,
            trace_pairs=splits.test,
            gamma_per_second=ppo_config.gamma_per_second,
            candidate_count=args.candidate_count,
            candidate_order=args.candidate_order,
            observation_schema=args.observation_schema,
            reward_schema=args.reward_schema,
            reward_config=reward_config,
            stay_mode=args.stay_mode,
            stay_redecision_ms=args.stay_redecision_ms,
        )
        test_env.load_reward_state(selected_reward_state)
        print("Running held-out test evaluation...", flush=True)
        test_metrics = selected_agent.evaluate(test_env)
    else:
        print(
            "Skipping held-out test evaluation (development ablation).",
            flush=True,
        )
    selected_agent.save_checkpoint(
        selected_export_path,
        extra={
            **audit_metadata,
            "selected_from": selected_path,
            "selection_reason": selection_reason,
            "selection_metadata": selection_metadata,
            "reward_state": selected_reward_state,
            "checkpoint_selection_summary": checkpoint_summary,
            "experiment_status": experiment_status,
            "validation": asdict(validation_metrics),
            "test": (
                asdict(test_metrics) if test_metrics is not None else None
            ),
        },
    )

    report = {
        "data_dir": data_dir,
        "experiment_status": experiment_status,
        "experiment_config": experiment,
        "source_state": source_state,
        "selected_speeds_kmh": selected_speeds,
        "dataset_manifest": manifest_paths,
        "ppo_config": asdict(ppo_config),
        "protocol_config": asdict(protocol_config),
        "objective": train_env.objective_spec,
        "training_reward_state": training_reward_state,
        "selected_reward_state": selected_reward_state,
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
        "test": asdict(test_metrics) if test_metrics is not None else None,
        "test_evaluated": test_metrics is not None,
        "checkpoints": {
            "best": checkpoint_summary["best_return"],
            "best_return": checkpoint_summary["best_return"],
            "best_qos": checkpoint_summary["best_qos"],
            "best_rlf": checkpoint_summary["best_rlf"],
            "all": checkpoint_summary["all_dir"],
            "pareto": checkpoint_summary["pareto_dir"],
            "latest": latest_path,
            "selected_from": selected_path,
            "selected_export": selected_export_path,
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
    print(f"Experiment status: {experiment_status}")
    print(f"Validation mean return: {validation_metrics.mean_return:.6f}")
    if test_metrics is not None:
        print(f"Test mean return: {test_metrics.mean_return:.6f}")
    else:
        print("Test mean return: not evaluated")
    print(f"Selected checkpoint export: {selected_export_path}")
    print(f"Dataset manifest: {manifest_paths['json']}")
    print(f"Training report: {report_path}")
    return 0
