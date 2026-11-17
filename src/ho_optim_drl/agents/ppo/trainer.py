"""Collection, optimization, evaluation, and checkpointing for SMDP PPO."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field, replace
import csv
import json
import math
import os
import random
import shutil
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from torch import nn

from ho_optim_drl.envs import HandoverSMDPEnv
from ho_optim_drl.smdp import HandoverAction

from .config import SMDPPPOConfig
from .network import ActorCritic
from .rollout import SMDPRolloutBuffer

ProgressCallback = Callable[[dict[str, Any]], None]
TransitionDiagnosticCallback = Callable[
    [HandoverSMDPEnv, np.ndarray, np.ndarray, int, np.ndarray],
    dict[str, Any],
]
EvaluationActionSelector = Callable[[np.ndarray, np.ndarray], int]


@dataclass(frozen=True)
class PPOUpdateMetrics:
    """Mean diagnostics from one PPO optimization phase."""

    policy_loss: float
    value_loss: float
    value_loss_normalized: float
    raw_value_mse: float
    entropy: float
    normalized_entropy: float
    approximate_kl: float
    clip_fraction: float
    gradient_norm: float
    explained_variance: float
    epochs_completed: int
    learning_diagnostics: dict[str, Any] | None = None


@dataclass(frozen=True)
class EpisodeRecord:
    """One completed training episode."""

    trace_id: str
    episode_return: float
    macro_steps: int
    duration_seconds: float
    speed_kmh: int | None = None
    relative_rate_ratio: float | None = None


@dataclass(frozen=True)
class EvaluationMetrics:
    """Deterministic metrics over a trajectory set."""

    episodes: int
    skipped_no_decision: int
    mean_return: float
    std_return: float
    mean_rate_reward: float
    mean_rlf_penalty: float
    mean_ping_pong_penalty: float
    mean_macro_steps: float
    handover_fraction: float
    stay_fraction: float
    mean_estimated_rate_mbps: float
    mean_max_estimated_rate_mbps: float
    mean_relative_rate_percent: float
    mean_connected_percent: float
    mean_estimated_rate_mbps_by_speed: dict[str, float]
    total_handover_attempts: int
    total_handover_successes: int
    total_handover_failures: int
    handover_failure_percent: float
    total_rlf: int
    total_rlf_recovery_time_seconds: float
    rlf_time_percent: float
    total_ping_pong: int
    ping_pong_percent: float
    mean_relative_rate_percent_by_speed: dict[str, float] = field(
        default_factory=dict
    )
    p10_relative_rate_percent: float = 0.0
    p10_relative_rate_percent_by_speed: dict[str, float] = field(
        default_factory=dict
    )
    action_counts: dict[str, int] = field(default_factory=dict)
    action_fractions: dict[str, float] = field(default_factory=dict)
    timing_stage_action_counts: dict[str, int] = field(default_factory=dict)
    timing_stage_action_fractions: dict[str, float] = field(
        default_factory=dict
    )
    mean_handover_penalty: float = 0.0
    mean_handover_failure_penalty: float = 0.0
    mean_rlf_recovery_penalty: float = 0.0


class ValidationCheckpointManager:
    """Persist validation checkpoints and maintain QoS/Pareto selections."""

    _PARETO_MAXIMIZE = ("mean_estimated_rate_mbps",)
    _PARETO_MINIMIZE = (
        "rlf_time_percent",
        "handover_failure_percent",
        "ping_pong_percent",
        "total_handover_attempts",
    )

    def __init__(
        self,
        output_dir: str | os.PathLike[str],
        *,
        reference_rate_mbps: float,
        reference_rates_by_speed: dict[int | str, float] | None = None,
        reference_relative_rate_percent: float | None = None,
        reference_relative_rates_by_speed: (
            dict[int | str, float] | None
        ) = None,
        qos_rate_ratio: float = 0.98,
        qos_rate_ratio_per_speed: float | None = None,
        qos_relative_rate_ratio: float | None = None,
        qos_relative_rate_ratio_per_speed: float | None = None,
        checkpoint_extra: dict[str, Any] | None = None,
        reference_label: str = "3gpp_always_ho",
    ) -> None:
        if not math.isfinite(reference_rate_mbps) or reference_rate_mbps <= 0:
            raise ValueError("reference_rate_mbps must be positive and finite.")
        if not 0 < qos_rate_ratio <= 1:
            raise ValueError("qos_rate_ratio must be in (0, 1].")
        if qos_rate_ratio_per_speed is None:
            qos_rate_ratio_per_speed = qos_rate_ratio
        if not 0 < qos_rate_ratio_per_speed <= 1:
            raise ValueError("qos_rate_ratio_per_speed must be in (0, 1].")
        if qos_relative_rate_ratio is None:
            qos_relative_rate_ratio = qos_rate_ratio
        if qos_relative_rate_ratio_per_speed is None:
            qos_relative_rate_ratio_per_speed = qos_rate_ratio_per_speed
        if not 0 < qos_relative_rate_ratio <= 1:
            raise ValueError("qos_relative_rate_ratio must be in (0, 1].")
        if not 0 < qos_relative_rate_ratio_per_speed <= 1:
            raise ValueError(
                "qos_relative_rate_ratio_per_speed must be in (0, 1]."
            )

        normalized_speed_rates = {
            str(speed): float(rate)
            for speed, rate in (reference_rates_by_speed or {}).items()
        }
        normalized_relative_rates = {
            str(speed): float(rate)
            for speed, rate in (
                reference_relative_rates_by_speed or {}
            ).items()
        }
        if any(
            not math.isfinite(rate) or rate <= 0
            for rate in normalized_speed_rates.values()
        ):
            raise ValueError(
                "Every reference rate by speed must be positive and finite."
            )
        if reference_relative_rate_percent is not None and (
            not math.isfinite(reference_relative_rate_percent)
            or reference_relative_rate_percent <= 0
        ):
            raise ValueError(
                "reference_relative_rate_percent must be positive and finite."
            )
        if any(
            not math.isfinite(rate) or rate <= 0
            for rate in normalized_relative_rates.values()
        ):
            raise ValueError(
                "Every relative-rate reference by speed must be positive "
                "and finite."
            )

        self.output_dir = os.path.abspath(os.fspath(output_dir))
        self.checkpoint_dir = os.path.join(self.output_dir, "checkpoints")
        self.all_dir = os.path.join(self.checkpoint_dir, "all")
        self.pareto_dir = os.path.join(self.checkpoint_dir, "pareto")
        self.metrics_csv_path = os.path.join(
            self.checkpoint_dir,
            "validation_metrics.csv",
        )
        self.metrics_json_path = os.path.join(
            self.checkpoint_dir,
            "validation_metrics.json",
        )
        self.pareto_json_path = os.path.join(
            self.checkpoint_dir,
            "pareto_front.json",
        )
        self.pareto_csv_path = os.path.join(
            self.checkpoint_dir,
            "pareto_front.csv",
        )
        self.best_return_path = os.path.join(
            self.output_dir,
            "smdp_ppo_best_return.pt",
        )
        self.best_qos_path = os.path.join(
            self.output_dir,
            "smdp_ppo_best_qos.pt",
        )
        self.best_rlf_path = os.path.join(
            self.output_dir,
            "smdp_ppo_best_rlf.pt",
        )
        # Preserve the historical filename as an alias of best-return.
        self.legacy_best_path = os.path.join(
            self.output_dir,
            "smdp_ppo_best.pt",
        )
        self.reference = {
            "label": reference_label,
            "mean_estimated_rate_mbps": float(reference_rate_mbps),
            "mean_estimated_rate_mbps_by_speed": normalized_speed_rates,
            "mean_relative_rate_percent": (
                None
                if reference_relative_rate_percent is None
                else float(reference_relative_rate_percent)
            ),
            "mean_relative_rate_percent_by_speed": normalized_relative_rates,
        }
        self.qos_rate_ratio = float(qos_rate_ratio)
        self.qos_rate_ratio_per_speed = float(qos_rate_ratio_per_speed)
        self.qos_relative_rate_ratio = float(qos_relative_rate_ratio)
        self.qos_relative_rate_ratio_per_speed = float(
            qos_relative_rate_ratio_per_speed
        )
        self.checkpoint_extra = dict(checkpoint_extra or {})
        self.records: list[dict[str, Any]] = []
        os.makedirs(self.all_dir, exist_ok=True)
        os.makedirs(self.pareto_dir, exist_ok=True)

    def record(
        self,
        agent: "SMDPPPO",
        evaluation: EvaluationMetrics,
    ) -> dict[str, Any]:
        """Save one validation model and refresh every selection artifact."""
        metrics = asdict(evaluation)
        step_name = f"step_{agent.total_macro_steps:09d}.pt"
        checkpoint_path = os.path.join(self.all_dir, step_name)
        aggregate_rate_ratio = (
            evaluation.mean_estimated_rate_mbps
            / self.reference["mean_estimated_rate_mbps"]
        )
        speed_ratios = self._speed_ratios(
            evaluation.mean_estimated_rate_mbps_by_speed,
            self.reference["mean_estimated_rate_mbps_by_speed"],
        )
        reference_speed_count = len(
            self.reference["mean_estimated_rate_mbps_by_speed"]
        )
        rate_feasible = (
            aggregate_rate_ratio >= self.qos_rate_ratio
            and all(
                ratio >= self.qos_rate_ratio_per_speed
                for ratio in speed_ratios.values()
            )
            and len(speed_ratios) == reference_speed_count
        )
        if reference_speed_count == 0:
            rate_feasible = aggregate_rate_ratio >= self.qos_rate_ratio

        relative_reference = self.reference["mean_relative_rate_percent"]
        relative_speed_reference = self.reference[
            "mean_relative_rate_percent_by_speed"
        ]
        aggregate_relative_ratio = (
            evaluation.mean_relative_rate_percent / relative_reference
            if relative_reference is not None
            else None
        )
        relative_speed_ratios = self._speed_ratios(
            evaluation.mean_relative_rate_percent_by_speed,
            relative_speed_reference,
        )
        relative_feasible = True
        if aggregate_relative_ratio is not None:
            relative_feasible = (
                aggregate_relative_ratio >= self.qos_relative_rate_ratio
                and all(
                    ratio >= self.qos_relative_rate_ratio_per_speed
                    for ratio in relative_speed_ratios.values()
                )
                and len(relative_speed_ratios) == len(relative_speed_reference)
            )
        qos_feasible = rate_feasible and relative_feasible

        record: dict[str, Any] = {
            "macro_steps": agent.total_macro_steps,
            "update": agent.update_count,
            "checkpoint_path": checkpoint_path,
            "qos_feasible": qos_feasible,
            "aggregate_rate_ratio_to_3gpp": aggregate_rate_ratio,
            "rate_ratio_to_3gpp_by_speed": speed_ratios,
            "aggregate_relative_rate_ratio_to_3gpp": aggregate_relative_ratio,
            "relative_rate_ratio_to_3gpp_by_speed": relative_speed_ratios,
            "validation": metrics,
            "is_best_return": False,
            "is_best_qos": False,
            "is_best_rlf": False,
            "is_pareto": False,
        }
        agent.save_checkpoint(
            checkpoint_path,
            extra={
                **self.checkpoint_extra,
                "validation": metrics,
                "validation_reference": self.reference,
                "checkpoint_selection": {
                    "qos_feasible": qos_feasible,
                    "qos_rate_ratio": self.qos_rate_ratio,
                    "qos_rate_ratio_per_speed": (
                        self.qos_rate_ratio_per_speed
                    ),
                    "qos_relative_rate_ratio": self.qos_relative_rate_ratio,
                    "qos_relative_rate_ratio_per_speed": (
                        self.qos_relative_rate_ratio_per_speed
                    ),
                    "aggregate_rate_ratio_to_3gpp": aggregate_rate_ratio,
                    "rate_ratio_to_3gpp_by_speed": speed_ratios,
                    "aggregate_relative_rate_ratio_to_3gpp": (
                        aggregate_relative_ratio
                    ),
                    "relative_rate_ratio_to_3gpp_by_speed": (
                        relative_speed_ratios
                    ),
                },
            },
        )
        self.records.append(record)
        self._refresh_selections()
        self._write_artifacts()
        return {
            "checkpoint_path": checkpoint_path,
            "qos_feasible": qos_feasible,
            "aggregate_rate_ratio_to_3gpp": aggregate_rate_ratio,
            "rate_ratio_to_3gpp_by_speed": speed_ratios,
            "aggregate_relative_rate_ratio_to_3gpp": aggregate_relative_ratio,
            "relative_rate_ratio_to_3gpp_by_speed": relative_speed_ratios,
            "is_best_return": record["is_best_return"],
            "is_best_qos": record["is_best_qos"],
            "is_best_rlf": record["is_best_rlf"],
            "is_pareto": record["is_pareto"],
        }

    def summary(self) -> dict[str, Any]:
        """Return paths and selections for the final training report."""
        return {
            "selection_policy": {
                "qos_rate_ratio": self.qos_rate_ratio,
                "qos_rate_ratio_per_speed": self.qos_rate_ratio_per_speed,
                "qos_relative_rate_ratio": self.qos_relative_rate_ratio,
                "qos_relative_rate_ratio_per_speed": (
                    self.qos_relative_rate_ratio_per_speed
                ),
                "best_qos_priority": [
                    "mean_relative_rate_percent",
                    "handover_failure_percent",
                    "ping_pong_percent",
                    "rlf_time_percent",
                    "total_handover_attempts",
                ],
                "best_qos_selection": "qos_first_v2",
                "pareto_maximize": list(self._PARETO_MAXIMIZE),
                "pareto_minimize": list(self._PARETO_MINIMIZE),
            },
            "validation_reference": self.reference,
            "validation_checkpoints": len(self.records),
            "qos_feasible_checkpoints": sum(
                bool(record["qos_feasible"]) for record in self.records
            ),
            "pareto_checkpoints": sum(
                bool(record["is_pareto"]) for record in self.records
            ),
            "all_dir": self.all_dir,
            "pareto_dir": self.pareto_dir,
            "best_return": (
                self.best_return_path
                if os.path.exists(self.best_return_path)
                else None
            ),
            "best_qos": (
                self.best_qos_path
                if os.path.exists(self.best_qos_path)
                else None
            ),
            "best_rlf": (
                self.best_rlf_path
                if os.path.exists(self.best_rlf_path)
                else None
            ),
            "metrics_csv": self.metrics_csv_path,
            "metrics_json": self.metrics_json_path,
            "pareto_json": self.pareto_json_path,
            "pareto_csv": self.pareto_csv_path,
        }

    @staticmethod
    def _speed_ratios(
        values_by_speed: dict[str, float],
        references_by_speed: dict[str, float],
    ) -> dict[str, float]:
        ratios: dict[str, float] = {}
        for speed, reference_value in references_by_speed.items():
            if speed in values_by_speed:
                ratios[speed] = (
                    float(values_by_speed[speed]) / reference_value
                )
        return ratios

    def _refresh_selections(self) -> None:
        for record in self.records:
            record["is_best_return"] = False
            record["is_best_qos"] = False
            record["is_best_rlf"] = False
            record["is_pareto"] = False

        best_return = max(
            self.records,
            key=lambda record: record["validation"]["mean_return"],
        )
        best_return["is_best_return"] = True
        best_rlf = min(self.records, key=self._reliability_key)
        best_rlf["is_best_rlf"] = True
        feasible = [
            record for record in self.records if record["qos_feasible"]
        ]
        best_qos = min(feasible, key=self._qos_key) if feasible else None
        if best_qos is not None:
            best_qos["is_best_qos"] = True

        for candidate in self.records:
            candidate["is_pareto"] = not any(
                other is not candidate and self._dominates(other, candidate)
                for other in self.records
            )

        self._copy_checkpoint(
            best_return["checkpoint_path"],
            self.best_return_path,
        )
        self._copy_checkpoint(
            best_return["checkpoint_path"],
            self.legacy_best_path,
        )
        self._copy_checkpoint(
            best_rlf["checkpoint_path"],
            self.best_rlf_path,
        )
        if best_qos is not None:
            self._copy_checkpoint(
                best_qos["checkpoint_path"],
                self.best_qos_path,
            )
        self._sync_pareto_directory()

    @classmethod
    def _dominates(
        cls,
        left: dict[str, Any],
        right: dict[str, Any],
    ) -> bool:
        left_metrics = left["validation"]
        right_metrics = right["validation"]
        no_worse = all(
            left_metrics[field] >= right_metrics[field]
            for field in cls._PARETO_MAXIMIZE
        ) and all(
            left_metrics[field] <= right_metrics[field]
            for field in cls._PARETO_MINIMIZE
        )
        strictly_better = any(
            left_metrics[field] > right_metrics[field]
            for field in cls._PARETO_MAXIMIZE
        ) or any(
            left_metrics[field] < right_metrics[field]
            for field in cls._PARETO_MINIMIZE
        )
        return no_worse and strictly_better

    @staticmethod
    def _qos_key(record: dict[str, Any]) -> tuple[float, ...]:
        """Rank feasible checkpoints by QoS, then mobility reliability."""
        metrics = record["validation"]
        return (
            -metrics["mean_relative_rate_percent"],
            metrics["handover_failure_percent"],
            metrics["ping_pong_percent"],
            metrics["rlf_time_percent"],
            float(metrics["total_handover_attempts"]),
        )

    @staticmethod
    def _reliability_key(record: dict[str, Any]) -> tuple[float, ...]:
        metrics = record["validation"]
        return (
            metrics["rlf_time_percent"],
            metrics["handover_failure_percent"],
            metrics["ping_pong_percent"],
            float(metrics["total_handover_attempts"]),
            -metrics["mean_estimated_rate_mbps"],
        )

    def _sync_pareto_directory(self) -> None:
        desired = {
            os.path.basename(record["checkpoint_path"]): record[
                "checkpoint_path"
            ]
            for record in self.records
            if record["is_pareto"]
        }
        for name in os.listdir(self.pareto_dir):
            path = os.path.join(self.pareto_dir, name)
            if name.startswith("step_") and name.endswith(".pt"):
                if name not in desired:
                    os.remove(path)
        for name, source in desired.items():
            destination = os.path.join(self.pareto_dir, name)
            if not os.path.exists(destination):
                try:
                    os.link(source, destination)
                except OSError:
                    shutil.copy2(source, destination)

    @staticmethod
    def _copy_checkpoint(source: str, destination: str) -> None:
        temporary = f"{destination}.tmp"
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)

    def _write_artifacts(self) -> None:
        self._write_json(self.metrics_json_path, self.records)
        self._write_csv(self.metrics_csv_path, self.records)
        pareto_records = [
            record for record in self.records if record["is_pareto"]
        ]
        self._write_json(self.pareto_json_path, pareto_records)
        self._write_csv(self.pareto_csv_path, pareto_records)

    @staticmethod
    def _write_json(path: str, records: list[dict[str, Any]]) -> None:
        with open(path, "w", encoding="utf-8") as output_file:
            json.dump(records, output_file, indent=2, allow_nan=False)

    @staticmethod
    def _write_csv(path: str, records: list[dict[str, Any]]) -> None:
        if not records:
            return
        metric_fields = list(records[0]["validation"])
        fieldnames = [
            "macro_steps",
            "update",
            "checkpoint_path",
            "qos_feasible",
            "aggregate_rate_ratio_to_3gpp",
            "rate_ratio_to_3gpp_by_speed",
            "aggregate_relative_rate_ratio_to_3gpp",
            "relative_rate_ratio_to_3gpp_by_speed",
            "is_best_return",
            "is_best_qos",
            "is_best_rlf",
            "is_pareto",
            *metric_fields,
        ]
        with open(path, "w", encoding="utf-8", newline="") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=fieldnames)
            writer.writeheader()
            for record in records:
                row = {
                    key: value
                    for key, value in record.items()
                    if key != "validation"
                }
                row["rate_ratio_to_3gpp_by_speed"] = json.dumps(
                    row["rate_ratio_to_3gpp_by_speed"],
                    sort_keys=True,
                )
                row["relative_rate_ratio_to_3gpp_by_speed"] = json.dumps(
                    row["relative_rate_ratio_to_3gpp_by_speed"],
                    sort_keys=True,
                )
                row.update(record["validation"])
                for field_name in metric_fields:
                    if isinstance(row[field_name], dict):
                        row[field_name] = json.dumps(
                            row[field_name],
                            sort_keys=True,
                        )
                writer.writerow(row)


class SMDPPPO:
    """PPO implementation whose GAE and bootstrapping respect macro duration."""

    def __init__(
        self,
        observation_size: int,
        action_size: int,
        config: SMDPPPOConfig | None = None,
    ) -> None:
        # Gymnasium's Discrete.n may be a numpy.int64.  Normalize model
        # dimensions before they reach checkpoint metadata so PyTorch's safe
        # weights-only loader does not need NumPy scalar reconstruction for
        # checkpoints created by current code.
        observation_size = int(observation_size)
        action_size = int(action_size)
        self.config = config or SMDPPPOConfig()
        self.config.validate()
        self.device = self._resolve_device(self.config.device)
        self._seed_everything(self.config.seed)

        self.model = ActorCritic(
            observation_size,
            action_size,
            hidden_sizes=self.config.hidden_sizes,
            activation=self.config.activation,
            architecture=self.config.network_architecture,
            candidate_feature_size=self.config.candidate_feature_size,
            value_target_scale=self.config.value_target_scale,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            eps=1e-5,
        )
        self.total_macro_steps = 0
        self.update_count = 0
        self._collector_env_id: int | None = None
        self._collector_observation: np.ndarray | None = None
        self._last_collection_metrics: dict[str, Any] = {}

    def predict(
        self,
        observation: np.ndarray,
        *,
        deterministic: bool = True,
        action_mask: np.ndarray | None = None,
    ) -> int:
        """Choose an action for evaluation or external use."""
        self.model.eval()
        action, _, _ = self.model.act(
            observation,
            device=self.device,
            deterministic=deterministic,
            action_mask=action_mask,
        )
        return action

    def collect_rollout(
        self,
        env: HandoverSMDPEnv,
        *,
        rollout_steps: int | None = None,
        reset_seed: int | None = None,
        progress_callback: ProgressCallback | None = None,
        progress_interval: int = 50,
        transition_diagnostic_callback: TransitionDiagnosticCallback | None = None,
    ) -> tuple[SMDPRolloutBuffer, list[EpisodeRecord]]:
        """Collect consecutive macro-transitions across episode boundaries."""
        self._validate_environment(env)
        capacity = rollout_steps or self.config.rollout_steps
        if capacity <= 0:
            raise ValueError("rollout_steps must be positive.")
        if progress_interval <= 0:
            raise ValueError("progress_interval must be positive.")

        # Adaptive reward parameters are snapshotted once and remain fixed
        # for every transition collected in this rollout.
        env.transition_builder.reward_model.begin_rollout()

        if (
            self._collector_env_id != id(env)
            or self._collector_observation is None
        ):
            observation, _ = env.reset(
                seed=self.config.seed if reset_seed is None else reset_seed
            )
            self._collector_env_id = id(env)
            self._collector_observation = observation

        buffer = SMDPRolloutBuffer(
            capacity,
            tuple(env.observation_space.shape),
            self.model.action_size,
        )
        completed_episodes: list[EpisodeRecord] = []
        primitive_ticks = 0
        simulated_seconds = 0.0
        action_counts: Counter[str] = Counter()
        timing_stage_action_counts: Counter[str] = Counter()
        speed_counts: Counter[str] = Counter()
        raw_reward_totals: Counter[str] = Counter()
        discounted_reward_totals: Counter[str] = Counter()
        durations_by_action: dict[str, list[float]] = {}
        valid_action_counts: list[int] = []
        action_probabilities: dict[str, list[float]] = {
            action.name: []
            for action in list(HandoverAction)[: self.model.action_size]
        }
        mask_availability: Counter[str] = Counter()
        conditional_ho_entropies: list[float] = []
        self.model.eval()
        self._emit(
            progress_callback,
            {
                "event": "rollout_started",
                "rollout_steps": capacity,
                "total_macro_steps": self.total_macro_steps,
                "update": self.update_count + 1,
            },
        )

        while not buffer.full:
            observation = self._collector_observation
            action_mask = env.action_mask()
            (
                action,
                log_probability,
                value,
                probabilities,
            ) = self.model.act_with_probabilities(
                observation,
                device=self.device,
                deterministic=False,
                action_mask=action_mask,
            )
            transition_metadata = (
                None
                if transition_diagnostic_callback is None
                else transition_diagnostic_callback(
                    env,
                    observation.copy(),
                    action_mask.copy(),
                    action,
                    probabilities.copy(),
                )
            )
            (
                next_observation,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(action)
            action_name = HandoverAction(action).name
            action_counts[action_name] += 1
            decision_info = info.get("decision") or {}
            decision_stage = decision_info.get("stage")
            if decision_stage is not None:
                timing_stage_action_counts[
                    f"{decision_stage}:{action_name}"
                ] += 1
            speed_counts[str(info.get("speed_kmh", "unknown"))] += 1
            primitive_ticks += int(info["duration_ticks"])
            duration_seconds = float(info["duration_seconds"])
            simulated_seconds += duration_seconds
            durations_by_action.setdefault(action_name, []).append(
                duration_seconds
            )
            action_names = list(action_probabilities)
            for index, name in enumerate(action_names):
                action_probabilities[name].append(float(probabilities[index]))
                mask_availability[name] += int(action_mask[index])
            if action != int(HandoverAction.STAY):
                candidate_probabilities = probabilities[1:][action_mask[1:]]
                candidate_mass = float(candidate_probabilities.sum())
                if candidate_mass > 0 and len(candidate_probabilities) > 1:
                    conditional = candidate_probabilities / candidate_mass
                    conditional_ho_entropies.append(
                        float(
                            -np.sum(
                                conditional
                                * np.log(np.maximum(conditional, 1e-12))
                            )
                            / np.log(len(conditional))
                        )
                    )
            valid_action_counts.append(int(np.count_nonzero(action_mask)))
            for component, value_item in info["reward"].items():
                raw_reward_totals[component] += float(value_item)
            for component, value_item in info["discounted_reward"].items():
                discounted_reward_totals[component] += float(value_item)

            next_value = (
                0.0
                if terminated
                else self.model.predict_value(
                    next_observation,
                    device=self.device,
                )
            )
            buffer.add(
                observation=observation,
                action=action,
                reward=reward,
                old_log_probability=log_probability,
                value=value,
                next_value=next_value,
                discount=float(info["discount"]),
                duration_seconds=float(info["duration_seconds"]),
                terminated=terminated,
                truncated=truncated,
                trace_id=str(info["trace_id"]),
                action_mask=action_mask,
                transition_metadata=transition_metadata,
            )
            self.total_macro_steps += 1

            if terminated or truncated:
                episode_metrics = info.get("episode_metrics", {})
                relative_rate_ratio = (
                    float(episode_metrics["achieved_capacity_rel"]) / 100.0
                    if "achieved_capacity_rel" in episode_metrics
                    else None
                )
                completed_episodes.append(
                    EpisodeRecord(
                        trace_id=str(info["trace_id"]),
                        episode_return=float(info["episode_return"]),
                        macro_steps=int(info["episode_macro_steps"]),
                        duration_seconds=float(info["episode_duration_seconds"]),
                        speed_kmh=(
                            int(info["speed_kmh"])
                            if "speed_kmh" in info
                            else None
                        ),
                        relative_rate_ratio=relative_rate_ratio,
                    )
                )
                next_observation, _ = env.reset()

            self._collector_observation = next_observation
            if buffer.full or buffer.size % progress_interval == 0:
                self._emit(
                    progress_callback,
                    {
                        "event": "rollout_progress",
                        "collected_steps": buffer.size,
                        "rollout_steps": capacity,
                        "total_macro_steps": self.total_macro_steps,
                        "completed_episodes": len(completed_episodes),
                        "update": self.update_count + 1,
                    },
                )

        buffer.compute_returns_and_advantages(
            self.config.lambda_per_second
        )
        self._last_collection_metrics = {
            "primitive_ticks": primitive_ticks,
            "simulated_seconds": simulated_seconds,
            "action_counts": dict(sorted(action_counts.items())),
            "timing_stage_action_counts": dict(
                sorted(timing_stage_action_counts.items())
            ),
            "macro_steps_by_speed": dict(sorted(speed_counts.items())),
            "mean_valid_actions": float(np.mean(valid_action_counts)),
            "mask_availability_fraction": {
                name: mask_availability[name] / capacity
                for name in action_probabilities
            },
            "action_probability_summary": {
                name: {
                    "mean": float(np.mean(values)),
                    "p10": float(np.percentile(values, 10)),
                    "p50": float(np.percentile(values, 50)),
                    "p90": float(np.percentile(values, 90)),
                }
                for name, values in action_probabilities.items()
            },
            "conditional_handover_candidate_entropy": (
                float(np.mean(conditional_ho_entropies))
                if conditional_ho_entropies
                else None
            ),
            "raw_reward_totals": dict(sorted(raw_reward_totals.items())),
            "discounted_reward_totals": dict(
                sorted(discounted_reward_totals.items())
            ),
            "duration_seconds_by_action": {
                name: {
                    "p50": float(np.percentile(values, 50)),
                    "p90": float(np.percentile(values, 90)),
                    "p99": float(np.percentile(values, 99)),
                }
                for name, values in sorted(durations_by_action.items())
            },
        }
        return buffer, completed_episodes

    def update(self, buffer: SMDPRolloutBuffer) -> PPOUpdateMetrics:
        """Run clipped PPO optimization on one completed rollout."""
        tensors = buffer.to_tensors(
            self.device,
            normalize_advantage=self.config.normalize_advantage,
        )
        batch_size = min(self.config.batch_size, buffer.size)
        metric_values: dict[str, list[float]] = {
            "policy_loss": [],
            "value_loss": [],
            "raw_value_mse": [],
            "entropy": [],
            "normalized_entropy": [],
            "approximate_kl": [],
            "clip_fraction": [],
            "gradient_norm": [],
        }
        epochs_completed = 0
        stop_early = False
        learning_diagnostics: dict[str, Any] | None = None
        self.model.train()
        pre_update_probabilities: torch.Tensor | None = None
        if self.config.learning_diagnostics:
            with torch.no_grad():
                pre_distribution, _ = self.model.distribution_and_value(
                    tensors.observations,
                    tensors.action_masks,
                )
                pre_update_probabilities = pre_distribution.probs.detach().clone()

        for _ in range(self.config.update_epochs):
            permutation = torch.randperm(buffer.size, device=self.device)
            for start in range(0, buffer.size, batch_size):
                indices = permutation[start : start + batch_size]
                distribution, values = self.model.distribution_and_value(
                    tensors.observations[indices],
                    (
                        None
                        if tensors.action_masks is None
                        else tensors.action_masks[indices]
                    ),
                )
                new_log_probabilities = distribution.log_prob(
                    tensors.actions[indices]
                )
                entropy_values = distribution.entropy()
                entropy = entropy_values.mean()
                if tensors.action_masks is None:
                    valid_counts = torch.full_like(
                        entropy_values,
                        float(self.model.action_size),
                    )
                else:
                    valid_counts = tensors.action_masks[indices].sum(
                        dim=-1
                    ).to(dtype=entropy_values.dtype)
                entropy_denominator = torch.log(valid_counts)
                normalizable = entropy_denominator > 0
                normalized_entropy = (
                    (
                        entropy_values[normalizable]
                        / entropy_denominator[normalizable]
                    ).mean()
                    if torch.any(normalizable)
                    else torch.zeros((), device=self.device)
                )
                log_ratio = (
                    new_log_probabilities
                    - tensors.old_log_probabilities[indices]
                )
                ratio = log_ratio.exp()

                advantages = tensors.advantages[indices]
                unclipped_loss = -advantages * ratio
                clipped_loss = -advantages * torch.clamp(
                    ratio,
                    1 - self.config.clip_range,
                    1 + self.config.clip_range,
                )
                policy_loss = torch.maximum(
                    unclipped_loss,
                    clipped_loss,
                ).mean()
                normalized_values = (
                    values / self.config.value_target_scale
                )
                normalized_returns = (
                    tensors.returns[indices]
                    / self.config.value_target_scale
                )
                value_loss = 0.5 * nn.functional.mse_loss(
                    normalized_values,
                    normalized_returns,
                )
                raw_value_mse = nn.functional.mse_loss(
                    values.detach(),
                    tensors.returns[indices],
                )
                loss = (
                    policy_loss
                    + self.config.value_coefficient * value_loss
                    - self.config.entropy_coefficient * entropy
                )

                if (
                    self.config.learning_diagnostics
                    and learning_diagnostics is None
                ):
                    learning_diagnostics = self._first_minibatch_diagnostics(
                        policy_loss=policy_loss,
                        value_loss=value_loss,
                        entropy=entropy,
                        values=values,
                        returns=tensors.returns[indices],
                        advantages=advantages,
                    )

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                gradient_norm = nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm,
                )
                self.optimizer.step()

                with torch.no_grad():
                    approximate_kl = (
                        (ratio - 1) - log_ratio
                    ).mean()
                    clip_fraction = (
                        (torch.abs(ratio - 1) > self.config.clip_range)
                        .float()
                        .mean()
                    )
                metric_values["policy_loss"].append(float(policy_loss.item()))
                metric_values["value_loss"].append(float(value_loss.item()))
                metric_values["raw_value_mse"].append(
                    float(raw_value_mse.item())
                )
                metric_values["entropy"].append(float(entropy.item()))
                metric_values["normalized_entropy"].append(
                    float(normalized_entropy.item())
                )
                metric_values["approximate_kl"].append(
                    float(approximate_kl.item())
                )
                metric_values["clip_fraction"].append(
                    float(clip_fraction.item())
                )
                metric_values["gradient_norm"].append(
                    float(gradient_norm.item())
                )

                if (
                    self.config.target_kl is not None
                    and approximate_kl.item() > self.config.target_kl
                ):
                    stop_early = True
                    break

            epochs_completed += 1
            if stop_early:
                break

        self.update_count += 1
        explained_variance = self._explained_variance(
            tensors.old_values,
            tensors.returns,
        )
        if learning_diagnostics is not None:
            with torch.no_grad():
                post_distribution, post_values = self.model.distribution_and_value(
                    tensors.observations,
                    tensors.action_masks,
                )
            learning_diagnostics["rollout_distributions"] = {
                "returns": self._tensor_summary(tensors.returns),
                "advantages": self._tensor_summary(tensors.advantages),
                "values_pre": self._tensor_summary(tensors.old_values),
                "values_post": self._tensor_summary(post_values),
            }
            learning_diagnostics["explained_variance_pre"] = (
                explained_variance
            )
            learning_diagnostics["explained_variance_post"] = (
                self._explained_variance(post_values, tensors.returns)
            )
            learning_diagnostics["action_update"] = (
                self._action_update_diagnostics(
                    tensors=tensors,
                    raw_advantages=torch.as_tensor(
                        buffer.advantages[: buffer.size],
                        dtype=torch.float32,
                        device=self.device,
                    ),
                    probabilities_pre=pre_update_probabilities,
                    probabilities_post=post_distribution.probs,
                )
            )
        return PPOUpdateMetrics(
            policy_loss=self._mean(metric_values["policy_loss"]),
            value_loss=self._mean(metric_values["value_loss"]),
            value_loss_normalized=self._mean(metric_values["value_loss"]),
            raw_value_mse=self._mean(metric_values["raw_value_mse"]),
            entropy=self._mean(metric_values["entropy"]),
            normalized_entropy=self._mean(
                metric_values["normalized_entropy"]
            ),
            approximate_kl=self._mean(metric_values["approximate_kl"]),
            clip_fraction=self._mean(metric_values["clip_fraction"]),
            gradient_norm=self._mean(metric_values["gradient_norm"]),
            explained_variance=explained_variance,
            epochs_completed=epochs_completed,
            learning_diagnostics=learning_diagnostics,
        )

    def _action_update_diagnostics(
        self,
        *,
        tensors: Any,
        raw_advantages: torch.Tensor,
        probabilities_pre: torch.Tensor | None,
        probabilities_post: torch.Tensor,
    ) -> dict[str, Any]:
        """Summarize PPO's update signal and probability movement per action."""
        if probabilities_pre is None:
            raise RuntimeError("Pre-update probabilities were not captured.")
        actions = tensors.actions
        masks = tensors.action_masks
        normalized_advantages = tensors.advantages
        delta = probabilities_post - probabilities_pre
        records: dict[str, Any] = {}
        for action_index in range(self.model.action_size):
            name = HandoverAction(action_index).name
            chosen = actions == action_index
            available = (
                torch.ones_like(chosen, dtype=torch.bool)
                if masks is None
                else masks[:, action_index]
            )
            chosen_count = int(chosen.sum().item())
            available_count = int(available.sum().item())
            record: dict[str, Any] = {
                "action_index": action_index,
                "chosen_count": chosen_count,
                "available_count": available_count,
                "chosen_fraction": float(chosen.float().mean().item()),
                "availability_fraction": float(available.float().mean().item()),
            }
            if chosen_count:
                selected_raw = raw_advantages[chosen]
                selected_normalized = normalized_advantages[chosen]
                selected_old_log = tensors.old_log_probabilities[chosen]
                selected_post_log = torch.log(
                    probabilities_post[chosen, action_index].clamp_min(1e-12)
                )
                record.update(
                    {
                        "raw_advantage": self._tensor_summary(selected_raw),
                        "normalized_advantage": self._tensor_summary(
                            selected_normalized
                        ),
                        "positive_advantage_fraction": float(
                            (selected_raw > 0).float().mean().item()
                        ),
                        "mean_probability_ratio_for_chosen": float(
                            torch.exp(selected_post_log - selected_old_log)
                            .mean()
                            .item()
                        ),
                    }
                )
            else:
                record.update(
                    {
                        "raw_advantage": None,
                        "normalized_advantage": None,
                        "positive_advantage_fraction": None,
                        "mean_probability_ratio_for_chosen": None,
                    }
                )
            if available_count:
                record.update(
                    {
                        "mean_probability_pre_when_available": float(
                            probabilities_pre[available, action_index].mean().item()
                        ),
                        "mean_probability_post_when_available": float(
                            probabilities_post[available, action_index].mean().item()
                        ),
                        "mean_probability_delta_when_available": float(
                            delta[available, action_index].mean().item()
                        ),
                        "positive_probability_delta_fraction": float(
                            (delta[available, action_index] > 0)
                            .float()
                            .mean()
                            .item()
                        ),
                    }
                )
            else:
                record.update(
                    {
                        "mean_probability_pre_when_available": None,
                        "mean_probability_post_when_available": None,
                        "mean_probability_delta_when_available": None,
                        "positive_probability_delta_fraction": None,
                    }
                )
            records[name] = record
        return {
            "rollout_size": int(actions.numel()),
            "actions": records,
        }

    def _first_minibatch_diagnostics(
        self,
        *,
        policy_loss: torch.Tensor,
        value_loss: torch.Tensor,
        entropy: torch.Tensor,
        values: torch.Tensor,
        returns: torch.Tensor,
        advantages: torch.Tensor,
    ) -> dict[str, Any]:
        """Measure objective gradients without mutating parameter ``.grad``."""
        named = list(self.model.named_parameters())
        parameters = [parameter for _, parameter in named]
        terms = {
            "actor": policy_loss,
            "critic": self.config.value_coefficient * value_loss,
            "entropy": -self.config.entropy_coefficient * entropy,
        }
        vectors: dict[str, torch.Tensor] = {}
        by_term: dict[str, Any] = {}
        for term_name, term in terms.items():
            gradients = torch.autograd.grad(
                term,
                parameters,
                retain_graph=True,
                allow_unused=True,
            )
            flat_parts = []
            group_squares = {
                "encoder": 0.0,
                "actor_encoder": 0.0,
                "critic_encoder": 0.0,
                "policy_head": 0.0,
                "stay_encoder": 0.0,
                "candidate_encoder": 0.0,
                "stay_head": 0.0,
                "timing_head": 0.0,
                "candidate_head": 0.0,
                "value_head": 0.0,
            }
            for (name, parameter), gradient in zip(named, gradients):
                value = (
                    torch.zeros_like(parameter)
                    if gradient is None
                    else gradient.detach()
                )
                flat_parts.append(value.reshape(-1))
                group = name.split(".", 1)[0]
                if group in group_squares:
                    group_squares[group] += float(
                        torch.sum(value * value).item()
                    )
            vector = torch.cat(flat_parts)
            vectors[term_name] = vector
            by_term[term_name] = {
                "objective_value": float(term.detach().item()),
                "global_norm": float(torch.linalg.vector_norm(vector).item()),
                "group_norms": {
                    group: math.sqrt(square)
                    for group, square in group_squares.items()
                },
            }
        actor = vectors["actor"]
        critic = vectors["critic"]
        denominator = float(
            torch.linalg.vector_norm(actor).item()
            * torch.linalg.vector_norm(critic).item()
        )
        cosine = (
            float(torch.dot(actor, critic).item()) / denominator
            if denominator > 0
            else None
        )
        total = actor + critic + vectors["entropy"]
        total_norm = float(torch.linalg.vector_norm(total).item())
        return {
            "scope": "first_minibatch_before_optimizer_step",
            "weighted_objective_terms": True,
            "terms": by_term,
            "actor_critic_cosine": cosine,
            "total_gradient_norm_before_clip": total_norm,
            "gradient_clipping_factor": (
                min(1.0, self.config.max_grad_norm / total_norm)
                if total_norm > 0
                else 1.0
            ),
            "minibatch_distributions": {
                "returns": self._tensor_summary(returns),
                "advantages": self._tensor_summary(advantages),
                "values": self._tensor_summary(values),
            },
        }

    @staticmethod
    def _tensor_summary(values: torch.Tensor) -> dict[str, float]:
        array = values.detach().float().cpu().numpy()
        return {
            "mean": float(np.mean(array)),
            "std": float(np.std(array)),
            "min": float(np.min(array)),
            "p10": float(np.percentile(array, 10)),
            "median": float(np.median(array)),
            "p90": float(np.percentile(array, 90)),
            "max": float(np.max(array)),
        }

    def evaluate(
        self,
        env: HandoverSMDPEnv,
        *,
        deterministic: bool = True,
        max_episodes: int | None = None,
        action_selector: EvaluationActionSelector | None = None,
    ) -> EvaluationMetrics:
        """Evaluate each trajectory at most once in stable index order."""
        if max_episodes is not None and max_episodes <= 0:
            raise ValueError("max_episodes must be positive or None.")
        self._validate_environment(env)

        # Evaluation mutates ``env``.  It invalidates the cached observation
        # only when this is the same environment used by the active collector.
        # A separate validation environment must not reset training collection
        # or the next rollout will seed the training RNG from the beginning.
        invalidates_active_collector = self._collector_env_id == id(env)

        returns: list[float] = []
        rate_rewards: list[float] = []
        rlf_penalties: list[float] = []
        ping_pong_penalties: list[float] = []
        handover_penalties: list[float] = []
        handover_failure_penalties: list[float] = []
        rlf_recovery_penalties: list[float] = []
        macro_steps: list[int] = []
        estimated_rates: list[float] = []
        max_estimated_rates: list[float] = []
        relative_rates: list[float] = []
        rates_by_speed: dict[str, list[float]] = {}
        relative_rates_by_speed: dict[str, list[float]] = {}
        total_actions = 0
        evaluation_action_counts: Counter[str] = Counter()
        timing_stage_action_counts: Counter[str] = Counter()
        handover_actions = 0
        stay_actions = 0
        total_samples = 0
        total_connected_samples = 0
        total_time_seconds = 0.0
        total_handover_attempts = 0
        total_handover_successes = 0
        total_handover_failures = 0
        total_rlf = 0
        total_rlf_recovery_time_seconds = 0.0
        total_ping_pong = 0
        skipped = 0
        self.model.eval()
        env.transition_builder.reward_model.begin_rollout()

        for trace_index in range(env.num_trajectories):
            if max_episodes is not None and len(returns) >= max_episodes:
                break
            try:
                observation, _ = env.reset(
                    options={"trace_index": trace_index}
                )
            except RuntimeError:
                skipped += 1
                continue

            episode_return = 0.0
            episode_discount = 1.0
            rate_reward = 0.0
            rlf_penalty = 0.0
            ping_pong_penalty = 0.0
            handover_penalty = 0.0
            handover_failure_penalty = 0.0
            rlf_recovery_penalty = 0.0
            episode_steps = 0
            while True:
                action_mask = env.action_mask()
                action = (
                    self.predict(
                        observation,
                        deterministic=deterministic,
                        action_mask=action_mask,
                    )
                    if action_selector is None
                    else int(action_selector(observation, action_mask))
                )
                if not 0 <= action < self.model.action_size:
                    raise ValueError("Evaluation action selector returned an invalid action.")
                if not bool(action_mask[action]):
                    raise ValueError("Evaluation action selector returned a masked action.")
                (
                    observation,
                    reward,
                    terminated,
                    truncated,
                    info,
                ) = env.step(action)
                episode_return += episode_discount * reward
                episode_discount *= float(info["discount"])
                rate_reward += float(info["reward"]["rate"])
                rlf_penalty += float(info["reward"]["rlf"])
                ping_pong_penalty += float(info["reward"]["ping_pong"])
                handover_penalty += float(
                    info["reward"].get("handover", 0.0)
                )
                handover_failure_penalty += float(
                    info["reward"].get("handover_failure", 0.0)
                )
                rlf_recovery_penalty += float(
                    info["reward"].get("rlf_recovery", 0.0)
                )
                episode_steps += 1
                total_actions += 1
                action_name = HandoverAction(action).name
                evaluation_action_counts[action_name] += 1
                decision_info = info.get("decision") or {}
                decision_stage = decision_info.get("stage")
                if decision_stage is not None:
                    timing_stage_action_counts[
                        f"{decision_stage}:{action_name}"
                    ] += 1
                handover_actions += int(action != HandoverAction.STAY)
                stay_actions += int(action == HandoverAction.STAY)
                if terminated or truncated:
                    metrics = env.engine.get_statistics(env.current_trace)
                    estimated_rate = float(metrics["estimated_rate_mbps"])
                    max_estimated_rate = float(
                        metrics["max_estimated_rate_mbps"]
                    )
                    estimated_rates.append(estimated_rate)
                    max_estimated_rates.append(max_estimated_rate)
                    relative_rate = (
                        100 * estimated_rate / max_estimated_rate
                        if max_estimated_rate > 0
                        else 0.0
                    )
                    relative_rates.append(relative_rate)
                    if "speed_kmh" in info:
                        speed = str(info["speed_kmh"])
                        rates_by_speed.setdefault(speed, []).append(
                            estimated_rate
                        )
                        relative_rates_by_speed.setdefault(speed, []).append(
                            relative_rate
                        )
                    total_samples += int(metrics["total_time"])
                    total_connected_samples += int(metrics["connected_time"])
                    total_time_seconds += float(metrics["total_time_s"])
                    total_handover_attempts += int(
                        metrics["num_ho_exe_started"]
                    )
                    total_handover_successes += int(
                        metrics["num_ho_exe_completed"]
                    )
                    total_handover_failures += int(metrics["num_hof"])
                    total_rlf += int(metrics["num_rlf"])
                    total_rlf_recovery_time_seconds += float(
                        metrics["rlf_recovery_time_s"]
                    )
                    total_ping_pong += int(metrics["num_pp"])
                    break

            returns.append(episode_return)
            rate_rewards.append(rate_reward)
            rlf_penalties.append(rlf_penalty)
            ping_pong_penalties.append(ping_pong_penalty)
            handover_penalties.append(handover_penalty)
            handover_failure_penalties.append(handover_failure_penalty)
            rlf_recovery_penalties.append(rlf_recovery_penalty)
            macro_steps.append(episode_steps)

        if invalidates_active_collector:
            self._collector_env_id = None
            self._collector_observation = None
        if not returns:
            raise RuntimeError("Evaluation set contains no actionable trajectory.")
        return EvaluationMetrics(
            episodes=len(returns),
            skipped_no_decision=skipped,
            mean_return=float(np.mean(returns)),
            std_return=float(np.std(returns)),
            mean_rate_reward=float(np.mean(rate_rewards)),
            mean_rlf_penalty=float(np.mean(rlf_penalties)),
            mean_ping_pong_penalty=float(np.mean(ping_pong_penalties)),
            mean_macro_steps=float(np.mean(macro_steps)),
            handover_fraction=(
                handover_actions / total_actions if total_actions else 0.0
            ),
            stay_fraction=(
                stay_actions / total_actions if total_actions else 0.0
            ),
            mean_estimated_rate_mbps=float(np.mean(estimated_rates)),
            mean_max_estimated_rate_mbps=float(
                np.mean(max_estimated_rates)
            ),
            mean_relative_rate_percent=float(np.mean(relative_rates)),
            mean_connected_percent=(
                100 * total_connected_samples / total_samples
                if total_samples
                else 0.0
            ),
            mean_estimated_rate_mbps_by_speed={
                speed: float(np.mean(values))
                for speed, values in sorted(rates_by_speed.items())
            },
            total_handover_attempts=total_handover_attempts,
            total_handover_successes=total_handover_successes,
            total_handover_failures=total_handover_failures,
            handover_failure_percent=(
                100 * total_handover_failures / total_handover_attempts
                if total_handover_attempts
                else 0.0
            ),
            total_rlf=total_rlf,
            total_rlf_recovery_time_seconds=(
                total_rlf_recovery_time_seconds
            ),
            rlf_time_percent=(
                100 * total_rlf_recovery_time_seconds / total_time_seconds
                if total_time_seconds
                else 0.0
            ),
            total_ping_pong=total_ping_pong,
            ping_pong_percent=(
                100 * total_ping_pong / total_handover_successes
                if total_handover_successes
                else 0.0
            ),
            mean_relative_rate_percent_by_speed={
                speed: float(np.mean(values))
                for speed, values in sorted(relative_rates_by_speed.items())
            },
            p10_relative_rate_percent=float(
                np.percentile(relative_rates, 10)
            ),
            p10_relative_rate_percent_by_speed={
                speed: float(np.percentile(values, 10))
                for speed, values in sorted(relative_rates_by_speed.items())
            },
            action_counts={
                action.name: int(evaluation_action_counts[action.name])
                for action in list(HandoverAction)[: self.model.action_size]
            },
            action_fractions={
                action.name: (
                    evaluation_action_counts[action.name] / total_actions
                    if total_actions
                    else 0.0
                )
                for action in list(HandoverAction)[: self.model.action_size]
            },
            timing_stage_action_counts=dict(
                sorted(timing_stage_action_counts.items())
            ),
            timing_stage_action_fractions={
                key: count / total_actions if total_actions else 0.0
                for key, count in sorted(timing_stage_action_counts.items())
            },
            mean_handover_penalty=float(np.mean(handover_penalties)),
            mean_handover_failure_penalty=float(
                np.mean(handover_failure_penalties)
            ),
            mean_rlf_recovery_penalty=float(
                np.mean(rlf_recovery_penalties)
            ),
        )

    def evaluate_stochastic_seeds(
        self,
        env: HandoverSMDPEnv,
        seeds: tuple[int, ...],
    ) -> list[dict[str, Any]]:
        """Evaluate sampled actions without perturbing the training RNG stream."""
        normalized_seeds = tuple(int(seed) for seed in seeds)
        if len(set(normalized_seeds)) != len(normalized_seeds):
            raise ValueError("Stochastic validation seeds must be unique.")
        if not normalized_seeds:
            return []

        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.random.get_rng_state()
        cuda_states = (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        )
        results: list[dict[str, Any]] = []
        try:
            for seed in normalized_seeds:
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
                results.append(
                    {
                        "seed": seed,
                        "metrics": asdict(
                            self.evaluate(env, deterministic=False)
                        ),
                    }
                )
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.random.set_rng_state(torch_state)
            if cuda_states is not None:
                torch.cuda.set_rng_state_all(cuda_states)
        return results

    def train(
        self,
        env: HandoverSMDPEnv,
        *,
        total_macro_steps: int,
        validation_env: HandoverSMDPEnv | None = None,
        evaluation_interval: int = 5_000,
        checkpoint_path: str | os.PathLike[str] | None = None,
        checkpoint_extra: dict[str, Any] | None = None,
        checkpoint_manager: ValidationCheckpointManager | None = None,
        stochastic_validation_seeds: tuple[int, ...] = (),
        progress_callback: ProgressCallback | None = None,
        progress_interval: int = 50,
    ) -> list[dict[str, Any]]:
        """Train until the requested number of macro decisions is collected."""
        if total_macro_steps <= 0:
            raise ValueError("total_macro_steps must be positive.")
        if evaluation_interval <= 0:
            raise ValueError("evaluation_interval must be positive.")
        if progress_interval <= 0:
            raise ValueError("progress_interval must be positive.")

        history: list[dict[str, Any]] = []
        next_evaluation = evaluation_interval
        last_validation_step: int | None = None
        best_validation_return = -math.inf
        self._emit(
            progress_callback,
            {
                "event": "training_started",
                "total_macro_steps": self.total_macro_steps,
                "target_macro_steps": total_macro_steps,
                "rollout_steps": self.config.rollout_steps,
                "device": str(self.device),
            },
        )
        while self.total_macro_steps < total_macro_steps:
            remaining = total_macro_steps - self.total_macro_steps
            rollout_size = min(self.config.rollout_steps, remaining)
            buffer, episodes = self.collect_rollout(
                env,
                rollout_steps=rollout_size,
                progress_callback=progress_callback,
                progress_interval=progress_interval,
            )
            update_metrics = self.update(buffer)
            reward_model = env.transition_builder.reward_model
            dual_update = reward_model.update_dual(episodes)
            trace_ids = [
                str(trace_id)
                for trace_id in buffer.trace_ids[: buffer.size]
                if trace_id is not None
            ]
            record: dict[str, Any] = {
                "macro_steps": self.total_macro_steps,
                "update": self.update_count,
                "completed_episodes": len(episodes),
                "completed_episode_trace_ids": [
                    episode.trace_id for episode in episodes
                ],
                "training_trace_macro_steps": dict(
                    sorted(Counter(trace_ids).items())
                ),
                "training_trace_segments": self._trace_segments(trace_ids),
                "rollout_diagnostics": dict(self._last_collection_metrics),
                "mean_training_episode_return": (
                    float(np.mean([episode.episode_return for episode in episodes]))
                    if episodes
                    else None
                ),
                **asdict(update_metrics),
            }
            if dual_update is not None:
                record["dual_update"] = dual_update
            self._emit(
                progress_callback,
                {
                    "event": "update_complete",
                    **record,
                },
            )

            scheduled_validation_due = (
                self.total_macro_steps >= next_evaluation
            )
            budget_end_validation_due = (
                self.total_macro_steps >= total_macro_steps
                and last_validation_step != self.total_macro_steps
            )
            if validation_env is not None and (
                scheduled_validation_due or budget_end_validation_due
            ):
                validation_trigger = (
                    "scheduled"
                    if scheduled_validation_due
                    else "budget_end"
                )
                reward_state = reward_model.state_dict()
                validation_env.load_reward_state(reward_state)
                if checkpoint_manager is not None:
                    checkpoint_manager.checkpoint_extra["reward_state"] = (
                        reward_state
                    )
                self._emit(
                    progress_callback,
                    {
                        "event": "validation_started",
                        "macro_steps": self.total_macro_steps,
                        "update": self.update_count,
                        "trajectories": validation_env.num_trajectories,
                        "validation_trigger": validation_trigger,
                    },
                )
                evaluation = self.evaluate(validation_env)
                record["validation"] = asdict(evaluation)
                checkpoint_selection = (
                    checkpoint_manager.record(self, evaluation)
                    if checkpoint_manager is not None
                    else None
                )
                if checkpoint_selection is not None:
                    record["checkpoint_selection"] = checkpoint_selection
                stochastic_validation: list[dict[str, Any]] = []
                if stochastic_validation_seeds:
                    self._emit(
                        progress_callback,
                        {
                            "event": "stochastic_validation_started",
                            "macro_steps": self.total_macro_steps,
                            "update": self.update_count,
                            "trajectories": validation_env.num_trajectories,
                            "seeds": list(stochastic_validation_seeds),
                            "validation_trigger": validation_trigger,
                        },
                    )
                    validation_env.load_reward_state(reward_state)
                    stochastic_validation = self.evaluate_stochastic_seeds(
                        validation_env,
                        stochastic_validation_seeds,
                    )
                    record["stochastic_validation"] = stochastic_validation
                    for stochastic_result in stochastic_validation:
                        self._emit(
                            progress_callback,
                            {
                                "event": "stochastic_validation_complete",
                                "macro_steps": self.total_macro_steps,
                                "update": self.update_count,
                                "seed": stochastic_result["seed"],
                                "validation_trigger": validation_trigger,
                                **stochastic_result["metrics"],
                            },
                        )
                while next_evaluation <= self.total_macro_steps:
                    next_evaluation += evaluation_interval
                last_validation_step = self.total_macro_steps
                if (
                    checkpoint_path is not None
                    and evaluation.mean_return > best_validation_return
                ):
                    best_validation_return = evaluation.mean_return
                    self.save_checkpoint(
                        checkpoint_path,
                        extra={
                            **(checkpoint_extra or {}),
                            "best_validation_return": best_validation_return,
                        },
                    )
                    is_best = True
                else:
                    is_best = False
                if checkpoint_selection is not None:
                    is_best = checkpoint_selection["is_best_return"]
                self._emit(
                    progress_callback,
                    {
                        "event": "validation_complete",
                        "macro_steps": self.total_macro_steps,
                        "update": self.update_count,
                        "is_best": is_best,
                        "validation_trigger": validation_trigger,
                        **(checkpoint_selection or {}),
                        **asdict(evaluation),
                    },
                )

            history.append(record)

        if checkpoint_path is not None and validation_env is None:
            self.save_checkpoint(checkpoint_path, extra=checkpoint_extra)
        self._emit(
            progress_callback,
            {
                "event": "training_complete",
                "total_macro_steps": self.total_macro_steps,
                "updates": self.update_count,
            },
        )
        return history

    @staticmethod
    def _trace_segments(trace_ids: list[str]) -> list[dict[str, Any]]:
        """Run-length encode trajectory usage within one rollout update."""
        segments: list[dict[str, Any]] = []
        for trace_id in trace_ids:
            if segments and segments[-1]["trace_id"] == trace_id:
                segments[-1]["macro_steps"] += 1
            else:
                segments.append({"trace_id": trace_id, "macro_steps": 1})
        return segments

    def save_checkpoint(
        self,
        path: str | os.PathLike[str],
        *,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Save model, optimizer, counters, and configuration."""
        path = os.fspath(path)
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        torch.save(
            {
                "format_version": 1,
                "config": asdict(self.config),
                "model_spec": {
                    "observation_size": self.model.observation_size,
                    "action_size": self.model.action_size,
                    "network_architecture": self.model.architecture,
                    "candidate_feature_size": self.model.candidate_feature_size,
                    "value_normalization": self.config.value_normalization,
                    "value_target_scale": self.config.value_target_scale,
                    "action_names": [
                        action.name
                        for action in list(HandoverAction)[
                            : self.model.action_size
                        ]
                    ],
                },
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "total_macro_steps": self.total_macro_steps,
                "update_count": self.update_count,
                "training_state": self._capture_rng_state(),
                "checkpoint_contract": {
                    "save_boundary": "ppo_update_boundary",
                    "resume_capability": (
                        "optimizer_and_rng_with_episode_restart"
                    ),
                    "exact_resume": False,
                    "missing_state": [
                        "environment_protocol_state",
                        "collector_observation_and_trace_cursor",
                        "checkpoint_manager_records_and_schedule",
                    ],
                },
                "extra": extra or {},
            },
            path,
        )

    def load_checkpoint(
        self,
        path: str | os.PathLike[str],
        *,
        load_optimizer: bool = True,
    ) -> dict[str, Any]:
        """Load a compatible checkpoint and return its extra metadata."""
        checkpoint = self._load_checkpoint_payload(path, self.device)
        if checkpoint.get("format_version") != 1:
            raise ValueError("Unsupported PPO checkpoint format.")
        saved_config = dict(checkpoint.get("config", {}))
        saved_architecture = saved_config.get(
            "network_architecture", "shared_mlp_v1"
        )
        saved_scale = float(saved_config.get("value_target_scale", 1.0))
        if saved_architecture != self.config.network_architecture:
            raise ValueError("Cannot load a checkpoint across architectures.")
        if saved_scale != self.config.value_target_scale:
            raise ValueError("Cannot load a checkpoint across value scales.")
        self.model.load_state_dict(checkpoint["model_state"])
        if load_optimizer:
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.total_macro_steps = int(checkpoint["total_macro_steps"])
        self.update_count = int(checkpoint["update_count"])
        if load_optimizer and "training_state" in checkpoint:
            self._restore_rng_state(checkpoint["training_state"])
        self._collector_env_id = None
        self._collector_observation = None
        return dict(checkpoint.get("extra", {}))

    @classmethod
    def from_checkpoint(
        cls,
        path: str | os.PathLike[str],
        *,
        observation_size: int | None = None,
        action_size: int | None = None,
        device: str | None = None,
        load_optimizer: bool = False,
    ) -> tuple["SMDPPPO", dict[str, Any]]:
        """Construct an agent from the configuration stored in a checkpoint."""
        checkpoint = cls._load_checkpoint_payload(path, "cpu")
        if checkpoint.get("format_version") != 1:
            raise ValueError("Unsupported PPO checkpoint format.")
        model_spec = dict(checkpoint.get("model_spec", {}))
        model_state = checkpoint["model_state"]
        saved_observation_size_value = model_spec.get("observation_size")
        if saved_observation_size_value is None:
            first_encoder_weight = next(
                (
                    value
                    for key, value in model_state.items()
                    if key.startswith("encoder.") and key.endswith(".weight")
                ),
                None,
            )
            saved_observation_size_value = (
                first_encoder_weight.shape[1]
                if first_encoder_weight is not None
                else model_state["policy_head.weight"].shape[1]
            )
        saved_action_size_value = model_spec.get("action_size")
        if saved_action_size_value is None:
            saved_action_size_value = model_state["policy_head.weight"].shape[0]
        saved_observation_size = int(saved_observation_size_value)
        saved_action_size = int(saved_action_size_value)
        if (
            observation_size is not None
            and observation_size != saved_observation_size
        ):
            raise ValueError(
                "Requested observation size does not match checkpoint: "
                f"{observation_size} != {saved_observation_size}."
            )
        if action_size is not None and action_size != saved_action_size:
            raise ValueError(
                "Requested action size does not match checkpoint: "
                f"{action_size} != {saved_action_size}."
            )
        config_values = dict(checkpoint["config"])
        config_values["hidden_sizes"] = tuple(config_values["hidden_sizes"])
        # Legacy checkpoints predate D3 and are exactly the shared, unscaled
        # control. Defaults make that contract explicit without key guessing.
        config_values.setdefault("network_architecture", "shared_mlp_v1")
        config_values.setdefault("candidate_feature_size", 15)
        config_values.setdefault("value_normalization", "none")
        config_values.setdefault("value_target_scale", 1.0)
        config = SMDPPPOConfig(**config_values)
        saved_architecture = model_spec.get("network_architecture")
        if (
            saved_architecture is not None
            and saved_architecture != config.network_architecture
        ):
            raise ValueError(
                "Checkpoint model_spec architecture disagrees with config."
            )
        if device is not None:
            config = replace(config, device=device)
        agent = cls(saved_observation_size, saved_action_size, config)
        agent.model.load_state_dict(model_state)
        if load_optimizer:
            agent.optimizer.load_state_dict(checkpoint["optimizer_state"])
        agent.total_macro_steps = int(checkpoint["total_macro_steps"])
        agent.update_count = int(checkpoint["update_count"])
        if load_optimizer and "training_state" in checkpoint:
            agent._restore_rng_state(checkpoint["training_state"])
        return agent, dict(checkpoint.get("extra", {}))

    @staticmethod
    def _capture_rng_state() -> dict[str, Any]:
        """Capture RNG state using only weights-only-safe primitives."""
        numpy_state = np.random.get_state()
        state: dict[str, Any] = {
            "python": random.getstate(),
            "numpy": {
                "bit_generator": str(numpy_state[0]),
                "keys": numpy_state[1].astype(np.uint32).tolist(),
                "position": int(numpy_state[2]),
                "has_gauss": int(numpy_state[3]),
                "cached_gaussian": float(numpy_state[4]),
            },
            "torch_cpu": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            state["torch_cuda"] = torch.cuda.get_rng_state_all()
        return state

    @staticmethod
    def _restore_rng_state(state: dict[str, Any]) -> None:
        """Restore the checkpoint RNG state for update-boundary restart."""
        random.setstate(tuple(state["python"]))
        numpy_state = state["numpy"]
        np.random.set_state(
            (
                str(numpy_state["bit_generator"]),
                np.asarray(numpy_state["keys"], dtype=np.uint32),
                int(numpy_state["position"]),
                int(numpy_state["has_gauss"]),
                float(numpy_state["cached_gaussian"]),
            )
        )
        torch.set_rng_state(state["torch_cpu"])
        if "torch_cuda" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["torch_cuda"])

    @staticmethod
    def _load_checkpoint_payload(
        path: str | os.PathLike[str],
        map_location: str | torch.device,
    ) -> dict[str, Any]:
        """Load tensors plus legacy numeric NumPy scalars in safe mode.

        Older project checkpoints stored ``gymnasium.spaces.Discrete.n``
        directly in ``model_spec``.  Recent Gymnasium versions expose it as a
        ``numpy.int64``.  The allowlist below is deliberately limited to
        NumPy's scalar constructor and numeric dtype classes; arbitrary Python
        globals remain blocked by ``weights_only=True``.
        """
        numeric_dtypes = (
            np.bool_,
            np.int8,
            np.int16,
            np.int32,
            np.int64,
            np.uint8,
            np.uint16,
            np.uint32,
            np.uint64,
            np.float16,
            np.float32,
            np.float64,
        )
        safe_globals = [np._core.multiarray.scalar, np.dtype]
        safe_globals.extend(
            {
                type(np.dtype(dtype))
                for dtype in numeric_dtypes
            }
        )
        with torch.serialization.safe_globals(safe_globals):
            return torch.load(
                os.fspath(path),
                map_location=map_location,
                weights_only=True,
            )

    def _validate_environment(self, env: HandoverSMDPEnv) -> None:
        environment_gamma = env.transition_builder.gamma_per_second
        if not math.isclose(
            environment_gamma,
            self.config.gamma_per_second,
            rel_tol=0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "Environment and PPO gamma_per_second must match: "
                f"{environment_gamma} != {self.config.gamma_per_second}."
            )
        if tuple(env.observation_space.shape) != (
            self.model.observation_size,
        ):
            raise ValueError(
                "Environment observation space does not match the model."
            )
        if env.action_space.n != self.model.action_size:
            raise ValueError("Environment action space does not match the model.")

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        resolved = torch.device(device)
        if resolved.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is not available.")
        return resolved

    @staticmethod
    def _seed_everything(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    @staticmethod
    def _mean(values: list[float]) -> float:
        if not values:
            raise RuntimeError("No PPO minibatch metrics were produced.")
        return float(np.mean(values))

    @staticmethod
    def _emit(
        callback: ProgressCallback | None,
        event: dict[str, Any],
    ) -> None:
        if callback is not None:
            callback(event)

    @staticmethod
    def _explained_variance(
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> float:
        target_variance = torch.var(targets, unbiased=False)
        if target_variance.item() == 0:
            return 0.0
        residual_variance = torch.var(
            targets - predictions,
            unbiased=False,
        )
        return float((1 - residual_variance / target_variance).item())
