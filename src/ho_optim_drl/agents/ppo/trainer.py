"""Collection, optimization, evaluation, and checkpointing for SMDP PPO."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
import os
import random
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from torch import nn

from ho_optim_drl.envs import HandoverSMDPEnv

from .config import SMDPPPOConfig
from .network import ActorCritic
from .rollout import SMDPRolloutBuffer

ProgressCallback = Callable[[dict[str, Any]], None]


@dataclass(frozen=True)
class PPOUpdateMetrics:
    """Mean diagnostics from one PPO optimization phase."""

    policy_loss: float
    value_loss: float
    entropy: float
    approximate_kl: float
    clip_fraction: float
    gradient_norm: float
    explained_variance: float
    epochs_completed: int


@dataclass(frozen=True)
class EpisodeRecord:
    """One completed training episode."""

    trace_id: str
    episode_return: float
    macro_steps: int
    duration_seconds: float


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
    mean_estimated_rate_mbps: float
    total_rlf: int
    total_ping_pong: int


class SMDPPPO:
    """PPO implementation whose GAE and bootstrapping respect macro duration."""

    def __init__(
        self,
        observation_size: int,
        action_size: int,
        config: SMDPPPOConfig | None = None,
    ) -> None:
        self.config = config or SMDPPPOConfig()
        self.config.validate()
        self.device = self._resolve_device(self.config.device)
        self._seed_everything(self.config.seed)

        self.model = ActorCritic(
            observation_size,
            action_size,
            hidden_sizes=self.config.hidden_sizes,
            activation=self.config.activation,
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

    def predict(
        self,
        observation: np.ndarray,
        *,
        deterministic: bool = True,
    ) -> int:
        """Choose an action for evaluation or external use."""
        self.model.eval()
        action, _, _ = self.model.act(
            observation,
            device=self.device,
            deterministic=deterministic,
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
    ) -> tuple[SMDPRolloutBuffer, list[EpisodeRecord]]:
        """Collect consecutive macro-transitions across episode boundaries."""
        self._validate_environment(env)
        capacity = rollout_steps or self.config.rollout_steps
        if capacity <= 0:
            raise ValueError("rollout_steps must be positive.")
        if progress_interval <= 0:
            raise ValueError("progress_interval must be positive.")

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
        )
        completed_episodes: list[EpisodeRecord] = []
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
            action, log_probability, value = self.model.act(
                observation,
                device=self.device,
                deterministic=False,
            )
            (
                next_observation,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(action)

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
            )
            self.total_macro_steps += 1

            if terminated or truncated:
                completed_episodes.append(
                    EpisodeRecord(
                        trace_id=str(info["trace_id"]),
                        episode_return=float(info["episode_return"]),
                        macro_steps=int(info["episode_macro_steps"]),
                        duration_seconds=float(info["episode_duration_seconds"]),
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
            "entropy": [],
            "approximate_kl": [],
            "clip_fraction": [],
            "gradient_norm": [],
        }
        epochs_completed = 0
        stop_early = False
        self.model.train()

        for _ in range(self.config.update_epochs):
            permutation = torch.randperm(buffer.size, device=self.device)
            for start in range(0, buffer.size, batch_size):
                indices = permutation[start : start + batch_size]
                distribution, values = self.model.distribution_and_value(
                    tensors.observations[indices]
                )
                new_log_probabilities = distribution.log_prob(
                    tensors.actions[indices]
                )
                entropy = distribution.entropy().mean()
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
                value_loss = 0.5 * nn.functional.mse_loss(
                    values,
                    tensors.returns[indices],
                )
                loss = (
                    policy_loss
                    + self.config.value_coefficient * value_loss
                    - self.config.entropy_coefficient * entropy
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
                metric_values["entropy"].append(float(entropy.item()))
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
        return PPOUpdateMetrics(
            policy_loss=self._mean(metric_values["policy_loss"]),
            value_loss=self._mean(metric_values["value_loss"]),
            entropy=self._mean(metric_values["entropy"]),
            approximate_kl=self._mean(metric_values["approximate_kl"]),
            clip_fraction=self._mean(metric_values["clip_fraction"]),
            gradient_norm=self._mean(metric_values["gradient_norm"]),
            explained_variance=explained_variance,
            epochs_completed=epochs_completed,
        )

    def evaluate(
        self,
        env: HandoverSMDPEnv,
        *,
        deterministic: bool = True,
        max_episodes: int | None = None,
    ) -> EvaluationMetrics:
        """Evaluate each trajectory at most once in stable index order."""
        if max_episodes is not None and max_episodes <= 0:
            raise ValueError("max_episodes must be positive or None.")

        returns: list[float] = []
        rate_rewards: list[float] = []
        rlf_penalties: list[float] = []
        ping_pong_penalties: list[float] = []
        macro_steps: list[int] = []
        estimated_rates: list[float] = []
        total_actions = 0
        handover_actions = 0
        total_rlf = 0
        total_ping_pong = 0
        skipped = 0
        self.model.eval()

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
            rate_reward = 0.0
            rlf_penalty = 0.0
            ping_pong_penalty = 0.0
            episode_steps = 0
            while True:
                action = self.predict(
                    observation,
                    deterministic=deterministic,
                )
                (
                    observation,
                    reward,
                    terminated,
                    truncated,
                    info,
                ) = env.step(action)
                episode_return += reward
                rate_reward += float(info["reward"]["rate"])
                rlf_penalty += float(info["reward"]["rlf"])
                ping_pong_penalty += float(info["reward"]["ping_pong"])
                episode_steps += 1
                total_actions += 1
                handover_actions += int(action == 1)
                if terminated or truncated:
                    metrics = info["episode_metrics"]
                    estimated_rates.append(
                        float(metrics["estimated_rate_mbps"])
                    )
                    total_rlf += int(metrics["num_rlf"])
                    total_ping_pong += int(metrics["num_pp"])
                    break

            returns.append(episode_return)
            rate_rewards.append(rate_reward)
            rlf_penalties.append(rlf_penalty)
            ping_pong_penalties.append(ping_pong_penalty)
            macro_steps.append(episode_steps)

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
            mean_estimated_rate_mbps=float(np.mean(estimated_rates)),
            total_rlf=total_rlf,
            total_ping_pong=total_ping_pong,
        )

    def train(
        self,
        env: HandoverSMDPEnv,
        *,
        total_macro_steps: int,
        validation_env: HandoverSMDPEnv | None = None,
        evaluation_interval: int = 5_000,
        checkpoint_path: str | os.PathLike[str] | None = None,
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
            record: dict[str, Any] = {
                "macro_steps": self.total_macro_steps,
                "update": self.update_count,
                "completed_episodes": len(episodes),
                "mean_training_episode_return": (
                    float(np.mean([episode.episode_return for episode in episodes]))
                    if episodes
                    else None
                ),
                **asdict(update_metrics),
            }
            self._emit(
                progress_callback,
                {
                    "event": "update_complete",
                    **record,
                },
            )

            if (
                validation_env is not None
                and self.total_macro_steps >= next_evaluation
            ):
                self._emit(
                    progress_callback,
                    {
                        "event": "validation_started",
                        "macro_steps": self.total_macro_steps,
                        "update": self.update_count,
                        "trajectories": validation_env.num_trajectories,
                    },
                )
                evaluation = self.evaluate(validation_env)
                record["validation"] = asdict(evaluation)
                while next_evaluation <= self.total_macro_steps:
                    next_evaluation += evaluation_interval
                if (
                    checkpoint_path is not None
                    and evaluation.mean_return > best_validation_return
                ):
                    best_validation_return = evaluation.mean_return
                    self.save_checkpoint(
                        checkpoint_path,
                        extra={
                            "best_validation_return": best_validation_return,
                        },
                    )
                    is_best = True
                else:
                    is_best = False
                self._emit(
                    progress_callback,
                    {
                        "event": "validation_complete",
                        "macro_steps": self.total_macro_steps,
                        "update": self.update_count,
                        "is_best": is_best,
                        **asdict(evaluation),
                    },
                )

            history.append(record)

        if checkpoint_path is not None and validation_env is None:
            self.save_checkpoint(checkpoint_path)
        self._emit(
            progress_callback,
            {
                "event": "training_complete",
                "total_macro_steps": self.total_macro_steps,
                "updates": self.update_count,
            },
        )
        return history

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
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "total_macro_steps": self.total_macro_steps,
                "update_count": self.update_count,
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
        checkpoint = torch.load(
            os.fspath(path),
            map_location=self.device,
            weights_only=True,
        )
        if checkpoint.get("format_version") != 1:
            raise ValueError("Unsupported PPO checkpoint format.")
        self.model.load_state_dict(checkpoint["model_state"])
        if load_optimizer:
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.total_macro_steps = int(checkpoint["total_macro_steps"])
        self.update_count = int(checkpoint["update_count"])
        self._collector_env_id = None
        self._collector_observation = None
        return dict(checkpoint.get("extra", {}))

    @classmethod
    def from_checkpoint(
        cls,
        path: str | os.PathLike[str],
        *,
        observation_size: int,
        action_size: int,
        device: str | None = None,
        load_optimizer: bool = False,
    ) -> tuple["SMDPPPO", dict[str, Any]]:
        """Construct an agent from the configuration stored in a checkpoint."""
        checkpoint = torch.load(
            os.fspath(path),
            map_location="cpu",
            weights_only=True,
        )
        if checkpoint.get("format_version") != 1:
            raise ValueError("Unsupported PPO checkpoint format.")
        config_values = dict(checkpoint["config"])
        config_values["hidden_sizes"] = tuple(config_values["hidden_sizes"])
        config = SMDPPPOConfig(**config_values)
        if device is not None:
            config = replace(config, device=device)
        agent = cls(observation_size, action_size, config)
        agent.model.load_state_dict(checkpoint["model_state"])
        if load_optimizer:
            agent.optimizer.load_state_dict(checkpoint["optimizer_state"])
        agent.total_macro_steps = int(checkpoint["total_macro_steps"])
        agent.update_count = int(checkpoint["update_count"])
        return agent, dict(checkpoint.get("extra", {}))

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
