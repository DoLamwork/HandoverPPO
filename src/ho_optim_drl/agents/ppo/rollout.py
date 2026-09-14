"""Rollout storage and duration-aware generalized advantage estimation."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch


@dataclass(frozen=True)
class RolloutTensors:
    """One complete rollout converted to PyTorch tensors."""

    observations: torch.Tensor
    actions: torch.Tensor
    old_log_probabilities: torch.Tensor
    old_values: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor


class SMDPRolloutBuffer:
    """Fixed-capacity rollout buffer with one discount per macro-step."""

    def __init__(self, capacity: int, observation_shape: tuple[int, ...]) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive.")
        if len(observation_shape) != 1 or observation_shape[0] <= 0:
            raise ValueError("observation_shape must describe a non-empty vector.")

        self.capacity = capacity
        self.observation_shape = observation_shape
        self.observations = np.empty(
            (capacity, *observation_shape),
            dtype=np.float32,
        )
        self.actions = np.empty(capacity, dtype=np.int64)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.old_log_probabilities = np.empty(capacity, dtype=np.float32)
        self.values = np.empty(capacity, dtype=np.float32)
        self.next_values = np.empty(capacity, dtype=np.float32)
        self.discounts = np.empty(capacity, dtype=np.float32)
        self.duration_seconds = np.empty(capacity, dtype=np.float32)
        self.terminated = np.empty(capacity, dtype=np.bool_)
        self.truncated = np.empty(capacity, dtype=np.bool_)
        self.trace_ids = np.empty(capacity, dtype=object)
        self.trace_ids.fill(None)
        self.advantages = np.empty(capacity, dtype=np.float32)
        self.returns = np.empty(capacity, dtype=np.float32)
        self.size = 0
        self._advantages_ready = False

    @property
    def full(self) -> bool:
        return self.size == self.capacity

    def add(
        self,
        *,
        observation: np.ndarray,
        action: int,
        reward: float,
        old_log_probability: float,
        value: float,
        next_value: float,
        discount: float,
        duration_seconds: float,
        terminated: bool,
        truncated: bool,
        trace_id: str | None = None,
    ) -> None:
        """Store one macro-transition."""
        if self.full:
            raise RuntimeError("The rollout buffer is already full.")

        observation = np.asarray(observation, dtype=np.float32)
        if observation.shape != self.observation_shape:
            raise ValueError(
                f"Expected observation shape {self.observation_shape}, "
                f"got {observation.shape}."
            )
        numeric = {
            "reward": reward,
            "old_log_probability": old_log_probability,
            "value": value,
            "next_value": next_value,
            "discount": discount,
            "duration_seconds": duration_seconds,
        }
        if not np.all(np.isfinite(observation)):
            raise ValueError("observation must contain only finite values.")
        for name, value_item in numeric.items():
            if not math.isfinite(value_item):
                raise ValueError(f"{name} must be finite.")
        if not 0 < discount <= 1:
            raise ValueError("discount must be in (0, 1].")
        if duration_seconds <= 0:
            raise ValueError("duration_seconds must be positive.")
        if trace_id is not None and not trace_id:
            raise ValueError("trace_id must be non-empty when provided.")

        index = self.size
        self.observations[index] = observation
        self.actions[index] = action
        self.rewards[index] = reward
        self.old_log_probabilities[index] = old_log_probability
        self.values[index] = value
        self.next_values[index] = next_value
        self.discounts[index] = discount
        self.duration_seconds[index] = duration_seconds
        self.terminated[index] = terminated
        self.truncated[index] = truncated
        self.trace_ids[index] = trace_id
        self.size += 1
        self._advantages_ready = False

    def compute_returns_and_advantages(
        self,
        lambda_per_second: float,
    ) -> None:
        """Compute GAE using physical-time gamma and lambda factors."""
        if self.size == 0:
            raise RuntimeError("Cannot compute advantages for an empty rollout.")
        if not math.isfinite(lambda_per_second) or not 0 < lambda_per_second <= 1:
            raise ValueError("lambda_per_second must be in (0, 1].")

        next_advantage = 0.0
        for index in range(self.size - 1, -1, -1):
            bootstrap_mask = 0.0 if self.terminated[index] else 1.0
            episode_continues = not (
                self.terminated[index] or self.truncated[index]
            )
            delta = (
                self.rewards[index]
                + self.discounts[index]
                * bootstrap_mask
                * self.next_values[index]
                - self.values[index]
            )
            lambda_discount = (
                lambda_per_second ** float(self.duration_seconds[index])
            )
            self.advantages[index] = (
                delta
                + self.discounts[index]
                * lambda_discount
                * float(episode_continues)
                * next_advantage
            )
            next_advantage = float(self.advantages[index])

        self.returns[: self.size] = (
            self.advantages[: self.size] + self.values[: self.size]
        )
        if not np.all(np.isfinite(self.advantages[: self.size])):
            raise RuntimeError("Computed advantages contain non-finite values.")
        self._advantages_ready = True

    def to_tensors(
        self,
        device: torch.device,
        *,
        normalize_advantage: bool,
    ) -> RolloutTensors:
        """Convert the populated portion of the buffer to model tensors."""
        if not self._advantages_ready:
            raise RuntimeError("Compute returns and advantages first.")

        advantages = torch.as_tensor(
            self.advantages[: self.size],
            dtype=torch.float32,
            device=device,
        )
        if normalize_advantage and self.size > 1:
            advantages = (
                advantages - advantages.mean()
            ) / (advantages.std(unbiased=False) + 1e-8)

        return RolloutTensors(
            observations=torch.as_tensor(
                self.observations[: self.size],
                dtype=torch.float32,
                device=device,
            ),
            actions=torch.as_tensor(
                self.actions[: self.size],
                dtype=torch.long,
                device=device,
            ),
            old_log_probabilities=torch.as_tensor(
                self.old_log_probabilities[: self.size],
                dtype=torch.float32,
                device=device,
            ),
            old_values=torch.as_tensor(
                self.values[: self.size],
                dtype=torch.float32,
                device=device,
            ),
            returns=torch.as_tensor(
                self.returns[: self.size],
                dtype=torch.float32,
                device=device,
            ),
            advantages=advantages,
        )
