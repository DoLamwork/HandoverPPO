"""Actor-critic network for categorical handover decisions."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    """Shared MLP encoder with categorical policy and scalar value heads."""

    def __init__(
        self,
        observation_size: int,
        action_size: int,
        *,
        hidden_sizes: Sequence[int] = (64, 64),
        activation: str = "tanh",
    ) -> None:
        super().__init__()
        if observation_size <= 0 or action_size <= 1:
            raise ValueError("Invalid observation or action size.")
        self.observation_size = observation_size
        self.action_size = action_size

        activation_type: type[nn.Module]
        if activation == "tanh":
            activation_type = nn.Tanh
        elif activation == "relu":
            activation_type = nn.ReLU
        else:
            raise ValueError("activation must be 'tanh' or 'relu'.")

        layers: list[nn.Module] = []
        input_size = observation_size
        for hidden_size in hidden_sizes:
            if hidden_size <= 0:
                raise ValueError("hidden_sizes must be positive.")
            linear = nn.Linear(input_size, hidden_size)
            nn.init.orthogonal_(linear.weight, gain=np.sqrt(2))
            nn.init.zeros_(linear.bias)
            layers.extend((linear, activation_type()))
            input_size = hidden_size

        self.encoder = nn.Sequential(*layers)
        self.policy_head = nn.Linear(input_size, action_size)
        self.value_head = nn.Linear(input_size, 1)
        # Exact equal logits make the initial valid-action distribution uniform.
        nn.init.zeros_(self.policy_head.weight)
        nn.init.zeros_(self.policy_head.bias)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)

    def forward(
        self,
        observations: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(observations)
        return self.policy_head(features), self.value_head(features).squeeze(-1)

    def distribution_and_value(
        self,
        observations: torch.Tensor,
        action_masks: torch.Tensor | None = None,
    ) -> tuple[Categorical, torch.Tensor]:
        logits, values = self(observations)
        if action_masks is not None:
            action_masks = action_masks.to(device=logits.device, dtype=torch.bool)
            if action_masks.shape != logits.shape:
                raise ValueError("Action masks must match the policy logits.")
            if not torch.all(action_masks.any(dim=-1)):
                raise ValueError("Each action mask must allow at least one action.")
            logits = logits.masked_fill(
                ~action_masks,
                torch.finfo(logits.dtype).min,
            )
        return Categorical(logits=logits), values

    @torch.no_grad()
    def act(
        self,
        observation: np.ndarray,
        *,
        device: torch.device,
        deterministic: bool = False,
        action_mask: np.ndarray | None = None,
    ) -> tuple[int, float, float]:
        tensor = torch.as_tensor(
            observation,
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)
        masks = (
            None
            if action_mask is None
            else torch.as_tensor(
                action_mask,
                dtype=torch.bool,
                device=device,
            ).unsqueeze(0)
        )
        distribution, value = self.distribution_and_value(tensor, masks)
        action = (
            torch.argmax(distribution.logits, dim=-1)
            if deterministic
            else distribution.sample()
        )
        return (
            int(action.item()),
            float(distribution.log_prob(action).item()),
            float(value.item()),
        )

    @torch.no_grad()
    def act_with_probabilities(
        self,
        observation: np.ndarray,
        *,
        device: torch.device,
        deterministic: bool = False,
        action_mask: np.ndarray | None = None,
    ) -> tuple[int, float, float, np.ndarray]:
        """Choose an action and expose its complete masked distribution."""
        tensor = torch.as_tensor(
            observation,
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)
        masks = (
            None
            if action_mask is None
            else torch.as_tensor(
                action_mask,
                dtype=torch.bool,
                device=device,
            ).unsqueeze(0)
        )
        distribution, value = self.distribution_and_value(tensor, masks)
        action = (
            torch.argmax(distribution.logits, dim=-1)
            if deterministic
            else distribution.sample()
        )
        return (
            int(action.item()),
            float(distribution.log_prob(action).item()),
            float(value.item()),
            distribution.probs.squeeze(0).cpu().numpy().copy(),
        )

    @torch.no_grad()
    def predict_value(
        self,
        observation: np.ndarray,
        *,
        device: torch.device,
    ) -> float:
        tensor = torch.as_tensor(
            observation,
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)
        _, value = self(tensor)
        return float(value.item())
