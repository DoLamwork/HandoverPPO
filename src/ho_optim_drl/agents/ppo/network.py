"""Actor-critic network for categorical handover decisions."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    """Categorical actor-critic with shared or separate MLP encoders."""

    def __init__(
        self,
        observation_size: int,
        action_size: int,
        *,
        hidden_sizes: Sequence[int] = (64, 64),
        activation: str = "tanh",
        architecture: str = "shared_mlp_v1",
        candidate_feature_size: int = 15,
        value_target_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if observation_size <= 0 or action_size <= 1:
            raise ValueError("Invalid observation or action size.")
        self.observation_size = observation_size
        self.action_size = action_size
        if not hidden_sizes:
            raise ValueError("hidden_sizes must not be empty.")
        if architecture not in {
            "shared_mlp_v1",
            "separate_mlp_v1",
            "candidate_scorer_v1",
        }:
            raise ValueError("Unsupported network architecture.")
        if not np.isfinite(value_target_scale) or value_target_scale < 1.0:
            raise ValueError("value_target_scale must be finite and >= 1.")
        self.architecture = architecture
        self.candidate_feature_size = int(candidate_feature_size)
        self.value_target_scale = float(value_target_scale)

        activation_type: type[nn.Module]
        if activation == "tanh":
            activation_type = nn.Tanh
        elif activation == "relu":
            activation_type = nn.ReLU
        else:
            raise ValueError("activation must be 'tanh' or 'relu'.")

        def build_encoder(encoder_input_size: int) -> nn.Sequential:
            layers: list[nn.Module] = []
            input_size = encoder_input_size
            for hidden_size in hidden_sizes:
                if hidden_size <= 0:
                    raise ValueError("hidden_sizes must be positive.")
                linear = nn.Linear(input_size, hidden_size)
                nn.init.orthogonal_(linear.weight, gain=np.sqrt(2))
                nn.init.zeros_(linear.bias)
                layers.extend((linear, activation_type()))
                input_size = hidden_size
            return nn.Sequential(*layers)

        output_size = int(hidden_sizes[-1])
        if architecture == "shared_mlp_v1":
            self.encoder = build_encoder(observation_size)
            self.policy_head = nn.Linear(output_size, action_size)
        elif architecture == "separate_mlp_v1":
            self.actor_encoder = build_encoder(observation_size)
            self.critic_encoder = build_encoder(observation_size)
            self.policy_head = nn.Linear(output_size, action_size)
        else:
            candidate_count = action_size - 1
            self.global_feature_size = (
                observation_size - candidate_count * self.candidate_feature_size
            )
            if self.global_feature_size <= 0:
                raise ValueError(
                    "Observation is too small for candidate_scorer_v1 layout."
                )
            self.stay_encoder = build_encoder(self.global_feature_size)
            self.candidate_encoder = build_encoder(
                self.global_feature_size + self.candidate_feature_size
            )
            self.critic_encoder = build_encoder(observation_size)
            self.stay_head = nn.Linear(output_size, 1)
            self.candidate_head = nn.Linear(output_size, 1)
        self.value_head = nn.Linear(output_size, 1)
        # Exact equal logits make the initial valid-action distribution uniform.
        if architecture == "candidate_scorer_v1":
            nn.init.zeros_(self.stay_head.weight)
            nn.init.zeros_(self.stay_head.bias)
            nn.init.zeros_(self.candidate_head.weight)
            nn.init.zeros_(self.candidate_head.bias)
        else:
            nn.init.zeros_(self.policy_head.weight)
            nn.init.zeros_(self.policy_head.bias)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)

    def forward(
        self,
        observations: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.architecture == "shared_mlp_v1":
            actor_features = critic_features = self.encoder(observations)
            logits = self.policy_head(actor_features)
        elif self.architecture == "separate_mlp_v1":
            actor_features = self.actor_encoder(observations)
            critic_features = self.critic_encoder(observations)
            logits = self.policy_head(actor_features)
        else:
            context = observations[:, : self.global_feature_size]
            candidate_blocks = observations[:, self.global_feature_size :].reshape(
                observations.shape[0],
                self.action_size - 1,
                self.candidate_feature_size,
            )
            expanded_context = context.unsqueeze(1).expand(
                -1, self.action_size - 1, -1
            )
            candidate_inputs = torch.cat(
                (expanded_context, candidate_blocks), dim=-1
            )
            candidate_features = self.candidate_encoder(candidate_inputs)
            candidate_logits = self.candidate_head(candidate_features).squeeze(-1)
            stay_logits = self.stay_head(self.stay_encoder(context))
            logits = torch.cat((stay_logits, candidate_logits), dim=-1)
            critic_features = self.critic_encoder(observations)
        normalized_value = self.value_head(critic_features).squeeze(-1)
        return (
            logits,
            normalized_value * self.value_target_scale,
        )

    def normalized_value(self, observations: torch.Tensor) -> torch.Tensor:
        """Return critic output in normalized target units."""
        if self.architecture == "shared_mlp_v1":
            features = self.encoder(observations)
        else:
            features = self.critic_encoder(observations)
        return self.value_head(features).squeeze(-1)

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
