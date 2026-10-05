"""Configuration for duration-aware SMDP PPO."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class SMDPPPOConfig:
    """Hyperparameters for the first trainable handover policy."""

    hidden_sizes: tuple[int, ...] = (64, 64)
    activation: str = "tanh"
    network_architecture: str = "shared_mlp_v1"
    candidate_feature_size: int = 15
    value_normalization: str = "none"
    value_target_scale: float = 1.0
    learning_rate: float = 3e-4
    rollout_steps: int = 512
    batch_size: int = 64
    update_epochs: int = 10
    clip_range: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    max_grad_norm: float = 0.5
    gamma_per_second: float = 0.99
    lambda_per_second: float = 0.95
    normalize_advantage: bool = True
    target_kl: float | None = 0.03
    learning_diagnostics: bool = False
    seed: int = 17
    device: str = "cpu"

    def validate(self) -> None:
        if not self.hidden_sizes or any(size <= 0 for size in self.hidden_sizes):
            raise ValueError("hidden_sizes must contain positive layer sizes.")
        if self.activation not in {"tanh", "relu"}:
            raise ValueError("activation must be 'tanh' or 'relu'.")
        if self.network_architecture not in {
            "shared_mlp_v1",
            "separate_mlp_v1",
            "candidate_scorer_v1",
        }:
            raise ValueError("Unsupported network_architecture.")
        if self.candidate_feature_size <= 0:
            raise ValueError("candidate_feature_size must be positive.")
        if self.value_normalization not in {"none", "fixed_scale_v1"}:
            raise ValueError("Unsupported value_normalization.")
        if (
            not math.isfinite(self.value_target_scale)
            or self.value_target_scale < 1.0
        ):
            raise ValueError("value_target_scale must be finite and >= 1.")
        if self.value_normalization == "none" and self.value_target_scale != 1.0:
            raise ValueError("value_normalization='none' requires scale 1.")

        positive = {
            "learning_rate": self.learning_rate,
            "rollout_steps": self.rollout_steps,
            "batch_size": self.batch_size,
            "update_epochs": self.update_epochs,
            "clip_range": self.clip_range,
            "value_coefficient": self.value_coefficient,
            "max_grad_norm": self.max_grad_norm,
        }
        for name, value in positive.items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        if not math.isfinite(self.entropy_coefficient):
            raise ValueError("entropy_coefficient must be finite.")
        if self.entropy_coefficient < 0:
            raise ValueError("entropy_coefficient must be non-negative.")
        for name, value in {
            "gamma_per_second": self.gamma_per_second,
            "lambda_per_second": self.lambda_per_second,
        }.items():
            if not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError(f"{name} must be in (0, 1].")
        if self.target_kl is not None and (
            not math.isfinite(self.target_kl) or self.target_kl <= 0
        ):
            raise ValueError("target_kl must be finite and positive or None.")
        if self.seed < 0:
            raise ValueError("seed must be non-negative.")
        if not isinstance(self.learning_diagnostics, bool):
            raise ValueError("learning_diagnostics must be boolean.")
