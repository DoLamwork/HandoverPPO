"""Primitive reward contract and auditable reward decomposition."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import math
from typing import TYPE_CHECKING

from ho_optim_drl.protocol.types import TickResult

if TYPE_CHECKING:
    from ho_optim_drl.config import Config


@dataclass(frozen=True)
class RewardBreakdown:
    """Additive reward components for one tick or one macro-step."""

    rate: float = 0.0
    rlf: float = 0.0
    ping_pong: float = 0.0

    def __post_init__(self) -> None:
        for name in ("rate", "rlf", "ping_pong"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"Reward component {name} must be finite.")

    @property
    def total(self) -> float:
        return self.rate + self.rlf + self.ping_pong

    def __add__(self, other: "RewardBreakdown") -> "RewardBreakdown":
        if not isinstance(other, RewardBreakdown):
            return NotImplemented
        return RewardBreakdown(
            rate=self.rate + other.rate,
            rlf=self.rlf + other.rlf,
            ping_pong=self.ping_pong + other.ping_pong,
        )


@dataclass(frozen=True)
class HandoverRewardConfig:
    """Interpretable scales for the first handover objective."""

    rate_scale_mbps: float = 100.0
    rlf_penalty: float = 5.0
    ping_pong_penalty: float = 1.0

    def validate(self) -> None:
        values = {
            "rate_scale_mbps": self.rate_scale_mbps,
            "rlf_penalty": self.rlf_penalty,
            "ping_pong_penalty": self.ping_pong_penalty,
        }
        for name, value in values.items():
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite.")
        if self.rate_scale_mbps <= 0:
            raise ValueError("rate_scale_mbps must be positive.")
        if self.rlf_penalty < 0 or self.ping_pong_penalty < 0:
            raise ValueError("Event penalties must be non-negative.")


class RewardModel(ABC):
    """Compute one reward contribution per primitive timestep."""

    @abstractmethod
    def reset(self) -> None:
        """Reset episode-local reward state."""

    @abstractmethod
    def compute(self, tick: TickResult) -> RewardBreakdown:
        """Return the reward components for one primitive timestep."""


class HandoverRewardModel(RewardModel):
    """
    Reward delivered data while penalizing RLF and ping-pong events.

    The rate term is normalized delivered data:

    ``delta_t_seconds * delivered_rate_mbps / rate_scale_mbps``.

    HO preparation/execution, HOF, disconnection, and RLF recovery already
    lose rate because the protocol engine reports zero delivered throughput.
    A separate RLF penalty represents the additional severity of link loss.
    """

    def __init__(
        self,
        protocol_config: "Config",
        reward_config: HandoverRewardConfig | None = None,
    ) -> None:
        self.protocol_config = protocol_config
        self.config = reward_config or HandoverRewardConfig()
        self.config.validate()
        if protocol_config.delta_t_ms <= 0:
            raise ValueError("delta_t_ms must be positive.")
        self._delta_t_seconds = protocol_config.delta_t_ms / 1000.0

    def reset(self) -> None:
        """The current model has no episode-local state."""

    def compute(self, tick: TickResult) -> RewardBreakdown:
        """Compute the auditable reward contribution for one radio tick."""
        delivered_rate = float(tick.delivered_rate_mbps)
        if not math.isfinite(delivered_rate) or delivered_rate < 0:
            raise ValueError(
                "delivered_rate_mbps must be finite and non-negative."
            )

        return RewardBreakdown(
            rate=(
                self._delta_t_seconds
                * delivered_rate
                / self.config.rate_scale_mbps
            ),
            rlf=-self.config.rlf_penalty if tick.rlf_started else 0.0,
            ping_pong=(
                -self.config.ping_pong_penalty
                if tick.ping_pong_detected
                else 0.0
            ),
        )
