"""Primitive reward contracts and auditable reward decomposition."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import math
from typing import TYPE_CHECKING, Any, Iterable

import numpy as np

from ho_optim_drl.data import RadioTrace
from ho_optim_drl.protocol.types import ProtocolPhase, TickResult

if TYPE_CHECKING:
    from ho_optim_drl.config import Config


LEGACY_REWARD_SCHEMA = "legacy_handover_reward_v1"
P3_E07_REWARD_SCHEMA = "trajectory_qos_reward_v1"
P3_E08_REWARD_SCHEMA = "trajectory_qos_events_reward_v1"
P3_E09_REWARD_SCHEMA = "trajectory_qos_recovery_reward_v1"
P3_E10_REWARD_SCHEMA = "trajectory_qos_dual_reward_v1"
SUPPORTED_REWARD_SCHEMAS = (
    LEGACY_REWARD_SCHEMA,
    P3_E07_REWARD_SCHEMA,
    P3_E08_REWARD_SCHEMA,
    P3_E09_REWARD_SCHEMA,
    P3_E10_REWARD_SCHEMA,
)


@dataclass(frozen=True)
class RewardBreakdown:
    """Additive reward components for one tick or one macro-step."""

    # Preserve the historical positional order for the first three fields.
    rate: float = 0.0
    rlf: float = 0.0
    ping_pong: float = 0.0
    handover: float = 0.0
    handover_failure: float = 0.0
    rlf_recovery: float = 0.0

    COMPONENTS = (
        "rate",
        "rlf",
        "ping_pong",
        "handover",
        "handover_failure",
        "rlf_recovery",
    )

    def __post_init__(self) -> None:
        for name in self.COMPONENTS:
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"Reward component {name} must be finite.")

    @property
    def total(self) -> float:
        return sum(getattr(self, name) for name in self.COMPONENTS)

    def __add__(self, other: "RewardBreakdown") -> "RewardBreakdown":
        if not isinstance(other, RewardBreakdown):
            return NotImplemented
        return RewardBreakdown(
            **{
                name: getattr(self, name) + getattr(other, name)
                for name in self.COMPONENTS
            }
        )

    def scaled(self, factor: float) -> "RewardBreakdown":
        """Return every component multiplied by one finite factor."""
        if not math.isfinite(factor):
            raise ValueError("Reward scale factor must be finite.")
        return RewardBreakdown(
            **{
                name: getattr(self, name) * factor
                for name in self.COMPONENTS
            }
        )

    def as_dict(self) -> dict[str, float]:
        """Return every component plus the additive total."""
        values = {name: float(getattr(self, name)) for name in self.COMPONENTS}
        values["total"] = float(self.total)
        return values


@dataclass(frozen=True)
class HandoverRewardConfig:
    """Interpretable scales for the legacy handover objective."""

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


@dataclass(frozen=True)
class TrajectoryQoSRewardConfig:
    """Static P3 reward contract shared by E07--E10."""

    utility_scale: float = 60.0
    handover_penalty: float = 0.0
    handover_failure_penalty: float = 0.0
    rlf_penalty: float = 5.0
    ping_pong_penalty: float = 1.0
    rlf_recovery_penalty_per_second: float = 0.0
    pure_rlf_only: bool = False
    dual_enabled: bool = False
    dual_target_ratio: float = 0.98
    dual_learning_rate: float = 0.5
    dual_max_coefficient: float = 10.0
    dual_min_episodes_per_speed: int = 4

    def validate(self) -> None:
        numeric = {
            "utility_scale": self.utility_scale,
            "handover_penalty": self.handover_penalty,
            "handover_failure_penalty": self.handover_failure_penalty,
            "rlf_penalty": self.rlf_penalty,
            "ping_pong_penalty": self.ping_pong_penalty,
            "rlf_recovery_penalty_per_second": (
                self.rlf_recovery_penalty_per_second
            ),
            "dual_target_ratio": self.dual_target_ratio,
            "dual_learning_rate": self.dual_learning_rate,
            "dual_max_coefficient": self.dual_max_coefficient,
        }
        for name, value in numeric.items():
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite.")
        if self.utility_scale <= 0:
            raise ValueError("utility_scale must be positive.")
        for name in (
            "handover_penalty",
            "handover_failure_penalty",
            "rlf_penalty",
            "ping_pong_penalty",
            "rlf_recovery_penalty_per_second",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative.")
        if not 0 < self.dual_target_ratio <= 1:
            raise ValueError("dual_target_ratio must be in (0, 1].")
        if self.dual_learning_rate <= 0:
            raise ValueError("dual_learning_rate must be positive.")
        if self.dual_max_coefficient <= 0:
            raise ValueError("dual_max_coefficient must be positive.")
        if (
            not isinstance(self.dual_min_episodes_per_speed, int)
            or isinstance(self.dual_min_episodes_per_speed, bool)
            or self.dual_min_episodes_per_speed <= 0
        ):
            raise ValueError(
                "dual_min_episodes_per_speed must be a positive integer."
            )


def trajectory_reward_config(
    schema: str,
    *,
    dual_target_ratio: float = 0.98,
    dual_learning_rate: float = 0.5,
    dual_max_coefficient: float = 10.0,
    dual_min_episodes_per_speed: int = 4,
) -> TrajectoryQoSRewardConfig:
    """Return the preregistered one-factor config for a P3 arm."""
    common = {
        "utility_scale": 60.0,
        "handover_penalty": 0.0,
        "dual_target_ratio": dual_target_ratio,
        "dual_learning_rate": dual_learning_rate,
        "dual_max_coefficient": dual_max_coefficient,
        "dual_min_episodes_per_speed": dual_min_episodes_per_speed,
    }
    if schema == P3_E07_REWARD_SCHEMA:
        return TrajectoryQoSRewardConfig(
            **common,
            handover_failure_penalty=0.0,
            rlf_penalty=5.0,
            ping_pong_penalty=1.0,
        )
    if schema == P3_E08_REWARD_SCHEMA:
        return TrajectoryQoSRewardConfig(
            **common,
            handover_failure_penalty=0.2,
            rlf_penalty=0.2,
            ping_pong_penalty=0.1,
            pure_rlf_only=True,
        )
    if schema in (P3_E09_REWARD_SCHEMA, P3_E10_REWARD_SCHEMA):
        return TrajectoryQoSRewardConfig(
            **common,
            handover_failure_penalty=0.2,
            rlf_penalty=0.2,
            ping_pong_penalty=0.1,
            rlf_recovery_penalty_per_second=1.0,
            pure_rlf_only=True,
            dual_enabled=schema == P3_E10_REWARD_SCHEMA,
        )
    raise ValueError(f"Unsupported P3 reward schema: {schema!r}.")


class RewardModel(ABC):
    """Compute one reward contribution per primitive timestep."""

    schema_name: str
    config: HandoverRewardConfig | TrajectoryQoSRewardConfig

    @abstractmethod
    def reset(self, trace: RadioTrace | None = None) -> None:
        """Reset episode-local reward state."""

    @abstractmethod
    def compute(self, tick: TickResult) -> RewardBreakdown:
        """Return the reward components for one primitive timestep."""

    def begin_rollout(self) -> None:
        """Freeze any adaptive reward state for one rollout."""

    def update_dual(self, episodes: Iterable[Any]) -> dict[str, Any] | None:
        """Update adaptive state after a rollout, if supported."""
        del episodes
        return None

    def state_dict(self) -> dict[str, Any]:
        """Return dynamic reward state stored in a checkpoint."""
        return {"schema": self.schema_name}

    def load_state_dict(self, state: dict[str, Any] | None) -> None:
        """Restore dynamic reward state from a checkpoint."""
        if not state:
            return
        schema = state.get("schema")
        if schema is not None and schema != self.schema_name:
            raise ValueError(
                f"Reward state schema {schema!r} does not match "
                f"{self.schema_name!r}."
            )


class HandoverRewardModel(RewardModel):
    """Legacy delivered-rate reward with RLF and ping-pong penalties."""

    schema_name = LEGACY_REWARD_SCHEMA

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

    def reset(self, trace: RadioTrace | None = None) -> None:
        del trace

    def compute(self, tick: TickResult) -> RewardBreakdown:
        delivered_rate = _validated_rate(tick.delivered_rate_mbps)
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


class TrajectoryQoSRewardModel(RewardModel):
    """P3 trajectory-normalized utility and explicit protocol costs."""

    def __init__(
        self,
        protocol_config: "Config",
        schema_name: str,
        reward_config: TrajectoryQoSRewardConfig | None = None,
    ) -> None:
        if schema_name not in SUPPORTED_REWARD_SCHEMAS[1:]:
            raise ValueError(f"Unsupported P3 reward schema: {schema_name!r}.")
        self.protocol_config = protocol_config
        self.schema_name = schema_name
        self.config = reward_config or trajectory_reward_config(schema_name)
        self.config.validate()
        if self.config.dual_enabled != (
            schema_name == P3_E10_REWARD_SCHEMA
        ):
            raise ValueError(
                "dual_enabled must be true only for the E10 reward schema."
            )
        if protocol_config.delta_t_ms <= 0:
            raise ValueError("delta_t_ms must be positive.")
        self._delta_t_seconds = protocol_config.delta_t_ms / 1000.0
        self._normalizer_mbit: float | None = None
        self._current_speed_kmh: int | None = None
        self._dual_lambdas: dict[int, float] = {}
        self._rollout_lambdas: dict[int, float] = {}

    def reset(self, trace: RadioTrace | None = None) -> None:
        if trace is None:
            raise ValueError(
                "Trajectory-normalized reward requires the current RadioTrace."
            )
        self._normalizer_mbit = self._best_delivered_data_mbit(trace)
        speed = trace.speed_kmh
        if self.config.dual_enabled and speed is None:
            raise ValueError("Dual QoS reward requires trace speed_kmh metadata.")
        self._current_speed_kmh = None if speed is None else int(speed)
        if self._current_speed_kmh is not None:
            self._dual_lambdas.setdefault(self._current_speed_kmh, 0.0)
            self._rollout_lambdas.setdefault(self._current_speed_kmh, 0.0)

    def begin_rollout(self) -> None:
        self._rollout_lambdas = dict(self._dual_lambdas)

    def compute(self, tick: TickResult) -> RewardBreakdown:
        delivered_rate = _validated_rate(tick.delivered_rate_mbps)
        if self._normalizer_mbit is None:
            raise RuntimeError("Call reset(trace) before computing P3 reward.")
        multiplier = 1.0
        if self.config.dual_enabled:
            if self._current_speed_kmh is None:
                raise RuntimeError("Dual reward has no current speed.")
            multiplier += self._rollout_lambdas.get(
                self._current_speed_kmh,
                self._dual_lambdas.get(self._current_speed_kmh, 0.0),
            )

        pure_rlf = tick.rlf_started and (
            not self.config.pure_rlf_only or not tick.handover_failed
        )
        return RewardBreakdown(
            rate=(
                multiplier
                * self.config.utility_scale
                * self._delta_t_seconds
                * delivered_rate
                / self._normalizer_mbit
            ),
            handover=(
                -self.config.handover_penalty
                if tick.handover_started
                else 0.0
            ),
            handover_failure=(
                -self.config.handover_failure_penalty
                if tick.handover_failed
                else 0.0
            ),
            rlf=-self.config.rlf_penalty if pure_rlf else 0.0,
            ping_pong=(
                -self.config.ping_pong_penalty
                if tick.ping_pong_detected
                else 0.0
            ),
            rlf_recovery=(
                -self.config.rlf_recovery_penalty_per_second
                * self._delta_t_seconds
                if tick.snapshot.phase is ProtocolPhase.RLF_RECOVERY
                else 0.0
            ),
        )

    def update_dual(self, episodes: Iterable[Any]) -> dict[str, Any] | None:
        if not self.config.dual_enabled:
            return None
        ratios: dict[int, list[float]] = {}
        for episode in episodes:
            speed = getattr(episode, "speed_kmh", None)
            ratio = getattr(episode, "relative_rate_ratio", None)
            if speed is None or ratio is None:
                continue
            speed_int = int(speed)
            ratio_float = float(ratio)
            if not math.isfinite(ratio_float) or ratio_float < 0:
                raise ValueError("Episode relative-rate ratio must be finite.")
            ratios.setdefault(speed_int, []).append(ratio_float)

        for speed in ratios:
            self._dual_lambdas.setdefault(speed, 0.0)
        before = dict(self._dual_lambdas)
        means: dict[str, float] = {}
        shortfalls: dict[str, float] = {}
        counts: dict[str, int] = {}
        updated: list[str] = []
        infeasible: list[str] = []
        for speed in sorted(self._dual_lambdas):
            values = ratios.get(speed, [])
            key = str(speed)
            counts[key] = len(values)
            if not values:
                continue
            mean_ratio = float(np.mean(values))
            means[key] = mean_ratio
            shortfall = self.config.dual_target_ratio - mean_ratio
            shortfalls[key] = shortfall
            if len(values) < self.config.dual_min_episodes_per_speed:
                continue
            self._dual_lambdas[speed] = float(
                np.clip(
                    self._dual_lambdas[speed]
                    + self.config.dual_learning_rate * shortfall,
                    0.0,
                    self.config.dual_max_coefficient,
                )
            )
            updated.append(key)
            if (
                shortfall > 0
                and self._dual_lambdas[speed]
                >= self.config.dual_max_coefficient
            ):
                infeasible.append(key)

        return {
            "target_ratio": self.config.dual_target_ratio,
            "sample_count_by_speed": counts,
            "mean_relative_rate_ratio_by_speed": means,
            "shortfall_by_speed": shortfalls,
            "lambda_before_by_speed": {
                str(speed): value for speed, value in sorted(before.items())
            },
            "lambda_after_by_speed": {
                str(speed): value
                for speed, value in sorted(self._dual_lambdas.items())
            },
            "updated_speeds": updated,
            "infeasible_speeds": infeasible,
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema_name,
            "dual_lambdas_by_speed": {
                str(speed): float(value)
                for speed, value in sorted(self._dual_lambdas.items())
            },
        }

    def load_state_dict(self, state: dict[str, Any] | None) -> None:
        super().load_state_dict(state)
        if not state:
            return
        raw = state.get("dual_lambdas_by_speed", {})
        restored: dict[int, float] = {}
        for speed, value in raw.items():
            coefficient = float(value)
            if (
                not math.isfinite(coefficient)
                or coefficient < 0
                or coefficient > self.config.dual_max_coefficient
            ):
                raise ValueError("Invalid saved dual coefficient.")
            restored[int(speed)] = coefficient
        self._dual_lambdas = restored
        self.begin_rollout()

    def _best_delivered_data_mbit(self, trace: RadioTrace) -> float:
        max_sinr_db = np.max(trace.sinr_rate_db, axis=0)
        spectral_efficiency = np.logaddexp(
            0.0,
            max_sinr_db * math.log(10.0) / 10.0,
        ) / math.log(2.0)
        best_rate_mbps = self.protocol_config.bw * spectral_efficiency / 1e6
        normalizer = float(self._delta_t_seconds * np.sum(best_rate_mbps))
        if not math.isfinite(normalizer) or normalizer <= 0:
            raise ValueError("Full-trace best-rate normalizer must be positive.")
        return normalizer


def build_reward_model(
    protocol_config: "Config",
    schema: str = LEGACY_REWARD_SCHEMA,
    reward_config: HandoverRewardConfig | TrajectoryQoSRewardConfig | None = None,
) -> RewardModel:
    """Build the versioned reward implementation for one environment."""
    if schema == LEGACY_REWARD_SCHEMA:
        if reward_config is not None and not isinstance(
            reward_config, HandoverRewardConfig
        ):
            raise TypeError("Legacy reward requires HandoverRewardConfig.")
        return HandoverRewardModel(protocol_config, reward_config)
    if schema in SUPPORTED_REWARD_SCHEMAS[1:]:
        if reward_config is not None and not isinstance(
            reward_config, TrajectoryQoSRewardConfig
        ):
            raise TypeError("P3 reward requires TrajectoryQoSRewardConfig.")
        return TrajectoryQoSRewardModel(
            protocol_config,
            schema,
            reward_config,
        )
    raise ValueError(f"Unsupported reward schema: {schema!r}.")


def reward_config_from_dict(
    schema: str,
    values: dict[str, Any] | None,
) -> HandoverRewardConfig | TrajectoryQoSRewardConfig | None:
    """Rebuild a frozen reward config stored in checkpoint metadata."""
    if not values:
        return None
    if schema == LEGACY_REWARD_SCHEMA:
        config = HandoverRewardConfig(**values)
        config.validate()
        return config
    if schema in SUPPORTED_REWARD_SCHEMAS[1:]:
        config = TrajectoryQoSRewardConfig(**values)
        config.validate()
        return config
    raise ValueError(f"Unsupported reward schema: {schema!r}.")


def _validated_rate(value: float) -> float:
    rate = float(value)
    if not math.isfinite(rate) or rate < 0:
        raise ValueError("delivered_rate_mbps must be finite and non-negative.")
    return rate
