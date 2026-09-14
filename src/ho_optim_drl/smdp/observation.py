"""Causal observation construction at an A3 decision point."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ho_optim_drl.protocol.types import (
    ProtocolPhase,
    ProtocolSnapshot,
    TickResult,
)

if TYPE_CHECKING:
    from ho_optim_drl.config import Config


class ObservationBuilder(ABC):
    """Build a causal observation at an A3 decision point."""

    @abstractmethod
    def reset(self) -> None:
        """Clear rolling histories at the beginning of an episode."""

    @abstractmethod
    def observe(self, tick: TickResult) -> None:
        """Update causal histories after one primitive timestep."""

    @property
    @abstractmethod
    def size(self) -> int:
        """Return the fixed observation-vector size."""

    @property
    @abstractmethod
    def feature_names(self) -> tuple[str, ...]:
        """Return feature names in vector order."""

    @property
    @abstractmethod
    def low(self) -> np.ndarray:
        """Return normalized lower bounds."""

    @property
    @abstractmethod
    def high(self) -> np.ndarray:
        """Return normalized upper bounds."""

    @abstractmethod
    def build(self, snapshot: ProtocolSnapshot) -> np.ndarray:
        """Build a normalized observation using no future samples."""


@dataclass(frozen=True)
class A3ObservationConfig:
    """Normalization and history settings for the first observation model."""

    trend_window_ticks: int = 10
    history_window_ticks: int = 1024
    margin_clip_db: float = 10.0
    margin_std_clip_db: float = 5.0
    slope_clip_db_per_s: float = 100.0
    quality_gap_clip_db: float = 20.0
    residence_time_clip_s: float = 5.0
    time_since_ho_clip_s: float = 5.0
    recent_ho_count_clip: int = 5

    def validate(self) -> None:
        numeric_positive = {
            "trend_window_ticks": self.trend_window_ticks,
            "history_window_ticks": self.history_window_ticks,
            "margin_clip_db": self.margin_clip_db,
            "margin_std_clip_db": self.margin_std_clip_db,
            "slope_clip_db_per_s": self.slope_clip_db_per_s,
            "quality_gap_clip_db": self.quality_gap_clip_db,
            "residence_time_clip_s": self.residence_time_clip_s,
            "time_since_ho_clip_s": self.time_since_ho_clip_s,
            "recent_ho_count_clip": self.recent_ho_count_clip,
        }
        for name, value in numeric_positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive.")
        if self.history_window_ticks < self.trend_window_ticks:
            raise ValueError(
                "history_window_ticks must be at least trend_window_ticks."
            )


@dataclass(frozen=True)
class _RadioRecord:
    tic: int
    serving_cell: int | None
    target_cell: int | None
    serving_sinr_db: float | None
    target_sinr_db: float | None
    a3_margin_db: float | None


class A3ObservationBuilder(ObservationBuilder):
    """Build the normalized 19-feature observation for SMDP decisions."""

    FEATURE_NAMES = (
        "serving_spectral_efficiency",
        "target_spectral_efficiency",
        "delta_spectral_efficiency",
        "a3_margin_now",
        "a3_margin_mean",
        "a3_margin_min",
        "a3_margin_slope",
        "a3_margin_std",
        "serving_sinr_slope",
        "target_sinr_slope",
        "serving_qout_gap",
        "target_qin_gap",
        "n310_progress",
        "t310_active",
        "t310_progress",
        "serving_residence_time",
        "time_since_last_ho",
        "target_is_previous_cell",
        "recent_ho_count",
    )

    LOW = np.array(
        [
            0.0,
            0.0,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            0.0,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        dtype=np.float32,
    )
    HIGH = np.ones(len(FEATURE_NAMES), dtype=np.float32)

    def __init__(
        self,
        protocol_config: "Config",
        observation_config: A3ObservationConfig | None = None,
    ) -> None:
        self.protocol_config = protocol_config
        self.config = observation_config or A3ObservationConfig()
        self.config.validate()
        if (
            self.protocol_config.sinr_upper_clip
            <= self.protocol_config.sinr_lower_clip
        ):
            raise ValueError("SINR upper clip must exceed the lower clip.")
        if self.config.history_window_ticks < self.protocol_config.a3_ttt + 1:
            raise ValueError(
                "history_window_ticks must cover the complete configured TTT."
            )

        self._history: deque[_RadioRecord] = deque(
            maxlen=self.config.history_window_ticks
        )
        self._max_spectral_efficiency = self._spectral_efficiency(
            self.protocol_config.sinr_upper_clip
        )

    def reset(self) -> None:
        self._history.clear()

    def observe(self, tick: TickResult) -> None:
        snapshot = tick.snapshot
        serving = snapshot.serving_cell
        target = snapshot.target_cell
        serving_sinr = (
            float(snapshot.sinr_rate_db[serving])
            if serving is not None
            else None
        )
        target_sinr = (
            float(snapshot.sinr_rate_db[target])
            if target is not None
            else None
        )
        margin = None
        if serving is not None and target is not None:
            margin = float(
                snapshot.rsrp_a3_db[target]
                - snapshot.rsrp_a3_db[serving]
                - self.protocol_config.a3_hys
                - self.protocol_config.a3_off
            )

        self._history.append(
            _RadioRecord(
                tic=snapshot.tic,
                serving_cell=serving,
                target_cell=target,
                serving_sinr_db=serving_sinr,
                target_sinr_db=target_sinr,
                a3_margin_db=margin,
            )
        )

    @property
    def size(self) -> int:
        return len(self.FEATURE_NAMES)

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self.FEATURE_NAMES

    @property
    def low(self) -> np.ndarray:
        return self.LOW.copy()

    @property
    def high(self) -> np.ndarray:
        return self.HIGH.copy()

    def build(self, snapshot: ProtocolSnapshot) -> np.ndarray:
        """Build an observation only for a recorded A3 decision point."""
        decision = snapshot.decision_point
        if snapshot.phase is not ProtocolPhase.DECISION or decision is None:
            raise RuntimeError("Observation can only be built at a decision point.")
        if snapshot.tic != decision.tic:
            raise RuntimeError("Decision and snapshot timestamps do not match.")
        if not self._history or self._history[-1].tic != snapshot.tic:
            raise RuntimeError(
                "The decision tick has not been observed. Call observe(tick) "
                "for every TickResult before build()."
            )

        serving = decision.serving_cell
        target = decision.target_cell
        serving_rate_sinr = float(snapshot.sinr_rate_db[serving])
        target_rate_sinr = float(snapshot.sinr_rate_db[target])
        serving_rlm_sinr = float(snapshot.sinr_rlm_db[serving])
        target_rlm_sinr = float(snapshot.sinr_rlm_db[target])
        serving_eff = self._normalized_spectral_efficiency(serving_rate_sinr)
        target_eff = self._normalized_spectral_efficiency(target_rate_sinr)

        pair_history = [
            record
            for record in self._history
            if decision.a3_start_tic <= record.tic <= decision.tic
            and record.serving_cell == serving
            and record.target_cell == target
            and record.a3_margin_db is not None
        ]
        if not pair_history:
            raise RuntimeError("No causal A3 history is available for the decision.")

        margin_values = np.array(
            [record.a3_margin_db for record in pair_history],
            dtype=np.float64,
        )
        margin_tics = np.array(
            [record.tic for record in pair_history],
            dtype=np.float64,
        )

        trend_history = pair_history[-self.config.trend_window_ticks :]
        trend_tics = np.array(
            [record.tic for record in trend_history],
            dtype=np.float64,
        )
        serving_values = np.array(
            [record.serving_sinr_db for record in trend_history],
            dtype=np.float64,
        )
        target_values = np.array(
            [record.target_sinr_db for record in trend_history],
            dtype=np.float64,
        )

        target_is_previous = float(snapshot.previous_cell == target)
        time_since_ho = (
            self.config.time_since_ho_clip_s
            if snapshot.previous_cell is None
            else snapshot.time_since_last_ho_ticks
            * self.protocol_config.delta_t_ms
            / 1000
        )
        residence_s = (
            snapshot.serving_residence_ticks
            * self.protocol_config.delta_t_ms
            / 1000
        )

        observation = np.array(
            [
                serving_eff,
                target_eff,
                np.clip(target_eff - serving_eff, -1.0, 1.0),
                self._signed_normalize(
                    decision.a3_margin_db, self.config.margin_clip_db
                ),
                self._signed_normalize(
                    float(np.mean(margin_values)),
                    self.config.margin_clip_db,
                ),
                self._signed_normalize(
                    float(np.min(margin_values)),
                    self.config.margin_clip_db,
                ),
                self._signed_normalize(
                    self._slope_db_per_s(margin_tics, margin_values),
                    self.config.slope_clip_db_per_s,
                ),
                self._unsigned_normalize(
                    float(np.std(margin_values)),
                    self.config.margin_std_clip_db,
                ),
                self._signed_normalize(
                    self._slope_db_per_s(trend_tics, serving_values),
                    self.config.slope_clip_db_per_s,
                ),
                self._signed_normalize(
                    self._slope_db_per_s(trend_tics, target_values),
                    self.config.slope_clip_db_per_s,
                ),
                self._signed_normalize(
                    serving_rlm_sinr - self.protocol_config.q_out_db,
                    self.config.quality_gap_clip_db,
                ),
                self._signed_normalize(
                    target_rlm_sinr - self.protocol_config.q_in_db,
                    self.config.quality_gap_clip_db,
                ),
                np.clip(snapshot.n310_progress, 0.0, 1.0),
                float(snapshot.t310_active),
                np.clip(snapshot.t310_progress, 0.0, 1.0),
                self._unsigned_normalize(
                    residence_s, self.config.residence_time_clip_s
                ),
                self._unsigned_normalize(
                    time_since_ho, self.config.time_since_ho_clip_s
                ),
                target_is_previous,
                self._unsigned_normalize(
                    snapshot.recent_ho_count,
                    self.config.recent_ho_count_clip,
                ),
            ],
            dtype=np.float32,
        )
        if not np.all(np.isfinite(observation)):
            raise RuntimeError("Observation contains a non-finite value.")
        if np.any(observation < self.LOW) or np.any(observation > self.HIGH):
            raise RuntimeError("Observation violates its declared bounds.")
        return observation

    def as_dict(self, observation: np.ndarray) -> dict[str, float]:
        """Map an observation vector to named scalar values for logging."""
        observation = np.asarray(observation)
        if observation.shape != (self.size,):
            raise ValueError(
                f"Expected observation shape {(self.size,)}, "
                f"got {observation.shape}."
            )
        return {
            name: float(value)
            for name, value in zip(self.FEATURE_NAMES, observation)
        }

    def _normalized_spectral_efficiency(self, sinr_db: float) -> float:
        clipped_sinr = float(
            np.clip(
                sinr_db,
                self.protocol_config.sinr_lower_clip,
                self.protocol_config.sinr_upper_clip,
            )
        )
        efficiency = self._spectral_efficiency(clipped_sinr)
        return float(np.clip(efficiency / self._max_spectral_efficiency, 0, 1))

    @staticmethod
    def _spectral_efficiency(sinr_db: float) -> float:
        return float(np.log2(1 + 10 ** (sinr_db / 10)))

    @staticmethod
    def _signed_normalize(value: float, clip: float) -> float:
        return float(np.clip(value / clip, -1.0, 1.0))

    @staticmethod
    def _unsigned_normalize(value: float, clip: float) -> float:
        return float(np.clip(value / clip, 0.0, 1.0))

    def _slope_db_per_s(
        self, tics: np.ndarray, values: np.ndarray
    ) -> float:
        if len(values) < 2 or tics[-1] == tics[0]:
            return 0.0
        time_s = (
            (tics - tics[0])
            * self.protocol_config.delta_t_ms
            / 1000
        )
        centered_time = time_s - np.mean(time_s)
        centered_values = values - np.mean(values)
        denominator = float(np.dot(centered_time, centered_time))
        if denominator == 0:
            return 0.0
        return float(np.dot(centered_time, centered_values) / denominator)
