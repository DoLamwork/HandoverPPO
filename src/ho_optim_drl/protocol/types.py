"""Shared immutable values exposed by a handover protocol engine."""

from dataclasses import dataclass
from enum import Enum, auto

import numpy as np


class ProtocolPhase(Enum):
    """High-level phase of the UE handover procedure."""

    DISCONNECTED = auto()
    CONNECTED = auto()
    A3_TTT = auto()
    DECISION = auto()
    HO_PREPARATION = auto()
    HO_EXECUTION = auto()
    RLF_RECOVERY = auto()
    TERMINATED = auto()


@dataclass(frozen=True)
class DecisionPoint:
    """A latched A3 decision offered to a policy."""

    tic: int
    serving_cell: int
    target_cell: int
    a3_start_tic: int
    a3_margin_db: float


@dataclass(frozen=True)
class ProtocolSnapshot:
    """Protocol state available to observation and reward builders."""

    tic: int
    phase: ProtocolPhase
    serving_cell: int | None
    target_cell: int | None
    decision_point: DecisionPoint | None
    rsrp_a3_db: np.ndarray
    sinr_rate_db: np.ndarray
    sinr_rlm_db: np.ndarray
    n310_progress: float
    t310_active: bool
    t310_progress: float
    serving_residence_ticks: int
    time_since_last_ho_ticks: int
    previous_cell: int | None
    recent_ho_count: int


@dataclass(frozen=True)
class TickResult:
    """Outcome produced by one primitive radio timestep."""

    snapshot: ProtocolSnapshot
    delivered_rate_mbps: float
    best_rate_mbps: float
    handover_started: bool = False
    handover_succeeded: bool = False
    handover_failed: bool = False
    rlf_started: bool = False
    ping_pong_detected: bool = False
    terminated: bool = False
