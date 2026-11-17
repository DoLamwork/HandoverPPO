"""Trace-driven handover protocol components."""

from .engine import HandoverEngine
from .early_timing import (
    EarlyTimingDecisionPoint,
    EarlyTimingHandoverEngine,
    ShadowA3DecisionPoint,
)
from .reference import A3ReferenceSimulator
from .types import DecisionPoint, ProtocolPhase, ProtocolSnapshot, TickResult

__all__ = [
    "A3ReferenceSimulator",
    "DecisionPoint",
    "EarlyTimingDecisionPoint",
    "EarlyTimingHandoverEngine",
    "HandoverEngine",
    "ProtocolPhase",
    "ProtocolSnapshot",
    "ShadowA3DecisionPoint",
    "TickResult",
]
