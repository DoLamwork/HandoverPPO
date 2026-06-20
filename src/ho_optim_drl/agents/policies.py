"""Small policies used to verify the SMDP environment before PPO."""

from typing import Protocol

import numpy as np

from ho_optim_drl.smdp.actions import HandoverAction


class DecisionPolicy(Protocol):
    """Policy contract independent of any RL library."""

    def predict(self, observation: np.ndarray) -> HandoverAction:
        """Choose one action at a decision point."""


class AlwaysHandoverPolicy:
    """Reference policy that accepts every completed A3 event."""

    def predict(self, observation: np.ndarray) -> HandoverAction:
        del observation
        return HandoverAction.HANDOVER


class AlwaysStayPolicy:
    """Diagnostic policy that rejects every completed A3 event."""

    def predict(self, observation: np.ndarray) -> HandoverAction:
        del observation
        return HandoverAction.STAY
