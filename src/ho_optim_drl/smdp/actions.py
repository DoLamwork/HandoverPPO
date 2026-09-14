"""Actions available at an A3 decision point."""

from enum import IntEnum


class HandoverAction(IntEnum):
    """Actions available after one A3 event completes its TTT."""

    STAY = 0
    HANDOVER = 1
    WAIT_50 = 2
