"""Actions available at an A3 decision point."""

from enum import IntEnum


class HandoverAction(IntEnum):
    """Binary action space for the first SMDP implementation."""

    STAY = 0
    HANDOVER = 1
