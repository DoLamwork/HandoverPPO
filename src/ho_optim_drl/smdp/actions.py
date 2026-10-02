"""Actions available at an A3 decision point."""

from enum import IntEnum


class HandoverAction(IntEnum):
    """Reject the A3 event or hand over to one frozen candidate slot."""

    STAY = 0
    HO_C1 = 1
    HO_C2 = 2
    HO_C3 = 3
    HO_C4 = 4
    HO_C5 = 5
    HO_C6 = 6
    HO_C7 = 7
    HO_C8 = 8
    HO_C9 = 9
    HO_C10 = 10

    # Compatibility name for the A3 target used by reference policies.
    HANDOVER = HO_C1

    @property
    def is_handover(self) -> bool:
        return int(self) > 0

    @property
    def candidate_slot(self) -> int | None:
        return int(self) - 1 if self.is_handover else None
