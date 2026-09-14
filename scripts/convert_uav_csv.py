"""Compatibility entry point; implementation belongs to ho_optim_drl.commands."""

import sys
from ho_optim_drl.commands import convert_uav_csv as _implementation

if __name__ == "__main__":
    raise SystemExit(_implementation.main())
else:
    sys.modules[__name__] = _implementation
