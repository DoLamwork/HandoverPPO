"""Compatibility entry point; implementation belongs to ho_optim_drl.commands."""

import sys
from ho_optim_drl.commands import plot_relative_rate_pdf as _implementation

if __name__ == "__main__":
    raise SystemExit(_implementation.main())
else:
    sys.modules[__name__] = _implementation
