"""Compatibility entry point; implementation belongs to ho_optim_drl.commands."""

import sys
from ho_optim_drl.commands import validate_3gpp as _implementation

if __name__ == "__main__":
    from ho_optim_drl.cli import run
    raise SystemExit(run(['validate_3gpp', *sys.argv[1:]]))
else:
    sys.modules[__name__] = _implementation
