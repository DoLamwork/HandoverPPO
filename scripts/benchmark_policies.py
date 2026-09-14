"""Compatibility entry point; implementation belongs to ho_optim_drl.commands."""

import sys
from ho_optim_drl.commands import benchmark_policies as _implementation

if __name__ == "__main__":
    from ho_optim_drl.cli import run
    raise SystemExit(run(['benchmark', *sys.argv[1:]]))
else:
    sys.modules[__name__] = _implementation
