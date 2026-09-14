"""Compatibility entry point; implementation belongs to ho_optim_drl.commands."""

import sys
from ho_optim_drl.commands import evaluate_smdp_ppo as _implementation

if __name__ == "__main__":
    from ho_optim_drl.cli import run
    raise SystemExit(run(['evaluate_ppo', *sys.argv[1:]]))
else:
    sys.modules[__name__] = _implementation
