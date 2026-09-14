> Previous HandoverOptimDRL guide. Current commands and the data-generator integration are documented in the root README.

# HandoverPPO

Trace-driven research framework for comparing an audited A3 handover reference
with a new event-triggered SMDP/PPO approach.

The PPO implementation from the original paper has been removed. It selected
an action every 10 ms and does not match the new decision model. The replacement
will be built incrementally on top of a shared primitive protocol engine.

## Current status

Available:

- CSV RSRP/SINR traces and deterministic preprocessing.
- Audited A3, TTT, RLM, RLF, handover, and ping-pong reference simulator.
- Action-controlled primitive `HandoverEngine` with A3 decision points.
- Causal 19-feature A3 observation builder.
- Primitive throughput/RLF/ping-pong reward with auditable breakdown.
- Decision-to-decision macro-transition builder with duration-aware discount.
- Gymnasium SMDP environment with seeded trajectory sampling.
- PyTorch PPO with duration-aware GAE, validation, and checkpointing.
- `AlwaysHandoverPolicy` and `AlwaysStayPolicy` for environment verification.

Not implemented yet:

- Hyperparameter tuning and final trained-model selection.

## Primitive engine

```python
from ho_optim_drl import Config
from ho_optim_drl.data import load_radio_trace
from ho_optim_drl.protocol import HandoverEngine

config = Config()
trace = load_radio_trace(
    config,
    data_dir="data/processed",
    rsrp_file="rsrp_30kmh_0.csv",
    sinr_file="sinr_30kmh_0.csv",
)
engine = HandoverEngine(config)
snapshot = engine.reset(trace)

while True:
    if engine.decision_point is not None:
        engine.start_handover()  # or engine.stay()

    tick = engine.advance_one_tick()
    if tick.terminated:
        break
```

`advance_one_tick()` refuses to continue while a decision is pending. A
rejected A3 event is suppressed until A3 leaving or a different target appears.

Observation details and normalization are documented in
[docs/observation.md](observation.md).

Reward and macro-transition semantics are documented in
[docs/reward.md](reward.md).

Gymnasium usage and the `info` contract are documented in
[docs/gymnasium.md](gymnasium.md).

PPO equations, defaults, and commands are documented in
[docs/ppo.md](ppo.md).

Matched policy benchmarking and metric aggregation are documented in
[docs/benchmark.md](benchmark.md).

See [docs/architecture.md](architecture.md) for the build order and
[docs/3gpp_baseline_audit.md](3gpp_baseline_audit.md) for standards
coverage and simulation assumptions.

## Installation

```bash
python -m pip install -e .
```

The implementation uses NumPy, SciPy, Gymnasium, and PyTorch. PPO is local to
this repository so its duration-aware GAE semantics remain explicit.

## Data

Runtime input is stored in `data/processed` as:

```text
sample,time_ms,bs_0,bs_1,bs_2,bs_3,bs_4
```

The processed CSV traces are versioned with the repository so a fresh clone
can train immediately. MATLAB source traces are intentionally excluded from
Git. If they are available locally, regenerate the CSV files with:

```bash
python scripts/convert_mat_to_csv.py \
  --input-dir path/to/mat/traces \
  --output-dir data/processed \
  --overwrite
```

## Validate the reference

```bash
python run.py validate_3gpp
```

To use the UAV simulator traces, normalize them into a separate directory
(the source files are left unchanged), then validate every UAV speed:

```bash
python scripts/convert_uav_csv.py \
  --input-dir data/UAV_sim \
  --output-dir data/uav_processed
python run.py validate_3gpp \
  --data-dir data/uav_processed \
  --speeds 30 50 70 90
```

Train PPO on the normalized UAV traces with a separate checkpoint directory:

```bash
python run.py train_ppo \
  --data-dir data/uav_processed \
  --output-dir results/models/uav_smdp_ppo
```

## Benchmark policies

After training or downloading a checkpoint:

```bash
python run.py benchmark \
  --checkpoint results/models/smdp_ppo/smdp_ppo_final.pt \
  --split test
```

This compares PPO with 3GPP Always-HO on the same held-out trajectories and
writes CSV/JSON tables to `results/benchmark`.

For a PPO model trained on normalized UAV traces, add
`--data-dir data/uav_processed` and choose a separate output directory, for
example `--output-dir results/benchmark/uav_smdp_ppo`.

## Test

```bash
python -m unittest discover -s tests -v
```

## Origin

The repository originated from the code accompanying:

> J. Voigt, P. J. Gu, and P. M. Rost, "A Deep Reinforcement Learning-Based
> Approach for Adaptive Handover Protocols," SCC 2025,
> doi:10.1109/IEEECONF62907.2025.10949111.

The current architecture deliberately does not preserve the paper's PPO
implementation or reported behavior.
