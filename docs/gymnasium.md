# Gymnasium SMDP Environment

`HandoverSMDPEnv` exposes the event-triggered handover process through the
standard Gymnasium API.

## Spaces

```text
action_space      = Discrete(2)
0                 = STAY
1                 = HANDOVER
observation_space = Box(shape=(19,), dtype=float32)
```

An observation is returned only after A3 has remained valid for TTT. Calling
`step(action)` executes that action and advances the protocol internally until
the next completed A3 event or the end of the trajectory.

## Dataset setup

Use complete trajectory splits so samples from one route never appear in
multiple datasets:

```python
from ho_optim_drl import Config, HandoverSMDPEnv
from ho_optim_drl.data import (
    discover_trace_pairs,
    stratified_trajectory_split,
)

pairs = discover_trace_pairs("data/processed")
splits = stratified_trajectory_split(pairs, seed=17)

env = HandoverSMDPEnv(
    Config(),
    data_dir="data/processed",
    trace_pairs=splits.train,
)
```

The environment loads CSV trajectories lazily. It can also receive a sequence
of in-memory `RadioTrace` objects through `traces=`.

## API

```python
observation, info = env.reset(seed=123)

terminated = False
truncated = False
while not (terminated or truncated):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
```

`reward` is the scalar `RewardBreakdown.total`. The `info` dictionary retains:

- rate, RLF, and ping-pong reward components;
- macro duration in ticks and seconds;
- continuous-time discount;
- current and next A3 decision metadata;
- cumulative episode return, macro-step count, and duration;
- final protocol metrics when the trace terminates.

`reset(options={"trace_index": i})` or
`reset(options={"trace_id": "30kmh_0"})` selects a specific trajectory for
evaluation. Normal seeded resets sample trajectories uniformly.

Trajectories without a completed A3 event contain no policy action. Random
resets skip them and report their IDs in `skipped_no_decision_traces`.
Explicitly requesting one raises `RuntimeError`.

At trace termination the returned observation is a zero vector inside the
declared observation space. The transition has `terminated=True`, so PPO must
not bootstrap its value.
