# Project Architecture

The project is split into layers so the SMDP environment can be verified
before a PPO implementation is introduced.

```text
src/ho_optim_drl/
|-- config.py
|-- data/
|   |-- loader.py
|   |-- split.py
|   `-- trace.py
|-- evaluation/
|   `-- benchmark.py
|-- protocol/
|   |-- primitives.py
|   |-- reference.py
|   |-- engine.py
|   `-- types.py
|-- smdp/
|   |-- actions.py
|   |-- observation.py
|   |-- reward.py
|   `-- transition.py
|-- agents/
|   |-- policies.py
|   `-- ppo/
|       |-- config.py
|       |-- network.py
|       |-- rollout.py
|       `-- trainer.py
`-- utils.py
```

## Layer responsibilities

- `data`: CSV validation, trajectory splitting, explicit radio streams, and
  RSRP L3 filtering.
- `protocol`: radio timers, A3, handover, RLF, and the audited reference.
- `smdp`: action, observation, reward, and macro-transition contracts.
- `agents`: policies that consume observations. Protocol logic does not belong here.
- `evaluation`: matched-trajectory policy benchmarking and aggregation.

## Engine invariants

- `reset()` accepts a `RadioTrace`; all streams use
  `(base_stations, timesteps)` layout.
- A3 uses filtered RSRP, rate uses raw SINR, and RLM uses its explicit SINR
  stream.
- A completed A3+TTT event changes the phase to `DECISION`.
- Time cannot advance until `stay()` or `start_handover()` is called.
- `start_handover()` latches the decision target through preparation/execution.
- `stay()` suppresses the same serving-target A3 event until it leaves.
- `AlwaysHandoverPolicy` is tick-for-tick identical to the audited reference.

## Build order

1. [Done] Implement the primitive `HandoverEngine`.
2. [Done] Verify `AlwaysHandoverPolicy` reproduces `A3ReferenceSimulator`.
3. [Done] Verify `AlwaysStayPolicy`, event suppression, and target latching.
4. [Done] Implement concrete observation and reward builders.
5. [Done] Implement and verify decision-to-decision macro-transitions.
6. [Done] Implement and verify the Gymnasium macro-step environment.
7. [Done] Add duration-aware GAE and rollout storage.
8. [Done] Add clipped SMDP PPO, evaluation, and checkpointing.
9. [Done] Add matched PPO/3GPP/Stay benchmark tables.
10. Tune reward and PPO hyperparameters on validation trajectories.
11. Repeat training across seeds and report confidence intervals.

The removed PPO implementation acted every 10 ms and is deliberately not used
as the foundation of this architecture.
