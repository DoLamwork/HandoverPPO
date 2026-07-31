# Policy Benchmark

The benchmark command evaluates three policies on the same trajectory split:

- `ppo`: deterministic action from the selected checkpoint;
- `3gpp_always_ho`: accept every completed A3 event;
- `always_stay`: reject every completed A3 event.

```bash
python run.py benchmark \
  --checkpoint results/models/smdp_ppo/smdp_ppo_final.pt \
  --data-dir data/processed \
  --split test \
  --split-seed 17
```

For a model trained on normalized UAV traces, use that dataset and a separate
output directory:

```bash
python run.py benchmark \
  --checkpoint results/models/smdp_ppo/smdp_ppo_final.pt \
  --data-dir data/uav_processed \
  --split test \
  --output-dir results/benchmark/uav_smdp_ppo
```

All policies use the same `HandoverSMDPEnv`, preprocessing, protocol timers,
reward model, and trajectory order. Protocol metrics cover the complete trace.
`controlled_return` matches the PPO training objective and begins at the first
A3 decision. `full_trace_return` also includes the policy-independent prefix
before that decision.

Outputs are written to `results/benchmark`:

```text
per_trajectory.csv
aggregate.csv
by_speed.csv
paired_comparison.csv
summary.json
```

## Aggregation

`mean_estimated_rate_mbps` is the arithmetic mean across trajectories and
matches PPO evaluation output. `duration_weighted_rate_mbps` weights each
trajectory by its physical duration.

Event rates are pooled:

```text
HOF rate = total HOF / total HO attempts
PP rate  = total ping-pong / total successful HO
RLF/100s = 100 * total RLF / total trace duration in seconds
```

This avoids averaging unstable per-trajectory ratios with small denominators.

## Current Test Result

The current single-seed checkpoint on the 17 held-out test trajectories gives:

| Policy | Mean rate | Relative rate | RLF | PP | HOF | Controlled return |
|---|---:|---:|---:|---:|---:|---:|
| 3GPP Always-HO | 67.259 Mbps | 98.614% | 35 | 54 | 35 | 14.063 |
| Always-Stay | 65.028 Mbps | 94.734% | 53 | 0 | 0 | 10.749 |
| PPO | 66.498 Mbps | 97.084% | 27 | 16 | 11 | 18.293 |

Relative to 3GPP Always-HO, PPO changes:

```text
mean rate          -0.761 Mbps
RLF                -8
ping-pong          -38
HOF                -24
controlled return  +4.230
```

The result is not uniform across speeds. PPO performs strongly at 30 and
50 km/h, trades some rate for reliability at 90 km/h, and performs worse on
RLF at 70 km/h. It is therefore evidence of a useful learned trade-off, not a
final claim that PPO dominates the baseline. Multiple training seeds and
confidence intervals are still required.
