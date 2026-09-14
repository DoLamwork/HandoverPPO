# Policy Benchmark

The benchmark command evaluates two policies on the same trajectory split:

- `ppo`: deterministic action from the selected checkpoint;
- `3gpp_always_ho`: accept every completed A3 event;

```bash
python run.py benchmark \
  --checkpoint results/models/smdp_ppo/smdp_ppo_final.pt \
  --data-dir data/uav_100m_grid100m_dev_processed \
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

To evaluate a previously trained checkpoint on every trajectory from unseen
UE speeds without retraining, use the `all` split together with an explicit
speed filter:

```bash
python run.py benchmark \
  --checkpoint results/models/smdp_ppo/smdp_ppo_final.pt \
  --data-dir data/uav_processed_all \
  --split all \
  --speeds 120 150 \
  --output-dir results/benchmark/uav_120_150_zero_shot
```

This is a zero-shot benchmark: `--split all` evaluates every trajectory at the
selected speeds, and the supplied checkpoint is loaded for inference only.

Both policies use the same `HandoverSMDPEnv`, preprocessing, protocol timers,
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

The four headline evaluation metrics are reported in percent. Event counts are
retained in the output files for auditing, while the pooled percentages are
defined as:

```text
Relative rate (%) = 100 * mean achieved rate / upper-bound rate
HOF (%)           = 100 * total HOF / total HO attempts
RLF time (%)      = 100 * RLF-recovery time / total trace duration
PP (%)            = 100 * total ping-pong / total successful HO
```

Pooling the event counts and durations avoids averaging unstable
per-trajectory ratios with small denominators.

The per-trajectory and aggregate files also report the fractions of `STAY`,
`HANDOVER`, and `WAIT_50`. For a legacy two-action checkpoint, `WAIT_50` is
necessarily 0%; only a newly trained three-action checkpoint can use it.

## Current Test Result

The values below are a historical two-action result and must not be presented
as the result of the new `STAY/HANDOVER/WAIT_50` policy. Re-run the benchmark
after training a three-action checkpoint.

The current single-seed checkpoint on the 17 held-out test trajectories gives:

| Policy | Mean rate | Relative rate | RLF | PP | HOF | Controlled return |
|---|---:|---:|---:|---:|---:|---:|
| 3GPP Always-HO | 67.259 Mbps | 98.614% | 35 | 54 | 35 | 14.063 |
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
