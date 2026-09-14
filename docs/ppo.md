# Duration-Aware SMDP PPO

The trainable policy uses a local PyTorch PPO implementation. It does not use
a conventional fixed-step PPO wrapper because one handover macro-step can last
from a few milliseconds to the remainder of a trajectory.

## Network

The default actor-critic has:

```text
input             19 normalized A3 observation features
shared encoder    Linear(19, 64) -> Tanh -> Linear(64, 64) -> Tanh
policy head       3 logits for STAY, HANDOVER, and WAIT_50
value head        1 scalar
```

Policy actions are sampled from a categorical distribution during training and
chosen by maximum probability during deterministic evaluation.

`WAIT_50` defers the current A3 opportunity by exactly 50 ms (five radio ticks)
and exposes it again only while the same serving-target event remains valid.
This gives the policy a short, explicit observation window between immediate
handover and the longer event suppression caused by `STAY`.

## Duration-aware GAE

For macro-step `t`, the environment supplies:

```text
gamma_t = gamma_per_second ** duration_seconds_t
lambda_t = lambda_per_second ** duration_seconds_t
```

The temporal-difference residual and advantage are:

```text
delta_t =
    reward_t
    + gamma_t * (1 - terminated_t) * V(next_observation_t)
    - V(observation_t)

advantage_t =
    delta_t
    + gamma_t
    * lambda_t
    * (1 - episode_end_t)
    * advantage_(t+1)
```

`episode_end` is true for both termination and truncation. A truncation can
bootstrap its own TD residual but cannot propagate advantage from the next
reset trajectory. A true trace termination never bootstraps.

## PPO update

The optimizer uses the standard clipped probability-ratio objective, value
loss, entropy bonus, gradient clipping, advantage normalization, and optional
early stopping by approximate KL divergence. Rollouts can span several
trajectories; terminal masks prevent GAE leakage between them.

The initial defaults are research starting points, not tuned results:

```text
rollout macro-steps  512
minibatch size       64
update epochs        10
learning rate        3e-4
clip range           0.2
gamma per second     0.99
lambda per second    0.95
entropy coefficient  0.01
```

## Training

```bash
python run.py train_ppo --total-macro-steps 100000
```

To train on normalized UAV traces instead of the default `data/processed`
dataset, select their directory explicitly:

```bash
python run.py train_ppo \
  --data-dir data/uav_processed \
  --total-macro-steps 100000 \
  --output-dir results/models/uav_smdp_ppo
```

Use `--speeds` to restrict every training-stage split to selected speed
strata. For example, the following keeps 120 and 150 km/h completely unseen
during training and checkpoint selection:

```bash
python run.py train_ppo \
  --data-dir data/uav_300m_processed \
  --speeds 30 50 70 90 \
  --total-macro-steps 100000 \
  --output-dir results/models/uav_300m_ppo_30_90
```

That four-speed example creates a deterministic trajectory split with seed 17,
trains on 87 trajectories, selects the best checkpoint on 20 validation
trajectories, and reports final metrics on 17 held-out test trajectories.

The split fractions are configurable. With 60 independent trajectories at
each selected speed, this command creates exactly 36 training, 12 validation,
and 12 test trajectories per speed:

```bash
python run.py train_ppo \
  --data-dir data/uav_100m_60_processed \
  --speeds 30 50 70 90 120 150 \
  --train-fraction 0.60 \
  --validation-fraction 0.20 \
  --test-fraction 0.20 \
  --split-seed 17 \
  --total-macro-steps 100000 \
  --device cuda \
  --output-dir results/models/uav_100m_ppo_all_speeds_36_12_12
```

Outputs are written to `results/models/smdp_ppo`:

```text
smdp_ppo_best.pt
smdp_ppo_best_return.pt
smdp_ppo_best_qos.pt
smdp_ppo_best_rlf.pt
smdp_ppo_latest.pt
smdp_ppo_final.pt
validation_3gpp_reference.json
dataset_split_manifest.json
dataset_split_manifest.csv
training_metrics.jsonl
training_report.json
checkpoints/
  all/                    # one model per validation point
  pareto/                 # current non-dominated models
  validation_metrics.csv
  validation_metrics.json
  pareto_front.csv
  pareto_front.json
```

Before training, the command evaluates 3GPP Always-HO on the validation split.
By default, a checkpoint passes QoS only when its aggregate rate and its rate at
every selected speed are at least 98% of that reference. Change the floors with
`--qos-rate-ratio` and `--qos-rate-ratio-per-speed`.

`best_return` has the highest validation return. `best_rlf` minimizes RLF time,
then HOF, ping-pong, and HO attempts. `best_qos` uses the same reliability order
but only among checkpoints that pass both rate floors. `smdp_ppo_best.pt` remains
an alias of `best_return` for compatibility. `final` is selected from `best_qos`,
falling back to `best_return` if no checkpoint passes QoS; held-out test data
never selects the model.

The Pareto set maximizes rate while minimizing RLF time, HOF, ping-pong, and HO
attempts. The full validation history remains in `checkpoints/all`, so removing a
model from the current Pareto set never destroys its original checkpoint.

Training prints live collection speed, episode return, PPO losses, entropy,
KL divergence, clipping fraction, explained variance, and validation metrics.
The same events are appended immediately to `training_metrics.jsonl`, so
partial logs survive an interrupted run.

Every new training run also writes an auditable dataset manifest. It records
the exact train, validation, and test RSRP/SINR filenames, file sizes, and
SHA-256 hashes. The manifest hash is embedded in every saved checkpoint.
Each `update_complete` record in `training_metrics.jsonl` and each item in the
report `history` additionally contains:

- `training_trace_macro_steps`: macro-step counts by trace for that update;
- `training_trace_segments`: run-length encoded trace order for the rollout;
- `completed_episode_trace_ids`: completed training episodes in order.

The report-level `training_trace_usage` aggregates macro-step and completed
episode counts over the full run. Together these fields distinguish the fixed
split membership from the trajectories actually sampled during optimization.

The default collection update is printed every 50 macro-steps. Change it with:

```bash
python run.py train_ppo \
  --total-macro-steps 100000 \
  --progress-interval 10
```

Useful overrides:

```bash
python run.py train_ppo \
  --total-macro-steps 200000 \
  --rollout-steps 1024 \
  --batch-size 128 \
  --device cuda
```

## Evaluation

```bash
python run.py evaluate_ppo \
  --checkpoint results/models/smdp_ppo/smdp_ppo_final.pt \
  --split test
```

Evaluation is deterministic and visits each selected trajectory at most once.
Reported return excludes the uncontrollable prefix before the first A3
decision. Estimated rate, RLF, and ping-pong metrics still cover the complete
protocol trajectory.
