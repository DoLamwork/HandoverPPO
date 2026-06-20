# Duration-Aware SMDP PPO

The trainable policy uses a local PyTorch PPO implementation. It does not use
a conventional fixed-step PPO wrapper because one handover macro-step can last
from a few milliseconds to the remainder of a trajectory.

## Network

The default actor-critic has:

```text
input             19 normalized A3 observation features
shared encoder    Linear(19, 64) -> Tanh -> Linear(64, 64) -> Tanh
policy head       2 logits for STAY and HANDOVER
value head        1 scalar
```

Policy actions are sampled from a categorical distribution during training and
chosen by maximum probability during deterministic evaluation.

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

The command creates the deterministic trajectory split with seed 17, trains on
87 trajectories, selects the best checkpoint on 20 validation trajectories,
and reports final metrics on 17 held-out test trajectories.

Outputs are written to `results/models/smdp_ppo`:

```text
smdp_ppo_best.pt
smdp_ppo_latest.pt
smdp_ppo_final.pt
training_metrics.jsonl
training_report.json
```

`best` is selected only by validation mean return. `latest` stores the final
optimizer state. `final` is a copy of the selected checkpoint annotated with
validation and held-out test metrics; test data never selects the model.

Training prints live collection speed, episode return, PPO losses, entropy,
KL divergence, clipping fraction, explained variance, and validation metrics.
The same events are appended immediately to `training_metrics.jsonl`, so
partial logs survive an interrupted run.

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
