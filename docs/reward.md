# SMDP Reward and Macro-Transition

## Primitive reward

The reward model is evaluated once for every native 10 ms radio tick:

```text
rate_reward =
    delta_t_seconds * delivered_rate_mbps / rate_scale_mbps

tick_reward =
    rate_reward
    - rlf_penalty * I(RLF starts)
    - ping_pong_penalty * I(ping-pong is detected)
```

Default values are:

```text
rate_scale_mbps = 100
rlf_penalty = 5
ping_pong_penalty = 1
```

The rate term is normalized delivered data, not a second definition of data
rate. At 100 Mbps, one second of successful delivery contributes `+1`.
Therefore the default RLF and ping-pong penalties are directly interpretable
as fixed opportunity costs.

HO preparation/execution, HOF, disconnection, and RLF recovery already reduce
the reward because `delivered_rate_mbps` is zero while the UE cannot receive
data. An HOF starts RLF recovery in the current protocol model, so it also
receives the RLF event penalty.

## Macro reward

An action is selected only after A3 has completed TTT. The environment then
runs all primitive ticks internally until the next completed A3 event or the
end of the trace:

```text
macro_reward = sum(tick_reward)
```

There is no separate macro reward formula. `RewardBreakdown` preserves the
summed rate, RLF, and ping-pong components for analysis, while `.total`
provides the scalar used by reinforcement learning.

The prefix before the first A3 decision is uncontrollable and is not assigned
to an action. Its duration and diagnostic reward remain available as
`uncontrolled_duration_ticks` and `uncontrolled_reward`. The final transition
includes every remaining tick through the end of the trace.

A trace that never produces an A3 decision returns `None` from
`MacroTransitionBuilder.reset()`. Such a trace remains useful for baseline
evaluation but has no policy action and is skipped by the PPO
episode sampler.

## Duration-aware discount

Macro-steps have different durations. `MacroTransitionBuilder` stores:

```text
duration_seconds = duration_ticks * delta_t_seconds
discount = gamma_per_second ** duration_seconds
```

This makes discounting depend on elapsed physical time rather than on the
number of A3 decisions. The default `gamma_per_second` is `0.99`.
