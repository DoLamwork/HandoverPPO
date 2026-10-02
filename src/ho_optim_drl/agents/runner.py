"""Policy runner used before a trainable agent exists."""

import numpy as np

from ho_optim_drl.protocol import HandoverEngine, TickResult
from ho_optim_drl.smdp import HandoverAction, ObservationBuilder

from .policies import DecisionPolicy


def run_policy(
    engine: HandoverEngine,
    policy: DecisionPolicy,
    observation_builder: ObservationBuilder | None = None,
) -> list[TickResult]:
    """Run a policy to the end of the trace and return primitive tick results."""
    if observation_builder is not None:
        observation_builder.reset()

    results: list[TickResult] = []
    while True:
        if engine.decision_point is not None:
            observation = (
                observation_builder.build(engine.snapshot)
                if observation_builder is not None
                else np.empty(0, dtype=np.float32)
            )
            action = policy.predict(observation)
            if action == HandoverAction.STAY:
                engine.stay()
            elif action == HandoverAction.HO_C1:
                # Diagnostic reference policies always accept the A3 target.
                engine.start_handover()
            else:
                raise ValueError(
                    "run_policy has no candidate mapping; only STAY and HO_C1 "
                    f"are supported, got {action}."
                )

        tick = engine.advance_one_tick()
        if observation_builder is not None:
            observation_builder.observe(tick)
        results.append(tick)
        if tick.terminated:
            return results
