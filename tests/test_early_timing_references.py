"""Correctness tests for shadow-A3 Rpost/Rearly lookahead references."""

import unittest

import numpy as np

from ho_optim_drl import Config
from ho_optim_drl.commands.audit_early_timing_references import (
    choose_schedule,
    simulate_schedule,
)
from ho_optim_drl.data import RadioTrace
from ho_optim_drl.protocol import EarlyTimingDecisionPoint, EarlyTimingHandoverEngine


def _trace(neighbor: np.ndarray) -> RadioTrace:
    samples = len(neighbor)
    rsrp = np.vstack((np.full(samples, -80.0), neighbor))
    # Cell 0 must be the initial serving cell selected by RRC cell search.
    sinr = np.vstack((np.full(samples, 5.0), np.full(samples, 4.0)))
    return RadioTrace.from_arrays(rsrp, sinr, delta_t_ms=10)


class EarlyTimingReferenceTests(unittest.TestCase):
    def test_delayed_early_plan_executes_at_shadow_maturity(self) -> None:
        engine = EarlyTimingHandoverEngine(
            Config(), enable_early_decisions=True, enable_shadow_decisions=True
        )
        engine.reset(_trace(np.full(30, -77.0)))
        engine.advance_one_tick()
        self.assertIsInstance(engine.decision_point, EarlyTimingDecisionPoint)

        outcome, ticks = simulate_schedule(
            engine, requested_delay_ticks=4, horizon_ticks=20, mode="early"
        )

        self.assertEqual(ticks, 20)
        self.assertEqual(outcome["first_action"], "WAIT_20")
        self.assertEqual(outcome["executed_tic"], 4)
        self.assertEqual(outcome["actual_delay_ms"], 40)
        self.assertFalse(outcome["request_cancelled"])
        # Deep-copy replay must not consume or mutate the source decision.
        self.assertIsInstance(engine.decision_point, EarlyTimingDecisionPoint)
        self.assertEqual(engine.decision_point.tic, 0)

    def test_delayed_request_is_cancelled_when_early_window_closes(self) -> None:
        engine = EarlyTimingHandoverEngine(
            Config(), enable_early_decisions=True, enable_shadow_decisions=True
        )
        engine.reset(
            _trace(np.array([-77.0, -77.0, -80.0, -80.0, -80.0, -80.0]))
        )
        engine.advance_one_tick()

        outcome, _ = simulate_schedule(
            engine, requested_delay_ticks=8, horizon_ticks=5, mode="early"
        )

        self.assertTrue(outcome["request_cancelled"])
        self.assertIsNone(outcome["executed_tic"])
        self.assertEqual(outcome["handover_attempts"], 0)

    def test_selector_retains_baseline_on_tie_or_risk(self) -> None:
        control = {
            "delivered_mbit": 1.0,
            "handover_failures": 0,
            "ping_pong_count": 0,
            "recovery_seconds": 0.0,
        }
        tied = {**control, "requested_delay_ticks": 2}
        risky = {
            **control,
            "delivered_mbit": 2.0,
            "handover_failures": 1,
            "requested_delay_ticks": 0,
        }

        choice = choose_schedule(control, [tied, risky])

        self.assertEqual(choice["kind"], "baseline")

    def test_selector_maximizes_safe_data_then_shortest_delay(self) -> None:
        control = {
            "delivered_mbit": 1.0,
            "handover_failures": 0,
            "ping_pong_count": 0,
            "recovery_seconds": 0.0,
        }
        candidates = [
            {**control, "delivered_mbit": 1.2, "requested_delay_ticks": 8},
            {**control, "delivered_mbit": 1.2, "requested_delay_ticks": 2},
        ]

        choice = choose_schedule(control, candidates)

        self.assertEqual(choice["kind"], "schedule")
        self.assertEqual(choice["delay_ticks"], 2)


if __name__ == "__main__":
    unittest.main()
