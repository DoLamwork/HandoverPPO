"""Behavioral tests for the opt-in B2/B3 early timing adapter."""

import unittest

import numpy as np

from ho_optim_drl import Config
from ho_optim_drl.agents import AlwaysHandoverPolicy, run_policy
from ho_optim_drl.data import RadioTrace
from ho_optim_drl.protocol import (
    EarlyTimingDecisionPoint,
    EarlyTimingHandoverEngine,
    ProtocolPhase,
    ShadowA3DecisionPoint,
    A3ReferenceSimulator,
)


def _trace(neighbor_rsrp: np.ndarray, *, third_rsrp: np.ndarray | None = None) -> RadioTrace:
    samples = len(neighbor_rsrp)
    rows = [np.full(samples, -80.0), np.asarray(neighbor_rsrp, dtype=np.float64)]
    if third_rsrp is not None:
        rows.append(np.asarray(third_rsrp, dtype=np.float64))
    sinr = np.full((len(rows), samples), 0.0)
    sinr[0] = 5.0
    return RadioTrace.from_arrays(np.vstack(rows), sinr, delta_t_ms=10)


class EarlyTimingEngineTests(unittest.TestCase):
    def test_entry_is_strictly_above_one_db_and_is_not_a3_decision(self) -> None:
        engine = EarlyTimingHandoverEngine(Config())
        engine.reset(_trace(np.array([-79.0, -78.99, -78.99])))

        first = engine.advance_one_tick()
        second = engine.advance_one_tick()

        self.assertIsNone(first.snapshot.decision_point)
        self.assertIsInstance(second.snapshot.decision_point, EarlyTimingDecisionPoint)
        self.assertEqual(second.snapshot.decision_point.trigger, "early_window")
        self.assertEqual(second.snapshot.decision_point.stage, "entry")
        self.assertFalse(engine.cntr["tttc"].pending)

    def test_b2_starts_handover_immediately(self) -> None:
        engine = EarlyTimingHandoverEngine(Config())
        engine.reset(_trace(np.full(12, -78.5)))
        engine.advance_one_tick()
        decision = engine.decision_point

        engine.start_handover()

        self.assertIsInstance(decision, EarlyTimingDecisionPoint)
        self.assertEqual(engine.snapshot.phase, ProtocolPhase.HO_PREPARATION)
        self.assertEqual(engine.snapshot.target_cell, 1)
        self.assertEqual(engine.early_audit_state()["executions"], 1)

        while True:
            tick = engine.advance_one_tick()
            self.assertIsNone(engine.decision_point)
            if tick.handover_succeeded or tick.terminated:
                break
        self.assertTrue(tick.handover_succeeded)

    def test_b3_reoffers_after_four_ticks_and_then_executes(self) -> None:
        engine = EarlyTimingHandoverEngine(Config())
        engine.reset(_trace(np.full(12, -78.5)))
        engine.advance_one_tick()
        engine.defer_early_handover(delay_ticks=4)

        for _ in range(3):
            engine.advance_one_tick()
            self.assertIsNone(engine.decision_point)
        engine.advance_one_tick()

        self.assertIsInstance(engine.decision_point, EarlyTimingDecisionPoint)
        self.assertEqual(engine.decision_point.tic, 4)
        self.assertEqual(engine.decision_point.stage, "scheduled")
        self.assertEqual(engine.decision_point.scheduled_delay_ticks, 4)
        engine.start_handover()
        self.assertEqual(engine.early_audit_state()["executions"], 1)

    def test_b3_cancels_when_window_closes_before_due_tick(self) -> None:
        engine = EarlyTimingHandoverEngine(Config())
        engine.reset(_trace(np.array([-78.5, -79.5, -80.0, -80.0, -80.0, -80.0])))
        engine.advance_one_tick()
        engine.defer_early_handover(delay_ticks=4)

        while not engine.advance_one_tick().terminated:
            self.assertIsNone(engine.decision_point)

        state = engine.early_audit_state()
        self.assertEqual(state["schedule_cancellations"], 1)
        self.assertEqual(state["executions"], 0)

    def test_target_change_cancels_old_schedule_and_opens_new_episode(self) -> None:
        engine = EarlyTimingHandoverEngine(Config())
        engine.reset(
            _trace(
                np.array([-78.5, -78.5, -90.0, -90.0, -90.0, -90.0]),
                third_rsrp=np.array([-90.0, -90.0, -78.0, -78.0, -78.0, -78.0]),
            )
        )
        engine.advance_one_tick()
        engine.defer_early_handover(delay_ticks=4)
        engine.advance_one_tick()
        engine.advance_one_tick()

        self.assertIsInstance(engine.decision_point, EarlyTimingDecisionPoint)
        self.assertEqual(engine.decision_point.target_cell, 2)
        self.assertEqual(engine.decision_point.stage, "entry")
        state = engine.early_audit_state()
        self.assertEqual(state["schedule_cancellations"], 1)
        self.assertEqual(state["target_changes"], 1)
        self.assertEqual(state["entries"], 2)

    def test_shadow_only_immediate_acceptance_matches_b0(self) -> None:
        trace = _trace(
            np.r_[np.full(20, -77.0), np.full(20, -82.0), np.full(20, -77.0)]
        )
        reference = A3ReferenceSimulator(Config())
        reference.predict(trace)
        shadow = EarlyTimingHandoverEngine(
            Config(), enable_early_decisions=False, enable_shadow_decisions=True
        )
        shadow.reset(trace)
        run_policy(shadow, AlwaysHandoverPolicy())

        np.testing.assert_equal(shadow.bs_idxs, reference.bs_idxs)
        self.assertEqual(shadow.ho_idxs, reference.ho_idxs)
        self.assertEqual(shadow.rlfr_idxs, reference.rlfr_idxs)
        self.assertEqual(shadow.successful_ho_history, reference.successful_ho_history)
        self.assertEqual(shadow.ping_pong_idxs, reference.ping_pong_idxs)
        for name in ("n310", "n311", "t310", "rlfr", "ho_prep", "ho_exec"):
            self.assertEqual(shadow.cntr[name].start_idxs, reference.cntr[name].start_idxs)
            self.assertEqual(shadow.cntr[name].done_idxs, reference.cntr[name].done_idxs)
            self.assertEqual(shadow.cntr[name].error_idxs, reference.cntr[name].error_idxs)
        self.assertEqual(
            shadow._shadow_ttt.start_idxs, reference.cntr["tttc"].start_idxs
        )
        self.assertEqual(
            shadow._shadow_ttt.terminated_idxs,
            reference.cntr["tttc"].terminated_idxs,
        )
        self.assertEqual(
            [item[3] for item in shadow.shadow_maturities],
            reference.cntr["ho_prep"].start_idxs,
        )

    def test_shadow_maturity_wins_over_same_tick_early_redecision(self) -> None:
        engine = EarlyTimingHandoverEngine(
            Config(), enable_early_decisions=True, enable_shadow_decisions=True
        )
        engine.reset(_trace(np.full(12, -77.0)))
        engine.advance_one_tick()
        self.assertIsInstance(engine.decision_point, EarlyTimingDecisionPoint)
        engine.wait_for_redecision(delay_ticks=2)
        engine.advance_one_tick()
        engine.advance_one_tick()
        self.assertIsInstance(engine.decision_point, EarlyTimingDecisionPoint)
        engine.wait_for_redecision(delay_ticks=2)
        engine.advance_one_tick()
        engine.advance_one_tick()

        self.assertIsInstance(engine.decision_point, ShadowA3DecisionPoint)
        self.assertEqual(engine.decision_point.tic, 4)
        self.assertEqual(engine.decision_point.stage, "maturity")
        engine.wait_for_redecision(delay_ticks=2)
        engine.advance_one_tick()
        self.assertIsNone(engine.decision_point)
        engine.advance_one_tick()
        self.assertIsInstance(engine.decision_point, ShadowA3DecisionPoint)
        self.assertEqual(engine.decision_point.tic, 6)
        self.assertEqual(engine.decision_point.stage, "redecision")
        self.assertEqual(engine.decision_point.scheduled_delay_ticks, 2)


if __name__ == "__main__":
    unittest.main()
