"""Behavioral tests for the action-controlled handover engine."""

import unittest

import numpy as np

from ho_optim_drl import Config
from ho_optim_drl.agents import AlwaysHandoverPolicy, run_policy
from ho_optim_drl.data import RadioTrace
from ho_optim_drl.protocol import (
    A3ReferenceSimulator,
    HandoverEngine,
    ProtocolPhase,
)


def _constant_sinr(n_base_stations: int, n_samples: int) -> np.ndarray:
    sinr = np.full((n_base_stations, n_samples), -2.0)
    sinr[0, :] = 5.0
    return sinr


def _trace(
    rsrp: np.ndarray,
    sinr_rate: np.ndarray,
    sinr_rlm: np.ndarray | None = None,
    delta_t_ms: int = 10,
) -> RadioTrace:
    return RadioTrace.from_arrays(
        rsrp,
        sinr_rate,
        sinr_rlm_db=sinr_rlm,
        delta_t_ms=delta_t_ms,
    )


class DecisionTests(unittest.TestCase):
    def test_snapshot_handles_zero_t310_and_protects_trace_values(self) -> None:
        config = Config()
        config.t_t310_ms = 0
        rsrp = np.vstack((np.full(4, -80.0), np.full(4, -90.0)))
        sinr = _constant_sinr(2, 4)
        engine = HandoverEngine(config)
        snapshot = engine.reset(_trace(rsrp, sinr))

        rsrp[0, 0] = 123.0

        self.assertEqual(snapshot.t310_progress, 0.0)
        self.assertEqual(engine.snapshot.rsrp_a3_db[0], -80.0)
        with self.assertRaises(ValueError):
            engine.snapshot.rsrp_a3_db[0] = 0.0

    def test_engine_pauses_until_pending_decision_is_applied(self) -> None:
        rsrp = np.vstack(
            (
                np.full(12, -80.0),
                np.full(12, -77.0),
            )
        )
        engine = HandoverEngine(Config())
        engine.reset(_trace(rsrp, _constant_sinr(2, 12)))

        for _ in range(5):
            tick = engine.advance_one_tick()

        self.assertEqual(tick.snapshot.phase, ProtocolPhase.DECISION)
        self.assertEqual(engine.decision_point.serving_cell, 0)
        self.assertEqual(engine.decision_point.target_cell, 1)
        self.assertAlmostEqual(engine.decision_point.a3_margin_db, 1.0)
        with self.assertRaises(RuntimeError):
            engine.advance_one_tick()

    def test_stay_suppresses_same_event_until_a3_rearms(self) -> None:
        rsrp = np.vstack(
            (
                np.full(16, -80.0),
                np.array(
                    [-77.0] * 8
                    + [-81.0]
                    + [-77.0] * 7
                ),
            )
        )
        engine = HandoverEngine(Config())
        engine.reset(_trace(rsrp, _constant_sinr(2, 16)))

        for _ in range(5):
            engine.advance_one_tick()
        first_decision_tic = engine.decision_point.tic
        engine.stay()

        for _ in range(8):
            tick = engine.advance_one_tick()
            self.assertNotEqual(tick.snapshot.phase, ProtocolPhase.DECISION)

        tick = engine.advance_one_tick()
        self.assertEqual(tick.snapshot.phase, ProtocolPhase.DECISION)
        self.assertEqual(first_decision_tic, 4)
        self.assertEqual(engine.decision_point.tic, 13)

class HandoverTests(unittest.TestCase):
    def test_handover_uses_latched_target_when_strongest_cell_changes(self) -> None:
        config = Config()
        config.t_ho_prep_ms = 20
        config.t_ho_exec_ms = 10

        rsrp = np.vstack(
            (
                np.full(12, -80.0),
                np.full(12, -77.0),
                np.array([-90.0] * 5 + [-70.0] * 7),
            )
        )
        sinr = np.vstack(
            (
                np.full(12, 5.0),
                np.full(12, 2.0),
                np.array([-10.0] * 5 + [8.0] * 7),
            )
        )
        engine = HandoverEngine(config)
        engine.reset(_trace(rsrp, sinr))

        for _ in range(5):
            engine.advance_one_tick()
        engine.start_handover()

        self.assertEqual(engine.snapshot.phase, ProtocolPhase.HO_PREPARATION)
        self.assertEqual(engine.snapshot.target_cell, 1)

        results = [engine.advance_one_tick() for _ in range(3)]

        self.assertEqual(engine.rrc.pcell, 1)
        self.assertTrue(results[-1].handover_succeeded)
        self.assertEqual(engine.successful_ho_history, [(7, 0, 1)])

    def test_execution_outage_has_zero_delivered_rate(self) -> None:
        config = Config()
        config.t_ho_prep_ms = 10
        config.t_ho_exec_ms = 20
        rsrp = np.vstack((np.full(12, -80.0), np.full(12, -77.0)))
        sinr = _constant_sinr(2, 12)
        engine = HandoverEngine(config)
        engine.reset(_trace(rsrp, sinr))

        for _ in range(5):
            engine.advance_one_tick()
        engine.start_handover()

        prep_end = engine.advance_one_tick()
        execution = engine.advance_one_tick()

        self.assertEqual(prep_end.snapshot.phase, ProtocolPhase.HO_EXECUTION)
        self.assertEqual(prep_end.delivered_rate_mbps, 0.0)
        self.assertEqual(execution.delivered_rate_mbps, 0.0)
        self.assertGreater(execution.best_rate_mbps, 0.0)


class ReferenceParityTests(unittest.TestCase):
    def test_always_handover_matches_reference_tick_for_tick(self) -> None:
        config = Config()
        trace = _trace(
            np.vstack(
                (
                    np.array([-80.0] * 10 + [-75.0] * 10),
                    np.array([-77.0] * 10 + [-80.0] * 10),
                )
            ),
            np.vstack(
                (
                    np.array([5.0] * 10 + [8.0] * 10),
                    np.array([2.0] * 10 + [0.0] * 10),
                )
            ),
        )

        reference = A3ReferenceSimulator(config)
        reference.predict(trace)

        engine = HandoverEngine(config)
        engine.reset(trace)
        results = run_policy(engine, AlwaysHandoverPolicy())

        self.assertEqual(len(results), trace.num_samples)
        np.testing.assert_equal(
            np.asarray(engine.bs_idxs),
            np.asarray(reference.bs_idxs),
        )
        self.assertEqual(engine.ho_idxs, reference.ho_idxs)
        self.assertEqual(engine.rlfr_idxs, reference.rlfr_idxs)
        self.assertEqual(
            engine.successful_ho_history,
            reference.successful_ho_history,
        )
        self.assertEqual(engine.ping_pong_idxs, reference.ping_pong_idxs)
        for counter in ("tttc", "ho_prep", "ho_exec", "rlfr"):
            self.assertEqual(
                engine.cntr[counter].start_idxs,
                reference.cntr[counter].start_idxs,
            )
            self.assertEqual(
                engine.cntr[counter].done_idxs,
                reference.cntr[counter].done_idxs,
            )
            self.assertEqual(
                engine.cntr[counter].error_idxs,
                reference.cntr[counter].error_idxs,
            )

    def test_rate_and_rlm_use_separate_sinr_streams(self) -> None:
        rsrp = np.vstack((np.full(6, -80.0), np.full(6, -90.0)))
        sinr_rate = np.vstack((np.full(6, 10.0), np.full(6, -10.0)))
        sinr_rlm = np.vstack(
            (
                np.array([0.0, -9.0, -9.0, -9.0, -9.0, -9.0]),
                np.full(6, -10.0),
            )
        )
        engine = HandoverEngine(Config())
        engine.reset(_trace(rsrp, sinr_rate, sinr_rlm))

        engine.advance_one_tick()
        tick = engine.advance_one_tick()

        self.assertGreater(tick.delivered_rate_mbps, 30.0)
        self.assertEqual(engine.cntr["n310"].cnt, 1)
        self.assertEqual(tick.snapshot.sinr_rate_db[0], 10.0)
        self.assertEqual(tick.snapshot.sinr_rlm_db[0], -9.0)


if __name__ == "__main__":
    unittest.main()
