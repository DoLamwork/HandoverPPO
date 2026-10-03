"""Tests for the causal A3 observation builder."""

import unittest

import numpy as np

from ho_optim_drl import Config
from ho_optim_drl.agents import run_policy
from ho_optim_drl.data import RadioTrace
from ho_optim_drl.protocol import (
    DecisionPoint,
    HandoverEngine,
    ProtocolPhase,
    ProtocolSnapshot,
    TickResult,
)
from ho_optim_drl.smdp import (
    A3ObservationBuilder,
    A3ObservationConfig,
    HandoverAction,
)


def _run_to_decision(
    rsrp: np.ndarray,
    sinr: np.ndarray,
) -> tuple[HandoverEngine, A3ObservationBuilder, np.ndarray]:
    config = Config()
    engine = HandoverEngine(config)
    builder = A3ObservationBuilder(config)
    engine.reset(RadioTrace.from_arrays(rsrp, sinr))
    builder.reset()

    while engine.decision_point is None:
        tick = engine.advance_one_tick()
        builder.observe(tick)

    return engine, builder, builder.build(engine.snapshot)


class ObservationValueTests(unittest.TestCase):
    def test_known_a3_ramp_produces_expected_features(self) -> None:
        rsrp = np.vstack(
            (
                np.full(8, -80.0),
                np.array([-77.0, -76.0, -75.0, -74.0, -73.0, -73.0, -73.0, -73.0]),
            )
        )
        sinr = np.vstack(
            (
                np.array([5.0, 4.0, 3.0, 2.0, 1.0, 1.0, 1.0, 1.0]),
                np.array([-2.0, -1.0, 0.0, 1.0, 2.0, 2.0, 2.0, 2.0]),
            )
        )
        _, builder, observation = _run_to_decision(rsrp, sinr)
        values = builder.as_dict(observation)

        self.assertEqual(observation.shape, (19,))
        self.assertEqual(observation.dtype, np.float32)
        self.assertTrue(np.all(observation >= builder.low))
        self.assertTrue(np.all(observation <= builder.high))
        self.assertAlmostEqual(values["a3_margin_now"], 0.5, places=6)
        self.assertAlmostEqual(values["a3_margin_mean"], 0.3, places=6)
        self.assertAlmostEqual(values["a3_margin_min"], 0.1, places=6)
        self.assertAlmostEqual(values["a3_margin_slope"], 1.0, places=6)
        self.assertAlmostEqual(
            values["a3_margin_std"],
            np.std([1, 2, 3, 4, 5]) / 5,
            places=6,
        )
        self.assertAlmostEqual(values["serving_sinr_slope"], -1.0, places=6)
        self.assertAlmostEqual(values["target_sinr_slope"], 1.0, places=6)
        self.assertAlmostEqual(values["serving_qout_gap"], 0.45, places=6)
        self.assertAlmostEqual(values["target_qin_gap"], 0.4, places=6)
        self.assertEqual(values["n310_progress"], 0.0)
        self.assertEqual(values["t310_active"], 0.0)
        self.assertEqual(values["t310_progress"], 0.0)
        self.assertAlmostEqual(
            values["serving_residence_time"], 0.04 / 5.0, places=6
        )
        self.assertEqual(values["time_since_last_ho"], 1.0)
        self.assertEqual(values["target_is_previous_cell"], 0.0)
        self.assertEqual(values["recent_ho_count"], 0.0)

    def test_ping_pong_history_features_are_exposed(self) -> None:
        config = Config()
        builder = A3ObservationBuilder(config)
        builder.reset()
        decision = DecisionPoint(4, 0, 1, 0, 1.0)

        for tic in range(5):
            phase = ProtocolPhase.DECISION if tic == 4 else ProtocolPhase.A3_TTT
            point = decision if tic == 4 else None
            snapshot = ProtocolSnapshot(
                tic=tic,
                phase=phase,
                serving_cell=0,
                target_cell=1,
                decision_point=point,
                rsrp_a3_db=np.array([-80.0, -77.0]),
                sinr_rate_db=np.array([3.0, 4.0]),
                sinr_rlm_db=np.array([3.0, 4.0]),
                n310_progress=0.0,
                n311_progress=0.0,
                t310_active=False,
                t310_progress=0.0,
                serving_residence_ticks=20,
                time_since_last_ho_ticks=10,
                previous_cell=1,
                recent_ho_count=2,
            )
            builder.observe(TickResult(snapshot, 0.0, 0.0))

        values = builder.as_dict(builder.build(snapshot))

        self.assertEqual(values["target_is_previous_cell"], 1.0)
        self.assertAlmostEqual(values["time_since_last_ho"], 0.02, places=6)
        self.assertAlmostEqual(values["recent_ho_count"], 0.4, places=6)


class ObservationCausalityTests(unittest.TestCase):
    def test_future_trace_values_do_not_change_current_observation(self) -> None:
        prefix_rsrp = np.vstack(
            (
                np.full(5, -80.0),
                np.full(5, -77.0),
            )
        )
        prefix_sinr = np.vstack(
            (
                np.full(5, 5.0),
                np.full(5, 2.0),
            )
        )
        rsrp_a = np.hstack((prefix_rsrp, np.full((2, 5), -120.0)))
        rsrp_b = np.hstack((prefix_rsrp, np.full((2, 5), 20.0)))
        sinr_a = np.hstack((prefix_sinr, np.full((2, 5), -20.0)))
        sinr_b = np.hstack((prefix_sinr, np.full((2, 5), 30.0)))

        _, _, observation_a = _run_to_decision(rsrp_a, sinr_a)
        _, _, observation_b = _run_to_decision(rsrp_b, sinr_b)

        np.testing.assert_array_equal(observation_a, observation_b)

    def test_reset_removes_previous_episode_history(self) -> None:
        rsrp = np.vstack((np.full(8, -80.0), np.full(8, -77.0)))
        sinr = np.vstack((np.full(8, 5.0), np.full(8, 2.0)))
        engine, builder, _ = _run_to_decision(rsrp, sinr)

        builder.reset()

        with self.assertRaises(RuntimeError):
            builder.build(engine.snapshot)


class RunnerIntegrationTests(unittest.TestCase):
    def test_runner_passes_built_observation_to_policy(self) -> None:
        class RecordingPolicy:
            def __init__(self) -> None:
                self.observations: list[np.ndarray] = []

            def predict(self, observation: np.ndarray) -> HandoverAction:
                self.observations.append(observation.copy())
                return HandoverAction.STAY

        config = Config()
        rsrp = np.vstack((np.full(12, -80.0), np.full(12, -77.0)))
        sinr = np.vstack((np.full(12, 5.0), np.full(12, 2.0)))
        engine = HandoverEngine(config)
        engine.reset(RadioTrace.from_arrays(rsrp, sinr))
        builder = A3ObservationBuilder(config)
        policy = RecordingPolicy()

        run_policy(engine, policy, builder)

        self.assertEqual(len(policy.observations), 1)
        self.assertEqual(policy.observations[0].shape, (builder.size,))

    def test_history_must_cover_ttt(self) -> None:
        config = Config()
        config.a3_ttt_ms = 5120

        with self.assertRaises(ValueError):
            A3ObservationBuilder(
                config,
                A3ObservationConfig(history_window_ticks=500),
            )


if __name__ == "__main__":
    unittest.main()
