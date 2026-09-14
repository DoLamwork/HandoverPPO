"""Tests for decision-to-decision SMDP transitions."""

import unittest

import numpy as np

from ho_optim_drl import Config
from ho_optim_drl.data import RadioTrace
from ho_optim_drl.protocol import HandoverEngine
from ho_optim_drl.smdp import (
    A3ObservationBuilder,
    HandoverAction,
    HandoverRewardModel,
    MacroTransition,
    MacroTransitionBuilder,
    RewardBreakdown,
)


def _build_runner(
    config: Config,
    *,
    gamma_per_second: float = 0.99,
) -> MacroTransitionBuilder:
    return MacroTransitionBuilder(
        HandoverEngine(config),
        A3ObservationBuilder(config),
        HandoverRewardModel(config),
        gamma_per_second=gamma_per_second,
    )


class MacroTransitionTests(unittest.TestCase):
    def test_wait_50_creates_a_five_tick_macro_transition(self) -> None:
        config = Config()
        rsrp = np.vstack(
            (
                np.full(14, -80.0),
                np.full(14, -77.0),
            )
        )
        sinr = np.vstack((np.full(14, 10.0), np.full(14, 0.0)))
        runner = _build_runner(config, gamma_per_second=0.95)

        runner.reset(RadioTrace.from_arrays(rsrp, sinr))
        transition = runner.step(HandoverAction.WAIT_50)

        expected_rate_mbps = config.bw * np.log2(1 + 10.0) / 1e6
        expected_rate_reward = (
            5 * config.delta_t_ms / 1000 * expected_rate_mbps / 100.0
        )
        self.assertEqual(transition.action, HandoverAction.WAIT_50)
        self.assertEqual(transition.duration_ticks, 5)
        self.assertAlmostEqual(transition.duration_seconds, 0.05)
        self.assertAlmostEqual(transition.discount, 0.95**0.05)
        self.assertAlmostEqual(transition.reward.rate, expected_rate_reward)
        self.assertFalse(transition.terminated)
        self.assertIsNotNone(transition.next_observation)
        self.assertEqual(runner.engine.decision_point.tic, 9)

    def test_stay_advances_exactly_to_next_rearmed_a3_decision(self) -> None:
        config = Config()
        rsrp = np.vstack(
            (
                np.full(16, -80.0),
                np.array([-77.0] * 8 + [-81.0] + [-77.0] * 7),
            )
        )
        sinr = np.vstack((np.full(16, 10.0), np.full(16, 0.0)))
        runner = _build_runner(config, gamma_per_second=0.95)

        first_observation = runner.reset(RadioTrace.from_arrays(rsrp, sinr))
        self.assertEqual(runner.uncontrolled_duration_ticks, 5)
        self.assertGreater(runner.uncontrolled_reward.rate, 0.0)
        transition = runner.step(HandoverAction.STAY)

        expected_rate_mbps = config.bw * np.log2(1 + 10.0) / 1e6
        expected_rate_reward = (
            9 * config.delta_t_ms / 1000 * expected_rate_mbps / 100.0
        )
        self.assertIsNotNone(first_observation)
        self.assertEqual(transition.duration_ticks, 9)
        self.assertAlmostEqual(transition.duration_seconds, 0.09)
        self.assertAlmostEqual(transition.reward.rate, expected_rate_reward)
        self.assertEqual(transition.reward.rlf, 0.0)
        self.assertEqual(transition.reward.ping_pong, 0.0)
        self.assertAlmostEqual(transition.discount, 0.95**0.09)
        self.assertFalse(transition.terminated)
        self.assertFalse(transition.truncated)
        self.assertIsNotNone(transition.next_observation)
        np.testing.assert_array_equal(
            transition.next_observation,
            runner.current_observation,
        )
        with self.assertRaises(ValueError):
            transition.observation[0] = 0.0

    def test_terminal_tail_is_included_in_last_transition(self) -> None:
        config = Config()
        rsrp = np.vstack(
            (
                np.full(8, -80.0),
                np.full(8, -77.0),
            )
        )
        sinr = np.vstack((np.full(8, 10.0), np.full(8, 0.0)))
        runner = _build_runner(config)

        runner.reset(RadioTrace.from_arrays(rsrp, sinr))
        prefix_ticks = runner.uncontrolled_duration_ticks
        transition = runner.step(HandoverAction.STAY)

        self.assertEqual(transition.duration_ticks, 3)
        self.assertEqual(prefix_ticks + transition.duration_ticks, 8)
        self.assertTrue(transition.terminated)
        self.assertFalse(transition.truncated)
        self.assertIsNone(transition.next_observation)
        self.assertIsNone(runner.current_observation)
        self.assertTrue(runner.terminated)
        with self.assertRaises(RuntimeError):
            runner.step(HandoverAction.STAY)

    def test_handover_outage_and_post_ho_rate_share_one_macro_step(self) -> None:
        config = Config()
        config.t_ho_prep_ms = 10
        config.t_ho_exec_ms = 10
        rsrp = np.vstack(
            (
                np.full(8, -80.0),
                np.full(8, -77.0),
            )
        )
        sinr = np.vstack((np.full(8, 5.0), np.full(8, 2.0)))
        runner = _build_runner(config)

        runner.reset(RadioTrace.from_arrays(rsrp, sinr))
        transition = runner.step(HandoverAction.HANDOVER)

        target_rate_mbps = config.bw * np.log2(1 + 10 ** (2.0 / 10)) / 1e6
        expected_rate_reward = (
            2 * config.delta_t_ms / 1000 * target_rate_mbps / 100.0
        )
        self.assertTrue(transition.terminated)
        self.assertEqual(transition.duration_ticks, 3)
        self.assertEqual(runner.engine.successful_ho_history, [(6, 0, 1)])
        self.assertAlmostEqual(transition.reward.rate, expected_rate_reward)
        self.assertEqual(transition.reward.rlf, 0.0)

    def test_trace_without_decision_terminates_during_reset(self) -> None:
        config = Config()
        rsrp = np.vstack((np.full(6, -80.0), np.full(6, -90.0)))
        sinr = np.vstack((np.full(6, 5.0), np.full(6, 0.0)))
        runner = _build_runner(config)

        observation = runner.reset(RadioTrace.from_arrays(rsrp, sinr))

        self.assertIsNone(observation)
        self.assertTrue(runner.terminated)
        self.assertEqual(runner.uncontrolled_duration_ticks, 6)
        with self.assertRaises(RuntimeError):
            runner.step(HandoverAction.STAY)

    def test_integer_actions_are_supported_and_invalid_actions_rejected(self) -> None:
        config = Config()
        rsrp = np.vstack((np.full(8, -80.0), np.full(8, -77.0)))
        sinr = np.vstack((np.full(8, 5.0), np.full(8, 0.0)))
        runner = _build_runner(config)
        runner.reset(RadioTrace.from_arrays(rsrp, sinr))

        with self.assertRaises(ValueError):
            runner.step(7)

        transition = runner.step(int(HandoverAction.STAY))
        self.assertEqual(transition.action, HandoverAction.STAY)

    def test_invalid_continuous_time_discount_is_rejected(self) -> None:
        config = Config()
        for gamma in (0.0, 1.1, float("nan")):
            with self.subTest(gamma=gamma):
                with self.assertRaises(ValueError):
                    _build_runner(config, gamma_per_second=gamma)

    def test_invalid_transition_payload_is_rejected(self) -> None:
        observation = np.zeros(19, dtype=np.float32)
        with self.assertRaises(ValueError):
            MacroTransition(
                observation=observation,
                action=HandoverAction.STAY,
                reward=RewardBreakdown(),
                next_observation=None,
                duration_ticks=1,
                duration_seconds=0.01,
                discount=0.99,
                terminated=False,
                truncated=False,
            )
        with self.assertRaises(ValueError):
            MacroTransition(
                observation=np.array([float("nan")]),
                action=HandoverAction.STAY,
                reward=RewardBreakdown(),
                next_observation=None,
                duration_ticks=1,
                duration_seconds=0.01,
                discount=0.99,
                terminated=True,
                truncated=False,
            )


if __name__ == "__main__":
    unittest.main()
