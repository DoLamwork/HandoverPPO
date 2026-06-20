"""Tests for the primitive handover reward model."""

import unittest

import numpy as np

from ho_optim_drl import Config
from ho_optim_drl.protocol import ProtocolPhase, ProtocolSnapshot, TickResult
from ho_optim_drl.smdp import (
    HandoverRewardConfig,
    HandoverRewardModel,
    RewardBreakdown,
)


def _tick(
    delivered_rate_mbps: float,
    *,
    rlf_started: bool = False,
    ping_pong_detected: bool = False,
) -> TickResult:
    snapshot = ProtocolSnapshot(
        tic=0,
        phase=ProtocolPhase.CONNECTED,
        serving_cell=0,
        target_cell=None,
        decision_point=None,
        rsrp_a3_db=np.array([-80.0, -90.0]),
        sinr_rate_db=np.array([5.0, -5.0]),
        sinr_rlm_db=np.array([5.0, -5.0]),
        n310_progress=0.0,
        t310_active=False,
        t310_progress=0.0,
        serving_residence_ticks=0,
        time_since_last_ho_ticks=1,
        previous_cell=None,
        recent_ho_count=0,
    )
    return TickResult(
        snapshot=snapshot,
        delivered_rate_mbps=delivered_rate_mbps,
        best_rate_mbps=max(delivered_rate_mbps, 0.0),
        rlf_started=rlf_started,
        ping_pong_detected=ping_pong_detected,
    )


class RewardBreakdownTests(unittest.TestCase):
    def test_breakdowns_add_component_by_component(self) -> None:
        first = RewardBreakdown(rate=0.2, rlf=-5.0)
        second = RewardBreakdown(rate=0.3, ping_pong=-1.0)

        total = first + second

        self.assertEqual(total, RewardBreakdown(0.5, -5.0, -1.0))
        self.assertAlmostEqual(total.total, -5.5)

    def test_non_finite_component_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            RewardBreakdown(rate=float("nan"))


class HandoverRewardModelTests(unittest.TestCase):
    def test_rate_is_normalized_delivered_data(self) -> None:
        model = HandoverRewardModel(
            Config(),
            HandoverRewardConfig(
                rate_scale_mbps=100.0,
                rlf_penalty=5.0,
                ping_pong_penalty=1.0,
            ),
        )

        reward = model.compute(_tick(50.0))

        self.assertAlmostEqual(reward.rate, 0.01 * 50.0 / 100.0)
        self.assertEqual(reward.rlf, 0.0)
        self.assertEqual(reward.ping_pong, 0.0)

    def test_event_penalties_are_applied_once_on_event_tick(self) -> None:
        model = HandoverRewardModel(Config())

        reward = model.compute(
            _tick(
                50.0,
                rlf_started=True,
                ping_pong_detected=True,
            )
        )

        self.assertAlmostEqual(reward.rate, 0.005)
        self.assertEqual(reward.rlf, -5.0)
        self.assertEqual(reward.ping_pong, -1.0)
        self.assertAlmostEqual(reward.total, -5.995)

    def test_zero_rate_represents_outage_without_extra_hidden_penalty(self) -> None:
        model = HandoverRewardModel(Config())

        reward = model.compute(_tick(0.0))

        self.assertEqual(reward, RewardBreakdown())

    def test_invalid_reward_configuration_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            HandoverRewardModel(
                Config(),
                HandoverRewardConfig(rate_scale_mbps=0.0),
            )
        with self.assertRaises(ValueError):
            HandoverRewardModel(
                Config(),
                HandoverRewardConfig(rlf_penalty=-1.0),
            )

    def test_invalid_delivered_rate_is_rejected(self) -> None:
        model = HandoverRewardModel(Config())

        with self.assertRaises(ValueError):
            model.compute(_tick(float("nan")))
        with self.assertRaises(ValueError):
            model.compute(_tick(-1.0))


if __name__ == "__main__":
    unittest.main()
