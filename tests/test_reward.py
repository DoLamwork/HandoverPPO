"""Tests for the primitive handover reward model."""

import unittest
from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from ho_optim_drl import Config
from ho_optim_drl.data import RadioTrace
from ho_optim_drl.protocol import ProtocolPhase, ProtocolSnapshot, TickResult
from ho_optim_drl.smdp import (
    HandoverRewardConfig,
    HandoverRewardModel,
    P3_E07_REWARD_SCHEMA,
    P3_E08_REWARD_SCHEMA,
    P3_E09_REWARD_SCHEMA,
    P3_E10_REWARD_SCHEMA,
    RewardBreakdown,
    TrajectoryQoSRewardModel,
    trajectory_reward_config,
)


def _tick(
    delivered_rate_mbps: float,
    *,
    rlf_started: bool = False,
    ping_pong_detected: bool = False,
    handover_started: bool = False,
    handover_failed: bool = False,
    phase: ProtocolPhase = ProtocolPhase.CONNECTED,
) -> TickResult:
    snapshot = ProtocolSnapshot(
        tic=0,
        phase=phase,
        serving_cell=0,
        target_cell=None,
        decision_point=None,
        rsrp_a3_db=np.array([-80.0, -90.0]),
        sinr_rate_db=np.array([5.0, -5.0]),
        sinr_rlm_db=np.array([5.0, -5.0]),
        n310_progress=0.0,
        n311_progress=0.0,
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
        handover_started=handover_started,
        handover_failed=handover_failed,
        rlf_started=rlf_started,
        ping_pong_detected=ping_pong_detected,
    )


def _trace(*, speed_kmh: float | None = 30.0) -> RadioTrace:
    # SINR=0 dB gives exactly 1 bit/s/Hz, hence 10 Mbps at 10 MHz.
    values = np.zeros((2, 2), dtype=np.float64)
    return RadioTrace.from_arrays(
        values,
        values,
        speed_kmh=speed_kmh,
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


class TrajectoryQoSRewardModelTests(unittest.TestCase):
    def test_e07_oracle_trajectory_sums_to_sixty(self) -> None:
        model = TrajectoryQoSRewardModel(
            Config(),
            P3_E07_REWARD_SCHEMA,
        )
        model.reset(_trace())

        total = sum(model.compute(_tick(10.0)).rate for _ in range(2))

        self.assertAlmostEqual(total, 60.0)

    def test_e08_does_not_double_count_hof_as_pure_rlf(self) -> None:
        model = TrajectoryQoSRewardModel(
            Config(),
            P3_E08_REWARD_SCHEMA,
        )
        model.reset(_trace())

        reward = model.compute(
            _tick(
                0.0,
                handover_failed=True,
                rlf_started=True,
                ping_pong_detected=True,
            )
        )

        self.assertEqual(reward.handover_failure, -0.2)
        self.assertEqual(reward.rlf, 0.0)
        self.assertEqual(reward.ping_pong, -0.1)

    def test_e09_charges_recovery_duration_per_primitive_tick(self) -> None:
        model = TrajectoryQoSRewardModel(
            Config(),
            P3_E09_REWARD_SCHEMA,
        )
        model.reset(_trace())

        reward = model.compute(
            _tick(0.0, phase=ProtocolPhase.RLF_RECOVERY)
        )

        self.assertAlmostEqual(reward.rlf_recovery, -0.01)

    def test_e10_updates_per_speed_dual_only_between_rollouts(self) -> None:
        config = trajectory_reward_config(
            P3_E10_REWARD_SCHEMA,
            dual_min_episodes_per_speed=2,
        )
        model = TrajectoryQoSRewardModel(
            Config(),
            P3_E10_REWARD_SCHEMA,
            config,
        )
        model.begin_rollout()
        model.reset(_trace(speed_kmh=30.0))
        before = model.compute(_tick(10.0)).rate

        update = model.update_dual(
            [
                SimpleNamespace(speed_kmh=30.0, relative_rate_ratio=0.8),
                SimpleNamespace(speed_kmh=30.0, relative_rate_ratio=0.8),
            ]
        )
        still_frozen = model.compute(_tick(10.0)).rate
        model.begin_rollout()
        after = model.compute(_tick(10.0)).rate

        assert update is not None
        self.assertEqual(update["sample_count_by_speed"], {"30": 2})
        self.assertAlmostEqual(
            update["lambda_after_by_speed"]["30"],
            0.09,
        )
        self.assertAlmostEqual(still_frozen, before)
        self.assertAlmostEqual(after, before * 1.09)

    def test_e10_state_round_trip_preserves_dual_coefficients(self) -> None:
        config = trajectory_reward_config(
            P3_E10_REWARD_SCHEMA,
            dual_min_episodes_per_speed=1,
        )
        source = TrajectoryQoSRewardModel(
            Config(),
            P3_E10_REWARD_SCHEMA,
            config,
        )
        source.update_dual(
            [SimpleNamespace(speed_kmh=50.0, relative_rate_ratio=0.78)]
        )
        restored = TrajectoryQoSRewardModel(
            Config(),
            P3_E10_REWARD_SCHEMA,
            config,
        )

        restored.load_state_dict(source.state_dict())

        self.assertEqual(restored.state_dict(), source.state_dict())

    def test_dual_reward_requires_speed_metadata(self) -> None:
        model = TrajectoryQoSRewardModel(
            Config(),
            P3_E10_REWARD_SCHEMA,
        )

        with self.assertRaises(ValueError):
            model.reset(_trace(speed_kmh=None))

    def test_dual_schema_contract_and_infeasible_flag(self) -> None:
        non_dual = trajectory_reward_config(P3_E09_REWARD_SCHEMA)
        with self.assertRaisesRegex(ValueError, "dual_enabled"):
            TrajectoryQoSRewardModel(
                Config(),
                P3_E10_REWARD_SCHEMA,
                non_dual,
            )

        config = replace(
            trajectory_reward_config(
                P3_E10_REWARD_SCHEMA,
                dual_min_episodes_per_speed=1,
            ),
            dual_learning_rate=1.0,
            dual_max_coefficient=0.05,
        )
        model = TrajectoryQoSRewardModel(
            Config(),
            P3_E10_REWARD_SCHEMA,
            config,
        )

        update = model.update_dual(
            [SimpleNamespace(speed_kmh=30.0, relative_rate_ratio=0.0)]
        )

        assert update is not None
        self.assertEqual(update["infeasible_speeds"], ["30"])
        self.assertEqual(update["lambda_before_by_speed"], {"30": 0.0})


if __name__ == "__main__":
    unittest.main()
