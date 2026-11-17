"""Topology-aware candidate actions and masked-policy tests."""

from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch

from ho_optim_drl import Config
from ho_optim_drl.agents.ppo.network import ActorCritic
from ho_optim_drl.commands.audit_early_timing_learnability import (
    causal_features,
    feature_names,
)
from ho_optim_drl.commands.audit_t4d_compact_causal_probe import (
    RISK_FEATURE_NAMES,
)
from ho_optim_drl.data import RadioTrace, load_cell_topology
from ho_optim_drl.protocol import HandoverEngine
from ho_optim_drl.smdp import (
    HandoverAction,
    HandoverRewardModel,
    MacroTransitionBuilder,
    TopologyCandidateObservationBuilder,
)


ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DATA = ROOT / "data" / "uav_100m_grid100m_dev_processed"


def _twenty_one_cell_trace() -> RadioTrace:
    num_samples = 16
    rsrp = np.full((21, num_samples), -100.0, dtype=np.float64)
    rsrp[0] = -80.0
    rsrp[1] = -91.0
    rsrp[2] = -92.0
    rsrp[3] = -79.0
    rsrp[4] = -80.5
    rsrp[5] = -81.0
    rsrp[6] = -77.0
    rsrp[7] = -84.0
    rsrp[8] = -85.0
    rsrp[9] = -78.0
    rsrp[10] = -82.0
    rsrp[11] = -83.0

    # Initial camping uses SINR.  Keep cell 0 as the serving cell while RSRP
    # makes cell 6 the completed-A3 target.
    sinr = np.full((21, num_samples), -5.0, dtype=np.float64)
    sinr[0] = 10.0
    sinr[6] = 5.0
    sinr[9] = 4.0
    return RadioTrace.from_arrays(
        rsrp, sinr, trace_id="topology21", speed_kmh=30.0
    )


def _delayed_a3_trace() -> RadioTrace:
    num_samples = 80
    rsrp = np.full((21, num_samples), -110.0, dtype=np.float64)
    rsrp[0] = -80.0
    rsrp[6, :60] = -90.0
    rsrp[6, 60:] = -77.0
    rsrp[9] = -84.0
    rsrp[4] = -86.0

    sinr = np.full((21, num_samples), -8.0, dtype=np.float64)
    sinr[0] = 12.0
    sinr[6, :54] = 0.0
    sinr[6, 54:] = np.linspace(0.0, 40.0, num_samples - 54)
    sinr[9] = 5.0
    sinr[4, :55] = -5.0
    sinr[4, 55:] = np.linspace(-5.0, 8.0, num_samples - 55)
    return RadioTrace.from_arrays(
        rsrp,
        sinr,
        trace_id="delayed_a3",
        speed_kmh=30.0,
    )


class TopologyCandidateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.topology = load_cell_topology(PROCESSED_DATA)
        if self.topology is None:
            self.fail("The processed 1060-route dataset has no topology metadata.")

    def _runner(self) -> tuple[MacroTransitionBuilder, np.ndarray]:
        config = Config()
        builder = TopologyCandidateObservationBuilder(
            config,
            topology=self.topology,
        )
        runner = MacroTransitionBuilder(
            HandoverEngine(config),
            builder,
            HandoverRewardModel(config),
        )
        observation = runner.reset(_twenty_one_cell_trace())
        self.assertIsNotNone(observation)
        return runner, observation

    def _configured_runner(
        self,
        *,
        candidate_count: int,
        candidate_order: str,
        observation_schema: str = "legacy_v1",
        trace: RadioTrace | None = None,
    ) -> tuple[MacroTransitionBuilder, np.ndarray]:
        config = Config()
        builder = TopologyCandidateObservationBuilder(
            config,
            topology=self.topology,
            candidate_count=candidate_count,
            candidate_order=candidate_order,
            observation_schema=observation_schema,
        )
        runner = MacroTransitionBuilder(
            HandoverEngine(config),
            builder,
            HandoverRewardModel(config),
        )
        observation = runner.reset(trace or _twenty_one_cell_trace())
        self.assertIsNotNone(observation)
        return runner, observation

    def test_compact_metadata_maps_seven_sites_and_twenty_one_cells(self) -> None:
        self.assertEqual(len(self.topology.sites), 7)
        self.assertEqual(self.topology.num_cells, 21)
        self.assertEqual(self.topology.cells_for_site(3), (9, 10, 11))
        self.assertEqual(
            set(self.topology.neighboring_site_ids(3)),
            {0, 1, 2, 4, 5, 6},
        )

    def test_candidate_order_is_causal_unique_and_topology_aware(self) -> None:
        runner, observation = self._runner()
        builder = runner.observation_builder

        self.assertEqual(observation.shape, (109,))
        self.assertEqual(
            builder.candidate_cells,
            (6, 1, 2, 7, 8, 9, 10, 11, 3, 4),
        )
        self.assertEqual(len(set(builder.candidate_cells)), 10)
        self.assertNotIn(runner.engine.decision_point.serving_cell, builder.candidate_cells)
        np.testing.assert_array_equal(builder.action_mask(), np.ones(11, dtype=bool))

    def test_candidate_action_passes_the_frozen_target_to_engine(self) -> None:
        runner, _ = self._runner()
        with patch.object(
            runner.engine,
            "start_handover",
            wraps=runner.engine.start_handover,
        ) as start_handover:
            transition = runner.step(HandoverAction.HO_C2)

        start_handover.assert_called_once_with(target_cell=1)
        self.assertEqual(transition.action, HandoverAction.HO_C2)

    def test_future_samples_do_not_change_candidates_observation_or_mask(
        self,
    ) -> None:
        original = _twenty_one_cell_trace()
        changed_rsrp = np.array(original.rsrp_a3_db, copy=True)
        changed_sinr = np.array(original.sinr_rate_db, copy=True)
        changed_rsrp[:, 5:] = np.flip(changed_rsrp[:, 5:], axis=0) + 7.0
        changed_sinr[:, 5:] = np.flip(changed_sinr[:, 5:], axis=0) - 4.0
        changed = RadioTrace.from_arrays(changed_rsrp, changed_sinr)

        first_runner, first_observation = self._runner()
        second_builder = TopologyCandidateObservationBuilder(
            Config(),
            topology=self.topology,
        )
        second_runner = MacroTransitionBuilder(
            HandoverEngine(Config()),
            second_builder,
            HandoverRewardModel(Config()),
        )
        second_observation = second_runner.reset(changed)

        np.testing.assert_array_equal(first_observation, second_observation)
        self.assertEqual(
            first_runner.observation_builder.candidate_cells,
            second_builder.candidate_cells,
        )
        np.testing.assert_array_equal(
            first_runner.observation_builder.action_mask(),
            second_builder.action_mask(),
        )

    def test_zero_policy_head_is_uniform_over_only_valid_actions(self) -> None:
        model = ActorCritic(109, 11, hidden_sizes=(8,))
        observations = torch.zeros((1, 109), dtype=torch.float32)

        all_valid = torch.ones((1, 11), dtype=torch.bool)
        distribution, _ = model.distribution_and_value(observations, all_valid)
        torch.testing.assert_close(
            distribution.probs,
            torch.full((1, 11), 1.0 / 11.0),
        )

        two_valid = torch.tensor(
            [[True, True, False, False, False, False, False, False, False, False, False]]
        )
        distribution, _ = model.distribution_and_value(observations, two_valid)
        torch.testing.assert_close(
            distribution.probs,
            torch.tensor([[0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
        )

    def test_ranked_k3_k5_are_prefixes_of_one_causal_k10_order(self) -> None:
        trace = _twenty_one_cell_trace()
        changed_sinr = np.array(trace.sinr_rate_db, copy=True)
        changed_sinr[4, :5] = np.array([-20.0, -15.0, -10.0, -5.0, 3.0])
        changed_sinr[4, 5:] = 3.0
        trace = RadioTrace.from_arrays(trace.rsrp_a3_db, changed_sinr)

        configured = {
            count: self._configured_runner(
                candidate_count=count,
                candidate_order="ranked_v1",
                trace=trace,
            )
            for count in (3, 5, 10)
        }
        cells = {
            count: runner.observation_builder.candidate_cells
            for count, (runner, _) in configured.items()
        }

        self.assertEqual(cells[10][:3], cells[3])
        self.assertEqual(cells[10][:5], cells[5])
        self.assertEqual(cells[10][0], 6)
        self.assertEqual(cells[10][1], 9)
        self.assertEqual(cells[10][2], 4)
        self.assertEqual(configured[3][1].shape, (39,))
        self.assertEqual(configured[5][1].shape, (59,))
        self.assertEqual(configured[10][1].shape, (109,))

    def test_ranked_order_does_not_read_future_samples(self) -> None:
        original = _twenty_one_cell_trace()
        changed_sinr = np.array(original.sinr_rate_db, copy=True)
        changed_rsrp = np.array(original.rsrp_a3_db, copy=True)
        changed_sinr[:, 5:] = np.flip(changed_sinr[:, 5:], axis=0) + 30.0
        changed_rsrp[:, 5:] = np.flip(changed_rsrp[:, 5:], axis=0) + 30.0
        changed = RadioTrace.from_arrays(changed_rsrp, changed_sinr)

        first_runner, first_observation = self._configured_runner(
            candidate_count=10,
            candidate_order="ranked_v1",
            trace=original,
        )
        second_runner, second_observation = self._configured_runner(
            candidate_count=10,
            candidate_order="ranked_v1",
            trace=changed,
        )

        np.testing.assert_array_equal(first_observation, second_observation)
        self.assertEqual(
            first_runner.observation_builder.candidate_cells,
            second_runner.observation_builder.candidate_cells,
        )
        np.testing.assert_array_equal(
            first_runner.observation_builder.action_mask(),
            second_runner.observation_builder.action_mask(),
        )

    def test_candidate_configuration_rejects_unsupported_values(self) -> None:
        with self.assertRaises(ValueError):
            TopologyCandidateObservationBuilder(
                Config(),
                topology=self.topology,
                candidate_count=4,
            )
        with self.assertRaises(ValueError):
            TopologyCandidateObservationBuilder(
                Config(),
                topology=self.topology,
                candidate_order="future_oracle",
            )
        with self.assertRaises(ValueError):
            TopologyCandidateObservationBuilder(
                Config(),
                topology=self.topology,
                observation_schema="future_oracle",
            )

    def test_e04_uses_cell_id_history_before_a3_without_changing_size(self) -> None:
        trace = _delayed_a3_trace()
        legacy_runner, legacy_observation = self._configured_runner(
            candidate_count=3,
            candidate_order="ranked_v1",
            observation_schema="legacy_v1",
            trace=trace,
        )
        rolling_runner, rolling_observation = self._configured_runner(
            candidate_count=3,
            candidate_order="ranked_v1",
            observation_schema="rolling_v1",
            trace=trace,
        )
        legacy = legacy_runner.observation_builder.as_dict(legacy_observation)
        rolling = rolling_runner.observation_builder.as_dict(
            rolling_observation
        )

        self.assertEqual(legacy_observation.shape, (39,))
        self.assertEqual(rolling_observation.shape, (39,))
        self.assertLess(
            rolling["a3_margin_mean"], legacy["a3_margin_mean"]
        )
        self.assertEqual(
            rolling_runner.observation_builder.candidate_cells,
            legacy_runner.observation_builder.candidate_cells,
        )

    def test_e05_unclipped_monotonic_se_distinguishes_high_sinr(self) -> None:
        trace = _delayed_a3_trace()
        rolling_runner, rolling_observation = self._configured_runner(
            candidate_count=3,
            candidate_order="ranked_v1",
            observation_schema="rolling_v1",
            trace=trace,
        )
        monotonic_runner, monotonic_observation = self._configured_runner(
            candidate_count=3,
            candidate_order="ranked_v1",
            observation_schema="monotonic_v1",
            trace=trace,
        )
        rolling = rolling_runner.observation_builder.as_dict(
            rolling_observation
        )
        monotonic = monotonic_runner.observation_builder.as_dict(
            monotonic_observation
        )

        self.assertAlmostEqual(
            rolling["serving_spectral_efficiency"], 1.0
        )
        self.assertAlmostEqual(
            rolling["target_spectral_efficiency"], 1.0
        )
        self.assertAlmostEqual(rolling["delta_spectral_efficiency"], 0.0)
        self.assertGreater(
            monotonic["target_spectral_efficiency"],
            monotonic["serving_spectral_efficiency"],
        )
        self.assertGreater(monotonic["delta_spectral_efficiency"], 0.0)
        self.assertLess(monotonic["target_spectral_efficiency"], 1.0)

    def test_e06_has_17_global_and_15_features_per_candidate(self) -> None:
        runner, observation = self._configured_runner(
            candidate_count=3,
            candidate_order="ranked_v1",
            observation_schema="qos_full_v1",
            trace=_delayed_a3_trace(),
        )
        builder = runner.observation_builder
        values = builder.as_dict(observation)

        self.assertEqual(observation.shape, (62,))
        self.assertEqual(len(builder.feature_names), 17 + 15 * 3)
        self.assertTrue(np.all(observation >= builder.low))
        self.assertTrue(np.all(observation <= builder.high))
        self.assertAlmostEqual(values["history_coverage_500ms"], 1.0)
        self.assertAlmostEqual(values["speed_magnitude_kmh"], 0.25)
        self.assertEqual(values["c1_valid"], 1.0)
        self.assertEqual(values["c1_is_a3_target"], 1.0)
        self.assertEqual(values["c2_is_a3_target"], 0.0)
        self.assertEqual(values["c3_is_a3_target"], 0.0)

    def test_e06_requires_speed_metadata(self) -> None:
        trace = _twenty_one_cell_trace()
        trace_without_speed = RadioTrace.from_arrays(
            trace.rsrp_a3_db,
            trace.sinr_rate_db,
            trace_id="missing_speed",
        )
        with self.assertRaisesRegex(ValueError, "speed_kmh"):
            self._configured_runner(
                candidate_count=3,
                candidate_order="ranked_v1",
                observation_schema="qos_full_v1",
                trace=trace_without_speed,
            )

    def test_t4d1_k1_matches_frozen_compact_causal_features(self) -> None:
        trace = _delayed_a3_trace()
        runner, observation = self._configured_runner(
            candidate_count=1,
            candidate_order="ranked_v1",
            observation_schema="t4d1_compact_v1",
            trace=trace,
        )
        snapshot = runner.engine.snapshot
        decision = snapshot.decision_point
        self.assertIsNotNone(decision)
        full = causal_features(
            trace,
            tic=decision.tic,
            serving=decision.serving_cell,
            target=decision.target_cell,
            a3_start_tic=decision.a3_start_tic,
            speed_kmh=int(trace.speed_kmh),
            q_out_db=runner.engine.config.q_out_db,
            dt_ms=runner.engine.config.delta_t_ms,
            snapshot=snapshot,
            schema="t4b_protocol_history_v1",
        )
        names = feature_names("t4b_protocol_history_v1")
        expected = full[[names.index(name) for name in RISK_FEATURE_NAMES]]
        self.assertEqual(observation.shape, (19,))
        self.assertEqual(
            runner.observation_builder.feature_names,
            RISK_FEATURE_NAMES,
        )
        np.testing.assert_allclose(observation, expected, rtol=0.0, atol=1e-7)
        np.testing.assert_array_equal(
            runner.observation_builder.action_mask(),
            np.asarray([True, True]),
        )

    def test_e04_e06_do_not_read_suffix_after_decision(self) -> None:
        original = _delayed_a3_trace()
        reference_runner, _ = self._configured_runner(
            candidate_count=3,
            candidate_order="ranked_v1",
            observation_schema="qos_full_v1",
            trace=original,
        )
        decision_tic = reference_runner.engine.decision_point.tic
        changed_rsrp = np.array(original.rsrp_a3_db, copy=True)
        changed_sinr = np.array(original.sinr_rate_db, copy=True)
        changed_rsrp[:, decision_tic + 1 :] += 1000.0
        changed_sinr[:, decision_tic + 1 :] -= 1000.0
        changed = RadioTrace.from_arrays(
            changed_rsrp,
            changed_sinr,
            trace_id="changed_suffix",
            speed_kmh=original.speed_kmh,
        )

        for schema in ("rolling_v1", "monotonic_v1", "qos_full_v1"):
            first_runner, first = self._configured_runner(
                candidate_count=3,
                candidate_order="ranked_v1",
                observation_schema=schema,
                trace=original,
            )
            second_runner, second = self._configured_runner(
                candidate_count=3,
                candidate_order="ranked_v1",
                observation_schema=schema,
                trace=changed,
            )
            np.testing.assert_array_equal(first, second)
            self.assertEqual(
                first_runner.observation_builder.candidate_cells,
                second_runner.observation_builder.candidate_cells,
            )


if __name__ == "__main__":
    unittest.main()
