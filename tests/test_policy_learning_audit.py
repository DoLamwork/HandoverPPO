"""D1/D2 fixed-state policy-learning diagnostic tests."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ho_optim_drl import Config
from ho_optim_drl.commands.audit_candidates import simulate_counterfactual
from ho_optim_drl.commands.audit_policy_learning import (
    _D2_REWARDS,
    _average_precision,
    _route_folds,
    _select_action_from_probabilities,
    build_state_bank,
    load_state_bank,
    simulate_reward_counterfactual,
)
from ho_optim_drl.data import RadioTrace
from ho_optim_drl.protocol import HandoverEngine
from ho_optim_drl.smdp import LEGACY_REWARD_SCHEMA


def _pending_engine() -> HandoverEngine:
    config = Config()
    config.t_ho_prep_ms = 10
    config.t_ho_exec_ms = 10
    rsrp = np.vstack((np.full(30, -80.0), np.full(30, -77.0)))
    sinr = np.vstack((np.full(30, 5.0), np.full(30, 1.0)))
    engine = HandoverEngine(config)
    engine.reset(RadioTrace.from_arrays(rsrp, sinr, speed_kmh=30))
    while engine.decision_point is None:
        engine.advance_one_tick()
    return engine


class CounterfactualRewardAuditTests(unittest.TestCase):
    def test_action_selection_decoders_expose_joint_hierarchical_disagreement(
        self,
    ) -> None:
        probabilities = np.asarray([0.28, 0.2664, 0.2304, 0.2232])
        self.assertEqual(
            _select_action_from_probabilities(probabilities, "joint_argmax"),
            0,
        )
        self.assertEqual(
            _select_action_from_probabilities(
                probabilities, "hierarchical_argmax"
            ),
            1,
        )
        first = _select_action_from_probabilities(
            probabilities,
            "stochastic_joint",
            rng=np.random.default_rng(17),
        )
        second = _select_action_from_probabilities(
            probabilities,
            "stochastic_joint",
            rng=np.random.default_rng(17),
        )
        self.assertEqual(first, second)

    def test_probe_metrics_and_route_split_are_deterministic(self) -> None:
        labels = np.asarray([0, 1, 0, 1], dtype=np.int64)
        scores = np.asarray([0.1, 0.9, 0.2, 0.8])
        self.assertEqual(_average_precision(labels, scores), 1.0)
        arrays = {
            "speed_kmh": np.asarray([30, 30, 30, 30, 50, 50, 50, 50]),
            "trajectory_id": np.asarray([1, 1, 2, 2, 1, 1, 2, 2]),
        }
        fit, holdout, folds = _route_folds(arrays, seed=29)
        self.assertTrue(np.all(fit == ~holdout))
        self.assertEqual(sum(value == "holdout" for value in folds.values()), 2)
        self.assertEqual(fit[0], fit[1])
        self.assertEqual(fit[4], fit[5])

    def test_reward_branch_matches_physical_helper_and_preserves_source(self) -> None:
        engine = _pending_engine()
        source_tic = engine.snapshot.tic
        source_target = engine.decision_point.target_cell

        existing = simulate_counterfactual(
            engine, initial_target=None, horizon_ticks=(5, 10)
        )
        audited = simulate_reward_counterfactual(
            engine,
            initial_target=None,
            horizon_ticks=(5, 10),
            reward_schemas=_D2_REWARDS,
        )

        self.assertEqual(engine.snapshot.tic, source_tic)
        self.assertEqual(engine.decision_point.target_cell, source_target)
        self.assertEqual(audited[5]["physical"], existing[5].__dict__)
        self.assertEqual(audited[10]["physical"], existing[10].__dict__)
        expected_rate_reward = existing[5].delivered_mbit / 100.0
        self.assertAlmostEqual(
            audited[5]["reward_raw"][LEGACY_REWARD_SCHEMA]["rate"],
            expected_rate_reward,
        )

    def test_discount_starts_at_exponent_zero(self) -> None:
        engine = _pending_engine()
        audited = simulate_reward_counterfactual(
            engine,
            initial_target=None,
            horizon_ticks=(1,),
            reward_schemas=(LEGACY_REWARD_SCHEMA,),
            gamma_per_second=0.5,
        )
        raw = audited[1]["reward_raw"][LEGACY_REWARD_SCHEMA]
        discounted = audited[1]["reward_discounted"][LEGACY_REWARD_SCHEMA]
        self.assertEqual(raw, discounted)

    def test_horizon_past_terminal_is_marked_without_extra_outcomes(self) -> None:
        engine = _pending_engine()
        audited = simulate_reward_counterfactual(
            engine,
            initial_target=None,
            horizon_ticks=(10, 100),
            reward_schemas=(LEGACY_REWARD_SCHEMA,),
        )

        self.assertTrue(audited[100]["physical"]["trace_terminated"])
        self.assertLess(audited[100]["physical"]["elapsed_seconds"], 1.0)
        self.assertGreaterEqual(
            audited[100]["physical"]["delivered_mbit"],
            audited[10]["physical"]["delivered_mbit"],
        )

    def test_counterfactual_honors_a3_redecision_delay(self) -> None:
        engine = _pending_engine()
        audited = simulate_reward_counterfactual(
            engine,
            initial_target=None,
            horizon_ticks=(15,),
            reward_schemas=(LEGACY_REWARD_SCHEMA,),
            stay_redecision_ticks=10,
        )

        self.assertAlmostEqual(
            audited[15]["time_to_next_decision_seconds"],
            0.1,
        )


class StateBankIntegrationTests(unittest.TestCase):
    @staticmethod
    def _write_dataset(root: Path) -> None:
        header = "sample,time_ms,bs_0,bs_1\n"
        for trajectory in range(8):
            rsrp_rows = [
                f"{sample},{10 * sample},-80,-77"
                for sample in range(30)
            ]
            sinr_rows = [
                f"{sample},{10 * sample},5,1"
                for sample in range(30)
            ]
            (root / f"rsrp_30kmh_{trajectory}.csv").write_text(
                header + "\n".join(rsrp_rows) + "\n", encoding="utf-8"
            )
            (root / f"sinr_30kmh_{trajectory}.csv").write_text(
                header + "\n".join(sinr_rows) + "\n", encoding="utf-8"
            )

    def test_build_bank_is_train_only_reproducible_and_pickle_free(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            output = root / "bank"
            data.mkdir()
            self._write_dataset(data)

            manifest = build_state_bank(
                data_dir=str(data),
                output_dir=str(output),
                speeds=[30],
                split_seed=17,
                sample_seed=17,
                routes_per_speed=1,
                decision_indices=(0,),
                candidate_count=3,
                candidate_order="ranked_v1",
                schemas=("rolling_v1", "qos_full_v1"),
                source_checkpoints={},
            )
            loaded_manifest, arrays = load_state_bank(str(output))

            self.assertEqual(manifest["split"], "train")
            self.assertEqual(manifest["validation_or_test_rows_read"], 0)
            self.assertFalse(manifest["npz_allow_pickle"])
            self.assertEqual(loaded_manifest["npz_sha256"], manifest["npz_sha256"])
            self.assertEqual(len(arrays["state_id"]), 1)
            self.assertEqual(arrays["action_mask"].dtype, np.bool_)
            self.assertTrue(
                all(array.dtype != object for array in arrays.values())
            )
            serialized = json.loads(
                (output / "state_bank_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(serialized["state_count"], 1)


if __name__ == "__main__":
    unittest.main()
