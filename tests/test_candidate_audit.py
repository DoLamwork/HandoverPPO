"""Tests for the train-only P1 candidate counterfactual audit."""

import unittest

import numpy as np

from ho_optim_drl import Config
from ho_optim_drl.commands.audit_candidates import (
    simulate_counterfactual,
    summarize_coverage,
)
from ho_optim_drl.data import RadioTrace
from ho_optim_drl.protocol import HandoverEngine


class CandidateAuditTests(unittest.TestCase):
    def test_counterfactual_branch_does_not_mutate_source_engine(self) -> None:
        config = Config()
        config.t_ho_prep_ms = 10
        config.t_ho_exec_ms = 10
        rsrp = np.vstack(
            (np.full(20, -80.0), np.full(20, -77.0))
        )
        sinr = np.vstack(
            (np.full(20, 5.0), np.full(20, 1.0))
        )
        engine = HandoverEngine(config)
        engine.reset(RadioTrace.from_arrays(rsrp, sinr))
        for _ in range(5):
            engine.advance_one_tick()

        stay = simulate_counterfactual(
            engine, initial_target=None, horizon_ticks=(5, 10)
        )
        handover = simulate_counterfactual(
            engine, initial_target=1, horizon_ticks=(5, 10)
        )

        self.assertIsNotNone(engine.decision_point)
        self.assertEqual(set(stay), {5, 10})
        self.assertEqual(set(handover), {5, 10})
        self.assertGreater(stay[5].delivered_mbit, handover[5].delivered_mbit)

    def test_coverage_is_conditional_on_positive_states(self) -> None:
        common = {
            "source": "baseline",
            "speed_kmh": 30,
            "horizon_seconds": 1.0,
        }
        rows = [
            {
                **common,
                "useful_slots_vs_c1": "[2,7]",
                "useful_slots_vs_best_stay_c1": "[7]",
                "coverage_k3_vs_c1": True,
                "coverage_k5_vs_c1": True,
                "coverage_k10_vs_c1": True,
                "coverage_k3_vs_best_stay_c1": False,
                "coverage_k5_vs_best_stay_c1": False,
                "coverage_k10_vs_best_stay_c1": True,
            },
            {
                **common,
                "useful_slots_vs_c1": "[]",
                "useful_slots_vs_best_stay_c1": "[]",
                "coverage_k3_vs_c1": False,
                "coverage_k5_vs_c1": False,
                "coverage_k10_vs_c1": False,
                "coverage_k3_vs_best_stay_c1": False,
                "coverage_k5_vs_best_stay_c1": False,
                "coverage_k10_vs_best_stay_c1": False,
            },
        ]

        overall = next(
            row
            for row in summarize_coverage(rows)
            if row["group_by"] == "overall"
        )
        self.assertEqual(overall["state_count"], 2)
        self.assertEqual(overall["positive_states_vs_c1"], 1)
        self.assertEqual(overall["coverage_k3_vs_c1"], 1.0)
        self.assertEqual(overall["coverage_k5_vs_best_stay_c1"], 0.0)
        self.assertEqual(overall["coverage_k10_vs_best_stay_c1"], 1.0)


if __name__ == "__main__":
    unittest.main()
