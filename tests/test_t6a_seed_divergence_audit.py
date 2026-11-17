from __future__ import annotations

import unittest

import numpy as np

from ho_optim_drl.commands.audit_t6a_seed_divergence import (
    comparison_summary,
    summarize_training_signals,
)


class T6ASeedDivergenceAuditTests(unittest.TestCase):
    def test_same_state_summary_separates_argmax_from_probability_shift(self) -> None:
        seed17 = np.asarray([0.51, 0.49, 0.70, 0.20], dtype=np.float32)
        seed29 = np.asarray([0.49, 0.48, 0.60, 0.55], dtype=np.float32)

        report = comparison_summary(seed17, seed29)

        self.assertEqual(report["states"], 4)
        self.assertAlmostEqual(report["deterministic_disagreement_percent"], 50.0)
        self.assertAlmostEqual(report["seed17_ho_seed29_stay_percent"], 25.0)
        self.assertAlmostEqual(report["seed17_stay_seed29_ho_percent"], 25.0)
        self.assertAlmostEqual(
            report["p_ho_seed29_minus_seed17"]["mean"], 0.055, places=6
        )

    def test_training_signal_summary_preserves_early_and_late_segments(self) -> None:
        rows = []
        for update in range(1, 9):
            row = {
                "update": update,
                "macro_steps": update * 1024,
                "entropy": 0.7 - update * 0.01,
                "normalized_entropy": 1.0 - update * 0.01,
                "explained_variance": update * 0.01,
            }
            for action in ("stay", "ho_c1"):
                row[f"{action}_chosen_fraction"] = 0.5
                row[f"{action}_normalized_advantage_mean"] = float(update)
                row[f"{action}_raw_advantage_mean"] = float(update * 2)
                row[f"{action}_positive_advantage_fraction"] = 0.5
                row[f"{action}_probability_pre"] = 0.5
                row[f"{action}_probability_delta"] = 0.001
            rows.append(row)

        report = summarize_training_signals(rows)

        self.assertEqual(report["first_quarter"]["updates"], 2)
        self.assertEqual(report["last_quarter"]["first_update"], 7)
        self.assertAlmostEqual(
            report["last_quarter"]["stay_normalized_advantage_mean"]["mean"],
            7.5,
        )


if __name__ == "__main__":
    unittest.main()
