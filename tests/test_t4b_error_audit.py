"""Tests for frozen T4B error categorization."""

import unittest

import numpy as np

from ho_optim_drl.commands.analyze_t4b_errors import (
    benefit_reasons,
    confusion_labels,
    risk_reasons,
)


class T4BErrorAuditTests(unittest.TestCase):
    def test_confusion_labels_cover_all_four_cases(self) -> None:
        actual = confusion_labels(
            np.asarray([True, False, True, False]),
            np.asarray([True, True, False, False]),
        )
        self.assertEqual(actual.tolist(), ["tp", "fp", "fn", "tn"])

    def test_risk_reasons_preserve_gain_and_multiple_risks(self) -> None:
        label = {
            "delta_delivered_mbit": -0.1,
            "risk_deltas": {
                "handover_failures": 1.0,
                "ping_pong_count": 1.0,
                "rlf_starts": 0.0,
                "recovery_seconds": 0.2,
                "interruption_seconds": 0.0,
            },
        }
        self.assertEqual(
            risk_reasons(label),
            ["no_positive_data_gain", "increased_hof", "increased_ping_pong", "increased_recovery"],
        )

    def test_benefit_reasons_preserve_multiple_avoided_risks(self) -> None:
        label = {
            "delta_delivered_mbit": 0.2,
            "risk_deltas": {
                "handover_failures": -1.0,
                "ping_pong_count": 0.0,
                "rlf_starts": -1.0,
                "recovery_seconds": -0.2,
                "interruption_seconds": -0.08,
            },
        }
        self.assertEqual(
            benefit_reasons(label),
            ["positive_data_gain", "avoided_hof", "avoided_rlf", "reduced_recovery", "reduced_interruption"],
        )


if __name__ == "__main__":
    unittest.main()
