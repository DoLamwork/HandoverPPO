from __future__ import annotations

import unittest

from ho_optim_drl.commands.audit_t5a_benefit_sources import (
    _opportunity_bootstrap,
    _opportunity_records,
    _opportunity_summary,
    _source_classification,
)


def _physical(*, data: float, failures: int = 0, pp: int = 0) -> dict:
    return {
        "delivered_mbit": data,
        "relative_rate_percent": data * 10.0,
        "handover_failures": failures,
        "ping_pong_count": pp,
        "rlf_starts": failures,
        "recovery_seconds": float(failures),
        "interruption_seconds": 0.08 * failures,
    }


class T5ABenefitSourceAuditTests(unittest.TestCase):
    def test_opportunity_requires_positive_data_and_no_worse_risk(self) -> None:
        anchors = [{
        "state_id": "s0",
        "physical_state_fingerprint": "abc",
        "source": "b0",
        "category": "post_ttt",
        "speed_kmh": 30,
        "trajectory_id": 7,
        "tic": 12,
    }]
        outcomes = [
        {
            "state_id": "s0", "horizon_seconds": 1.0,
            "branch": "baseline_continuation", "physical": _physical(data=1.0),
        },
        {
            "state_id": "s0", "horizon_seconds": 1.0,
            "branch": "timing_schedule", "requested_delay_ms": 20,
            "physical": _physical(data=1.2, failures=1),
        },
        {
            "state_id": "s0", "horizon_seconds": 1.0,
            "branch": "timing_schedule", "requested_delay_ms": 40,
            "physical": _physical(data=1.1),
        },
    ]
        records = _opportunity_records(anchors, outcomes, horizon_seconds=1.0)
        self.assertEqual(len(records), 1)
        self.assertTrue(records[0]["has_safe_positive_opportunity"])
        self.assertEqual(records[0]["best_delay_ms"], 40)
        self.assertAlmostEqual(records[0]["best_delivered_mbit_gain"], 0.1)


    def test_opportunity_summary_keeps_pre_and_post_semantics_separate(self) -> None:
        rows = [
        {
            "category": "early_pre_a3", "has_safe_positive_opportunity": True,
            "physical_state_fingerprint": "a", "best_delivered_mbit_gain": 0.2,
            "best_relative_rate_gain_pp": 2.0, "best_delay_ms": 0,
        },
        {
            "category": "a3_pending_ttt", "has_safe_positive_opportunity": False,
            "physical_state_fingerprint": "b", "best_delivered_mbit_gain": 0.0,
            "best_relative_rate_gain_pp": 0.0, "best_delay_ms": None,
        },
        {
            "category": "post_ttt", "has_safe_positive_opportunity": True,
            "physical_state_fingerprint": "c", "best_delivered_mbit_gain": 0.1,
            "best_relative_rate_gain_pp": 1.0, "best_delay_ms": 20,
        },
    ]
        summary = _opportunity_summary(rows)
        self.assertEqual(summary["pre_a3_combined"]["positive_percent"], 50.0)
        self.assertEqual(summary["post_ttt"]["positive_percent"], 100.0)
        self.assertIn("post_ttt_combined", summary)

    def test_opportunity_bootstrap_clusters_complete_routes(self) -> None:
        rows = []
        for speed, trajectory in ((30, 1), (30, 2), (50, 1), (50, 2)):
            for category, positive in (
                ("early_pre_a3", True),
                ("post_ttt", False),
            ):
                rows.append({
                    "speed_kmh": speed,
                    "trajectory_id": trajectory,
                    "category": category,
                    "has_safe_positive_opportunity": positive,
                    "best_delivered_mbit_gain": 1.0 if positive else 0.0,
                })
        result = _opportunity_bootstrap(rows, resamples=20, seed=9)
        self.assertEqual(
            result["point"]["delta_pre_minus_post_positive_pp"], 100.0
        )
        self.assertEqual(
            result["ci95"]["delta_pre_minus_post_positive_pp"],
            [100.0, 100.0],
        )


    def test_realised_benefit_source_marks_reliability_rate_tradeoff(self) -> None:
        b0 = {
        "relative_rate_percent": 90.0,
        "handover_failures": 2,
        "ping_pong_count": 3,
        "rlf_recovery_samples": 40,
    }
        policy = {
        "relative_rate_percent": 89.5,
        "handover_failures": 1,
        "ping_pong_count": 2,
        "rlf_recovery_samples": 20,
    }
        self.assertEqual(
            _source_classification(b0, policy),
            "reliability_gain_rate_tradeoff",
        )


if __name__ == "__main__":
    unittest.main()
