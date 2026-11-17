"""Tests for T2 sampling and T3 route-cluster inference."""

import unittest

from ho_optim_drl.commands.audit_early_timing_bank import _balanced_selection
from ho_optim_drl.commands.audit_early_timing_routes import evaluate_gates, paired_bootstrap


def _metric(rate: float, *, failures: int = 0, ping_pongs: int = 0) -> dict:
    attempts = 2
    successes = attempts - failures
    return {
        "relative_rate_percent": rate,
        "estimated_rate_mbps": rate / 4.0,
        "max_estimated_rate_mbps": 25.0,
        "total_samples": 100,
        "rlf_recovery_samples": failures * 2,
        "handover_attempts": attempts,
        "handover_successes": successes,
        "handover_failures": failures,
        "ping_pong_count": ping_pongs,
        "rlf_starts": failures,
        "ttt_starts": 2,
        "ttt_aborts": 0,
        "ho_prep_start_tics": [],
    }


class EarlyTimingMainAuditTests(unittest.TestCase):
    def test_balanced_bank_selection_keeps_each_available_category(self) -> None:
        pools = {
            "early_pre_a3": [
                {"score": 4, "record": {"tic": 4}},
                {"score": 1, "record": {"tic": 1}},
            ],
            "a3_pending_ttt": [{"score": 2, "record": {"tic": 2}}],
            "post_ttt": [{"score": 3, "record": {"tic": 3}}],
        }

        selected = _balanced_selection(pools, 3)

        self.assertEqual({item["record"]["tic"] for item in selected}, {1, 2, 3})

    def test_bootstrap_is_route_clustered_stratified_and_reproducible(self) -> None:
        rows = []
        for speed in (30, 50):
            for trajectory in (1, 2, 3):
                rows.append({
                    "speed_kmh": speed,
                    "trajectory_id": trajectory,
                    "b0": _metric(90.0, failures=1, ping_pongs=1),
                    "rearly": _metric(91.0, failures=0, ping_pongs=0),
                })

        first = paired_bootstrap(
            rows, left="b0", right="rearly", dt_ms=10,
            resamples=50, seed=29,
        )
        second = paired_bootstrap(
            rows, left="b0", right="rearly", dt_ms=10,
            resamples=50, seed=29,
        )

        self.assertEqual(first, second)
        self.assertTrue(first["stratified_by_speed"])
        self.assertEqual(
            first["delta_right_minus_left"]["mean_route_relative_rate_percent"],
            1.0,
        )
        self.assertEqual(
            first["ci95"]["mean_route_relative_rate_percent"], [1.0, 1.0]
        )

    def test_qos_gate_uses_retention_vs_b0_not_absolute_relative_rate(self) -> None:
        aggregate = {
            "b0": {
                "mean_route_estimated_rate_mbps": 20.0,
                "mean_route_relative_rate_percent": 90.0,
                "handover_failure_percent": 10.0,
                "ping_pong_percent": 20.0,
            },
            "rearly": {
                "mean_route_estimated_rate_mbps": 20.0,
                "mean_route_relative_rate_percent": 90.0,
                "handover_failure_percent": 8.0,
                "ping_pong_percent": 15.0,
            },
        }
        delta = {
            "mean_route_relative_rate_percent": 1.0,
            "handover_failure_percent": -2.0,
            "ping_pong_percent": -5.0,
            "rlf_time_percent": -1.0,
        }
        comparisons = {
            "rearly_minus_b0": {
                "delta_right_minus_left": delta,
                "ci95": {
                    "mean_route_relative_rate_percent": [0.5, 1.5],
                    "handover_failure_percent": [-3.0, -1.0],
                    "ping_pong_percent": [-6.0, -4.0],
                    "rlf_time_percent": [-1.5, -0.5],
                },
            },
            "rearly_minus_rpost": {
                "delta_right_minus_left": delta,
                "ci95": {
                    "mean_route_relative_rate_percent": [0.1, 0.9],
                    "handover_failure_percent": [-1.0, 0.0],
                    "ping_pong_percent": [-1.0, 0.0],
                    "rlf_time_percent": [-1.0, 0.0],
                },
            },
        }
        per_speed = {
            "30": {
                "b0": {
                    "mean_route_estimated_rate_mbps": 20.0,
                    "mean_route_relative_rate_percent": 90.0,
                },
                "rearly": {
                    "mean_route_estimated_rate_mbps": 19.8,
                    "mean_route_relative_rate_percent": 89.0,
                },
            }
        }

        gates = evaluate_gates(aggregate, comparisons, per_speed)

        self.assertTrue(gates["g_qos_guard"]["passed"])
        self.assertEqual(
            gates["g_qos_guard"]["definition_version"],
            "qos_retention_vs_b0_v2",
        )
        self.assertAlmostEqual(
            gates["g_qos_guard"]["aggregate_relative_rate_retention_percent"],
            100.0,
        )


if __name__ == "__main__":
    unittest.main()
