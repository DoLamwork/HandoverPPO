"""Tests for the post-TTT confirmation full-route SINR-rule audit."""

import unittest

from ho_optim_drl.commands.audit_sinr_rule_fullroute import (
    _evaluate_gates,
    _fullroute_parity,
)


class SinrRuleFullRouteTests(unittest.TestCase):
    def test_terminal_censored_prep_is_the_only_tolerated_parity_delta(self) -> None:
        class Counter:
            def __init__(self, starts):
                self.start_idxs = starts

        class Fake:
            pass

        b0 = Fake(); b0a = Fake(); trace = Fake(); trace.num_samples = 10
        b0.cntr = {"ho_prep": Counter([3, 9])}
        b0a.cntr = {"ho_prep": Counter([3])}
        import ho_optim_drl.commands.audit_sinr_rule_fullroute as module
        old_shadow = module._shadow_b0_parity
        old_route = module._route_metrics
        old_metrics = module._metrics
        study = {
            "relative_rate_percent": 90.0, "estimated_rate_mbps": 10.0,
            "max_estimated_rate_mbps": 11.0, "total_samples": 10,
            "rlf_recovery_samples": 0, "handover_attempts": 1,
            "handover_successes": 1, "handover_failures": 0,
            "ping_pong_count": 0, "rlf_starts": 0,
        }
        try:
            module._shadow_b0_parity = lambda left, right: {
                "passed": False, "mismatches": ["ho_prep.start_idxs"]
            }
            module._route_metrics = lambda simulator, value: study
            module._metrics = lambda simulator, value: dict(study)
            result = _fullroute_parity(b0, b0a, trace)
            self.assertTrue(result["passed"])
            self.assertEqual(result["mode"], "terminal_censored_preparation")
            b0.cntr["ho_prep"].start_idxs = [3, 8]
            self.assertFalse(_fullroute_parity(b0, b0a, trace)["passed"])
        finally:
            module._shadow_b0_parity = old_shadow
            module._route_metrics = old_route
            module._metrics = old_metrics

    def test_gate_requires_qos_and_all_reliability_bounds(self) -> None:
        aggregate = {
            "b0": {
                "mean_route_estimated_rate_mbps": 10.0,
                "mean_route_relative_rate_percent": 90.0,
            },
            "sinr_rule": {
                "mean_route_estimated_rate_mbps": 10.1,
                "mean_route_relative_rate_percent": 91.0,
            },
        }
        per_speed = {
            "30": {"b0": aggregate["b0"], "sinr_rule": aggregate["sinr_rule"]}
        }
        metrics = {
            "mean_route_relative_rate_percent": 1.0,
            "mean_route_estimated_rate_mbps": 0.1,
            "handover_failure_percent": -1.0,
            "ping_pong_percent": -1.0,
            "rlf_time_percent": -0.1,
        }
        comparison = {
            "delta_right_minus_left": metrics,
            "ci95": {
                name: ([0.1, 1.5] if name == "mean_route_relative_rate_percent" else [-2.0, 0.0])
                for name in metrics
            },
        }
        gates = _evaluate_gates(
            aggregate, per_speed, comparison, parity_passed=True
        )
        self.assertEqual(gates["strict_all"], "PASSED")
        comparison["ci95"]["ping_pong_percent"] = [-2.0, 0.1]
        gates = _evaluate_gates(
            aggregate, per_speed, comparison, parity_passed=True
        )
        self.assertEqual(gates["strict_all"], "NOT_PASSED")


if __name__ == "__main__":
    unittest.main()
