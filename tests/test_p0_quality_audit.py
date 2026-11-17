import unittest

import numpy as np

from ho_optim_drl.commands.audit_p0_quality import (
    _normalized_entropy,
    _percentiles,
    _quality_gates,
)


class P0QualityAuditTests(unittest.TestCase):
    def test_percentiles_and_entropy(self):
        values = _percentiles(np.arange(100, dtype=float))
        self.assertAlmostEqual(values["p50"], 49.5)
        self.assertAlmostEqual(_normalized_entropy([10, 10, 10]), 1.0)
        self.assertEqual(_normalized_entropy([10]), 0.0)

    def test_quality_gates_require_every_speed(self):
        group = {
            "radio": {"rank_switches": 1, "unique_winner_cells": 3},
            "protocol": {
                "ttt_starts": 2,
                "a3_maturities": 1,
                "mean_route_relative_rate_percent": 90.0,
                "handover_failure_percent": 5.0,
                "rlf_time_percent": 1.0,
            },
        }
        aggregate = {
            "routes": 120,
            "radio": {
                "best_rsrp_dbm": {"p1": -90.0, "p99": -60.0},
                "best_sinr_db": {"p1": -10.0, "p99": 10.0},
            },
        }
        complete = {str(speed): group for speed in (30, 50, 70, 90, 120, 150)}
        self.assertTrue(_quality_gates(aggregate, complete)["ready_for_main_or_training_pilot"])
        del complete["150"]
        self.assertFalse(_quality_gates(aggregate, complete)["ready_for_main_or_training_pilot"])


if __name__ == "__main__":
    unittest.main()

