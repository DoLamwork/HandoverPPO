"""Tests for the B0/B0A early-HO baseline parity audit."""

import unittest

import numpy as np

from ho_optim_drl import Config
from ho_optim_drl.commands.audit_baseline_parity import run_pair
from ho_optim_drl.data import RadioTrace


class BaselineParityAuditTests(unittest.TestCase):
    def test_b0a_is_exactly_equal_to_b0_on_protocol_stress_trace(self) -> None:
        samples = 80
        rsrp = np.vstack(
            (
                np.r_[np.full(25, -80.0), np.full(30, -75.0), np.full(25, -82.0)],
                np.r_[np.full(25, -77.0), np.full(30, -81.0), np.full(25, -78.0)],
                np.full(samples, -90.0),
            )
        )
        rate = np.vstack(
            (np.full(samples, 4.0), np.full(samples, 2.0), np.full(samples, -8.0))
        )
        rlm = rate.copy()
        trace = RadioTrace.from_arrays(rsrp, rate, sinr_rlm_db=rlm, delta_t_ms=10)

        comparison, _, _ = run_pair(Config(), trace)

        self.assertTrue(comparison["passed"], comparison["mismatches"])
        self.assertEqual(comparison["mismatch_count"], 0)


if __name__ == "__main__":
    unittest.main()
