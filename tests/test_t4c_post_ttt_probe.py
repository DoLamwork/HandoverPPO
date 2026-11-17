"""Tests for T4C split and paired local inference."""

import unittest

import numpy as np

from ho_optim_drl.commands.audit_t4c_post_ttt_probe import (
    _folds_from_split,
    _paired_selection_bootstrap,
)


class T4CPostTttProbeTests(unittest.TestCase):
    def test_folds_are_read_from_frozen_split(self) -> None:
        split = {"selected_routes": [
            {"speed_kmh": 30, "trajectory_id": 1, "fold": "fit"},
            {"speed_kmh": 30, "trajectory_id": 2, "fold": "tuning"},
            {"speed_kmh": 30, "trajectory_id": 3, "fold": "confirmation"},
        ]}
        self.assertEqual(_folds_from_split(split), {
            (30, 1): "fit", (30, 2): "tuning", (30, 3): "confirmation",
        })

    def test_paired_bootstrap_preserves_route_clusters(self) -> None:
        samples = []
        for speed in (30, 50):
            for route in (1, 2):
                samples.append({
                    "speed_kmh": speed, "trajectory_id": route,
                    "label": {
                        "delta_delivered_mbit": 1.0,
                        "risk_deltas": {
                            "handover_failures": 0.0, "ping_pong_count": 0.0,
                            "rlf_starts": 0.0, "recovery_seconds": 0.0,
                            "interruption_seconds": 0.0,
                        },
                    },
                })
        result = _paired_selection_bootstrap(
            samples, np.zeros(4, dtype=bool), np.ones(4, dtype=bool),
            seed=29, resamples=20,
        )
        self.assertEqual(result["routes"], 4)
        self.assertTrue(result["stratified_by_speed"])
        self.assertEqual(result["mean"]["delta_delivered_mbit"], 1.0)
        self.assertEqual(result["ci95"]["delta_delivered_mbit"], [1.0, 1.0])


if __name__ == "__main__":
    unittest.main()
