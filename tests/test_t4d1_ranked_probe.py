"""Tests for T4D.1 route splitting, calibration and selector locking."""

import unittest

import numpy as np

from ho_optim_drl.commands.audit_t4d1_ranked_probe import (
    _calibrate_temperature,
    _inner_route_masks,
    _lock_selector,
    _ranking_loss,
)


def _label(gain: float, risk: float = 0.0) -> dict:
    return {
        "delta_delivered_mbit": gain,
        "delta_relative_rate_percent": gain,
        "risk_deltas": {
            "handover_failures": risk,
            "ping_pong_count": 0.0,
            "rlf_starts": 0.0,
            "recovery_seconds": 0.0,
            "interruption_seconds": 0.0,
        },
        "wait_useful": gain > 0 and risk <= 0,
    }


class T4D1RankedProbeTests(unittest.TestCase):
    def test_inner_split_keeps_routes_disjoint_and_speed_stratified(self) -> None:
        rows = []
        for speed in (30, 50):
            for route in range(10):
                rows.extend([
                    {"speed_kmh": speed, "trajectory_id": route},
                    {"speed_kmh": speed, "trajectory_id": route},
                ])
        fit = np.ones(len(rows), dtype=bool)
        train, validation = _inner_route_masks(rows, fit, seed=29)
        train_routes = {
            (row["speed_kmh"], row["trajectory_id"])
            for row, keep in zip(rows, train) if keep
        }
        validation_routes = {
            (row["speed_kmh"], row["trajectory_id"])
            for row, keep in zip(rows, validation) if keep
        }
        self.assertFalse(train_routes & validation_routes)
        self.assertEqual({speed for speed, _ in validation_routes}, {30, 50})

    def test_ranking_loss_prefers_ordered_logits(self) -> None:
        import torch

        labels = torch.tensor([True, True, False, False])
        speeds = torch.tensor([30, 30, 30, 30])
        mask = torch.ones(4, dtype=torch.bool)
        good = _ranking_loss(torch.tensor([2.0, 1.0, -1.0, -2.0]), labels, speeds, mask)
        bad = _ranking_loss(torch.tensor([-2.0, -1.0, 1.0, 2.0]), labels, speeds, mask)
        self.assertLess(float(good), float(bad))

    def test_temperature_calibration_is_selected_only_from_grid(self) -> None:
        value, report = _calibrate_temperature(
            np.asarray([4.0, 3.0, -4.0, -3.0]),
            np.asarray([1.0, 0.0, 0.0, 1.0]),
        )
        self.assertIn(value, {row["temperature"] for row in report["candidates"]})
        self.assertEqual(value, report["selected"]["temperature"])

    def test_selector_requires_action_support_and_filters_risk(self) -> None:
        labels = [_label(1.0), _label(0.5), _label(-1.0), _label(0.8, 1.0)]
        qos = np.asarray([0.95, 0.90, 0.10, 0.99])
        risk = np.asarray([0.05, 0.10, 0.05, 0.95])
        qt, rt, report = _lock_selector(
            labels, qos, risk, minimum_wait_actions=2
        )
        choose = (qos >= qt) & (risk <= rt)
        self.assertGreaterEqual(int(np.sum(choose)), 2)
        self.assertFalse(bool(choose[3]))
        self.assertTrue(report["selected"]["reliability_not_worse"])


if __name__ == "__main__":
    unittest.main()
