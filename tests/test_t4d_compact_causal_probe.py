"""Unit tests for T4D compact causal QoS/risk helpers."""

import unittest

import numpy as np

from ho_optim_drl.commands.audit_t4d_compact_causal_probe import (
    QOS_FEATURE_NAMES,
    RISK_FEATURE_NAMES,
    _feature_indices,
    _lock_decision_thresholds,
    _meaningful_wait,
    _route_weights,
)


def _label(gain: float, risk: float = 0.0) -> dict:
    risks = {
        "handover_failures": risk,
        "ping_pong_count": 0.0,
        "rlf_starts": 0.0,
        "recovery_seconds": 0.0,
        "interruption_seconds": 0.0,
    }
    return {
        "label": {
            "delta_delivered_mbit": gain,
            "delta_relative_rate_percent": gain,
            "risk_deltas": risks,
            "wait_useful": gain > 0 and risk <= 0,
        }
    }


class T4DCompactCausalProbeTests(unittest.TestCase):
    def test_compact_feature_contract_is_unique_and_available(self) -> None:
        qos = _feature_indices(QOS_FEATURE_NAMES)
        risk = _feature_indices(RISK_FEATURE_NAMES)
        self.assertEqual(len(qos), len(set(qos.tolist())))
        self.assertEqual(len(risk), len(set(risk.tolist())))
        self.assertNotIn("speed_kmh_div_150", QOS_FEATURE_NAMES)
        self.assertNotIn("route_progress", RISK_FEATURE_NAMES)
        self.assertNotIn("serving_rlm_margin_qout_div_20", RISK_FEATURE_NAMES)

    def test_meaningful_wait_requires_material_gain_and_safety(self) -> None:
        self.assertFalse(_meaningful_wait(_label(0.01), 0.05))
        self.assertTrue(_meaningful_wait(_label(0.05), 0.05))
        self.assertFalse(_meaningful_wait(_label(0.50, 1.0), 0.05))

    def test_route_weights_equalize_route_mass(self) -> None:
        rows = [
            {"speed_kmh": 30, "trajectory_id": 1},
            {"speed_kmh": 30, "trajectory_id": 1},
            {"speed_kmh": 30, "trajectory_id": 1},
            {"speed_kmh": 30, "trajectory_id": 2},
        ]
        weights = _route_weights(rows, np.ones(4, dtype=bool))
        self.assertAlmostEqual(float(np.sum(weights[:3])), float(weights[3]))
        self.assertAlmostEqual(float(np.mean(weights)), 1.0)

    def test_threshold_lock_rejects_predicted_risk(self) -> None:
        labels = [_label(1.0)["label"], _label(-1.0)["label"], _label(0.5, 1.0)["label"]]
        gain = np.asarray([0.8, -0.5, 0.9])
        risk = np.asarray([0.01, 0.01, 0.9])
        gain_threshold, risk_threshold, report = _lock_decision_thresholds(
            labels, gain, risk, minimum_gain_mbit=0.05
        )
        choose = (gain >= gain_threshold) & (risk <= risk_threshold)
        self.assertEqual(choose.tolist(), [True, False, False])
        self.assertTrue(report["selected"]["reliability_not_worse"])


if __name__ == "__main__":
    unittest.main()
