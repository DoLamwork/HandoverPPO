"""Tests for T4 post-A3 causal learnability helpers."""

import unittest
from types import SimpleNamespace

import numpy as np

from ho_optim_drl.commands.audit_early_timing_learnability import (
    _classification_groups,
    causal_features,
    stratified_route_folds,
    timing_label,
)


class _Trace:
    def __init__(self) -> None:
        self.num_samples = 20
        base = np.arange(20, dtype=float)
        self.rsrp_a3_db = np.vstack((-100 + base, -98 + 2 * base))
        self.sinr_rate_db = np.vstack((-5 + base / 10, -4 + base / 5))
        self.sinr_rlm_db = self.sinr_rate_db.copy()


class EarlyTimingLearnabilityTests(unittest.TestCase):
    def test_features_do_not_read_future(self) -> None:
        trace = _Trace()
        before = causal_features(trace, tic=10, serving=0, target=1, a3_start_tic=6, speed_kmh=50, q_out_db=-8, dt_ms=10)
        trace.rsrp_a3_db[:, 11:] = 9999
        trace.sinr_rate_db[:, 11:] = -9999
        trace.sinr_rlm_db[:, 11:] = 9999
        after = causal_features(trace, tic=10, serving=0, target=1, a3_start_tic=6, speed_kmh=50, q_out_db=-8, dt_ms=10)
        np.testing.assert_array_equal(before, after)

    def test_t4b_features_require_snapshot_and_remain_causal(self) -> None:
        trace = _Trace()
        decision = SimpleNamespace()
        snapshot = SimpleNamespace(
            tic=10, decision_point=decision, n310_progress=0.2,
            n311_progress=0.0, t310_active=False, t310_progress=0.0,
            serving_residence_ticks=100, time_since_last_ho_ticks=80,
            previous_cell=1, recent_ho_count=1,
        )
        kwargs = dict(
            tic=10, serving=0, target=1, a3_start_tic=6,
            speed_kmh=50, q_out_db=-8, dt_ms=10,
            snapshot=snapshot, schema="t4b_protocol_history_v1",
        )
        before = causal_features(trace, **kwargs)
        self.assertEqual(len(before), 40)
        trace.rsrp_a3_db[:, 11:] = 9999
        trace.sinr_rate_db[:, 11:] = -9999
        trace.sinr_rlm_db[:, 11:] = 9999
        np.testing.assert_array_equal(before, causal_features(trace, **kwargs))
        with self.assertRaises(ValueError):
            causal_features(trace, **{**kwargs, "snapshot": None})

    def test_route_split_is_stratified_reproducible_and_disjoint(self) -> None:
        routes = [(speed, route) for speed in (30, 50, 70, 90) for route in range(20)]
        first = stratified_route_folds(routes, seed=29)
        second = stratified_route_folds(routes, seed=29)
        self.assertEqual(first, second)
        for speed in (30, 50, 70, 90):
            counts = {fold: sum(value == fold for (item_speed, _), value in first.items() if item_speed == speed) for fold in ("fit", "tuning", "holdout")}
            self.assertEqual(counts, {"fit": 12, "tuning": 4, "holdout": 4})

    def test_wait_label_keeps_gain_and_risk_separate(self) -> None:
        now = {"delivered_mbit": 1.0, "relative_rate_percent": 80.0, **{name: 1.0 for name in ("handover_failures", "ping_pong_count", "rlf_starts", "recovery_seconds", "interruption_seconds")}}
        wait = {"delivered_mbit": 1.2, "relative_rate_percent": 82.0, **{name: 1.0 for name in ("handover_failures", "ping_pong_count", "rlf_starts", "recovery_seconds", "interruption_seconds")}}
        result = timing_label(now, wait)
        self.assertTrue(result["wait_useful"])
        self.assertAlmostEqual(result["delta_delivered_mbit"], 0.2)
        wait["handover_failures"] = 2.0
        self.assertFalse(timing_label(now, wait)["wait_useful"])

    def test_classification_groups_keep_overall_speed_and_source(self) -> None:
        samples = [
            {"speed_kmh": 30, "source": "b0"},
            {"speed_kmh": 50, "source": "b2"},
        ]
        result = _classification_groups(
            samples,
            np.asarray([True, False]),
            np.asarray([0.9, 0.1]),
            np.asarray([True, True]),
            0.5,
        )
        self.assertEqual(
            set(result),
            {"overall", "speed_kmh:30", "speed_kmh:50", "source:b0", "source:b2"},
        )


if __name__ == "__main__":
    unittest.main()
