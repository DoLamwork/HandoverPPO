import tempfile
import unittest
from pathlib import Path

import numpy as np

from ho_optim_drl.commands.validate_d63_3_feature import (
    _json_write,
    _prepare_destination,
    candidate_thresholds,
    lock_threshold,
)


def _arrays() -> dict[str, np.ndarray]:
    # Two states, each with registered C1 control plus C2/C3 alternatives.
    return {
        "state_id": np.asarray(["a", "a", "a", "b", "b", "b"]),
        "is_control": np.asarray([True, False, False, True, False, False]),
        "requested_action_valid": np.ones(6, dtype=bool),
        "strict_useful": np.asarray([False, True, False, False, False, False]),
        "delta_delivered_mbit": np.asarray([0, 1, -1, 0, -2, -3], dtype=float),
        "delta_relative_rate_percent": np.asarray([0, 2, -2, 0, -3, -4], dtype=float),
        "speed_kmh": np.asarray([30] * 6),
        "trajectory_id": np.asarray([1, 1, 1, 2, 2, 2]),
        "source": np.asarray(["d61"] * 6),
        **{field: np.zeros(6, dtype=float) for field in (
            "delta_handover_failures", "delta_rlf_starts",
            "delta_pure_rlf_starts", "delta_ping_pong_count",
            "delta_recovery_seconds", "delta_handover_interruption_seconds",
        )},
    }


class D633FeatureValidationTests(unittest.TestCase):
    def test_thresholds_include_control_only_sentinel(self):
        values = candidate_thresholds(np.asarray([-2.0, 1.0, 1.0]))
        self.assertTrue(np.isinf(values[-1]))
        np.testing.assert_array_equal(values[:-1], np.asarray([-2.0, 1.0]))

    def test_fit_locks_safe_profitable_threshold(self):
        arrays = _arrays()
        scores = np.asarray([0, 2, -1, 0, -2, -3], dtype=float)
        threshold, audit = lock_threshold(arrays, scores)
        self.assertEqual(threshold, 2.0)
        self.assertEqual(audit["selection_fold"], "fit")
        self.assertEqual(audit["selected"]["changed_actions"], 1)
        self.assertGreater(audit["selected"]["mean_delta_delivered_mbit"], 0)
        self.assertEqual(
            audit["best_non_control_reliability_feasible"]["changed_actions"], 1
        )

    def test_output_refuses_overwrite(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            path = Path(temporary) / "output"
            path.mkdir()
            (path / "keep.txt").write_text("keep", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                _prepare_destination(path)

    def test_json_encodes_control_only_threshold(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            path = Path(temporary) / "report.json"
            _json_write(path, {"threshold": float("inf")})
            self.assertIn('"CONTROL_ONLY"', path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
