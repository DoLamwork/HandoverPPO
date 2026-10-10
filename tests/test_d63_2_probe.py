import tempfile
import unittest
from pathlib import Path

import numpy as np

from ho_optim_drl.commands.probe_d63_2 import (
    _prepare_destination,
    build_features,
    choose_actions,
    select_threshold,
)


class D632ProbeTests(unittest.TestCase):
    def test_features_only_add_delay_and_slot_to_observation(self):
        arrays = {
            "observation": np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            "planned_delay_ms": np.asarray([0, 300]),
            "planned_slot": np.asarray([1, 3]),
            "source": np.asarray(["forbidden_a", "forbidden_b"]),
        }
        features = build_features(arrays)
        self.assertEqual(features.shape, (2, 6))
        np.testing.assert_array_equal(features[:, :2], arrays["observation"])
        np.testing.assert_array_equal(features[:, 2], np.asarray([0.0, 1.0]))
        np.testing.assert_array_equal(features[:, 3:], np.asarray([[1, 0, 0], [0, 0, 1]]))

    def test_action_selection_uses_one_branch_or_control_per_state(self):
        arrays = {
            "state_id": np.asarray(["a", "a", "a", "b", "b"]),
            "is_control": np.asarray([True, False, False, True, False]),
            "requested_action_valid": np.asarray([True, True, True, True, True]),
        }
        chosen = choose_actions(arrays, np.asarray([0.0, 0.6, 0.8, 0.0, 0.4]), 0.5)
        np.testing.assert_array_equal(chosen, np.asarray([2, 3]))

    def test_threshold_tuning_has_safe_control_fallback(self):
        arrays = {
            "state_id": np.asarray(["a", "a"]),
            "is_control": np.asarray([True, False]),
            "requested_action_valid": np.asarray([True, True]),
            "delta_delivered_mbit": np.asarray([0.0, 1.0]),
            "delta_relative_rate_percent": np.asarray([0.0, 1.0]),
            "strict_useful": np.asarray([False, False]),
        }
        for field in (
            "delta_handover_failures", "delta_rlf_starts", "delta_pure_rlf_starts",
            "delta_ping_pong_count", "delta_recovery_seconds",
            "delta_handover_interruption_seconds",
        ):
            arrays[field] = np.asarray([0.0, 1.0])
        threshold, record = select_threshold(arrays, np.asarray([0.0, 0.99]))
        self.assertGreater(threshold, 1.0)
        self.assertEqual(record["selected"]["changed_actions"], 0)
        self.assertTrue(record["selected"]["reliability_not_worse"])

    def test_output_refuses_overwrite(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            path = Path(temporary) / "output"
            path.mkdir()
            (path / "keep.txt").write_text("keep", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                _prepare_destination(path)


if __name__ == "__main__":
    unittest.main()
