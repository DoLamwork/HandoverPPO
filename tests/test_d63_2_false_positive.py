import tempfile
import unittest
from pathlib import Path

import numpy as np

from ho_optim_drl.commands.analyze_d63_2_false_positives import (
    _prepare_destination,
    analyze_operating_point,
    false_positive_reasons,
)


class D632FalsePositiveTests(unittest.TestCase):
    def _arrays(self):
        arrays = {
            "state_id": np.asarray(["a", "a", "b", "b"]),
            "is_control": np.asarray([True, False, True, False]),
            "requested_action_valid": np.asarray([True, True, True, True]),
            "strict_useful": np.asarray([False, False, False, True]),
            "source": np.asarray(["d61", "d61", "d62", "d62"]),
            "speed_kmh": np.asarray([50, 50, 70, 70]),
            "planned_slot": np.asarray([1, 2, 1, 3]),
            "delta_delivered_mbit": np.asarray([0.0, -1.0, 0.0, 2.0]),
            "delta_relative_rate_percent": np.asarray([0.0, -1.0, 0.0, 2.0]),
        }
        for field in (
            "delta_handover_failures", "delta_rlf_starts", "delta_pure_rlf_starts",
            "delta_ping_pong_count", "delta_recovery_seconds",
            "delta_handover_interruption_seconds",
        ):
            arrays[field] = np.asarray([0.0, 1.0 if field != "delta_ping_pong_count" else 0.0, 0.0, 0.0])
        return arrays

    def test_action_false_positive_breaks_down_speed_source_and_risk(self):
        report, indices = analyze_operating_point(
            self._arrays(), np.asarray([0.0, 0.9, 0.0, 0.8]), 0.5
        )
        self.assertEqual(report["changed_actions"], 2)
        self.assertEqual(report["false_positive_actions"], 1)
        self.assertEqual(report["breakdown"]["speed"]["50"]["false_positive_actions"], 1)
        self.assertEqual(report["breakdown"]["source"]["d61"]["false_positive_actions"], 1)
        self.assertEqual(report["false_positive_reasons"]["delta_handover_failures"]["rows_worse"], 1)
        np.testing.assert_array_equal(indices, np.asarray([1]))

    def test_empty_false_positive_reason_is_well_formed(self):
        reasons = false_positive_reasons(self._arrays(), np.asarray([], dtype=int))
        self.assertEqual(reasons["no_delivered_data_gain"]["rows"], 0)
        self.assertIsNone(reasons["delta_rlf_starts"]["mean_delta_all_false_positives"])

    def test_output_refuses_overwrite(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            path = Path(temporary) / "output"
            path.mkdir()
            (path / "keep.txt").write_text("keep", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                _prepare_destination(path)


if __name__ == "__main__":
    unittest.main()
