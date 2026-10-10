import tempfile
import unittest
from pathlib import Path

import numpy as np

from ho_optim_drl.commands.replay_d63_2_failure_cases import (
    _prepare_destination,
    causal_radio_features,
)
from ho_optim_drl.data import RadioTrace


class D632FailureReplayTests(unittest.TestCase):
    def test_causal_feature_does_not_read_future_samples(self):
        samples = 80
        rsrp = np.vstack((np.linspace(-90, -80, samples), np.linspace(-88, -78, samples)))
        sinr = np.vstack((np.linspace(-4, 4, samples), np.linspace(-2, 6, samples)))
        trace = RadioTrace.from_arrays(rsrp, sinr, delta_t_ms=10)
        first = causal_radio_features(
            trace=trace, serving_cell=0, target_cell=1, anchor_tic=50,
            q_in_db=-6.0, delta_t_ms=10,
        )
        changed = np.array(sinr, copy=True)
        changed[:, 51:] = -100.0
        changed_trace = RadioTrace.from_arrays(rsrp, changed, delta_t_ms=10)
        second = causal_radio_features(
            trace=changed_trace, serving_cell=0, target_cell=1, anchor_tic=50,
            q_in_db=-6.0, delta_t_ms=10,
        )
        self.assertEqual(first, second)

    def test_execution_safety_margin_penalizes_long_window_volatility(self):
        samples = 60
        rsrp = np.zeros((2, samples))
        stable = np.vstack((np.zeros(samples), np.ones(samples)))
        volatile = np.array(stable, copy=True)
        volatile[1, :51] += np.tile(np.asarray([-5.0, 5.0]), 26)[:51]
        stable_feature = causal_radio_features(
            trace=RadioTrace.from_arrays(rsrp, stable), serving_cell=0,
            target_cell=1, anchor_tic=50, q_in_db=-6.0, delta_t_ms=10,
        )
        volatile_feature = causal_radio_features(
            trace=RadioTrace.from_arrays(rsrp, volatile), serving_cell=0,
            target_cell=1, anchor_tic=50, q_in_db=-6.0, delta_t_ms=10,
        )
        self.assertLess(
            volatile_feature["execution_safety_margin_v1_db"],
            stable_feature["execution_safety_margin_v1_db"],
        )

    def test_output_refuses_overwrite(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            path = Path(temporary) / "output"
            path.mkdir()
            (path / "keep.txt").write_text("keep", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                _prepare_destination(path)


if __name__ == "__main__":
    unittest.main()
