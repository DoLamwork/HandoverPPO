"""Contract tests for the B1 reduced-A3-offset control."""

from dataclasses import asdict
import unittest

import numpy as np

from ho_optim_drl import Config
from ho_optim_drl.commands.audit_b1_lower_threshold import _route_metrics, b1_config
from ho_optim_drl.data import RadioTrace
from ho_optim_drl.protocol import A3ReferenceSimulator


class B1ContractTests(unittest.TestCase):
    def test_b1_changes_only_a3_offset(self) -> None:
        b0 = Config()
        b1 = b1_config(b0)
        expected = asdict(b0)
        expected["a3_off"] = 0.0

        self.assertEqual(asdict(b1), expected)
        self.assertEqual(b0.a3_off, 1.0)

    def test_b1_triggers_when_delta_is_between_one_and_two_db(self) -> None:
        samples = 12
        trace = RadioTrace.from_arrays(
            np.vstack((np.full(samples, -80.0), np.full(samples, -78.5))),
            np.vstack((np.full(samples, 4.0), np.full(samples, 3.0))),
            delta_t_ms=10,
        )
        b0 = A3ReferenceSimulator(Config())
        b1 = A3ReferenceSimulator(b1_config(Config()))

        b0.predict(trace)
        b1.predict(trace)

        self.assertEqual(b0.cntr["ho_prep"].start_idxs, [])
        self.assertEqual(b1.cntr["ho_prep"].start_idxs, [4])
        metrics = _route_metrics(b1, trace)
        self.assertEqual(metrics["ttt_starts"], 1)
        self.assertEqual(metrics["ttt_aborts"], 0)


if __name__ == "__main__":
    unittest.main()
