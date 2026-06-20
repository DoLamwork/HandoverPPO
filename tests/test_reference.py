"""Focused regression tests for the A3 reference implementation."""

import unittest

import numpy as np

from ho_optim_drl.config import Config
from ho_optim_drl.protocol.primitives import HOEventHandler
from ho_optim_drl.protocol.reference import A3ReferenceSimulator


class A3EventTests(unittest.TestCase):
    def test_a3_entering_deadband_and_leaving(self) -> None:
        event = HOEventHandler(hys_a3=1.0, off_a3=1.0, debug_lvl=0)

        self.assertFalse(event.event_a3(np.array([-80.0, -78.0]), 0, 1))
        self.assertTrue(event.event_a3(np.array([-80.0, -77.9]), 0, 1))
        self.assertTrue(event.event_a3(np.array([-80.0, -79.0]), 0, 1))
        self.assertFalse(event.event_a3(np.array([-80.0, -80.1]), 0, 1))


class ConfigurationTests(unittest.TestCase):
    def test_default_timer_units_are_converted_to_steps(self) -> None:
        config = Config()

        self.assertEqual(config.a3_ttt_ms, 40)
        self.assertEqual(config.a3_ttt, 4)
        self.assertEqual(config.t_t310_ms, 1000)
        self.assertEqual(config.t_t310, 100)

    def test_non_standard_ttt_is_rejected(self) -> None:
        config = Config()
        config.a3_ttt_ms = 30

        with self.assertRaises(ValueError):
            config.validate()


class RadioLinkMonitoringTests(unittest.TestCase):
    def setUp(self) -> None:
        config = Config()
        config.n310 = 2
        config.n311 = 2
        self.env = A3ReferenceSimulator(config)
        self.env.rrc.pcell = 0

    def test_no_indication_does_not_advance_n310(self) -> None:
        self.env._radio_link_monitoring(np.array([-9.0]), 0)
        self.assertEqual(self.env.cntr["n310"].cnt, 1)

        self.env._radio_link_monitoring(np.array([-7.0]), 1)
        self.assertEqual(self.env.cntr["n310"].cnt, 1)
        self.assertFalse(self.env.cntr["n310"].reached_max)

        self.env._radio_link_monitoring(np.array([-9.0]), 2)
        self.assertTrue(self.env.cntr["n310"].reached_max)

    def test_n311_stops_t310_after_required_indications(self) -> None:
        self.env.cntr["t310"].start(0)

        self.env._radio_link_monitoring(np.array([-5.0]), 1)
        self.assertEqual(self.env.cntr["n311"].cnt, 1)
        self.env._radio_link_monitoring(np.array([-5.0]), 2)
        self.env._t310_state_machine(np.array([-5.0]), 2)

        self.assertFalse(self.env.cntr["t310"].pending)


class HandoverMetricTests(unittest.TestCase):
    def test_ping_pong_requires_a_b_a(self) -> None:
        env = A3ReferenceSimulator(Config())

        env._record_successful_handover(0, 1, 100)
        env._record_successful_handover(1, 2, 150)
        self.assertEqual(env.ping_pong_idxs, [])

        env._record_successful_handover(2, 1, 200)
        self.assertEqual(env.ping_pong_idxs, [200])

    def test_execution_connects_to_latched_target(self) -> None:
        config = Config()
        config.t_ho_exec_ms = 10
        env = A3ReferenceSimulator(config)
        env.event_handler.target_cell = 1
        env.ho_source_cell = 0
        env.rrc.ncell = 2
        env.cntr["ho_exec"].start(0)
        env.cntr["ho_exec"].step(1)

        env._ho_state_machine(np.array([0.0, 0.0, 0.0]), 1)

        self.assertEqual(env.rrc.pcell, 1)
        self.assertEqual(env.successful_ho_history, [(1, 0, 1)])


if __name__ == "__main__":
    unittest.main()
