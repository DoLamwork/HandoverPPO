"""Tests for the new SMDP package boundaries."""

import unittest

import numpy as np

from ho_optim_drl.agents import AlwaysHandoverPolicy, AlwaysStayPolicy
from ho_optim_drl.smdp import HandoverAction, RewardBreakdown


class PolicyTests(unittest.TestCase):
    def test_diagnostic_policies_use_binary_smdp_actions(self) -> None:
        observation = np.zeros(1, dtype=np.float32)

        self.assertEqual(
            AlwaysHandoverPolicy().predict(observation),
            HandoverAction.HANDOVER,
        )
        self.assertEqual(
            AlwaysStayPolicy().predict(observation),
            HandoverAction.STAY,
        )

    def test_reward_breakdown_has_one_total(self) -> None:
        reward = RewardBreakdown(rate=-0.1, rlf=-2.0, ping_pong=-0.25)

        self.assertAlmostEqual(reward.total, -2.35)


if __name__ == "__main__":
    unittest.main()
