"""Tests for fair policy benchmark accounting and output tables."""

import csv
import json
import os
import tempfile
import unittest

import numpy as np

from ho_optim_drl import Config, HandoverSMDPEnv
from ho_optim_drl.data import RadioTrace
from ho_optim_drl.evaluation import (
    BenchmarkPolicy,
    TrajectoryDescriptor,
    aggregate_benchmark_rows,
    benchmark_policy,
    build_paired_comparisons,
    write_benchmark_outputs,
)
from ho_optim_drl.smdp import HandoverAction


def _trace(trace_id: str, target_sinr_db: float) -> RadioTrace:
    rsrp = np.vstack(
        (
            np.full(12, -80.0),
            np.full(12, -77.0),
        )
    )
    sinr = np.vstack(
        (
            np.full(12, 5.0),
            np.full(12, target_sinr_db),
        )
    )
    return RadioTrace.from_arrays(rsrp, sinr, trace_id=trace_id)


class BenchmarkTests(unittest.TestCase):
    def setUp(self) -> None:
        self.traces = [_trace("30kmh_0", 3.0), _trace("50kmh_0", 1.0)]
        self.descriptors = [
            TrajectoryDescriptor("30kmh_0", 30, 0),
            TrajectoryDescriptor("50kmh_0", 50, 0),
        ]

    def _env(self) -> HandoverSMDPEnv:
        return HandoverSMDPEnv(Config(), traces=self.traces)

    def test_fixed_policies_have_expected_action_fractions(self) -> None:
        handover_rows = benchmark_policy(
            self._env(),
            BenchmarkPolicy(
                "3gpp_always_ho",
                lambda observation: HandoverAction.HANDOVER,
            ),
            self.descriptors,
        )
        stay_rows = benchmark_policy(
            self._env(),
            BenchmarkPolicy(
                "always_stay",
                lambda observation: HandoverAction.STAY,
            ),
            self.descriptors,
        )

        self.assertTrue(
            all(row["handover_fraction"] == 1.0 for row in handover_rows)
        )
        self.assertTrue(
            all(row["handover_fraction"] == 0.0 for row in stay_rows)
        )
        self.assertTrue(all(row["actionable"] for row in handover_rows))
        self.assertTrue(
            all(
                np.isclose(
                    row["full_trace_return"],
                    row["controlled_return"]
                    + row["full_trace_reward_rate"]
                    - row["controlled_reward_rate"]
                    + row["full_trace_reward_rlf"]
                    - row["controlled_reward_rlf"]
                    + row["full_trace_reward_ping_pong"]
                    - row["controlled_reward_ping_pong"],
                )
                for row in handover_rows
            )
        )

    def test_aggregate_uses_pooled_event_denominators(self) -> None:
        rows = [
            {
                "policy": "policy",
                "speed_kmh": 30,
                "actionable": True,
                "total_time_s": 10.0,
                "controlled_return": 1.0,
                "full_trace_return": 2.0,
                "controlled_reward_rate": 3.0,
                "controlled_reward_rlf": -1.0,
                "controlled_reward_ping_pong": 0.0,
                "estimated_rate_mbps": 10.0,
                "relative_rate_percent": 50.0,
                "connected_fraction": 0.8,
                "macro_steps": 2,
                "handover_actions": 1,
                "stay_actions": 1,
                "ho_attempts": 2,
                "ho_successes": 1,
                "handover_failures": 1,
                "rlf_events": 1,
                "ping_pong_events": 1,
            },
            {
                "policy": "policy",
                "speed_kmh": 30,
                "actionable": True,
                "total_time_s": 30.0,
                "controlled_return": 3.0,
                "full_trace_return": 4.0,
                "controlled_reward_rate": 5.0,
                "controlled_reward_rlf": -1.0,
                "controlled_reward_ping_pong": 0.0,
                "estimated_rate_mbps": 30.0,
                "relative_rate_percent": 75.0,
                "connected_fraction": 1.0,
                "macro_steps": 4,
                "handover_actions": 3,
                "stay_actions": 1,
                "ho_attempts": 3,
                "ho_successes": 3,
                "handover_failures": 0,
                "rlf_events": 2,
                "ping_pong_events": 1,
            },
        ]

        aggregate = aggregate_benchmark_rows(rows, by_speed=False)[0]

        self.assertAlmostEqual(
            aggregate["duration_weighted_rate_mbps"],
            25.0,
        )
        self.assertAlmostEqual(aggregate["rlf_rate_per_100s"], 7.5)
        self.assertAlmostEqual(aggregate["handover_failure_rate"], 0.2)
        self.assertAlmostEqual(aggregate["ping_pong_rate"], 0.5)
        self.assertAlmostEqual(aggregate["handover_fraction"], 2 / 3)

    def test_paired_comparison_matches_trajectory_ids(self) -> None:
        rows = [
            {
                "policy": "3gpp_always_ho",
                "trace_id": "a",
                "estimated_rate_mbps": 10.0,
                "controlled_return": 1.0,
                "rlf_events": 2,
                "ping_pong_events": 1,
            },
            {
                "policy": "ppo",
                "trace_id": "a",
                "estimated_rate_mbps": 12.0,
                "controlled_return": 3.0,
                "rlf_events": 1,
                "ping_pong_events": 0,
            },
        ]

        comparison = build_paired_comparisons(
            rows,
            reference_policy="3gpp_always_ho",
        )[0]

        self.assertEqual(comparison["mean_rate_delta_mbps"], 2.0)
        self.assertEqual(comparison["total_rlf_delta"], -1)
        self.assertEqual(comparison["total_ping_pong_delta"], -1)
        self.assertEqual(comparison["return_win_fraction"], 1.0)

    def test_output_files_are_machine_readable(self) -> None:
        rows = benchmark_policy(
            self._env(),
            BenchmarkPolicy(
                "always_stay",
                lambda observation: HandoverAction.STAY,
            ),
            self.descriptors,
        )
        aggregate = aggregate_benchmark_rows(rows, by_speed=False)
        by_speed = aggregate_benchmark_rows(rows, by_speed=True)
        comparisons = [
            {
                "policy": "always_stay",
                "reference_policy": "3gpp_always_ho",
                "paired_trajectories": 2,
            }
        ]

        with tempfile.TemporaryDirectory() as directory:
            paths = write_benchmark_outputs(
                directory,
                rows,
                aggregate,
                by_speed,
                comparisons,
                metadata={"split": "test"},
            )
            with open(
                paths["per_trajectory"],
                newline="",
                encoding="utf-8",
            ) as csv_file:
                csv_rows = list(csv.DictReader(csv_file))
            with open(paths["summary"], encoding="utf-8") as summary_file:
                summary = json.load(summary_file)

        self.assertEqual(len(csv_rows), 2)
        self.assertEqual(summary["metadata"]["split"], "test")
        self.assertEqual(summary["aggregate"][0]["policy"], "always_stay")


if __name__ == "__main__":
    unittest.main()
