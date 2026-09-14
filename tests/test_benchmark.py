"""Tests for fair policy benchmark accounting and output tables."""

import csv
import json
import os
import tempfile
import unittest
from unittest.mock import Mock

import numpy as np

from ho_optim_drl import Config, HandoverSMDPEnv
from ho_optim_drl.data import RadioTrace, TracePair
from ho_optim_drl.evaluation import (
    BenchmarkPolicy,
    TrajectoryDescriptor,
    aggregate_benchmark_rows,
    benchmark_policy,
    build_paired_comparisons,
    write_benchmark_outputs,
)
from ho_optim_drl.smdp import HandoverAction
from scripts.benchmark_policies import (
    build_benchmark_policies,
    select_benchmark_pairs,
    select_mixed_benchmark_pairs,
)


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

    def test_benchmark_command_only_uses_ppo_and_3gpp(self) -> None:
        policies = build_benchmark_policies(Mock())

        self.assertEqual(
            [policy.name for policy in policies],
            ["ppo", "3gpp_always_ho"],
        )

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
                "fixed_stay_test_policy",
                lambda observation: HandoverAction.STAY,
            ),
            self.descriptors,
        )
        wait_rows = benchmark_policy(
            self._env(),
            BenchmarkPolicy(
                "fixed_wait_test_policy",
                lambda observation: HandoverAction.WAIT_50,
            ),
            self.descriptors,
        )

        self.assertTrue(
            all(
                row["handover_action_percent"] == 100.0
                for row in handover_rows
            )
        )
        self.assertTrue(
            all(
                row["handover_action_percent"] == 0.0
                for row in stay_rows
            )
        )
        self.assertTrue(all(row["actionable"] for row in handover_rows))
        self.assertTrue(
            all(row["wait_action_percent"] == 100.0 for row in wait_rows)
        )
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
                "connected_percent": 80.0,
                "macro_steps": 2,
                "handover_actions": 1,
                "stay_actions": 1,
                "wait_actions": 0,
                "ho_attempts": 2,
                "ho_successes": 1,
                "handover_failures": 1,
                "rlf_events": 1,
                "rlf_recovery_time_s": 1.0,
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
                "connected_percent": 100.0,
                "macro_steps": 4,
                "handover_actions": 3,
                "stay_actions": 1,
                "wait_actions": 2,
                "ho_attempts": 3,
                "ho_successes": 3,
                "handover_failures": 0,
                "rlf_events": 2,
                "rlf_recovery_time_s": 2.0,
                "ping_pong_events": 1,
            },
        ]

        aggregate = aggregate_benchmark_rows(rows, by_speed=False)[0]

        self.assertAlmostEqual(
            aggregate["duration_weighted_rate_mbps"],
            25.0,
        )
        self.assertAlmostEqual(aggregate["rlf_time_percent"], 7.5)
        self.assertAlmostEqual(aggregate["handover_failure_percent"], 20.0)
        self.assertAlmostEqual(aggregate["ping_pong_percent"], 50.0)
        self.assertAlmostEqual(
            aggregate["handover_action_percent"],
            50.0,
        )
        self.assertAlmostEqual(aggregate["wait_action_percent"], 25.0)

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
                "ppo",
                lambda observation: HandoverAction.HANDOVER,
            ),
            self.descriptors,
        )
        aggregate = aggregate_benchmark_rows(rows, by_speed=False)
        by_speed = aggregate_benchmark_rows(rows, by_speed=True)
        comparisons = [
            {
                "policy": "ppo",
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
        self.assertEqual(summary["aggregate"][0]["policy"], "ppo")

    def test_all_split_can_select_unseen_speeds(self) -> None:
        pairs = [
            TracePair(120, 0, "rsrp_120kmh_0.csv", "sinr_120kmh_0.csv"),
            TracePair(120, 1, "rsrp_120kmh_1.csv", "sinr_120kmh_1.csv"),
            TracePair(150, 0, "rsrp_150kmh_0.csv", "sinr_150kmh_0.csv"),
            TracePair(90, 0, "rsrp_90kmh_0.csv", "sinr_90kmh_0.csv"),
        ]

        selected = select_benchmark_pairs(
            pairs,
            split="all",
            split_seed=17,
            speeds=[120, 150],
        )

        self.assertEqual(len(selected), 3)
        self.assertEqual({pair.speed_kmh for pair in selected}, {120, 150})

    def test_speed_selection_rejects_missing_data(self) -> None:
        pairs = [
            TracePair(120, 0, "rsrp_120kmh_0.csv", "sinr_120kmh_0.csv"),
        ]

        with self.assertRaisesRegex(ValueError, "150 km/h"):
            select_benchmark_pairs(
                pairs,
                split="all",
                split_seed=17,
                speeds=[120, 150],
            )

    def test_custom_test_split_selects_12_routes_per_speed(self) -> None:
        speeds = (30, 50, 70, 90, 120, 150)
        pairs = [
            TracePair(
                speed,
                trajectory_id,
                f"rsrp_{speed}kmh_{trajectory_id}.csv",
                f"sinr_{speed}kmh_{trajectory_id}.csv",
            )
            for speed in speeds
            for trajectory_id in range(60)
        ]

        selected = select_benchmark_pairs(
            pairs,
            split="test",
            split_seed=17,
            speeds=None,
            train_fraction=0.60,
            validation_fraction=0.20,
            test_fraction=0.20,
        )

        self.assertEqual(len(selected), 72)
        for speed in speeds:
            self.assertEqual(
                sum(pair.speed_kmh == speed for pair in selected),
                12,
            )

    def test_mixed_split_combines_seen_test_and_all_zero_shot_routes(
        self,
    ) -> None:
        counts = {30: 420, 50: 255, 70: 180, 90: 145, 120: 30, 150: 30}
        pairs = [
            TracePair(
                speed,
                trajectory_id,
                f"rsrp_{speed}kmh_{trajectory_id}.csv",
                f"sinr_{speed}kmh_{trajectory_id}.csv",
            )
            for speed, count in counts.items()
            for trajectory_id in range(count)
        ]

        selected = select_mixed_benchmark_pairs(
            pairs,
            split="test",
            split_seed=17,
            speeds=[30, 50, 70, 90],
            zero_shot_speeds=[120, 150],
            train_fraction=0.60,
            validation_fraction=0.20,
            test_fraction=0.20,
        )

        expected_counts = {30: 84, 50: 51, 70: 36, 90: 29, 120: 30, 150: 30}
        self.assertEqual(len(selected), 260)
        for speed, expected in expected_counts.items():
            self.assertEqual(
                sum(pair.speed_kmh == speed for pair in selected),
                expected,
            )

    def test_mixed_split_rejects_overlapping_speed_groups(self) -> None:
        pairs = [
            TracePair(90, 0, "rsrp_90kmh_0.csv", "sinr_90kmh_0.csv"),
        ]

        with self.assertRaisesRegex(ValueError, "overlap"):
            select_mixed_benchmark_pairs(
                pairs,
                split="test",
                split_seed=17,
                speeds=[90],
                zero_shot_speeds=[90],
            )


if __name__ == "__main__":
    unittest.main()
