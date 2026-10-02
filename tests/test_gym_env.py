"""Gymnasium API tests for the handover SMDP environment."""

import os
import tempfile
import unittest

import gymnasium as gym
import numpy as np
from gymnasium.utils.env_checker import check_env

from ho_optim_drl import Config, HandoverSMDPEnv
from ho_optim_drl.data import RadioTrace, discover_trace_pairs
from ho_optim_drl.smdp import HandoverAction


def _decision_trace(
    trace_id: str,
    *,
    num_samples: int = 16,
    serving_sinr_db: float = 10.0,
    target_sinr_db: float = 0.0,
) -> RadioTrace:
    rsrp = np.vstack(
        (
            np.full(num_samples, -80.0),
            np.full(num_samples, -77.0),
        )
    )
    sinr = np.vstack(
        (
            np.full(num_samples, serving_sinr_db),
            np.full(num_samples, target_sinr_db),
        )
    )
    return RadioTrace.from_arrays(rsrp, sinr, trace_id=trace_id)


def _no_decision_trace(trace_id: str, num_samples: int = 8) -> RadioTrace:
    rsrp = np.vstack(
        (
            np.full(num_samples, -80.0),
            np.full(num_samples, -90.0),
        )
    )
    sinr = np.vstack(
        (
            np.full(num_samples, 5.0),
            np.full(num_samples, 0.0),
        )
    )
    return RadioTrace.from_arrays(rsrp, sinr, trace_id=trace_id)


def _full_candidate_trace(trace_id: str, num_samples: int = 16) -> RadioTrace:
    """Return a 21-cell trace for APIs that sample without an action mask."""
    rsrp = np.full((21, num_samples), -95.0, dtype=np.float64)
    rsrp[0] = -80.0
    rsrp[1] = -77.0
    sinr = np.full((21, num_samples), -5.0, dtype=np.float64)
    sinr[0] = 10.0
    return RadioTrace.from_arrays(rsrp, sinr, trace_id=trace_id)


class HandoverGymContractTests(unittest.TestCase):
    def test_checkpoint_objective_schema_mismatch_is_rejected(self) -> None:
        env = HandoverSMDPEnv(
            traces=[_decision_trace("schema", target_sinr_db=2.0)]
        )
        objective = env.objective_spec
        objective["candidate_policy"] = "different_order"

        with self.assertRaisesRegex(
            ValueError,
            "Checkpoint objective schema differs",
        ):
            env.validate_checkpoint_objective({"objective": objective})

    def test_spaces_reset_and_terminal_step_follow_gymnasium_contract(self) -> None:
        trace = _decision_trace("terminal", num_samples=8)
        env = HandoverSMDPEnv(Config(), traces=[trace])

        observation, reset_info = env.reset(seed=7)

        self.assertIsInstance(env.action_space, gym.spaces.Discrete)
        self.assertEqual(env.action_space.n, 11)
        self.assertEqual(observation.shape, (109,))
        self.assertEqual(reset_info["candidate_cells"], [1] + [None] * 9)
        self.assertEqual(
            reset_info["action_mask"],
            [True, True] + [False] * 9,
        )
        self.assertTrue(env.observation_space.contains(observation))
        self.assertEqual(reset_info["trace_id"], "terminal")
        self.assertEqual(reset_info["uncontrolled_duration_ticks"], 5)
        self.assertEqual(reset_info["decision"]["tic"], 4)

        next_observation, reward, terminated, truncated, info = env.step(
            int(HandoverAction.STAY)
        )

        self.assertTrue(env.observation_space.contains(next_observation))
        np.testing.assert_array_equal(
            next_observation,
            np.zeros(env.observation_space.shape, dtype=np.float32),
        )
        self.assertIsInstance(reward, float)
        self.assertAlmostEqual(reward, info["discounted_reward"]["total"])
        self.assertLessEqual(reward, info["reward"]["total"])
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["duration_ticks"], 3)
        self.assertEqual(info["action"], "stay")
        self.assertAlmostEqual(info["episode_return"], reward)
        self.assertEqual(info["episode_macro_steps"], 1)
        self.assertEqual(info["episode_duration_ticks"], 8)
        self.assertIsNone(info["next_decision"])
        self.assertIn("episode_metrics", info)

    def test_episode_return_applies_discount_between_macro_steps(self) -> None:
        rsrp = np.vstack(
            (
                np.full(16, -80.0),
                np.array([-77.0] * 8 + [-81.0] + [-77.0] * 7),
            )
        )
        sinr = np.vstack((np.full(16, 10.0), np.full(16, 0.0)))
        env = HandoverSMDPEnv(
            Config(),
            traces=[RadioTrace.from_arrays(rsrp, sinr)],
            gamma_per_second=0.95,
        )
        env.reset()

        _, first_reward, first_terminated, _, first_info = env.step(
            int(HandoverAction.STAY)
        )
        _, second_reward, second_terminated, _, second_info = env.step(
            int(HandoverAction.STAY)
        )

        self.assertFalse(first_terminated)
        self.assertTrue(second_terminated)
        self.assertAlmostEqual(
            second_info["episode_return"],
            first_reward + first_info["discount"] * second_reward,
        )

    def test_reset_seed_reproduces_trace_and_observation(self) -> None:
        traces = [
            _decision_trace("first", target_sinr_db=0.0),
            _decision_trace("second", target_sinr_db=4.0),
        ]
        env = HandoverSMDPEnv(traces=traces)

        first_observation, first_info = env.reset(seed=123)
        second_observation, second_info = env.reset(seed=123)

        self.assertEqual(first_info["trace_id"], second_info["trace_id"])
        np.testing.assert_array_equal(first_observation, second_observation)

    def test_reset_can_select_trace_by_index_or_id(self) -> None:
        traces = [
            _decision_trace("first"),
            _decision_trace("second", target_sinr_db=4.0),
        ]
        env = HandoverSMDPEnv(traces=traces)

        _, by_index = env.reset(options={"trace_index": 1})
        _, by_id = env.reset(options={"trace_id": "first"})

        self.assertEqual(by_index["trace_id"], "second")
        self.assertEqual(by_id["trace_index"], 0)
        with self.assertRaises(ValueError):
            env.reset(options={"trace_index": 0, "trace_id": "first"})
        with self.assertRaises(ValueError):
            env.reset(options={"unknown": 1})

    def test_random_reset_skips_trace_without_a3_decision(self) -> None:
        env = HandoverSMDPEnv(
            traces=[
                _no_decision_trace("no_decision"),
                _decision_trace("valid"),
            ]
        )

        observation, info = env.reset(seed=0)

        self.assertTrue(env.observation_space.contains(observation))
        self.assertEqual(info["trace_id"], "valid")
        self.assertEqual(
            info["skipped_no_decision_traces"],
            ("no_decision",),
        )
        with self.assertRaises(RuntimeError):
            env.reset(options={"trace_id": "no_decision"})

    def test_invalid_action_and_step_before_reset_are_rejected(self) -> None:
        env = HandoverSMDPEnv(traces=[_decision_trace("trace")])

        with self.assertRaises(RuntimeError):
            env.step(0)
        env.reset()
        with self.assertRaises(ValueError):
            env.step(3)

    def test_constructor_rejects_ambiguous_or_empty_sources(self) -> None:
        trace = _decision_trace("trace")
        with self.assertRaises(ValueError):
            HandoverSMDPEnv(traces=[])
        with self.assertRaises(ValueError):
            HandoverSMDPEnv(traces=[trace], data_dir="data/processed")
        with self.assertRaises(ValueError):
            HandoverSMDPEnv()

    def test_csv_trace_pair_path_exposes_dataset_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            header = "sample,time_ms,bs_0,bs_1\n"
            measurements = {
                "rsrp": (-80, -77),
                "sinr": (5, 2),
            }
            for kind, values in measurements.items():
                rows = "".join(
                    f"{sample},{sample * 10},{values[0]},{values[1]}\n"
                    for sample in range(12)
                )
                with open(
                    os.path.join(directory, f"{kind}_30kmh_0.csv"),
                    "w",
                    encoding="utf-8",
                ) as output_file:
                    output_file.write(header + rows)
            pair = discover_trace_pairs(directory)[0]
            env = HandoverSMDPEnv(
                data_dir=directory,
                trace_pairs=[pair],
            )

            observation, info = env.reset(options={"trace_index": 0})

            self.assertTrue(env.observation_space.contains(observation))
            self.assertEqual(info["speed_kmh"], pair.speed_kmh)
            self.assertEqual(info["trajectory_id"], pair.trajectory_id)
            self.assertEqual(
                info["trace_id"],
                f"{pair.speed_kmh}kmh_{pair.trajectory_id}",
            )

    def test_environment_passes_gymnasium_checker(self) -> None:
        env = HandoverSMDPEnv(
            traces=[
                _full_candidate_trace("first"),
                _full_candidate_trace("second"),
            ]
        )

        check_env(env, skip_render_check=True)


if __name__ == "__main__":
    unittest.main()
