"""Mathematical and integration tests for duration-aware SMDP PPO."""

import math
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout

import numpy as np
import torch

from ho_optim_drl import Config, HandoverSMDPEnv
from ho_optim_drl.agents.ppo import (
    SMDPPPO,
    SMDPPPOConfig,
    SMDPRolloutBuffer,
)
from ho_optim_drl.data import RadioTrace
from scripts.train_smdp_ppo import TrainingProgressLogger


def _trace(trace_id: str, target_sinr_db: float) -> RadioTrace:
    rsrp = np.vstack((np.full(10, -80.0), np.full(10, -77.0)))
    sinr = np.vstack(
        (
            np.full(10, 5.0),
            np.full(10, target_sinr_db),
        )
    )
    return RadioTrace.from_arrays(rsrp, sinr, trace_id=trace_id)


def _add_transition(
    buffer: SMDPRolloutBuffer,
    *,
    reward: float,
    value: float,
    next_value: float,
    discount: float,
    duration_seconds: float,
    terminated: bool = False,
    truncated: bool = False,
) -> None:
    buffer.add(
        observation=np.array([0.0], dtype=np.float32),
        action=0,
        reward=reward,
        old_log_probability=-0.5,
        value=value,
        next_value=next_value,
        discount=discount,
        duration_seconds=duration_seconds,
        terminated=terminated,
        truncated=truncated,
    )


class DurationAwareGAETests(unittest.TestCase):
    def test_gae_matches_hand_calculation(self) -> None:
        buffer = SMDPRolloutBuffer(3, (1,))
        _add_transition(
            buffer,
            reward=0.5,
            value=1.0,
            next_value=2.0,
            discount=0.9,
            duration_seconds=1.0,
        )
        _add_transition(
            buffer,
            reward=1.0,
            value=2.0,
            next_value=3.0,
            discount=0.8,
            duration_seconds=2.0,
        )
        _add_transition(
            buffer,
            reward=-1.0,
            value=3.0,
            next_value=0.0,
            discount=0.7,
            duration_seconds=1.0,
            terminated=True,
        )

        buffer.compute_returns_and_advantages(lambda_per_second=0.5)

        np.testing.assert_allclose(
            buffer.advantages,
            np.array([1.57, 0.6, -4.0]),
            rtol=0,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            buffer.returns,
            np.array([2.57, 2.6, -1.0]),
            rtol=0,
            atol=1e-6,
        )

    def test_episode_boundary_blocks_advantage_leakage(self) -> None:
        buffer = SMDPRolloutBuffer(2, (1,))
        _add_transition(
            buffer,
            reward=1.0,
            value=0.0,
            next_value=99.0,
            discount=0.9,
            duration_seconds=1.0,
            terminated=True,
        )
        _add_transition(
            buffer,
            reward=10.0,
            value=0.0,
            next_value=0.0,
            discount=0.9,
            duration_seconds=1.0,
            terminated=True,
        )

        buffer.compute_returns_and_advantages(lambda_per_second=0.95)

        self.assertAlmostEqual(buffer.advantages[0], 1.0)
        self.assertAlmostEqual(buffer.advantages[1], 10.0)

    def test_truncation_bootstraps_delta_but_stops_recursion(self) -> None:
        buffer = SMDPRolloutBuffer(2, (1,))
        _add_transition(
            buffer,
            reward=1.0,
            value=2.0,
            next_value=3.0,
            discount=0.5,
            duration_seconds=1.0,
            truncated=True,
        )
        _add_transition(
            buffer,
            reward=100.0,
            value=0.0,
            next_value=0.0,
            discount=0.5,
            duration_seconds=1.0,
            terminated=True,
        )

        buffer.compute_returns_and_advantages(lambda_per_second=1.0)

        self.assertAlmostEqual(buffer.advantages[0], 0.5)
        self.assertAlmostEqual(buffer.advantages[1], 100.0)

    def test_longer_macro_step_has_stronger_lambda_decay(self) -> None:
        short = 0.95**0.1
        long = 0.95**2.0

        self.assertLess(long, short)


class PPOIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.env = HandoverSMDPEnv(
            Config(),
            traces=[_trace("first", 2.0), _trace("second", 4.0)],
            gamma_per_second=0.99,
        )
        self.config = SMDPPPOConfig(
            hidden_sizes=(16,),
            learning_rate=1e-3,
            rollout_steps=4,
            batch_size=2,
            update_epochs=2,
            target_kl=None,
            seed=3,
            device="cpu",
        )

    def test_collection_uses_environment_discount_and_update_changes_model(
        self,
    ) -> None:
        agent = SMDPPPO(19, 2, self.config)
        before = {
            name: parameter.detach().clone()
            for name, parameter in agent.model.named_parameters()
        }

        buffer, episodes = agent.collect_rollout(
            self.env,
            reset_seed=11,
        )
        metrics = agent.update(buffer)

        self.assertEqual(buffer.size, 4)
        self.assertGreaterEqual(len(episodes), 1)
        for discount, duration in zip(
            buffer.discounts,
            buffer.duration_seconds,
        ):
            self.assertAlmostEqual(
                float(discount),
                0.99 ** float(duration),
                places=6,
            )
        self.assertTrue(math.isfinite(metrics.policy_loss))
        self.assertTrue(math.isfinite(metrics.value_loss))
        self.assertTrue(math.isfinite(metrics.approximate_kl))
        self.assertTrue(
            any(
                not torch.equal(before[name], parameter)
                for name, parameter in agent.model.named_parameters()
            )
        )

    def test_evaluation_is_deterministic_and_covers_each_trace_once(self) -> None:
        agent = SMDPPPO(19, 2, self.config)

        first = agent.evaluate(self.env)
        second = agent.evaluate(self.env)

        self.assertEqual(first.episodes, 2)
        self.assertEqual(first.skipped_no_decision, 0)
        self.assertEqual(first, second)
        self.assertGreaterEqual(first.handover_fraction, 0.0)
        self.assertLessEqual(first.handover_fraction, 1.0)

    def test_checkpoint_round_trip_preserves_prediction(self) -> None:
        agent = SMDPPPO(19, 2, self.config)
        observation, _ = self.env.reset(options={"trace_index": 0})
        action_before = agent.predict(observation)

        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "ppo.pt")
            agent.total_macro_steps = 7
            agent.save_checkpoint(path, extra={"tag": "test"})

            restored = SMDPPPO(19, 2, self.config)
            extra = restored.load_checkpoint(path)
            action_after = restored.predict(observation)

        self.assertEqual(action_before, action_after)
        self.assertEqual(restored.total_macro_steps, 7)
        self.assertEqual(extra, {"tag": "test"})

    def test_gamma_mismatch_is_rejected(self) -> None:
        mismatched = SMDPPPO(
            19,
            2,
            SMDPPPOConfig(
                hidden_sizes=(8,),
                rollout_steps=2,
                batch_size=2,
                update_epochs=1,
                gamma_per_second=0.9,
            ),
        )

        with self.assertRaises(ValueError):
            mismatched.collect_rollout(self.env)

    def test_training_loop_saves_loadable_checkpoint(self) -> None:
        validation_env = HandoverSMDPEnv(
            Config(),
            traces=[_trace("validation", 3.0)],
            gamma_per_second=0.99,
        )
        agent = SMDPPPO(19, 2, self.config)
        events: list[dict] = []

        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "best.pt")
            history = agent.train(
                self.env,
                total_macro_steps=4,
                validation_env=validation_env,
                evaluation_interval=4,
                checkpoint_path=path,
                progress_callback=events.append,
                progress_interval=1,
            )
            restored, extra = SMDPPPO.from_checkpoint(
                path,
                observation_size=19,
                action_size=2,
            )

        self.assertEqual(agent.total_macro_steps, 4)
        self.assertEqual(len(history), 1)
        self.assertIn("validation", history[0])
        self.assertEqual(restored.total_macro_steps, 4)
        self.assertIn("best_validation_return", extra)
        event_names = [event["event"] for event in events]
        self.assertEqual(event_names[0], "training_started")
        self.assertIn("rollout_progress", event_names)
        self.assertIn("update_complete", event_names)
        self.assertIn("validation_started", event_names)
        self.assertIn("validation_complete", event_names)
        self.assertEqual(event_names[-1], "training_complete")

    def test_progress_logger_prints_and_persists_json_lines(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "metrics.jsonl")
            logger = TrainingProgressLogger(path, target_macro_steps=10)
            output = io.StringIO()
            with redirect_stdout(output):
                logger(
                    {
                        "event": "training_started",
                        "total_macro_steps": 0,
                        "target_macro_steps": 10,
                        "rollout_steps": 4,
                        "device": "cpu",
                    }
                )
                logger(
                    {
                        "event": "rollout_progress",
                        "collected_steps": 2,
                        "rollout_steps": 4,
                        "total_macro_steps": 2,
                        "completed_episodes": 1,
                        "update": 1,
                    }
                )

            with open(path, encoding="utf-8") as metrics_file:
                records = [json.loads(line) for line in metrics_file]

        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["event"], "training_started")
        self.assertEqual(records[1]["total_macro_steps"], 2)
        self.assertIn("[train]", output.getvalue())
        self.assertIn("[collect 0001]", output.getvalue())


if __name__ == "__main__":
    unittest.main()
