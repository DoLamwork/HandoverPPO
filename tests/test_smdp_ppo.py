"""Mathematical and integration tests for duration-aware SMDP PPO."""

import math
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import replace

import numpy as np
import torch

from ho_optim_drl import Config, HandoverSMDPEnv
from ho_optim_drl.agents.ppo import (
    ActorCritic,
    EvaluationMetrics,
    SMDPPPO,
    SMDPPPOConfig,
    SMDPRolloutBuffer,
    ValidationCheckpointManager,
)
from ho_optim_drl.data import RadioTrace, TracePair, TraceSplits
from ho_optim_drl.smdp import HandoverAction
from scripts.train_smdp_ppo import (
    TrainingProgressLogger,
    aggregate_training_trace_usage,
    build_dataset_manifest,
    fit_fixed_value_target_scale,
    select_value_scale_calibration_pairs,
    write_dataset_manifest,
)


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


def _evaluation_metrics(
    *,
    mean_return: float,
    rate_mbps: float,
    rlf_time_percent: float,
    handover_failure_percent: float,
    ping_pong_percent: float,
    handover_attempts: int,
    relative_rate_percent: float | None = None,
) -> EvaluationMetrics:
    relative_rate_percent = (
        100 * rate_mbps / 12.0
        if relative_rate_percent is None
        else relative_rate_percent
    )
    return EvaluationMetrics(
        episodes=2,
        skipped_no_decision=0,
        mean_return=mean_return,
        std_return=0.1,
        mean_rate_reward=1.0,
        mean_rlf_penalty=-1.0,
        mean_ping_pong_penalty=-0.5,
        mean_macro_steps=3.0,
        handover_fraction=0.5,
        stay_fraction=0.5,
        mean_estimated_rate_mbps=rate_mbps,
        mean_max_estimated_rate_mbps=12.0,
        mean_relative_rate_percent=relative_rate_percent,
        mean_connected_percent=98.0,
        mean_estimated_rate_mbps_by_speed={"30": rate_mbps},
        total_handover_attempts=handover_attempts,
        total_handover_successes=max(handover_attempts - 1, 0),
        total_handover_failures=1,
        handover_failure_percent=handover_failure_percent,
        total_rlf=2,
        total_rlf_recovery_time_seconds=1.0,
        rlf_time_percent=rlf_time_percent,
        total_ping_pong=1,
        ping_pong_percent=ping_pong_percent,
        mean_relative_rate_percent_by_speed={
            "30": relative_rate_percent
        },
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
        self.observation_size = self.env.observation_space.shape[0]

    def test_collection_uses_environment_discount_and_update_changes_model(
        self,
    ) -> None:
        agent = SMDPPPO(self.observation_size, len(HandoverAction), self.config)
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
        self.assertIsNone(metrics.learning_diagnostics)
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
        agent = SMDPPPO(self.observation_size, len(HandoverAction), self.config)

        first = agent.evaluate(self.env)
        second = agent.evaluate(self.env)

        self.assertEqual(first.episodes, 2)
        self.assertEqual(first.skipped_no_decision, 0)
        self.assertEqual(first, second)
        self.assertGreaterEqual(first.handover_fraction, 0.0)
        self.assertLessEqual(first.handover_fraction, 1.0)

    def test_separate_validation_env_preserves_training_collector(self) -> None:
        agent = SMDPPPO(self.observation_size, len(HandoverAction), self.config)
        agent.collect_rollout(self.env, rollout_steps=1, reset_seed=11)
        collector_env_id = agent._collector_env_id
        collector_observation = agent._collector_observation.copy()
        validation_env = HandoverSMDPEnv(
            Config(),
            traces=[_trace("validation", 3.0)],
            gamma_per_second=0.99,
        )

        agent.evaluate(validation_env)

        self.assertEqual(agent._collector_env_id, collector_env_id)
        np.testing.assert_array_equal(
            agent._collector_observation,
            collector_observation,
        )

    def test_evaluation_on_training_env_invalidates_training_collector(self) -> None:
        agent = SMDPPPO(self.observation_size, len(HandoverAction), self.config)
        agent.collect_rollout(self.env, rollout_steps=1, reset_seed=11)

        agent.evaluate(self.env)

        self.assertIsNone(agent._collector_env_id)
        self.assertIsNone(agent._collector_observation)

    def test_checkpoint_round_trip_preserves_prediction(self) -> None:
        agent = SMDPPPO(self.observation_size, len(HandoverAction), self.config)
        observation, _ = self.env.reset(options={"trace_index": 0})
        action_before = agent.predict(observation)

        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "ppo.pt")
            agent.total_macro_steps = 7
            agent.save_checkpoint(path, extra={"tag": "test"})

            restored = SMDPPPO(
                self.observation_size,
                len(HandoverAction),
                self.config,
            )
            extra = restored.load_checkpoint(path)
            action_after = restored.predict(observation)

        self.assertEqual(action_before, action_after)
        self.assertEqual(restored.total_macro_steps, 7)
        self.assertEqual(extra, {"tag": "test"})

    def test_checkpoint_without_new_diagnostics_flag_remains_loadable(self) -> None:
        agent = SMDPPPO(self.observation_size, len(HandoverAction), self.config)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "legacy_config.pt")
            agent.save_checkpoint(path)
            checkpoint = torch.load(path, weights_only=True)
            checkpoint["config"].pop("learning_diagnostics", None)
            checkpoint["config"].pop("network_architecture", None)
            checkpoint["config"].pop("candidate_feature_size", None)
            checkpoint["config"].pop("value_normalization", None)
            checkpoint["config"].pop("value_target_scale", None)
            checkpoint["model_spec"].pop("network_architecture", None)
            checkpoint["model_spec"].pop("candidate_feature_size", None)
            checkpoint["model_spec"].pop("value_normalization", None)
            checkpoint["model_spec"].pop("value_target_scale", None)
            torch.save(checkpoint, path)

            restored, _ = SMDPPPO.from_checkpoint(path)

        self.assertFalse(restored.config.learning_diagnostics)
        self.assertEqual(restored.config.network_architecture, "shared_mlp_v1")
        self.assertEqual(restored.config.value_normalization, "none")
        self.assertEqual(restored.config.value_target_scale, 1.0)

    def test_separate_encoder_gradients_do_not_cross_heads(self) -> None:
        model = ActorCritic(
            self.observation_size,
            len(HandoverAction),
            hidden_sizes=(16,),
            architecture="separate_mlp_v1",
        )
        observations = torch.randn(5, self.observation_size)
        logits, values = model(observations)

        actor_gradients = torch.autograd.grad(
            logits.square().sum(),
            tuple(model.critic_encoder.parameters()),
            allow_unused=True,
        )
        critic_gradients = torch.autograd.grad(
            values.square().sum(),
            tuple(model.actor_encoder.parameters()),
            allow_unused=True,
        )

        self.assertTrue(all(value is None for value in actor_gradients))
        self.assertTrue(all(value is None for value in critic_gradients))

    def test_separate_checkpoint_round_trip_auto_reconstructs_model(self) -> None:
        config = replace(
            self.config,
            network_architecture="separate_mlp_v1",
        )
        agent = SMDPPPO(self.observation_size, len(HandoverAction), config)
        observation, _ = self.env.reset(options={"trace_index": 0})
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "separate.pt")
            agent.save_checkpoint(path)
            restored, _ = SMDPPPO.from_checkpoint(path)

        self.assertEqual(restored.model.architecture, "separate_mlp_v1")
        before = agent.model.act_with_probabilities(
            observation, device=agent.device, deterministic=True
        )
        after = restored.model.act_with_probabilities(
            observation, device=restored.device, deterministic=True
        )
        self.assertEqual(before[:3], after[:3])
        np.testing.assert_array_equal(before[3], after[3])

    def test_fixed_value_scale_keeps_public_values_raw_and_scales_loss(self) -> None:
        scale = 5.0
        config = replace(
            self.config,
            value_normalization="fixed_scale_v1",
            value_target_scale=scale,
            update_epochs=1,
        )
        agent = SMDPPPO(self.observation_size, len(HandoverAction), config)
        with torch.no_grad():
            agent.model.value_head.weight.zero_()
            agent.model.value_head.bias.fill_(2.0)
        observation, _ = self.env.reset(options={"trace_index": 0})
        self.assertAlmostEqual(
            agent.model.predict_value(observation, device=agent.device),
            10.0,
        )
        buffer, _ = agent.collect_rollout(self.env, reset_seed=11)
        metrics = agent.update(buffer)
        self.assertAlmostEqual(
            metrics.raw_value_mse,
            2.0 * metrics.value_loss * scale * scale,
            places=4,
        )

    def test_value_scale_calibration_uses_train_returns_and_is_deterministic(self) -> None:
        traces = [_trace("cal-a", 2.0), _trace("cal-b", 4.0)]
        records = [{"trace_id": trace.trace_id} for trace in traces]
        first = fit_fixed_value_target_scale(
            HandoverSMDPEnv(Config(), traces=traces),
            trace_records=records,
        )
        second = fit_fixed_value_target_scale(
            HandoverSMDPEnv(Config(), traces=traces),
            trace_records=records,
        )
        self.assertGreaterEqual(first["value_target_scale"], 1.0)
        self.assertEqual(first, second)
        self.assertEqual(first["split"], "train")

    def test_value_scale_calibration_subset_is_speed_stratified(self) -> None:
        pairs = tuple(
            TracePair(speed, route, f"r{speed}-{route}", f"s{speed}-{route}")
            for speed in (50, 30)
            for route in ("b", "a", "c")
        )
        selected = select_value_scale_calibration_pairs(
            pairs, routes_per_speed=2
        )
        self.assertEqual(
            [(pair.speed_kmh, pair.trajectory_id) for pair in selected],
            [(30, "a"), (30, "b"), (50, "a"), (50, "b")],
        )

    def test_numpy_model_dimensions_are_saved_safely_and_legacy_loads(self) -> None:
        agent = SMDPPPO(
            np.int64(self.observation_size),
            np.int64(len(HandoverAction)),
            self.config,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "numpy_dimensions.pt")
            agent.save_checkpoint(path)

            checkpoint = torch.load(path, weights_only=True)
            self.assertIs(type(checkpoint["model_spec"]["observation_size"]), int)
            self.assertIs(type(checkpoint["model_spec"]["action_size"]), int)

            # Reproduce checkpoints written before dimensions were normalized.
            checkpoint["model_spec"]["action_size"] = np.int64(
                len(HandoverAction)
            )
            torch.save(checkpoint, path)
            restored, _ = SMDPPPO.from_checkpoint(path)

        self.assertEqual(restored.model.action_size, len(HandoverAction))

    def test_two_action_checkpoint_without_model_spec_is_inferred(self) -> None:
        binary_agent = SMDPPPO(19, 2, self.config)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "binary_without_spec.pt")
            binary_agent.save_checkpoint(path, extra={"tag": "binary"})
            checkpoint = torch.load(path, weights_only=True)
            checkpoint.pop("model_spec")
            torch.save(checkpoint, path)

            restored, extra = SMDPPPO.from_checkpoint(path)

        self.assertEqual(restored.model.observation_size, 19)
        self.assertEqual(restored.model.action_size, 2)
        self.assertEqual(extra["tag"], "binary")
        with self.assertRaises(ValueError):
            restored.evaluate(self.env)

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
        agent = SMDPPPO(self.observation_size, len(HandoverAction), self.config)
        events: list[dict] = []

        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "best.pt")
            history = agent.train(
                self.env,
                total_macro_steps=4,
                validation_env=validation_env,
                evaluation_interval=4,
                checkpoint_path=path,
                checkpoint_extra={"dataset_manifest_sha256": "abc123"},
                progress_callback=events.append,
                progress_interval=1,
            )
            restored, extra = SMDPPPO.from_checkpoint(
                path,
                observation_size=self.observation_size,
                action_size=len(HandoverAction),
            )

        self.assertEqual(agent.total_macro_steps, 4)
        self.assertEqual(len(history), 1)
        self.assertIn("validation", history[0])
        self.assertEqual(
            sum(history[0]["training_trace_macro_steps"].values()),
            4,
        )
        self.assertEqual(
            sum(
                segment["macro_steps"]
                for segment in history[0]["training_trace_segments"]
            ),
            4,
        )
        self.assertEqual(
            len(history[0]["completed_episode_trace_ids"]),
            history[0]["completed_episodes"],
        )
        usage = aggregate_training_trace_usage(history)
        self.assertEqual(usage["total_logged_macro_steps"], 4)
        self.assertEqual(restored.total_macro_steps, 4)
        self.assertIn("best_validation_return", extra)
        self.assertEqual(extra["dataset_manifest_sha256"], "abc123")
        event_names = [event["event"] for event in events]
        self.assertEqual(event_names[0], "training_started")
        self.assertIn("rollout_progress", event_names)
        self.assertIn("update_complete", event_names)
        self.assertIn("validation_started", event_names)
        self.assertIn("validation_complete", event_names)
        self.assertEqual(event_names[-1], "training_complete")

    def test_budget_end_validation_is_added_when_interval_does_not_divide(self) -> None:
        validation_env = HandoverSMDPEnv(
            Config(), traces=[_trace("validation", 3.0)], gamma_per_second=0.99
        )
        agent = SMDPPPO(self.observation_size, len(HandoverAction), self.config)
        events: list[dict] = []

        agent.train(
            self.env,
            total_macro_steps=5,
            validation_env=validation_env,
            evaluation_interval=4,
            progress_callback=events.append,
        )

        completed = [
            event for event in events if event["event"] == "validation_complete"
        ]
        self.assertEqual([event["macro_steps"] for event in completed], [4, 5])
        self.assertEqual(completed[-1]["validation_trigger"], "budget_end")

    def test_budget_end_validation_is_not_duplicated_on_interval(self) -> None:
        validation_env = HandoverSMDPEnv(
            Config(), traces=[_trace("validation", 3.0)], gamma_per_second=0.99
        )
        agent = SMDPPPO(self.observation_size, len(HandoverAction), self.config)
        events: list[dict] = []

        agent.train(
            self.env,
            total_macro_steps=4,
            validation_env=validation_env,
            evaluation_interval=4,
            progress_callback=events.append,
        )

        completed = [
            event for event in events if event["event"] == "validation_complete"
        ]
        self.assertEqual([event["macro_steps"] for event in completed], [4])
        self.assertEqual(completed[0]["validation_trigger"], "scheduled")

    def test_opt_in_learning_diagnostics_separate_actor_and_critic(self) -> None:
        config = SMDPPPOConfig(
            hidden_sizes=(16,),
            learning_rate=1e-3,
            rollout_steps=4,
            batch_size=2,
            update_epochs=1,
            target_kl=None,
            learning_diagnostics=True,
            seed=3,
        )
        agent = SMDPPPO(self.observation_size, len(HandoverAction), config)
        buffer, _ = agent.collect_rollout(self.env, reset_seed=11)

        metrics = agent.update(buffer)

        diagnostics = metrics.learning_diagnostics
        self.assertIsNotNone(diagnostics)
        self.assertEqual(
            set(diagnostics["terms"]), {"actor", "critic", "entropy"}
        )
        self.assertIn("encoder", diagnostics["terms"]["actor"]["group_norms"])
        self.assertIn("explained_variance_pre", diagnostics)
        self.assertIn("explained_variance_post", diagnostics)
        self.assertEqual(diagnostics["action_update"]["rollout_size"], 4)
        self.assertIn("STAY", diagnostics["action_update"]["actions"])
        self.assertIn("HO_C1", diagnostics["action_update"]["actions"])
        self.assertEqual(
            sum(
                action["chosen_count"]
                for action in diagnostics["action_update"]["actions"].values()
            ),
            4,
        )

    def test_transition_diagnostic_callback_is_aligned_with_buffer(self) -> None:
        agent = SMDPPPO(
            self.observation_size,
            len(HandoverAction),
            replace(self.config, rollout_steps=4),
        )

        def capture(env, observation, mask, action, probabilities):
            return {
                "tic": env.engine.snapshot.tic,
                "action": action,
                "probability_sum": float(probabilities.sum()),
            }

        buffer, _ = agent.collect_rollout(
            self.env,
            reset_seed=11,
            transition_diagnostic_callback=capture,
        )
        self.assertEqual(len(buffer.transition_metadata[: buffer.size]), 4)
        for index, metadata in enumerate(buffer.transition_metadata[: buffer.size]):
            self.assertEqual(metadata["action"], int(buffer.actions[index]))
            self.assertAlmostEqual(metadata["probability_sum"], 1.0)

    def test_candidate_scorer_candidate_logits_are_permutation_equivariant(self) -> None:
        model = ActorCritic(
            62,
            4,
            hidden_sizes=(16,),
            architecture="candidate_scorer_v1",
            candidate_feature_size=15,
        )
        with torch.no_grad():
            model.candidate_head.weight.normal_()
        observation = torch.randn(5, 62)
        permutation = torch.tensor([2, 0, 1])
        permuted = torch.cat(
            (
                observation[:, :17],
                observation[:, 17:].reshape(5, 3, 15)[:, permutation].reshape(5, 45),
            ),
            dim=1,
        )
        logits, _ = model(observation)
        permuted_logits, _ = model(permuted)
        torch.testing.assert_close(permuted_logits[:, 0], logits[:, 0])
        torch.testing.assert_close(permuted_logits[:, 1:], logits[:, 1:][:, permutation])

    def test_candidate_scorer_checkpoint_round_trip(self) -> None:
        config = replace(
            self.config,
            network_architecture="candidate_scorer_v1",
            candidate_feature_size=15,
        )
        agent = SMDPPPO(62, 4, config)
        observation = np.zeros(62, dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "candidate_scorer.pt")
            agent.save_checkpoint(path)
            restored, _ = SMDPPPO.from_checkpoint(path)
        self.assertEqual(restored.model.architecture, "candidate_scorer_v1")
        self.assertEqual(restored.model.candidate_feature_size, 15)
        self.assertEqual(
            agent.predict(observation), restored.predict(observation)
        )

    def test_learning_diagnostics_do_not_change_the_ppo_update(self) -> None:
        collector = SMDPPPO(
            self.observation_size, len(HandoverAction), self.config
        )
        buffer, _ = collector.collect_rollout(self.env, reset_seed=11)
        plain = SMDPPPO(
            self.observation_size, len(HandoverAction), self.config
        )
        instrumented = SMDPPPO(
            self.observation_size,
            len(HandoverAction),
            replace(self.config, learning_diagnostics=True),
        )
        instrumented.model.load_state_dict(plain.model.state_dict())
        instrumented.optimizer.load_state_dict(plain.optimizer.state_dict())

        torch.manual_seed(991)
        plain_metrics = plain.update(buffer)
        torch.manual_seed(991)
        instrumented_metrics = instrumented.update(buffer)

        for (plain_name, plain_parameter), (
            instrumented_name,
            instrumented_parameter,
        ) in zip(
            plain.model.named_parameters(),
            instrumented.model.named_parameters(),
        ):
            self.assertEqual(plain_name, instrumented_name)
            torch.testing.assert_close(plain_parameter, instrumented_parameter)
        self.assertAlmostEqual(
            plain_metrics.policy_loss, instrumented_metrics.policy_loss
        )
        self.assertAlmostEqual(
            plain_metrics.value_loss, instrumented_metrics.value_loss
        )

    def test_checkpoint_manager_saves_all_and_tracks_pareto_qos(self) -> None:
        agent = SMDPPPO(19, len(HandoverAction), self.config)
        with tempfile.TemporaryDirectory() as directory:
            manager = ValidationCheckpointManager(
                directory,
                reference_rate_mbps=10.0,
                reference_rates_by_speed={30: 10.0},
                reference_relative_rate_percent=80.0,
                reference_relative_rates_by_speed={30: 80.0},
                qos_rate_ratio=0.98,
                qos_rate_ratio_per_speed=0.98,
                checkpoint_extra={"dataset_manifest_sha256": "abc123"},
            )
            agent.total_macro_steps = 10
            agent.update_count = 1
            first = manager.record(
                agent,
                _evaluation_metrics(
                    mean_return=1.0,
                    rate_mbps=9.9,
                    rlf_time_percent=2.0,
                    handover_failure_percent=10.0,
                    ping_pong_percent=20.0,
                    handover_attempts=10,
                ),
            )
            agent.total_macro_steps = 20
            agent.update_count = 2
            second = manager.record(
                agent,
                _evaluation_metrics(
                    mean_return=2.0,
                    rate_mbps=9.7,
                    rlf_time_percent=1.0,
                    handover_failure_percent=8.0,
                    ping_pong_percent=15.0,
                    handover_attempts=8,
                ),
            )
            agent.total_macro_steps = 30
            agent.update_count = 3
            third = manager.record(
                agent,
                _evaluation_metrics(
                    mean_return=0.0,
                    rate_mbps=9.8,
                    rlf_time_percent=2.5,
                    handover_failure_percent=11.0,
                    ping_pong_percent=21.0,
                    handover_attempts=11,
                ),
            )
            summary = manager.summary()
            all_names = sorted(os.listdir(summary["all_dir"]))
            pareto_names = sorted(os.listdir(summary["pareto_dir"]))
            _, best_return_extra = SMDPPPO.from_checkpoint(
                summary["best_return"],
                observation_size=19,
                action_size=len(HandoverAction),
            )
            best_qos_agent, _ = SMDPPPO.from_checkpoint(
                summary["best_qos"],
                observation_size=19,
                action_size=len(HandoverAction),
            )
            best_rlf_agent, _ = SMDPPPO.from_checkpoint(
                summary["best_rlf"],
                observation_size=19,
                action_size=len(HandoverAction),
            )
            with open(summary["metrics_json"], encoding="utf-8") as source:
                records = json.load(source)

        self.assertTrue(first["qos_feasible"])
        self.assertFalse(second["qos_feasible"])
        self.assertFalse(third["is_pareto"])
        self.assertEqual(len(all_names), 3)
        self.assertEqual(
            pareto_names,
            ["step_000000010.pt", "step_000000020.pt"],
        )
        self.assertEqual(best_return_extra["dataset_manifest_sha256"], "abc123")
        self.assertEqual(best_qos_agent.total_macro_steps, 10)
        self.assertEqual(best_rlf_agent.total_macro_steps, 20)
        self.assertEqual(summary["qos_feasible_checkpoints"], 2)
        self.assertEqual(summary["pareto_checkpoints"], 2)
        self.assertTrue(records[0]["is_best_qos"])
        self.assertTrue(records[1]["is_best_return"])
        self.assertTrue(records[1]["is_best_rlf"])
        self.assertFalse(records[2]["is_pareto"])

    def test_qos_gate_requires_relative_rate_and_absolute_mbps(self) -> None:
        agent = SMDPPPO(19, len(HandoverAction), self.config)
        with tempfile.TemporaryDirectory() as directory:
            manager = ValidationCheckpointManager(
                directory,
                reference_rate_mbps=10.0,
                reference_rates_by_speed={30: 10.0},
                reference_relative_rate_percent=80.0,
                reference_relative_rates_by_speed={30: 80.0},
                qos_rate_ratio=0.98,
                qos_rate_ratio_per_speed=0.98,
                qos_relative_rate_ratio=0.98,
                qos_relative_rate_ratio_per_speed=0.98,
            )
            agent.total_macro_steps = 10
            result = manager.record(
                agent,
                _evaluation_metrics(
                    mean_return=1.0,
                    rate_mbps=10.0,
                    relative_rate_percent=75.0,
                    rlf_time_percent=1.0,
                    handover_failure_percent=2.0,
                    ping_pong_percent=3.0,
                    handover_attempts=4,
                ),
            )
            summary = manager.summary()

        self.assertFalse(result["qos_feasible"])
        self.assertAlmostEqual(
            result["aggregate_rate_ratio_to_3gpp"], 1.0
        )
        self.assertAlmostEqual(
            result["aggregate_relative_rate_ratio_to_3gpp"],
            75.0 / 80.0,
        )
        self.assertIsNone(summary["best_qos"])

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

    def test_dataset_manifest_records_split_files_and_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            pairs = []
            for trajectory_id in range(3):
                rsrp_file = f"rsrp_30kmh_{trajectory_id}.csv"
                sinr_file = f"sinr_30kmh_{trajectory_id}.csv"
                with open(
                    os.path.join(directory, rsrp_file),
                    "wb",
                ) as source_file:
                    source_file.write(f"rsrp-{trajectory_id}".encode())
                with open(
                    os.path.join(directory, sinr_file),
                    "wb",
                ) as source_file:
                    source_file.write(f"sinr-{trajectory_id}".encode())
                pairs.append(
                    TracePair(
                        30,
                        trajectory_id,
                        rsrp_file,
                        sinr_file,
                    )
                )
            splits = TraceSplits(
                train=(pairs[0],),
                validation=(pairs[1],),
                test=(pairs[2],),
            )
            provenance_path = os.path.join(
                directory,
                "dataset_provenance.json",
            )
            with open(provenance_path, "w", encoding="utf-8") as output_file:
                json.dump({"topology": "test"}, output_file)

            manifest = build_dataset_manifest(
                directory,
                splits,
                selected_speeds=[30],
                split_seed=17,
                split_fractions={
                    "train": 1 / 3,
                    "validation": 1 / 3,
                    "test": 1 / 3,
                },
            )
            repeated_manifest = build_dataset_manifest(
                directory,
                splits,
                selected_speeds=[30],
                split_seed=17,
                split_fractions={
                    "train": 1 / 3,
                    "validation": 1 / 3,
                    "test": 1 / 3,
                },
            )
            paths = write_dataset_manifest(directory, manifest)

            self.assertEqual(manifest["schema_version"], 2)
            self.assertEqual(
                manifest["dataset_provenance"]["file"],
                "dataset_provenance.json",
            )
            self.assertEqual(
                len(manifest["dataset_provenance"]["sha256"]),
                64,
            )
            self.assertEqual(manifest["split_sizes"], {
                "train": 1,
                "validation": 1,
                "test": 1,
            })
            self.assertEqual(manifest["split_fractions"]["train"], 1 / 3)
            self.assertEqual(
                manifest["splits"]["train"][0]["trace_id"],
                "30kmh_0",
            )
            self.assertEqual(
                len(manifest["splits"]["train"][0]["rsrp_sha256"]),
                64,
            )
            self.assertEqual(len(paths["sha256"]), 64)
            self.assertEqual(
                manifest["dataset_content_sha256"],
                repeated_manifest["dataset_content_sha256"],
            )
            self.assertEqual(
                paths["dataset_content_sha256"],
                manifest["dataset_content_sha256"],
            )
            self.assertTrue(os.path.exists(paths["json"]))
            self.assertTrue(os.path.exists(paths["csv"]))


if __name__ == "__main__":
    unittest.main()
