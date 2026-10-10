"""D63 timing x target replay and counterfactual contract tests."""

from types import SimpleNamespace
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ho_optim_drl import Config
from ho_optim_drl.commands.audit_policy_learning import simulate_reward_counterfactual
from ho_optim_drl.commands.audit_timing_target import (
    _apply_source_action,
    _prepare_destination,
    simulate_timing_target,
    summarize_rows,
)
from ho_optim_drl.commands.audit_timing_target_main import (
    _aggregate_route_rows,
    _cluster_bootstrap_means,
    build_local_paired_records,
    decide_gate_g1,
)
from ho_optim_drl.data import CellTopology, RadioTrace
from ho_optim_drl.protocol import HandoverEngine
from ho_optim_drl.smdp import TopologyCandidateObservationBuilder


def _pending_state(
    *, num_cells: int = 4, samples: int = 80
) -> tuple[HandoverEngine, TopologyCandidateObservationBuilder]:
    config = Config()
    config.t_ho_prep_ms = 10
    config.t_ho_exec_ms = 10
    rsrp = np.vstack(
        [
            np.full(samples, value)
            for value in (-80.0, -76.0, -77.0, -78.0)[:num_cells]
        ]
    )
    sinr = np.vstack(
        [
            np.full(samples, value)
            for value in (5.0, 3.0, 2.0, 1.0)[:num_cells]
        ]
    )
    trace = RadioTrace.from_arrays(rsrp, sinr, speed_kmh=30)
    engine = HandoverEngine(config)
    engine.reset(trace)
    builder = TopologyCandidateObservationBuilder(
        config,
        topology=CellTopology.contiguous_sector_groups(num_cells),
        candidate_count=3,
        candidate_order="ranked_v1",
        observation_schema="qos_full_v1",
    )
    builder.reset()
    builder.start_episode(trace)
    while engine.decision_point is None:
        tick = engine.advance_one_tick()
        builder.observe(tick)
    builder.build(engine.snapshot)
    return engine, builder


class D63CounterfactualTests(unittest.TestCase):
    def test_delay_zero_c1_matches_existing_physical_counterfactual(self) -> None:
        engine, builder = _pending_state()
        target = builder.candidate_cells[0]
        self.assertIsNotNone(target)
        existing = simulate_reward_counterfactual(
            engine,
            initial_target=int(target),
            horizon_ticks=(20,),
            reward_schemas=("trajectory_qos_reward_v1",),
        )[20]["physical"]
        audited, _ = simulate_timing_target(
            engine,
            builder,
            delay_ticks=0,
            requested_slot=1,
            horizon_ticks=(20,),
        )
        physical = audited[20]["physical"]
        self.assertAlmostEqual(physical["delivered_mbit"], existing["delivered_mbit"])
        self.assertEqual(
            physical["handover_failures"], existing["handover_failures"]
        )
        self.assertEqual(physical["rlf_starts"], existing["rlf_count"])
        self.assertEqual(physical["ping_pong_count"], existing["ping_pong_count"])
        self.assertAlmostEqual(physical["recovery_seconds"], existing["recovery_seconds"])

    def test_delay_uses_redecision_and_recomputes_execution_candidate(self) -> None:
        engine, builder = _pending_state()
        original_tic = engine.snapshot.tic
        audited, _ = simulate_timing_target(
            engine,
            builder,
            delay_ticks=10,
            requested_slot=1,
            horizon_ticks=(15,),
        )
        execution = audited[15]["execution"]
        self.assertGreaterEqual(execution["executed_tic"], original_tic + 10)
        self.assertEqual(execution["actual_delay_ms"], 100)
        self.assertIsNone(execution["fallback"])

    def test_unavailable_requested_slot_falls_back_to_current_c1(self) -> None:
        engine, builder = _pending_state(num_cells=2)
        audited, _ = simulate_timing_target(
            engine,
            builder,
            delay_ticks=0,
            requested_slot=3,
            horizon_ticks=(5,),
        )
        execution = audited[5]["execution"]
        self.assertEqual(execution["executed_action"], "C1")
        self.assertEqual(execution["fallback"], "c1_unavailable_requested_slot")
        self.assertIsNotNone(execution["executed_target_cell"])

    def test_counterfactual_does_not_mutate_engine_or_builder(self) -> None:
        engine, builder = _pending_state()
        tic = engine.snapshot.tic
        target = engine.decision_point.target_cell
        cells = tuple(builder.candidate_cells)
        simulate_timing_target(
            engine,
            builder,
            delay_ticks=10,
            requested_slot=2,
            horizon_ticks=(10,),
        )
        self.assertEqual(engine.snapshot.tic, tic)
        self.assertEqual(engine.decision_point.target_cell, target)
        self.assertEqual(tuple(builder.candidate_cells), cells)

    def test_trace_end_before_delayed_action_is_censored(self) -> None:
        engine, builder = _pending_state(samples=8)
        audited, _ = simulate_timing_target(
            engine,
            builder,
            delay_ticks=10,
            requested_slot=2,
            horizon_ticks=(20,),
        )
        record = audited[20]
        self.assertTrue(record["horizon_censored"])
        self.assertEqual(
            record["execution"]["status"], "trace_ended_before_execution"
        )
        self.assertIsNone(record["execution"]["executed_tic"])

    def test_source_stay_applies_explicit_redecision_contract(self) -> None:
        engine, builder = _pending_state()
        observation = builder.build(engine.snapshot)
        source = {
            "agent": SimpleNamespace(predict=lambda *args, **kwargs: 0),
            "stay_mode": "a3_redecision_v1",
            "stay_redecision_ms": 100,
        }
        action, event = _apply_source_action(engine, builder, observation, source)
        self.assertEqual(action, 0)
        self.assertIsNotNone(event)
        self.assertIsNone(engine.decision_point)
        for _ in range(10):
            tick = engine.advance_one_tick()
            builder.observe(tick)
        self.assertIsNotNone(engine.decision_point)

    def test_pilot_output_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            destination = Path(directory, "output")
            _prepare_destination(destination)
            (destination / "artifact.json").write_text("{}", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                _prepare_destination(destination)

    def test_summary_keeps_pilot_gate_unresolved(self) -> None:
        common = {
            "state_id": "s0",
            "source": "d61",
            "speed_kmh": 30,
            "trajectory_id": 1,
            "horizon_seconds": 1.0,
            "execution": {"fallback": None, "executed_tic": 10},
        }
        physical = {
            "delivered_mbit": 1.0,
            "handover_failures": 0,
            "rlf_starts": 0,
            "pure_rlf_starts": 0,
            "ping_pong_count": 0,
            "recovery_seconds": 0.0,
            "handover_interruption_seconds": 0.0,
        }
        rows = []
        for delay in (0, 100):
            for slot in (1, 2, 3):
                candidate = dict(physical)
                if delay == 100 and slot == 1:
                    candidate["delivered_mbit"] = 1.1
                rows.append(
                    {
                        **common,
                        "delay_ms": delay,
                        "requested_slot": slot,
                        "physical": candidate,
                    }
                )
        summary = summarize_rows(rows)
        record = summary["summaries"][0]
        self.assertEqual(record["strict_positive_states"], 1)
        self.assertEqual(record["strict_useful_branch_counts"]["timing_only"], 1)
        self.assertEqual(
            record["strict_positive_state_counts_by_category"]["timing_only"], 1
        )
        self.assertEqual(
            summary["gate_g1"]["status"],
            "NOT_EVALUATED_PILOT_CORRECTNESS_ONLY",
        )

    def test_main_paired_records_separate_timing_target_and_joint(self) -> None:
        physical = {
            "delivered_mbit": 1.0,
            "relative_rate_percent": 50.0,
            "handover_failures": 0,
            "rlf_starts": 0,
            "pure_rlf_starts": 0,
            "ping_pong_count": 0,
            "recovery_seconds": 0.0,
            "handover_interruption_seconds": 0.0,
        }
        rows = []
        for delay in (0, 100):
            for slot in (1, 2, 3):
                candidate = dict(physical)
                if delay == 100 and slot == 1:
                    candidate["delivered_mbit"] = 1.1
                if delay == 0 and slot == 2:
                    candidate["delivered_mbit"] = 1.2
                if delay == 100 and slot == 2:
                    candidate["delivered_mbit"] = 1.3
                rows.append(
                    {
                        "state_id": "s0",
                        "source": "d61",
                        "speed_kmh": 30,
                        "trajectory_id": 7,
                        "horizon_seconds": 1.0,
                        "delay_ms": delay,
                        "requested_slot": slot,
                        "physical": candidate,
                        "execution": {"executed_slot": slot, "fallback": None},
                        "horizon_censored": False,
                    }
                )
        records = {
            row["category"]: row for row in build_local_paired_records(rows)
        }
        self.assertEqual(records["timing_only"]["best_delay_ms"], 100)
        self.assertEqual(records["timing_only"]["best_requested_slot"], 1)
        self.assertEqual(records["target_only"]["best_requested_slot"], 2)
        self.assertEqual(records["joint"]["best_delay_ms"], 100)
        self.assertEqual(records["joint"]["best_requested_slot"], 2)
        self.assertAlmostEqual(records["joint"]["delta"]["delivered_mbit"], 0.3)

    def test_route_cluster_bootstrap_is_deterministic(self) -> None:
        records = [
            {"speed_kmh": speed, "trajectory_id": route, "value": speed + route}
            for speed in (30, 50)
            for route in (1, 2, 3)
            for _ in range(2)
        ]
        kwargs = {
            "metric_getters": {"value": lambda row: float(row["value"])},
            "resamples": 100,
            "seed": 29,
        }
        first = _cluster_bootstrap_means(records, **kwargs)
        second = _cluster_bootstrap_means(records, **kwargs)
        self.assertEqual(first, second)
        self.assertAlmostEqual(first["value"][0], 42.0)

    def test_gate_requires_point_threshold_and_interval_direction(self) -> None:
        control = {
            "mean_rate_mbps": 20.0,
            "mean_relative_rate_percent": 90.0,
            "handover_failure_percent": 10.0,
            "rlf_time_percent": 2.0,
            "ping_pong_percent": 30.0,
        }
        treatment = dict(control)
        treatment.update(
            mean_rate_mbps=20.2,
            mean_relative_rate_percent=90.6,
            handover_failure_percent=9.0,
            rlf_time_percent=1.8,
            ping_pong_percent=29.0,
        )
        paired = {
            "mean_relative_rate_percent": {"delta": 0.6, "ci95": [0.1, 1.0]},
            "handover_failure_percent": {"delta": -1.0, "ci95": [-2.0, 0.0]},
            "rlf_time_percent": {"delta": -0.2, "ci95": [-0.4, 0.0]},
            "ping_pong_percent": {"delta": -1.0, "ci95": [-2.0, 0.0]},
        }
        summary = {
            "aggregate": {
                "always_ho_c1": control,
                "timing_only_reference": treatment,
                "joint_reference": treatment,
            },
            "paired_comparisons": {
                name: {
                    "aggregate": paired,
                    "qos_retention": {
                        "mean_rate_ratio": 1.01,
                        "mean_relative_rate_ratio": 1.006,
                        "relative_rate_ratio_by_speed": {"30": 1.0},
                    },
                }
                for name in ("timing_only_reference", "joint_reference")
            },
        }
        self.assertEqual(decide_gate_g1(summary)["status"], "PASS")

    def test_full_route_metric_definitions_use_raw_denominators(self) -> None:
        rows = [
            {
                "estimated_rate_mbps": 10.0,
                "relative_rate_percent": 80.0,
                "total_time_s": 10.0,
                "handover_attempts": 2,
                "handover_successes": 1,
                "handover_failures": 1,
                "rlf_starts": 1,
                "pure_rlf_starts": 0,
                "recovery_seconds": 1.0,
                "ping_pong_count": 1,
                "handover_interruption_seconds": 0.4,
                "decision_count": 3,
                "stay_actions": 1,
                "handover_actions": 2,
            },
            {
                "estimated_rate_mbps": 20.0,
                "relative_rate_percent": 100.0,
                "total_time_s": 30.0,
                "handover_attempts": 8,
                "handover_successes": 7,
                "handover_failures": 1,
                "rlf_starts": 1,
                "pure_rlf_starts": 0,
                "recovery_seconds": 1.0,
                "ping_pong_count": 1,
                "handover_interruption_seconds": 0.8,
                "decision_count": 9,
                "stay_actions": 1,
                "handover_actions": 8,
            },
        ]
        aggregate = _aggregate_route_rows(rows)
        self.assertAlmostEqual(aggregate["mean_rate_mbps"], 15.0)
        self.assertAlmostEqual(aggregate["mean_relative_rate_percent"], 90.0)
        self.assertAlmostEqual(aggregate["handover_failure_percent"], 20.0)
        self.assertAlmostEqual(aggregate["rlf_time_percent"], 5.0)
        self.assertAlmostEqual(aggregate["ping_pong_percent"], 25.0)


if __name__ == "__main__":
    unittest.main()
