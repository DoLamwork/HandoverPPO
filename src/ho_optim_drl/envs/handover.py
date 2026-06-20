"""Gymnasium wrapper around the decision-to-decision handover process."""

from __future__ import annotations

import copy
import os
from collections.abc import Sequence
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ho_optim_drl.config import Config
from ho_optim_drl.data import (
    RadioTrace,
    TracePair,
    discover_trace_pairs,
    load_radio_trace,
)
from ho_optim_drl.protocol import DecisionPoint, HandoverEngine
from ho_optim_drl.smdp import (
    A3ObservationBuilder,
    A3ObservationConfig,
    HandoverAction,
    HandoverRewardConfig,
    HandoverRewardModel,
    MacroTransition,
    MacroTransitionBuilder,
    RewardBreakdown,
)


class HandoverSMDPEnv(gym.Env[np.ndarray, int]):
    """
    Binary STAY/HANDOVER environment evaluated only at completed A3 events.

    A reset advances through the uncontrollable trace prefix to the first A3
    decision. A step executes one action and advances internally to the next
    decision or the end of the trajectory.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        protocol_config: Config | None = None,
        *,
        traces: Sequence[RadioTrace] | None = None,
        data_dir: str | os.PathLike[str] | None = None,
        trace_pairs: Sequence[TracePair] | None = None,
        observation_config: A3ObservationConfig | None = None,
        reward_config: HandoverRewardConfig | None = None,
        gamma_per_second: float = 0.99,
    ) -> None:
        super().__init__()
        self.config = copy.deepcopy(protocol_config or Config())
        self.config.validate()

        self._data_dir: str | None = None
        self._traces: tuple[RadioTrace, ...] | None = None
        self._trace_pairs: tuple[TracePair, ...] | None = None
        self._configure_sources(traces, data_dir, trace_pairs)

        observation_builder = A3ObservationBuilder(
            self.config,
            observation_config,
        )
        reward_model = HandoverRewardModel(self.config, reward_config)
        self.transition_builder = MacroTransitionBuilder(
            HandoverEngine(self.config),
            observation_builder,
            reward_model,
            gamma_per_second=gamma_per_second,
        )

        self.action_space = spaces.Discrete(len(HandoverAction))
        self.observation_space = spaces.Box(
            low=observation_builder.low,
            high=observation_builder.high,
            dtype=np.float32,
        )

        self._current_source_index: int | None = None
        self._current_trace: RadioTrace | None = None
        self._episode_started = False
        self._known_no_decision_sources: set[int] = set()
        self._episode_return = 0.0
        self._episode_macro_steps = 0
        self._episode_duration_ticks = 0

    @property
    def engine(self) -> HandoverEngine:
        """Expose the shared protocol engine for evaluation and diagnostics."""
        return self.transition_builder.engine

    @property
    def num_trajectories(self) -> int:
        return (
            len(self._traces)
            if self._traces is not None
            else len(self._trace_pairs or ())
        )

    @property
    def current_trace(self) -> RadioTrace | None:
        return self._current_trace

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Select a trajectory and advance it to the first A3 decision."""
        super().reset(seed=seed)
        self._episode_started = False
        self._current_source_index = None
        self._current_trace = None
        self._episode_return = 0.0
        self._episode_macro_steps = 0
        self._episode_duration_ticks = 0

        candidates = self._candidate_indices(options or {})
        skipped: list[str] = []
        for source_index in candidates:
            if source_index in self._known_no_decision_sources:
                skipped.append(self._source_id(source_index))
                continue

            trace = self._load_trace(source_index)
            observation = self.transition_builder.reset(trace)
            if observation is None:
                self._known_no_decision_sources.add(source_index)
                skipped.append(self._trace_id(source_index, trace))
                continue

            self._current_source_index = source_index
            self._current_trace = trace
            self._episode_started = True
            self._episode_duration_ticks = (
                self.transition_builder.uncontrolled_duration_ticks
            )
            info = self._reset_info(source_index, trace, skipped)
            return self._observation_copy(observation), info

        requested = "requested trajectory" if options else "trajectory set"
        raise RuntimeError(
            f"The {requested} contains no completed A3 decision."
        )

    def step(
        self,
        action: int,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Execute one macro action and return the next decision boundary."""
        if not self._episode_started or self._current_trace is None:
            raise RuntimeError("Call reset() before step().")
        if not self.action_space.contains(action):
            raise ValueError(f"Action {action!r} is outside {self.action_space}.")

        decision = self.engine.decision_point
        transition = self.transition_builder.step(HandoverAction(int(action)))
        self._episode_return += transition.reward.total
        self._episode_macro_steps += 1
        self._episode_duration_ticks += transition.duration_ticks
        observation = (
            self._observation_copy(transition.next_observation)
            if transition.next_observation is not None
            else np.zeros(self.observation_space.shape, dtype=np.float32)
        )
        info = self._step_info(transition, decision)
        return (
            observation,
            float(transition.reward.total),
            bool(transition.terminated),
            bool(transition.truncated),
            info,
        )

    def _configure_sources(
        self,
        traces: Sequence[RadioTrace] | None,
        data_dir: str | os.PathLike[str] | None,
        trace_pairs: Sequence[TracePair] | None,
    ) -> None:
        if traces is not None:
            if data_dir is not None or trace_pairs is not None:
                raise ValueError(
                    "Use either in-memory traces or data_dir/trace_pairs."
                )
            self._traces = tuple(traces)
            if not self._traces:
                raise ValueError("At least one RadioTrace is required.")
            if not all(isinstance(trace, RadioTrace) for trace in self._traces):
                raise TypeError("traces must contain only RadioTrace objects.")
            return

        if data_dir is None:
            raise ValueError(
                "data_dir is required when in-memory traces are not provided."
            )
        self._data_dir = os.fspath(data_dir)
        pairs = (
            tuple(trace_pairs)
            if trace_pairs is not None
            else tuple(discover_trace_pairs(self._data_dir))
        )
        if not pairs:
            raise ValueError("At least one TracePair is required.")
        if not all(isinstance(pair, TracePair) for pair in pairs):
            raise TypeError("trace_pairs must contain only TracePair objects.")
        self._trace_pairs = pairs

    def _candidate_indices(self, options: dict[str, Any]) -> list[int]:
        allowed = {"trace_index", "trace_id"}
        unknown = set(options) - allowed
        if unknown:
            raise ValueError(f"Unsupported reset options: {sorted(unknown)}")
        if "trace_index" in options and "trace_id" in options:
            raise ValueError("Specify only one of trace_index or trace_id.")

        if "trace_index" in options:
            index = options["trace_index"]
            if (
                not isinstance(index, (int, np.integer))
                or isinstance(index, bool)
                or not 0 <= int(index) < self.num_trajectories
            ):
                raise ValueError("trace_index is outside the trajectory set.")
            return [int(index)]

        if "trace_id" in options:
            trace_id = options["trace_id"]
            if not isinstance(trace_id, str):
                raise TypeError("trace_id must be a string.")
            matches = [
                index
                for index in range(self.num_trajectories)
                if self._source_id(index) == trace_id
            ]
            if not matches:
                raise ValueError(f"Unknown trace_id: {trace_id!r}.")
            return matches

        return [
            int(index)
            for index in self.np_random.permutation(self.num_trajectories)
        ]

    def _load_trace(self, source_index: int) -> RadioTrace:
        if self._traces is not None:
            trace = self._traces[source_index]
        else:
            pair = self._trace_pairs[source_index]
            trace = load_radio_trace(
                self.config,
                self._data_dir,
                pair.rsrp_file,
                pair.sinr_file,
            )
        if trace.delta_t_ms != self.config.delta_t_ms:
            raise ValueError(
                f"Trace uses {trace.delta_t_ms} ms samples, expected "
                f"{self.config.delta_t_ms} ms."
            )
        return trace

    def _source_id(self, source_index: int) -> str:
        if self._traces is not None:
            return self._traces[source_index].trace_id or f"memory_{source_index}"
        pair = self._trace_pairs[source_index]
        return f"{pair.speed_kmh}kmh_{pair.trajectory_id}"

    def _trace_id(self, source_index: int, trace: RadioTrace) -> str:
        return trace.trace_id or self._source_id(source_index)

    def _source_metadata(self, source_index: int) -> dict[str, Any]:
        if self._trace_pairs is None:
            return {}
        pair = self._trace_pairs[source_index]
        return {
            "speed_kmh": pair.speed_kmh,
            "trajectory_id": pair.trajectory_id,
        }

    def _reset_info(
        self,
        source_index: int,
        trace: RadioTrace,
        skipped: list[str],
    ) -> dict[str, Any]:
        info = {
            "trace_index": source_index,
            "trace_id": self._trace_id(source_index, trace),
            "num_samples": trace.num_samples,
            "uncontrolled_duration_ticks": (
                self.transition_builder.uncontrolled_duration_ticks
            ),
            "uncontrolled_duration_seconds": (
                self.transition_builder.uncontrolled_duration_ticks
                * self.config.delta_t_ms
                / 1000.0
            ),
            "uncontrolled_reward": self._reward_dict(
                self.transition_builder.uncontrolled_reward
            ),
            "skipped_no_decision_traces": tuple(skipped),
            "decision": self._decision_dict(self.engine.decision_point),
        }
        info.update(self._source_metadata(source_index))
        return info

    def _step_info(
        self,
        transition: MacroTransition,
        decision: DecisionPoint | None,
    ) -> dict[str, Any]:
        info = {
            "trace_index": self._current_source_index,
            "trace_id": self._trace_id(
                self._current_source_index,
                self._current_trace,
            ),
            "action": transition.action.name.lower(),
            "reward": self._reward_dict(transition.reward),
            "duration_ticks": transition.duration_ticks,
            "duration_seconds": transition.duration_seconds,
            "discount": transition.discount,
            "episode_return": float(self._episode_return),
            "episode_macro_steps": self._episode_macro_steps,
            "episode_duration_ticks": self._episode_duration_ticks,
            "episode_duration_seconds": (
                self._episode_duration_ticks
                * self.config.delta_t_ms
                / 1000.0
            ),
            "decision": self._decision_dict(decision),
            "next_decision": self._decision_dict(self.engine.decision_point),
        }
        info.update(self._source_metadata(self._current_source_index))
        if transition.terminated:
            info["episode_metrics"] = self.engine.get_log()
        return info

    @staticmethod
    def _reward_dict(reward: RewardBreakdown) -> dict[str, float]:
        return {
            "rate": float(reward.rate),
            "rlf": float(reward.rlf),
            "ping_pong": float(reward.ping_pong),
            "total": float(reward.total),
        }

    @staticmethod
    def _decision_dict(
        decision: DecisionPoint | None,
    ) -> dict[str, int | float] | None:
        if decision is None:
            return None
        return {
            "tic": decision.tic,
            "serving_cell": decision.serving_cell,
            "target_cell": decision.target_cell,
            "a3_start_tic": decision.a3_start_tic,
            "a3_margin_db": decision.a3_margin_db,
        }

    def _observation_copy(self, observation: np.ndarray) -> np.ndarray:
        copied = np.array(observation, dtype=np.float32, copy=True)
        if not self.observation_space.contains(copied):
            raise RuntimeError("Observation is outside the declared space.")
        return copied
