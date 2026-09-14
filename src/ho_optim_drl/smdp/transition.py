"""Decision-to-decision execution and SMDP transition storage."""

from dataclasses import dataclass
import math

import numpy as np

from ho_optim_drl.data import RadioTrace
from ho_optim_drl.protocol import HandoverEngine

from .actions import HandoverAction
from .observation import ObservationBuilder
from .reward import RewardBreakdown, RewardModel


@dataclass(frozen=True)
class MacroTransition:
    """One decision-to-decision transition with variable duration."""

    observation: np.ndarray
    action: HandoverAction
    reward: RewardBreakdown
    next_observation: np.ndarray | None
    duration_ticks: int
    duration_seconds: float
    discount: float
    terminated: bool
    truncated: bool

    def __post_init__(self) -> None:
        if (
            not isinstance(self.duration_ticks, int)
            or isinstance(self.duration_ticks, bool)
            or self.duration_ticks <= 0
        ):
            raise ValueError("A macro-transition must contain at least one tick.")
        if (
            not math.isfinite(self.duration_seconds)
            or self.duration_seconds <= 0
        ):
            raise ValueError("duration_seconds must be finite and positive.")
        if not math.isfinite(self.discount) or not 0 < self.discount <= 1:
            raise ValueError("discount must be in (0, 1].")
        if self.terminated and self.next_observation is not None:
            raise ValueError(
                "A terminated transition cannot expose a next decision."
            )
        if (
            not self.terminated
            and not self.truncated
            and self.next_observation is None
        ):
            raise ValueError(
                "A continuing transition must expose the next decision."
            )

        object.__setattr__(
            self,
            "observation",
            self._readonly_observation(self.observation),
        )
        if self.next_observation is not None:
            object.__setattr__(
                self,
                "next_observation",
                self._readonly_observation(self.next_observation),
            )

    @staticmethod
    def _readonly_observation(observation: np.ndarray) -> np.ndarray:
        copied = np.array(observation, dtype=np.float32, copy=True)
        if copied.ndim != 1 or not np.all(np.isfinite(copied)):
            raise ValueError(
                "Macro-transition observations must be finite 1-D arrays."
            )
        copied.setflags(write=False)
        return copied


class MacroTransitionBuilder:
    """
    Execute one action and advance to the next A3 decision or trace end.

    The uncontrollable prefix before the first A3 decision is consumed by
    ``reset()`` and is not assigned to an action. Every later primitive tick
    belongs to exactly one macro-transition.
    """

    def __init__(
        self,
        engine: HandoverEngine,
        observation_builder: ObservationBuilder,
        reward_model: RewardModel,
        *,
        gamma_per_second: float = 0.99,
    ) -> None:
        if not math.isfinite(gamma_per_second):
            raise ValueError("gamma_per_second must be finite.")
        if not 0 < gamma_per_second <= 1:
            raise ValueError("gamma_per_second must be in (0, 1].")

        self.engine = engine
        self.observation_builder = observation_builder
        self.reward_model = reward_model
        self.gamma_per_second = gamma_per_second
        self._current_observation: np.ndarray | None = None
        self._terminated = False
        self._uncontrolled_duration_ticks = 0
        self._uncontrolled_reward = RewardBreakdown()

    @property
    def current_observation(self) -> np.ndarray | None:
        """Return a protected copy of the current decision observation."""
        if self._current_observation is None:
            return None
        return np.array(self._current_observation, copy=True)

    @property
    def terminated(self) -> bool:
        return self._terminated

    @property
    def uncontrolled_duration_ticks(self) -> int:
        """Ticks before the first decision, which no action can influence."""
        return self._uncontrolled_duration_ticks

    @property
    def uncontrolled_reward(self) -> RewardBreakdown:
        """Diagnostic reward accumulated before the first decision."""
        return self._uncontrolled_reward

    def reset(self, trace: RadioTrace) -> np.ndarray | None:
        """Reset all components and advance to the first A3 decision."""
        self.engine.reset(trace)
        self.observation_builder.reset()
        self.reward_model.reset()
        self._current_observation = None
        self._terminated = False
        self._uncontrolled_duration_ticks = 0
        self._uncontrolled_reward = RewardBreakdown()

        while True:
            tick = self.engine.advance_one_tick()
            self.observation_builder.observe(tick)
            self._uncontrolled_reward = (
                self._uncontrolled_reward + self.reward_model.compute(tick)
            )
            self._uncontrolled_duration_ticks += 1

            if tick.terminated:
                self._terminated = True
                return None
            if self.engine.decision_point is not None:
                self._current_observation = self.observation_builder.build(
                    self.engine.snapshot
                )
                return self.current_observation

    def step(self, action: HandoverAction | int) -> MacroTransition:
        """Apply one decision action and collect its complete macro outcome."""
        if self._terminated:
            raise RuntimeError("The trace episode has already terminated.")
        if (
            self.engine.decision_point is None
            or self._current_observation is None
        ):
            raise RuntimeError("No A3 decision is available for a macro-step.")

        try:
            selected_action = HandoverAction(action)
        except ValueError as error:
            raise ValueError(f"Unsupported handover action: {action}") from error

        observation = self._current_observation
        self._apply_action(selected_action)
        accumulated_reward = RewardBreakdown()
        duration_ticks = 0

        while True:
            tick = self.engine.advance_one_tick()
            self.observation_builder.observe(tick)
            accumulated_reward = (
                accumulated_reward + self.reward_model.compute(tick)
            )
            duration_ticks += 1

            if tick.terminated:
                self._current_observation = None
                self._terminated = True
                break
            if self.engine.decision_point is not None:
                self._current_observation = self.observation_builder.build(
                    self.engine.snapshot
                )
                break

        duration_seconds = (
            duration_ticks * self.engine.config.delta_t_ms / 1000.0
        )
        return MacroTransition(
            observation=observation,
            action=selected_action,
            reward=accumulated_reward,
            next_observation=self._current_observation,
            duration_ticks=duration_ticks,
            duration_seconds=duration_seconds,
            discount=self.gamma_per_second**duration_seconds,
            terminated=self._terminated,
            truncated=False,
        )

    def _apply_action(self, action: HandoverAction) -> None:
        if action is HandoverAction.HANDOVER:
            self.engine.start_handover()
        elif action is HandoverAction.WAIT_50:
            self.engine.wait_50()
        elif action is HandoverAction.STAY:
            self.engine.stay()
        else:
            raise ValueError(f"Unsupported handover action: {action}")
