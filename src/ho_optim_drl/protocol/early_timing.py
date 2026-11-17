"""Opt-in early-window timing adapter used by the B2/B3 controls."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .engine import HandoverEngine
from .primitives import GeneralCounter, HOEventHandler
from .reference import A3ReferenceSimulator
from .types import DecisionPoint

if TYPE_CHECKING:
    from ho_optim_drl.config import Config


@dataclass(frozen=True)
class EarlyTimingDecisionPoint(DecisionPoint):
    """A decision opened by the early RSRP window, not completed A3+TTT.

    The inherited ``a3_start_tic`` and ``a3_margin_db`` fields are retained for
    compatibility with causal observation builders.  For this subtype they
    mean early-window start tic and margin above the early entry threshold.
    ``trigger`` makes that distinction explicit.
    """

    trigger: str = "early_window"
    stage: str = "entry"
    scheduled_delay_ticks: int = 0


@dataclass(frozen=True)
class ShadowA3DecisionPoint(DecisionPoint):
    """A decision emitted by an observational A3+TTT shadow tracker."""

    trigger: str = "shadow_a3_maturity"
    stage: str = "maturity"
    scheduled_delay_ticks: int = 0


class EarlyTimingHandoverEngine(HandoverEngine):
    """Handover engine offering decisions before A3+TTT.

    The physical RLM, HO preparation/execution, RLF recovery, rate accounting,
    and trace stepping are inherited unchanged.  Only measurement-triggered
    decision creation is replaced by a separate early-window adapter.
    """

    def __init__(
        self,
        protocol_config: "Config",
        *,
        enter_delta_db: float = 1.0,
        exit_delta_db: float = 0.0,
        enable_early_decisions: bool = True,
        enable_shadow_decisions: bool = False,
    ) -> None:
        if not np.isfinite(enter_delta_db) or not np.isfinite(exit_delta_db):
            raise ValueError("Early-window thresholds must be finite.")
        if exit_delta_db >= enter_delta_db:
            raise ValueError("exit_delta_db must be below enter_delta_db.")
        super().__init__(protocol_config)
        self.enter_delta_db = float(enter_delta_db)
        self.exit_delta_db = float(exit_delta_db)
        self.enable_early_decisions = bool(enable_early_decisions)
        self.enable_shadow_decisions = bool(enable_shadow_decisions)
        self._early_episode: tuple[int, int, int] | None = None
        self._early_due_tic: int | None = None
        self._early_delay_ticks: int | None = None
        self.early_entries = 0
        self.early_schedules = 0
        self.early_schedule_cancellations = 0
        self.early_target_changes = 0
        self.early_executions = 0
        self.early_execution_delays_ticks: list[int] = []
        self._shadow_event_handler = HOEventHandler(
            hys_a3=self.config.a3_hys,
            off_a3=self.config.a3_off,
            debug_lvl=0,
        )
        self._shadow_ttt = GeneralCounter(
            self.config.a3_ttt, 1, "ShadowTTT", False, 0
        )
        self._shadow_matured: tuple[int, int, int, int] | None = None
        self._shadow_due_tic: int | None = None
        self._shadow_delay_ticks: int | None = None
        self.shadow_maturities: list[tuple[int, int, int, int]] = []

    def reset(self, trace):  # type: ignore[no-untyped-def]
        snapshot = super().reset(trace)
        self._early_episode = None
        self._early_due_tic = None
        self._early_delay_ticks = None
        self.early_entries = 0
        self.early_schedules = 0
        self.early_schedule_cancellations = 0
        self.early_target_changes = 0
        self.early_executions = 0
        self.early_execution_delays_ticks = []
        self._shadow_event_handler.reset(0)
        self._shadow_ttt.reset(0, clear_history=True)
        self._shadow_matured = None
        self._shadow_due_tic = None
        self._shadow_delay_ticks = None
        self.shadow_maturities = []
        return snapshot

    def defer_early_handover(self, *, delay_ticks: int) -> None:
        """Schedule another decision if this exact early episode stays valid."""
        decision = self._require_early_decision()
        if decision.stage != "entry":
            raise RuntimeError("Only an entry decision can be deferred.")
        if (
            not isinstance(delay_ticks, int)
            or isinstance(delay_ticks, bool)
            or delay_ticks <= 0
        ):
            raise ValueError("delay_ticks must be a positive integer.")
        self._early_due_tic = decision.tic + delay_ticks
        self._early_delay_ticks = delay_ticks
        self._decision_point = None
        self.early_schedules += 1

    def wait_for_redecision(self, *, delay_ticks: int) -> None:
        """Take one WAIT action and re-offer a valid timing state later."""
        decision = self._require_decision()
        if (
            not isinstance(delay_ticks, int)
            or isinstance(delay_ticks, bool)
            or delay_ticks <= 0
        ):
            raise ValueError("delay_ticks must be a positive integer.")
        due_tic = decision.tic + delay_ticks
        if isinstance(decision, EarlyTimingDecisionPoint):
            self._early_due_tic = due_tic
            self._early_delay_ticks = delay_ticks
        elif isinstance(decision, ShadowA3DecisionPoint):
            self._shadow_due_tic = due_tic
            self._shadow_delay_ticks = delay_ticks
            if self.enable_early_decisions and self._early_episode is not None:
                self._early_due_tic = due_tic
                self._early_delay_ticks = delay_ticks
        else:
            raise RuntimeError("The pending decision is not a timing decision.")
        self._decision_point = None

    def start_handover(self, target_cell: int | None = None) -> None:
        decision = self._require_decision()
        is_early = isinstance(decision, EarlyTimingDecisionPoint)
        is_timing = is_early or isinstance(decision, ShadowA3DecisionPoint)
        super().start_handover(target_cell)
        if is_timing:
            if is_early:
                self.early_executions += 1
                self.early_execution_delays_ticks.append(
                    decision.tic - decision.a3_start_tic
                )
            self._clear_timing_trackers(decision.tic)
        if is_early:
            return

    def early_audit_state(self) -> dict[str, int | list[int]]:
        return {
            "entries": self.early_entries,
            "schedules": self.early_schedules,
            "schedule_cancellations": self.early_schedule_cancellations,
            "target_changes": self.early_target_changes,
            "executions": self.early_executions,
            "execution_delays_ticks": list(self.early_execution_delays_ticks),
            "shadow_maturities": len(self.shadow_maturities),
            "shadow_ttt_starts": len(self._shadow_ttt.start_idxs),
            "shadow_ttt_aborts": len(self._shadow_ttt.aborted_idxs),
        }

    def _measurement_event_handler(
        self, rsrp_db: np.ndarray, tic: int
    ) -> None:
        """Update observational A3 shadow and causal early-window decisions."""
        self._update_shadow_a3(rsrp_db, tic)
        if (
            not self.rrc.is_connected
            or self.cntr["ho_prep"].pending
            or self.cntr["ho_exec"].pending
            or self.cntr["rlfr"].pending
        ):
            self._cancel_early_episode()
            return
        if not self.enable_early_decisions:
            self._cancel_early_episode()
            return
        serving = self.rrc.pcell
        target = self.rrc.ncell
        if serving is None or target is None:
            self._cancel_early_episode()
            return
        delta_db = float(rsrp_db[target] - rsrp_db[serving])

        if self._early_episode is not None:
            episode_serving, episode_target, _ = self._early_episode
            if serving != episode_serving or target != episode_target:
                self.early_target_changes += int(target != episode_target)
                self._cancel_early_episode()
            elif delta_db <= self.exit_delta_db:
                self._cancel_early_episode()
                return

        if self._early_episode is None:
            if delta_db <= self.enter_delta_db:
                return
            self._early_episode = (serving, target, tic)
            self.early_entries += 1
            if self._decision_point is None:
                self._decision_point = self._new_early_decision(
                    tic=tic,
                    serving=serving,
                    target=target,
                    start_tic=tic,
                    delta_db=delta_db,
                    stage="entry",
                    delay_ticks=0,
                )
            return

        if self._early_due_tic is not None and tic >= self._early_due_tic:
            if self._decision_point is not None:
                # A shadow maturity/redecision on the same tick has priority.
                self._early_due_tic = None
                self._early_delay_ticks = None
                return
            _, _, start_tic = self._early_episode
            delay_ticks = self._early_delay_ticks
            if delay_ticks is None:
                raise RuntimeError("Scheduled early decision has no delay.")
            self._decision_point = self._new_early_decision(
                tic=tic,
                serving=serving,
                target=target,
                start_tic=start_tic,
                delta_db=delta_db,
                stage="scheduled",
                delay_ticks=delay_ticks,
            )
            self._early_due_tic = None
            self._early_delay_ticks = None

    def _action_controlled_ho_state_machine(
        self,
        sinr_rlm_db: np.ndarray,
        sinr_rate_db: np.ndarray,
        tic: int,
    ) -> None:
        """Advance only physical HO state; early adapter owns trigger timing."""
        A3ReferenceSimulator._ho_state_machine(
            self, sinr_rlm_db, tic, sinr_rate_db
        )

    def _cancel_early_episode(self) -> None:
        if self._early_due_tic is not None:
            self.early_schedule_cancellations += 1
        self._early_episode = None
        self._early_due_tic = None
        self._early_delay_ticks = None

    def _update_shadow_a3(self, rsrp_db: np.ndarray, tic: int) -> None:
        """Mirror B0 A3/TTT without starting physical handover."""
        if self._shadow_ttt.active:
            self._shadow_ttt.step(tic)
        if not self.rrc.is_connected:
            self._reset_shadow(tic)
            return
        serving = self.rrc.pcell
        target = self.rrc.ncell
        entered = self._shadow_event_handler.event_a3(
            rsrp_db, serving, target, tic
        )
        if (
            self.cntr["ho_prep"].pending
            or self.cntr["ho_exec"].pending
            or self.cntr["rlfr"].pending
        ):
            # B0 continues to update the A3 hysteresis condition during HO
            # preparation, but it does not start another TTT.  Preserve that
            # state: resetting the event handler here changes the next A3
            # entry after a successful handover.
            self._reset_shadow(tic)
            return

        if self._shadow_matured is not None:
            mature_serving, mature_target, _, _ = self._shadow_matured
            if not entered or serving != mature_serving or target != mature_target:
                self._shadow_matured = None
                self._shadow_due_tic = None
                self._shadow_delay_ticks = None
                self._shadow_event_handler.target_cell = None
            else:
                if (
                    self.enable_shadow_decisions
                    and self._shadow_due_tic is not None
                    and tic >= self._shadow_due_tic
                ):
                    _, _, start_tic, _ = self._shadow_matured
                    delay_ticks = self._shadow_delay_ticks
                    if delay_ticks is None:
                        raise RuntimeError("Shadow redecision has no delay.")
                    self._offer_shadow_decision(
                        tic=tic,
                        serving=int(serving),
                        target=int(target),
                        start_tic=start_tic,
                        rsrp_db=rsrp_db,
                        stage="redecision",
                        delay_ticks=delay_ticks,
                    )
                    self._shadow_due_tic = None
                    self._shadow_delay_ticks = None
                return

        if entered:
            if not self._shadow_ttt.pending:
                self._shadow_event_handler.target_cell = target
                self._shadow_ttt.start(tic)
            if target != self._shadow_event_handler.target_cell:
                if self._shadow_ttt.pending:
                    self._shadow_ttt.reset(tic)
        else:
            if self._shadow_ttt.pending:
                self._shadow_ttt.reset(tic)

        if self._shadow_ttt.reached_max:
            latched_target = self._shadow_event_handler.target_cell
            if serving is None or latched_target is None:
                self._shadow_ttt.reset(tic)
                return
            start_tic = int(self._shadow_ttt.start_idxs[-1])
            self._shadow_ttt.reset(tic)
            maturity = (int(serving), int(latched_target), start_tic, tic)
            self._shadow_matured = maturity
            self.shadow_maturities.append(maturity)
            if self.enable_shadow_decisions:
                self._offer_shadow_decision(
                    tic=tic,
                    serving=int(serving),
                    target=int(latched_target),
                    start_tic=start_tic,
                    rsrp_db=rsrp_db,
                    stage="maturity",
                    delay_ticks=0,
                )

    def _offer_shadow_decision(
        self,
        *,
        tic: int,
        serving: int,
        target: int,
        start_tic: int,
        rsrp_db: np.ndarray,
        stage: str,
        delay_ticks: int,
    ) -> None:
        if self._decision_point is not None:
            return
        margin_db = float(
            rsrp_db[target] - rsrp_db[serving]
            - self.config.a3_hys - self.config.a3_off
        )
        self._decision_point = ShadowA3DecisionPoint(
            tic=tic,
            serving_cell=serving,
            target_cell=target,
            a3_start_tic=start_tic,
            a3_margin_db=margin_db,
            stage=stage,
            scheduled_delay_ticks=delay_ticks,
        )
        if self._early_due_tic is not None and self._early_due_tic <= tic:
            self._early_due_tic = None
            self._early_delay_ticks = None

    def _reset_shadow(self, tic: int) -> None:
        if self._shadow_ttt.pending:
            self._shadow_ttt.reset(tic)
        self._shadow_matured = None
        self._shadow_due_tic = None
        self._shadow_delay_ticks = None

    def _clear_timing_trackers(self, tic: int) -> None:
        self._early_episode = None
        self._early_due_tic = None
        self._early_delay_ticks = None
        self._reset_shadow(tic)

    def _require_early_decision(self) -> EarlyTimingDecisionPoint:
        decision = self._require_decision()
        if not isinstance(decision, EarlyTimingDecisionPoint):
            raise RuntimeError("The pending decision is not an early-window decision.")
        return decision

    def _new_early_decision(
        self,
        *,
        tic: int,
        serving: int,
        target: int,
        start_tic: int,
        delta_db: float,
        stage: str,
        delay_ticks: int,
    ) -> EarlyTimingDecisionPoint:
        return EarlyTimingDecisionPoint(
            tic=tic,
            serving_cell=serving,
            target_cell=target,
            a3_start_tic=start_tic,
            a3_margin_db=delta_db - self.enter_delta_db,
            stage=stage,
            scheduled_delay_ticks=delay_ticks,
        )
