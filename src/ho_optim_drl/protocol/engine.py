"""Action-controlled primitive-timestep handover engine."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ho_optim_drl.data import RadioTrace

from .reference import A3ReferenceSimulator
from .types import DecisionPoint, ProtocolPhase, ProtocolSnapshot, TickResult

if TYPE_CHECKING:
    from ho_optim_drl.config import Config


class HandoverEngine(A3ReferenceSimulator):
    """
    Shared protocol engine for heuristic and learned handover policies.

    The engine consumes a ``RadioTrace``, advances one radio sample at a time,
    and pauses at a completed A3+TTT event until ``stay()`` or
    ``start_handover()`` is called.
    """

    def __init__(self, protocol_config: "Config") -> None:
        super().__init__(protocol_config)
        self._trace: RadioTrace | None = None
        self._tic = 0
        self._last_processed_tic = -1
        self._terminated = False
        self._decision_point: DecisionPoint | None = None
        self._suppressed_event: tuple[int, int] | None = None
        self._handover_started_pending = False
        self._serving_residence_start_tic = 0
        self._last_ho_tic: int | None = None
        self._previous_cell: int | None = None

    def reset(self, trace: RadioTrace) -> ProtocolSnapshot:
        """Reset the engine and camp on the strongest suitable initial cell."""
        if trace.delta_t_ms != self.config.delta_t_ms:
            raise ValueError(
                f"Trace uses {trace.delta_t_ms} ms samples, expected "
                f"{self.config.delta_t_ms} ms."
            )

        super().reset(0)
        self._trace = trace
        self._tic = 0
        self._last_processed_tic = -1
        self._terminated = False
        self._decision_point = None
        self._suppressed_event = None
        self._handover_started_pending = False
        self._serving_residence_start_tic = 0
        self._last_ho_tic = None
        self._previous_cell = None

        max_sinr_lin = 10 ** (np.max(trace.sinr_rate_db, axis=0) / 10)
        self.max_mean_c = float(np.mean(np.log2(1 + max_sinr_lin)))
        self.rrc.cell_search(trace.sinr_rlm_db[:, 0], 0)
        self.rrc.reconnection(0)
        return self.snapshot

    @property
    def decision_point(self) -> DecisionPoint | None:
        """Return the pending A3 decision."""
        return self._decision_point

    @property
    def trace(self) -> RadioTrace | None:
        """Return the immutable trace currently owned by the engine."""
        return self._trace

    @property
    def snapshot(self) -> ProtocolSnapshot:
        """Return a causal copy of the current protocol state."""
        self._require_trace()
        sample_tic = max(self._last_processed_tic, 0)
        sample_tic = min(sample_tic, self._num_samples - 1)
        rsrp_a3_db = self._readonly_copy(
            self._trace.rsrp_a3_db[:, sample_tic]
        )
        sinr_rate_db = self._readonly_copy(
            self._trace.sinr_rate_db[:, sample_tic]
        )
        sinr_rlm_db = self._readonly_copy(
            self._trace.sinr_rlm_db[:, sample_tic]
        )

        n310_progress = self._counter_progress("n310")
        t310_progress = self._counter_progress("t310")
        residence_ticks = max(
            0, sample_tic - self._serving_residence_start_tic
        )
        since_last_ho = (
            sample_tic + 1
            if self._last_ho_tic is None
            else max(0, sample_tic - self._last_ho_tic)
        )
        recent_ho_count = sum(
            sample_tic - ho_tic <= self.config.t_mts
            for ho_tic, _, _ in self.successful_ho_history
        )

        return ProtocolSnapshot(
            tic=sample_tic,
            phase=self._phase,
            serving_cell=self.rrc.pcell,
            target_cell=self._target_cell,
            decision_point=self._decision_point,
            rsrp_a3_db=rsrp_a3_db,
            sinr_rate_db=sinr_rate_db,
            sinr_rlm_db=sinr_rlm_db,
            n310_progress=float(n310_progress),
            t310_active=self.cntr["t310"].pending,
            t310_progress=float(t310_progress),
            serving_residence_ticks=residence_ticks,
            time_since_last_ho_ticks=since_last_ho,
            previous_cell=self._previous_cell,
            recent_ho_count=recent_ho_count,
        )

    def stay(self) -> None:
        """Reject the current A3 event until it leaves or the target changes."""
        decision = self._require_decision()
        self._suppressed_event = (
            decision.serving_cell,
            decision.target_cell,
        )
        self._decision_point = None
        self.event_handler.target_cell = None

    def start_handover(self) -> None:
        """Start preparation toward the target latched at the decision point."""
        decision = self._require_decision()
        if self.rrc.pcell != decision.serving_cell:
            raise RuntimeError("Serving cell changed while a decision was pending.")

        self.event_handler.target_cell = decision.target_cell
        self.cntr["ho_prep"].start(decision.tic)
        self._decision_point = None
        self._suppressed_event = None
        self._handover_started_pending = True

    def advance_one_tick(self) -> TickResult:
        """Advance one primitive timestep and emit protocol events."""
        self._require_trace()
        if self._decision_point is not None:
            raise RuntimeError(
                "A decision is pending; call stay() or start_handover() first."
            )
        if self._terminated:
            raise RuntimeError("The trace episode has already terminated.")

        tic = self._tic
        rsrp_a3_db = self._trace.rsrp_a3_db[:, tic]
        sinr_rate_db = self._trace.sinr_rate_db[:, tic]
        sinr_rlm_db = self._trace.sinr_rlm_db[:, tic]

        before_ho_done = len(self.cntr["ho_exec"].done_idxs)
        before_ho_error = len(self.cntr["ho_exec"].error_idxs)
        before_rlf = len(self.cntr["rlfr"].start_idxs)
        before_pp = len(self.ping_pong_idxs)
        before_success = len(self.successful_ho_history)

        if self.rrc.is_connected:
            self.rrc.rrc_measurement(rsrp_a3_db, tic)

        self._step_counter_tasks(tic)
        self._radio_link_monitoring(sinr_rlm_db, tic)
        self._t310_state_machine(sinr_rlm_db, tic)
        self._measurement_event_handler(rsrp_a3_db, tic)
        self._action_controlled_ho_state_machine(
            sinr_rlm_db, sinr_rate_db, tic
        )
        self._rlf_recovery(sinr_rlm_db, tic)

        if len(self.successful_ho_history) > before_success:
            _, source_cell, _ = self.successful_ho_history[-1]
            self._previous_cell = source_cell
            self._last_ho_tic = tic
            self._serving_residence_start_tic = tic
        elif (
            self.rrc.is_connected
            and self._last_processed_tic >= 0
            and np.isnan(self.bs_idxs[-1])
        ):
            self._serving_residence_start_tic = tic

        self._append_timeline(rsrp_a3_db, sinr_rate_db)
        self._last_processed_tic = tic
        self._tic += 1
        self._terminated = self._tic >= self._num_samples

        handover_started = self._handover_started_pending
        self._handover_started_pending = False
        handover_succeeded = len(self.cntr["ho_exec"].done_idxs) > before_ho_done
        handover_failed = len(self.cntr["ho_exec"].error_idxs) > before_ho_error
        rlf_started = len(self.cntr["rlfr"].start_idxs) > before_rlf
        ping_pong = len(self.ping_pong_idxs) > before_pp

        return TickResult(
            snapshot=self.snapshot,
            delivered_rate_mbps=self._delivered_rate_mbps(sinr_rate_db),
            best_rate_mbps=self._best_rate_mbps(sinr_rate_db),
            handover_started=handover_started,
            handover_succeeded=handover_succeeded,
            handover_failed=handover_failed,
            rlf_started=rlf_started,
            ping_pong_detected=ping_pong,
            terminated=self._terminated,
        )

    def predict(self, *args: object, **kwargs: object) -> float:
        """Prevent accidental use of the reference simulator's auto-HO API."""
        del args, kwargs
        raise RuntimeError(
            "HandoverEngine is action-controlled. Use reset(), "
            "advance_one_tick(), stay(), and start_handover()."
        )

    def _measurement_event_handler(
        self, rsrp_db: np.ndarray, tic: int
    ) -> None:
        """Run A3/TTT while honoring a rejected-event latch."""
        if self._suppressed_event is None:
            self._msr_event_handler(rsrp_db, tic)
            return

        suppressed_serving, suppressed_target = self._suppressed_event
        pcell = self.rrc.pcell
        ncell = self.rrc.ncell
        if pcell != suppressed_serving:
            self._clear_suppression()
            self._msr_event_handler(rsrp_db, tic)
            return

        if ncell != suppressed_target:
            self._clear_suppression(reset_a3=True)
            self._msr_event_handler(rsrp_db, tic)
            return

        a3_entered = self.event_handler.event_a3(
            rsrp_db, pcell, ncell, tic
        )
        if not a3_entered:
            self._clear_suppression()

    def _action_controlled_ho_state_machine(
        self,
        sinr_rlm_db: np.ndarray,
        sinr_rate_db: np.ndarray,
        tic: int,
    ) -> None:
        """Pause at TTT completion; otherwise reuse the reference HO machine."""
        if self.cntr["tttc"].reached_max:
            target_cell = self.event_handler.target_cell
            serving_cell = self.rrc.pcell
            a3_start_tic = self.cntr["tttc"].start_idxs[-1]
            self.cntr["tttc"].reset(tic)

            if target_cell is None or serving_cell is None:
                return
            margin_db = (
                self._trace.rsrp_a3_db[target_cell, tic]
                - self._trace.rsrp_a3_db[serving_cell, tic]
                - self.config.a3_hys
                - self.config.a3_off
            )
            self._decision_point = DecisionPoint(
                tic=tic,
                serving_cell=serving_cell,
                target_cell=target_cell,
                a3_start_tic=a3_start_tic,
                a3_margin_db=float(margin_db),
            )
            return

        super()._ho_state_machine(sinr_rlm_db, tic, sinr_rate_db)

    def _append_timeline(
        self, rsrp_db: np.ndarray, sinr_db: np.ndarray
    ) -> None:
        self.ho_idxs.append(self.cntr["ho_exec"].pending)
        self.rlfr_idxs.append(self.cntr["rlfr"].pending)
        if self.rrc.is_connected:
            self.bs_idxs.append(self.rrc.pcell)
            self.rsrp_timeline.append(rsrp_db[self.rrc.pcell])
            self.sinr_timeline.append(sinr_db[self.rrc.pcell])
        else:
            self.bs_idxs.append(np.nan)
            self.rsrp_timeline.append(np.nan)
            self.sinr_timeline.append(np.nan)

    def _clear_suppression(self, reset_a3: bool = False) -> None:
        self._suppressed_event = None
        self.event_handler.target_cell = None
        if reset_a3:
            self.event_handler.conditions["a3"] = False

    def _delivered_rate_mbps(self, sinr_db: np.ndarray) -> float:
        if not self.rrc.is_connected:
            return 0.0
        sinr_linear = 10 ** (sinr_db[self.rrc.pcell] / 10)
        return float(self.config.bw * np.log2(1 + sinr_linear) / 1e6)

    def _best_rate_mbps(self, sinr_db: np.ndarray) -> float:
        sinr_linear = 10 ** (np.max(sinr_db) / 10)
        return float(self.config.bw * np.log2(1 + sinr_linear) / 1e6)

    @property
    def _phase(self) -> ProtocolPhase:
        if self._terminated:
            return ProtocolPhase.TERMINATED
        if self._decision_point is not None:
            return ProtocolPhase.DECISION
        if self.cntr["rlfr"].pending:
            return ProtocolPhase.RLF_RECOVERY
        if self.cntr["ho_exec"].pending:
            return ProtocolPhase.HO_EXECUTION
        if self.cntr["ho_prep"].pending:
            return ProtocolPhase.HO_PREPARATION
        if self.cntr["tttc"].pending:
            return ProtocolPhase.A3_TTT
        if self.rrc.is_connected:
            return ProtocolPhase.CONNECTED
        return ProtocolPhase.DISCONNECTED

    @property
    def _target_cell(self) -> int | None:
        if self._decision_point is not None:
            return self._decision_point.target_cell
        return self.event_handler.target_cell

    @property
    def _num_samples(self) -> int:
        self._require_trace()
        return self._trace.num_samples

    def _require_trace(self) -> None:
        if self._trace is None:
            raise RuntimeError("Call reset(trace) before using the engine.")

    def _require_decision(self) -> DecisionPoint:
        if self._decision_point is None:
            raise RuntimeError("No A3 decision is currently pending.")
        return self._decision_point

    def _counter_progress(self, name: str) -> float:
        counter = self.cntr[name]
        if counter.max_val == 0:
            return float(counter.reached_max)
        return float(counter.cnt / counter.max_val)

    @staticmethod
    def _readonly_copy(values: np.ndarray) -> np.ndarray:
        copied = np.array(values, dtype=np.float64, copy=True)
        copied.setflags(write=False)
        return copied
