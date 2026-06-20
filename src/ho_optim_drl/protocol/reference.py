"""Audited A3-based reference simulator."""

from typing import Any, TYPE_CHECKING

import numpy as np

from ho_optim_drl.data import RadioTrace

from .primitives import (
    GeneralCounter,
    HOEventHandler,
    RadioResourceControl,
    SyncSignal,
)

if TYPE_CHECKING:
    from ..config import Config


class A3ReferenceSimulator:
    """
    3GPP handover protocol.

    This class simulates the 3GPP handover protocol and includes:
    - Radio link monitoring (RLM)
    - Radio link failure detection and recovery (RLF)
    - A3 Event handler
    - Counter N310
    - Counter N311
    - Timer T310
    - Timer TTT
    """

    def __init__(self, protocol_config: "Config") -> None:
        # Parameters
        self.config = protocol_config
        self.config.validate()
        self.debug = False
        self.verbose = 0

        self.permit_ho_prep_abort = self.config.baseline_permit_ho_prep_abort

        # General tasks
        self.sync_source = SyncSignal(
            self.config.q_in_db,
            self.config.q_out_db,
            self.debug,
            self.verbose,
        )
        self.rrc = RadioResourceControl(
            self.config.q_in_db,
            self.config.q_out_db,
            self.debug,
            self.verbose,
        )
        self.event_handler = HOEventHandler(
            debug=self.debug,
            hys_a3=self.config.a3_hys,
            off_a3=self.config.a3_off,
            debug_lvl=self.verbose,
        )
        self.cntr = {
            "n310": GeneralCounter(
                self.config.n310,
                1,
                "N310",
                self.debug,
                self.verbose,
            ),
            "n311": GeneralCounter(
                self.config.n311,
                1,
                "N311",
                self.debug,
                self.verbose,
            ),
            "t310": GeneralCounter(
                self.config.t_t310,
                1,
                "T310",
                self.debug,
                self.verbose,
            ),
            "rlfr": GeneralCounter(
                self.config.t_rlfr,
                1,
                "RLFRecovery",
                self.debug,
                self.verbose,
            ),
            "tttc": GeneralCounter(
                self.config.a3_ttt,
                1,
                "TTT",
                self.debug,
                self.verbose,
            ),
            "ho_prep": GeneralCounter(
                self.config.t_ho_prep,
                1,
                "HOPrep",
                self.debug,
                self.verbose,
            ),
            "ho_exec": GeneralCounter(
                self.config.t_ho_exec,
                1,
                "HOExec",
                self.debug,
                self.verbose,
            ),
        }

        # Timeline
        self.bs_idxs = []
        self.ho_idxs = []
        self.rlfr_idxs = []
        self.rsrp_timeline = []
        self.sinr_timeline = []
        self.sinr_at_ho_exe_pcell = []
        self.sinr_after_ho_exe_tcell = []
        self.ho_source_cell = None
        self.successful_ho_history = []
        self.ping_pong_idxs = []

        # Benchmark
        self.max_mean_c = 0.0
        self._trace: RadioTrace | None = None

    def reset(self, tic: int) -> None:
        """Reset task."""
        self.sync_source.reset(tic)
        self.event_handler.reset(tic)
        self.rrc.reset(tic)
        for task in self.cntr.values():
            task.reset(tic, clear_history=True)
        self.bs_idxs = []
        self.ho_idxs = []
        self.rlfr_idxs = []
        self.rsrp_timeline = []
        self.sinr_timeline = []
        self.sinr_at_ho_exe_pcell = []
        self.sinr_after_ho_exe_tcell = []
        self.ho_source_cell = None
        self.successful_ho_history = []
        self.ping_pong_idxs = []
        self._trace = None

    def _step_counter_tasks(self, tic: int) -> None:
        """Step counter tasks."""
        for name, task in self.cntr.items():
            if name in {"n310", "n311"}:
                continue
            task.step(tic)

    def _radio_link_monitoring(self, sinr_db: np.ndarray, tic: int) -> None:
        """Radio link monitoring (RLM)."""
        if self.rrc.is_connected:
            indication = self.sync_source.get_indication(
                sinr_db, self.rrc.pcell, tic
            )
            if indication == "out_of_sync":
                self.cntr["n311"].reset(tic)
                if not self.cntr["t310"].pending:
                    self.cntr["n310"].start(tic)
                    self.cntr["n310"].step(tic)
            elif indication == "in_sync":
                self.cntr["n310"].reset(tic)
                if self.cntr["t310"].pending:
                    self.cntr["n311"].start(tic)
                    self.cntr["n311"].step(tic)
        else:  # Disconnected from cell, reset N310, N311, T310
            self.cntr["n310"].reset(tic)
            self.cntr["n311"].reset(tic)
            self.cntr["t310"].reset(tic)

    def _msr_event_handler(self, rsrp_db: np.ndarray, tic: int) -> None:
        """Event handler."""
        if self.rrc.is_connected:  # Connected to cell
            pcell = self.rrc.pcell
            ncell = self.rrc.ncell
            # Event A3: neighbor cell becomes offset better than SpCell
            if self.event_handler.event_a3(rsrp_db, pcell, ncell, tic):
                # Start TTT if not in progress or in HO preparation
                if not self.cntr["ho_prep"].pending and not self.cntr["tttc"].pending:
                    self.event_handler.target_cell = ncell
                    self.cntr["tttc"].start(tic)
                if ncell != self.event_handler.target_cell:  # Target cell changed
                    if self.cntr["tttc"].pending:  # Reset TTT
                        self.cntr["tttc"].reset(tic)
                    if (
                        self.permit_ho_prep_abort
                        and self.cntr["ho_prep"].pending
                    ):
                        self.cntr["ho_prep"].reset(tic)
            else:  # A3 leaving condition
                if self.cntr["tttc"].pending:  # Reset TTT
                    self.cntr["tttc"].reset(tic)
                if self.permit_ho_prep_abort and self.cntr["ho_prep"].pending:
                    self.cntr["ho_prep"].reset(tic)
        else:  # Disconnected from cell, reset TTT and HO preparation
            self.cntr["tttc"].reset(tic)
            self.cntr["ho_prep"].reset(tic)

    def _t310_state_machine(self, sinr_db: np.ndarray, tic: int) -> None:
        """Timer T310 state machine."""
        if self.cntr["n310"].reached_max:  # N310 max reached, start timer T310
            self.cntr["n310"].reset(tic)
            self.cntr["n311"].reset(tic)
            self.cntr["t310"].start(tic)
        elif self.cntr["n311"].reached_max:  # N311 max reached, reset timer T310
            self.cntr["n310"].reset(tic)
            self.cntr["n311"].reset(tic)
            self.cntr["t310"].reset(tic)
        elif self.cntr["t310"].reached_max:  # T310 expired, declare RLF
            self.cntr["n310"].reset(tic)
            self.cntr["n311"].reset(tic)
            self.cntr["t310"].reset(tic)
            self._rlf_recovery(sinr_db, tic, start=True)

    def _rlf_recovery(
        self, sinr_db: np.ndarray, tic: int, start: bool = False
    ) -> None:
        """RLF recovery."""
        if start:  # Start RLF recovery
            self.cntr["tttc"].reset(tic)
            self.cntr["ho_prep"].reset(tic)
            self.cntr["ho_exec"].reset(tic)
            self.cntr["rlfr"].start(tic)
            self.rrc.disconnect(tic)
        else:
            if self.cntr["rlfr"].reached_max:
                self.rrc.cell_search(sinr_db, tic)
                if self.rrc.cell_detected:
                    self.rrc.reconnection(tic)
                    self.cntr["rlfr"].reset(tic)

    def _ho_state_machine(
        self,
        sinr_rlm_db: np.ndarray,
        tic: int,
        sinr_rate_db: np.ndarray | None = None,
    ) -> None:
        """Handover state machine."""
        if sinr_rate_db is None:
            sinr_rate_db = sinr_rlm_db
        if self.cntr["tttc"].reached_max:
            # TTT expired, start HO preparation and reset TTT
            self.cntr["tttc"].reset(tic)
            self.cntr["ho_prep"].start(tic)
        elif self.cntr["ho_prep"].reached_max:
            target_cell = self.event_handler.target_cell
            if target_cell is None or self.rrc.pcell is None:
                self.cntr["ho_prep"].reset(tic)
                return

            # Reception of reconfigurationWithSync stops T310.
            self.cntr["ho_prep"].reset(tic)
            self.cntr["n310"].reset(tic)
            self.cntr["n311"].reset(tic)
            self.cntr["t310"].reset(tic)
            self.cntr["ho_exec"].start(tic)
            self.ho_source_cell = self.rrc.pcell
            self.sinr_at_ho_exe_pcell.append(sinr_rate_db[self.rrc.pcell])
            self.cntr["ho_prep"].set_done(tic)
            self.rrc.disconnect(tic)
        elif self.cntr["ho_exec"].reached_max:
            target_cell = self.event_handler.target_cell
            self.cntr["ho_exec"].reset(tic)
            if target_cell is None:
                self._rlf_recovery(sinr_rlm_db, tic, start=True)
                return
            self.sinr_after_ho_exe_tcell.append(sinr_rate_db[target_cell])
            if not self.sync_source.is_suitable(sinr_rlm_db, target_cell):
                self._rlf_recovery(sinr_rlm_db, tic, start=True)
            else:
                self.cntr["ho_exec"].set_done(tic)
                self.rrc.reconnection(tic, target_cell)
                self._record_successful_handover(
                    self.ho_source_cell, target_cell, tic
                )

    def _record_successful_handover(
        self, source_cell: int | None, target_cell: int, tic: int
    ) -> None:
        """Record a successful HO and detect an A-B-A return."""
        if source_cell is None:
            return
        if self.successful_ho_history:
            previous_tic, previous_source, previous_target = (
                self.successful_ho_history[-1]
            )
            is_return = (
                previous_source == target_cell
                and previous_target == source_cell
            )
            if is_return and tic - previous_tic <= self.config.t_mts:
                self.ping_pong_idxs.append(tic)
        self.successful_ho_history.append((tic, source_cell, target_cell))

    def predict(
        self,
        trace: RadioTrace,
        **config: dict[str, Any],
    ) -> float:
        """Simulate handover."""
        if trace.delta_t_ms != self.config.delta_t_ms:
            raise ValueError(
                f"Trace uses {trace.delta_t_ms} ms samples, expected "
                f"{self.config.delta_t_ms} ms."
            )
        if len(config):
            self._update_config(**config)
        else:
            self.reset(0)
        self._trace = trace

        # Max capacity (upper bound)
        max_sinr_lin = 10 ** (np.max(trace.sinr_rate_db, axis=0) / 10)
        self.max_mean_c = np.mean(np.log2(1 + max_sinr_lin))

        # Initial state
        self.rrc.cell_search(trace.sinr_rlm_db[:, 0], 0)
        self.rrc.reconnection(0)  # Init: reconnection to the best cell

        for tic in range(trace.num_samples):
            rsrp_i_db = trace.rsrp_a3_db[:, tic]
            sinr_rate_i_db = trace.sinr_rate_db[:, tic]
            sinr_rlm_i_db = trace.sinr_rlm_db[:, tic]
            if self.rrc.is_connected:
                self.rrc.rrc_measurement(rsrp_i_db, tic)

            # Timer and counter tasks
            self._step_counter_tasks(tic)

            # Monitoring
            self._radio_link_monitoring(sinr_rlm_i_db, tic)

            # State machines
            self._t310_state_machine(sinr_rlm_i_db, tic)

            # Measurement report event handler
            self._msr_event_handler(rsrp_i_db, tic)

            self._ho_state_machine(
                sinr_rlm_i_db, tic, sinr_rate_i_db
            )

            # RLF recovery
            self._rlf_recovery(sinr_rlm_i_db, tic)

            # Results
            self.ho_idxs.append(self.cntr["ho_exec"].pending)
            self.rlfr_idxs.append(self.cntr["rlfr"].pending)
            if self.rrc.is_connected:
                self.bs_idxs.append(self.rrc.pcell)
                self.rsrp_timeline.append(rsrp_i_db[self.rrc.pcell])
                self.sinr_timeline.append(sinr_rate_i_db[self.rrc.pcell])
            else:
                self.bs_idxs.append(np.nan)
                self.rsrp_timeline.append(np.nan)
                self.sinr_timeline.append(np.nan)

        return self._mean_capacity()

    def _update_config(self, **config: dict[str, Any]) -> None:
        """Update configuration."""
        for key, value in config.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"Attribute {key} not found.")
        self.reset(0)

    def _mean_capacity(self) -> float:
        """Mean Shannon spectral efficiency in bit/s/Hz."""
        sinr_t_lin = 10 ** (np.array(self.sinr_timeline) / 10)
        sinr_t_lin[np.isnan(sinr_t_lin)] = 0
        return np.mean(np.log2(1 + sinr_t_lin)).item() if len(sinr_t_lin) > 0 else 0.0

    def _event_metrics(self) -> dict[str, float | int]:
        """Return explicitly defined HO, RLF, and ping-pong metrics."""
        num_ho_attempts = len(self.cntr["ho_exec"].start_idxs)
        num_ho_success = len(self.cntr["ho_exec"].done_idxs)
        num_hof = len(self.cntr["ho_exec"].error_idxs)
        num_rlf = len(self.cntr["rlfr"].start_idxs)
        num_pp = len(self.ping_pong_idxs)
        duration_s = len(self.bs_idxs) * self.config.delta_t_ms / 1000
        return {
            "num_hof": num_hof,
            "hof_rate": num_hof / num_ho_attempts if num_ho_attempts else 0.0,
            "num_rlf": num_rlf,
            "rlf_rate_per_100s": (
                100 * num_rlf / duration_s if duration_s > 0 else 0.0
            ),
            "num_pp": num_pp,
            "pp_rate": num_pp / num_ho_success if num_ho_success else 0.0,
        }

    def get_log(self) -> dict[str, Any]:
        """Get log."""
        mean_c = self._mean_capacity()
        mean_c_rel = 100 * mean_c / self.max_mean_c if self.max_mean_c > 0 else 0
        log = {
            "max_capacity": self.max_mean_c,
            "achieved_capacity": mean_c,
            "achieved_capacity_rel": mean_c_rel,
            "max_estimated_rate_mbps": self.max_mean_c * self.config.bw / 1e6,
            "estimated_rate_mbps": mean_c * self.config.bw / 1e6,
            "num_ttt_started": len(self.cntr["tttc"].start_idxs),
            "num_ttt_finished": len(self.cntr["tttc"].terminated_idxs),
            "num_ttt_abort": len(self.cntr["tttc"].aborted_idxs),
            "num_ho_prep_started": len(self.cntr["ho_prep"].start_idxs),
            "num_ho_prep_terminated": len(self.cntr["ho_prep"].terminated_idxs),
            "num_ho_prep_completed": len(self.cntr["ho_prep"].done_idxs),
            "num_ho_prep_aborted": len(self.cntr["ho_prep"].aborted_idxs),
            "num_ho_prep_failed": len(self.cntr["ho_prep"].failed_idxs),
            "num_ho_prep_error": len(self.cntr["ho_prep"].error_idxs),
            "num_ho_exe_started": len(self.cntr["ho_exec"].start_idxs),
            "num_ho_exe_terminated": len(self.cntr["ho_exec"].terminated_idxs),
            "num_ho_exe_completed": len(self.cntr["ho_exec"].done_idxs),
            "num_ho_exe_aborted": len(self.cntr["ho_exec"].aborted_idxs),
            "num_ho_exe_failed": len(self.cntr["ho_exec"].failed_idxs),
            "num_ho_exe_error": len(self.cntr["ho_exec"].error_idxs),
        }
        log.update(self._event_metrics())
        return log

    def get_statistics(self, trace: RadioTrace | None = None) -> dict[str, Any]:
        """Get statistics."""
        trace = self._trace if trace is None else trace
        if trace is None:
            raise RuntimeError("Run predict(trace) before requesting statistics.")
        sinr_db_mat = trace.sinr_rate_db
        n_base_stations, n_samples = sinr_db_mat.shape

        # Statistics
        max_sinr_db = np.max(sinr_db_mat, axis=0)
        max_c = np.log2(1 + 10 ** (max_sinr_db / 10))
        max_c_mean = np.mean(max_c)

        sinr_t_lin = 10 ** (np.array(self.sinr_timeline) / 10)
        sinr_t_lin[np.isnan(sinr_t_lin)] = 0
        mean_c = np.mean(np.log2(1 + sinr_t_lin))

        q1_sinr, q3_sinr = np.quantile(sinr_db_mat, [0.25, 0.75])
        stats = {
            "num_base_stations": int(n_base_stations),
            "num_samples": int(n_samples),
            "mean_sinr": float(np.mean(sinr_db_mat)),
            "variance_sinr": float(np.var(sinr_db_mat)),
            "median_sinr": float(np.median(sinr_db_mat)),
            "q1_sinr": float(q1_sinr),
            "q3_sinr": float(q3_sinr),
            "max_spectral_eff": float(max_c_mean),
            "spectral_eff": float(mean_c),
            "max_estimated_rate_mbps": float(max_c_mean * self.config.bw / 1e6),
            "estimated_rate_mbps": float(mean_c * self.config.bw / 1e6),
            "total_time": int(0 + len(self.bs_idxs)),
            "total_time_s": float(
                len(self.bs_idxs) * self.config.delta_t_ms / 1000
            ),
            "connected_time": int(
                0 + len(self.bs_idxs) - len(np.where((np.isnan(self.bs_idxs)))[0])
            ),
            "num_ho_prep_started": int(0 + len(self.cntr["ho_prep"].start_idxs)),
            "num_ho_prep_terminated": int(
                0 + len(self.cntr["ho_prep"].terminated_idxs)
            ),
            "num_ho_prep_completed": int(0 + len(self.cntr["ho_prep"].done_idxs)),
            "num_ho_prep_aborted": int(0 + len(self.cntr["ho_prep"].aborted_idxs)),
            "num_ho_prep_failed": int(0 + len(self.cntr["ho_prep"].failed_idxs)),
            "num_ho_prep_error": int(0 + len(self.cntr["ho_prep"].error_idxs)),
            "num_ho_exe_started": int(0 + len(self.cntr["ho_exec"].start_idxs)),
            "num_ho_exe_terminated": int(0 + len(self.cntr["ho_exec"].terminated_idxs)),
            "num_ho_exe_completed": int(0 + len(self.cntr["ho_exec"].done_idxs)),
            "num_ho_exe_aborted": int(0 + len(self.cntr["ho_exec"].aborted_idxs)),
            "num_ho_exe_failed": int(0 + len(self.cntr["ho_exec"].failed_idxs)),
            "num_ho_exe_error": int(0 + len(self.cntr["ho_exec"].error_idxs)),
        }
        stats.update(self._event_metrics())
        return stats

    def get_stats_dict(self) -> dict[str, int | float | None]:
        """Return empty dict."""
        return {
            "num_base_stations": None,
            "num_samples": None,
            "mean_sinr": None,
            "mean_sinr_all_bs": None,
            "variance_sinr": None,
            "median_sinr": None,
            "q1_sinr": None,
            "q3_sinr": None,
            "max_spectral_eff": None,
            "spectral_eff": None,
            "max_estimated_rate_mbps": None,
            "estimated_rate_mbps": None,
            "total_time": None,
            "total_time_s": None,
            "connected_time": None,
            "num_ho_prep_started": None,
            "num_ho_prep_terminated": None,
            "num_ho_prep_completed": None,
            "num_ho_prep_aborted": None,
            "num_ho_prep_failed": None,
            "num_ho_prep_error": None,
            "num_ho_exe_started": None,
            "num_ho_exe_terminated": None,
            "num_ho_exe_completed": None,
            "num_ho_exe_aborted": None,
            "num_ho_exe_failed": None,
            "num_ho_exe_error": None,
            "num_hof": None,
            "hof_rate": None,
            "num_rlf": None,
            "rlf_rate_per_100s": None,
            "num_pp": None,
            "pp_rate": None,
        }


def print_statistics(
    simulator: A3ReferenceSimulator, trace: RadioTrace | None = None
) -> None:
    """Print statistics."""
    stats = simulator.get_statistics(trace)
    for key, value in stats.items():
        print(f"{key}: {value}")
