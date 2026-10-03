"""Topology-aware configurable candidate extension of the A3 observation."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from ho_optim_drl.data import CellTopology, RadioTrace
from ho_optim_drl.protocol.types import ProtocolSnapshot, TickResult

from .observation import A3ObservationBuilder, A3ObservationConfig


_MAX_CANDIDATES = 10
_SUPPORTED_CANDIDATE_COUNTS = (3, 5, 10)
_SUPPORTED_CANDIDATE_ORDERS = ("legacy", "ranked_v1")
_SUPPORTED_OBSERVATION_SCHEMAS = (
    "legacy_v1",
    "rolling_v1",
    "monotonic_v1",
    "qos_full_v1",
)
_ADDITIONAL_FEATURES = (
    "spectral_efficiency",
    "delta_spectral_efficiency",
    "a3_margin_now",
    "a3_margin_mean",
    "a3_margin_min",
    "a3_margin_slope",
    "a3_margin_std",
    "sinr_slope",
    "qin_gap",
    "is_previous_cell",
)
_FULL_GLOBAL_FEATURES = (
    "serving_spectral_efficiency",
    "serving_qout_gap",
    "serving_sinr_slope_100ms",
    "serving_sinr_slope_500ms",
    "serving_sinr_std_500ms",
    "serving_qout_fraction_500ms",
    "n310_progress",
    "n311_progress",
    "t310_active",
    "t310_remaining",
    "serving_residence_time",
    "time_since_last_ho",
    "ping_pong_window_remaining",
    "successful_ho_count_1s",
    "best_measured_spectral_efficiency",
    "history_coverage_500ms",
    "speed_magnitude_kmh",
)
_FULL_CANDIDATE_FEATURES = (
    "valid",
    "spectral_efficiency",
    "delta_spectral_efficiency",
    "qin_gap",
    "sinr_slope_100ms",
    "sinr_slope_500ms",
    "sinr_std_500ms",
    "qin_suitable_fraction_500ms",
    "sinr_p10_qin_gap_500ms",
    "rsrp_gap_now",
    "rsrp_gap_slope_500ms",
    "projected_qin_margin_90ms",
    "is_previous_cell",
    "is_same_site_as_serving",
    "is_a3_target",
)


@dataclass(frozen=True)
class _AllCellRecord:
    tic: int
    serving_cell: int | None
    rsrp_a3_db: np.ndarray
    sinr_rate_db: np.ndarray
    sinr_rlm_db: np.ndarray


class TopologyCandidateObservationBuilder(A3ObservationBuilder):
    """Keep the original C1 features and append causal candidate blocks.

    The legacy pool is deterministic: the A3 target, other sectors at the
    serving site, all sectors from the two strongest first-ring sites, then the
    strongest remaining measured cells. ``ranked_v1`` keeps the exact same
    ten-cell pool but places the best-current-SINR alternative second and a
    distinct best projected-SINR (+90 ms, causal linear trend) alternative
    third. K=3 and K=5 are prefixes of that frozen ten-cell list.
    """

    MAX_CANDIDATES = _MAX_CANDIDATES
    ADDITIONAL_FEATURES = _ADDITIONAL_FEATURES
    FEATURE_NAMES = A3ObservationBuilder.FEATURE_NAMES + tuple(
        f"c{slot}_{feature}"
        for slot in range(2, _MAX_CANDIDATES + 1)
        for feature in _ADDITIONAL_FEATURES
    )
    _ADDITIONAL_LOW = np.array(
        [0.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.0, -1.0, -1.0, 0.0],
        dtype=np.float32,
    )
    LOW = np.concatenate(
        (
            A3ObservationBuilder.LOW,
            np.tile(_ADDITIONAL_LOW, _MAX_CANDIDATES - 1),
        )
    ).astype(np.float32)
    HIGH = np.ones(len(FEATURE_NAMES), dtype=np.float32)
    CANDIDATE_POLICY = "a3_same_site_two_neighbor_sites_fill_rsrp_v1"
    CANDIDATE_ORDERS = _SUPPORTED_CANDIDATE_ORDERS
    OBSERVATION_SCHEMAS = _SUPPORTED_OBSERVATION_SCHEMAS

    def __init__(
        self,
        protocol_config,
        observation_config: A3ObservationConfig | None = None,
        *,
        topology: CellTopology | None = None,
        candidate_count: int = _MAX_CANDIDATES,
        candidate_order: str = "legacy",
        observation_schema: str = "legacy_v1",
    ) -> None:
        super().__init__(protocol_config, observation_config)
        if candidate_count not in _SUPPORTED_CANDIDATE_COUNTS:
            raise ValueError(
                "candidate_count must be one of "
                f"{_SUPPORTED_CANDIDATE_COUNTS}."
            )
        if candidate_order not in _SUPPORTED_CANDIDATE_ORDERS:
            raise ValueError(
                "candidate_order must be one of "
                f"{_SUPPORTED_CANDIDATE_ORDERS}."
            )
        if observation_schema not in _SUPPORTED_OBSERVATION_SCHEMAS:
            raise ValueError(
                "observation_schema must be one of "
                f"{_SUPPORTED_OBSERVATION_SCHEMAS}."
            )
        self.topology = topology
        self.candidate_count = int(candidate_count)
        self.candidate_order = candidate_order
        self.observation_schema = observation_schema
        if observation_schema == "qos_full_v1":
            self._feature_names = self._full_feature_names()
            self._low = self._full_low()
        else:
            self._feature_names = A3ObservationBuilder.FEATURE_NAMES + tuple(
                f"c{slot}_{feature}"
                for slot in range(2, self.candidate_count + 1)
                for feature in self.ADDITIONAL_FEATURES
            )
            self._low = np.concatenate(
                (
                    A3ObservationBuilder.LOW,
                    np.tile(
                        self._ADDITIONAL_LOW,
                        self.candidate_count - 1,
                    ),
                )
            ).astype(np.float32)
        self._high = np.ones(len(self._feature_names), dtype=np.float32)
        self._all_cell_history: deque[_AllCellRecord] = deque(
            maxlen=self.config.history_window_ticks
        )
        self.candidate_pool_cells: tuple[int | None, ...] = (
            (None,) * self.MAX_CANDIDATES
        )
        self.legacy_candidate_pool_cells: tuple[int | None, ...] = (
            (None,) * self.MAX_CANDIDATES
        )
        self.candidate_cells: tuple[int | None, ...] = (
            (None,) * self.candidate_count
        )
        self._candidate_tic: int | None = None
        self._cached_observation: np.ndarray | None = None
        self._speed_kmh: float | None = None

    def reset(self) -> None:
        super().reset()
        self._all_cell_history.clear()
        self.candidate_pool_cells = (None,) * self.MAX_CANDIDATES
        self.legacy_candidate_pool_cells = (None,) * self.MAX_CANDIDATES
        self.candidate_cells = (None,) * self.candidate_count
        self._candidate_tic = None
        self._cached_observation = None
        self._speed_kmh = None

    def start_episode(self, trace: RadioTrace) -> None:
        self._speed_kmh = trace.speed_kmh
        if self.observation_schema == "qos_full_v1" and self._speed_kmh is None:
            raise ValueError(
                "qos_full_v1 requires speed_kmh in RadioTrace metadata."
            )

    @property
    def size(self) -> int:
        return len(self._feature_names)

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self._feature_names

    @property
    def low(self) -> np.ndarray:
        return self._low.copy()

    @property
    def high(self) -> np.ndarray:
        return self._high.copy()

    @property
    def candidate_order_spec(self) -> str:
        if self.candidate_order == "legacy":
            return "legacy_topology_pool_order_v1"
        return "c1_current_sinr_projected_sinr_90ms_then_legacy_v1"

    @property
    def observation_schema_spec(self) -> str:
        return self.observation_schema

    def observe(self, tick: TickResult) -> None:
        super().observe(tick)
        snapshot = tick.snapshot
        self._all_cell_history.append(
            _AllCellRecord(
                tic=snapshot.tic,
                serving_cell=snapshot.serving_cell,
                rsrp_a3_db=np.array(snapshot.rsrp_a3_db, copy=True),
                sinr_rate_db=np.array(snapshot.sinr_rate_db, copy=True),
                sinr_rlm_db=np.array(snapshot.sinr_rlm_db, copy=True),
            )
        )

    def build(self, snapshot: ProtocolSnapshot) -> np.ndarray:
        if self._candidate_tic == snapshot.tic:
            if self._cached_observation is None:
                raise RuntimeError("Candidate observation cache is incomplete.")
            return self._cached_observation.copy()

        decision = snapshot.decision_point
        if decision is None:
            raise RuntimeError("A candidate observation requires a decision.")
        if not self._all_cell_history or self._all_cell_history[-1].tic != snapshot.tic:
            raise RuntimeError("All-cell history is missing the decision tick.")

        num_cells = len(snapshot.rsrp_a3_db)
        topology = self.topology or CellTopology.contiguous_sector_groups(num_cells)
        if topology.num_cells != num_cells:
            raise ValueError(
                f"Topology declares {topology.num_cells} cells but trace has "
                f"{num_cells}."
            )
        legacy_pool = self._select_candidate_pool(snapshot, topology)
        self.legacy_candidate_pool_cells = legacy_pool
        self.candidate_pool_cells = self._order_candidate_pool(
            snapshot,
            legacy_pool,
        )
        self.candidate_cells = self.candidate_pool_cells[
            : self.candidate_count
        ]

        if self.observation_schema == "legacy_v1":
            base = super().build(snapshot)
            history = [
                record
                for record in self._all_cell_history
                if decision.a3_start_tic <= record.tic <= decision.tic
                and record.serving_cell == decision.serving_cell
            ]
            if not history:
                raise RuntimeError("No causal all-cell history is available.")
            values = list(base)
            for cell in self.candidate_cells[1:]:
                values.extend(
                    self._candidate_features(snapshot, history, cell)
                )
        elif self.observation_schema in {"rolling_v1", "monotonic_v1"}:
            history = self._window_records(
                snapshot.tic, self.config.long_window_ticks
            )
            values = list(self._rolling_base_features(snapshot, history))
            for cell in self.candidate_cells[1:]:
                values.extend(
                    self._candidate_features(snapshot, history, cell)
                )
        else:
            history = self._window_records(
                snapshot.tic, self.config.long_window_ticks
            )
            values = list(self._full_global_features(snapshot, history))
            for cell in self.candidate_cells:
                values.extend(
                    self._full_candidate_features(snapshot, history, cell)
                )
        observation = np.asarray(values, dtype=np.float32)
        if observation.shape != (self.size,):
            raise RuntimeError(
                f"Expected {self.size} features, got "
                f"{observation.shape}."
            )
        if not np.all(np.isfinite(observation)):
            raise RuntimeError("Candidate observation contains a non-finite value.")
        observation = np.clip(observation, self._low, self._high)
        self._candidate_tic = snapshot.tic
        self._cached_observation = observation.copy()
        return observation

    def action_mask(self) -> np.ndarray:
        """Return STAY plus one validity bit for each frozen candidate."""
        return np.asarray(
            [True, *(cell is not None for cell in self.candidate_cells)],
            dtype=np.bool_,
        )

    def prediction_diagnostics(
        self, snapshot: ProtocolSnapshot
    ) -> list[dict[str, float | int]]:
        """Expose causal +90 ms predictor inputs for train/validation audit."""
        if snapshot.decision_point is None:
            raise RuntimeError("Prediction diagnostics require a decision.")
        if self._candidate_tic != snapshot.tic:
            self.build(snapshot)
        records = self._window_records(
            snapshot.tic, self.config.long_window_ticks
        )
        short = self._short_records(records)
        tics = np.asarray(
            [record.tic for record in short], dtype=np.float64
        )
        rows: list[dict[str, float | int]] = []
        for slot, cell in enumerate(self.candidate_cells, start=1):
            if cell is None:
                continue
            values = np.asarray(
                [record.sinr_rlm_db[cell] for record in short],
                dtype=np.float64,
            )
            slope = self._slope_db_per_s(tics, values)
            mean = float(np.mean(values))
            std = float(np.std(values))
            predicted = (
                mean
                + self.config.projection_horizon_seconds * slope
                - self.config.projection_uncertainty_kappa * std
            )
            rows.append(
                {
                    "slot": slot,
                    "cell": int(cell),
                    "mean_sinr_100ms_db": mean,
                    "slope_sinr_100ms_db_per_s": slope,
                    "std_sinr_100ms_db": std,
                    "predicted_sinr_90ms_db": predicted,
                    "predicted_qin_margin_90ms_db": (
                        predicted - self.protocol_config.q_in_db
                    ),
                }
            )
        return rows

    def _full_feature_names(self) -> tuple[str, ...]:
        return _FULL_GLOBAL_FEATURES + tuple(
            f"c{slot}_{feature}"
            for slot in range(1, self.candidate_count + 1)
            for feature in _FULL_CANDIDATE_FEATURES
        )

    def _full_low(self) -> np.ndarray:
        global_low = np.asarray(
            [
                0.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
            dtype=np.float32,
        )
        candidate_low = np.asarray(
            [
                0.0, 0.0, -1.0, -1.0, -1.0, -1.0, 0.0, 0.0,
                -1.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0,
            ],
            dtype=np.float32,
        )
        return np.concatenate(
            (global_low, np.tile(candidate_low, self.candidate_count))
        ).astype(np.float32)

    def _window_records(
        self, decision_tic: int, lookback_ticks: int
    ) -> list[_AllCellRecord]:
        first_tic = decision_tic - lookback_ticks
        records = [
            record
            for record in self._all_cell_history
            if first_tic <= record.tic <= decision_tic
        ]
        if not records or records[-1].tic != decision_tic:
            raise RuntimeError("No causal rolling history is available.")
        return records

    def _short_records(
        self, records: list[_AllCellRecord]
    ) -> list[_AllCellRecord]:
        first_tic = records[-1].tic - self.config.short_window_ticks
        return [record for record in records if record.tic >= first_tic]

    def _uses_monotonic_normalization(self) -> bool:
        return self.observation_schema in {"monotonic_v1", "qos_full_v1"}

    def _se_feature(self, sinr_db: float) -> float:
        if not self._uses_monotonic_normalization():
            return self._normalized_spectral_efficiency(sinr_db)
        efficiency = self._spectral_efficiency(sinr_db)
        return self._unsigned_saturating(
            efficiency, self.config.spectral_efficiency_scale
        )

    def _delta_se_feature(
        self, candidate_sinr_db: float, serving_sinr_db: float
    ) -> float:
        if not self._uses_monotonic_normalization():
            return float(
                np.clip(
                    self._normalized_spectral_efficiency(candidate_sinr_db)
                    - self._normalized_spectral_efficiency(serving_sinr_db),
                    -1.0,
                    1.0,
                )
            )
        delta = self._spectral_efficiency(candidate_sinr_db) - (
            self._spectral_efficiency(serving_sinr_db)
        )
        return self._signed_saturating(
            delta, self.config.spectral_efficiency_scale
        )

    def _signed_feature(self, value: float, scale: float) -> float:
        if self._uses_monotonic_normalization():
            return self._signed_saturating(value, scale)
        return self._signed_normalize(value, scale)

    def _unsigned_feature(self, value: float, scale: float) -> float:
        if self._uses_monotonic_normalization():
            return self._unsigned_saturating(value, scale)
        return self._unsigned_normalize(value, scale)

    def _rolling_base_features(
        self,
        snapshot: ProtocolSnapshot,
        records: list[_AllCellRecord],
    ) -> tuple[float, ...]:
        decision = snapshot.decision_point
        if decision is None:
            raise RuntimeError("Rolling features require an A3 decision.")
        serving = decision.serving_cell
        target = decision.target_cell
        tics = np.asarray([record.tic for record in records], dtype=np.float64)
        margins = np.asarray(
            [
                record.rsrp_a3_db[target]
                - record.rsrp_a3_db[serving]
                - self.protocol_config.a3_hys
                - self.protocol_config.a3_off
                for record in records
            ],
            dtype=np.float64,
        )
        short = self._short_records(records)
        short_tics = np.asarray(
            [record.tic for record in short], dtype=np.float64
        )
        serving_short = np.asarray(
            [record.sinr_rate_db[serving] for record in short],
            dtype=np.float64,
        )
        target_short = np.asarray(
            [record.sinr_rate_db[target] for record in short],
            dtype=np.float64,
        )
        serving_sinr = float(snapshot.sinr_rate_db[serving])
        target_sinr = float(snapshot.sinr_rate_db[target])
        time_since_ho = (
            self.config.time_since_ho_clip_s
            if snapshot.previous_cell is None
            else snapshot.time_since_last_ho_ticks
            * self.protocol_config.delta_t_ms
            / 1000.0
        )
        residence_s = (
            snapshot.serving_residence_ticks
            * self.protocol_config.delta_t_ms
            / 1000.0
        )
        return (
            self._se_feature(serving_sinr),
            self._se_feature(target_sinr),
            self._delta_se_feature(target_sinr, serving_sinr),
            self._signed_feature(margins[-1], self.config.margin_clip_db),
            self._signed_feature(
                float(np.mean(margins)), self.config.margin_clip_db
            ),
            self._signed_feature(
                float(np.min(margins)), self.config.margin_clip_db
            ),
            self._signed_feature(
                self._slope_db_per_s(tics, margins),
                self.config.slope_clip_db_per_s,
            ),
            self._unsigned_feature(
                float(np.std(margins)), self.config.margin_std_clip_db
            ),
            self._signed_feature(
                self._slope_db_per_s(short_tics, serving_short),
                self.config.slope_clip_db_per_s,
            ),
            self._signed_feature(
                self._slope_db_per_s(short_tics, target_short),
                self.config.slope_clip_db_per_s,
            ),
            self._signed_feature(
                float(snapshot.sinr_rlm_db[serving])
                - self.protocol_config.q_out_db,
                self.config.quality_gap_clip_db,
            ),
            self._signed_feature(
                float(snapshot.sinr_rlm_db[target])
                - self.protocol_config.q_in_db,
                self.config.quality_gap_clip_db,
            ),
            float(np.clip(snapshot.n310_progress, 0.0, 1.0)),
            float(snapshot.t310_active),
            float(np.clip(snapshot.t310_progress, 0.0, 1.0)),
            self._unsigned_feature(
                residence_s, self.config.residence_time_clip_s
            ),
            self._unsigned_feature(
                time_since_ho, self.config.time_since_ho_clip_s
            ),
            float(snapshot.previous_cell == target),
            self._unsigned_feature(
                snapshot.recent_ho_count,
                self.config.recent_ho_count_clip,
            ),
        )

    def _full_global_features(
        self,
        snapshot: ProtocolSnapshot,
        records: list[_AllCellRecord],
    ) -> tuple[float, ...]:
        decision = snapshot.decision_point
        if decision is None:
            raise RuntimeError("Full observation requires an A3 decision.")
        serving = decision.serving_cell
        short = self._short_records(records)
        long_tics = np.asarray(
            [record.tic for record in records], dtype=np.float64
        )
        short_tics = np.asarray(
            [record.tic for record in short], dtype=np.float64
        )
        serving_long = np.asarray(
            [record.sinr_rate_db[serving] for record in records],
            dtype=np.float64,
        )
        serving_short = np.asarray(
            [record.sinr_rate_db[serving] for record in short],
            dtype=np.float64,
        )
        serving_rlm = np.asarray(
            [record.sinr_rlm_db[serving] for record in records],
            dtype=np.float64,
        )
        residence_s = (
            snapshot.serving_residence_ticks
            * self.protocol_config.delta_t_ms
            / 1000.0
        )
        no_previous = snapshot.previous_cell is None
        since_ho_s = (
            self.config.time_since_ho_clip_s
            if no_previous
            else snapshot.time_since_last_ho_ticks
            * self.protocol_config.delta_t_ms
            / 1000.0
        )
        pp_remaining = (
            0.0
            if no_previous or self.protocol_config.t_mts <= 0
            else max(
                0.0,
                1.0
                - snapshot.time_since_last_ho_ticks
                / self.protocol_config.t_mts,
            )
        )
        t310_remaining = (
            max(0.0, 1.0 - snapshot.t310_progress)
            if snapshot.t310_active
            else 0.0
        )
        expected_samples = self.config.long_window_ticks + 1
        coverage = min(1.0, len(records) / expected_samples)
        best_sinr = float(np.max(snapshot.sinr_rate_db))
        return (
            self._se_feature(float(snapshot.sinr_rate_db[serving])),
            self._signed_feature(
                float(snapshot.sinr_rlm_db[serving])
                - self.protocol_config.q_out_db,
                self.config.quality_gap_clip_db,
            ),
            self._signed_feature(
                self._slope_db_per_s(short_tics, serving_short),
                self.config.slope_clip_db_per_s,
            ),
            self._signed_feature(
                self._slope_db_per_s(long_tics, serving_long),
                self.config.slope_clip_db_per_s,
            ),
            self._unsigned_feature(
                float(np.std(serving_long)), self.config.margin_std_clip_db
            ),
            float(np.mean(serving_rlm < self.protocol_config.q_out_db)),
            float(np.clip(snapshot.n310_progress, 0.0, 1.0)),
            float(np.clip(snapshot.n311_progress, 0.0, 1.0)),
            float(snapshot.t310_active),
            float(t310_remaining),
            self._unsigned_feature(
                residence_s, self.config.residence_time_clip_s
            ),
            self._unsigned_feature(
                since_ho_s, self.config.time_since_ho_clip_s
            ),
            float(pp_remaining),
            self._unsigned_feature(
                snapshot.recent_ho_count,
                self.config.recent_ho_count_clip,
            ),
            self._se_feature(best_sinr),
            float(coverage),
            self._unsigned_saturating(
                float(self._speed_kmh), self.config.speed_scale_kmh
            ),
        )

    def _full_candidate_features(
        self,
        snapshot: ProtocolSnapshot,
        records: list[_AllCellRecord],
        cell: int | None,
    ) -> tuple[float, ...]:
        if cell is None:
            return (0.0,) * len(_FULL_CANDIDATE_FEATURES)
        decision = snapshot.decision_point
        if decision is None:
            raise RuntimeError("Candidate features require an A3 decision.")
        serving = decision.serving_cell
        short = self._short_records(records)
        long_tics = np.asarray(
            [record.tic for record in records], dtype=np.float64
        )
        short_tics = np.asarray(
            [record.tic for record in short], dtype=np.float64
        )
        candidate_long = np.asarray(
            [record.sinr_rate_db[cell] for record in records],
            dtype=np.float64,
        )
        candidate_short = np.asarray(
            [record.sinr_rate_db[cell] for record in short],
            dtype=np.float64,
        )
        candidate_rlm_long = np.asarray(
            [record.sinr_rlm_db[cell] for record in records],
            dtype=np.float64,
        )
        candidate_rlm_short = np.asarray(
            [record.sinr_rlm_db[cell] for record in short],
            dtype=np.float64,
        )
        rsrp_gap = np.asarray(
            [
                record.rsrp_a3_db[cell] - record.rsrp_a3_db[serving]
                for record in records
            ],
            dtype=np.float64,
        )
        short_slope = self._slope_db_per_s(
            short_tics, candidate_rlm_short
        )
        projected_margin = (
            float(np.mean(candidate_rlm_short))
            + self.config.projection_horizon_seconds * short_slope
            - self.config.projection_uncertainty_kappa
            * float(np.std(candidate_rlm_short))
            - self.protocol_config.q_in_db
        )
        topology = self.topology or CellTopology.contiguous_sector_groups(
            len(snapshot.rsrp_a3_db)
        )
        same_site = topology.site_id_for_cell(cell) == (
            topology.site_id_for_cell(serving)
        )
        candidate_sinr = float(snapshot.sinr_rate_db[cell])
        serving_sinr = float(snapshot.sinr_rate_db[serving])
        return (
            1.0,
            self._se_feature(candidate_sinr),
            self._delta_se_feature(candidate_sinr, serving_sinr),
            self._signed_feature(
                float(snapshot.sinr_rlm_db[cell])
                - self.protocol_config.q_in_db,
                self.config.quality_gap_clip_db,
            ),
            self._signed_feature(
                self._slope_db_per_s(short_tics, candidate_short),
                self.config.slope_clip_db_per_s,
            ),
            self._signed_feature(
                self._slope_db_per_s(long_tics, candidate_long),
                self.config.slope_clip_db_per_s,
            ),
            self._unsigned_feature(
                float(np.std(candidate_long)),
                self.config.margin_std_clip_db,
            ),
            float(np.mean(candidate_rlm_long > self.protocol_config.q_in_db)),
            self._signed_feature(
                float(np.percentile(candidate_rlm_long, 10))
                - self.protocol_config.q_in_db,
                self.config.quality_gap_clip_db,
            ),
            self._signed_feature(
                rsrp_gap[-1], self.config.margin_clip_db
            ),
            self._signed_feature(
                self._slope_db_per_s(long_tics, rsrp_gap),
                self.config.slope_clip_db_per_s,
            ),
            self._signed_feature(
                projected_margin, self.config.quality_gap_clip_db
            ),
            float(snapshot.previous_cell == cell),
            float(same_site),
            float(cell == decision.target_cell),
        )

    def _select_candidate_pool(
        self,
        snapshot: ProtocolSnapshot,
        topology: CellTopology,
    ) -> tuple[int | None, ...]:
        decision = snapshot.decision_point
        if decision is None:
            raise RuntimeError("Candidate selection requires an A3 decision.")
        serving = decision.serving_cell
        trigger = decision.target_cell
        chosen: list[int] = []

        def add(cell: int) -> None:
            cell = int(cell)
            if (
                cell != serving
                and 0 <= cell < topology.num_cells
                and cell not in chosen
                and len(chosen) < self.MAX_CANDIDATES
            ):
                chosen.append(cell)

        add(trigger)
        serving_site = topology.site_id_for_cell(serving)
        for cell in topology.cells_for_site(serving_site):
            add(cell)

        neighbor_sites = topology.neighboring_site_ids(serving_site)
        ranked_sites = sorted(
            neighbor_sites,
            key=lambda site_id: (
                -max(
                    float(snapshot.rsrp_a3_db[cell])
                    for cell in topology.cells_for_site(site_id)
                ),
                site_id,
            ),
        )
        for site_id in ranked_sites[:2]:
            for cell in topology.cells_for_site(site_id):
                add(cell)

        remaining = sorted(
            (
                cell
                for cell in range(topology.num_cells)
                if cell != serving and cell not in chosen
            ),
            key=lambda cell: (-float(snapshot.rsrp_a3_db[cell]), cell),
        )
        for cell in remaining:
            add(cell)
        padded: list[int | None] = chosen[: self.MAX_CANDIDATES]
        padded.extend([None] * (self.MAX_CANDIDATES - len(padded)))
        return tuple(padded)

    def _order_candidate_pool(
        self,
        snapshot: ProtocolSnapshot,
        legacy_pool: tuple[int | None, ...],
    ) -> tuple[int | None, ...]:
        if self.candidate_order == "legacy":
            return legacy_pool
        if not legacy_pool or legacy_pool[0] is None:
            raise RuntimeError("The candidate pool has no A3 target.")

        c1 = int(legacy_pool[0])
        alternatives = [
            int(cell) for cell in legacy_pool[1:] if cell is not None
        ]
        current_rank = sorted(
            alternatives,
            key=lambda cell: (
                -float(snapshot.sinr_rate_db[cell]),
                cell,
            ),
        )
        projected_rank = sorted(
            alternatives,
            key=lambda cell: (
                -self._projected_sinr_db(cell, horizon_seconds=0.09),
                cell,
            ),
        )
        ordered: list[int] = [c1]
        if current_rank:
            ordered.append(current_rank[0])
        projected_alternative = next(
            (cell for cell in projected_rank if cell not in ordered),
            None,
        )
        if projected_alternative is not None:
            ordered.append(projected_alternative)
        ordered.extend(
            int(cell)
            for cell in legacy_pool[1:]
            if cell is not None and int(cell) not in ordered
        )
        padded: list[int | None] = ordered[: self.MAX_CANDIDATES]
        padded.extend([None] * (self.MAX_CANDIDATES - len(padded)))
        return tuple(padded)

    def _projected_sinr_db(
        self,
        cell: int,
        *,
        horizon_seconds: float,
    ) -> float:
        records = list(self._all_cell_history)[
            -self.config.trend_window_ticks :
        ]
        current = float(records[-1].sinr_rate_db[cell])
        if len(records) < 2:
            return current
        tics = np.asarray([record.tic for record in records], dtype=np.float64)
        values = np.asarray(
            [record.sinr_rate_db[cell] for record in records],
            dtype=np.float64,
        )
        slope = self._slope_db_per_s(tics, values)
        return current + slope * horizon_seconds

    def _candidate_features(
        self,
        snapshot: ProtocolSnapshot,
        records: list[_AllCellRecord],
        cell: int | None,
    ) -> tuple[float, ...]:
        if cell is None:
            return (0.0,) * len(self.ADDITIONAL_FEATURES)
        decision = snapshot.decision_point
        if decision is None:
            raise RuntimeError("Candidate features require an A3 decision.")
        serving = decision.serving_cell
        tics = np.asarray([record.tic for record in records], dtype=np.float64)
        margins = np.asarray(
            [
                record.rsrp_a3_db[cell]
                - record.rsrp_a3_db[serving]
                - self.protocol_config.a3_hys
                - self.protocol_config.a3_off
                for record in records
            ],
            dtype=np.float64,
        )
        if self.observation_schema == "legacy_v1":
            trend_records = records[-self.config.trend_window_ticks :]
        else:
            trend_records = self._short_records(records)
        trend_tics = np.asarray(
            [record.tic for record in trend_records], dtype=np.float64
        )
        candidate_sinr = np.asarray(
            [record.sinr_rate_db[cell] for record in trend_records],
            dtype=np.float64,
        )
        serving_sinr = float(snapshot.sinr_rate_db[serving])
        candidate_current_sinr = float(snapshot.sinr_rate_db[cell])
        return (
            self._se_feature(candidate_current_sinr),
            self._delta_se_feature(candidate_current_sinr, serving_sinr),
            self._signed_feature(margins[-1], self.config.margin_clip_db),
            self._signed_feature(
                float(np.mean(margins)), self.config.margin_clip_db
            ),
            self._signed_feature(
                float(np.min(margins)), self.config.margin_clip_db
            ),
            self._signed_feature(
                self._slope_db_per_s(tics, margins),
                self.config.slope_clip_db_per_s,
            ),
            self._unsigned_feature(
                float(np.std(margins)), self.config.margin_std_clip_db
            ),
            self._signed_feature(
                self._slope_db_per_s(trend_tics, candidate_sinr),
                self.config.slope_clip_db_per_s,
            ),
            self._signed_feature(
                float(snapshot.sinr_rlm_db[cell]) - self.protocol_config.q_in_db,
                self.config.quality_gap_clip_db,
            ),
            float(snapshot.previous_cell == cell),
        )
