"""Topology-aware configurable candidate extension of the A3 observation."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from ho_optim_drl.data import CellTopology
from ho_optim_drl.protocol.types import ProtocolSnapshot, TickResult

from .observation import A3ObservationBuilder, A3ObservationConfig


_MAX_CANDIDATES = 10
_SUPPORTED_CANDIDATE_COUNTS = (3, 5, 10)
_SUPPORTED_CANDIDATE_ORDERS = ("legacy", "ranked_v1")
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


@dataclass(frozen=True)
class _AllCellRecord:
    tic: int
    serving_cell: int | None
    rsrp_a3_db: np.ndarray
    sinr_rate_db: np.ndarray


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

    def __init__(
        self,
        protocol_config,
        observation_config: A3ObservationConfig | None = None,
        *,
        topology: CellTopology | None = None,
        candidate_count: int = _MAX_CANDIDATES,
        candidate_order: str = "legacy",
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
        self.topology = topology
        self.candidate_count = int(candidate_count)
        self.candidate_order = candidate_order
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

    def reset(self) -> None:
        super().reset()
        self._all_cell_history.clear()
        self.candidate_pool_cells = (None,) * self.MAX_CANDIDATES
        self.legacy_candidate_pool_cells = (None,) * self.MAX_CANDIDATES
        self.candidate_cells = (None,) * self.candidate_count
        self._candidate_tic = None
        self._cached_observation = None

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

    def observe(self, tick: TickResult) -> None:
        super().observe(tick)
        snapshot = tick.snapshot
        self._all_cell_history.append(
            _AllCellRecord(
                tic=snapshot.tic,
                serving_cell=snapshot.serving_cell,
                rsrp_a3_db=np.array(snapshot.rsrp_a3_db, copy=True),
                sinr_rate_db=np.array(snapshot.sinr_rate_db, copy=True),
            )
        )

    def build(self, snapshot: ProtocolSnapshot) -> np.ndarray:
        if self._candidate_tic == snapshot.tic:
            if self._cached_observation is None:
                raise RuntimeError("Candidate observation cache is incomplete.")
            return self._cached_observation.copy()

        base = super().build(snapshot)
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

        event_history = [
            record
            for record in self._all_cell_history
            if decision.a3_start_tic <= record.tic <= decision.tic
            and record.serving_cell == decision.serving_cell
        ]
        if not event_history:
            raise RuntimeError("No causal all-cell history is available.")

        values = list(base)
        for cell in self.candidate_cells[1:]:
            values.extend(
                self._candidate_features(
                    snapshot,
                    event_history,
                    cell,
                )
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
        trend_records = records[-self.config.trend_window_ticks :]
        trend_tics = np.asarray(
            [record.tic for record in trend_records], dtype=np.float64
        )
        candidate_sinr = np.asarray(
            [record.sinr_rate_db[cell] for record in trend_records],
            dtype=np.float64,
        )
        serving_eff = self._normalized_spectral_efficiency(
            float(snapshot.sinr_rate_db[serving])
        )
        candidate_eff = self._normalized_spectral_efficiency(
            float(snapshot.sinr_rate_db[cell])
        )
        return (
            candidate_eff,
            float(np.clip(candidate_eff - serving_eff, -1.0, 1.0)),
            self._signed_normalize(margins[-1], self.config.margin_clip_db),
            self._signed_normalize(
                float(np.mean(margins)), self.config.margin_clip_db
            ),
            self._signed_normalize(
                float(np.min(margins)), self.config.margin_clip_db
            ),
            self._signed_normalize(
                self._slope_db_per_s(tics, margins),
                self.config.slope_clip_db_per_s,
            ),
            self._unsigned_normalize(
                float(np.std(margins)), self.config.margin_std_clip_db
            ),
            self._signed_normalize(
                self._slope_db_per_s(trend_tics, candidate_sinr),
                self.config.slope_clip_db_per_s,
            ),
            self._signed_normalize(
                float(snapshot.sinr_rlm_db[cell]) - self.protocol_config.q_in_db,
                self.config.quality_gap_clip_db,
            ),
            float(snapshot.previous_cell == cell),
        )
