"""Validated radio inputs consumed by the protocol engine."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RadioTrace:
    """
    Three explicit radio streams in ``(base_stations, timesteps)`` layout.

    ``rsrp_a3_db`` is L3-filtered and is used only by A3/TTT.
    ``sinr_rate_db`` is unfiltered and is used for estimated delivered rate.
    ``sinr_rlm_db`` is the RLM proxy used by Qin/Qout, N310/N311, and T310.
    """

    rsrp_a3_db: np.ndarray
    sinr_rate_db: np.ndarray
    sinr_rlm_db: np.ndarray
    delta_t_ms: int
    trace_id: str | None = None

    def __post_init__(self) -> None:
        if self.delta_t_ms <= 0:
            raise ValueError("delta_t_ms must be positive.")

        arrays = {}
        for name in ("rsrp_a3_db", "sinr_rate_db", "sinr_rlm_db"):
            value = np.asarray(getattr(self, name), dtype=np.float64)
            if value.ndim != 2:
                raise ValueError(f"{name} must be a 2-D BS-by-time array.")
            if value.shape[0] == 0 or value.shape[1] == 0:
                raise ValueError(f"{name} must not be empty.")
            if not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must contain only finite values.")
            copied = np.array(value, copy=True)
            copied.setflags(write=False)
            arrays[name] = copied

        shapes = {value.shape for value in arrays.values()}
        if len(shapes) != 1:
            raise ValueError(
                "RSRP A3, SINR rate, and SINR RLM arrays must have equal shapes."
            )

        for name, value in arrays.items():
            object.__setattr__(self, name, value)

    @property
    def shape(self) -> tuple[int, int]:
        return self.rsrp_a3_db.shape

    @property
    def num_base_stations(self) -> int:
        return self.shape[0]

    @property
    def num_samples(self) -> int:
        return self.shape[1]

    @classmethod
    def from_arrays(
        cls,
        rsrp_a3_db: np.ndarray,
        sinr_rate_db: np.ndarray,
        *,
        sinr_rlm_db: np.ndarray | None = None,
        delta_t_ms: int = 10,
        trace_id: str | None = None,
    ) -> "RadioTrace":
        """Construct a trace; by default RLM uses the raw rate SINR stream."""
        if sinr_rlm_db is None:
            sinr_rlm_db = sinr_rate_db
        return cls(
            rsrp_a3_db=rsrp_a3_db,
            sinr_rate_db=sinr_rate_db,
            sinr_rlm_db=sinr_rlm_db,
            delta_t_ms=delta_t_ms,
            trace_id=trace_id,
        )
