"""Validated physical-site topology associated with a radio-trace dataset."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path


@dataclass(frozen=True)
class CellSite:
    """One physical site and the directional sector cells hosted by it."""

    site_id: int
    position_xyz_m: tuple[float, float, float] | None
    cell_indices: tuple[int, ...]


class CellTopology:
    """Cell-to-site mapping used to construct physically local candidates."""

    def __init__(
        self,
        sites: tuple[CellSite, ...],
        *,
        sector_bearings_deg: tuple[float, ...],
        nearest_site_spacing_m: float | None = None,
    ) -> None:
        if not sites:
            raise ValueError("Topology must contain at least one physical site.")
        if not sector_bearings_deg:
            raise ValueError("Topology must contain sector bearings.")
        if len({site.site_id for site in sites}) != len(sites):
            raise ValueError("Topology site IDs must be unique.")

        cells = [cell for site in sites for cell in site.cell_indices]
        if len(set(cells)) != len(cells) or sorted(cells) != list(range(len(cells))):
            raise ValueError("Topology cells must be unique and contiguous from zero.")
        if any(len(site.cell_indices) == 0 for site in sites):
            raise ValueError("Every topology site must contain at least one cell.")
        if nearest_site_spacing_m is not None and (
            not math.isfinite(nearest_site_spacing_m)
            or nearest_site_spacing_m <= 0
        ):
            raise ValueError("nearest_site_spacing_m must be positive.")

        self.sites = tuple(sorted(sites, key=lambda site: site.site_id))
        self.sector_bearings_deg = tuple(float(x) for x in sector_bearings_deg)
        self.nearest_site_spacing_m = nearest_site_spacing_m
        self._site_by_id = {site.site_id: site for site in self.sites}
        self._site_id_by_cell = {
            cell: site.site_id for site in self.sites for cell in site.cell_indices
        }

    @property
    def num_cells(self) -> int:
        return len(self._site_id_by_cell)

    def site_id_for_cell(self, cell: int) -> int:
        try:
            return self._site_id_by_cell[int(cell)]
        except KeyError as error:
            raise ValueError(f"Cell {cell} is absent from the topology.") from error

    def cells_for_site(self, site_id: int) -> tuple[int, ...]:
        try:
            return self._site_by_id[int(site_id)].cell_indices
        except KeyError as error:
            raise ValueError(f"Unknown site ID: {site_id}.") from error

    def neighboring_site_ids(self, site_id: int) -> tuple[int, ...]:
        """Return first-ring physical neighbors; fall back to all other sites."""
        source = self._site_by_id[int(site_id)]
        if source.position_xyz_m is None:
            return tuple(site.site_id for site in self.sites if site != source)

        distances: list[tuple[int, float]] = []
        for site in self.sites:
            if site.site_id == source.site_id or site.position_xyz_m is None:
                continue
            distance = math.dist(source.position_xyz_m[:2], site.position_xyz_m[:2])
            distances.append((site.site_id, distance))
        if not distances:
            return ()
        spacing = self.nearest_site_spacing_m or min(
            distance for _, distance in distances
        )
        tolerance = max(1e-6, spacing * 1e-6)
        return tuple(
            site_id
            for site_id, distance in sorted(distances)
            if distance <= spacing + tolerance
        )

    @classmethod
    def contiguous_sector_groups(
        cls,
        num_cells: int,
        *,
        sectors_per_site: int = 3,
    ) -> "CellTopology":
        """Construct a signal-only fallback when no position metadata exists."""
        if num_cells <= 0 or sectors_per_site <= 0:
            raise ValueError("Cell and sector counts must be positive.")
        sites = tuple(
            CellSite(
                site_id=site_id,
                position_xyz_m=None,
                cell_indices=tuple(
                    range(
                        site_id * sectors_per_site,
                        min((site_id + 1) * sectors_per_site, num_cells),
                    )
                ),
            )
            for site_id in range(math.ceil(num_cells / sectors_per_site))
        )
        return cls(
            sites,
            sector_bearings_deg=tuple(
                360.0 * index / sectors_per_site
                for index in range(sectors_per_site)
            ),
        )


def load_cell_topology(data_dir: str | Path) -> CellTopology | None:
    """Load compact topology metadata, returning ``None`` for legacy folders."""
    path = Path(data_dir) / "dataset_provenance.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    topology = payload.get("topology")
    if not isinstance(topology, dict):
        raise ValueError(f"{path} does not contain a topology object.")

    sites = []
    for item in topology.get("sites", []):
        position = item.get("position_xyz_m")
        sites.append(
            CellSite(
                site_id=int(item["site_id"]),
                position_xyz_m=(
                    tuple(float(value) for value in position)
                    if position is not None
                    else None
                ),
                cell_indices=tuple(int(value) for value in item["cell_indices"]),
            )
        )
    result = CellTopology(
        tuple(sites),
        sector_bearings_deg=tuple(
            float(value) for value in topology["sector_bearings_deg"]
        ),
        nearest_site_spacing_m=(
            float(topology["nearest_site_spacing_m"])
            if topology.get("nearest_site_spacing_m") is not None
            else None
        ),
    )
    declared_cells = int(topology.get("sector_cell_count", result.num_cells))
    if declared_cells != result.num_cells:
        raise ValueError("Declared and mapped topology cell counts disagree.")
    return result
