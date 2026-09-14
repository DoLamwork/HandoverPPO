import argparse
import math
import re
from pathlib import Path

from uav_simulation.paths import DATA_100M_DIR

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button, TextBox

from uav_simulation import config
from uav_simulation.hexagons import axial_to_pixel, build_hex_cover
from uav_simulation.mobility import GridMobility
from uav_simulation.route_tools import reconstruct_seeded_grid_route


DATA_FILE_RE = re.compile(
    r'^ue(?P<ue>\d+)_(?P<metric>rsrp|sinr)_'
    r'(?P<speed>\d+(?:\.\d+)?)kmh_(?P<route>\d+)\.csv$',
    re.IGNORECASE,
)


def discover_route_ids(data_dir: Path, speed_kmh: float, ue_idx: int = 0):
    """Discover route IDs from RSRP/SINR names and de-duplicate each pair."""
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise ValueError(f'Data folder does not exist: {data_dir}')

    metrics_by_route: dict[int, set[str]] = {}
    for path in data_dir.glob('*.csv'):
        match = DATA_FILE_RE.match(path.name)
        if match is None or int(match.group('ue')) != int(ue_idx):
            continue
        file_speed = float(match.group('speed'))
        if not math.isclose(file_speed, float(speed_kmh), abs_tol=1e-9):
            continue
        route_id = int(match.group('route'))
        metrics_by_route.setdefault(route_id, set()).add(
            match.group('metric').lower()
        )

    route_ids = sorted(metrics_by_route)
    if not route_ids:
        raise ValueError(
            f'No UE{ue_idx} RSRP/SINR files found for {float(speed_kmh):g} km/h '
            f'in {data_dir}'
        )
    complete_pairs = sum(
        {'rsrp', 'sinr'}.issubset(metrics) for metrics in metrics_by_route.values()
    )
    return route_ids, complete_pairs


class MobilityHeatmapViewer:
    """Aggregate seeded UE routes into a mobility-only spatial heatmap."""

    def __init__(
        self,
        *,
        data_dir: str | Path = DATA_100M_DIR,
        speed_kmh: float = 30.0,
        rect_len_m: float = 2000.0,
        rect_wid_m: float = 1667.0,
        grid_spacing_m: float = 500.0,
        steps: int = 6000,
        time_per_step_s: float = config.DELTA_T_MS / 1000.0,
        bin_size_m: float = 25.0,
        ue_idx: int = 0,
    ):
        self.project_dir = Path.cwd()
        self.data_dir = self._resolve_data_dir(data_dir)
        self.speed_kmh = float(speed_kmh)
        self.width = float(rect_len_m)
        self.height = float(rect_wid_m)
        self.grid_spacing_m = float(grid_spacing_m)
        self.steps = int(steps)
        self.time_per_step_s = float(time_per_step_s)
        self.bin_size_m = float(bin_size_m)
        self.ue_idx = int(ue_idx)

        if self.width <= 0.0 or self.height <= 0.0:
            raise ValueError('Map dimensions must be positive.')
        if self.grid_spacing_m <= 0.0 or self.bin_size_m <= 0.0:
            raise ValueError('Grid spacing and heatmap bin size must be positive.')

        self.total_size = 10000.0
        self.center = self.total_size / 2.0
        self.rect_xmin = self.center - self.width / 2.0
        self.rect_xmax = self.center + self.width / 2.0
        self.rect_ymin = self.center - self.height / 2.0
        self.rect_ymax = self.center + self.height / 2.0
        self.x_lines = GridMobility._make_lines(
            self.rect_xmin, self.rect_xmax, self.grid_spacing_m
        )
        self.y_lines = GridMobility._make_lines(
            self.rect_ymin, self.rect_ymax, self.grid_spacing_m
        )

        self.fig = plt.figure(figsize=(12, 8.5))
        self.ax = self.fig.add_axes([0.07, 0.18, 0.78, 0.75])
        self.colorbar_ax = self.fig.add_axes([0.88, 0.20, 0.025, 0.70])
        self.colorbar_ax.set_visible(False)

        data_ax = self.fig.add_axes([0.17, 0.085, 0.30, 0.05])
        speed_ax = self.fig.add_axes([0.61, 0.085, 0.10, 0.05])
        show_ax = self.fig.add_axes([0.75, 0.085, 0.13, 0.05])
        self.data_box = TextBox(
            data_ax, 'Data folder', initial=self._display_data_dir(self.data_dir)
        )
        self.speed_box = TextBox(
            speed_ax, 'Speed (km/h)', initial=f'{self.speed_kmh:g}'
        )
        self.show_button = Button(show_ax, 'Show Heatmap', color='lightskyblue')
        self.show_button.on_clicked(self.show_heatmap_from_inputs)
        self.speed_box.on_submit(self.show_heatmap_from_inputs)
        self.status_text = self.fig.text(
            0.07, 0.035, 'Ready. Route positions are reconstructed from seed; radio is not run.',
            fontsize=9, va='center'
        )

        self.heatmap_artist = None
        self.colorbar = None
        self._draw_base_map()

    def _resolve_data_dir(self, data_dir: str | Path) -> Path:
        path = Path(str(data_dir).strip()).expanduser()
        if not path.is_absolute():
            path = self.project_dir / path
        return path.resolve()

    def _display_data_dir(self, data_dir: Path) -> str:
        try:
            return str(data_dir.relative_to(self.project_dir))
        except ValueError:
            return str(data_dir)

    def _draw_base_map(self):
        for x_abs in self.x_lines:
            x = x_abs - self.rect_xmin
            self.ax.plot(
                [x, x], [0.0, self.height], color='0.35',
                linewidth=0.8, alpha=0.6, zorder=5
            )
        for y_abs in self.y_lines:
            y = y_abs - self.rect_ymin
            self.ax.plot(
                [0.0, self.width], [y, y], color='0.35',
                linewidth=0.8, alpha=0.6, zorder=5
            )

        self.ax.add_patch(Rectangle(
            (0.0, 0.0), self.width, self.height,
            linewidth=2.0, edgecolor='tab:red', facecolor='none',
            linestyle='--', zorder=7
        ))

        scale_factor = 500.0
        hexes = build_hex_cover(
            scale_factor, self.width, self.height, self.center,
            self.rect_xmin, self.rect_xmax, self.rect_ymin, self.rect_ymax
        )
        for site_idx, hex_obj in enumerate(hexes):
            x_abs, y_abs = axial_to_pixel(hex_obj.q, hex_obj.r, scale_factor)
            x = x_abs + self.center - self.rect_xmin
            y = y_abs + self.center - self.rect_ymin
            self.ax.plot(
                x, y, marker='^', color='tab:cyan', markeredgecolor='black',
                markersize=7, zorder=9
            )
            self.ax.text(
                x + 18.0, y + 18.0, f'S{site_idx}', fontsize=8,
                color='black', zorder=9
            )
            angles = np.radians(np.arange(0.0, 361.0, 60.0))
            xs = x + scale_factor * np.cos(angles)
            ys = y + scale_factor * np.sin(angles)
            self.ax.plot(
                xs, ys, color='0.2', linestyle='--', linewidth=0.8,
                alpha=0.5, zorder=4
            )

        margin_m = max(25.0, 0.04 * max(self.width, self.height))
        self.ax.set_xlim(-margin_m, self.width + margin_m)
        self.ax.set_ylim(-margin_m, self.height + margin_m)
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.set_xlabel('Map x (m)')
        self.ax.set_ylabel('Map y (m)')
        self.ax.set_title('UE mobility heatmap')

    def _clear_heatmap(self):
        if self.heatmap_artist is not None:
            self.heatmap_artist.remove()
            self.heatmap_artist = None
        if self.colorbar is not None:
            self.colorbar.remove()
            self.colorbar = None
        self.colorbar_ax.clear()
        self.colorbar_ax.set_visible(False)

    def build_heatmap(self, speed_kmh: float, data_dir: Path):
        route_ids, complete_pairs = discover_route_ids(
            data_dir, speed_kmh, self.ue_idx
        )
        x_edges = np.linspace(
            0.0, self.width, int(math.ceil(self.width / self.bin_size_m)) + 1
        )
        y_edges = np.linspace(
            0.0, self.height, int(math.ceil(self.height / self.bin_size_m)) + 1
        )
        histogram = np.zeros((len(x_edges) - 1, len(y_edges) - 1), dtype=float)
        start_points = set()

        offset = np.array([self.rect_xmin, self.rect_ymin], dtype=float)
        for route_id in route_ids:
            points, _ = reconstruct_seeded_grid_route(
                speed_kmh=speed_kmh,
                route_id=route_id,
                x_lines=self.x_lines,
                y_lines=self.y_lines,
                rect_xmin=self.rect_xmin,
                rect_xmax=self.rect_xmax,
                rect_ymin=self.rect_ymin,
                rect_ymax=self.rect_ymax,
                steps=self.steps,
                time_per_step_s=self.time_per_step_s,
            )
            points_local = points - offset
            start_points.add(tuple(np.round(points_local[0], 6)))

            # points[1:] correspond exactly to CSV Step 0..Step 5999.
            route_histogram, _, _ = np.histogram2d(
                points_local[1:, 0], points_local[1:, 1],
                bins=(x_edges, y_edges)
            )
            histogram += route_histogram

        return {
            'histogram': histogram,
            'x_edges': x_edges,
            'y_edges': y_edges,
            'route_ids': route_ids,
            'complete_pairs': complete_pairs,
            'unique_starts': len(start_points),
        }

    def show_heatmap(self, speed_kmh: float, data_dir: str | Path | None = None):
        speed_kmh = float(speed_kmh)
        if not np.isfinite(speed_kmh) or speed_kmh < 0.0:
            raise ValueError('Speed must be a non-negative finite number.')
        data_dir = self.data_dir if data_dir is None else self._resolve_data_dir(data_dir)

        result = self.build_heatmap(speed_kmh, data_dir)
        histogram = result['histogram']
        positive = histogram[histogram > 0.0]
        if positive.size == 0:
            raise ValueError('The selected routes contain no movement samples.')

        self._clear_heatmap()
        masked = np.ma.masked_less_equal(histogram.T, 0.0)
        self.heatmap_artist = self.ax.pcolormesh(
            result['x_edges'], result['y_edges'], masked,
            cmap='YlOrRd',
            norm=LogNorm(vmin=max(1.0, float(positive.min())),
                         vmax=max(2.0, float(positive.max()))),
            shading='flat', alpha=0.82, zorder=3
        )
        self.colorbar_ax.set_visible(True)
        self.colorbar = self.fig.colorbar(
            self.heatmap_artist, cax=self.colorbar_ax
        )
        self.colorbar.set_label(
            f'UE samples per ~{self.bin_size_m:g} m cell (log scale)'
        )

        self.data_dir = data_dir
        self.speed_kmh = speed_kmh
        route_ids = result['route_ids']
        route_range = (
            str(route_ids[0]) if len(route_ids) == 1
            else f'{route_ids[0]}-{route_ids[-1]}'
        )
        total_samples = len(route_ids) * self.steps
        total_intersections = len(self.x_lines) * len(self.y_lines)
        occupied_bins = int(np.count_nonzero(histogram))
        self.status_text.set_color('black')
        self.status_text.set_text(
            f'{len(route_ids)} routes (IDs {route_range}); '
            f'{result["complete_pairs"]} complete RSRP/SINR pairs; '
            f'{total_samples:,} positions; starts '
            f'{result["unique_starts"]}/{total_intersections}; '
            f'{occupied_bins} occupied heatmap cells.'
        )
        self.ax.set_title(
            f'UE mobility heatmap — {speed_kmh:g} km/h — {data_dir.name}'
        )
        self.fig.canvas.draw_idle()
        return result

    def show_heatmap_from_inputs(self, event=None):
        try:
            speed_kmh = float(self.speed_box.text.strip())
            data_dir = self._resolve_data_dir(self.data_box.text)
            return self.show_heatmap(speed_kmh, data_dir)
        except (OSError, TypeError, ValueError) as exc:
            self.status_text.set_color('tab:red')
            self.status_text.set_text(f'Heatmap error: {exc}')
            self.fig.canvas.draw_idle()
            return None


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Display a mobility-only heatmap reconstructed from route seeds.'
    )
    parser.add_argument('--data-dir', default=str(DATA_100M_DIR))
    parser.add_argument('--speed', type=float, default=30.0)
    parser.add_argument('--width', type=float, default=2000.0)
    parser.add_argument('--height', type=float, default=1667.0)
    parser.add_argument('--grid-spacing', type=float, default=500.0)
    parser.add_argument('--steps', type=int, default=6000)
    parser.add_argument('--time-per-step', type=float, default=0.010)
    parser.add_argument('--bin-size', type=float, default=25.0)
    parser.add_argument('--save', type=Path, default=None)
    parser.add_argument('--no-show', action='store_true')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    viewer = MobilityHeatmapViewer(
        data_dir=args.data_dir,
        speed_kmh=args.speed,
        rect_len_m=args.width,
        rect_wid_m=args.height,
        grid_spacing_m=args.grid_spacing,
        steps=args.steps,
        time_per_step_s=args.time_per_step,
        bin_size_m=args.bin_size,
    )
    viewer.show_heatmap(args.speed)
    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        viewer.fig.savefig(args.save, dpi=160, bbox_inches='tight')
    if not args.no_show:
        plt.show(block=True)


if __name__ == '__main__':
    main()
