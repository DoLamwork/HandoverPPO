"""Create a publication-ready A2G system-model figure.

The figure is generated from the same topology and seeded mobility helpers used
by the simulator.  Radio calculations are intentionally not executed.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from uav_simulation.paths import FIGURES_DIR

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Arc, FancyArrowPatch, Rectangle, Wedge

from uav_simulation import config
from uav_simulation.hexagons import axial_to_pixel, build_hex_cover
from uav_simulation.mobility import GridMobility
from uav_simulation.route_tools import reconstruct_seeded_grid_route


MAP_CENTER_M = 5000.0
SITE_SPACING_SCALE_M = 500.0
SECTOR_SYMBOL_RADIUS_M = 600.0


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a top/side-view A2G system-model figure."
    )
    parser.add_argument("--speed", type=float, default=30.0)
    parser.add_argument("--route-id", type=int, default=0)
    parser.add_argument("--width", type=float, default=2000.0)
    parser.add_argument("--height", type=float, default=1667.0)
    parser.add_argument("--grid-spacing", type=float, default=100.0)
    parser.add_argument("--altitude", type=float, default=float(config.H_UT))
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--time-step", type=float, default=0.010)
    parser.add_argument("--output-dir", default=str(FIGURES_DIR))
    parser.add_argument("--name", default="system_model")
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args(argv)


def _topology(width: float, height: float, grid_spacing: float):
    xmin = MAP_CENTER_M - width / 2.0
    xmax = MAP_CENTER_M + width / 2.0
    ymin = MAP_CENTER_M - height / 2.0
    ymax = MAP_CENTER_M + height / 2.0

    x_lines = GridMobility._make_lines(xmin, xmax, grid_spacing)
    y_lines = GridMobility._make_lines(ymin, ymax, grid_spacing)
    hexes = build_hex_cover(
        SITE_SPACING_SCALE_M,
        width,
        height,
        MAP_CENTER_M,
        xmin,
        xmax,
        ymin,
        ymax,
    )
    sites = []
    for hex_obj in hexes:
        x_offset, y_offset = axial_to_pixel(
            hex_obj.q, hex_obj.r, SITE_SPACING_SCALE_M
        )
        sites.append((x_offset + MAP_CENTER_M - xmin, y_offset + MAP_CENTER_M - ymin))
    return xmin, xmax, ymin, ymax, x_lines, y_lines, sites


def _draw_sector_symbol(ax, x: float, y: float, bearing_deg: float, color: str):
    half_beamwidth = float(config.BS_ANTENNA_PHI_3DB_DEG) / 2.0
    # This radius is only a drawing scale for the schematic sector symbol. It
    # is not a serving-cell candidate range or a connection boundary.
    radius = SECTOR_SYMBOL_RADIUS_M
    ax.add_patch(
        Wedge(
            (x, y),
            radius,
            bearing_deg - half_beamwidth,
            bearing_deg + half_beamwidth,
            facecolor=color,
            edgecolor=color,
            linewidth=0.8,
            alpha=0.035,
            zorder=3,
        )
    )
    arrow_length = radius
    angle_rad = math.radians(bearing_deg)
    ax.annotate(
        "",
        xy=(x + arrow_length * math.cos(angle_rad), y + arrow_length * math.sin(angle_rad)),
        xytext=(x, y),
            arrowprops=dict(arrowstyle="-|>", color=color, lw=1.1, mutation_scale=8),
        zorder=5,
    )


def draw_top_view(ax, args: argparse.Namespace):
    xmin, xmax, ymin, ymax, x_lines, y_lines, sites = _topology(
        args.width, args.height, args.grid_spacing
    )

    for x_abs in x_lines:
        x = x_abs - xmin
        ax.plot([x, x], [0.0, args.height], color="0.82", lw=0.45, zorder=0)
    for y_abs in y_lines:
        y = y_abs - ymin
        ax.plot([0.0, args.width], [y, y], color="0.82", lw=0.45, zorder=0)

    ax.add_patch(
        Rectangle(
            (0.0, 0.0),
            args.width,
            args.height,
            facecolor="none",
            edgecolor="0.15",
            linewidth=1.3,
            zorder=2,
        )
    )

    sector_colors = ("#0072B2", "#D55E00", "#009E73")
    bearings = tuple(float(value) for value in config.BS_SECTOR_BEARINGS_DEG)
    for site_idx, (x, y) in enumerate(sites):
        for sector_idx, bearing in enumerate(bearings):
            _draw_sector_symbol(
                ax, x, y, bearing, sector_colors[sector_idx % len(sector_colors)]
            )
        ax.plot(
            x,
            y,
            marker="^",
            markersize=7,
            markerfacecolor="white",
            markeredgecolor="black",
            markeredgewidth=1.0,
            linestyle="None",
            zorder=7,
        )
        ax.text(x + 18.0, y + 18.0, f"S{site_idx}", fontsize=7.5, zorder=8)

    route, seed = reconstruct_seeded_grid_route(
        speed_kmh=args.speed,
        route_id=args.route_id,
        x_lines=x_lines,
        y_lines=y_lines,
        rect_xmin=xmin,
        rect_xmax=xmax,
        rect_ymin=ymin,
        rect_ymax=ymax,
        steps=args.steps,
        time_per_step_s=args.time_step,
    )
    route = route - np.array([xmin, ymin])
    ax.plot(route[:, 0], route[:, 1], color="#6A3D9A", lw=1.7, zorder=6)
    ax.scatter(
        route[0, 0], route[0, 1], s=38, c="#2CA02C", edgecolors="black",
        linewidths=0.7, label="Route start", zorder=9
    )
    ax.scatter(
        route[-1, 0], route[-1, 1], s=48, c="#CC0000", marker="X",
        edgecolors="black", linewidths=0.7, label="Route end", zorder=9
    )

    sample_idx = np.linspace(0, max(0, len(route) - 2), num=12, dtype=int)
    vectors = route[sample_idx + 1] - route[sample_idx]
    moving = np.linalg.norm(vectors, axis=1) > 1e-12
    if np.any(moving):
        vectors = vectors[moving]
        norms = np.linalg.norm(vectors, axis=1)
        vectors = vectors / norms[:, None] * 40.0
        idx = sample_idx[moving]
        ax.quiver(
            route[idx, 0], route[idx, 1], vectors[:, 0], vectors[:, 1],
            angles="xy", scale_units="xy", scale=1.0, color="#6A3D9A",
            width=0.003, headwidth=4.0, headlength=5.0, zorder=7
        )

    legend_items = [
        Line2D([0], [0], marker="^", color="none", markerfacecolor="white",
               markeredgecolor="black", markersize=7, label="Ground BS site (3 sectors)"),
        Line2D(
            [0], [0], color=sector_colors[0], lw=1.5,
            label="Sector boresight / 65° HPBW (schematic)",
        ),
        Line2D([0], [0], color="#6A3D9A", lw=1.8,
               label=f"UAV route ({args.speed:g} km/h, seed {seed})"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#2CA02C",
               markeredgecolor="black", markersize=6, label="Start"),
        Line2D([0], [0], marker="X", color="none", markerfacecolor="#CC0000",
               markeredgecolor="black", markersize=6, label="End"),
    ]
    ax.legend(handles=legend_items, loc="upper right", fontsize=7.2, framealpha=0.95)
    # Keep only a small border around the simulated rectangle. Sector arrows
    # use a 600 m drawing length, but portions outside the plotted
    # map are clipped so the exported top view does not contain excess whitespace.
    margin = 75.0
    ax.set_xlim(-margin, args.width + margin)
    ax.set_ylim(-margin, args.height + margin)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Map x (m)")
    ax.set_ylabel("Map y (m)")
    ax.set_title("(a) Network topology and UAV mobility", fontsize=11)


def _double_arrow(
    ax,
    start,
    end,
    label,
    *,
    color="0.2",
    label_offset=(0.0, 0.0),
    label_rotation=0.0,
):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="<->",
            mutation_scale=10,
            color=color,
            linewidth=1.0,
            zorder=6,
        )
    )
    mx = (start[0] + end[0]) / 2.0 + label_offset[0]
    my = (start[1] + end[1]) / 2.0 + label_offset[1]
    ax.text(
        mx,
        my,
        label,
        fontsize=8.5,
        color=color,
        ha="center",
        va="center",
        rotation=label_rotation,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=1.0),
        zorder=8,
    )


def draw_side_view(ax, args: argparse.Namespace):
    h_bs = float(config.H_BS)
    h_uav = float(args.altitude)
    bs_x = 120.0
    uav_x = 820.0
    horizontal_distance = uav_x - bs_x
    elevation_deg = math.degrees(math.atan2(h_uav - h_bs, horizontal_distance))

    ax.axhspan(0.0, 4.0, color="#8C6D31", alpha=0.65, zorder=0)
    ax.plot([0.0, 950.0], [4.0, 4.0], color="0.25", lw=1.0, zorder=1)

    # Conceptual buildings only: the simulator uses a statistical urban model.
    buildings = ((285.0, 52.0, 75.0), (425.0, 31.0, 90.0), (590.0, 45.0, 68.0))
    for x, h, w in buildings:
        ax.add_patch(
            Rectangle((x, 4.0), w, h, facecolor="0.82", edgecolor="0.55",
                      linewidth=0.7, zorder=1)
        )

    ax.plot([bs_x, bs_x], [4.0, h_bs], color="0.15", lw=3.0, zorder=4)
    ax.plot([bs_x - 9.0, bs_x + 9.0], [h_bs, h_bs], color="#0072B2", lw=5.0, zorder=5)
    ax.text(
        bs_x + 10.0,
        8.0,
        "Ground BS",
        ha="left",
        va="bottom",
        fontsize=8.5,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=1.0),
        zorder=8,
    )

    # UAV symbol.
    ax.plot([uav_x - 22.0, uav_x + 22.0], [h_uav, h_uav], color="#6A3D9A", lw=2.0)
    ax.plot([uav_x, uav_x], [h_uav - 7.0, h_uav + 7.0], color="#6A3D9A", lw=1.5)
    for rotor_x in (uav_x - 22.0, uav_x + 22.0):
        ax.add_patch(Arc((rotor_x, h_uav), 24.0, 5.0, color="#6A3D9A", lw=1.2))
    ax.scatter([uav_x], [h_uav], s=25, c="#6A3D9A", zorder=7)
    ax.annotate(
        "UAV-UE",
        xy=(uav_x, h_uav),
        xytext=(uav_x + 22.0, h_uav + 4.0),
        fontsize=8.5,
        ha="left",
        va="center",
        arrowprops=dict(arrowstyle="-", color="0.3", lw=0.8),
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=1.0),
        zorder=8,
    )

    ax.plot(
        [bs_x, uav_x], [h_bs, h_uav], color="#CC0000", lw=1.7,
        label=r"A2G link $d_{3D}$", zorder=5
    )
    ax.text(
        (bs_x + uav_x) / 2.0,
        (h_bs + h_uav) / 2.0 + 8.0,
        r"LoS/NLoS channel, $d_{3D}$",
        color="#CC0000",
        fontsize=8.5,
        ha="center",
        rotation=elevation_deg,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.78, pad=1.0),
        zorder=8,
    )

    # Horizontal reference and elevation angle.
    ax.plot([bs_x, bs_x + 155.0], [h_bs, h_bs], color="0.35", ls=":", lw=1.0)
    ax.add_patch(
        Arc(
            (bs_x, h_bs), 115.0, 115.0, angle=0.0,
            theta1=0.0, theta2=elevation_deg,
            color="#CC0000", lw=1.0, zorder=6
        )
    )
    ax.text(
        bs_x + 67.0,
        h_bs + 7.0,
        r"$\theta$",
        fontsize=9,
        color="#CC0000",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=0.5),
        zorder=8,
    )

    # Electrical downtilt is shown separately from the UAV link direction.
    downtilt = float(config.BS_ANTENNA_ELECTRICAL_DOWNTILT_DEG)
    beam_length = 135.0
    ax.plot(
        [bs_x, bs_x + beam_length * math.cos(math.radians(downtilt))],
        [h_bs, h_bs - beam_length * math.sin(math.radians(downtilt))],
        color="#0072B2", ls="--", lw=1.2, zorder=4
    )
    ax.text(
        bs_x + 142.0, h_bs - 3.0,
        f"BS boresight ({downtilt:g}° downtilt)",
        color="#0072B2", fontsize=7.8, ha="left", va="center",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=1.0),
        zorder=8,
    )

    _double_arrow(
        ax, (78.0, 4.0), (78.0, h_bs),
        rf"$h_{{BS}}={h_bs:g}$ m", label_offset=(-20.0, 0.0),
        label_rotation=90.0,
    )
    _double_arrow(
        ax, (900.0, 4.0), (900.0, h_uav),
        rf"$h_U={h_uav:g}$ m", label_offset=(20.0, 0.0),
        label_rotation=90.0,
    )
    _double_arrow(
        ax, (bs_x, -7.0), (uav_x, -7.0),
        rf"horizontal distance $r={horizontal_distance:g}$ m",
        label_offset=(0.0, -6.0),
    )

    ax.set_xlim(0.0, 975.0)
    ax.set_ylim(-20.0, max(132.0, h_uav + 32.0))
    ax.set_xlabel("Horizontal distance (m)")
    ax.set_ylabel("Height (m)")
    ax.set_title("(b) A2G geometry and antenna orientation", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)


def main(argv=None):
    args = parse_args(argv)
    if args.width <= 0.0 or args.height <= 0.0:
        raise ValueError("Map dimensions must be positive.")
    if args.grid_spacing <= 0.0 or args.altitude <= 0.0:
        raise ValueError("Grid spacing and altitude must be positive.")
    if args.speed < 0.0 or args.route_id < 0:
        raise ValueError("Speed and route ID must be non-negative.")

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13.6, 6.2),
        gridspec_kw={"width_ratios": (1.45, 1.0)},
        constrained_layout=True,
    )
    draw_top_view(axes[0], args)
    draw_side_view(axes[1], args)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{args.name}.png"
    pdf_path = output_dir / f"{args.name}.pdf"
    fig.savefig(png_path, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    print(f"Saved PNG: {png_path.resolve()}")
    print(f"Saved PDF: {pdf_path.resolve()}")

    # Export each panel as a true standalone vector figure.  Redrawing the
    # panels gives Matplotlib enough room to lay out their labels correctly;
    # cropping the combined image would retain the narrower subplot geometry.
    top_fig, top_ax = plt.subplots(figsize=(8.6, 7.4), constrained_layout=True)
    draw_top_view(top_ax, args)
    top_pdf_path = output_dir / f"{args.name}_top_view.pdf"
    top_fig.savefig(top_pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(top_fig)

    side_fig, side_ax = plt.subplots(figsize=(8.6, 5.8), constrained_layout=True)
    draw_side_view(side_ax, args)
    side_pdf_path = output_dir / f"{args.name}_side_view.pdf"
    side_fig.savefig(side_pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(side_fig)

    print(f"Saved top-view PDF: {top_pdf_path.resolve()}")
    print(f"Saved side-view PDF: {side_pdf_path.resolve()}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
