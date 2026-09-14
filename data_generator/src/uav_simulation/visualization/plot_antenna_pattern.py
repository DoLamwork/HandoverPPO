"""Render the 3GPP TR 38.901 single-element and panel radiation patterns."""

import argparse
from pathlib import Path

from uav_simulation.paths import FIGURES_DIR

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors, cm
import numpy as np

from uav_simulation import config
from uav_simulation.antenna_model import ThreeGPPPanelArray, single_element_gain_db


def pattern_surface(gain_db, theta_deg, phi_deg):
    relative_db = np.maximum(gain_db - np.max(gain_db), -40.0)
    radius = np.power(10.0, relative_db / 20.0)
    theta = np.deg2rad(theta_deg)
    phi = np.deg2rad(phi_deg)
    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)
    return x, y, z


def add_surface(ax, x, y, z, gain_db, norm):
    ax.plot_surface(
        x,
        y,
        z,
        facecolors=cm.viridis(norm(gain_db)),
        rstride=1,
        cstride=1,
        linewidth=0.0,
        antialiased=False,
        shade=False,
    )
    ax.set_box_aspect((1.0, 1.0, 1.0))
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_zlim(-1.0, 1.0)
    ax.set_xlabel('GCS x')
    ax.set_ylabel('GCS y')
    ax.set_zlabel('GCS z')
    ax.view_init(elev=24.0, azim=42.0)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--output', type=Path,
        default=FIGURES_DIR / 'antenna_pattern_3gpp_3d.png',
        help='Destination PNG file.',
    )
    args = parser.parse_args(argv)
    theta_axis = np.linspace(0.0, 180.0, 121)
    phi_axis = np.linspace(-180.0, 180.0, 241)
    theta, phi = np.meshgrid(theta_axis, phi_axis, indexing='ij')

    element_gain_db = single_element_gain_db(theta, phi)
    panel = ThreeGPPPanelArray(
        num_vertical=config.BS_ANTENNA_NUM_VERTICAL,
        num_horizontal=config.BS_ANTENNA_NUM_HORIZONTAL,
        spacing_vertical_lambda=config.BS_ANTENNA_SPACING_VERTICAL_LAMBDA,
        spacing_horizontal_lambda=config.BS_ANTENNA_SPACING_HORIZONTAL_LAMBDA,
        electrical_downtilt_deg=config.BS_ANTENNA_ELECTRICAL_DOWNTILT_DEG,
        polarization_slant_deg=config.BS_ANTENNA_POLARIZATION_SLANT_DEG,
        polarization_model=config.BS_ANTENNA_POLARIZATION_MODEL,
    )
    panel_gain_db = panel.power_gain_db_gcs(theta, phi)

    all_gain = np.concatenate((element_gain_db.ravel(), panel_gain_db.ravel()))
    norm = colors.Normalize(vmin=max(float(np.min(all_gain)), -40.0), vmax=float(np.max(all_gain)))
    figure = plt.figure(figsize=(13.5, 6.5), constrained_layout=True)
    element_ax = figure.add_subplot(1, 2, 1, projection='3d')
    panel_ax = figure.add_subplot(1, 2, 2, projection='3d')

    add_surface(
        element_ax,
        *pattern_surface(element_gain_db, theta, phi),
        element_gain_db,
        norm,
    )
    element_ax.set_title('3GPP single element (Table 7.3-1)')
    add_surface(
        panel_ax,
        *pattern_surface(panel_gain_db, theta, phi),
        panel_gain_db,
        norm,
    )
    panel_ax.set_title(
        f'Panel {panel.num_vertical}x{panel.num_horizontal}, '
        f'electrical downtilt {panel.electrical_downtilt_deg:g} deg'
    )

    colorbar = figure.colorbar(cm.ScalarMappable(norm=norm, cmap='viridis'), ax=[element_ax, panel_ax], shrink=0.72)
    colorbar.set_label('Gain (dBi)')
    output_path = args.output.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220)
    plt.close(figure)
    print(f'Saved {output_path.resolve()}')
    print(f'Element peak: {np.max(element_gain_db):.3f} dBi')
    print(f'Panel sampled peak: {np.max(panel_gain_db):.3f} dBi')


if __name__ == '__main__':
    main()
