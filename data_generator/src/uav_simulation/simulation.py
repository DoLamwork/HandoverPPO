import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button, TextBox

from uav_simulation.hexagons import axial_to_pixel, build_hex_cover
from uav_simulation.antenna_model import ThreeGPPPanelArray
from uav_simulation import config

from uav_simulation.mobility import GridMobility
from uav_simulation.route_tools import reconstruct_seeded_grid_route
from uav_simulation.radio_models import RadioModel
from uav_simulation.logger import SimulationLogger
from uav_simulation.paths import DATA_DIR

class CellularNetworkReceivedPower:
    """Generate per-cell RSRP/SINR traces without connection or handover state."""

    def __init__(
        self,
        num_ues: int = 5,
        rect_len_m: float = 4000.0,
        rect_wid_m: float = 3000.0,
        grid_spacing_m: float = 200.0,
        add_uav_cover: bool = False,
        uav_radius_m: float = 750.0,
        uav_altitude_m: float = 100.0,
        uav_ptx_dbm: float = 40.0,
        fast_mode: bool = True,
        headless: bool = False,
        uav_ue_speed_kmh: float | None = None,
        uav_ue_altitude_m: float | None = None,
        seed: int | None = None,
        output_run_idx: int | None = None,
        output_data_dir: str = str(DATA_DIR),
        train_ready_output: bool = False,
    ):
        self.seed = None if seed is None else int(seed)
        if seed is not None:
            np.random.seed(self.seed)
        self.rng = np.random.default_rng(self.seed)
        if add_uav_cover:
            raise ValueError("This project supports UAV-UE to Ground-BS links only.")
        self.output_run_idx = (
            None if output_run_idx is None else int(output_run_idx)
        )
        if self.output_run_idx is not None and self.output_run_idx < 0:
            raise ValueError("output_run_idx must be non-negative.")
        self.output_data_dir = str(output_data_dir).strip()
        if not self.output_data_dir:
            raise ValueError("output_data_dir must not be empty.")
        self.train_ready_output = bool(train_ready_output)



        # Core sim canvas
        self.total_size = 10000.0
        self.center = self.total_size / 2.0

        # Region (HCN)
        self.width, self.height = float(rect_len_m), float(rect_wid_m)
        self.rect_xmin = self.center - self.width / 2.0
        self.rect_xmax = self.center + self.width / 2.0
        self.rect_ymin = self.center - self.height / 2.0
        self.rect_ymax = self.center + self.height / 2.0

        self.mobility = GridMobility(self)
        self.radio_model = RadioModel(self)
        self.logger = SimulationLogger(self)

        # Road grid
        self.grid_spacing_m = float(max(10.0, grid_spacing_m))
        self.x_lines = GridMobility._make_lines(self.rect_xmin, self.rect_xmax, self.grid_spacing_m)
        self.y_lines = GridMobility._make_lines(self.rect_ymin, self.rect_ymax, self.grid_spacing_m)

        # UAV deployment (optional)
        self.add_uav_cover = bool(add_uav_cover)
        self.uav_radius_m = float(uav_radius_m)
        self.uav_altitude_m = float(uav_altitude_m)
        self.uav_ptx_dbm = float(uav_ptx_dbm)

        # Radio params
        self.ptx, self.gtx, self.grx = config.PTX, config.GTX, config.GRX
        self.fc = config.FC
        self.use_3gpp_3d_antenna = bool(config.USE_3GPP_3D_ANTENNA)
        self.bs_sector_bearings_deg = (
            tuple(float(value) for value in config.BS_SECTOR_BEARINGS_DEG)
            if self.use_3gpp_3d_antenna
            else (0.0,)
        )
        if not self.bs_sector_bearings_deg:
            raise ValueError("BS_SECTOR_BEARINGS_DEG must not be empty.")
        self.bs_antenna_kwargs = {
            'num_vertical': int(config.BS_ANTENNA_NUM_VERTICAL),
            'num_horizontal': int(config.BS_ANTENNA_NUM_HORIZONTAL),
            'spacing_vertical_lambda': float(config.BS_ANTENNA_SPACING_VERTICAL_LAMBDA),
            'spacing_horizontal_lambda': float(config.BS_ANTENNA_SPACING_HORIZONTAL_LAMBDA),
            'mechanical_downtilt_deg': float(config.BS_ANTENNA_MECHANICAL_DOWNTILT_DEG),
            'mechanical_slant_deg': float(config.BS_ANTENNA_MECHANICAL_SLANT_DEG),
            'electrical_downtilt_deg': float(config.BS_ANTENNA_ELECTRICAL_DOWNTILT_DEG),
            'polarization_slant_deg': float(config.BS_ANTENNA_POLARIZATION_SLANT_DEG),
            'polarization_model': int(config.BS_ANTENNA_POLARIZATION_MODEL),
            'max_element_gain_dbi': float(config.BS_ANTENNA_MAX_ELEMENT_GAIN_DBI),
            'theta_3db_deg': float(config.BS_ANTENNA_THETA_3DB_DEG),
            'phi_3db_deg': float(config.BS_ANTENNA_PHI_3DB_DEG),
            'side_lobe_attenuation_db': float(config.BS_ANTENNA_VERTICAL_SIDE_LOBE_ATTENUATION_DB),
            'front_back_attenuation_db': float(config.BS_ANTENNA_FRONT_BACK_ATTENUATION_DB),
        }

        # Full-channel noise and the NR resource grid used to derive RSRP.
        self.bandwidth_hz = float(config.BANDWIDTH_MHZ) * 1e6
        self.num_resource_blocks = int(config.NUM_RESOURCE_BLOCKS)
        self.subcarriers_per_rb = int(config.SUBCARRIERS_PER_RB)
        self.noise_figure_db = float(config.NOISE_FIGURE_DB)
        self.thermal_noise_density_dbm_hz = float(config.THERMAL_NOISE_DENSITY_DBM_HZ)
        self.reference_signal_power_offset_db = float(config.REFERENCE_SIGNAL_POWER_OFFSET_DB)
        if not np.isfinite(self.bandwidth_hz) or self.bandwidth_hz <= 0.0:
            raise ValueError("BANDWIDTH_MHZ must be a positive finite value.")
        if self.num_resource_blocks <= 0 or self.subcarriers_per_rb <= 0:
            raise ValueError("NUM_RESOURCE_BLOCKS and SUBCARRIERS_PER_RB must be positive.")
        self.noise_dbm = (
            self.thermal_noise_density_dbm_hz
            + 10.0 * np.log10(self.bandwidth_hz)
            + self.noise_figure_db
        )
        self.rsrp_resource_elements = self.num_resource_blocks * self.subcarriers_per_rb

        self.h_bs = float(config.H_BS)
        self.h_ut = float(config.H_UT if uav_ue_altitude_m is None else uav_ue_altitude_m)
        if not np.isfinite(self.h_ut) or self.h_ut <= 0.0:
            raise ValueError("UAV-UE altitude must be a positive finite value.")

        # Statistical urban-building model for UAV-UE to ground-BS links
        self.building_area_ratio = float(config.BUILDING_AREA_RATIO)
        self.building_density_km2 = float(config.BUILDING_DENSITY_KM2)
        self.building_height_rayleigh_m = float(config.BUILDING_HEIGHT_RAYLEIGH_M)
        self.excess_loss_los_db = float(config.EXCESS_LOSS_LOS_DB)
        self.excess_loss_nlos_db = float(config.EXCESS_LOSS_NLOS_DB)
        self.pathloss_mode = str(config.PATHLOSS_MODE).strip().lower()
        self.los_state_correlation_distance_m = float(
            config.LOS_STATE_CORRELATION_DISTANCE_M
        )
        self.los_state_shared_per_site = bool(config.LOS_STATE_SHARED_PER_SITE)
        if self.pathloss_mode not in {'average', 'correlated_state'}:
            raise ValueError("PATHLOSS_MODE must be 'average' or 'correlated_state'.")
        if self.los_state_correlation_distance_m <= 0.0:
            raise ValueError("LOS_STATE_CORRELATION_DISTANCE_M must be positive.")
        if not 0.0 <= self.building_area_ratio <= 1.0:
            raise ValueError("BUILDING_AREA_RATIO must be in [0, 1].")
        if self.building_density_km2 < 0.0:
            raise ValueError("BUILDING_DENSITY_KM2 must be non-negative.")
        if self.building_height_rayleigh_m <= 0.0:
            raise ValueError("BUILDING_HEIGHT_RAYLEIGH_M must be positive.")

        # Ground BS: cố định NLOS

        # UAV BS: luôn LOS (giữ như bản gốc)

        # Shadow fading
        self.sf_sigma = config.SF_SIGMA
        self.sf_decorr = float(max(20.0, 0.25 * self.grid_spacing_m))
        self.sf_cache = {}
        self.los_state_cache = {}
        self.fast_fading_cache = {}

        # Flat small-scale fading: Rician in LoS, Rayleigh in NLoS.
        self.enable_small_scale_fading = bool(config.ENABLE_SMALL_SCALE_FADING)
        self.rician_k_factor_db = float(config.RICIAN_K_FACTOR_DB)
        self.fading_num_sinusoids = int(config.FADING_NUM_SINUSOIDS)
        self.fading_substep_s = float(config.FADING_SUBSTEP_MS) / 1000.0
        self.speed_of_light_m_s = float(config.SPEED_OF_LIGHT_M_S)
        if not np.isfinite(self.rician_k_factor_db):
            raise ValueError("RICIAN_K_FACTOR_DB must be finite.")
        if self.fading_num_sinusoids < 4:
            raise ValueError("FADING_NUM_SINUSOIDS must be at least 4.")
        if self.fading_substep_s <= 0.0 or self.speed_of_light_m_s <= 0.0:
            raise ValueError("Fading time step and speed of light must be positive.")

        # UE
        self.num_ues = int(num_ues)
        if self.train_ready_output and self.num_ues != 1:
            raise ValueError(
                "Train-ready output represents one trajectory and requires num_ues=1."
            )
        self.ue_colors = ['green', 'purple', 'orange', 'cyan', 'magenta', 'yellow', 'black']
        if uav_ue_speed_kmh is None:
            self.ue_speeds = np.random.uniform(20.0, 60.0, self.num_ues)  # km/h
        else:
            speed_kmh = float(uav_ue_speed_kmh)
            if speed_kmh < 0.0:
                raise ValueError("UAV-UE speed must be non-negative.")
            self.ue_speeds = np.full(self.num_ues, speed_kmh, dtype=float)
        self.ue_speeds_ms = self.ue_speeds * (1000.0 / 3600.0)

        self.steps = 6000
        self.time_per_step = float(config.DELTA_T_MS) / 1000.0
        self.pause_s = (0.03 if fast_mode else 0.08)
        self.headless = bool(headless)

        # BS hex layout
        self.scale_factor = 500.0
        self.hexes = build_hex_cover(self.scale_factor, self.width, self.height, self.center, self.rect_xmin, self.rect_xmax, self.rect_ymin, self.rect_ymax)
        self.bs_positions, self.bs_heights, self.bs_ptx = [], [], []
        self.bs_is_uav, self.bs_site_ids, self.bs_sector_ids, self.bs_antennas = [], [], [], []

        # UE state
        self.ue_positions = []
        self.ue_dir = []  # 0:E,1:N,2:W,3:S

        # UI + plot
        self.route_speed_box = None
        self.route_id_box = None
        self.show_route_button = None
        self.clear_route_button = None
        self.route_status_text = None
        self.route_direction_arrows = None
        if self.headless:
            # A standalone Figure has no GUI manager and therefore opens no window.
            self.fig = Figure(figsize=(10, 10))
            self.ax = self.fig.add_axes([0.08, 0.1, 0.84, 0.82])
            self.toggle_ax = self.restart_ax = None
            self.toggle_button = self.restart_button = None
        else:
            plt.ion()
            self.fig = plt.figure(figsize=(10, 10))
            self.ax = self.fig.add_axes([0.08, 0.18, 0.84, 0.74])
            self.toggle_ax = self.fig.add_axes([0.08, 0.02, 0.12, 0.05])
            self.restart_ax = self.fig.add_axes([0.22, 0.02, 0.14, 0.05])
            self.toggle_button = Button(self.toggle_ax, 'Stop', color='lightcoral')
            self.restart_button = Button(self.restart_ax, 'Restart', color='lightgreen')
            self.toggle_button.on_clicked(self.toggle_animation)
            self.restart_button.on_clicked(self.restart_animation)

            route_speed_ax = self.fig.add_axes([0.20, 0.10, 0.12, 0.045])
            route_id_ax = self.fig.add_axes([0.48, 0.10, 0.10, 0.045])
            show_route_ax = self.fig.add_axes([0.64, 0.10, 0.13, 0.045])
            clear_route_ax = self.fig.add_axes([0.79, 0.10, 0.13, 0.045])
            self.route_speed_box = TextBox(
                route_speed_ax, 'Speed (km/h)', initial='30'
            )
            self.route_id_box = TextBox(route_id_ax, 'Route ID', initial='0')
            self.show_route_button = Button(
                show_route_ax, 'Show Route', color='lightskyblue'
            )
            self.clear_route_button = Button(
                clear_route_ax, 'Clear Route', color='lightgray'
            )
            self.show_route_button.on_clicked(self.show_route_from_inputs)
            self.clear_route_button.on_clicked(self.clear_route)
            self.route_speed_box.on_submit(self.show_route_from_inputs)
            self.route_id_box.on_submit(self.show_route_from_inputs)
            self.route_status_text = self.fig.text(
                0.40, 0.045, 'Route Viewer: mobility only; no RSRP/SINR calculation.',
                fontsize=9, va='center'
            )
        self.animation_running = True
        self.current_frame = -1

        self.setup_plot()
        self.setup_ues()
        self.logger.setup_log()

    def setup_plot(self):
        # Roads (grid)
        for x in self.x_lines:
            self.ax.plot([x, x], [self.rect_ymin, self.rect_ymax], 'gray', linestyle='-', alpha=0.35, linewidth=1.0, zorder=1)
        for y in self.y_lines:
            self.ax.plot([self.rect_xmin, self.rect_xmax], [y, y], 'gray', linestyle='-', alpha=0.35, linewidth=1.0, zorder=1)

        # HCN boundary
        self.ax.add_patch(Rectangle((self.rect_xmin, self.rect_ymin), self.width, self.height,
                                    linewidth=2, edgecolor='red', facecolor='none',
                                    linestyle='--', zorder=5))

        # BS from hex grid
        self.bs_positions.clear()
        self.bs_heights.clear()
        self.bs_ptx.clear()
        self.bs_is_uav.clear()
        self.bs_site_ids.clear()
        self.bs_sector_ids.clear()
        self.bs_antennas.clear()
        bs_colors = ['red', 'green', 'blue', 'cyan', 'magenta', 'yellow', 'black', 'orange', 'purple', 'brown']
        for site_idx, hex_obj in enumerate(self.hexes):
            x, y = axial_to_pixel(hex_obj.q, hex_obj.r, self.scale_factor)
            cx, cy = x + self.center, y + self.center
            for sector_idx, bearing_deg in enumerate(self.bs_sector_bearings_deg):
                self.bs_positions.append((cx, cy))
                self.bs_heights.append(self.h_bs)
                self.bs_ptx.append(self.ptx)
                self.bs_is_uav.append(False)
                self.bs_site_ids.append(site_idx)
                self.bs_sector_ids.append(sector_idx)
                self.bs_antennas.append(ThreeGPPPanelArray(
                    bearing_deg=bearing_deg,
                    **self.bs_antenna_kwargs,
                ))
            self.ax.plot(cx, cy, marker='^', color=bs_colors[site_idx % len(bs_colors)],
                         markersize=5, zorder=6)

            # viền hex (đứt)
            size = self.scale_factor
            angles = [60 * i for i in range(7)]
            xs = [cx + size * math.cos(math.radians(a)) for a in angles]
            ys = [cy + size * math.sin(math.radians(a)) for a in angles]
            self.ax.plot(xs, ys, color='black', linestyle='--', linewidth=1.0, alpha=0.6, zorder=2)


        # --- UAV BS cover (optional): marker 'v' đỏ ---
        if self.add_uav_cover:
            R = float(self.uav_radius_m)
            dx = math.sqrt(3.0) * R
            dy = 1.5 * R
            y = self.rect_ymin + R
            row = 0
            while y <= self.rect_ymax - R:
                x = self.rect_xmin + R + (dx / 2.0 if (row % 2 == 1) else 0.0)
                while x <= self.rect_xmax - R:
                    self.bs_positions.append((x, y))
                    self.bs_heights.append(float(self.uav_altitude_m))
                    self.bs_ptx.append(float(self.uav_ptx_dbm))
                    self.bs_is_uav.append(True)
                    self.ax.plot(x, y, marker='v', color='tab:red', markersize=6, zorder=7)

                    # Viền vùng phủ UAV (xanh nhạt, liền)
                    size = self.scale_factor
                    angles = [60 * i for i in range(7)]
                    xs = [x + size * math.cos(math.radians(a)) for a in angles]
                    ys = [y + size * math.sin(math.radians(a)) for a in angles]
                    self.ax.plot(xs, ys, color='skyblue', linestyle='-', linewidth=1.2, alpha=0.9, zorder=2)
                    x += dx
                y += dy
                row += 1

        # UE artists
        self.ue_points = []
        for i in range(self.num_ues):
            color = self.ue_colors[i % len(self.ue_colors)]
            p, = self.ax.plot([], [], 'o', color=color, markersize=7, zorder=7)
            self.ue_points.append(p)

        # Route Viewer artists are independent from the live UE/radio artists.
        self.route_line, = self.ax.plot(
            [], [], color='tab:blue', linewidth=2.2, alpha=0.9, zorder=8
        )
        self.route_start, = self.ax.plot(
            [], [], marker='o', color='tab:green', markersize=8,
            markeredgecolor='black', linestyle='None', zorder=10
        )
        self.route_end, = self.ax.plot(
            [], [], marker='X', color='tab:red', markersize=9,
            markeredgecolor='black', linestyle='None', zorder=10
        )

        self.time_text = self.ax.text(self.rect_xmin + 20, self.rect_ymax + 20, '', fontsize=10, zorder=8)

        margin_m = max(25.0, 0.05 * max(self.width, self.height))
        self.ax.set_xlim(self.rect_xmin - margin_m, self.rect_xmax + margin_m)
        self.ax.set_ylim(self.rect_ymin - margin_m, self.rect_ymax + margin_m)
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.set_xlabel('Khoảng cách (m)')
        self.ax.set_ylabel('Khoảng cách (m)')
        self.ax.set_title(
            f'UAV-UE grid flight | h={self.h_ut:g} m | Ground BS | {int(self.width)}x{int(self.height)} m | grid_spacing={int(self.grid_spacing_m)} m'
        )
        if not self.headless:
            plt.draw()

    def setup_ues(self):
        self.ue_positions.clear()
        self.ue_dir.clear()

        # Spawn tại giao điểm ngẫu nhiên
        xs = self.x_lines
        ys = self.y_lines
        for _ in range(self.num_ues):
            x = float(np.random.choice(xs))
            y = float(np.random.choice(ys))
            self.ue_positions.append([x, y])
            allowed = self.mobility.allowed_dirs_at_intersection(x, y)
            self.ue_dir.append(int(np.random.choice(allowed)) if allowed else int(np.random.randint(0, 4)))

    def reconstruct_route(self, speed_kmh: float, route_id: int):
        """Recreate one seeded grid route without invoking radio or logging."""
        return reconstruct_seeded_grid_route(
            speed_kmh=speed_kmh,
            route_id=route_id,
            x_lines=self.x_lines,
            y_lines=self.y_lines,
            rect_xmin=self.rect_xmin,
            rect_xmax=self.rect_xmax,
            rect_ymin=self.rect_ymin,
            rect_ymax=self.rect_ymax,
            steps=self.steps,
            time_per_step_s=self.time_per_step,
        )

    def _set_route_direction_arrows(self, points: np.ndarray):
        if self.route_direction_arrows is not None:
            self.route_direction_arrows.remove()
            self.route_direction_arrows = None

        if len(points) < 2:
            return

        sample_count = min(18, len(points) - 1)
        sample_idx = np.linspace(
            0, len(points) - 2, num=sample_count, dtype=int
        )
        vectors = points[sample_idx + 1] - points[sample_idx]
        magnitudes = np.linalg.norm(vectors, axis=1)
        moving = magnitudes > 1e-12
        if not np.any(moving):
            return

        sample_idx = sample_idx[moving]
        vectors = vectors[moving]
        magnitudes = magnitudes[moving]
        arrow_length_m = max(10.0, min(self.width, self.height) * 0.025)
        vectors = vectors / magnitudes[:, None] * arrow_length_m
        self.route_direction_arrows = self.ax.quiver(
            points[sample_idx, 0], points[sample_idx, 1],
            vectors[:, 0], vectors[:, 1],
            angles='xy', scale_units='xy', scale=1.0,
            color='tab:blue', width=0.0035,
            headwidth=4.0, headlength=5.0, zorder=9
        )

    def show_route(self, speed_kmh: float, route_id: int):
        """Draw a complete mobility-only route on the existing map."""
        if self.headless:
            raise RuntimeError('Route Viewer requires headless=False.')

        points, seed = self.reconstruct_route(speed_kmh, route_id)
        self.route_line.set_data(points[:, 0], points[:, 1])
        self.route_start.set_data([points[0, 0]], [points[0, 1]])
        self.route_end.set_data([points[-1, 0]], [points[-1, 1]])
        self._set_route_direction_arrows(points)

        margin_m = max(25.0, 0.05 * max(self.width, self.height))
        self.ax.set_xlim(self.rect_xmin - margin_m, self.rect_xmax + margin_m)
        self.ax.set_ylim(self.rect_ymin - margin_m, self.rect_ymax + margin_m)
        self.ax.set_aspect('equal', adjustable='box')

        duration_s = self.steps * self.time_per_step
        distance_m = float(speed_kmh) * (1000.0 / 3600.0) * duration_s
        if self.route_status_text is not None:
            self.route_status_text.set_color('black')
            self.route_status_text.set_text(
                f'Route {int(route_id)} | seed={seed} | {float(speed_kmh):g} km/h '
                f'| {duration_s:g} s | {distance_m:.1f} m | mobility only'
            )
        self.fig.canvas.draw_idle()
        return points

    def show_route_from_inputs(self, event=None):
        if self.headless:
            return None
        try:
            speed_kmh = float(self.route_speed_box.text.strip())
            route_id_text = self.route_id_box.text.strip()
            route_id = int(route_id_text)
            if str(route_id) != route_id_text and route_id_text not in {
                f'+{route_id}', f'{route_id}.0'
            }:
                raise ValueError('Route ID must be an integer.')
            return self.show_route(speed_kmh, route_id)
        except (TypeError, ValueError) as exc:
            if self.route_status_text is not None:
                self.route_status_text.set_text(f'Route Viewer error: {exc}')
                self.route_status_text.set_color('tab:red')
            self.fig.canvas.draw_idle()
            return None

    def clear_route(self, event=None):
        if self.headless:
            return
        self.route_line.set_data([], [])
        self.route_start.set_data([], [])
        self.route_end.set_data([], [])
        if self.route_direction_arrows is not None:
            self.route_direction_arrows.remove()
            self.route_direction_arrows = None
        if self.route_status_text is not None:
            self.route_status_text.set_color('black')
            self.route_status_text.set_text(
                'Route Viewer cleared; mobility only; no RSRP/SINR calculation.'
            )
        self.fig.canvas.draw_idle()

    def run_route_viewer(self, speed_kmh: float = 30.0, route_id: int = 0):
        """Open the GUI as a dedicated viewer and never run the radio loop."""
        if self.headless:
            raise RuntimeError('Route Viewer requires headless=False.')
        self.route_speed_box.set_val(f'{float(speed_kmh):g}')
        self.route_id_box.set_val(str(int(route_id)))
        self.show_route(speed_kmh, route_id)
        plt.show(block=True)

    def toggle_animation(self, event):
        self.animation_running = not getattr(self, 'animation_running', False)
        if self.animation_running:
            self.toggle_button.label.set_text('Stop')
            self.run_animation()
        else:
            self.toggle_button.label.set_text('Continue')
        self.toggle_button.ax.figure.canvas.draw()

    def restart_animation(self, event):
        self.animation_running = True
        self.current_frame = 0

        self.sf_cache = {}
        self.los_state_cache = {}
        self.fast_fading_cache = {}

        self.logger.setup_log()
        self.setup_ues()

        self.toggle_button.label.set_text('Stop')
        self.toggle_button.color = 'lightcoral'
        self.toggle_button.ax.figure.canvas.draw()
        self.run_animation()

    def update(self, frame: int):
        if not getattr(self, 'animation_running', True):
            return

        self.current_frame = frame
        self.logger.log_step(frame)

        for ue_idx in range(self.num_ues):
            base_d = float(self.ue_speeds_ms[ue_idx] * self.time_per_step)
            d = base_d

            self.mobility.step_manhattan_on_grid(ue_idx, d)

            x, y = self.ue_positions[ue_idx]
            if not self.headless:
                self.ue_points[ue_idx].set_data([x], [y])

            rsrp_dbm, sinr_db = self.radio_model.calculate_all_bs_measurements(ue_idx)
            self.logger.log_all_bs_data(ue_idx, rsrp_dbm, sinr_db)

        if not self.headless:
            self.time_text.set_text(f'Step: {frame}')
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

    def run_animation(self):
        self.animation_running = True
        for frame in range(self.steps):
            if not self.animation_running:
                break
            self.update(frame)
            if not self.headless:
                plt.pause(self.pause_s)
        try:
            return self.logger.save_data_to_csv()
        except Exception as e:
            if self.headless:
                raise
            print('CSV save failed:', e)
            return []
