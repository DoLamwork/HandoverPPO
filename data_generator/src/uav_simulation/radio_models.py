import math
import numpy as np

class RadioModel:
    def __init__(self, sim):
        self.sim = sim

    def calculate_distance_idx(self, ue_x, ue_y, bs_idx: int):
        bs_x, bs_y = self.sim.bs_positions[bs_idx]
        d2 = float(np.hypot(ue_x - bs_x, ue_y - bs_y))
        dz = float(self.sim.bs_heights[bs_idx] - self.sim.h_ut)
        return max(float(np.hypot(d2, dz)), 1.0)

    def calculate_los_probability_idx(self, ue_x, ue_y, bs_idx: int):
        """Return the statistical LoS probability for a UAV-UE to ground-BS link."""
        bs_x, bs_y = self.sim.bs_positions[bs_idx]
        horizontal_distance_m = float(np.hypot(ue_x - bs_x, ue_y - bs_y))
        horizontal_distance_km = horizontal_distance_m / 1000.0

        expected_crossings = horizontal_distance_km * np.sqrt(
            self.sim.building_area_ratio * self.sim.building_density_km2
        )
        num_buildings = max(0, int(np.floor(expected_crossings)))
        if num_buildings == 0:
            return 1.0

        h_uav = float(self.sim.h_ut)
        h_bs = float(self.sim.bs_heights[bs_idx])
        omega_m = float(self.sim.building_height_rayleigh_m)
        log_p_los = 0.0

        for n in range(num_buildings):
            h_ray = h_uav - ((n + 0.5) / num_buildings) * (h_uav - h_bs)
            exponent = -(max(h_ray, 0.0) ** 2) / (2.0 * omega_m ** 2)
            clear_probability = float(-np.expm1(exponent))
            if clear_probability <= 0.0:
                return 0.0
            log_p_los += float(np.log(clear_probability))

        return float(np.clip(np.exp(log_p_los), 0.0, 1.0))

    def _large_scale_cache_key(self, ue_idx: int, bs_idx: int):
        if self.sim.los_state_shared_per_site and hasattr(self.sim, 'bs_site_ids'):
            return ue_idx, int(self.sim.bs_site_ids[bs_idx])
        return ue_idx, bs_idx

    def calculate_path_loss_components_idx(self, ue_x, ue_y, bs_idx: int):
        """Return LoS/NLoS branch losses and statistical LoS probability."""
        d3 = self.calculate_distance_idx(ue_x, ue_y, bs_idx)
        fspl_db = 32.4 + 20.0 * np.log10(self.sim.fc) + 20.0 * np.log10(d3)
        p_los = self.calculate_los_probability_idx(ue_x, ue_y, bs_idx)
        pl_los_db = fspl_db + self.sim.excess_loss_los_db
        pl_nlos_db = fspl_db + self.sim.excess_loss_nlos_db
        return float(pl_los_db), float(pl_nlos_db), float(p_los)

    def update_correlated_los_state(self, ue_idx, bs_idx, p_los, ue_pos):
        """Return a spatially correlated LoS/NLoS realization for one link/site."""
        p_los = float(np.clip(p_los, 0.0, 1.0))
        key = self._large_scale_cache_key(ue_idx, bs_idx)
        state = self.sim.los_state_cache.get(key)

        if state is None:
            latent_gaussian = float(self.sim.rng.normal())
        else:
            delta_m = float(np.hypot(
                float(ue_pos[0]) - float(state['x']),
                float(ue_pos[1]) - float(state['y']),
            ))
            if delta_m <= 1e-12:
                return bool(state['is_los'])
            rho = float(np.exp(
                -delta_m / self.sim.los_state_correlation_distance_m
            ))
            innovation = float(self.sim.rng.normal())
            latent_gaussian = (
                rho * float(state['z'])
                + math.sqrt(max(0.0, 1.0 - rho ** 2)) * innovation
            )

        uniform_value = 0.5 * (
            1.0 + math.erf(latent_gaussian / math.sqrt(2.0))
        )
        is_los = bool(uniform_value < p_los)
        self.sim.los_state_cache[key] = {
            'x': float(ue_pos[0]),
            'y': float(ue_pos[1]),
            'z': float(latent_gaussian),
            'u': float(uniform_value),
            'p_los': p_los,
            'is_los': is_los,
        }
        return is_los

    def calculate_path_loss_idx(self, ue_x, ue_y, bs_idx: int, ue_idx: int):
        """Return realized pathloss, LoS probability, and the branch state."""
        pl_los_db, pl_nlos_db, p_los = self.calculate_path_loss_components_idx(
            ue_x, ue_y, bs_idx
        )
        if self.sim.pathloss_mode == 'correlated_state':
            is_los = self.update_correlated_los_state(
                ue_idx, bs_idx, p_los, (ue_x, ue_y)
            )
            path_loss_db = pl_los_db if is_los else pl_nlos_db
            return float(path_loss_db), float(p_los), bool(is_los)

        pl_avg_db = p_los * pl_los_db + (1.0 - p_los) * pl_nlos_db
        return float(pl_avg_db), float(p_los), None

    def shadow_fading(self, ue_idx, bs_idx, p_los, ue_pos, is_los=None):
        p_los = float(np.clip(p_los, 0.0, 1.0))
        sigma_los = float(self.sim.sf_sigma['LOS'])
        sigma_nlos = float(self.sim.sf_sigma['NLOS'])
        if is_los is None:
            sigma = float(np.sqrt(
                p_los * sigma_los ** 2 + (1.0 - p_los) * sigma_nlos ** 2
            ))
            branch = 'average'
        else:
            branch = 'LOS' if is_los else 'NLOS'
            sigma = sigma_los if is_los else sigma_nlos

        key = self._large_scale_cache_key(ue_idx, bs_idx)
        state = self.sim.sf_cache.get(key)
        if state is None or state.get('branch') != branch:
            val = float(self.sim.rng.normal(0.0, sigma))
            val = float(np.clip(val, -3 * sigma, 3 * sigma))
            self.sim.sf_cache[key] = {
                'x': float(ue_pos[0]),
                'y': float(ue_pos[1]),
                'val': val,
                'branch': branch,
            }
            return val

        delta = float(np.hypot(
            float(ue_pos[0]) - float(state['x']),
            float(ue_pos[1]) - float(state['y']),
        ))
        if delta <= 1e-12:
            return float(state['val'])

        rho = 0.0 if self.sim.sf_decorr <= 0 else float(np.exp(-delta / self.sim.sf_decorr))
        innov = float(self.sim.rng.normal(0.0, sigma))
        newval = rho * float(state['val']) + (np.sqrt(max(0.0, 1.0 - rho ** 2)) * innov)
        newval = float(np.clip(newval, -3 * sigma, 3 * sigma))
        self.sim.sf_cache[key] = {
            'x': float(ue_pos[0]),
            'y': float(ue_pos[1]),
            'val': newval,
            'branch': branch,
        }
        return newval

    def _ue_velocity_vector(self, ue_idx: int):
        """Return the current horizontal UAV velocity in GCS coordinates."""
        direction_vectors = (
            (1.0, 0.0),   # east
            (0.0, 1.0),   # north
            (-1.0, 0.0),  # west
            (0.0, -1.0),  # south
        )
        direction = int(self.sim.ue_dir[ue_idx])
        unit_x, unit_y = direction_vectors[direction]
        speed_m_s = float(self.sim.ue_speeds_ms[ue_idx])
        return speed_m_s * unit_x, speed_m_s * unit_y

    def calculate_doppler_hz(self, ue_idx: int, bs_idx: int, ue_pos):
        """Return direct-path and maximum Doppler shifts for one UAV link."""
        velocity_x, velocity_y = self._ue_velocity_vector(ue_idx)
        carrier_hz = float(self.sim.fc) * 1e9
        doppler_scale = carrier_hz / float(self.sim.speed_of_light_m_s)
        max_doppler_hz = doppler_scale * float(np.hypot(velocity_x, velocity_y))

        bs_x, bs_y = self.sim.bs_positions[bs_idx]
        dx = float(ue_pos[0]) - float(bs_x)
        dy = float(ue_pos[1]) - float(bs_y)
        dz = float(self.sim.h_ut) - float(self.sim.bs_heights[bs_idx])
        distance_3d = max(float(np.sqrt(dx * dx + dy * dy + dz * dz)), 1e-12)
        radial_velocity_m_s = (velocity_x * dx + velocity_y * dy) / distance_3d
        direct_doppler_hz = doppler_scale * radial_velocity_m_s
        return float(direct_doppler_hz), float(max_doppler_hz)

    def small_scale_fading_db(self, ue_idx, bs_idx, is_los, ue_pos):
        """Return a step-averaged Rician/Rayleigh gain with Doppler."""
        if not self.sim.enable_small_scale_fading or is_los is None:
            return 0.0

        key = (int(ue_idx), int(bs_idx))
        epoch = int(getattr(self.sim, 'current_frame', -1))
        branch = 'LOS' if bool(is_los) else 'NLOS'
        state = self.sim.fast_fading_cache.get(key)

        if state is not None:
            same_position = (
                abs(float(ue_pos[0]) - float(state['x'])) <= 1e-12
                and abs(float(ue_pos[1]) - float(state['y'])) <= 1e-12
            )
            if (
                int(state['epoch']) == epoch
                and state['branch'] == branch
                and same_position
            ):
                return float(state['gain_db'])

        num_waves = int(self.sim.fading_num_sinusoids)
        if state is None:
            angular_offset = float(self.sim.rng.random())
            scatterer_angles = (
                2.0 * np.pi * (np.arange(num_waves) + angular_offset) / num_waves
            )
            diffuse_phases = self.sim.rng.uniform(0.0, 2.0 * np.pi, num_waves)
            direct_phase = float(self.sim.rng.uniform(0.0, 2.0 * np.pi))
            previous_epoch = epoch - 1
        else:
            scatterer_angles = np.asarray(state['scatterer_angles'], dtype=float)
            diffuse_phases = np.asarray(state['diffuse_phases'], dtype=float)
            direct_phase = float(state['direct_phase'])
            previous_epoch = int(state['epoch'])

        velocity_x, velocity_y = self._ue_velocity_vector(ue_idx)
        carrier_hz = float(self.sim.fc) * 1e9
        doppler_scale = carrier_hz / float(self.sim.speed_of_light_m_s)
        diffuse_doppler_hz = doppler_scale * (
            velocity_x * np.cos(scatterer_angles)
            + velocity_y * np.sin(scatterer_angles)
        )
        direct_doppler_hz, max_doppler_hz = self.calculate_doppler_hz(
            ue_idx, bs_idx, ue_pos
        )

        # If a link was not evaluated in an intermediate frame, advance its
        # phases to the beginning of the current measurement interval.
        skipped_steps = max(0, epoch - previous_epoch - 1)
        skipped_duration_s = skipped_steps * float(self.sim.time_per_step)
        if skipped_duration_s > 0.0:
            diffuse_phases = diffuse_phases + (
                2.0 * np.pi * diffuse_doppler_hz * skipped_duration_s
            )
            direct_phase += 2.0 * np.pi * direct_doppler_hz * skipped_duration_s

        num_samples = max(1, int(np.ceil(
            float(self.sim.time_per_step) / float(self.sim.fading_substep_s)
        )))
        sample_dt_s = float(self.sim.time_per_step) / num_samples
        sample_numbers = np.arange(1, num_samples + 1, dtype=float)
        diffuse_phase_samples = (
            diffuse_phases[None, :]
            + sample_numbers[:, None]
            * (2.0 * np.pi * diffuse_doppler_hz[None, :] * sample_dt_s)
        )
        diffuse_channel = (
            np.exp(1j * diffuse_phase_samples).sum(axis=1) / np.sqrt(num_waves)
        )

        if bool(is_los):
            k_linear = 10.0 ** (float(self.sim.rician_k_factor_db) / 10.0)
            direct_phase_samples = (
                direct_phase
                + sample_numbers * 2.0 * np.pi * direct_doppler_hz * sample_dt_s
            )
            channel = (
                np.sqrt(k_linear / (k_linear + 1.0))
                * np.exp(1j * direct_phase_samples)
                + np.sqrt(1.0 / (k_linear + 1.0)) * diffuse_channel
            )
            direct_phase = float(direct_phase_samples[-1])
        else:
            channel = diffuse_channel
            direct_phase += 2.0 * np.pi * direct_doppler_hz * float(
                self.sim.time_per_step
            )

        mean_power_gain = float(np.mean(np.abs(channel) ** 2))
        gain_db = float(10.0 * np.log10(max(mean_power_gain, 1e-12)))
        self.sim.fast_fading_cache[key] = {
            'x': float(ue_pos[0]),
            'y': float(ue_pos[1]),
            'epoch': epoch,
            'branch': branch,
            'gain_db': gain_db,
            'mean_power_gain': mean_power_gain,
            'scatterer_angles': scatterer_angles,
            'diffuse_phases': np.mod(diffuse_phase_samples[-1], 2.0 * np.pi),
            'direct_phase': float(np.mod(direct_phase, 2.0 * np.pi)),
            'direct_doppler_hz': direct_doppler_hz,
            'max_doppler_hz': max_doppler_hz,
        }
        return gain_db

    def calculate_bs_antenna_gain_idx(self, ue_x, ue_y, bs_idx: int):
        """Return the BS transmit gain toward the UAV-UE in the GCS."""
        if not self.sim.use_3gpp_3d_antenna:
            return float(self.sim.gtx)
        bs_x, bs_y = self.sim.bs_positions[bs_idx]
        dx, dy = float(ue_x - bs_x), float(ue_y - bs_y)
        dz = float(self.sim.h_ut - self.sim.bs_heights[bs_idx])
        distance_3d = max(float(np.sqrt(dx * dx + dy * dy + dz * dz)), 1e-12)
        theta_gcs_deg = float(np.rad2deg(np.arccos(np.clip(dz / distance_3d, -1.0, 1.0))))
        phi_gcs_deg = float(np.rad2deg(np.arctan2(dy, dx)))
        return float(self.sim.bs_antennas[bs_idx].power_gain_db_gcs(
            theta_gcs_deg, phi_gcs_deg
        ))

    def calculate_received_power_idx(self, bs_idx: int, path_loss_db: float, ue_x, ue_y):
        """Return total received power over the configured channel bandwidth."""
        ptx = float(self.sim.bs_ptx[bs_idx])
        gtx = self.calculate_bs_antenna_gain_idx(ue_x, ue_y, bs_idx)
        return ptx + gtx + self.sim.grx - float(path_loss_db)

    def calculate_rsrp_dbm(self, total_received_power_dbm):
        """Convert full-channel received power to per-reference-RE RSRP."""
        return (
            np.asarray(total_received_power_dbm, dtype=float)
            - 10.0 * np.log10(float(self.sim.rsrp_resource_elements))
            + self.sim.reference_signal_power_offset_db
        )

    def calculate_all_bs_measurements(self, ue_idx: int):
        """Return RSRP (dBm) and co-channel SINR (dB) for every ground BS."""
        ue_x, ue_y = self.sim.ue_positions[ue_idx]
        num_bs = len(self.sim.bs_positions)
        total_received_power_dbm = np.empty(num_bs, dtype=float)

        for bs_idx in range(num_bs):
            path_loss_db, p_los, is_los = self.calculate_path_loss_idx(
                ue_x, ue_y, bs_idx, ue_idx
            )
            shadow_db = self.shadow_fading(
                ue_idx, bs_idx, p_los, (ue_x, ue_y), is_los
            )
            fading_db = self.small_scale_fading_db(
                ue_idx, bs_idx, is_los, (ue_x, ue_y)
            )
            total_received_power_dbm[bs_idx] = fading_db + self.calculate_received_power_idx(
                bs_idx, path_loss_db + shadow_db, ue_x, ue_y
            )

        if num_bs == 0:
            return total_received_power_dbm, np.empty(0, dtype=float)

        # SINR is evaluated over the full co-channel bandwidth. Therefore the
        # signal/interference powers remain full-channel powers and the noise
        # uses that same bandwidth.
        received_mw = np.power(10.0, total_received_power_dbm / 10.0)
        noise_mw = 10.0 ** (self.sim.noise_dbm / 10.0)
        total_received_mw = float(np.sum(received_mw))
        interference_mw = np.maximum(total_received_mw - received_mw, 0.0)
        sinr_linear = received_mw / (interference_mw + noise_mw)
        sinr_db = 10.0 * np.log10(sinr_linear)
        rsrp_dbm = self.calculate_rsrp_dbm(total_received_power_dbm)
        return rsrp_dbm, sinr_db
