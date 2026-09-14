import unittest
from types import SimpleNamespace

import numpy as np

from uav_simulation.radio_models import RadioModel


def make_fading_radio(
    seed=123,
    speed_kmh=30.0,
    direction=0,
    enabled=True,
    num_ues=1,
):
    speed_m_s = float(speed_kmh) / 3.6
    sim = SimpleNamespace(
        fc=4.0,
        speed_of_light_m_s=299792458.0,
        ue_speeds_ms=np.full(num_ues, speed_m_s, dtype=float),
        ue_dir=[int(direction)] * num_ues,
        bs_positions=[(0.0, 0.0)],
        bs_heights=[25.0],
        h_ut=100.0,
        enable_small_scale_fading=bool(enabled),
        rician_k_factor_db=10.0,
        fading_num_sinusoids=16,
        fading_substep_s=0.001,
        time_per_step=0.010,
        current_frame=0,
        fast_fading_cache={},
        rng=np.random.default_rng(seed),
    )
    return RadioModel(sim), sim


class DopplerTests(unittest.TestCase):
    def test_maximum_doppler_matches_speed_and_carrier(self):
        for speed_kmh in (30.0, 60.0, 90.0, 120.0):
            radio, sim = make_fading_radio(speed_kmh=speed_kmh)
            direct_hz, maximum_hz = radio.calculate_doppler_hz(
                0, 0, (1000.0, 0.0)
            )
            expected = (
                (speed_kmh / 3.6) * sim.fc * 1e9
                / sim.speed_of_light_m_s
            )
            self.assertAlmostEqual(maximum_hz, expected, places=10)
            # The height difference makes the direct projection slightly less
            # than the theoretical horizontal maximum.
            self.assertGreater(direct_hz, 0.99 * maximum_hz)
            self.assertLessEqual(direct_hz, maximum_hz)

    def test_direct_doppler_is_zero_for_tangential_flight(self):
        radio, _ = make_fading_radio(speed_kmh=120.0, direction=1)
        direct_hz, maximum_hz = radio.calculate_doppler_hz(
            0, 0, (1000.0, 0.0)
        )
        self.assertAlmostEqual(direct_hz, 0.0, places=12)
        self.assertGreater(maximum_hz, 400.0)


class SmallScaleFadingTests(unittest.TestCase):
    def test_disabled_model_returns_zero_gain(self):
        radio, sim = make_fading_radio(enabled=False)
        self.assertEqual(
            radio.small_scale_fading_db(0, 0, True, (1000.0, 0.0)),
            0.0,
        )
        self.assertEqual(sim.fast_fading_cache, {})

    def test_average_pathloss_has_no_small_scale_branch(self):
        radio, sim = make_fading_radio()
        self.assertEqual(
            radio.small_scale_fading_db(0, 0, None, (1000.0, 0.0)),
            0.0,
        )
        self.assertEqual(sim.fast_fading_cache, {})

    def test_same_frame_reuses_identical_sample(self):
        radio, sim = make_fading_radio(seed=9)
        first = radio.small_scale_fading_db(0, 0, True, (1000.0, 0.0))
        phases = sim.fast_fading_cache[(0, 0)]['diffuse_phases'].copy()
        second = radio.small_scale_fading_db(0, 0, True, (1000.0, 0.0))
        self.assertEqual(first, second)
        np.testing.assert_array_equal(
            phases, sim.fast_fading_cache[(0, 0)]['diffuse_phases']
        )

    def test_seed_reproduces_time_sequence(self):
        radio_a, sim_a = make_fading_radio(seed=77, speed_kmh=90.0)
        radio_b, sim_b = make_fading_radio(seed=77, speed_kmh=90.0)
        sequence_a, sequence_b = [], []
        for frame in range(100):
            sim_a.current_frame = frame
            sim_b.current_frame = frame
            sequence_a.append(
                radio_a.small_scale_fading_db(0, 0, True, (1000.0, 0.0))
            )
            sequence_b.append(
                radio_b.small_scale_fading_db(0, 0, True, (1000.0, 0.0))
            )
        np.testing.assert_array_equal(sequence_a, sequence_b)

    def test_zero_speed_channel_is_static_but_moving_channel_evolves(self):
        static_radio, static_sim = make_fading_radio(seed=5, speed_kmh=0.0)
        moving_radio, moving_sim = make_fading_radio(seed=5, speed_kmh=120.0)
        static_values, moving_values = [], []
        for frame in range(50):
            static_sim.current_frame = frame
            moving_sim.current_frame = frame
            static_values.append(
                static_radio.small_scale_fading_db(0, 0, False, (1000.0, 0.0))
            )
            moving_values.append(
                moving_radio.small_scale_fading_db(0, 0, False, (1000.0, 0.0))
            )
        self.assertEqual(len(set(static_values)), 1)
        self.assertGreater(float(np.std(moving_values)), 0.05)

    def test_rayleigh_and_rician_have_unit_mean_power(self):
        num_links = 2000
        rayleigh_radio, _ = make_fading_radio(
            seed=100, speed_kmh=0.0, num_ues=num_links
        )
        rician_radio, _ = make_fading_radio(
            seed=200, speed_kmh=0.0, num_ues=num_links
        )
        rayleigh_power = np.array([
            10.0 ** (
                rayleigh_radio.small_scale_fading_db(
                    ue_idx, 0, False, (1000.0, 0.0)
                ) / 10.0
            )
            for ue_idx in range(num_links)
        ])
        rician_power = np.array([
            10.0 ** (
                rician_radio.small_scale_fading_db(
                    ue_idx, 0, True, (1000.0, 0.0)
                ) / 10.0
            )
            for ue_idx in range(num_links)
        ])
        self.assertAlmostEqual(float(np.mean(rayleigh_power)), 1.0, delta=0.08)
        self.assertAlmostEqual(float(np.mean(rician_power)), 1.0, delta=0.05)
        self.assertLess(float(np.var(rician_power)), float(np.var(rayleigh_power)))


class FullSimulationFadingIntegrationTests(unittest.TestCase):
    def test_every_ue_sector_has_one_cached_fading_process(self):
        from uav_simulation.simulation import CellularNetworkReceivedPower

        sim = CellularNetworkReceivedPower(
            num_ues=1,
            rect_len_m=1000.0,
            rect_wid_m=1000.0,
            grid_spacing_m=200.0,
            uav_ue_speed_kmh=120.0,
            headless=True,
            seed=41,
        )
        first_rsrp, first_sinr = sim.radio_model.calculate_all_bs_measurements(0)
        second_rsrp, second_sinr = sim.radio_model.calculate_all_bs_measurements(0)
        self.assertEqual(len(sim.fast_fading_cache), len(sim.bs_positions))
        np.testing.assert_array_equal(first_rsrp, second_rsrp)
        np.testing.assert_array_equal(first_sinr, second_sinr)


if __name__ == '__main__':
    unittest.main()
