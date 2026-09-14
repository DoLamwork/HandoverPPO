import unittest
from types import SimpleNamespace

import numpy as np

from uav_simulation.radio_models import RadioModel


def make_radio(seed=123, mode='correlated_state'):
    sim = SimpleNamespace(
        bs_positions=[(0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (1000.0, 0.0)],
        bs_heights=[25.0, 25.0, 25.0, 25.0],
        bs_site_ids=[0, 0, 0, 1],
        h_ut=100.0,
        fc=4.0,
        building_area_ratio=0.3,
        building_density_km2=500.0,
        building_height_rayleigh_m=15.0,
        excess_loss_los_db=1.0,
        excess_loss_nlos_db=20.0,
        pathloss_mode=mode,
        los_state_correlation_distance_m=50.0,
        los_state_shared_per_site=True,
        los_state_cache={},
        sf_cache={},
        sf_sigma={'LOS': 4.6, 'NLOS': 10.0},
        sf_decorr=50.0,
        rng=np.random.default_rng(seed),
    )
    return RadioModel(sim), sim


class CorrelatedLosStateTests(unittest.TestCase):
    def test_three_sectors_at_same_site_share_state(self):
        radio, sim = make_radio()
        state_0 = radio.update_correlated_los_state(0, 0, 0.5, (100.0, 20.0))
        cached_z = sim.los_state_cache[(0, 0)]['z']
        state_1 = radio.update_correlated_los_state(0, 1, 0.5, (100.0, 20.0))
        state_2 = radio.update_correlated_los_state(0, 2, 0.5, (100.0, 20.0))
        self.assertEqual(state_0, state_1)
        self.assertEqual(state_0, state_2)
        self.assertEqual(len(sim.los_state_cache), 1)
        self.assertEqual(sim.los_state_cache[(0, 0)]['z'], cached_z)

    def test_no_motion_does_not_update_latent_variable(self):
        radio, sim = make_radio()
        first = radio.update_correlated_los_state(0, 0, 0.4, (25.0, 30.0))
        snapshot = dict(sim.los_state_cache[(0, 0)])
        second = radio.update_correlated_los_state(0, 0, 0.4, (25.0, 30.0))
        self.assertEqual(first, second)
        self.assertEqual(sim.los_state_cache[(0, 0)], snapshot)

    def test_seed_reproduces_same_spatial_sequence(self):
        radio_a, _ = make_radio(seed=77)
        radio_b, _ = make_radio(seed=77)
        sequence_a, sequence_b = [], []
        for x in np.linspace(0.0, 500.0, 501):
            sequence_a.append(
                radio_a.update_correlated_los_state(0, 0, 0.55, (x, 0.0))
            )
            sequence_b.append(
                radio_b.update_correlated_los_state(0, 0, 0.55, (x, 0.0))
            )
        self.assertEqual(sequence_a, sequence_b)

    def test_small_spatial_steps_have_persistent_state(self):
        radio, _ = make_radio(seed=8)
        sequence = [
            radio.update_correlated_los_state(0, 0, 0.5, (step * 0.1, 0.0))
            for step in range(2000)
        ]
        transitions = sum(a != b for a, b in zip(sequence, sequence[1:]))
        self.assertLess(transitions, 100)

    def test_initial_realizations_preserve_target_probability(self):
        radio, _ = make_radio(seed=999)
        samples = [
            radio.update_correlated_los_state(ue_idx, 0, 0.3, (0.0, 0.0))
            for ue_idx in range(5000)
        ]
        self.assertAlmostEqual(float(np.mean(samples)), 0.3, delta=0.02)

    def test_realized_pathloss_selects_exact_branch(self):
        radio, _ = make_radio(seed=2)
        ue_x, ue_y, bs_idx, ue_idx = 750.0, 0.0, 0, 0
        pl_los, pl_nlos, _ = radio.calculate_path_loss_components_idx(
            ue_x, ue_y, bs_idx
        )
        pathloss, _, is_los = radio.calculate_path_loss_idx(
            ue_x, ue_y, bs_idx, ue_idx
        )
        self.assertEqual(pathloss, pl_los if is_los else pl_nlos)

    def test_average_mode_retains_weighted_pathloss(self):
        radio, sim = make_radio(mode='average')
        ue_x, ue_y, bs_idx = 750.0, 0.0, 0
        pl_los, pl_nlos, p_los = radio.calculate_path_loss_components_idx(
            ue_x, ue_y, bs_idx
        )
        pathloss, returned_probability, is_los = radio.calculate_path_loss_idx(
            ue_x, ue_y, bs_idx, 0
        )
        expected = p_los * pl_los + (1.0 - p_los) * pl_nlos
        self.assertAlmostEqual(pathloss, expected)
        self.assertEqual(returned_probability, p_los)
        self.assertIsNone(is_los)
        self.assertEqual(sim.los_state_cache, {})

    def test_shadow_fading_is_shared_by_co_sited_sectors(self):
        radio, sim = make_radio(seed=44)
        value_0 = radio.shadow_fading(0, 0, 0.7, (100.0, 20.0), True)
        value_1 = radio.shadow_fading(0, 1, 0.7, (100.0, 20.0), True)
        value_2 = radio.shadow_fading(0, 2, 0.7, (100.0, 20.0), True)
        self.assertEqual(value_0, value_1)
        self.assertEqual(value_0, value_2)
        self.assertEqual(len(sim.sf_cache), 1)


class FullSimulationIntegrationTests(unittest.TestCase):
    def test_all_sectors_share_one_large_scale_state_per_site(self):
        from uav_simulation.simulation import CellularNetworkReceivedPower

        sim = CellularNetworkReceivedPower(
            num_ues=1,
            rect_len_m=1000.0,
            rect_wid_m=1000.0,
            grid_spacing_m=200.0,
            uav_ue_speed_kmh=30.0,
            headless=True,
            seed=31,
        )
        rsrp_first, sinr_first = sim.radio_model.calculate_all_bs_measurements(0)
        rsrp_second, sinr_second = sim.radio_model.calculate_all_bs_measurements(0)
        self.assertEqual(len(sim.los_state_cache), len(sim.hexes))
        self.assertEqual(len(sim.sf_cache), len(sim.hexes))
        np.testing.assert_array_equal(rsrp_first, rsrp_second)
        np.testing.assert_array_equal(sinr_first, sinr_second)


if __name__ == '__main__':
    unittest.main()
