import unittest

import numpy as np

from uav_simulation.antenna_model import (
    ThreeGPPPanelArray,
    gcs_angles_to_lcs,
    polarized_element_field_lcs,
    rotation_matrix_lcs_to_gcs,
    single_element_gain_db,
    spherical_unit_vector,
    transform_field_lcs_to_gcs,
    vertical_port_weights,
)


class CoordinateTransformTests(unittest.TestCase):
    def test_identity_rotation(self):
        np.testing.assert_allclose(rotation_matrix_lcs_to_gcs(), np.eye(3), atol=1e-14)
        theta_lcs, phi_lcs = gcs_angles_to_lcs(73.0, -41.0)
        self.assertAlmostEqual(float(theta_lcs), 73.0, places=12)
        self.assertAlmostEqual(float(phi_lcs), -41.0, places=12)

    def test_bearing_rotation(self):
        theta_lcs, phi_lcs = gcs_angles_to_lcs(
            90.0, 90.0, bearing_deg=90.0
        )
        self.assertAlmostEqual(float(theta_lcs), 90.0, places=12)
        self.assertAlmostEqual(float(phi_lcs), 0.0, places=12)

    def test_rotation_is_orthogonal(self):
        rotation = rotation_matrix_lcs_to_gcs(37.0, 11.0, -23.0)
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-14)
        self.assertAlmostEqual(float(np.linalg.det(rotation)), 1.0, places=12)

    def test_field_transform_conserves_power(self):
        f_theta_lcs = 1.2 + 0.4j
        f_phi_lcs = -0.3 + 0.8j
        f_theta_gcs, f_phi_gcs = transform_field_lcs_to_gcs(
            f_theta_lcs,
            f_phi_lcs,
            61.0,
            -28.0,
            bearing_deg=31.0,
            downtilt_deg=9.0,
            slant_deg=-17.0,
        )
        power_lcs = abs(f_theta_lcs) ** 2 + abs(f_phi_lcs) ** 2
        power_gcs = abs(f_theta_gcs) ** 2 + abs(f_phi_gcs) ** 2
        self.assertAlmostEqual(float(power_gcs), power_lcs, places=12)


class ElementPatternTests(unittest.TestCase):
    def test_table_7_3_1_reference_directions(self):
        self.assertAlmostEqual(float(single_element_gain_db(90.0, 0.0)), 8.0)
        self.assertAlmostEqual(float(single_element_gain_db(90.0, 180.0)), -22.0)

    def test_model_2_preserves_element_power(self):
        theta, phi = 76.0, 24.0
        f_theta, f_phi = polarized_element_field_lcs(
            theta, phi, polarization_slant_deg=45.0, polarization_model=2
        )
        expected = 10.0 ** (float(single_element_gain_db(theta, phi)) / 10.0)
        self.assertAlmostEqual(float(abs(f_theta) ** 2 + abs(f_phi) ** 2), expected)

    def test_model_1_and_model_2_match_at_boresight(self):
        for model in (1, 2):
            f_theta, f_phi = polarized_element_field_lcs(
                90.0, 0.0, polarization_slant_deg=45.0, polarization_model=model
            )
            expected_power = 10.0 ** (8.0 / 10.0)
            self.assertAlmostEqual(float(abs(f_theta) ** 2 + abs(f_phi) ** 2), expected_power)
            self.assertAlmostEqual(float(abs(f_theta)), float(abs(f_phi)), places=12)


class AntennaPortTests(unittest.TestCase):
    def test_vertical_weights_are_unit_norm(self):
        weights = vertical_port_weights(8, 0.5, 100.0)
        self.assertAlmostEqual(float(np.sum(np.abs(weights) ** 2)), 1.0, places=12)

    def test_array_factor_has_expected_steering_gain(self):
        antenna = ThreeGPPPanelArray(num_vertical=8, num_horizontal=4)
        factor = antenna.array_factor_lcs(
            antenna.electrical_steering_zenith_deg, 0.0
        )
        self.assertAlmostEqual(
            float(abs(factor) ** 2),
            antenna.num_vertical * antenna.num_horizontal,
            places=10,
        )

    def test_oriented_panel_peak_is_in_expected_direction(self):
        antenna = ThreeGPPPanelArray(
            bearing_deg=120.0,
            mechanical_downtilt_deg=5.0,
            electrical_downtilt_deg=10.0,
        )
        local_direction = spherical_unit_vector(
            antenna.electrical_steering_zenith_deg, 0.0
        )
        global_direction = rotation_matrix_lcs_to_gcs(120.0, 5.0, 0.0) @ local_direction
        theta_gcs = np.rad2deg(np.arccos(global_direction[2]))
        phi_gcs = np.rad2deg(np.arctan2(global_direction[1], global_direction[0]))
        gain_db = antenna.power_gain_db_gcs(theta_gcs, phi_gcs)
        expected = (
            single_element_gain_db(antenna.electrical_steering_zenith_deg, 0.0)
            + 10.0 * np.log10(antenna.num_vertical * antenna.num_horizontal)
        )
        self.assertAlmostEqual(float(gain_db), float(expected), places=10)

    def test_vectorized_3d_grid(self):
        antenna = ThreeGPPPanelArray()
        theta, phi = np.meshgrid(
            np.linspace(0.0, 180.0, 19),
            np.linspace(-180.0, 180.0, 37),
            indexing='ij',
        )
        gain = antenna.power_gain_db_gcs(theta, phi)
        self.assertEqual(gain.shape, theta.shape)
        self.assertTrue(np.all(np.isfinite(gain)))


if __name__ == '__main__':
    unittest.main()
