"""3GPP TR 38.901 three-dimensional antenna model.

Implements the coordinate-system transformation in Clause 7.1.3, the
single-element pattern in Table 7.3-1, legacy vertical port mapping in Clause
7.3.1, and both polarized-element alternatives in Clause 7.3.2.
"""

from dataclasses import dataclass

import numpy as np


_POWER_FLOOR = 1e-30


def wrap_azimuth_deg(phi_deg):
    """Wrap azimuth to the 3GPP interval [-180, 180)."""
    phi = np.asarray(phi_deg, dtype=float)
    return (phi + 180.0) % 360.0 - 180.0


def rotation_matrix_lcs_to_gcs(
    bearing_deg: float = 0.0,
    downtilt_deg: float = 0.0,
    slant_deg: float = 0.0,
):
    """Return R = Rz(alpha) Ry(beta) Rx(gamma), Clause 7.1.3.

    Positive beta rotates the local +x broadside direction downward, which is
    the 3GPP mechanical-downtilt convention.
    """
    alpha, beta, gamma = np.deg2rad([bearing_deg, downtilt_deg, slant_deg])
    ca, sa = np.cos(alpha), np.sin(alpha)
    cb, sb = np.cos(beta), np.sin(beta)
    cg, sg = np.cos(gamma), np.sin(gamma)

    rz = np.array([[ca, -sa, 0.0], [sa, ca, 0.0], [0.0, 0.0, 1.0]])
    ry = np.array([[cb, 0.0, sb], [0.0, 1.0, 0.0], [-sb, 0.0, cb]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cg, -sg], [0.0, sg, cg]])
    return rz @ ry @ rx


def spherical_unit_vector(theta_deg, phi_deg):
    """Return r-hat for 3GPP zenith theta and azimuth phi angles."""
    theta, phi = np.broadcast_arrays(
        np.deg2rad(np.asarray(theta_deg, dtype=float)),
        np.deg2rad(np.asarray(phi_deg, dtype=float)),
    )
    return np.stack(
        (
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta),
        ),
        axis=-1,
    )


def spherical_basis(theta_deg, phi_deg):
    """Return the theta-hat and phi-hat spherical basis vectors."""
    theta, phi = np.broadcast_arrays(
        np.deg2rad(np.asarray(theta_deg, dtype=float)),
        np.deg2rad(np.asarray(phi_deg, dtype=float)),
    )
    e_theta = np.stack(
        (
            np.cos(theta) * np.cos(phi),
            np.cos(theta) * np.sin(phi),
            -np.sin(theta),
        ),
        axis=-1,
    )
    e_phi = np.stack(
        (-np.sin(phi), np.cos(phi), np.zeros_like(phi)),
        axis=-1,
    )
    return e_theta, e_phi


def gcs_angles_to_lcs(
    theta_gcs_deg,
    phi_gcs_deg,
    bearing_deg: float = 0.0,
    downtilt_deg: float = 0.0,
    slant_deg: float = 0.0,
):
    """Transform a GCS ray direction to LCS spherical angles.

    This is the vector form of equations (7.1-7) and (7.1-8). Using R.T also
    avoids special-case algebra near the spherical-coordinate poles.
    """
    rotation = rotation_matrix_lcs_to_gcs(bearing_deg, downtilt_deg, slant_deg)
    direction_gcs = spherical_unit_vector(theta_gcs_deg, phi_gcs_deg)
    direction_lcs = np.einsum('ij,...j->...i', rotation.T, direction_gcs)
    theta_lcs = np.rad2deg(np.arccos(np.clip(direction_lcs[..., 2], -1.0, 1.0)))
    phi_lcs = np.rad2deg(np.arctan2(direction_lcs[..., 1], direction_lcs[..., 0]))
    return theta_lcs, wrap_azimuth_deg(phi_lcs)


def transform_field_lcs_to_gcs(
    f_theta_lcs,
    f_phi_lcs,
    theta_gcs_deg,
    phi_gcs_deg,
    bearing_deg: float = 0.0,
    downtilt_deg: float = 0.0,
    slant_deg: float = 0.0,
):
    """Transform polarized field components from LCS to GCS.

    Equivalent to the polarization-basis rotation in equations (7.1-9) to
    (7.1-11), implemented through Cartesian vector projection.
    """
    theta_lcs, phi_lcs = gcs_angles_to_lcs(
        theta_gcs_deg,
        phi_gcs_deg,
        bearing_deg,
        downtilt_deg,
        slant_deg,
    )
    e_theta_lcs, e_phi_lcs = spherical_basis(theta_lcs, phi_lcs)
    f_theta_lcs, f_phi_lcs = np.broadcast_arrays(
        np.asarray(f_theta_lcs, dtype=complex),
        np.asarray(f_phi_lcs, dtype=complex),
    )
    field_lcs = (
        f_theta_lcs[..., np.newaxis] * e_theta_lcs
        + f_phi_lcs[..., np.newaxis] * e_phi_lcs
    )

    rotation = rotation_matrix_lcs_to_gcs(bearing_deg, downtilt_deg, slant_deg)
    field_gcs_cartesian = np.einsum('ij,...j->...i', rotation, field_lcs)
    e_theta_gcs, e_phi_gcs = spherical_basis(theta_gcs_deg, phi_gcs_deg)
    f_theta_gcs = np.sum(field_gcs_cartesian * e_theta_gcs, axis=-1)
    f_phi_gcs = np.sum(field_gcs_cartesian * e_phi_gcs, axis=-1)
    return f_theta_gcs, f_phi_gcs


def single_element_cuts_db(
    theta_lcs_deg,
    phi_lcs_deg,
    theta_3db_deg: float = 65.0,
    phi_3db_deg: float = 65.0,
    side_lobe_attenuation_db: float = 30.0,
    front_back_attenuation_db: float = 30.0,
):
    """Return vertical and horizontal cuts from TR 38.901 Table 7.3-1."""
    theta = np.asarray(theta_lcs_deg, dtype=float)
    phi = wrap_azimuth_deg(phi_lcs_deg)
    vertical_db = -np.minimum(
        12.0 * ((theta - 90.0) / theta_3db_deg) ** 2,
        side_lobe_attenuation_db,
    )
    horizontal_db = -np.minimum(
        12.0 * (phi / phi_3db_deg) ** 2,
        front_back_attenuation_db,
    )
    return vertical_db, horizontal_db


def single_element_gain_db(
    theta_lcs_deg,
    phi_lcs_deg,
    max_element_gain_dbi: float = 8.0,
    theta_3db_deg: float = 65.0,
    phi_3db_deg: float = 65.0,
    side_lobe_attenuation_db: float = 30.0,
    front_back_attenuation_db: float = 30.0,
):
    """Return the single-element 3D power gain in dBi, Table 7.3-1."""
    vertical_db, horizontal_db = single_element_cuts_db(
        theta_lcs_deg,
        phi_lcs_deg,
        theta_3db_deg,
        phi_3db_deg,
        side_lobe_attenuation_db,
        front_back_attenuation_db,
    )
    attenuation_db = np.minimum(
        -(vertical_db + horizontal_db),
        front_back_attenuation_db,
    )
    return max_element_gain_dbi - attenuation_db


def polarized_element_field_lcs(
    theta_lcs_deg,
    phi_lcs_deg,
    polarization_slant_deg: float = 45.0,
    polarization_model: int = 2,
    **element_pattern_kwargs,
):
    """Return (F_theta, F_phi) for Clause 7.3.2 Model-1 or Model-2."""
    if polarization_model not in (1, 2):
        raise ValueError("polarization_model must be 1 or 2.")

    if polarization_model == 2:
        gain_db = single_element_gain_db(
            theta_lcs_deg, phi_lcs_deg, **element_pattern_kwargs
        )
        amplitude = np.sqrt(np.power(10.0, gain_db / 10.0))
        zeta = np.deg2rad(polarization_slant_deg)
        return (
            amplitude * np.cos(zeta) + 0j,
            amplitude * np.sin(zeta) + 0j,
        )

    # Model-1 treats the slanted element as a coordinate system rotated by
    # zeta around local broadside. It is pure theta-polarized in that system.
    theta_double, phi_double = gcs_angles_to_lcs(
        theta_lcs_deg,
        phi_lcs_deg,
        slant_deg=polarization_slant_deg,
    )
    gain_db = single_element_gain_db(
        theta_double, phi_double, **element_pattern_kwargs
    )
    amplitude = np.sqrt(np.power(10.0, gain_db / 10.0)) + 0j
    return transform_field_lcs_to_gcs(
        amplitude,
        np.zeros_like(amplitude, dtype=complex),
        theta_lcs_deg,
        phi_lcs_deg,
        slant_deg=polarization_slant_deg,
    )


def vertical_port_weights(
    num_vertical: int,
    spacing_vertical_lambda: float,
    electrical_steering_zenith_deg: float,
):
    """Return normalized Clause 7.3.1 vertical port weights, equation 7.3-1."""
    if num_vertical <= 0:
        raise ValueError("num_vertical must be positive.")
    if spacing_vertical_lambda <= 0.0:
        raise ValueError("spacing_vertical_lambda must be positive.")
    element_index = np.arange(num_vertical, dtype=float)
    theta_etilt = np.deg2rad(electrical_steering_zenith_deg)
    phase = -2j * np.pi * element_index * spacing_vertical_lambda * np.cos(theta_etilt)
    return np.exp(phase) / np.sqrt(float(num_vertical))


@dataclass(frozen=True)
class ThreeGPPPanelArray:
    """Single rectangular 3GPP panel and one polarized antenna port.

    The implemented array tuple is (Mg=1, Ng=1, M=num_vertical,
    N=num_horizontal, P=2). One of the two slanted polarization ports is
    selected by polarization_slant_deg; use +45 and -45 degrees for dual-pol.
    """

    num_vertical: int = 8
    num_horizontal: int = 4
    spacing_vertical_lambda: float = 0.5
    spacing_horizontal_lambda: float = 0.5
    bearing_deg: float = 0.0
    mechanical_downtilt_deg: float = 0.0
    mechanical_slant_deg: float = 0.0
    electrical_downtilt_deg: float = 10.0
    polarization_slant_deg: float = 45.0
    polarization_model: int = 2
    max_element_gain_dbi: float = 8.0
    theta_3db_deg: float = 65.0
    phi_3db_deg: float = 65.0
    side_lobe_attenuation_db: float = 30.0
    front_back_attenuation_db: float = 30.0

    def __post_init__(self):
        if self.num_vertical <= 0 or self.num_horizontal <= 0:
            raise ValueError("Antenna array dimensions must be positive.")
        if self.spacing_vertical_lambda <= 0.0 or self.spacing_horizontal_lambda <= 0.0:
            raise ValueError("Antenna element spacings must be positive.")
        if self.polarization_model not in (1, 2):
            raise ValueError("polarization_model must be 1 or 2.")

    @property
    def electrical_steering_zenith_deg(self):
        return 90.0 + self.electrical_downtilt_deg

    def _element_kwargs(self):
        return {
            'max_element_gain_dbi': self.max_element_gain_dbi,
            'theta_3db_deg': self.theta_3db_deg,
            'phi_3db_deg': self.phi_3db_deg,
            'side_lobe_attenuation_db': self.side_lobe_attenuation_db,
            'front_back_attenuation_db': self.front_back_attenuation_db,
        }

    def array_factor_lcs(self, theta_lcs_deg, phi_lcs_deg):
        """Return the normalized complex M-by-N panel array factor."""
        direction = spherical_unit_vector(theta_lcs_deg, phi_lcs_deg)
        vertical_positions = (
            np.arange(self.num_vertical, dtype=float) * self.spacing_vertical_lambda
        )
        horizontal_positions = (
            np.arange(self.num_horizontal, dtype=float) * self.spacing_horizontal_lambda
        )
        weights_vertical = vertical_port_weights(
            self.num_vertical,
            self.spacing_vertical_lambda,
            self.electrical_steering_zenith_deg,
        )
        weights = np.broadcast_to(
            weights_vertical[:, np.newaxis] / np.sqrt(float(self.num_horizontal)),
            (self.num_vertical, self.num_horizontal),
        )
        sample_shape = direction.shape[:-1]
        expansion = (1,) * len(sample_shape)
        z = vertical_positions.reshape((self.num_vertical, 1) + expansion)
        y = horizontal_positions.reshape((1, self.num_horizontal) + expansion)
        u_z = direction[..., 2].reshape((1, 1) + sample_shape)
        u_y = direction[..., 1].reshape((1, 1) + sample_shape)
        spatial_phase = 2j * np.pi * (
            z * u_z + y * u_y
        )
        phase = np.exp(spatial_phase)
        weight_shape = (self.num_vertical, self.num_horizontal) + (1,) * len(sample_shape)
        return np.sum(weights.reshape(weight_shape) * phase, axis=(0, 1))

    def field_pattern_gcs(self, theta_gcs_deg, phi_gcs_deg):
        """Return complex (F_theta, F_phi) of the complete antenna port."""
        theta_lcs, phi_lcs = gcs_angles_to_lcs(
            theta_gcs_deg,
            phi_gcs_deg,
            self.bearing_deg,
            self.mechanical_downtilt_deg,
            self.mechanical_slant_deg,
        )
        f_theta_lcs, f_phi_lcs = polarized_element_field_lcs(
            theta_lcs,
            phi_lcs,
            self.polarization_slant_deg,
            self.polarization_model,
            **self._element_kwargs(),
        )
        array_factor = self.array_factor_lcs(theta_lcs, phi_lcs)
        return transform_field_lcs_to_gcs(
            f_theta_lcs * array_factor,
            f_phi_lcs * array_factor,
            theta_gcs_deg,
            phi_gcs_deg,
            self.bearing_deg,
            self.mechanical_downtilt_deg,
            self.mechanical_slant_deg,
        )

    def power_gain_linear_gcs(self, theta_gcs_deg, phi_gcs_deg):
        f_theta, f_phi = self.field_pattern_gcs(theta_gcs_deg, phi_gcs_deg)
        return np.abs(f_theta) ** 2 + np.abs(f_phi) ** 2

    def power_gain_db_gcs(self, theta_gcs_deg, phi_gcs_deg):
        power = np.maximum(self.power_gain_linear_gcs(theta_gcs_deg, phi_gcs_deg), _POWER_FLOOR)
        return 10.0 * np.log10(power)
