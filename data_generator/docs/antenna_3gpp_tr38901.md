# 3GPP TR 38.901 3D antenna model

The implementation in `antenna_model.py` follows ETSI TR 138 901 / 3GPP TR
38.901 V18.0.0, Clauses 7.1.3, 7.3.1, 7.3.2, and Table 7.3-1. Angles use the
3GPP convention: `theta` is the zenith angle and `phi` is azimuth.

Official source:
https://www.etsi.org/deliver/etsi_tr/138900_138999/138901/18.00.00_60/tr_138901v180000p.pdf

## LCS to GCS transformation ? Clause 7.1.3

The antenna orientation is defined by bearing `alpha`, mechanical downtilt
`beta`, and slant `gamma`. The forward rotation from LCS to GCS is:

```text
R = Rz(alpha) Ry(beta) Rx(gamma)
```

For a GCS ray with angles `(theta, phi)`:

```text
r_GCS = [sin(theta)cos(phi), sin(theta)sin(phi), cos(theta)]^T
r_LCS = R^T r_GCS
theta' = acos(r_LCS,z)
phi'   = atan2(r_LCS,y, r_LCS,x)
```

This vector implementation is equivalent to equations (7.1-7) and (7.1-8)
and remains numerically stable near the spherical-coordinate poles.

Polarized fields are transformed through the Cartesian field vector:

```text
E_LCS = F_theta' e_theta' + F_phi' e_phi'
E_GCS = R E_LCS
F_theta = E_GCS . e_theta
F_phi   = E_GCS . e_phi
```

This is equivalent to the polarization-basis rotation of equations (7.1-9)
through (7.1-11), without explicitly evaluating the displacement angle `psi`.

## Single-element pattern ? Table 7.3-1

The vertical and horizontal cuts in dB are:

```text
A_EV(theta') = -min(12((theta' - 90 deg)/theta_3dB)^2, SLA_V)
A_EH(phi')   = -min(12(phi'/phi_3dB)^2, A_max)
```

The complete element gain is:

```text
A_E(theta', phi') = G_E,max - min(-(A_EV + A_EH), A_max)
```

3GPP defaults used by the project:

```text
theta_3dB = 65 deg
phi_3dB   = 65 deg
SLA_V     = 30 dB
A_max     = 30 dB
G_E,max   = 8 dBi
```

## Polarized antenna modelling ? Clause 7.3.2

Both standard alternatives are implemented.

Model-1 rotates a pure theta-polarized element coordinate system by the
polarization slant angle `zeta`, evaluates the element pattern in that rotated
system, and transforms its vector field back to the antenna LCS. This produces
the angle-dependent polarization mapping in equation (7.3-3).

Model-2 uses angle-independent polarization in the antenna LCS:

```text
F_theta' = sqrt(A_E,linear) cos(zeta)
F_phi'   = sqrt(A_E,linear) sin(zeta)
```

and satisfies equation (7.3-2):

```text
A_E,linear = |F_theta'|^2 + |F_phi'|^2
```

The default port uses Model-2 with `zeta = +45 deg`. A second port can be
constructed with `zeta = -45 deg` for a cross-polarized pair.

## Antenna port mapping ? Clause 7.3.1

For `M` vertical elements, electrical steering zenith angle `theta_etilt`, and
vertical spacing `d_V`, the normalized port weights are:

```text
w_m = 1/sqrt(M) exp(-j 2 pi m (d_V/lambda) cos(theta_etilt))
```

where the implementation uses zero-based `m = 0, ..., M-1`. A positive
electrical downtilt `delta` is converted to:

```text
theta_etilt = 90 deg + delta
```

For the implemented single rectangular panel with `M` vertical and `N`
horizontal elements, uniform horizontal port mapping gives:

```text
AF(theta', phi') = sum_m sum_n w_m/sqrt(N)
    exp(j 2 pi [m d_V cos(theta')
              + n d_H sin(theta')sin(phi')]/lambda)
```

The full polarized port field is the element field multiplied by `AF`. The
project default is `(Mg, Ng, M, N, P) = (1, 1, 8, 4, 2)`, with half-wavelength
element spacing and 10-degree electrical downtilt.

## Network integration

Each ground site is represented by three co-located sectors with GCS bearings
`0`, `120`, and `240` degrees. For each BS-to-UAV ray, `radio_models.py`
computes GCS zenith/azimuth from the 3D geometry and uses the sector's complete
panel gain instead of the former constant `GTX`.

The current link budget uses total polarized port power. Polarization mismatch,
cross-polarization ratio, and a polarized UAV receive array require a future
dual-ended channel model and are not approximated here.
