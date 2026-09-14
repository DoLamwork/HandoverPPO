# Spatially correlated LoS/NLoS realization

The project supports two large-scale pathloss modes through `PATHLOSS_MODE` in
`config.py`:

```text
average
correlated_state
```

`average` retains the original weighted pathloss:

```text
PL_avg = P_LOS PL_LOS + (1 - P_LOS) PL_NLOS
```

`correlated_state`, which is now the default, generates one spatially
persistent LoS/NLoS realization for every UAV-to-ground-site link.

## Latent Gaussian process

For horizontal UAV displacement `delta_d`, the correlation coefficient is:

```text
rho = exp(-delta_d / d_corr)
```

The latent standard Gaussian variable is updated as:

```text
z_t = rho z_(t-1) + sqrt(1 - rho^2) epsilon_t
epsilon_t ~ N(0,1)
```

It is transformed to a uniform random variable with the standard-normal CDF:

```text
u_t = Phi(z_t)
```

The realized link state is:

```text
LoS  if u_t < P_LOS(x_t, y_t)
NLoS otherwise
```

Because `u_t` is marginally uniform, the initial LoS probability remains
`P_LOS`. Because `z_t` is correlated by traveled distance, nearby samples do
not make independent blockage decisions.

The default `d_corr` is 50 m. With the project's 10 ms sample interval, the
per-step correlations are approximately 0.9983 at 30 km/h and 0.9934 at
120 km/h.

## Site-level consistency

The cache key is `(ue_idx, site_idx)`, not `(ue_idx, sector_idx)`. Therefore the
three co-located sectors at bearings 0, 120, and 240 degrees share:

- `P_LOS`;
- the realized LoS/NLoS state;
- the selected pathloss branch; and
- shadow fading.

They still have different received powers because their 3D antenna gains are
different.

Repeated radio calculations at the same UAV position return the cached state
and shadow value without consuming another random sample. This is important
because serving-cell selection and all-BS logging both evaluate the links in
the same simulation step.

## Realized pathloss and shadow fading

For a correlated-state realization:

```text
if state == LoS:
    PL = PL_LOS
    sigma_SF = sigma_LOS
else:
    PL = PL_NLOS
    sigma_SF = sigma_NLOS
```

Shadow correlation now uses Euclidean horizontal displacement:

```text
delta_d = hypot(x_t - x_(t-1), y_t - y_(t-1))
```

No artificial turn penalty is applied. When the realized branch changes, a
new shadow value is drawn with the standard deviation of the new branch.

## Reproducibility and limitation

The LoS process uses the simulation's dedicated NumPy random generator. Equal
simulation seeds reproduce the same state sequence.

This is a trajectory-correlated stochastic model, not a fixed 2D obstruction
map. If a UAV later revisits the same coordinate, the state is not guaranteed
to match an earlier visit. A pre-generated two-dimensional Gaussian field is
the appropriate later extension when repeat-location consistency or correlated
blockage between multiple UAVs is required.

The approach follows the spatial-consistency principle and correlation-distance
concept in 3GPP TR 38.901 Clause 7.6.3 while retaining the project's ITU-style
statistical building-based `P_LOS` calculation.
