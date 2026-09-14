"""
Global Configuration variables for Cellular Network Simulation
"""

import ho_optim_drl.config as _protocol_config

# Radio Params
PTX = 35.0       # Power Transmit (dBm)
GTX = 2.0        # Gain Transmit
GRX = 2.0        # Gain Receive

# 3GPP TR 38.901 single-panel sector antenna (Clauses 7.1.3 and 7.3)
USE_3GPP_3D_ANTENNA = True
BS_SECTOR_BEARINGS_DEG = (0.0, 120.0, 240.0)
BS_ANTENNA_NUM_VERTICAL = 8
BS_ANTENNA_NUM_HORIZONTAL = 4
BS_ANTENNA_SPACING_VERTICAL_LAMBDA = 0.5
BS_ANTENNA_SPACING_HORIZONTAL_LAMBDA = 0.5
BS_ANTENNA_MECHANICAL_DOWNTILT_DEG = 0.0
BS_ANTENNA_MECHANICAL_SLANT_DEG = 0.0
BS_ANTENNA_ELECTRICAL_DOWNTILT_DEG = 10.0
BS_ANTENNA_POLARIZATION_SLANT_DEG = 45.0
BS_ANTENNA_POLARIZATION_MODEL = 2
BS_ANTENNA_MAX_ELEMENT_GAIN_DBI = 8.0
BS_ANTENNA_THETA_3DB_DEG = 65.0
BS_ANTENNA_PHI_3DB_DEG = 65.0
BS_ANTENNA_VERTICAL_SIDE_LOBE_ATTENUATION_DB = 30.0
BS_ANTENNA_FRONT_BACK_ATTENUATION_DB = 30.0

FC = _protocol_config.DEFAULT_CARRIER_HZ / 1e9  # GHz
DELTA_T_MS = _protocol_config.DEFAULT_DELTA_T_MS

H_BS = 25.0     # Height of BS_ground (m)
H_UT = 100.0    # Fixed altitude of UAV-UE (m)

# Air-to-ground urban environment (ITU-style statistical building model)
BUILDING_AREA_RATIO = 0.3          # zeta: built-up land fraction [0, 1]
BUILDING_DENSITY_KM2 = 500.0       # xi: mean buildings per km^2
BUILDING_HEIGHT_RAYLEIGH_M = 15.0  # Omega: Rayleigh building-height scale (m)

# Large-scale pathloss realization. "average" retains the old weighted model;
# "correlated_state" selects a spatially persistent LoS or NLoS branch.
PATHLOSS_MODE = "correlated_state"
LOS_STATE_CORRELATION_DISTANCE_M = 50.0
LOS_STATE_SHARED_PER_SITE = True

# Excess loss added to free-space path loss (dB)
EXCESS_LOSS_LOS_DB = 1.0
EXCESS_LOSS_NLOS_DB = 20.0

# Shadow Fading
SF_SIGMA = {'LOS': 4.6, 'NLOS': 10.0}

# Flat small-scale fading with Doppler. The LoS branch uses Rician fading and
# the NLoS branch uses Rayleigh fading. Power is averaged over 1 ms internal
# samples before each radio measurement is logged.
ENABLE_SMALL_SCALE_FADING = True
RICIAN_K_FACTOR_DB = 10.0
FADING_NUM_SINUSOIDS = 16
FADING_SUBSTEP_MS = 1.0
SPEED_OF_LIGHT_M_S = 299792458.0

# Channel bandwidth and NR resource grid. PTX is the total BS transmit power
# distributed uniformly over the active subcarriers in this basic model.
BANDWIDTH_MHZ = _protocol_config.DEFAULT_BANDWIDTH_HZ / 1e6
NUM_RESOURCE_BLOCKS = 51
SUBCARRIERS_PER_RB = 12

# Receiver noise: N = N0 + 10*log10(BW) + noise figure.
THERMAL_NOISE_DENSITY_DBM_HZ = -174.0
NOISE_FIGURE_DB = 7.0

# Optional power offset for the reference-signal REs relative to data REs.
REFERENCE_SIGNAL_POWER_OFFSET_DB = 0.0
