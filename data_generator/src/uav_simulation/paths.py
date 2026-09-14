"""Simulator artifact locations relative to the HandoverOptimDRL root.

Run commands from the principal project root. The data_generator component
contains source code; shared data and training results live in the parent.
"""

from pathlib import Path

DATA_DIR = Path("data/UAV_single_runs")
DATA_100M_DIR = Path("data/UAV_data_100m")
DATA_300M_DIR = Path("data/UAV_data_300m")
DATA_GRID100_H100_DIR = Path("data/UAV_data_100m_grid100m")
DATA_GRID100_H100_V2_DIR = Path("data/UAV_data_100m_grid100m_v2")
FIGURES_DIR = Path("figures/uav_simulation")
VALIDATION_DIR = Path("results/channel_validation")
