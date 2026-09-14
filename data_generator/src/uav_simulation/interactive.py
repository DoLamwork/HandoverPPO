import argparse
from pathlib import Path

from uav_simulation.paths import DATA_DIR
from uav_simulation.simulation import CellularNetworkReceivedPower
from uav_simulation.utils import _read_float, _read_int

def main(argv=None):
    parser = argparse.ArgumentParser(description='Run a UAV simulation with interactive parameter prompts.')
    parser.add_argument('--headless', action='store_true', help='Run without opening a simulation window.')
    parser.add_argument('--output-dir', type=Path, default=DATA_DIR, help='Directory for RSRP/SINR CSV files.')
    args = parser.parse_args(argv)
    L = _read_float("Enter RECTANGLE LENGTH (m): ", 2000.0)
    W = _read_float("Enter RECTANGLE WIDTH (m): ", 1667.0)
    num_ues = _read_int("Enter number of UEs: ", 1)
    uav_ue_speed_kmh = max(0.0, _read_float("Enter UAV-UE speed (km/h, default 30): ", 30.0))
    uav_ue_altitude_m = max(1.0, _read_float("Enter UAV-UE altitude (m, default 100): ", 100.0))
    grid_sp = _read_float("Enter road grid spacing (m, default 500): ", 500.0)

    sim = CellularNetworkReceivedPower(
        num_ues=num_ues,
        uav_ue_speed_kmh=uav_ue_speed_kmh,
        uav_ue_altitude_m=uav_ue_altitude_m,
        rect_len_m=L,
        rect_wid_m=W,
        grid_spacing_m=grid_sp,
        headless=args.headless,
        output_data_dir=str(args.output_dir),
        fast_mode=True
    )
    paths = sim.run_animation()
    print(f"Simulation complete. Generated {len(paths)} CSV file(s).")


if __name__ == "__main__":
    main()
