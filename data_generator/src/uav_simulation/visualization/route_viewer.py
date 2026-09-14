import argparse

from uav_simulation.simulation import CellularNetworkReceivedPower


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Reconstruct and display one seeded UAV-UE route without radio.'
    )
    parser.add_argument('--speed', type=float, default=30.0)
    parser.add_argument('--route-id', type=int, default=0)
    parser.add_argument('--width', type=float, default=2000.0)
    parser.add_argument('--height', type=float, default=1667.0)
    parser.add_argument('--grid-spacing', type=float, default=500.0)
    parser.add_argument('--altitude', type=float, default=100.0)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    sim = CellularNetworkReceivedPower(
        num_ues=1,
        uav_ue_speed_kmh=args.speed,
        uav_ue_altitude_m=args.altitude,
        rect_len_m=args.width,
        rect_wid_m=args.height,
        grid_spacing_m=args.grid_spacing,
        headless=False,
    )
    sim.run_route_viewer(args.speed, args.route_id)


if __name__ == '__main__':
    main()
