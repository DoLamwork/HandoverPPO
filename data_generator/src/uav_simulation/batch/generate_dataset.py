"""Generate seeded single-UAV datasets for the integrated PPO workflow."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import math
from pathlib import Path

from uav_simulation.manifest import (
    build_generation_manifest,
    ensure_generation_manifest,
    train_ready_csv_is_complete,
    write_train_ready_dataset_metadata,
)


def _generate_route(parameters, speed, route_id, output_dir):
    from uav_simulation.simulation import CellularNetworkReceivedPower
    sim = CellularNetworkReceivedPower(
        num_ues=1, rect_len_m=parameters["width_m"], rect_wid_m=parameters["height_m"],
        grid_spacing_m=parameters["grid_spacing_m"], uav_ue_altitude_m=parameters["altitude_m"],
        uav_ue_speed_kmh=speed, seed=speed * 1000 + route_id,
        headless=True,
        output_run_idx=route_id, output_data_dir=str(output_dir),
        train_ready_output=True,
    )
    sim.steps = parameters["steps"]
    return [str(path) for path in sim.run_animation()]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/uav_generated"))
    parser.add_argument("--speeds", nargs="+", type=int, default=[30, 50, 70, 90, 120, 150])
    parser.add_argument("--routes-per-speed", type=int, default=60)
    parser.add_argument("--start-route-id", type=int, default=0)
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--altitude", type=float, default=100.0)
    parser.add_argument("--grid-spacing", type=float, default=500.0)
    parser.add_argument("--width", type=float, default=2000.0)
    parser.add_argument("--height", type=float, default=1667.0)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    if min(args.routes_per_speed, args.steps, args.workers) <= 0 or args.start_route_id < 0 or min(args.speeds) <= 0:
        parser.error("speeds, routes, steps and workers must be positive; start-route-id must be non-negative")
    if any(not math.isfinite(value) or value <= 0 for value in (args.altitude, args.grid_spacing, args.width, args.height)):
        parser.error("map dimensions, altitude and grid spacing must be positive finite values")
    if args.grid_spacing < 10:
        parser.error("grid-spacing must be at least 10 m, matching the simulator")
    parameters = {
        "speeds_kmh": sorted(set(args.speeds)), "routes_per_speed": args.routes_per_speed,
        "start_route_id": args.start_route_id, "steps": args.steps,
        "altitude_m": args.altitude, "grid_spacing_m": args.grid_spacing,
        "width_m": args.width, "height_m": args.height,
    }
    manifest = build_generation_manifest(
        parameters,
        entrypoint="uav_simulation.batch.generate_dataset",
    )
    directory = args.output_dir.resolve()
    ensure_generation_manifest(directory, manifest)
    pending = []
    expected_keys = set()
    for speed in parameters["speeds_kmh"]:
        for route_id in range(args.start_route_id, args.start_route_id + args.routes_per_speed):
            expected_keys.add((speed, route_id))
            paths = [directory / f"{kind}_{speed}kmh_{route_id}.csv" for kind in ("rsrp", "sinr")]
            if any(path.exists() for path in paths):
                if all(train_ready_csv_is_complete(path, args.steps) for path in paths):
                    continue
                raise FileExistsError(f"Incomplete route {speed}/{route_id}; move its partial files before resuming.")
            pending.append((speed, route_id))
    if args.workers == 1:
        for speed, route_id in pending:
            _generate_route(parameters, speed, route_id, directory)
    elif pending:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(_generate_route, parameters, speed, route_id, directory) for speed, route_id in pending]
            for future in as_completed(futures):
                future.result()
    metadata_path = write_train_ready_dataset_metadata(
        directory,
        manifest,
        expected_keys=expected_keys,
    )
    print(f"Dataset ready: {directory}; generated {len(pending)} routes.")
    print(f"Training metadata: {metadata_path}")
    return 0
