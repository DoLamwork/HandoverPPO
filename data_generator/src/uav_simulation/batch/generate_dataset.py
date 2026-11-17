"""Generate seeded single-UAV datasets for the integrated PPO workflow."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
import math
from pathlib import Path

from uav_simulation import config
from uav_simulation.manifest import (
    build_generation_manifest,
    ensure_generation_manifest,
    train_ready_csv_is_complete,
    write_train_ready_dataset_metadata,
)


P0_CANONICAL_PROFILE = "p0_canonical_v1"
P1_DENSE_URBAN_PROFILE = "p1_dense_urban_v1"
P2_HIGH_MOBILITY_PROFILE = "p2_high_mobility_v1"
P3_FAST_CHANNEL_PROFILE = "p3_fast_channel_v1"
P4_COMBINED_HARSH_PROFILE = "p4_combined_harsh_v1"
P0_CANONICAL_PARAMETERS = {
    "steps": 6000,
    "altitude_m": 100.0,
    "grid_spacing_m": 100.0,
    "width_m": 2000.0,
    "height_m": 1667.0,
}
P0_CANONICAL_RADIO = {
    "carrier_hz": 2.1e9,
    "bandwidth_hz": 10e6,
    "delta_t_ms": 10.0,
}
P0_CANONICAL_URBAN = {
    "building_area_ratio": 0.3,
    "building_density_km2": 500.0,
    "building_height_rayleigh_m": 15.0,
}
P1_DENSE_URBAN = {
    "building_area_ratio": 0.5,
    "building_density_km2": 300.0,
    "building_height_rayleigh_m": 20.0,
}
BASE_CHANNEL = {
    "los_state_correlation_distance_m": 50.0,
    "rician_k_factor_db": 10.0,
}
FAST_CHANNEL = {
    "los_state_correlation_distance_m": 25.0,
    "rician_k_factor_db": 6.0,
}
STANDARD_SPEEDS_KMH = (30, 50, 70, 90, 120, 150)
HIGH_MOBILITY_SPEEDS_KMH = (90, 120, 150, 180, 210, 240)

LOCKED_PROFILE_PARAMETERS = {
    P0_CANONICAL_PROFILE: P0_CANONICAL_PARAMETERS,
    P1_DENSE_URBAN_PROFILE: P0_CANONICAL_PARAMETERS,
    P2_HIGH_MOBILITY_PROFILE: P0_CANONICAL_PARAMETERS,
    P3_FAST_CHANNEL_PROFILE: P0_CANONICAL_PARAMETERS,
    P4_COMBINED_HARSH_PROFILE: P0_CANONICAL_PARAMETERS,
}
LOCKED_PROFILE_URBAN = {
    P0_CANONICAL_PROFILE: P0_CANONICAL_URBAN,
    P1_DENSE_URBAN_PROFILE: P1_DENSE_URBAN,
    P2_HIGH_MOBILITY_PROFILE: P0_CANONICAL_URBAN,
    P3_FAST_CHANNEL_PROFILE: P0_CANONICAL_URBAN,
    P4_COMBINED_HARSH_PROFILE: P1_DENSE_URBAN,
}
LOCKED_PROFILE_CHANNEL = {
    P0_CANONICAL_PROFILE: BASE_CHANNEL,
    P1_DENSE_URBAN_PROFILE: BASE_CHANNEL,
    P2_HIGH_MOBILITY_PROFILE: BASE_CHANNEL,
    P3_FAST_CHANNEL_PROFILE: FAST_CHANNEL,
    P4_COMBINED_HARSH_PROFILE: FAST_CHANNEL,
}
LOCKED_PROFILE_SPEEDS = {
    P2_HIGH_MOBILITY_PROFILE: HIGH_MOBILITY_SPEEDS_KMH,
    P3_FAST_CHANNEL_PROFILE: STANDARD_SPEEDS_KMH,
    P4_COMBINED_HARSH_PROFILE: HIGH_MOBILITY_SPEEDS_KMH,
}
_URBAN_CONFIG_NAMES = {
    "building_area_ratio": "BUILDING_AREA_RATIO",
    "building_density_km2": "BUILDING_DENSITY_KM2",
    "building_height_rayleigh_m": "BUILDING_HEIGHT_RAYLEIGH_M",
}
_CHANNEL_CONFIG_NAMES = {
    "los_state_correlation_distance_m": "LOS_STATE_CORRELATION_DISTANCE_M",
    "rician_k_factor_db": "RICIAN_K_FACTOR_DB",
}


def _validate_locked_profile_contract(parameters, parser):
    profile = parameters["profile"]
    mismatches = []
    for name, expected in LOCKED_PROFILE_PARAMETERS[profile].items():
        actual = float(parameters[name])
        if not math.isclose(actual, float(expected), rel_tol=0.0, abs_tol=1e-9):
            mismatches.append(f"{name}={actual:g} (expected {expected:g})")

    runtime = {
        "carrier_hz": float(config.FC) * 1e9,
        "bandwidth_hz": float(config.BANDWIDTH_MHZ) * 1e6,
        "delta_t_ms": float(config.DELTA_T_MS),
    }
    for name, expected in P0_CANONICAL_RADIO.items():
        actual = runtime[name]
        if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-6):
            mismatches.append(f"{name}={actual:g} (expected {expected:g})")

    expected_speeds = LOCKED_PROFILE_SPEEDS.get(profile)
    if expected_speeds is not None:
        actual_speeds = tuple(sorted(set(parameters["speeds_kmh"])))
        if actual_speeds != tuple(expected_speeds):
            mismatches.append(
                f"speeds_kmh={list(actual_speeds)} "
                f"(expected {list(expected_speeds)})"
            )

    if mismatches:
        parser.error(
            f"{profile} contract mismatch: " + "; ".join(mismatches)
        )


@contextmanager
def _locked_profile_runtime(profile):
    urban = LOCKED_PROFILE_URBAN.get(profile)
    channel = LOCKED_PROFILE_CHANNEL.get(profile)
    if urban is None and channel is None:
        yield
        return
    overrides = {}
    if urban is not None:
        overrides.update({
            _URBAN_CONFIG_NAMES[name]: value for name, value in urban.items()
        })
    if channel is not None:
        overrides.update({
            _CHANNEL_CONFIG_NAMES[name]: value for name, value in channel.items()
        })
    original = {name: getattr(config, name) for name in overrides}
    try:
        for config_name, value in overrides.items():
            setattr(config, config_name, value)
        yield
    finally:
        for config_name, value in original.items():
            setattr(config, config_name, value)


def _generate_route(parameters, speed, route_id, output_dir):
    with _locked_profile_runtime(parameters["profile"]):
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
    parser.add_argument(
        "--profile",
        choices=[
            "custom",
            P0_CANONICAL_PROFILE,
            P1_DENSE_URBAN_PROFILE,
            P2_HIGH_MOBILITY_PROFILE,
            P3_FAST_CHANNEL_PROFILE,
            P4_COMBINED_HARSH_PROFILE,
        ],
        default="custom",
        help="Lock and record a named generation profile (default: custom).",
    )
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
        "profile": args.profile,
        "speeds_kmh": sorted(set(args.speeds)), "routes_per_speed": args.routes_per_speed,
        "start_route_id": args.start_route_id, "steps": args.steps,
        "altitude_m": args.altitude, "grid_spacing_m": args.grid_spacing,
        "width_m": args.width, "height_m": args.height,
    }
    if args.profile in LOCKED_PROFILE_PARAMETERS:
        _validate_locked_profile_contract(parameters, parser)
        parameters["urban_profile"] = dict(LOCKED_PROFILE_URBAN[args.profile])
        parameters["channel_profile"] = dict(LOCKED_PROFILE_CHANNEL[args.profile])
    with _locked_profile_runtime(args.profile):
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
