"""Generate the 300 m UAV-altitude RSRP/SINR dataset batch."""

from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from uav_simulation.manifest import (
    build_generation_manifest,
    ensure_generation_manifest,
    train_ready_csv_is_complete,
    write_train_ready_dataset_metadata,
)
from uav_simulation.simulation import CellularNetworkReceivedPower


SPEEDS_KMH = (30, 50, 70, 90, 120, 150)
RUNS_PER_SPEED = 30
FILES_PER_RUN = 2
EXPECTED_ROWS_PER_FILE = 6000
DATA_DIR = Path("data/uav_300m_generated")


def route_paths(speed_kmh: int, run_idx: int, data_dir: str | Path = DATA_DIR):
    directory = Path(data_dir)
    return (
        directory / f"rsrp_{speed_kmh}kmh_{run_idx}.csv",
        directory / f"sinr_{speed_kmh}kmh_{run_idx}.csv",
    )


def count_data_rows(path: Path):
    with path.open("r", encoding="utf-8-sig") as csv_file:
        return sum(1 for _ in csv_file) - 1


def route_is_complete(speed_kmh: int, run_idx: int, data_dir: str | Path = DATA_DIR):
    paths = route_paths(speed_kmh, run_idx, data_dir)
    if not all(path.exists() for path in paths):
        return False
    return all(train_ready_csv_is_complete(path, EXPECTED_ROWS_PER_FILE) for path in paths)


def remove_incomplete_route(speed_kmh: int, run_idx: int, data_dir: str | Path = DATA_DIR):
    paths = route_paths(speed_kmh, run_idx, data_dir)
    if route_is_complete(speed_kmh, run_idx, data_dir):
        return
    for path in paths:
        if path.exists():
            path.unlink()


def generate_route(speed_kmh: int, run_idx: int, data_dir: str | Path = DATA_DIR):
    directory = Path(data_dir)
    started = time.perf_counter()
    sim = CellularNetworkReceivedPower(
        num_ues=1,
        rect_len_m=2000.0,
        rect_wid_m=1667.0,
        grid_spacing_m=500.0,
        uav_ue_speed_kmh=float(speed_kmh),
        uav_ue_altitude_m=300.0,
        headless=True,
        seed=(speed_kmh * 1000) + run_idx,
        output_run_idx=run_idx,
        output_data_dir=str(directory),
        train_ready_output=True,
    )
    paths = tuple(sim.run_animation())
    expected_paths = route_paths(speed_kmh, run_idx, directory)
    if set(map(Path, paths)) != set(expected_paths):
        raise RuntimeError(
            f"Route {speed_kmh}/{run_idx} wrote unexpected files: {paths}"
        )
    for path in expected_paths:
        rows = count_data_rows(path)
        if rows != EXPECTED_ROWS_PER_FILE:
            raise RuntimeError(
                f"{path} contains {rows} rows, expected {EXPECTED_ROWS_PER_FILE}."
            )
    return {
        "speed_kmh": speed_kmh,
        "run_idx": run_idx,
        "seconds": time.perf_counter() - started,
        "paths": tuple(str(path) for path in expected_paths),
    }


def expected_csv_paths(data_dir: str | Path = DATA_DIR):
    return {
        path
        for speed_kmh in SPEEDS_KMH
        for run_idx in range(RUNS_PER_SPEED)
        for path in route_paths(speed_kmh, run_idx, data_dir)
    }


def verify_complete_batch(data_dir: str | Path = DATA_DIR):
    directory = Path(data_dir)
    expected = expected_csv_paths(directory)
    actual = set(directory.glob("*.csv"))
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing or unexpected:
        raise RuntimeError(
            f"Batch file-set mismatch: missing={missing}, unexpected={unexpected}"
        )
    for path in sorted(expected):
        if not train_ready_csv_is_complete(path, EXPECTED_ROWS_PER_FILE):
            raise RuntimeError(
                f"{path} does not satisfy the train-ready schema with "
                f"{EXPECTED_ROWS_PER_FILE} rows."
            )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workers",
        type=int,
        default=min(4, os.cpu_count() or 1),
        help="Parallel route workers (default: up to 4).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DATA_DIR,
        help="New or exactly matching manifested train-ready dataset directory.",
    )
    args = parser.parse_args(argv)
    if args.workers <= 0:
        raise ValueError("workers must be positive")

    data_dir = args.output_dir.resolve()
    parameters = {
        "profile": "h300_grid500",
        "speeds_kmh": list(SPEEDS_KMH),
        "routes_per_speed": RUNS_PER_SPEED,
        "start_route_id": 0,
        "steps": EXPECTED_ROWS_PER_FILE,
        "altitude_m": 300.0,
        "grid_spacing_m": 500.0,
        "width_m": 2000.0,
        "height_m": 1667.0,
    }
    manifest = build_generation_manifest(
        parameters,
        entrypoint="uav_simulation.batch.generate_batch_300m",
    )
    ensure_generation_manifest(data_dir, manifest)
    total_routes = len(SPEEDS_KMH) * RUNS_PER_SPEED
    pending = []
    completed = 0
    for speed_kmh in SPEEDS_KMH:
        for run_idx in range(RUNS_PER_SPEED):
            if route_is_complete(speed_kmh, run_idx, data_dir):
                completed += 1
                print(
                    f"Resume: speed={speed_kmh} km/h, route={run_idx}/29 already complete",
                    flush=True,
                )
            else:
                remove_incomplete_route(speed_kmh, run_idx, data_dir)
                pending.append((speed_kmh, run_idx))

    print(
        f"Batch plan: {len(SPEEDS_KMH) * RUNS_PER_SPEED} routes, "
        f"completed={completed}, pending={len(pending)}, workers={args.workers}",
        flush=True,
    )
    batch_started = time.perf_counter()
    if pending:
        with ProcessPoolExecutor(max_workers=min(args.workers, len(pending))) as executor:
            futures = {
                executor.submit(generate_route, speed_kmh, run_idx, data_dir): (speed_kmh, run_idx)
                for speed_kmh, run_idx in pending
            }
            for future in as_completed(futures):
                result = future.result()
                completed += 1
                print(
                    f"Completed {completed}/{total_routes}: "
                    f"speed={result['speed_kmh']} km/h, "
                    f"route={result['run_idx']}/29, elapsed={result['seconds']:.1f}s",
                    flush=True,
                )

    verify_complete_batch(data_dir)
    metadata_path = write_train_ready_dataset_metadata(
        data_dir,
        manifest,
        expected_keys={
            (speed_kmh, run_idx)
            for speed_kmh in SPEEDS_KMH
            for run_idx in range(RUNS_PER_SPEED)
        },
    )
    elapsed = time.perf_counter() - batch_started
    print(
        f"Batch complete: {total_routes} routes, "
        f"{total_routes * FILES_PER_RUN} CSV files, 60 files per speed, "
        f"6000 rows per file, elapsed={elapsed:.1f}s",
        flush=True,
    )
    print(f"Training metadata: {metadata_path}", flush=True)


if __name__ == "__main__":
    main()
