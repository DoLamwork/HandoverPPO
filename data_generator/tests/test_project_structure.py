"""Integration coverage for installed commands, output paths, and batch workers."""

from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager, redirect_stdout
from importlib import import_module
import io
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from uav_simulation.cli import COMMANDS
from uav_simulation.batch.generate_dataset import main as generate_dataset
from uav_simulation.simulation import CellularNetworkReceivedPower


@contextmanager
def working_directory(path):
    previous = Path.cwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(previous)


class InstalledCommandTests(unittest.TestCase):
    def test_help_works_outside_repository_for_every_command(self):
        with TemporaryDirectory() as directory:
            for command in (None, *COMMANDS):
                with self.subTest(command=command):
                    args = [] if command is None else [command]
                    result = subprocess.run(
                        [sys.executable, '-m', 'uav_simulation', *args, '--help'],
                        cwd=directory,
                        env={**os.environ, 'MPLBACKEND': 'Agg'},
                        capture_output=True,
                        text=True,
                        timeout=30,
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn('usage:', result.stdout)
            self.assertEqual(list(Path(directory).iterdir()), [])


class OutputIntegrationTests(unittest.TestCase):
    def test_headless_csv_paths_schema_and_seed_survive_package_move(self):
        with TemporaryDirectory() as directory, working_directory(directory):
            runs = []
            with redirect_stdout(io.StringIO()):
                for _ in range(2):
                    sim = CellularNetworkReceivedPower(
                        num_ues=2,
                        rect_len_m=1000.0,
                        rect_wid_m=1000.0,
                        grid_spacing_m=200.0,
                        uav_ue_speed_kmh=30.0,
                        headless=True,
                        seed=30000,
                    )
                    self.assertFalse(hasattr(sim, 'ue_serving_bs'))
                    self.assertFalse(hasattr(sim.radio_model, 'get_serving_bs'))
                    sim.steps = 4
                    paths = sim.run_animation()
                    self.assertEqual(len(paths), 4)
                    for path in paths:
                        self.assertEqual(path.parent, Path('data/UAV_single_runs'))
                        frame = pd.read_csv(path)
                        self.assertEqual(list(frame['Step']), list(range(4)))
                        self.assertEqual(
                            list(frame.columns),
                            ['Step', *[f'bs{i}' for i in range(len(sim.bs_positions))]],
                        )
                        self.assertTrue(np.isfinite(frame.to_numpy()).all())
                    runs.append(paths)
            self.assertTrue(set(runs[0]).isdisjoint(runs[1]))
            self.assertEqual(len(list(Path('data/UAV_single_runs').glob('*.csv'))), 8)
            for first, second in zip(*runs):
                self.assertEqual(first.read_bytes(), second.read_bytes())

    def test_batch_generator_writes_train_ready_schema_and_metadata(self):
        from ho_optim_drl.data import discover_trace_pairs, load_radio_trace
        from ho_optim_drl.data.project import load_protocol_config

        with TemporaryDirectory() as directory:
            output = Path(directory) / "dataset"
            with redirect_stdout(io.StringIO()):
                result = generate_dataset([
                    "--output-dir", str(output),
                    "--speeds", "30",
                    "--routes-per-speed", "1",
                    "--steps", "4",
                    "--grid-spacing", "200",
                    "--width", "1000",
                    "--height", "1000",
                    "--workers", "1",
                ])
            self.assertEqual(result, 0)
            pairs = discover_trace_pairs(str(output))
            self.assertEqual(len(pairs), 1)
            self.assertEqual(
                (pairs[0].rsrp_file, pairs[0].sinr_file),
                ("rsrp_30kmh_0.csv", "sinr_30kmh_0.csv"),
            )
            frame = pd.read_csv(output / pairs[0].rsrp_file)
            self.assertEqual(list(frame.columns[:2]), ["sample", "time_ms"])
            self.assertEqual(list(frame["sample"]), list(range(4)))
            self.assertEqual(list(frame["time_ms"]), [0, 10, 20, 30])
            self.assertEqual(
                list(frame.columns[2:]),
                [f"bs_{index}" for index in range(len(frame.columns) - 2)],
            )
            protocol = load_protocol_config(output)
            trace = load_radio_trace(
                protocol,
                str(output),
                pairs[0].rsrp_file,
                pairs[0].sinr_file,
            )
            self.assertEqual(trace.num_samples, 4)
            metadata = json.loads(
                (output / "dataset_metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["dataset_format"], "handover_ppo_train_ready_v1")
            self.assertEqual(metadata["trajectory_count"], 1)

    def test_p0_profile_is_recorded_and_rejects_noncanonical_geometry(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "p0"
            with patch.dict(
                "uav_simulation.batch.generate_dataset.P0_CANONICAL_PARAMETERS",
                {"steps": 4},
            ), redirect_stdout(io.StringIO()):
                result = generate_dataset([
                        "--output-dir", str(output),
                        "--profile", "p0_canonical_v1",
                        "--speeds", "30",
                        "--routes-per-speed", "1",
                        "--steps", "4",
                        "--altitude", "100",
                        "--grid-spacing", "100",
                        "--width", "2000",
                        "--height", "1667",
                        "--workers", "1",
                    ])
            self.assertEqual(result, 0)
            metadata = json.loads(
                (output / "dataset_metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                metadata["generation"]["parameters"]["profile"],
                "p0_canonical_v1",
            )

            with patch.dict(
                "uav_simulation.batch.generate_dataset.P0_CANONICAL_PARAMETERS",
                {"steps": 4},
            ), self.assertRaises(SystemExit):
                generate_dataset([
                        "--output-dir", str(Path(directory) / "invalid"),
                        "--profile", "p0_canonical_v1",
                        "--speeds", "30",
                        "--routes-per-speed", "1",
                        "--steps", "4",
                        "--altitude", "99",
                        "--grid-spacing", "100",
                        "--width", "2000",
                        "--height", "1667",
                        "--workers", "1",
                    ])

    def test_p1_profile_records_dense_urban_contract_and_restores_config(self):
        from uav_simulation import config

        original = (
            config.BUILDING_AREA_RATIO,
            config.BUILDING_DENSITY_KM2,
            config.BUILDING_HEIGHT_RAYLEIGH_M,
        )
        with TemporaryDirectory() as directory:
            output = Path(directory) / "p1"
            with patch.dict(
                "uav_simulation.batch.generate_dataset.P0_CANONICAL_PARAMETERS",
                {"steps": 4},
            ), redirect_stdout(io.StringIO()):
                result = generate_dataset([
                    "--output-dir", str(output),
                    "--profile", "p1_dense_urban_v1",
                    "--speeds", "30",
                    "--routes-per-speed", "1",
                    "--steps", "4",
                    "--altitude", "100",
                    "--grid-spacing", "100",
                    "--width", "2000",
                    "--height", "1667",
                    "--workers", "1",
                ])
            self.assertEqual(result, 0)
            metadata = json.loads(
                (output / "dataset_metadata.json").read_text(encoding="utf-8")
            )
            parameters = metadata["generation"]["parameters"]
            self.assertEqual(parameters["profile"], "p1_dense_urban_v1")
            self.assertEqual(
                parameters["urban_profile"],
                {
                    "building_area_ratio": 0.5,
                    "building_density_km2": 300.0,
                    "building_height_rayleigh_m": 20.0,
                },
            )
            simulation_config = metadata["generation"]["simulation_config"]
            self.assertEqual(simulation_config["BUILDING_AREA_RATIO"], 0.5)
            self.assertEqual(simulation_config["BUILDING_DENSITY_KM2"], 300.0)
            self.assertEqual(simulation_config["BUILDING_HEIGHT_RAYLEIGH_M"], 20.0)
        self.assertEqual(
            (
                config.BUILDING_AREA_RATIO,
                config.BUILDING_DENSITY_KM2,
                config.BUILDING_HEIGHT_RAYLEIGH_M,
            ),
            original,
        )

    def test_p2_p3_p4_profiles_record_stress_contract_and_restore_config(self):
        from uav_simulation import config

        original = (
            config.BUILDING_AREA_RATIO,
            config.BUILDING_DENSITY_KM2,
            config.BUILDING_HEIGHT_RAYLEIGH_M,
            config.LOS_STATE_CORRELATION_DISTANCE_M,
            config.RICIAN_K_FACTOR_DB,
        )
        specs = {
            "p2_high_mobility_v1": {
                "urban": (0.3, 500.0, 15.0),
                "channel": (50.0, 10.0),
            },
            "p3_fast_channel_v1": {
                "urban": (0.3, 500.0, 15.0),
                "channel": (25.0, 6.0),
            },
            "p4_combined_harsh_v1": {
                "urban": (0.5, 300.0, 20.0),
                "channel": (25.0, 6.0),
            },
        }
        with TemporaryDirectory() as directory:
            for profile, expected in specs.items():
                with self.subTest(profile=profile):
                    output = Path(directory) / profile
                    with patch.dict(
                        "uav_simulation.batch.generate_dataset.P0_CANONICAL_PARAMETERS",
                        {"steps": 4},
                    ), patch.dict(
                        "uav_simulation.batch.generate_dataset.LOCKED_PROFILE_SPEEDS",
                        {profile: (30,)},
                    ), redirect_stdout(io.StringIO()):
                        result = generate_dataset([
                            "--output-dir", str(output),
                            "--profile", profile,
                            "--speeds", "30",
                            "--routes-per-speed", "1",
                            "--steps", "4",
                            "--altitude", "100",
                            "--grid-spacing", "100",
                            "--width", "2000",
                            "--height", "1667",
                            "--workers", "1",
                        ])
                    self.assertEqual(result, 0)
                    metadata = json.loads(
                        (output / "dataset_metadata.json").read_text(encoding="utf-8")
                    )
                    parameters = metadata["generation"]["parameters"]
                    self.assertEqual(parameters["profile"], profile)
                    self.assertEqual(
                        tuple(parameters["urban_profile"].values()),
                        expected["urban"],
                    )
                    self.assertEqual(
                        tuple(parameters["channel_profile"].values()),
                        expected["channel"],
                    )
                    simulation_config = metadata["generation"]["simulation_config"]
                    self.assertEqual(
                        simulation_config["LOS_STATE_CORRELATION_DISTANCE_M"],
                        expected["channel"][0],
                    )
                    self.assertEqual(
                        simulation_config["RICIAN_K_FACTOR_DB"],
                        expected["channel"][1],
                    )
        self.assertEqual(
            (
                config.BUILDING_AREA_RATIO,
                config.BUILDING_DENSITY_KM2,
                config.BUILDING_HEIGHT_RAYLEIGH_M,
                config.LOS_STATE_CORRELATION_DISTANCE_M,
                config.RICIAN_K_FACTOR_DB,
            ),
            original,
        )

    def test_spawn_workers_find_complete_and_incomplete_routes(self):
        modules = [import_module(f'uav_simulation.batch.{name}') for name in (
            'generate_batch', 'generate_batch_300m', 'generate_batch_grid100_h100',
        )]
        with TemporaryDirectory() as directory, working_directory(directory):
            for module in modules:
                for path in module.route_paths(30, 0):
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(
                        'sample,time_ms,bs_0\n' + '0,0,-100\n' * 6000,
                        encoding='utf-8',
                    )
                incomplete = module.route_paths(30, 1)
                for path in incomplete:
                    path.write_text('sample,time_ms,bs_0\n0,0,-100\n', encoding='utf-8')
            with ProcessPoolExecutor(
                max_workers=2, mp_context=multiprocessing.get_context('spawn'),
            ) as executor:
                for module in modules:
                    self.assertTrue(executor.submit(module.route_is_complete, 30, 0).result(timeout=30))
                    self.assertFalse(executor.submit(module.route_is_complete, 30, 1).result(timeout=30))


if __name__ == '__main__':
    unittest.main()
