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
