"""Contracts between the UAV simulator, PPO dataset, and held-out evaluation."""

from contextlib import redirect_stdout
from dataclasses import asdict
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from ho_optim_drl import Config, HandoverSMDPEnv
from ho_optim_drl.commands.train_smdp_ppo import build_dataset_manifest, write_dataset_manifest
from ho_optim_drl.data import discover_trace_pairs, load_radio_trace, stratified_trajectory_split
from ho_optim_drl.data.project import checkpoint_pairs, load_protocol_config
from ho_optim_drl.commands.prepare_uav_data import main as prepare
from uav_simulation import config as generator_config
from uav_simulation.manifest import (
    build_generation_manifest,
    ensure_generation_manifest,
)


class SharedRadioConfigTests(unittest.TestCase):
    def test_generator_defaults_match_handover_environment(self):
        protocol = Config()

        self.assertEqual(
            (protocol.fc, protocol.bw, protocol.delta_t_ms),
            (2.1e9, 10e6, 10),
        )
        self.assertEqual(generator_config.FC * 1e9, protocol.fc)
        self.assertEqual(generator_config.BANDWIDTH_MHZ * 1e6, protocol.bw)
        self.assertEqual(generator_config.DELTA_T_MS, protocol.delta_t_ms)


class GeneratorManifestTests(unittest.TestCase):
    def test_manifest_pins_generator_source_parameters_and_radio(self):
        parameters = {
            "speeds_kmh": [30, 50],
            "routes_per_speed": 2,
            "steps": 6000,
        }

        manifest = build_generation_manifest(
            parameters,
            entrypoint="uav_simulation.batch.generate_dataset",
        )

        self.assertEqual(manifest["schema_version"], 3)
        self.assertEqual(manifest["generator"], "uav_simulation")
        self.assertEqual(
            manifest["output_contract"],
            "handover_ppo_train_ready_v1",
        )
        self.assertEqual(manifest["csv_schema"], "sample,time_ms,bs_0,...")
        self.assertEqual(manifest["parameters"], parameters)
        self.assertEqual(
            manifest["radio"],
            {
                "carrier_hz": 2.1e9,
                "bandwidth_hz": 10e6,
                "delta_t_ms": 10,
            },
        )
        provenance = manifest["generator_provenance"]
        self.assertEqual(
            provenance["entrypoint"],
            "uav_simulation.batch.generate_dataset",
        )
        self.assertEqual(len(provenance["source_tree_sha256"]), 64)
        self.assertIn("manifest.py", provenance["source_files_sha256"])

        with TemporaryDirectory() as directory:
            first = ensure_generation_manifest(directory, manifest)
            second = ensure_generation_manifest(directory, manifest)
            self.assertEqual(first, second)
            self.assertEqual(json.loads(first.read_text()), manifest)

    def test_manifest_refuses_to_label_existing_unverified_csv(self):
        manifest = build_generation_manifest(
            {"speeds_kmh": [30]},
            entrypoint="uav_simulation.batch.generate_dataset",
        )
        with TemporaryDirectory() as directory:
            Path(directory, "ue0_rsrp_30kmh_0.csv").write_text(
                "Step,bs0\n0,-80\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(FileExistsError, "unverified provenance"):
                ensure_generation_manifest(directory, manifest)

    def test_protocol_config_can_use_generator_manifest_directly(self):
        manifest = build_generation_manifest(
            {"speeds_kmh": [30], "steps": 4},
            entrypoint="uav_simulation.batch.generate_dataset",
        )
        with TemporaryDirectory() as directory:
            ensure_generation_manifest(directory, manifest)
            protocol = load_protocol_config(directory)
            self.assertEqual(
                (protocol.fc, protocol.bw, protocol.delta_t_ms),
                (2.1e9, 10e6, 10),
            )


class IntegratedDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.raw = self.root / "raw"
        self.data = self.root / "prepared"
        self.raw.mkdir()
        for route in range(6):
            for kind, value in (("rsrp", -80.0), ("sinr", 5.0)):
                values = np.column_stack((np.arange(20), np.full(20, value), np.full(20, value + 3)))
                np.savetxt(self.raw / f"ue0_{kind}_30kmh_{route}.csv", values,
                           delimiter=",", header="Step,bs0,bs1", comments="", fmt="%.2f")
        (self.raw / "simulation_manifest.json").write_text(json.dumps({
            "schema_version": 3,
            "generator": "uav_simulation",
            "generator_provenance": {
                "entrypoint": "test.fixture",
                "source_tree_sha256": "a" * 64,
            },
            "radio": {"carrier_hz": 4e9, "bandwidth_hz": 20e6, "delta_t_ms": 10},
        }), encoding="utf-8")
        with redirect_stdout(io.StringIO()):
            prepare(["--input-dir", str(self.raw), "--output-dir", str(self.data)])

    def test_conversion_preserves_samples_radio_values_and_source_ids(self):
        metadata = json.loads((self.data / "dataset_metadata.json").read_text())
        self.assertEqual(metadata["radio_provenance"], "generation_manifest")
        self.assertEqual(metadata["generation"]["generator"], "uav_simulation")
        self.assertEqual(metadata["trajectory_count"], 6)
        self.assertEqual([row["source_trace_id"] for row in metadata["trajectories"]], list(range(6)))
        radio = load_protocol_config(self.data)
        self.assertEqual((radio.fc, radio.bw, radio.delta_t_ms), (4e9, 20e6, 10))
        trace = load_radio_trace(radio, str(self.data), "rsrp_30kmh_5.csv", "sinr_30kmh_5.csv")
        expected = np.loadtxt(self.raw / "ue0_sinr_30kmh_5.csv", delimiter=",", skiprows=1)[:, 1:].T
        np.testing.assert_array_equal(trace.sinr_rate_db, expected)
        self.assertEqual(HandoverSMDPEnv(data_dir=self.data).config.bw, 20e6)

    def test_conflicting_radio_declaration_is_rejected_before_writing(self):
        destination = self.root / "bad"
        with self.assertRaisesRegex(ValueError, "conflicts"):
            prepare(["--input-dir", str(self.raw), "--output-dir", str(destination), "--bandwidth-mhz", "10"])
        self.assertFalse(destination.exists())

    def test_existing_prepared_dataset_cannot_be_overwritten(self):
        with self.assertRaises(FileExistsError):
            prepare(["--input-dir", str(self.raw), "--output-dir", str(self.data)])

    def test_checkpoint_protocol_must_match_dataset_radio(self):
        with self.assertRaisesRegex(ValueError, "differs from checkpoint"):
            load_protocol_config(self.data, {"protocol_config": asdict(Config())})

    def _checkpoint_manifest(self):
        pairs = discover_trace_pairs(str(self.data))
        splits = stratified_trajectory_split(pairs, seed=9)
        fractions = {"train": 0.7, "validation": 0.15, "test": 0.15}
        manifest = build_dataset_manifest(str(self.data), splits, selected_speeds=[30], split_seed=9, split_fractions=fractions)
        paths = write_dataset_manifest(str(self.root), manifest)
        extra = {"dataset_manifest_file": Path(paths["json"]).name,
                 "dataset_manifest_sha256": paths["sha256"], "split_seed": 9,
                 "split_fractions": fractions, "selected_speeds_kmh": [30]}
        return splits, extra

    def test_evaluation_reuses_saved_split_when_new_routes_are_added(self):
        splits, extra = self._checkpoint_manifest()
        for kind in ("rsrp", "sinr"):
            (self.data / f"{kind}_10kmh_0.csv").write_bytes((self.data / f"{kind}_30kmh_0.csv").read_bytes())
        selected = checkpoint_pairs(self.data, self.root / "model.pt", extra, split="test")
        self.assertEqual(selected, splits.test)
        self.assertFalse({pair.key for pair in selected} & {pair.key for pair in splits.train})

    def test_changed_held_out_data_and_different_seed_are_rejected(self):
        splits, extra = self._checkpoint_manifest()
        with self.assertRaisesRegex(ValueError, "split_seed"):
            checkpoint_pairs(self.data, self.root / "model.pt", extra, split="test", split_seed=10)
        path = self.data / splits.test[0].rsrp_file
        path.write_text(path.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "fingerprint changed"):
            checkpoint_pairs(self.data, self.root / "model.pt", extra, split="test")


class IntegratedCommandTests(unittest.TestCase):
    def test_pipeline_commands_are_available_outside_project(self):
        with TemporaryDirectory() as directory:
            for command in ("generate_data", "prepare_data", "train_ppo", "evaluate_ppo", "benchmark", "validate_3gpp"):
                with self.subTest(command=command):
                    result = subprocess.run([sys.executable, "-m", "ho_optim_drl", command, "--help"],
                                            cwd=directory, capture_output=True, text=True,
                                            env={**os.environ, "MPLBACKEND": "Agg"}, timeout=40)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn("usage:", result.stdout)


if __name__ == "__main__":
    unittest.main()
