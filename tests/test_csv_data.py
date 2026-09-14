"""CSV trace loader regression tests."""

import os
import tempfile
import unittest

import numpy as np

from ho_optim_drl.config import Config
from ho_optim_drl.data import (
    RadioTrace,
    TracePair,
    discover_trace_pairs,
    filter_trace_pairs_by_speed,
    loader,
    stratified_trajectory_split,
)


def _write_pair(
    directory: str,
    speed_kmh: int,
    trajectory_id: int,
    *,
    num_samples: int = 12,
    num_cells: int = 5,
) -> None:
    header = "sample,time_ms," + ",".join(
        f"bs_{cell}" for cell in range(num_cells)
    ) + "\n"
    for kind in ("rsrp", "sinr"):
        rows = []
        for sample in range(num_samples):
            if kind == "rsrp":
                values = [
                    -90.0 + cell + 0.25 * sample
                    for cell in range(num_cells)
                ]
            else:
                values = [
                    -2.0 + cell + 0.10 * sample
                    for cell in range(num_cells)
                ]
            rows.append(
                f"{sample},{sample * 10},"
                + ",".join(str(value) for value in values)
                + "\n"
            )
        with open(
            os.path.join(
                directory,
                f"{kind}_{speed_kmh}kmh_{trajectory_id}.csv",
            ),
            "w",
            encoding="utf-8",
        ) as output_file:
            output_file.write(header + "".join(rows))


class _CsvFixtureTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.data_dir = self._temporary_directory.name
        for speed_kmh in (30, 50, 70, 90):
            for trajectory_id in range(8):
                _write_pair(self.data_dir, speed_kmh, trajectory_id)

    def tearDown(self) -> None:
        self._temporary_directory.cleanup()


class CsvDataTests(_CsvFixtureTestCase):
    def test_all_csv_traces_form_rsrp_sinr_pairs(self) -> None:
        pairs = discover_trace_pairs(self.data_dir)

        self.assertEqual(len(pairs), 32)
        self.assertEqual(
            {pair.speed_kmh for pair in pairs},
            {30, 50, 70, 90},
        )

    def test_loader_builds_three_explicit_bs_by_time_streams(self) -> None:
        trace = loader.load_radio_trace(
            Config(),
            self.data_dir,
            "rsrp_30kmh_0.csv",
            "sinr_30kmh_0.csv",
        )

        self.assertEqual(trace.shape, (5, 12))
        self.assertEqual(trace.delta_t_ms, 10)
        self.assertEqual(trace.trace_id, "30kmh_0")
        self.assertFalse(np.shares_memory(trace.rsrp_a3_db, trace.sinr_rate_db))
        np.testing.assert_array_equal(trace.sinr_rate_db, trace.sinr_rlm_db)
        self.assertFalse(trace.rsrp_a3_db.flags.writeable)
        self.assertFalse(trace.sinr_rate_db.flags.writeable)

    def test_loader_filters_only_rsrp(self) -> None:
        config = Config()
        trace = loader.load_radio_trace(
            config,
            self.data_dir,
            "rsrp_30kmh_0.csv",
            "sinr_30kmh_0.csv",
        )
        raw_sinr, _ = loader._load_csv_measurements(
            os.path.join(self.data_dir, "sinr_30kmh_0.csv")
        )
        raw_rsrp, _ = loader._load_csv_measurements(
            os.path.join(self.data_dir, "rsrp_30kmh_0.csv")
        )

        np.testing.assert_array_equal(trace.sinr_rate_db, raw_sinr.T)
        self.assertFalse(np.array_equal(trace.rsrp_a3_db, raw_rsrp.T))

    def test_loader_rejects_wrong_sampling_interval(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            header = "sample,time_ms,bs_0\n"
            rows = "0,0,-80\n1,20,-79\n"
            for prefix in ("rsrp", "sinr"):
                with open(
                    os.path.join(directory, f"{prefix}_30kmh_0.csv"),
                    "w",
                    encoding="utf-8",
                ) as file:
                    file.write(header + rows)

            with self.assertRaises(ValueError):
                loader.load_radio_trace(
                    Config(),
                    directory,
                    "rsrp_30kmh_0.csv",
                    "sinr_30kmh_0.csv",
                )

    def test_loader_rejects_mismatched_trajectory_pair(self) -> None:
        with self.assertRaises(ValueError):
            loader.load_radio_trace(
                Config(),
                self.data_dir,
                "rsrp_30kmh_0.csv",
                "sinr_30kmh_1.csv",
            )


class DatasetSplitTests(_CsvFixtureTestCase):
    def test_filter_trace_pairs_selects_only_requested_speeds(self) -> None:
        pairs = discover_trace_pairs(self.data_dir)

        selected = filter_trace_pairs_by_speed(pairs, [30, 70])

        self.assertEqual({pair.speed_kmh for pair in selected}, {30, 70})
        self.assertEqual(
            selected,
            [pair for pair in pairs if pair.speed_kmh in {30, 70}],
        )

    def test_filter_trace_pairs_rejects_unavailable_speed(self) -> None:
        pairs = discover_trace_pairs(self.data_dir)

        with self.assertRaisesRegex(ValueError, "120"):
            filter_trace_pairs_by_speed(pairs, [30, 120])

    def test_split_is_reproducible_disjoint_and_stratified(self) -> None:
        pairs = discover_trace_pairs(self.data_dir)

        first = stratified_trajectory_split(pairs, seed=17)
        second = stratified_trajectory_split(pairs, seed=17)

        self.assertEqual(first, second)
        self.assertEqual(
            len(first.train) + len(first.validation) + len(first.test),
            len(pairs),
        )
        first.validate()
        all_speeds = {pair.speed_kmh for pair in pairs}
        for split in (first.train, first.validation, first.test):
            self.assertEqual({pair.speed_kmh for pair in split}, all_speeds)

    def test_custom_split_gives_36_12_12_per_speed(self) -> None:
        speeds = (30, 50, 70, 90, 120, 150)
        pairs = [
            TracePair(
                speed,
                trajectory_id,
                f"rsrp_{speed}kmh_{trajectory_id}.csv",
                f"sinr_{speed}kmh_{trajectory_id}.csv",
            )
            for speed in speeds
            for trajectory_id in range(60)
        ]

        splits = stratified_trajectory_split(
            pairs,
            train_fraction=0.60,
            validation_fraction=0.20,
            test_fraction=0.20,
            seed=17,
        )

        splits.validate()
        for speed in speeds:
            self.assertEqual(
                sum(pair.speed_kmh == speed for pair in splits.train),
                36,
            )
            self.assertEqual(
                sum(pair.speed_kmh == speed for pair in splits.validation),
                12,
            )
            self.assertEqual(
                sum(pair.speed_kmh == speed for pair in splits.test),
                12,
            )

    def test_radio_trace_rejects_stream_shape_mismatch(self) -> None:
        with self.assertRaises(ValueError):
            RadioTrace.from_arrays(
                np.zeros((2, 4)),
                np.zeros((2, 3)),
            )


if __name__ == "__main__":
    unittest.main()
