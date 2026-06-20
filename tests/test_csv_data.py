"""CSV trace loader regression tests."""

import os
import tempfile
import unittest

import numpy as np

from ho_optim_drl.config import Config
from ho_optim_drl.data import (
    RadioTrace,
    discover_trace_pairs,
    loader,
    stratified_trajectory_split,
)


class CsvDataTests(unittest.TestCase):
    def test_all_csv_traces_form_rsrp_sinr_pairs(self) -> None:
        pairs = discover_trace_pairs("data/processed")

        self.assertEqual(len(pairs), 124)
        self.assertEqual(
            {pair.speed_kmh for pair in pairs},
            {30, 50, 70, 90},
        )

    def test_loader_builds_three_explicit_bs_by_time_streams(self) -> None:
        trace = loader.load_radio_trace(
            Config(),
            "data/processed",
            "rsrp_30kmh_0.csv",
            "sinr_30kmh_0.csv",
        )

        self.assertEqual(trace.shape, (5, 5400))
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
            "data/processed",
            "rsrp_30kmh_0.csv",
            "sinr_30kmh_0.csv",
        )
        raw_sinr, _ = loader._load_csv_measurements(
            "data/processed/sinr_30kmh_0.csv"
        )
        raw_rsrp, _ = loader._load_csv_measurements(
            "data/processed/rsrp_30kmh_0.csv"
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
                "data/processed",
                "rsrp_30kmh_0.csv",
                "sinr_30kmh_1.csv",
            )


class DatasetSplitTests(unittest.TestCase):
    def test_split_is_reproducible_disjoint_and_stratified(self) -> None:
        pairs = discover_trace_pairs("data/processed")

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

    def test_radio_trace_rejects_stream_shape_mismatch(self) -> None:
        with self.assertRaises(ValueError):
            RadioTrace.from_arrays(
                np.zeros((2, 4)),
                np.zeros((2, 3)),
            )


if __name__ == "__main__":
    unittest.main()
