"""Tests for unused-route and temporally spaced post-TTT bank selection."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from ho_optim_drl.commands.build_post_ttt_train_bank import (
    lock_confirmation_route_split,
    select_spaced_candidates,
    select_unused_train_routes,
)
from ho_optim_drl.data import TracePair


def _pair(speed: int, trajectory: int) -> TracePair:
    return TracePair(speed, trajectory, f"rsrp_{speed}_{trajectory}.csv", f"sinr_{speed}_{trajectory}.csv")


class PostTttTrainBankTests(unittest.TestCase):
    def test_spaced_selection_is_hash_ranked_and_respects_distance(self) -> None:
        candidates = [
            {"score": score, "record": {"tic": tic}}
            for score, tic in ((1, 10), (2, 12), (3, 20), (4, 31))
        ]
        selected = select_spaced_candidates(
            candidates, max_anchors=3, min_separation_ticks=10
        )
        self.assertEqual([row["record"]["tic"] for row in selected], [10, 20, 31])

    def test_route_selection_excludes_old_routes_and_locks_folds(self) -> None:
        pairs = [_pair(30, index) for index in range(20)]
        first, folds, available = select_unused_train_routes(
            pairs, speeds=[30], split_seed=17, route_seed=43,
            excluded={(30, 0)}, fit_per_speed=2,
            tuning_per_speed=1, confirmation_per_speed=1,
        )
        second, second_folds, _ = select_unused_train_routes(
            pairs, speeds=[30], split_seed=17, route_seed=43,
            excluded={(30, 0)}, fit_per_speed=2,
            tuning_per_speed=1, confirmation_per_speed=1,
        )
        self.assertEqual([pair.key for pair in first], [pair.key for pair in second])
        self.assertEqual(folds, second_folds)
        self.assertNotIn((30, 0), {pair.key for pair in first})
        self.assertEqual(dict(__import__("collections").Counter(folds.values())), {
            "fit": 2, "tuning": 1, "confirmation": 1,
        })
        self.assertGreaterEqual(available[30], 4)

    def test_split_only_locks_routes_without_outcomes(self) -> None:
        with TemporaryDirectory(dir=Path.cwd()) as temporary:
            root = Path(temporary)
            data_dir = root / "data"
            data_dir.mkdir()
            for trajectory in range(20):
                for kind in ("rsrp", "sinr"):
                    (data_dir / f"{kind}_30kmh_{trajectory}.csv").write_text(
                        "sample,time_ms,bs0\n0,0,-50\n", encoding="utf-8"
                    )
            excluded_path = root / "excluded.json"
            excluded_path.write_text(__import__("json").dumps({
                "split": "train", "validation_or_test_rows_read": 0,
                "selected_routes": [{"speed_kmh": 30, "trajectory_id": 0}],
            }), encoding="utf-8")
            output_dir = root / "locked"
            manifest = lock_confirmation_route_split(
                root=root, data_dir=data_dir, output_dir=output_dir, speeds=[30],
                split_seed=17, route_seed=101, sample_seed=17,
                exclude_route_splits=[excluded_path], confirmation_per_speed=2,
            )
            payload = __import__("json").loads(
                (output_dir / "route_split.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["selected_route_count"], 2)
            self.assertEqual(payload["fold_counts_per_speed"]["confirmation"], 2)
            self.assertNotIn(
                (30, 0),
                {(row["speed_kmh"], row["trajectory_id"]) for row in payload["selected_routes"]},
            )
            self.assertFalse((output_dir / "local_outcomes.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
