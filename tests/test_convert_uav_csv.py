"""UAV CSV subset-selection regression tests."""

from pathlib import Path
import unittest

from scripts.convert_uav_csv import _SourcePair, _select_source_pairs


def _pair(speed_kmh: int, source_trace_id: int, ue: int = 0) -> _SourcePair:
    stem = f"ue{ue}_{{kind}}_{speed_kmh}kmh_{source_trace_id}.csv"
    return _SourcePair(
        ue=ue,
        speed_kmh=speed_kmh,
        source_trace_id=source_trace_id,
        rsrp_path=Path(stem.format(kind="rsrp")),
        sinr_path=Path(stem.format(kind="sinr")),
    )


class UavSubsetSelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pairs = [
            _pair(speed, trace_id)
            for speed in (30, 50)
            for trace_id in (30, 31)
        ]

    def test_selects_requested_ids_at_every_speed(self) -> None:
        selected = _select_source_pairs(
            self.pairs,
            speeds=[30, 50],
            source_trace_ids=[31],
        )

        self.assertEqual(
            [(pair.speed_kmh, pair.source_trace_id) for pair in selected],
            [(30, 31), (50, 31)],
        )

    def test_rejects_id_missing_from_one_speed(self) -> None:
        pairs = [
            pair
            for pair in self.pairs
            if (pair.speed_kmh, pair.source_trace_id) != (50, 31)
        ]

        with self.assertRaisesRegex(ValueError, "50"):
            _select_source_pairs(
                pairs,
                speeds=[30, 50],
                source_trace_ids=[31],
            )

    def test_rejects_unavailable_speed(self) -> None:
        with self.assertRaisesRegex(ValueError, "70"):
            _select_source_pairs(
                self.pairs,
                speeds=[30, 70],
                source_trace_ids=[30],
            )


if __name__ == "__main__":
    unittest.main()
