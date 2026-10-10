import tempfile
import unittest
from pathlib import Path

from ho_optim_drl.commands.prepare_d63_2_labels import (
    build_route_split,
    compare_branch,
    _prepare_destination,
)


def _row(*, delivered=10.0, failures=0, rlf=0, pure_rlf=0, pp=0,
         recovery=0.0, interruption=0.1, slot=1, control=False,
         status="executed", fallback=None):
    return {
        "requested_slot": slot,
        "is_control": control,
        "physical": {
            "delivered_mbit": delivered,
            "relative_rate_percent": delivered * 10.0,
            "connected_seconds": 0.9,
            "handover_interruption_seconds": interruption,
            "recovery_seconds": recovery,
            "handover_attempts": 1,
            "handover_successes": 1 - failures,
            "handover_failures": failures,
            "rlf_starts": rlf,
            "pure_rlf_starts": pure_rlf,
            "ping_pong_count": pp,
        },
        "execution": {
            "status": status,
            "fallback": fallback,
            "executed_slot": slot,
        },
    }


class D632LabelTests(unittest.TestCase):
    def test_route_split_is_stratified_reproducible_and_leakage_safe(self):
        routes = [(speed, route) for speed in (30, 50, 70, 90) for route in range(20)]
        first, manifest = build_route_split(routes, seed=29)
        second, _ = build_route_split(reversed(routes), seed=29)
        self.assertEqual(first, second)
        self.assertEqual({fold: list(first.values()).count(fold) for fold in ("fit", "tuning", "holdout")},
                         {"fit": 48, "tuning": 16, "holdout": 16})
        for speed in (30, 50, 70, 90):
            self.assertEqual(
                {fold: len(manifest["per_speed"][str(speed)][fold]) for fold in ("fit", "tuning", "holdout")},
                {"fit": 12, "tuning": 4, "holdout": 4},
            )

    def test_gain_and_risk_are_preserved_separately(self):
        control = _row(control=True)
        safe_gain = compare_branch(_row(delivered=11.0, slot=2), control)
        self.assertTrue(safe_gain["strict_useful"])
        self.assertEqual(safe_gain["pareto_class"], "strict_dominant")
        risky_gain = compare_branch(_row(delivered=11.0, failures=1, slot=2), control)
        self.assertFalse(risky_gain["strict_useful"])
        self.assertTrue(risky_gain["risk_worse"])
        self.assertEqual(risky_gain["pareto_class"], "rate_risk_tradeoff")
        self.assertAlmostEqual(risky_gain["delta_delivered_mbit"], 1.0)
        self.assertAlmostEqual(risky_gain["delta_handover_failures"], 1.0)

    def test_fallback_cannot_be_labeled_as_useful_requested_candidate(self):
        control = _row(control=True)
        fallback = _row(delivered=12.0, slot=2, fallback="C1")
        result = compare_branch(fallback, control)
        self.assertFalse(result["requested_action_valid"])
        self.assertFalse(result["strict_useful"])
        self.assertEqual(result["pareto_class"], "unavailable_or_fallback")

    def test_output_refuses_overwrite(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            path = Path(temporary) / "output"
            path.mkdir()
            (path / "existing.txt").write_text("keep", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                _prepare_destination(path)


if __name__ == "__main__":
    unittest.main()
