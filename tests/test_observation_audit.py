import unittest

from ho_optim_drl.commands.audit_observations import (
    _prediction_summary,
    _sample_routes,
)
from ho_optim_drl.data import TracePair


class ObservationAuditTests(unittest.TestCase):
    def test_route_sampling_is_stratified_bounded_and_reproducible(self) -> None:
        pairs = tuple(
            TracePair(speed, route, f"r{speed}_{route}", f"s{speed}_{route}")
            for speed in (30, 50)
            for route in range(5)
        )

        first = _sample_routes(pairs, routes_per_speed=2, seed=17)
        second = _sample_routes(pairs, routes_per_speed=2, seed=17)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 4)
        self.assertEqual(sum(pair.speed_kmh == 30 for pair in first), 2)
        self.assertEqual(sum(pair.speed_kmh == 50 for pair in first), 2)

    def test_prediction_summary_reports_signed_bias_and_error_norms(self) -> None:
        summary = _prediction_summary(
            [{"error_db": -2.0}, {"error_db": 4.0}]
        )

        self.assertEqual(summary["samples"], 2)
        self.assertEqual(summary["mae_db"], 3.0)
        self.assertAlmostEqual(summary["rmse_db"], 10 ** 0.5)
        self.assertEqual(summary["bias_db"], 1.0)


if __name__ == "__main__":
    unittest.main()
