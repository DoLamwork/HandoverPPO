import csv
import json
import tempfile
import unittest
from pathlib import Path

from ho_optim_drl.commands.summarize_ablation import summarize_arm, summarize_arms


class AblationSummaryTests(unittest.TestCase):
    def _write_arm(
        self,
        root: Path,
        experiment_id: str = "E03_ranked_k3_s17",
        *,
        test_evaluated: bool = False,
    ) -> Path:
        arm = root / experiment_id
        checkpoints = arm / "checkpoints"
        checkpoints.mkdir(parents=True)
        config = {
            "experiment_id": experiment_id,
            "parent_id": "E02_ranked_k5_s17",
            "changed_factor": "candidate_count_5_to_3",
            "candidate": {"K": 3, "candidate_order_key": "ranked_v1"},
            "action_schema": ["STAY", "HO_C1", "HO_C2", "HO_C3"],
            "observation_schema": {"size": 39},
            "seeds": {"training": 17},
            "budget": {"total_macro_steps": 100000},
        }
        (arm / "experiment_config.json").write_text(json.dumps(config))
        selected = {
            "mean_return": -14.0,
            "mean_estimated_rate_mbps": 23.0,
            "mean_relative_rate_percent": 91.0,
            "p10_relative_rate_percent": 84.0,
            "rlf_time_percent": 1.3,
            "handover_failure_percent": 10.9,
            "ping_pong_percent": 37.0,
            "handover_fraction": 1.0,
            "action_fractions": {
                "STAY": 0.0,
                "HO_C1": 1.0,
                "HO_C2": 0.0,
                "HO_C3": 0.0,
            },
        }
        latest = arm / "smdp_ppo_latest.pt"
        exported = arm / "smdp_ppo_selected.pt"
        latest.touch()
        exported.touch()
        report = {
            "experiment_status": "QOS_FEASIBLE",
            "experiment_config": {"config": config},
            "checkpoint_selection": {
                "validation_checkpoints": 2,
                "qos_feasible_checkpoints": 1,
            },
            "selection_reason": "best_qos",
            "validation": selected,
            "test_evaluated": test_evaluated,
            "checkpoints": {
                "latest": str(latest),
                "selected_export": str(exported),
            },
        }
        (arm / "training_report.json").write_text(json.dumps(report))
        fields = [
            "macro_steps",
            "qos_feasible",
            "aggregate_rate_ratio_to_3gpp",
            "aggregate_relative_rate_ratio_to_3gpp",
            "mean_return",
            "mean_estimated_rate_mbps",
            "mean_relative_rate_percent",
            "p10_relative_rate_percent",
            "rlf_time_percent",
            "handover_failure_percent",
            "ping_pong_percent",
            "handover_fraction",
            "action_counts",
            "action_fractions",
        ]
        with (checkpoints / "validation_metrics.csv").open(
            "w", newline="", encoding="utf-8"
        ) as target:
            writer = csv.DictWriter(target, fieldnames=fields)
            writer.writeheader()
            writer.writerow(
                {
                    "macro_steps": 51200,
                    "qos_feasible": True,
                    "aggregate_rate_ratio_to_3gpp": 1.0,
                    "aggregate_relative_rate_ratio_to_3gpp": 1.0,
                    **selected,
                    "action_counts": json.dumps({"HO_C1": 100}),
                    "action_fractions": json.dumps(selected["action_fractions"]),
                }
            )
            writer.writerow(
                {
                    "macro_steps": 92160,
                    "qos_feasible": False,
                    "aggregate_rate_ratio_to_3gpp": 0.84,
                    "aggregate_relative_rate_ratio_to_3gpp": 0.81,
                    "mean_return": -12.0,
                    "mean_estimated_rate_mbps": 19.0,
                    "mean_relative_rate_percent": 74.0,
                    "p10_relative_rate_percent": 59.0,
                    "rlf_time_percent": 1.7,
                    "handover_failure_percent": 0.0,
                    "ping_pong_percent": 0.0,
                    "handover_fraction": 0.0,
                    "action_counts": json.dumps({"STAY": 100}),
                    "action_fractions": json.dumps({"STAY": 1.0, "HO_C1": 0.0}),
                }
            )
        return arm

    def test_summary_exposes_selected_and_last_validation_modes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            arm = self._write_arm(Path(directory))
            record = summarize_arm(arm, expected_budget=100000)
        self.assertEqual(record["selected_step"], 51200)
        self.assertEqual(record["last_scheduled_validation_step"], 92160)
        self.assertEqual(
            record["selected_policy_mode"],
            "HO_C1_COLLAPSE_3GPP_EQUIVALENT",
        )
        self.assertEqual(record["last_validation_policy_mode"], "STAY_COLLAPSE")
        self.assertTrue(record["selected_matches_3gpp"])

    def test_summary_rejects_test_evaluated_arm(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            arm = self._write_arm(Path(directory), test_evaluated=True)
            with self.assertRaisesRegex(ValueError, "test split"):
                summarize_arm(arm)

    def test_summary_writes_registry_formats(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_arm(root)
            output = root / "summary"
            records = summarize_arms(
                root,
                ["E03_ranked_k3_s17"],
                output,
                expected_budget=100000,
            )
            self.assertEqual(len(records), 1)
            self.assertTrue((output / "ablation_registry.json").is_file())
            self.assertTrue((output / "ablation_registry.csv").is_file())
            self.assertIn(
                "HO_C1_COLLAPSE_3GPP_EQUIVALENT",
                (output / "ablation_registry.md").read_text(),
            )


if __name__ == "__main__":
    unittest.main()
