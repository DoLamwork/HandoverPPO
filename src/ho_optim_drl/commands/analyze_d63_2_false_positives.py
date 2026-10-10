"""Post-hoc, no-refit false-positive analysis for the frozen D63-2 target MLP."""

from __future__ import annotations

from argparse import Namespace
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np

from ho_optim_drl.commands.audit_policy_learning import _sha256_file
from ho_optim_drl.commands.probe_d63_2 import RISK_FIELDS, choose_actions


SCHEMA = "d63_2_target_mlp_false_positive_audit_v1"
EPS = 1e-9
CONTINUOUS_FIELDS = ("delta_delivered_mbit", *RISK_FIELDS)


def _json_write(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _prepare_destination(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        return {name: loaded[name] for name in loaded.files}


def false_positive_reasons(
    arrays: dict[str, np.ndarray], indices: np.ndarray
) -> dict[str, Any]:
    reasons: dict[str, Any] = {}
    no_rate_gain = arrays["delta_delivered_mbit"][indices] <= EPS
    reasons["no_delivered_data_gain"] = {
        "rows": int(np.sum(no_rate_gain)),
        "mean_delta": float(np.mean(arrays["delta_delivered_mbit"][indices])) if len(indices) else None,
        "sum_delta": float(np.sum(arrays["delta_delivered_mbit"][indices])),
    }
    for field in RISK_FIELDS:
        worse = arrays[field][indices] > EPS
        reasons[field] = {
            "rows_worse": int(np.sum(worse)),
            "mean_delta_all_false_positives": float(np.mean(arrays[field][indices])) if len(indices) else None,
            "sum_positive_delta": float(np.sum(arrays[field][indices][worse])),
        }
    return reasons


def _group_breakdown(
    arrays: dict[str, np.ndarray], chosen: np.ndarray
) -> dict[str, Any]:
    changed = ~arrays["is_control"][chosen]
    false_positive = changed & ~arrays["strict_useful"][chosen]
    true_positive = changed & arrays["strict_useful"][chosen]
    groups: dict[str, Any] = {}
    definitions = {
        "speed": arrays["speed_kmh"][chosen],
        "source": arrays["source"][chosen],
        "slot": arrays["planned_slot"][chosen],
    }
    for name, values in definitions.items():
        records = {}
        for value in sorted(set(values.tolist()), key=str):
            mask = values == value
            records[str(value)] = {
                "states": int(np.sum(mask)),
                "changed_actions": int(np.sum(mask & changed)),
                "true_positive_actions": int(np.sum(mask & true_positive)),
                "false_positive_actions": int(np.sum(mask & false_positive)),
            }
        groups[name] = records
    return groups


def analyze_operating_point(
    arrays: dict[str, np.ndarray], scores: np.ndarray, threshold: float
) -> tuple[dict[str, Any], np.ndarray]:
    chosen = choose_actions(arrays, scores, threshold)
    changed = ~arrays["is_control"][chosen]
    false_positive = changed & ~arrays["strict_useful"][chosen]
    true_positive = changed & arrays["strict_useful"][chosen]
    fp_indices = chosen[false_positive]
    return {
        "threshold": float(threshold),
        "states": int(len(chosen)),
        "changed_actions": int(np.sum(changed)),
        "true_positive_actions": int(np.sum(true_positive)),
        "false_positive_actions": int(np.sum(false_positive)),
        "precision_among_changed": (
            float(np.sum(true_positive) / np.sum(changed)) if np.sum(changed) else None
        ),
        "mean_delta_delivered_mbit": float(np.mean(arrays["delta_delivered_mbit"][chosen])),
        "total_delta_delivered_mbit": float(np.sum(arrays["delta_delivered_mbit"][chosen])),
        "risk_deltas": {
            field: float(np.mean(arrays[field][chosen])) for field in RISK_FIELDS
        },
        "breakdown": _group_breakdown(arrays, chosen),
        "false_positive_reasons": false_positive_reasons(arrays, fp_indices),
    }, fp_indices


def _row_level_breakdown(
    arrays: dict[str, np.ndarray], scores: np.ndarray, threshold: float
) -> dict[str, Any]:
    predicted = (
        (~arrays["is_control"])
        & arrays["requested_action_valid"]
        & (scores >= threshold)
    )
    fp = predicted & ~arrays["strict_useful"]
    tp = predicted & arrays["strict_useful"]
    result = {
        "threshold": float(threshold),
        "predicted_rows": int(np.sum(predicted)),
        "true_positive_rows": int(np.sum(tp)),
        "false_positive_rows": int(np.sum(fp)),
        "false_positive_reasons": false_positive_reasons(arrays, np.flatnonzero(fp)),
        "by_speed": {},
        "by_source": {},
        "by_slot": {},
    }
    for field, output_name in (
        ("speed_kmh", "by_speed"), ("source", "by_source"), ("planned_slot", "by_slot")
    ):
        for value in sorted(set(arrays[field].tolist()), key=str):
            mask = arrays[field] == value
            result[output_name][str(value)] = {
                "predicted_rows": int(np.sum(predicted & mask)),
                "true_positive_rows": int(np.sum(tp & mask)),
                "false_positive_rows": int(np.sum(fp & mask)),
            }
    return result


def run_analysis(*, labels_dir: Path, probe_dir: Path, output_dir: Path) -> dict[str, Any]:
    _prepare_destination(output_dir)
    labels_path = labels_dir / "target_labels.npz"
    scores_path = probe_dir / "probe_scores.npz"
    probe_report_path = probe_dir / "probe_report.json"
    for path in (labels_path, scores_path, probe_report_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    labels = _load_npz(labels_path)
    score_data = _load_npz(scores_path)
    scores = score_data["target__mlp_small__score"]
    continuous_prediction = score_data["target__mlp_small__continuous"]
    if len(scores) != len(labels["state_id"]):
        raise ValueError("Target score and label rows do not align.")
    report = json.loads(probe_report_path.read_text(encoding="utf-8"))
    target_mlp = report["families"]["target"]["models"]["mlp_small"]
    safe_threshold = float(
        target_mlp["tuning_threshold_selection"]["selected"]["threshold"]
    )
    unconstrained = max(
        target_mlp["tuning_threshold_selection"]["candidates"],
        key=lambda record: record["mean_delta_delivered_mbit"],
    )
    diagnostic_threshold = float(unconstrained["threshold"])
    holdout = labels["fold"] == "holdout"
    holdout_labels = {name: values[holdout] for name, values in labels.items()}
    holdout_scores = scores[holdout]
    holdout_prediction = continuous_prediction[holdout]

    operating_points = {}
    fp_indices_by_name: dict[str, np.ndarray] = {}
    for name, threshold in (
        ("safe_registered", safe_threshold),
        ("fixed_0_5", 0.5),
        ("tuning_best_gain_diagnostic", diagnostic_threshold),
    ):
        operating_points[name], fp_indices_by_name[name] = analyze_operating_point(
            holdout_labels, holdout_scores, threshold
        )
    diagnostic_fp = fp_indices_by_name["tuning_best_gain_diagnostic"]
    cases = []
    for index in diagnostic_fp:
        cases.append(
            {
                "state_id": str(holdout_labels["state_id"][index]),
                "source": str(holdout_labels["source"][index]),
                "speed_kmh": int(holdout_labels["speed_kmh"][index]),
                "trajectory_id": int(holdout_labels["trajectory_id"][index]),
                "planned_slot": int(holdout_labels["planned_slot"][index]),
                "score": float(holdout_scores[index]),
                "actual": {
                    field: float(holdout_labels[field][index])
                    for field in CONTINUOUS_FIELDS
                },
                "predicted": {
                    field: float(holdout_prediction[index, column])
                    for column, field in enumerate(CONTINUOUS_FIELDS)
                },
            }
        )
    output = {
        "schema_name": SCHEMA,
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "analysis_only_no_refit": True,
        "validation_or_test_rows_read": 0,
        "fold": "TRAIN-holdout frozen in D63-2 v1",
        "policy_status": {
            "registered_safe_threshold": safe_threshold,
            "registered_policy_changes_actions": False,
            "diagnostic_threshold": diagnostic_threshold,
            "diagnostic_threshold_source": "maximum tuning mean gain without reliability constraint",
            "diagnostic_threshold_is_not_approved_policy": True,
        },
        "inputs": {
            "target_labels": {"path": str(labels_path.resolve()), "sha256": _sha256_file(labels_path)},
            "probe_scores": {"path": str(scores_path.resolve()), "sha256": _sha256_file(scores_path)},
            "probe_report": {"path": str(probe_report_path.resolve()), "sha256": _sha256_file(probe_report_path)},
        },
        "holdout": {
            "rows": int(np.sum(holdout)),
            "states": int(len(np.unique(holdout_labels["state_id"]))),
            "routes": int(len(set(zip(holdout_labels["speed_kmh"].tolist(), holdout_labels["trajectory_id"].tolist())))),
            "positive_rows": int(np.sum(holdout_labels["strict_useful"])),
        },
        "row_level": {
            "fixed_0_5": _row_level_breakdown(holdout_labels, holdout_scores, 0.5),
            "tuning_best_gain_diagnostic": _row_level_breakdown(
                holdout_labels, holdout_scores, diagnostic_threshold
            ),
        },
        "action_level_operating_points": operating_points,
        "diagnostic_false_positive_cases": cases,
    }
    report_path = output_dir / "false_positive_report.json"
    _json_write(report_path, output)
    with (output_dir / "false_positive_cases.jsonl").open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(case, ensure_ascii=False, allow_nan=False) + "\n")
    _json_write(
        output_dir / "manifest.json",
        {
            "schema_name": "d63_2_false_positive_manifest_v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "files": [
                {"path": path.name, "size_bytes": path.stat().st_size, "sha256": _sha256_file(path)}
                for path in sorted(output_dir.iterdir()) if path.is_file()
            ],
        },
    )
    return output


def main(root_path: str, args: Namespace) -> int:
    root = Path(root_path)
    report = run_analysis(
        labels_dir=(root / args.labels_dir).resolve(),
        probe_dir=(root / args.probe_dir).resolve(),
        output_dir=(root / args.output_dir).resolve(),
    )
    diagnostic = report["action_level_operating_points"]["tuning_best_gain_diagnostic"]
    print(
        "Target-MLP FP audit: "
        f"{diagnostic['changed_actions']} changed, "
        f"{diagnostic['true_positive_actions']} TP, "
        f"{diagnostic['false_positive_actions']} FP at tuning-derived threshold "
        f"{diagnostic['threshold']:.3f}."
    )
    return 0
