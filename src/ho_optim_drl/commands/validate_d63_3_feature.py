"""Validate the frozen D63 execution-safety hypothesis on TRAIN fit/tuning."""

from __future__ import annotations

from argparse import Namespace
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np

from ho_optim_drl.commands.audit_policy_learning import (
    _average_precision,
    _sha256_file,
)
from ho_optim_drl.commands.audit_timing_target import (
    _source_specs,
    replay_d63_anchors,
)
from ho_optim_drl.commands.probe_d63_2 import (
    RISK_FIELDS,
    _route_bootstrap_selection,
    _selection_metrics,
    _subset,
)
from ho_optim_drl.commands.replay_d63_2_failure_cases import (
    causal_radio_features,
)


SCHEMA = "d63_3_execution_safety_validation_v1"
FEATURE = "execution_safety_margin_v1_db"
EPS = 1e-9


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        return {name: loaded[name] for name in loaded.files}


def _json_write(path: Path, value: Any) -> None:
    def compatible(item: Any) -> Any:
        if isinstance(item, dict):
            return {key: compatible(child) for key, child in item.items()}
        if isinstance(item, (list, tuple)):
            return [compatible(child) for child in item]
        if isinstance(item, (float, np.floating)) and not np.isfinite(item):
            if np.isposinf(item):
                return "CONTROL_ONLY"
            if np.isneginf(item):
                return "BELOW_ALL"
            return None
        return item
    path.write_text(
        json.dumps(
            compatible(value), indent=2, ensure_ascii=False, allow_nan=False
        ) + "\n",
        encoding="utf-8",
    )


def _prepare_destination(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {path}")
    path.mkdir(parents=True, exist_ok=True)


def candidate_thresholds(scores: np.ndarray) -> np.ndarray:
    """Return deterministic cut points plus a control-only sentinel."""
    finite = np.unique(scores[np.isfinite(scores)])
    if not len(finite):
        return np.asarray([np.inf], dtype=np.float64)
    if len(finite) > 200:
        finite = np.unique(np.quantile(finite, np.linspace(0.0, 1.0, 201)))
    return np.concatenate((finite, np.asarray([np.inf])))


def lock_threshold(
    arrays: dict[str, np.ndarray], scores: np.ndarray
) -> tuple[float, dict[str, Any]]:
    """Lock a fit-only threshold using the registered physical objective."""
    evaluated = []
    for threshold in candidate_thresholds(scores[~arrays["is_control"]]):
        metrics = _selection_metrics(arrays, scores, float(threshold))
        evaluated.append({"threshold": float(threshold), **metrics})
    feasible = [row for row in evaluated if row["reliability_not_worse"]]
    selected = max(
        feasible,
        key=lambda row: (
            row["mean_delta_delivered_mbit"],
            -row["false_alternative_selections"],
            -row["changed_actions"],
            row["threshold"],
        ),
    )
    best_unconstrained = max(
        evaluated,
        key=lambda row: (
            row["mean_delta_delivered_mbit"],
            -row["false_alternative_selections"],
            -row["changed_actions"],
        ),
    )
    changed_feasible = [
        row for row in feasible if row["changed_actions"] > 0
    ]
    return float(selected["threshold"]), {
        "selection_fold": "fit",
        "objective": "maximize mean delivered Mbit subject to every mean risk delta <= 0",
        "selected": selected,
        "best_unconstrained": best_unconstrained,
        "best_non_control_reliability_feasible": (
            max(
                changed_feasible,
                key=lambda row: (
                    row["mean_delta_delivered_mbit"],
                    -row["false_alternative_selections"],
                    -row["changed_actions"],
                ),
            )
            if changed_feasible else None
        ),
        "thresholds_evaluated": len(evaluated),
    }


def classification_summary(
    arrays: dict[str, np.ndarray], scores: np.ndarray, threshold: float
) -> dict[str, Any]:
    alternative = ~arrays["is_control"]
    labels = arrays["strict_useful"][alternative].astype(np.int8)
    values = scores[alternative]
    predicted = values >= threshold
    tp = int(np.sum(predicted & (labels == 1)))
    fp = int(np.sum(predicted & (labels == 0)))
    fn = int(np.sum(~predicted & (labels == 1)))
    prevalence = float(np.mean(labels)) if len(labels) else 0.0
    ap = _average_precision(labels, values) if np.any(labels) else 0.0
    return {
        "alternative_rows": int(len(labels)),
        "positive_rows": int(np.sum(labels)),
        "prevalence": prevalence,
        "average_precision": float(ap),
        "lift_over_prevalence": float(ap / prevalence) if prevalence > 0 else None,
        "threshold": float(threshold),
        "predicted_positive_rows": int(np.sum(predicted)),
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
    }


def _group_selection(
    arrays: dict[str, np.ndarray], scores: np.ndarray, threshold: float
) -> dict[str, Any]:
    groups: dict[str, Any] = {
        "overall": _selection_metrics(arrays, scores, threshold)
    }
    for speed in sorted(set(arrays["speed_kmh"].tolist())):
        mask = arrays["speed_kmh"] == speed
        groups[f"speed:{int(speed)}"] = _selection_metrics(
            _subset(arrays, mask), scores[mask], threshold
        )
    for source in sorted(set(arrays["source"].tolist())):
        mask = arrays["source"] == source
        groups[f"source:{source}"] = _selection_metrics(
            _subset(arrays, mask), scores[mask], threshold
        )
    return groups


def extract_feature_scores(
    *, bank: dict[str, np.ndarray], manifest: dict[str, Any],
    state_ids: set[str], d61_checkpoint: Path, d62_checkpoint: Path,
) -> dict[tuple[str, int], float]:
    """Replay only fit/tuning anchors and compute causal per-slot margins."""
    mask = np.asarray([str(value) in state_ids for value in bank["state_id"]])
    filtered = {name: values[mask] for name, values in bank.items()}
    sources = _source_specs(d61_checkpoint, d62_checkpoint)
    features: dict[tuple[str, int], float] = {}
    for index, engine, builder in replay_d63_anchors(manifest, filtered, sources):
        builder.build(engine.snapshot)
        state_id = str(filtered["state_id"][index])
        serving = int(filtered["serving_cell"][index])
        anchor_tic = int(filtered["tic"][index])
        for slot, cell in enumerate(filtered["candidate_cells"][index], start=1):
            if int(cell) < 0:
                continue
            row = causal_radio_features(
                trace=engine.trace,
                serving_cell=serving,
                target_cell=int(cell),
                anchor_tic=anchor_tic,
                q_in_db=engine.config.q_in_db,
                delta_t_ms=engine.config.delta_t_ms,
            )
            features[(state_id, slot)] = float(row[FEATURE])
    return features


def run_validation(
    *, bank_dir: Path, labels_dir: Path, output_dir: Path
) -> dict[str, Any]:
    _prepare_destination(output_dir)
    bank_path = bank_dir / "state_bank.npz"
    manifest_path = bank_dir / "state_bank_manifest.json"
    labels_path = labels_dir / "target_labels.npz"
    split_path = labels_dir / "route_split_manifest.json"
    for path in (bank_path, manifest_path, labels_path, split_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    bank = _load_npz(bank_path)
    labels = _load_npz(labels_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    allowed = np.isin(labels["fold"], np.asarray(["fit", "tuning"]))
    arrays = {name: values[allowed] for name, values in labels.items()}
    if set(arrays["fold"].tolist()) != {"fit", "tuning"}:
        raise ValueError("D63-3 requires non-empty TRAIN fit and tuning folds.")
    state_ids = set(str(value) for value in arrays["state_id"])
    feature_map = extract_feature_scores(
        bank=bank,
        manifest=manifest,
        state_ids=state_ids,
        d61_checkpoint=Path(manifest["sources"]["d61"]["checkpoint"]),
        d62_checkpoint=Path(manifest["sources"]["d62"]["checkpoint"]),
    )
    scores = np.asarray(
        [feature_map[(str(state), int(slot))] for state, slot in zip(
            arrays["state_id"], arrays["planned_slot"]
        )],
        dtype=np.float64,
    )
    fit_mask = arrays["fold"] == "fit"
    tuning_mask = arrays["fold"] == "tuning"
    fit_arrays = _subset(arrays, fit_mask)
    tuning_arrays = _subset(arrays, tuning_mask)
    threshold, fit_selection = lock_threshold(fit_arrays, scores[fit_mask])
    tuning_selection = _group_selection(
        tuning_arrays, scores[tuning_mask], threshold
    )
    tuning_overall = tuning_selection["overall"]
    tuning_overall["route_bootstrap_gain_95_interval"] = (
        _route_bootstrap_selection(
            tuning_arrays, scores[tuning_mask], threshold, seed=63
        )
    )
    fit_overall = fit_selection["selected"]
    positive_speed_groups = sum(
        row["mean_delta_delivered_mbit"] > 0
        for name, row in tuning_selection.items() if name.startswith("speed:")
    )
    gate_pass = bool(
        fit_overall["changed_actions"] > 0
        and fit_overall["mean_delta_delivered_mbit"] > 0
        and fit_overall["reliability_not_worse"]
        and tuning_overall["changed_actions"] > 0
        and tuning_overall["mean_delta_delivered_mbit"] > 0
        and tuning_overall["reliability_not_worse"]
        and positive_speed_groups >= 2
    )
    gate = {
        "status": (
            "OPEN_D4_SINGLE_OBSERVATION_ABLATION"
            if gate_pass else "CLOSE_D63_NO_GENERALIZING_FEATURE_EVIDENCE"
        ),
        "criteria": {
            "fit": "changed actions > 0, mean gain > 0 and every mean risk delta <= 0",
            "tuning": "changed actions > 0, mean gain > 0 and every mean risk delta <= 0",
            "speed_coverage": "positive tuning gain in at least two speed groups",
            "data_scope": "holdout must not be read",
        },
        "observed": {
            "fit_pass": bool(
                fit_overall["changed_actions"] > 0
                and fit_overall["mean_delta_delivered_mbit"] > 0
                and fit_overall["reliability_not_worse"]
            ),
            "tuning_pass": bool(
                tuning_overall["changed_actions"] > 0
                and tuning_overall["mean_delta_delivered_mbit"] > 0
                and tuning_overall["reliability_not_worse"]
            ),
            "speed_coverage_pass": bool(positive_speed_groups >= 2),
            "holdout_rows_read": 0,
        },
        "observed_positive_tuning_speed_groups": int(positive_speed_groups),
        "holdout_rows_read": 0,
    }
    artifact_path = output_dir / "execution_safety_scores.npz"
    np.savez_compressed(
        artifact_path,
        state_id=arrays["state_id"],
        planned_slot=arrays["planned_slot"],
        fold=arrays["fold"],
        execution_safety_margin_v1_db=scores.astype(np.float32),
    )
    report = {
        "schema_name": SCHEMA,
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "scope": "TRAIN_fit_and_tuning_only",
        "validation_or_test_or_zero_shot_rows_read": 0,
        "holdout_rows_read": 0,
        "feature": {
            "name": FEATURE,
            "definition": "target SINR RLM gap to Qin now minus target SINR RLM standard deviation over causal 500 ms",
            "larger_is_safer": True,
            "causal_at_decision": True,
        },
        "rows": int(len(scores)),
        "states": int(len(np.unique(arrays["state_id"]))),
        "fit": {
            "classification": classification_summary(
                fit_arrays, scores[fit_mask], threshold
            ),
            "threshold_selection": fit_selection,
        },
        "tuning": {
            "classification": classification_summary(
                tuning_arrays, scores[tuning_mask], threshold
            ),
            "selection": tuning_selection,
        },
        "gate_d63_3": gate,
        "inputs": {
            "state_bank": {"path": str(bank_path.resolve()), "sha256": _sha256_file(bank_path)},
            "state_bank_manifest": {"path": str(manifest_path.resolve()), "sha256": _sha256_file(manifest_path)},
            "target_labels": {"path": str(labels_path.resolve()), "sha256": _sha256_file(labels_path)},
            "route_split": {"path": str(split_path.resolve()), "sha256": _sha256_file(split_path)},
        },
        "artifacts": {
            "scores": {"path": artifact_path.name, "sha256": _sha256_file(artifact_path)}
        },
        "interpretation_limit": (
            "This validates one prespecified causal feature on TRAIN fit/tuning. "
            "It does not reopen the frozen holdout and is not a deployable policy result."
        ),
    }
    _json_write(output_dir / "feature_validation_report.json", report)
    _json_write(output_dir / "decision_gate.json", gate)
    _json_write(
        output_dir / "manifest.json",
        {
            "schema_name": "d63_3_feature_validation_manifest_v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "files": [
                {"path": path.name, "size_bytes": path.stat().st_size, "sha256": _sha256_file(path)}
                for path in sorted(output_dir.iterdir()) if path.is_file()
            ],
        },
    )
    return report


def main(root_path: str, args: Namespace) -> int:
    root = Path(root_path)
    report = run_validation(
        bank_dir=(root / args.bank_dir).resolve(),
        labels_dir=(root / args.labels_dir).resolve(),
        output_dir=(root / args.output_dir).resolve(),
    )
    tuning = report["tuning"]["selection"]["overall"]
    print(
        "D63-3 feature validation complete: "
        f"gate={report['gate_d63_3']['status']}, "
        f"tuning gain={tuning['mean_delta_delivered_mbit']:+.6f} Mbit/state, "
        f"changed={tuning['changed_actions']}."
    )
    return 0
