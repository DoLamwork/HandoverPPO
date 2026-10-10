"""Fit leakage-safe linear/MLP probes on the frozen D63-2 TRAIN labels."""

from __future__ import annotations

from argparse import Namespace
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from ho_optim_drl.commands.audit_policy_learning import (
    _average_precision,
    _sha256_file,
)


SCHEMA = "d63_2_learnability_probe_v1"
FAMILIES = ("timing", "target", "joint")
MODELS = ("linear", "mlp_small")
RISK_FIELDS = (
    "delta_handover_failures",
    "delta_rlf_starts",
    "delta_pure_rlf_starts",
    "delta_ping_pong_count",
    "delta_recovery_seconds",
    "delta_handover_interruption_seconds",
)
EPS = 1e-9


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


def build_features(arrays: dict[str, np.ndarray]) -> np.ndarray:
    """Use only causal observation plus the planned delay/slot contract."""
    observation = arrays["observation"].astype(np.float32)
    delay = arrays["planned_delay_ms"].astype(np.float32)[:, None] / 300.0
    slots = arrays["planned_slot"].astype(np.int64)
    one_hot = np.zeros((len(slots), 3), dtype=np.float32)
    one_hot[np.arange(len(slots)), slots - 1] = 1.0
    return np.concatenate((observation, delay, one_hot), axis=1)


def _classification_metrics(
    labels: np.ndarray, scores: np.ndarray, threshold: float
) -> dict[str, Any]:
    labels = labels.astype(bool)
    predicted = scores >= threshold
    tp = int(np.sum(labels & predicted))
    fp = int(np.sum(~labels & predicted))
    fn = int(np.sum(labels & ~predicted))
    bins = np.minimum((scores * 10).astype(int), 9)
    ece = 0.0
    calibration = []
    for bin_index in range(10):
        mask = bins == bin_index
        if not np.any(mask):
            continue
        mean_score = float(np.mean(scores[mask]))
        fraction_positive = float(np.mean(labels[mask]))
        ece += float(np.mean(mask)) * abs(mean_score - fraction_positive)
        calibration.append(
            {
                "bin": bin_index,
                "rows": int(np.sum(mask)),
                "mean_score": mean_score,
                "positive_fraction": fraction_positive,
            }
        )
    return {
        "rows": int(len(labels)),
        "positive_rows": int(np.sum(labels)),
        "prevalence": float(np.mean(labels)),
        "average_precision": _average_precision(labels.astype(np.int8), scores),
        "threshold": float(threshold),
        "predicted_positive_rows": int(np.sum(predicted)),
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "brier_score": float(np.mean((scores - labels.astype(float)) ** 2)),
        "expected_calibration_error_10bin": float(ece),
        "calibration": calibration,
    }


class _MultiTaskProbe(nn.Module):
    def __init__(self, input_size: int, architecture: str):
        super().__init__()
        output_size = 2 + len(RISK_FIELDS)  # useful logit + gain + risks
        if architecture == "linear":
            self.network = nn.Linear(input_size, output_size)
        elif architecture == "mlp_small":
            self.network = nn.Sequential(
                nn.Linear(input_size, 32), nn.Tanh(), nn.Linear(32, output_size)
            )
        else:
            raise ValueError(architecture)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


def fit_probe(
    features: np.ndarray,
    useful: np.ndarray,
    continuous: np.ndarray,
    fit_mask: np.ndarray,
    *,
    architecture: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Fit one fixed multitask configuration without consulting tuning/holdout."""
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    x = torch.as_tensor(features, dtype=torch.float32)
    fit = torch.as_tensor(fit_mask, dtype=torch.bool)
    x_mean = x[fit].mean(dim=0)
    x_std = x[fit].std(dim=0, unbiased=False).clamp_min(1e-6)
    x_scaled = (x - x_mean) / x_std
    y_binary = torch.as_tensor(useful, dtype=torch.float32)
    y_continuous = torch.as_tensor(continuous, dtype=torch.float32)
    y_mean = y_continuous[fit].mean(dim=0)
    y_std = y_continuous[fit].std(dim=0, unbiased=False).clamp_min(1e-6)
    y_scaled = (y_continuous - y_mean) / y_std
    positives = int(y_binary[fit].sum().item())
    negatives = int(fit.sum().item()) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("Fit fold must contain both useful and non-useful rows.")
    model = _MultiTaskProbe(features.shape[1], architecture)
    learning_rate = 0.02 if architecture == "linear" else 0.005
    epochs = 350 if architecture == "linear" else 500
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    binary_loss = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(negatives / positives, dtype=torch.float32)
    )
    for _ in range(epochs):
        output = model(x_scaled[fit])
        loss_useful = binary_loss(output[:, 0], y_binary[fit])
        loss_continuous = torch.mean((output[:, 1:] - y_scaled[fit]) ** 2)
        loss = loss_useful + 0.25 * loss_continuous
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        output = model(x_scaled)
        scores = torch.sigmoid(output[:, 0]).cpu().numpy()
        continuous_prediction = (output[:, 1:] * y_std + y_mean).cpu().numpy()
    return scores, continuous_prediction, {
        "architecture": architecture,
        "hidden_size": None if architecture == "linear" else 32,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "weight_decay": 1e-4,
        "multitask_continuous_loss_weight": 0.25,
        "positive_class_weight": negatives / positives,
        "seed": seed,
    }


def choose_actions(
    arrays: dict[str, np.ndarray], scores: np.ndarray, threshold: float
) -> np.ndarray:
    """Choose at most one branch/state; fall back to registered control."""
    chosen: list[int] = []
    for state_id in np.unique(arrays["state_id"]):
        indices = np.flatnonzero(arrays["state_id"] == state_id)
        control = indices[arrays["is_control"][indices]][0]
        eligible = indices[
            (~arrays["is_control"][indices])
            & arrays["requested_action_valid"][indices]
            & (scores[indices] >= threshold)
        ]
        if len(eligible):
            chosen.append(int(eligible[np.argmax(scores[eligible])]))
        else:
            chosen.append(int(control))
    return np.asarray(chosen, dtype=np.int64)


def _selection_metrics(
    arrays: dict[str, np.ndarray], scores: np.ndarray, threshold: float
) -> dict[str, Any]:
    chosen = choose_actions(arrays, scores, threshold)
    risks = {field: float(np.mean(arrays[field][chosen])) for field in RISK_FIELDS}
    available_gain = 0.0
    for state_id in np.unique(arrays["state_id"]):
        state_mask = arrays["state_id"] == state_id
        useful_indices = np.flatnonzero(state_mask & arrays["strict_useful"])
        if len(useful_indices):
            available_gain += max(
                0.0, float(np.max(arrays["delta_delivered_mbit"][useful_indices]))
            )
    captured_positive_gain = float(
        np.sum(np.maximum(arrays["delta_delivered_mbit"][chosen], 0.0))
    )
    return {
        "states": int(len(chosen)),
        "changed_actions": int(np.sum(~arrays["is_control"][chosen])),
        "mean_delta_delivered_mbit": float(np.mean(arrays["delta_delivered_mbit"][chosen])),
        "total_delta_delivered_mbit": float(np.sum(arrays["delta_delivered_mbit"][chosen])),
        "mean_delta_relative_rate_percent": float(np.mean(arrays["delta_relative_rate_percent"][chosen])),
        "risk_deltas": risks,
        "reliability_not_worse": all(value <= EPS for value in risks.values()),
        "strict_useful_selections": int(np.sum(arrays["strict_useful"][chosen])),
        "false_alternative_selections": int(np.sum(
            (~arrays["is_control"][chosen]) & (~arrays["strict_useful"][chosen])
        )),
        "missed_gain_mbit": float(available_gain - captured_positive_gain),
    }


def select_threshold(
    arrays: dict[str, np.ndarray], scores: np.ndarray
) -> tuple[float, dict[str, Any]]:
    """Lock threshold on tuning: maximize gain subject to non-worse mean risks."""
    candidates = np.concatenate((np.linspace(0.05, 0.95, 19), np.asarray([1.000001])))
    evaluated = []
    for threshold in candidates:
        metrics = _selection_metrics(arrays, scores, float(threshold))
        evaluated.append({"threshold": float(threshold), **metrics})
    feasible = [record for record in evaluated if record["reliability_not_worse"]]
    best = max(
        feasible,
        key=lambda record: (
            record["mean_delta_delivered_mbit"],
            -record["false_alternative_selections"],
            -record["changed_actions"],
            record["threshold"],
        ),
    )
    return float(best["threshold"]), {"selected": best, "candidates": evaluated}


def _subset(arrays: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, np.ndarray]:
    return {name: values[mask] for name, values in arrays.items()}


def _route_bootstrap_selection(
    arrays: dict[str, np.ndarray], scores: np.ndarray, threshold: float, *, seed: int
) -> dict[str, Any]:
    route_keys = np.asarray([
        f"{int(speed)}:{int(route)}"
        for speed, route in zip(arrays["speed_kmh"], arrays["trajectory_id"])
    ])
    routes = np.unique(route_keys)
    per_route = []
    for route in routes:
        mask = route_keys == route
        metrics = _selection_metrics(_subset(arrays, mask), scores[mask], threshold)
        per_route.append(metrics["mean_delta_delivered_mbit"])
    values = np.asarray(per_route, dtype=float)
    rng = np.random.default_rng(seed)
    boot = np.asarray([
        np.mean(values[rng.integers(0, len(values), size=len(values))])
        for _ in range(2000)
    ])
    return {
        "unit": "route",
        "route_count": int(len(routes)),
        "resamples": 2000,
        "mean": float(np.mean(values)),
        "p2_5": float(np.percentile(boot, 2.5)),
        "median": float(np.median(boot)),
        "p97_5": float(np.percentile(boot, 97.5)),
    }


def _continuous_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    names = ("delta_delivered_mbit", *RISK_FIELDS)
    return {
        name: {
            "mae": float(np.mean(np.abs(actual[:, index] - predicted[:, index]))),
            "rmse": float(np.sqrt(np.mean((actual[:, index] - predicted[:, index]) ** 2))),
        }
        for index, name in enumerate(names)
    }


def run_probe(*, labels_dir: Path, output_dir: Path, seed: int) -> dict[str, Any]:
    _prepare_destination(output_dir)
    label_report_path = labels_dir / "label_report.json"
    registry_path = labels_dir / "input_registry.json"
    split_path = labels_dir / "route_split_manifest.json"
    for path in (label_report_path, registry_path, split_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    if registry["constraints"]["validation_test_zero_shot_used"]:
        raise ValueError("D63-2 probe accepts TRAIN-only labels only.")
    expected_allowed = ["observation", "planned_delay_ms", "planned_slot"]
    if registry["predictor_input_contract"]["allowed"] != expected_allowed:
        raise ValueError("Unexpected D63-2 predictor input contract.")

    family_reports: dict[str, Any] = {}
    score_artifacts: dict[str, np.ndarray] = {}
    for family_index, family in enumerate(FAMILIES):
        label_path = labels_dir / f"{family}_labels.npz"
        arrays = _load_npz(label_path)
        features = build_features(arrays)
        useful = arrays["strict_useful"].astype(np.float32)
        continuous = np.column_stack(
            (arrays["delta_delivered_mbit"], *(arrays[field] for field in RISK_FIELDS))
        ).astype(np.float32)
        masks = {fold: arrays["fold"] == fold for fold in ("fit", "tuning", "holdout")}
        if any(not np.any(mask) for mask in masks.values()):
            raise ValueError(f"{family}: an expected route fold is empty.")
        models = {}
        for model_index, architecture in enumerate(MODELS):
            scores, prediction, config = fit_probe(
                features,
                useful,
                continuous,
                masks["fit"],
                architecture=architecture,
                seed=seed + family_index * 10 + model_index,
            )
            tuning_arrays = _subset(arrays, masks["tuning"])
            threshold, tuning_selection = select_threshold(
                tuning_arrays, scores[masks["tuning"]]
            )
            holdout_arrays = _subset(arrays, masks["holdout"])
            holdout_scores = scores[masks["holdout"]]
            holdout_selection = _selection_metrics(
                holdout_arrays, holdout_scores, threshold
            )
            holdout_selection["route_bootstrap_gain_95_interval"] = (
                _route_bootstrap_selection(
                    holdout_arrays,
                    holdout_scores,
                    threshold,
                    seed=seed + 100 + family_index * 10 + model_index,
                )
            )
            groups = {
                "overall": _classification_metrics(
                    useful[masks["holdout"]], holdout_scores, threshold
                )
            }
            for speed in sorted(set(arrays["speed_kmh"][masks["holdout"]].tolist())):
                group = masks["holdout"] & (arrays["speed_kmh"] == speed)
                groups[f"speed:{int(speed)}"] = _classification_metrics(
                    useful[group], scores[group], threshold
                )
            for source in sorted(set(arrays["source"][masks["holdout"]].tolist())):
                group = masks["holdout"] & (arrays["source"] == source)
                groups[f"source:{source}"] = _classification_metrics(
                    useful[group], scores[group], threshold
                )
            models[architecture] = {
                "fit_config": config,
                "tuning_threshold_selection": tuning_selection,
                "holdout_classification": groups,
                "holdout_continuous_prediction": _continuous_metrics(
                    continuous[masks["holdout"]], prediction[masks["holdout"]]
                ),
                "holdout_action_selection": holdout_selection,
            }
            score_artifacts[f"{family}__{architecture}__score"] = scores.astype(np.float32)
            score_artifacts[f"{family}__{architecture}__continuous"] = prediction.astype(np.float32)
        family_reports[family] = {
            "label_file": label_path.name,
            "label_sha256": _sha256_file(label_path),
            "rows": int(len(useful)),
            "feature_count": int(features.shape[1]),
            "fold_rows": {fold: int(np.sum(mask)) for fold, mask in masks.items()},
            "fold_positive_rows": {fold: int(np.sum(useful[mask])) for fold, mask in masks.items()},
            "models": models,
        }

    scores_path = output_dir / "probe_scores.npz"
    np.savez_compressed(scores_path, **score_artifacts)
    report = {
        "schema_name": SCHEMA,
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "split": "train_route_fit_tuning_holdout",
        "validation_or_test_rows_read": 0,
        "route_split_seed": 29,
        "model_seed": seed,
        "holdout_access": "opened only after fit and tuning threshold were fixed in-process",
        "input_contract": expected_allowed,
        "forbidden_features_used": [],
        "inputs": {
            "label_report": {"path": str(label_report_path.resolve()), "sha256": _sha256_file(label_report_path)},
            "input_registry": {"path": str(registry_path.resolve()), "sha256": _sha256_file(registry_path)},
            "route_split_manifest": {"path": str(split_path.resolve()), "sha256": _sha256_file(split_path)},
        },
        "families": family_reports,
        "scores": {"path": scores_path.name, "sha256": _sha256_file(scores_path)},
    }
    report_path = output_dir / "probe_report.json"
    _json_write(report_path, report)
    manifest = {
        "schema_name": "d63_2_probe_manifest_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": [
            {"path": path.name, "size_bytes": path.stat().st_size, "sha256": _sha256_file(path)}
            for path in sorted(output_dir.iterdir()) if path.is_file()
        ],
    }
    _json_write(output_dir / "probe_manifest.json", manifest)
    return report


def main(root_path: str, args: Namespace) -> int:
    root = Path(root_path)
    report = run_probe(
        labels_dir=(root / args.labels_dir).resolve(),
        output_dir=(root / args.output_dir).resolve(),
        seed=int(args.seed),
    )
    for family, family_report in report["families"].items():
        values = []
        for model in MODELS:
            record = family_report["models"][model]
            ap = record["holdout_classification"]["overall"]["average_precision"]
            gain = record["holdout_action_selection"]["mean_delta_delivered_mbit"]
            values.append(f"{model}: AP={ap:.3f}, gain={gain:+.4f} Mbit")
        print(f"{family}: " + "; ".join(values))
    return 0
