"""T4D compact causal QoS/risk probe built from the frozen T4C samples."""

from __future__ import annotations

from argparse import Namespace
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from ho_optim_drl.commands.audit_baseline_parity import _git_state, _sha256
from ho_optim_drl.commands.audit_early_timing_learnability import (
    EPS,
    RISK_FIELDS,
    _average_precision,
    _rule_wait,
    _selection,
    feature_names,
)
from ho_optim_drl.commands.audit_t4c_post_ttt_probe import (
    _paired_selection_bootstrap,
)


SCHEMA_NAME = "early_ho_t4d_compact_causal_probe"
SCHEMA_VERSION = 1
SOURCE_SCHEMA = "t4b_protocol_history_v1"
MODEL_NAMES = ("linear", "mlp_small")

# Deliberately omit speed, route progress, residence/recent-HO counters and all
# affine RLM duplicates.  These features were strong shortcuts on T4C-fit but
# did not transfer to held-out routes.
QOS_FEATURE_NAMES = (
    "a3_elapsed_s",
    "rsrp_delta_db_div_20",
    "serving_sinr_rate_db_div_30",
    "target_minus_serving_sinr_div_30",
    "delta_rsrp_trend_20ms_div_10",
    "delta_rsrp_trend_100ms_div_10",
    "delta_rsrp_trend_200ms_div_10",
    "delta_rsrp_trend_500ms_div_10",
    "delta_sinr_trend_200ms_div_20",
    "delta_sinr_trend_500ms_div_20",
    "delta_rsrp_std_100ms_div_10",
    "delta_rsrp_std_500ms_div_10",
    "serving_sinr_min_200ms_div_30",
    "target_sinr_min_200ms_div_30",
)

# Risk head receives the compact radio evidence plus protocol state that can
# causally precede an RLF/HO failure.  It does not receive route identifiers.
RISK_EXTRA_FEATURE_NAMES = (
    "n310_progress",
    "n311_progress",
    "t310_active",
    "t310_progress",
    "target_is_previous",
)
RISK_FEATURE_NAMES = (*QOS_FEATURE_NAMES, *RISK_EXTRA_FEATURE_NAMES)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def _feature_indices(selected: tuple[str, ...]) -> np.ndarray:
    available = feature_names(SOURCE_SCHEMA)
    missing = sorted(set(selected) - set(available))
    if missing:
        raise ValueError(f"Unknown compact features: {missing}")
    if len(selected) != len(set(selected)):
        raise ValueError("Compact feature list contains duplicates.")
    return np.asarray([available.index(name) for name in selected], dtype=np.int64)


def _route_weights(rows: list[dict[str, Any]], mask: np.ndarray) -> np.ndarray:
    """Give every route equal total weight inside the selected fold."""
    grouped: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, (row, keep) in enumerate(zip(rows, mask)):
        if keep:
            grouped[(int(row["speed_kmh"]), int(row["trajectory_id"]))].append(index)
    if not grouped:
        raise ValueError("Route-balanced weights require a non-empty fold.")
    weights = np.zeros(len(rows), dtype=np.float32)
    for indices in grouped.values():
        weights[indices] = 1.0 / len(indices)
    weights *= float(np.sum(mask)) / float(np.sum(weights))
    return weights


def _risk_worse(row: dict[str, Any]) -> bool:
    return any(float(row["label"]["risk_deltas"][name]) > EPS for name in RISK_FIELDS)


def _meaningful_wait(row: dict[str, Any], minimum_gain_mbit: float) -> bool:
    return bool(
        float(row["label"]["delta_delivered_mbit"]) >= minimum_gain_mbit
        and not _risk_worse(row)
    )


class _Head(nn.Module):
    def __init__(self, size: int, architecture: str) -> None:
        super().__init__()
        if architecture == "linear":
            self.network = nn.Linear(size, 1)
        elif architecture == "mlp_small":
            self.network = nn.Sequential(
                nn.Linear(size, 16), nn.Tanh(), nn.Linear(16, 1)
            )
        else:
            raise ValueError(architecture)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value)[:, 0]


def _weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return torch.sum(values * weights) / torch.sum(weights)


def _fit_heads(
    qos_x: np.ndarray,
    risk_x: np.ndarray,
    gain: np.ndarray,
    risk: np.ndarray,
    rows: list[dict[str, Any]],
    fit_mask: np.ndarray,
    tuning_mask: np.ndarray,
    *,
    architecture: str,
    seed: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    """Fit separate QoS regression and risk classification heads.

    Early stopping observes tuning loss, but confirmation is never evaluated in
    this function. Route-balanced fit loss prevents routes with more anchors
    from dominating the probe.
    """
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    fit_weights = torch.as_tensor(_route_weights(rows, fit_mask), dtype=torch.float32)
    fit = torch.as_tensor(fit_mask, dtype=torch.bool)
    tuning = torch.as_tensor(tuning_mask, dtype=torch.bool)

    qx = torch.as_tensor(qos_x, dtype=torch.float32)
    rx = torch.as_tensor(risk_x, dtype=torch.float32)
    q_mean = qx[fit].mean(0)
    q_std = qx[fit].std(0, unbiased=False).clamp_min(1e-6)
    r_mean = rx[fit].mean(0)
    r_std = rx[fit].std(0, unbiased=False).clamp_min(1e-6)
    q_scaled = (qx - q_mean) / q_std
    r_scaled = (rx - r_mean) / r_std

    gain_tensor = torch.as_tensor(gain, dtype=torch.float32)
    gain_mean = gain_tensor[fit].mean()
    gain_std = gain_tensor[fit].std(unbiased=False).clamp_min(1e-6)
    gain_scaled = (gain_tensor - gain_mean) / gain_std
    risk_tensor = torch.as_tensor(risk.astype(np.float32))
    positives = int(risk_tensor[fit].sum().item())
    negatives = int(fit.sum().item()) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("Fit fold must contain both safe and risk-worse WAIT rows.")

    qos_head = _Head(qos_x.shape[1], architecture)
    risk_head = _Head(risk_x.shape[1], architecture)
    parameters = [*qos_head.parameters(), *risk_head.parameters()]
    learning_rate = 0.01 if architecture == "linear" else 0.003
    optimizer = torch.optim.Adam(parameters, lr=learning_rate, weight_decay=1e-3)
    risk_loss = nn.BCEWithLogitsLoss(
        reduction="none", pos_weight=torch.tensor(negatives / positives)
    )
    epochs = 500 if architecture == "linear" else 800
    patience = 60
    best_loss = float("inf")
    best_epoch = 0
    best_qos: dict[str, torch.Tensor] | None = None
    best_risk: dict[str, torch.Tensor] | None = None
    stale = 0
    for epoch in range(epochs):
        predicted_gain = qos_head(q_scaled)
        predicted_risk = risk_head(r_scaled)
        qos_per_row = nn.functional.smooth_l1_loss(
            predicted_gain, gain_scaled, reduction="none", beta=0.5
        )
        risk_per_row = risk_loss(predicted_risk, risk_tensor)
        loss = _weighted_mean(qos_per_row[fit], fit_weights[fit]) + _weighted_mean(
            risk_per_row[fit], fit_weights[fit]
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            tune_loss = (
                nn.functional.smooth_l1_loss(
                    qos_head(q_scaled)[tuning], gain_scaled[tuning], beta=0.5
                )
                + risk_loss(risk_head(r_scaled)[tuning], risk_tensor[tuning]).mean()
            )
            tune_value = float(tune_loss.item())
        if tune_value < best_loss - 1e-7:
            best_loss = tune_value
            best_epoch = epoch + 1
            best_qos = {name: value.detach().clone() for name, value in qos_head.state_dict().items()}
            best_risk = {name: value.detach().clone() for name, value in risk_head.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_qos is None or best_risk is None:
        raise RuntimeError("T4D early stopping did not capture model weights.")
    qos_head.load_state_dict(best_qos)
    risk_head.load_state_dict(best_risk)
    qos_head.eval()
    risk_head.eval()
    with torch.no_grad():
        predicted_gain = (
            qos_head(q_scaled) * gain_std + gain_mean
        ).cpu().numpy()
        predicted_risk = torch.sigmoid(risk_head(r_scaled)).cpu().numpy()
    models = {"qos": qos_head, "risk": risk_head}
    arrays = {
        "qos_feature_mean": q_mean.cpu().numpy(),
        "qos_feature_std": q_std.cpu().numpy(),
        "risk_feature_mean": r_mean.cpu().numpy(),
        "risk_feature_std": r_std.cpu().numpy(),
        "gain_mean": np.asarray([float(gain_mean.item())], dtype=np.float32),
        "gain_std": np.asarray([float(gain_std.item())], dtype=np.float32),
        "predicted_gain": predicted_gain,
        "predicted_risk": predicted_risk,
    }
    config = {
        "architecture": architecture,
        "seed": seed,
        "maximum_epochs": epochs,
        "best_epoch": best_epoch,
        "early_stopping_patience": patience,
        "learning_rate": learning_rate,
        "weight_decay": 1e-3,
        "route_balanced_fit_loss": True,
        "risk_positive_weight": negatives / positives,
        "tuning_used_for_early_stopping": True,
    }
    return models, arrays, config


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    unique, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    del unique
    if np.any(counts > 1):
        for group, count in enumerate(counts):
            if count > 1:
                ranks[inverse == group] = float(np.mean(ranks[inverse == group]))
    return ranks


def _correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    if len(left) < 2 or np.std(left) <= EPS or np.std(right) <= EPS:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _regression_report(
    actual: np.ndarray, predicted: np.ndarray, mask: np.ndarray
) -> dict[str, Any]:
    truth = actual[mask]
    estimate = predicted[mask]
    return {
        "rows": int(len(truth)),
        "mae_mbit": float(np.mean(np.abs(estimate - truth))),
        "rmse_mbit": float(np.sqrt(np.mean((estimate - truth) ** 2))),
        "pearson": _correlation(estimate, truth),
        "spearman": _correlation(_rank(estimate), _rank(truth)),
        "predicted_mean_mbit": float(np.mean(estimate)),
        "actual_mean_mbit": float(np.mean(truth)),
    }


def _classification_report(
    labels: np.ndarray, scores: np.ndarray, mask: np.ndarray, threshold: float
) -> dict[str, Any]:
    truth = labels[mask]
    estimate = scores[mask]
    predicted = estimate >= threshold
    tp = int(np.sum(truth & predicted))
    fp = int(np.sum(~truth & predicted))
    fn = int(np.sum(truth & ~predicted))
    return {
        "rows": int(len(truth)),
        "positive_rows": int(np.sum(truth)),
        "prevalence": float(np.mean(truth)),
        "average_precision": float(_average_precision(truth.astype(np.int8), estimate)),
        "threshold": float(threshold),
        "predicted_positive_rows": int(np.sum(predicted)),
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "brier_score": float(np.mean((estimate - truth.astype(float)) ** 2)),
    }


def _lock_decision_thresholds(
    labels: list[dict[str, Any]],
    predicted_gain: np.ndarray,
    predicted_risk: np.ndarray,
    *,
    minimum_gain_mbit: float,
) -> tuple[float, float, dict[str, Any]]:
    gain_candidates = sorted(set(
        [minimum_gain_mbit, 0.10, 0.20, 0.50]
        + [float(value) for value in np.quantile(predicted_gain, [0.5, 0.7, 0.8, 0.9, 0.95])]
    ))
    gain_candidates = [value for value in gain_candidates if value >= minimum_gain_mbit]
    risk_candidates = (0.05, 0.10, 0.20, 0.35, 0.50)
    records = []
    for gain_threshold in gain_candidates:
        for risk_threshold in risk_candidates:
            choose = (predicted_gain >= gain_threshold) & (
                predicted_risk <= risk_threshold
            )
            records.append({
                "gain_threshold_mbit": float(gain_threshold),
                "risk_threshold": float(risk_threshold),
                **_selection(labels, choose),
            })
    feasible = [row for row in records if row["reliability_not_worse"]]
    if not feasible:
        raise RuntimeError("No reliability-feasible T4D threshold pair on tuning.")
    selected = max(feasible, key=lambda row: (
        row["mean_delta_delivered_mbit"],
        -row["false_wait_actions"],
        -row["wait_actions"],
        row["gain_threshold_mbit"],
        -row["risk_threshold"],
    ))
    return (
        float(selected["gain_threshold_mbit"]),
        float(selected["risk_threshold"]),
        {"selected": selected, "candidates": records},
    )


def _route_bootstrap_correlation(
    rows: list[dict[str, Any]],
    actual: np.ndarray,
    predicted: np.ndarray,
    *,
    seed: int,
    resamples: int = 2000,
) -> dict[str, Any]:
    by_speed: dict[int, dict[tuple[int, int], list[int]]] = defaultdict(lambda: defaultdict(list))
    for index, row in enumerate(rows):
        by_speed[int(row["speed_kmh"])][
            (int(row["speed_kmh"]), int(row["trajectory_id"]))
        ].append(index)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(resamples):
        indices: list[int] = []
        for speed in sorted(by_speed):
            routes = list(by_speed[speed].values())
            for pick in rng.integers(0, len(routes), len(routes)):
                indices.extend(routes[int(pick)])
        value = _correlation(predicted[indices], actual[indices])
        values.append(0.0 if value is None else value)
    observed = _correlation(predicted, actual)
    return {
        "routes": int(sum(len(group) for group in by_speed.values())),
        "stratified_by_speed": True,
        "resamples": resamples,
        "seed": seed,
        "pearson": observed,
        "ci95": [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))],
    }


def run_probe(
    *,
    root: Path,
    t4c_dir: Path,
    output_dir: Path,
    model_seed: int,
    minimum_gain_mbit: float,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite T4D output: {output_dir}")
    if minimum_gain_mbit <= 0:
        raise ValueError("minimum_gain_mbit must be positive.")
    source_report = _read_json(t4c_dir / "t4c_report.json")
    if (
        source_report.get("status") != "COMPLETE"
        or source_report.get("validation_or_test_rows_read") != 0
        or source_report.get("feature_contract", {}).get("schema") != SOURCE_SCHEMA
        or source_report.get("split_contract", {}).get("confirmation_used_for_fit") is not False
    ):
        raise ValueError("T4D requires a complete, TRAIN-only frozen T4C artifact.")
    rows = _read_jsonl(t4c_dir / "samples.jsonl")
    if len(rows) != int(source_report["sample_counts"]["total"]):
        raise ValueError("T4C sample count does not match its report.")
    full = np.asarray([row["feature"] for row in rows], dtype=np.float32)
    qos_x = full[:, _feature_indices(QOS_FEATURE_NAMES)]
    risk_x = full[:, _feature_indices(RISK_FEATURE_NAMES)]
    gain = np.asarray([
        row["label"]["delta_delivered_mbit"] for row in rows
    ], dtype=np.float32)
    risk = np.asarray([_risk_worse(row) for row in rows], dtype=bool)
    meaningful = np.asarray([
        _meaningful_wait(row, minimum_gain_mbit) for row in rows
    ], dtype=bool)
    masks = {
        fold: np.asarray([row["fold"] == fold for row in rows], dtype=bool)
        for fold in ("fit", "tuning", "confirmation")
    }
    if any(not np.any(mask) for mask in masks.values()):
        raise ValueError("T4D requires non-empty frozen folds.")

    fit_reports: dict[str, Any] = {}
    frozen_arrays: dict[str, np.ndarray] = {}
    frozen_models: dict[str, tuple[dict[str, Any], dict[str, np.ndarray]]] = {}
    tuning_rows = [row["label"] for row, keep in zip(rows, masks["tuning"]) if keep]
    for index, architecture in enumerate(MODEL_NAMES):
        models, arrays, config = _fit_heads(
            qos_x, risk_x, gain, risk, rows, masks["fit"], masks["tuning"],
            architecture=architecture, seed=model_seed + index,
        )
        tune_gain = arrays["predicted_gain"][masks["tuning"]]
        tune_risk = arrays["predicted_risk"][masks["tuning"]]
        gain_threshold, risk_threshold, threshold_report = _lock_decision_thresholds(
            tuning_rows, tune_gain, tune_risk,
            minimum_gain_mbit=minimum_gain_mbit,
        )
        fit_reports[architecture] = {
            "fit_config": config,
            "threshold_selection_on_tuning": threshold_report,
        }
        frozen_models[architecture] = (models, arrays)
        frozen_arrays[f"{architecture}__gain_threshold_mbit"] = np.asarray(
            [gain_threshold], dtype=np.float32
        )
        frozen_arrays[f"{architecture}__risk_threshold"] = np.asarray(
            [risk_threshold], dtype=np.float32
        )
        for name, value in arrays.items():
            frozen_arrays[f"{architecture}__{name}"] = value
        for head_name, model in models.items():
            for name, tensor in model.state_dict().items():
                frozen_arrays[
                    f"{architecture}__{head_name}__{name}"
                ] = tensor.detach().cpu().numpy()

    # Architecture is selected using tuning only; confirmation remains sealed.
    selected_architecture = max(MODEL_NAMES, key=lambda architecture: (
        fit_reports[architecture]["threshold_selection_on_tuning"]["selected"]
        ["mean_delta_delivered_mbit"],
        -fit_reports[architecture]["threshold_selection_on_tuning"]["selected"]
        ["false_wait_actions"],
    ))
    confirmation_mask = masks["confirmation"]
    confirmation_rows = [row for row, keep in zip(rows, confirmation_mask) if keep]
    confirmation_labels = [row["label"] for row in confirmation_rows]
    model_reports: dict[str, Any] = {}
    for architecture in MODEL_NAMES:
        arrays = frozen_models[architecture][1]
        selected = fit_reports[architecture]["threshold_selection_on_tuning"]["selected"]
        gain_threshold = float(selected["gain_threshold_mbit"])
        risk_threshold = float(selected["risk_threshold"])
        predicted_gain = arrays["predicted_gain"]
        predicted_risk = arrays["predicted_risk"]
        choose = (
            predicted_gain[confirmation_mask] >= gain_threshold
        ) & (predicted_risk[confirmation_mask] <= risk_threshold)
        model_reports[architecture] = {
            **fit_reports[architecture],
            "qos_regression_confirmation": _regression_report(
                gain, predicted_gain, confirmation_mask
            ),
            "meaningful_wait_confirmation": _classification_report(
                meaningful,
                predicted_gain,
                confirmation_mask,
                minimum_gain_mbit,
            ),
            "risk_confirmation": _classification_report(
                risk, predicted_risk, confirmation_mask, risk_threshold
            ),
            "selection_confirmation": _selection(confirmation_labels, choose),
            "qos_correlation_route_bootstrap": _route_bootstrap_correlation(
                confirmation_rows,
                gain[confirmation_mask],
                predicted_gain[confirmation_mask],
                seed=model_seed + 100 + MODEL_NAMES.index(architecture),
            ),
        }
        # Existing route utility bootstrap is equivalent to comparison with
        # always-HO-now (all-false selector).
        model_reports[architecture]["selection_vs_ho_now"] = _paired_selection_bootstrap(
            confirmation_rows,
            np.zeros(len(confirmation_rows), dtype=bool),
            choose,
            seed=model_seed + 200 + MODEL_NAMES.index(architecture),
        )

    source_names = feature_names(SOURCE_SCHEMA)
    confirmation_features = full[confirmation_mask]
    rule_choose = np.asarray([
        _rule_wait(feature, "target_sinr_worse", names=source_names)
        for feature in confirmation_features
    ], dtype=bool)
    rule_report = {
        "name": "target_sinr_worse",
        "selection_confirmation": _selection(confirmation_labels, rule_choose),
        "selection_vs_ho_now": _paired_selection_bootstrap(
            confirmation_rows,
            np.zeros(len(confirmation_rows), dtype=bool),
            rule_choose,
            seed=model_seed + 300,
        ),
    }
    selected_arrays = frozen_models[selected_architecture][1]
    selected_thresholds = fit_reports[selected_architecture][
        "threshold_selection_on_tuning"
    ]["selected"]
    selected_choose = (
        selected_arrays["predicted_gain"][confirmation_mask]
        >= float(selected_thresholds["gain_threshold_mbit"])
    ) & (
        selected_arrays["predicted_risk"][confirmation_mask]
        <= float(selected_thresholds["risk_threshold"])
    )
    selected_vs_rule = _paired_selection_bootstrap(
        confirmation_rows,
        rule_choose,
        selected_choose,
        seed=model_seed + 400,
    )
    primary = model_reports[selected_architecture]
    regression = primary["qos_regression_confirmation"]
    ranking = primary["meaningful_wait_confirmation"]
    utility = primary["selection_vs_ho_now"]
    g_qos_signal = bool(
        regression["pearson"] is not None
        and regression["spearman"] is not None
        and regression["pearson"] >= 0.10
        and regression["spearman"] >= 0.10
        and primary["qos_correlation_route_bootstrap"]["ci95"][0] > 0
    )
    g_meaningful_ranking = bool(
        ranking["average_precision"] >= ranking["prevalence"] + 0.05
    )
    g_local_utility = bool(
        primary["selection_confirmation"]["reliability_not_worse"]
        and utility["ci95"]["delta_delivered_mbit"][0] > 0
    )
    g_beats_rule = bool(
        selected_vs_rule["ci95"]["delta_delivered_mbit"][0] > 0
        and all(
            selected_vs_rule["ci95"][field][1] <= 0 for field in RISK_FIELDS
        )
    )
    local_pass = bool(g_qos_signal and g_meaningful_ranking and g_local_utility)

    output_dir.mkdir(parents=True)
    np.savez_compressed(output_dir / "frozen_models.npz", **frozen_arrays)
    report = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "status": "COMPLETE",
        "scope": "train_fit_tuning_confirmation_compact_local_probe",
        "validation_or_test_rows_read": 0,
        "source": {
            "t4c_report": str((t4c_dir / "t4c_report.json").resolve()),
            "t4c_report_sha256": _sha256(t4c_dir / "t4c_report.json"),
            "samples": str((t4c_dir / "samples.jsonl").resolve()),
            "samples_sha256": _sha256(t4c_dir / "samples.jsonl"),
        },
        "feature_contract": {
            "source_schema": SOURCE_SCHEMA,
            "qos_features": list(QOS_FEATURE_NAMES),
            "risk_features": list(RISK_FEATURE_NAMES),
            "causal": True,
            "future_features": [],
            "excluded_shortcuts": [
                "speed_kmh_div_150", "route_progress",
                "serving_residence_s_div_10", "time_since_last_ho_s_div_10",
                "recent_ho_count_div_4", "candidate_persistence_500ms",
            ],
            "excluded_affine_duplicates": [
                "serving_rlm_margin_qout_div_20",
                "target_rlm_margin_qout_div_20",
                "serving_rlm_min_margin_qout_div_20",
                "target_rlm_min_margin_qout_div_20",
            ],
        },
        "label_contract": {
            "qos_target": "WAIT_20_minus_HO_NOW_delivered_mbit_at_1s",
            "risk_target": "any_WAIT_20_minus_HO_NOW_risk_delta_gt_0",
            "meaningful_wait": (
                f"delta_delivered_mbit >= {minimum_gain_mbit} and no risk worse"
            ),
            "minimum_meaningful_gain_mbit": minimum_gain_mbit,
        },
        "split_contract": {
            "inherited_from_t4c": True,
            "fit_used_for_weights": True,
            "tuning_used_for_early_stopping": True,
            "tuning_used_for_model_and_threshold_selection": True,
            "confirmation_used_for_fit": False,
            "confirmation_used_for_threshold": False,
            "confirmation_used_for_architecture_selection": False,
            "counts": {fold: int(np.sum(mask)) for fold, mask in masks.items()},
            "meaningful_positive_counts": {
                fold: int(np.sum(meaningful & mask)) for fold, mask in masks.items()
            },
            "risk_positive_counts": {
                fold: int(np.sum(risk & mask)) for fold, mask in masks.items()
            },
        },
        "models": model_reports,
        "selected_architecture_on_tuning": selected_architecture,
        "simple_rule_reference": rule_report,
        "selected_minus_simple_rule": selected_vs_rule,
        "gates": {
            "g_qos_signal": g_qos_signal,
            "g_meaningful_ranking": g_meaningful_ranking,
            "g_local_utility": g_local_utility,
            "g_beats_simple_rule": g_beats_rule,
            "overall_local_probe": "PASSED" if local_pass else "NOT_PASSED",
            "ppo_status": "CLOSED",
            "next_gate": (
                "compact full-route TRAIN audit" if local_pass
                else "revise compact target/features; do not train PPO"
            ),
        },
    }
    _write_json(output_dir / "t4d_report.json", report)
    manifest = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "validation_or_test_rows_read": 0,
        "model_seed": model_seed,
        "minimum_meaningful_gain_mbit": minimum_gain_mbit,
        "inputs": report["source"],
        "outputs": {
            name: {"path": str((output_dir / name).resolve()), "sha256": _sha256(output_dir / name)}
            for name in ("t4d_report.json", "frozen_models.npz")
        },
        "source_files": [
            {
                "path": path,
                "sha256": _sha256(root / path),
            }
            for path in (
                "src/ho_optim_drl/commands/audit_t4d_compact_causal_probe.py",
                "src/ho_optim_drl/commands/audit_t4c_post_ttt_probe.py",
                "src/ho_optim_drl/commands/audit_early_timing_learnability.py",
            )
        ],
        "git": _git_state(root),
    }
    _write_json(output_dir / "run_manifest.json", manifest)
    primary_selection = primary["selection_confirmation"]
    lines = [
        "# T4D compact causal QoS/risk probe",
        "",
        f"- Model được chọn chỉ bằng tuning: **{selected_architecture}**.",
        f"- Feature QoS/risk: **{len(QOS_FEATURE_NAMES)}/{len(RISK_FEATURE_NAMES)}**; minimum meaningful gain: **{minimum_gain_mbit:.3f} Mbit**.",
        f"- Confirmation QoS Pearson/Spearman: **{regression['pearson']} / {regression['spearman']}**.",
        f"- Meaningful AP/prevalence: **{ranking['average_precision']:.4f} / {ranking['prevalence']:.4f}**.",
        f"- Selection gain: **{primary_selection['mean_delta_delivered_mbit']:.6f} Mbit/state**; WAIT: **{primary_selection['wait_actions']}**.",
        f"- Gate QoS/ranking/utility/rule: **{g_qos_signal}/{g_meaningful_ranking}/{g_local_utility}/{g_beats_rule}**.",
        f"- Kết luận local: **{report['gates']['overall_local_probe']}**; PPO: **CLOSED**.",
        "",
        "Confirmation không tham gia fit, early stopping, chọn architecture hoặc khóa threshold.",
    ]
    (output_dir / "report_vi.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main(root_path: str, args: Namespace) -> int:
    root = Path(root_path).resolve()
    resolve = lambda value: (
        Path(value) if Path(value).is_absolute() else root / value
    ).resolve()
    report = run_probe(
        root=root,
        t4c_dir=resolve(args.t4c_dir),
        output_dir=resolve(args.output_dir),
        model_seed=int(args.model_seed),
        minimum_gain_mbit=float(args.minimum_meaningful_gain_mbit),
    )
    selected = report["selected_architecture_on_tuning"]
    print(json.dumps({
        "status": report["status"],
        "selected_architecture": selected,
        "confirmation": report["models"][selected]["selection_confirmation"],
        "gates": report["gates"],
    }, indent=2), flush=True)
    return 0
