"""T4D.1 material-gain ranking probe with calibrated risk selection."""

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
from ho_optim_drl.commands.audit_t4d_compact_causal_probe import (
    QOS_FEATURE_NAMES,
    RISK_FEATURE_NAMES,
    SOURCE_SCHEMA,
    _classification_report,
    _feature_indices,
    _route_weights,
)


SCHEMA_NAME = "early_ho_t4d1_ranked_probe"
SCHEMA_VERSION = 1


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _risk_worse(row: dict[str, Any]) -> bool:
    return any(float(row["label"]["risk_deltas"][name]) > EPS for name in RISK_FIELDS)


def _inner_route_masks(
    rows: list[dict[str, Any]], fit_mask: np.ndarray, *, seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a deterministic, speed-stratified 80/20 route split inside fit."""
    by_speed: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for row, keep in zip(rows, fit_mask):
        if keep:
            route = (int(row["speed_kmh"]), int(row["trajectory_id"]))
            if route not in by_speed[route[0]]:
                by_speed[route[0]].append(route)
    rng = np.random.default_rng(seed)
    validation_routes: set[tuple[int, int]] = set()
    for speed in sorted(by_speed):
        routes = sorted(by_speed[speed])
        count = max(1, int(round(0.20 * len(routes))))
        picks = rng.permutation(len(routes))[:count]
        validation_routes.update(routes[int(index)] for index in picks)
    inner_validation = np.asarray([
        bool(keep) and (int(row["speed_kmh"]), int(row["trajectory_id"])) in validation_routes
        for row, keep in zip(rows, fit_mask)
    ])
    inner_train = fit_mask & ~inner_validation
    if not np.any(inner_train) or not np.any(inner_validation):
        raise ValueError("T4D.1 inner route split is empty.")
    return inner_train, inner_validation


class _BinaryHead(nn.Module):
    def __init__(self, size: int) -> None:
        super().__init__()
        self.network = nn.Linear(size, 1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value)[:, 0]


def _weighted_mean(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return torch.sum(value * weight) / torch.sum(weight)


def _ranking_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    speeds: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Rank material-gain states above non-material states within each speed."""
    losses = []
    for speed in torch.unique(speeds[mask]):
        group = mask & (speeds == speed)
        positive = logits[group & labels]
        negative = logits[group & ~labels]
        if len(positive) and len(negative):
            losses.append(torch.nn.functional.softplus(
                -(positive[:, None] - negative[None, :])
            ).mean())
    if not losses:
        raise ValueError("Ranking loss requires both classes within a speed group.")
    return torch.stack(losses).mean()


def _train_epoch_range(
    qos_head: _BinaryHead,
    risk_head: _BinaryHead,
    q_scaled: torch.Tensor,
    r_scaled: torch.Tensor,
    material: torch.Tensor,
    risk: torch.Tensor,
    speeds: torch.Tensor,
    weights: torch.Tensor,
    train_mask: torch.Tensor,
    *,
    epochs: int,
    learning_rate: float,
    material_positive_weight: float,
    risk_positive_weight: float,
) -> None:
    optimizer = torch.optim.Adam(
        [*qos_head.parameters(), *risk_head.parameters()],
        lr=learning_rate,
        weight_decay=2e-3,
    )
    qos_bce = nn.BCEWithLogitsLoss(
        reduction="none", pos_weight=torch.tensor(material_positive_weight)
    )
    risk_bce = nn.BCEWithLogitsLoss(
        reduction="none", pos_weight=torch.tensor(risk_positive_weight)
    )
    for _ in range(epochs):
        qos_logits = qos_head(q_scaled)
        risk_logits = risk_head(r_scaled)
        classification = _weighted_mean(
            qos_bce(qos_logits, material.float())[train_mask], weights[train_mask]
        )
        risk_classification = _weighted_mean(
            risk_bce(risk_logits, risk.float())[train_mask], weights[train_mask]
        )
        ranking = _ranking_loss(qos_logits, material, speeds, train_mask)
        loss = classification + risk_classification + 0.25 * ranking
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()


def _validation_loss(
    qos_head: _BinaryHead,
    risk_head: _BinaryHead,
    q_scaled: torch.Tensor,
    r_scaled: torch.Tensor,
    material: torch.Tensor,
    risk: torch.Tensor,
    speeds: torch.Tensor,
    mask: torch.Tensor,
) -> float:
    with torch.no_grad():
        qos_logits = qos_head(q_scaled)
        risk_logits = risk_head(r_scaled)
        return float((
            nn.functional.binary_cross_entropy_with_logits(
                qos_logits[mask], material[mask].float()
            )
            + nn.functional.binary_cross_entropy_with_logits(
                risk_logits[mask], risk[mask].float()
            )
            + 0.25 * _ranking_loss(qos_logits, material, speeds, mask)
        ).item())


def _fit_ranked_heads(
    qos_x: np.ndarray,
    risk_x: np.ndarray,
    material: np.ndarray,
    risk: np.ndarray,
    rows: list[dict[str, Any]],
    fit_mask: np.ndarray,
    *,
    seed: int,
) -> tuple[dict[str, _BinaryHead], dict[str, np.ndarray], dict[str, Any]]:
    """Select epoch on an inner route holdout, then refit all fit routes."""
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    inner_train, inner_validation = _inner_route_masks(rows, fit_mask, seed=seed)
    fit = torch.as_tensor(fit_mask, dtype=torch.bool)
    train = torch.as_tensor(inner_train, dtype=torch.bool)
    validation = torch.as_tensor(inner_validation, dtype=torch.bool)
    qx = torch.as_tensor(qos_x, dtype=torch.float32)
    rx = torch.as_tensor(risk_x, dtype=torch.float32)
    q_mean = qx[fit].mean(0)
    q_std = qx[fit].std(0, unbiased=False).clamp_min(1e-6)
    r_mean = rx[fit].mean(0)
    r_std = rx[fit].std(0, unbiased=False).clamp_min(1e-6)
    q_scaled = (qx - q_mean) / q_std
    r_scaled = (rx - r_mean) / r_std
    material_t = torch.as_tensor(material, dtype=torch.bool)
    risk_t = torch.as_tensor(risk, dtype=torch.bool)
    speeds = torch.as_tensor([int(row["speed_kmh"]) for row in rows])

    def positive_weight(label: np.ndarray, mask: np.ndarray) -> float:
        positives = int(np.sum(label & mask))
        negatives = int(np.sum(~label & mask))
        if positives == 0 or negatives == 0:
            raise ValueError("T4D.1 fit requires both classes.")
        return negatives / positives

    material_weight = positive_weight(material, inner_train)
    risk_weight = positive_weight(risk, inner_train)
    inner_weights = torch.as_tensor(
        _route_weights(rows, inner_train), dtype=torch.float32
    )
    qos_head = _BinaryHead(qos_x.shape[1])
    risk_head = _BinaryHead(risk_x.shape[1])
    learning_rate = 0.01
    optimizer = torch.optim.Adam(
        [*qos_head.parameters(), *risk_head.parameters()],
        lr=learning_rate,
        weight_decay=2e-3,
    )
    qos_bce = nn.BCEWithLogitsLoss(
        reduction="none", pos_weight=torch.tensor(material_weight)
    )
    risk_bce = nn.BCEWithLogitsLoss(
        reduction="none", pos_weight=torch.tensor(risk_weight)
    )
    best_loss = float("inf")
    best_epoch = 1
    stale = 0
    patience = 50
    maximum_epochs = 500
    for epoch in range(maximum_epochs):
        qos_logits = qos_head(q_scaled)
        risk_logits = risk_head(r_scaled)
        loss = (
            _weighted_mean(qos_bce(qos_logits, material_t.float())[train], inner_weights[train])
            + _weighted_mean(risk_bce(risk_logits, risk_t.float())[train], inner_weights[train])
            + 0.25 * _ranking_loss(qos_logits, material_t, speeds, train)
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        validation_loss = _validation_loss(
            qos_head, risk_head, q_scaled, r_scaled,
            material_t, risk_t, speeds, validation,
        )
        if validation_loss < best_loss - 1e-7:
            best_loss = validation_loss
            best_epoch = epoch + 1
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    # Refit from the same deterministic initialization on every fit route.
    torch.manual_seed(seed)
    qos_head = _BinaryHead(qos_x.shape[1])
    risk_head = _BinaryHead(risk_x.shape[1])
    fit_weights = torch.as_tensor(_route_weights(rows, fit_mask), dtype=torch.float32)
    material_fit_weight = positive_weight(material, fit_mask)
    risk_fit_weight = positive_weight(risk, fit_mask)
    _train_epoch_range(
        qos_head, risk_head, q_scaled, r_scaled, material_t, risk_t,
        speeds, fit_weights, fit,
        epochs=best_epoch, learning_rate=learning_rate,
        material_positive_weight=material_fit_weight,
        risk_positive_weight=risk_fit_weight,
    )
    qos_head.eval()
    risk_head.eval()
    with torch.no_grad():
        qos_logits = qos_head(q_scaled).cpu().numpy()
        risk_logits = risk_head(r_scaled).cpu().numpy()
    arrays = {
        "qos_feature_mean": q_mean.numpy(),
        "qos_feature_std": q_std.numpy(),
        "risk_feature_mean": r_mean.numpy(),
        "risk_feature_std": r_std.numpy(),
        "qos_logits": qos_logits,
        "risk_logits": risk_logits,
    }
    config = {
        "architecture": "linear_material_classifier_plus_pairwise_ranking",
        "seed": seed,
        "maximum_epochs": maximum_epochs,
        "selected_epoch_on_inner_route_validation": best_epoch,
        "inner_validation_loss": best_loss,
        "inner_train_rows": int(np.sum(inner_train)),
        "inner_validation_rows": int(np.sum(inner_validation)),
        "learning_rate": learning_rate,
        "weight_decay": 2e-3,
        "ranking_loss_weight": 0.25,
        "route_balanced_loss": True,
        "refit_all_fit_routes": True,
    }
    return {"qos": qos_head, "risk": risk_head}, arrays, config


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -30.0, 30.0)))


def _calibrate_temperature(
    logits: np.ndarray, labels: np.ndarray
) -> tuple[float, dict[str, Any]]:
    """Lock risk temperature on tuning using NLL, with Brier as tie-break."""
    candidates = []
    for temperature in (0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0):
        probability = _sigmoid(logits / temperature)
        nll = float(-np.mean(
            labels * np.log(np.clip(probability, 1e-8, 1.0))
            + (1 - labels) * np.log(np.clip(1 - probability, 1e-8, 1.0))
        ))
        brier = float(np.mean((probability - labels) ** 2))
        candidates.append({"temperature": temperature, "nll": nll, "brier": brier})
    selected = min(candidates, key=lambda row: (row["nll"], row["brier"], row["temperature"]))
    return float(selected["temperature"]), {"selected": selected, "candidates": candidates}


def _lock_selector(
    labels: list[dict[str, Any]],
    qos_score: np.ndarray,
    risk_score: np.ndarray,
    *,
    minimum_wait_actions: int,
) -> tuple[float, float, dict[str, Any]]:
    if minimum_wait_actions <= 0:
        raise ValueError("minimum_wait_actions must be positive.")
    qos_thresholds = sorted(set(
        [0.50, 0.60, 0.70, 0.80, 0.90]
        + [float(value) for value in np.quantile(qos_score, [0.60, 0.70, 0.80, 0.90, 0.95])]
    ))
    risk_thresholds = sorted(set(
        [0.10, 0.20, 0.35, 0.50, 0.70, 0.90]
        + [float(value) for value in np.quantile(risk_score, [0.25, 0.50, 0.75])]
    ))
    candidates = []
    for qos_threshold in qos_thresholds:
        for risk_threshold in risk_thresholds:
            choose = (qos_score >= qos_threshold) & (risk_score <= risk_threshold)
            candidates.append({
                "qos_threshold": float(qos_threshold),
                "risk_threshold": float(risk_threshold),
                **_selection(labels, choose),
            })
    supported = [
        row for row in candidates if row["wait_actions"] >= minimum_wait_actions
    ]
    feasible = [row for row in supported if row["reliability_not_worse"]]
    pool = feasible or supported
    if not pool:
        raise RuntimeError("No T4D.1 selector candidate meets minimum action support.")
    selected = max(pool, key=lambda row: (
        int(row["reliability_not_worse"]),
        row["mean_delta_delivered_mbit"],
        -row["false_wait_actions"],
        -row["wait_actions"],
    ))
    return (
        float(selected["qos_threshold"]),
        float(selected["risk_threshold"]),
        {
            "minimum_wait_actions": minimum_wait_actions,
            "reliability_feasible_candidate_count": len(feasible),
            "selected": selected,
            "candidates": candidates,
        },
    )


def _group_classification(
    rows: list[dict[str, Any]],
    labels: np.ndarray,
    scores: np.ndarray,
    mask: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    report = {"overall": _classification_report(labels, scores, mask, threshold)}
    for field in ("speed_kmh", "source"):
        for value in sorted({row[field] for row, keep in zip(rows, mask) if keep}):
            group = mask & np.asarray([row[field] == value for row in rows])
            report[f"{field}:{value}"] = _classification_report(
                labels, scores, group, threshold
            )
    return report


def run_probe(
    *,
    root: Path,
    t4c_dir: Path,
    t4d_report_path: Path | None,
    output_dir: Path,
    model_seed: int,
    minimum_gain_mbit: float,
    minimum_tuning_wait_actions: int,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite T4D.1 output: {output_dir}")
    if minimum_gain_mbit <= 0:
        raise ValueError("minimum_gain_mbit must be positive.")
    source_report = _read_json(t4c_dir / "t4c_report.json")
    if (
        source_report.get("status") != "COMPLETE"
        or source_report.get("validation_or_test_rows_read") != 0
        or source_report.get("feature_contract", {}).get("schema") != SOURCE_SCHEMA
    ):
        raise ValueError("T4D.1 requires the complete TRAIN-only T4C artifact.")
    rows = _read_jsonl(t4c_dir / "samples.jsonl")
    if len(rows) != int(source_report["sample_counts"]["total"]):
        raise ValueError("T4C sample count does not match its report.")
    full = np.asarray([row["feature"] for row in rows], dtype=np.float32)
    qos_x = full[:, _feature_indices(QOS_FEATURE_NAMES)]
    risk_x = full[:, _feature_indices(RISK_FEATURE_NAMES)]
    gain = np.asarray([
        float(row["label"]["delta_delivered_mbit"]) for row in rows
    ])
    material = gain >= minimum_gain_mbit
    risk = np.asarray([_risk_worse(row) for row in rows], dtype=bool)
    meaningful = material & ~risk
    masks = {
        fold: np.asarray([row["fold"] == fold for row in rows], dtype=bool)
        for fold in ("fit", "tuning", "confirmation")
    }

    models, arrays, fit_config = _fit_ranked_heads(
        qos_x, risk_x, material, risk, rows, masks["fit"], seed=model_seed
    )
    qos_score = _sigmoid(arrays["qos_logits"])
    tuning_mask = masks["tuning"]
    temperature, calibration = _calibrate_temperature(
        arrays["risk_logits"][tuning_mask], risk[tuning_mask].astype(float)
    )
    risk_score = _sigmoid(arrays["risk_logits"] / temperature)
    tuning_rows = [row["label"] for row, keep in zip(rows, tuning_mask) if keep]
    qos_threshold, risk_threshold, threshold_report = _lock_selector(
        tuning_rows,
        qos_score[tuning_mask],
        risk_score[tuning_mask],
        minimum_wait_actions=minimum_tuning_wait_actions,
    )
    confirmation_mask = masks["confirmation"]
    confirmation_rows = [row for row, keep in zip(rows, confirmation_mask) if keep]
    confirmation_labels = [row["label"] for row in confirmation_rows]
    choose = (
        qos_score[confirmation_mask] >= qos_threshold
    ) & (risk_score[confirmation_mask] <= risk_threshold)
    selection = _selection(confirmation_labels, choose)
    meaningful_chosen = meaningful[confirmation_mask] & choose
    selection["meaningful_wait_actions"] = int(np.sum(meaningful_chosen))
    selection["nonmeaningful_wait_actions"] = int(np.sum(choose & ~meaningful[confirmation_mask]))
    selection_vs_now = _paired_selection_bootstrap(
        confirmation_rows,
        np.zeros(len(confirmation_rows), dtype=bool),
        choose,
        seed=model_seed + 100,
    )

    names = feature_names(SOURCE_SCHEMA)
    rule_choose = np.asarray([
        _rule_wait(feature, "target_sinr_worse", names=names)
        for feature in full[confirmation_mask]
    ], dtype=bool)
    rule_selection = _selection(confirmation_labels, rule_choose)
    selected_vs_rule = _paired_selection_bootstrap(
        confirmation_rows, rule_choose, choose, seed=model_seed + 200
    )
    material_report = _group_classification(
        rows, material, qos_score, confirmation_mask, qos_threshold
    )
    risk_uncalibrated = _classification_report(
        risk, _sigmoid(arrays["risk_logits"]), confirmation_mask, risk_threshold
    )
    risk_calibrated = _group_classification(
        rows, risk, risk_score, confirmation_mask, risk_threshold
    )
    primary_material = material_report["overall"]
    primary_risk = risk_calibrated["overall"]
    g_material_ranking = bool(
        primary_material["average_precision"]
        >= primary_material["prevalence"] + 0.05
    )
    g_risk_signal = bool(
        primary_risk["average_precision"] >= primary_risk["prevalence"] + 0.03
        and primary_risk["brier_score"] <= risk_uncalibrated["brier_score"] + EPS
    )
    g_local_utility = bool(
        selection["wait_actions"] > 0
        and selection["reliability_not_worse"]
        and selection_vs_now["ci95"]["delta_delivered_mbit"][0] > 0
    )
    g_beats_rule = bool(
        selected_vs_rule["ci95"]["delta_delivered_mbit"][0] > 0
        and all(selected_vs_rule["ci95"][field][1] <= 0 for field in RISK_FIELDS)
    )
    local_pass = bool(g_material_ranking and g_risk_signal and g_local_utility)

    output_dir.mkdir(parents=True)
    frozen = {
        **arrays,
        "risk_temperature": np.asarray([temperature], dtype=np.float32),
        "qos_threshold": np.asarray([qos_threshold], dtype=np.float32),
        "risk_threshold": np.asarray([risk_threshold], dtype=np.float32),
    }
    for head_name, model in models.items():
        for name, tensor in model.state_dict().items():
            frozen[f"{head_name}__{name}"] = tensor.detach().cpu().numpy()
    np.savez_compressed(output_dir / "frozen_model.npz", **frozen)
    report = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "status": "COMPLETE",
        "scope": "train_fit_inner_validation_tuning_confirmation_ranked_local_probe",
        "validation_or_test_rows_read": 0,
        "source": {
            "t4c_report": str((t4c_dir / "t4c_report.json").resolve()),
            "t4c_report_sha256": _sha256(t4c_dir / "t4c_report.json"),
            "samples": str((t4c_dir / "samples.jsonl").resolve()),
            "samples_sha256": _sha256(t4c_dir / "samples.jsonl"),
            "t4d_context": None if t4d_report_path is None else {
                "path": str(t4d_report_path.resolve()),
                "sha256": _sha256(t4d_report_path),
                "warning": "Context/provenance only; T4D confirmation was not used for T4D.1 selection.",
            },
        },
        "feature_contract": {
            "qos_features": list(QOS_FEATURE_NAMES),
            "risk_features": list(RISK_FEATURE_NAMES),
            "causal": True,
            "future_features": [],
        },
        "label_contract": {
            "material_gain": f"delta_delivered_mbit >= {minimum_gain_mbit}",
            "risk": "any WAIT_20-minus-HO_NOW risk delta > 0",
            "meaningful_wait": "material_gain and not risk",
            "horizon_seconds": 1.0,
        },
        "split_contract": {
            "fit_only_inner_route_validation": True,
            "fit_refit_after_epoch_lock": True,
            "tuning_used_for_temperature_and_thresholds": True,
            "confirmation_used_for_fit": False,
            "confirmation_used_for_epoch_selection": False,
            "confirmation_used_for_calibration": False,
            "confirmation_used_for_threshold": False,
            "counts": {fold: int(np.sum(mask)) for fold, mask in masks.items()},
            "material_positive_counts": {
                fold: int(np.sum(material & mask)) for fold, mask in masks.items()
            },
            "risk_positive_counts": {
                fold: int(np.sum(risk & mask)) for fold, mask in masks.items()
            },
        },
        "fit_config": fit_config,
        "risk_temperature_on_tuning": calibration,
        "selector_thresholds_on_tuning": threshold_report,
        "confirmation": {
            "material_gain_classification": material_report,
            "risk_uncalibrated": risk_uncalibrated,
            "risk_calibrated": risk_calibrated,
            "selection": selection,
            "selection_vs_ho_now": selection_vs_now,
            "simple_rule": {
                "name": "target_sinr_worse",
                "selection": rule_selection,
            },
            "selected_minus_simple_rule": selected_vs_rule,
        },
        "gates": {
            "g_material_ranking": g_material_ranking,
            "g_risk_signal_and_calibration": g_risk_signal,
            "g_local_utility": g_local_utility,
            "g_beats_simple_rule": g_beats_rule,
            "overall_local_probe": "PASSED" if local_pass else "NOT_PASSED",
            "ppo_status": "CLOSED",
            "next_gate": (
                "full-route TRAIN audit against SINR rule" if local_pass
                else "stop model expansion; audit label horizon/reference policy"
            ),
        },
    }
    _write_json(output_dir / "t4d1_report.json", report)
    manifest = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "validation_or_test_rows_read": 0,
        "model_seed": model_seed,
        "minimum_gain_mbit": minimum_gain_mbit,
        "minimum_tuning_wait_actions": minimum_tuning_wait_actions,
        "inputs": report["source"],
        "outputs": {
            name: {"path": str((output_dir / name).resolve()), "sha256": _sha256(output_dir / name)}
            for name in ("t4d1_report.json", "frozen_model.npz")
        },
        "source_files": [
            {"path": path, "sha256": _sha256(root / path)}
            for path in (
                "src/ho_optim_drl/commands/audit_t4d1_ranked_probe.py",
                "src/ho_optim_drl/commands/audit_t4d_compact_causal_probe.py",
            )
        ],
        "git": _git_state(root),
    }
    _write_json(output_dir / "run_manifest.json", manifest)
    lines = [
        "# T4D.1 ranked material-gain probe",
        "",
        f"- Epoch khóa bằng inner route validation: **{fit_config['selected_epoch_on_inner_route_validation']}**.",
        f"- Risk temperature khóa trên tuning: **{temperature:.3f}**.",
        f"- Threshold QoS/risk: **{qos_threshold:.4f}/{risk_threshold:.4f}**.",
        f"- Material AP/prevalence confirmation: **{primary_material['average_precision']:.4f}/{primary_material['prevalence']:.4f}**.",
        f"- Risk AP/prevalence confirmation: **{primary_risk['average_precision']:.4f}/{primary_risk['prevalence']:.4f}**.",
        f"- WAIT confirmation: **{selection['wait_actions']}**; mean gain **{selection['mean_delta_delivered_mbit']:.6f} Mbit/state**.",
        f"- Gate ranking/risk/utility/rule: **{g_material_ranking}/{g_risk_signal}/{g_local_utility}/{g_beats_rule}**.",
        f"- Kết luận: **{report['gates']['overall_local_probe']}**; PPO **CLOSED**.",
        "",
        "Confirmation không tham gia fit, epoch selection, calibration hoặc threshold selection.",
    ]
    (output_dir / "report_vi.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main(root_path: str, args: Namespace) -> int:
    root = Path(root_path).resolve()
    resolve = lambda value: (
        Path(value) if Path(value).is_absolute() else root / value
    ).resolve()
    context = getattr(args, "t4d_report", None)
    report = run_probe(
        root=root,
        t4c_dir=resolve(args.t4c_dir),
        t4d_report_path=None if context is None else resolve(context),
        output_dir=resolve(args.output_dir),
        model_seed=int(args.model_seed),
        minimum_gain_mbit=float(args.minimum_gain_mbit),
        minimum_tuning_wait_actions=int(args.minimum_tuning_wait_actions),
    )
    print(json.dumps({
        "status": report["status"],
        "selection": report["confirmation"]["selection"],
        "gates": report["gates"],
    }, indent=2), flush=True)
    return 0
