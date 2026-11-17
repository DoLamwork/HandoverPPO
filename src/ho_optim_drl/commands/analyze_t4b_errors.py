"""Post-hoc false-positive/false-negative audit for the frozen T4B MLP."""

from __future__ import annotations

from argparse import Namespace
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ho_optim_drl.commands.audit_baseline_parity import _git_state, _sha256
from ho_optim_drl.commands.audit_early_timing_learnability import (
    RISK_FIELDS,
    _Probe,
)
from ho_optim_drl.commands.audit_policy_learning import _average_precision


SCHEMA_NAME = "early_ho_t4b_error_audit"
SCHEMA_VERSION = 1


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def confusion_labels(labels: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    result = np.full(len(labels), "tn", dtype="<U2")
    result[labels & predicted] = "tp"
    result[~labels & predicted] = "fp"
    result[labels & ~predicted] = "fn"
    return result


def risk_reasons(label: dict[str, Any]) -> list[str]:
    reasons = []
    if float(label["delta_delivered_mbit"]) <= 0:
        reasons.append("no_positive_data_gain")
    mapping = {
        "handover_failures": "increased_hof",
        "ping_pong_count": "increased_ping_pong",
        "rlf_starts": "increased_rlf",
        "recovery_seconds": "increased_recovery",
        "interruption_seconds": "increased_interruption",
    }
    for field, reason in mapping.items():
        if float(label["risk_deltas"][field]) > 1e-12:
            reasons.append(reason)
    return reasons or ["strict_gain_tolerance_or_other"]


def benefit_reasons(label: dict[str, Any]) -> list[str]:
    """Describe which strict-safe benefits a missed WAIT decision contained."""
    reasons = []
    if float(label["delta_delivered_mbit"]) > 0:
        reasons.append("positive_data_gain")
    mapping = {
        "handover_failures": "avoided_hof",
        "ping_pong_count": "avoided_ping_pong",
        "rlf_starts": "avoided_rlf",
        "recovery_seconds": "reduced_recovery",
        "interruption_seconds": "reduced_interruption",
    }
    for field, reason in mapping.items():
        if float(label["risk_deltas"][field]) < -1e-12:
            reasons.append(reason)
    return reasons or ["strict_safe_tolerance_or_other"]


def _load_model(
    path: Path, *, input_size: int,
) -> tuple[_Probe, np.ndarray, np.ndarray, float]:
    with np.load(path, allow_pickle=False) as arrays:
        model = _Probe(input_size, "mlp_small")
        state = {
            name.removeprefix("mlp_small__"): torch.as_tensor(arrays[name].copy())
            for name in arrays.files if name.startswith("mlp_small__network.")
        }
        model.load_state_dict(state); model.eval()
        mean = arrays["mlp_small__feature_mean"].astype(np.float32)
        std = arrays["mlp_small__feature_std"].astype(np.float32)
        threshold = float(arrays["mlp_small__threshold"][0])
    return model, mean, std, threshold


def _scores_and_contributions(
    model: _Probe, features: np.ndarray, mean: np.ndarray, std: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    scaled = torch.as_tensor((features - mean) / std, dtype=torch.float32)
    scaled.requires_grad_(True)
    logits = model(scaled)[:, 0]
    logits.sum().backward()
    scores = torch.sigmoid(logits).detach().cpu().numpy()
    contributions = (scaled.grad * scaled).detach().cpu().numpy()
    return scores, contributions


def _confusion_summary(
    samples: list[dict[str, Any]], scores: np.ndarray, threshold: float,
) -> dict[str, Any]:
    labels = np.asarray([row["label"]["wait_useful"] for row in samples], dtype=bool)
    predicted = scores >= threshold
    groups = confusion_labels(labels, predicted)
    counts = Counter(groups.tolist())
    tp, fp, fn, tn = (counts[name] for name in ("tp", "fp", "fn", "tn"))
    return {
        "rows": len(samples), "threshold": threshold,
        "counts": {name: counts[name] for name in ("tp", "fp", "fn", "tn")},
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "specificity": tn / (tn + fp) if tn + fp else None,
        "average_precision": _average_precision(labels.astype(np.int8), scores),
        "prevalence": float(np.mean(labels)),
        "score_summary": {
            name: ({"mean": float(np.mean(scores[groups == name])), "min": float(np.min(scores[groups == name])), "max": float(np.max(scores[groups == name]))} if np.any(groups == name) else None)
            for name in ("tp", "fp", "fn", "tn")
        },
    }


def _breakdowns(
    samples: list[dict[str, Any]], scores: np.ndarray, threshold: float,
) -> dict[str, Any]:
    labels = np.asarray([row["label"]["wait_useful"] for row in samples], dtype=bool)
    predicted = scores >= threshold
    output: dict[str, Any] = {}
    for field in ("speed_kmh", "source"):
        for value in sorted({row[field] for row in samples}):
            mask = np.asarray([row[field] == value for row in samples])
            output[f"{field}:{value}"] = _confusion_summary(
                [row for row, keep in zip(samples, mask) if keep], scores[mask], threshold
            )
    return output


def _physical_error_costs(
    samples: list[dict[str, Any]], scores: np.ndarray, threshold: float,
) -> dict[str, Any]:
    labels = np.asarray([row["label"]["wait_useful"] for row in samples], dtype=bool)
    predicted = scores >= threshold
    groups = confusion_labels(labels, predicted)
    fp_indices = np.flatnonzero(groups == "fp")
    fn_indices = np.flatnonzero(groups == "fn")
    reason_counts: Counter[str] = Counter()
    for index in fp_indices:
        reason_counts.update(risk_reasons(samples[int(index)]["label"]))
    benefit_counts: Counter[str] = Counter()
    for index in fn_indices:
        benefit_counts.update(benefit_reasons(samples[int(index)]["label"]))
    def sums(indices: np.ndarray) -> dict[str, Any]:
        return {
            "rows": int(len(indices)),
            "delta_delivered_mbit_sum": float(sum(samples[int(i)]["label"]["delta_delivered_mbit"] for i in indices)),
            "delta_relative_rate_percent_sum": float(sum(samples[int(i)]["label"]["delta_relative_rate_percent"] for i in indices)),
            "risk_delta_sums": {field: float(sum(samples[int(i)]["label"]["risk_deltas"][field] for i in indices)) for field in RISK_FIELDS},
        }
    missed = sums(fn_indices)
    missed["foregone_positive_gain_mbit"] = float(sum(max(0.0, samples[int(i)]["label"]["delta_delivered_mbit"]) for i in fn_indices))
    missed["benefit_counts"] = dict(benefit_counts)
    return {"false_positive": {**sums(fp_indices), "reason_counts": dict(reason_counts)}, "false_negative": missed}


def _feature_diagnostics(
    samples: list[dict[str, Any]], features: np.ndarray, scores: np.ndarray,
    contributions: np.ndarray, threshold: float, names: list[str],
) -> dict[str, Any]:
    labels = np.asarray([row["label"]["wait_useful"] for row in samples], dtype=bool)
    groups = confusion_labels(labels, scores >= threshold)
    useful = features[labels]; not_useful = features[~labels]
    pooled = np.sqrt((np.var(useful, axis=0) + np.var(not_useful, axis=0)) / 2.0)
    effect = np.divide(np.mean(useful, axis=0) - np.mean(not_useful, axis=0), pooled, out=np.zeros_like(pooled), where=pooled > 1e-9)
    base_ap = _average_precision(labels.astype(np.int8), scores)
    separation = []
    for index, name in enumerate(names):
        separation.append({"feature": name, "absolute_label_effect_size": float(abs(effect[index])), "signed_label_effect_size": float(effect[index])})
    saliency = []
    for index, name in enumerate(names):
        saliency.append({
            "feature": name,
            "fp_mean_abs_gradient_x_input": float(np.mean(np.abs(contributions[groups == "fp", index]))) if np.any(groups == "fp") else None,
            "fn_mean_abs_gradient_x_input": float(np.mean(np.abs(contributions[groups == "fn", index]))) if np.any(groups == "fn") else None,
            "tp_mean_abs_gradient_x_input": float(np.mean(np.abs(contributions[groups == "tp", index]))) if np.any(groups == "tp") else None,
            "signed_label_effect_size": float(effect[index]),
        })
    return {
        "base_average_precision": base_ap,
        "label_separation_ranked": sorted(separation, key=lambda row: row["absolute_label_effect_size"], reverse=True),
        "error_saliency_ranked": sorted(saliency, key=lambda row: max(row["fp_mean_abs_gradient_x_input"] or 0.0, row["fn_mean_abs_gradient_x_input"] or 0.0), reverse=True),
        "interpretation": "Gradient×input describes frozen-model reliance; label effect size describes observed marginal separation. Neither is causal feature importance.",
    }


def _rule_audit(samples: list[dict[str, Any]]) -> dict[str, Any]:
    target_sinr_index = 19
    scores = np.asarray([-float(row["feature"][target_sinr_index]) for row in samples])
    # target_sinr_worse is exactly score > 0.  A tiny strict threshold preserves <0.
    return _confusion_summary(samples, scores, np.nextafter(0.0, 1.0))


def _case_records(
    samples: list[dict[str, Any]], scores: np.ndarray, threshold: float,
    contributions: np.ndarray, names: list[str], *, limit: int,
) -> list[dict[str, Any]]:
    labels = np.asarray([row["label"]["wait_useful"] for row in samples], dtype=bool)
    groups = confusion_labels(labels, scores >= threshold)
    records = []
    for index, (row, group) in enumerate(zip(samples, groups)):
        if group not in ("fp", "fn"):
            continue
        top = np.argsort(np.abs(contributions[index]))[::-1][:5]
        records.append({
            "error_type": str(group), "state_id": row["state_id"],
            "speed_kmh": row["speed_kmh"], "trajectory_id": row["trajectory_id"],
            "source": row["source"], "score": float(scores[index]),
            "threshold": threshold, "label": row["label"],
            "top_model_contributions": [
                {"feature": names[int(i)], "value": float(row["feature"][int(i)]), "gradient_x_input": float(contributions[index, int(i)])}
                for i in top
            ],
        })
    fp = sorted((row for row in records if row["error_type"] == "fp"), key=lambda row: (sum(max(0.0, float(v)) for v in row["label"]["risk_deltas"].values()), -float(row["label"]["delta_delivered_mbit"]), row["score"]), reverse=True)[:limit]
    fn = sorted((row for row in records if row["error_type"] == "fn"), key=lambda row: float(row["label"]["delta_delivered_mbit"]), reverse=True)[:limit]
    return [*fp, *fn]


def run_audit(
    *, root: Path, t4b_dir: Path, output_dir: Path, seed: int,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite T4B error audit: {output_dir}")
    output_dir.mkdir(parents=True)
    report_path = t4b_dir / "t4_report.json"
    samples_path = t4b_dir / "post_ttt_samples.jsonl"
    model_path = t4b_dir / "frozen_models.npz"
    source_report = _read_json(report_path)
    if source_report.get("validation_or_test_rows_read") != 0 or source_report["feature_contract"]["schema"] != "t4b_protocol_history_v1":
        raise ValueError("Error audit requires a TRAIN-only T4B artifact.")
    names = list(source_report["feature_contract"]["names"])
    samples = [row for row in _read_jsonl(samples_path) if row["fold"] == "holdout"]
    features = np.asarray([row["feature"] for row in samples], dtype=np.float32)
    model, mean, std, threshold = _load_model(model_path, input_size=len(names))
    scores, contributions = _scores_and_contributions(model, features, mean, std)
    registered = source_report["models"]["mlp_small"]["holdout_classification"]["overall"]
    summary = _confusion_summary(samples, scores, threshold)
    if summary["counts"]["tp"] + summary["counts"]["fp"] != registered["predicted_wait_rows"] or abs(summary["average_precision"] - registered["average_precision"]) > 1e-7:
        raise RuntimeError("Reloaded frozen model does not reproduce the registered T4B report.")
    cases = _case_records(samples, scores, threshold, contributions, names, limit=10)
    result = {
        "schema_name": SCHEMA_NAME, "schema_version": SCHEMA_VERSION,
        "status": "COMPLETE", "scope": "train_holdout_posthoc_no_refit",
        "validation_or_test_rows_read": 0, "model": "mlp_small",
        "threshold_source": "frozen T4B tuning selection", "threshold": threshold,
        "confusion": summary, "breakdown": _breakdowns(samples, scores, threshold),
        "physical_error_costs": _physical_error_costs(samples, scores, threshold),
        "simple_rule_confusion": _rule_audit(samples),
        "feature_diagnostics": _feature_diagnostics(samples, features, scores, contributions, threshold, names),
        "priority_case_count": len(cases),
        "conclusion_contract": "diagnostic only; does not approve a new feature, threshold, policy, or PPO run",
    }
    _write_json(output_dir / "audit_report.json", result)
    with (output_dir / "error_cases.jsonl").open("w", encoding="utf-8") as stream:
        for row in cases:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    counts = summary["counts"]; fp_cost = result["physical_error_costs"]["false_positive"]; fn_cost = result["physical_error_costs"]["false_negative"]
    fp_reasons = ", ".join(
        f"{name}={count}" for name, count in sorted(
            fp_cost["reason_counts"].items(), key=lambda item: (-item[1], item[0])
        )
    ) or "không có"
    fn_benefits = ", ".join(
        f"{name}={count}" for name, count in sorted(
            fn_cost["benefit_counts"].items(), key=lambda item: (-item[1], item[0])
        )
    ) or "không có"
    separation = result["feature_diagnostics"]["label_separation_ranked"][:5]
    saliency = result["feature_diagnostics"]["error_saliency_ranked"][:5]
    breakdown_lines = []
    for key, value in result["breakdown"].items():
        item_counts = value["counts"]
        breakdown_lines.append(
            f"- `{key}`: TP/FP/FN/TN = "
            f"{item_counts['tp']}/{item_counts['fp']}/{item_counts['fn']}/{item_counts['tn']}; "
            f"precision={value['precision'] if value['precision'] is not None else 'NA'}, "
            f"recall={value['recall'] if value['recall'] is not None else 'NA'}."
        )
    lines = [
        "# Audit false-positive/false-negative T4B",
        "",
        f"- TP/FP/FN/TN: **{counts['tp']}/{counts['fp']}/{counts['fn']}/{counts['tn']}**.",
        f"- Precision: {summary['precision']:.4f}; recall: {summary['recall']:.4f}; AP: {summary['average_precision']:.4f}.",
        f"- FP delta data tổng: {fp_cost['delta_delivered_mbit_sum']:+.6f} Mbit.",
        f"- Nguyên nhân FP (một case có thể có nhiều nguyên nhân): {fp_reasons}.",
        f"- FN bỏ lỡ tổng: {fn_cost['foregone_positive_gain_mbit']:.6f} Mbit.",
        f"- Lợi ích bị bỏ lỡ ở FN (một case có thể có nhiều lợi ích): {fn_benefits}.",
        f"- Simple rule precision/recall: {result['simple_rule_confusion']['precision']:.4f}/{result['simple_rule_confusion']['recall']:.4f}.",
        "",
        "## Theo speed và nguồn state",
        "",
        *breakdown_lines,
        "",
        "## Feature diagnostic",
        "",
        "Feature phân tách nhãn mạnh nhất: " + ", ".join(
            f"`{row['feature']}` (|d|={row['absolute_label_effect_size']:.3f})" for row in separation
        ) + ".",
        "Feature model dựa nhiều nhất ở các lỗi: " + ", ".join(
            f"`{row['feature']}`" for row in saliency
        ) + ".",
        "",
        "## Diễn giải",
        "",
        "FP cho biết model WAIT ở state không đem lại strict safe gain; FN cho biết state WAIT hữu ích nhưng model bỏ qua. Danh sách case ưu tiên và feature contribution nằm trong error_cases.jsonl.",
        "Kết quả chỉ dùng để chọn giả thuyết observation tiếp theo; không được dùng để đổi threshold trên holdout.",
    ]
    (output_dir / "report_vi.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = {
        "schema_name": SCHEMA_NAME, "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "validation_or_test_rows_read": 0, "seed": seed,
        "inputs": {
            "t4b_report": {"path": str(report_path.resolve()), "sha256": _sha256(report_path)},
            "samples": {"path": str(samples_path.resolve()), "sha256": _sha256(samples_path)},
            "frozen_models": {"path": str(model_path.resolve()), "sha256": _sha256(model_path)},
        },
        "source": {"path": str(Path(__file__).resolve().relative_to(root)), "sha256": _sha256(Path(__file__).resolve())},
        "git": _git_state(root),
    }
    _write_json(output_dir / "run_manifest.json", manifest)
    return result


def main(root_path: str, args: Namespace) -> int:
    root = Path(root_path).resolve()
    def resolve(value: str) -> Path:
        path = Path(value); return (path if path.is_absolute() else root / path).resolve()
    report = run_audit(root=root, t4b_dir=resolve(args.t4b_dir), output_dir=resolve(args.output_dir), seed=int(args.seed))
    print(json.dumps({"status": report["status"], "confusion": report["confusion"], "physical_error_costs": report["physical_error_costs"]}, indent=2), flush=True)
    return 0
