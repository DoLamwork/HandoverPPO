"""T4C frozen probe on the expanded post-TTT-only TRAIN bank."""

from __future__ import annotations

from argparse import Namespace
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np

from ho_optim_drl.commands.audit_baseline_parity import _git_state, _sha256
from ho_optim_drl.commands.audit_early_timing_learnability import (
    MODEL_NAMES,
    RISK_FIELDS,
    _classification_groups,
    _fit_probe,
    _lock_threshold,
    _predict,
    _route_bootstrap_gain,
    _rule_wait,
    _selection,
    causal_features,
    feature_names,
    timing_label,
)
from ho_optim_drl.commands.build_post_ttt_train_bank import (
    SOURCES,
    collect_post_ttt_source_anchors,
)
from ho_optim_drl.data import TracePair, load_radio_trace
from ho_optim_drl.data.project import load_protocol_config


SCHEMA_NAME = "early_ho_t4c_post_ttt_probe"
SCHEMA_VERSION = 1
FEATURE_SCHEMA = "t4b_protocol_history_v1"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _folds_from_split(split: dict[str, Any]) -> dict[tuple[int, int], str]:
    folds: dict[tuple[int, int], str] = {}
    for row in split["selected_routes"]:
        route = (int(row["speed_kmh"]), int(row["trajectory_id"]))
        fold = str(row["fold"])
        if fold not in ("fit", "tuning", "confirmation"):
            raise ValueError(f"Unexpected T4C fold: {fold}")
        if route in folds:
            raise ValueError(f"Duplicate route in T4C split: {route}")
        folds[route] = fold
    return folds


def _paired_selection_bootstrap(
    samples: list[dict[str, Any]], left: np.ndarray, right: np.ndarray,
    *, seed: int, resamples: int = 2000,
) -> dict[str, Any]:
    fields = ("delta_delivered_mbit", *RISK_FIELDS)
    by_speed: dict[int, dict[tuple[int, int], list[int]]] = defaultdict(lambda: defaultdict(list))
    for index, row in enumerate(samples):
        by_speed[int(row["speed_kmh"])][
            (int(row["speed_kmh"]), int(row["trajectory_id"]))
        ].append(index)
    route_deltas: dict[int, list[list[float]]] = defaultdict(list)
    for speed, routes in by_speed.items():
        for indices in routes.values():
            values = []
            for field in fields:
                if field == "delta_delivered_mbit":
                    base = np.asarray([
                        samples[index]["label"][field] for index in indices
                    ], dtype=float)
                else:
                    base = np.asarray([
                        samples[index]["label"]["risk_deltas"][field]
                        for index in indices
                    ], dtype=float)
                values.append(float(np.mean(
                    np.where(right[indices], base, 0.0)
                    - np.where(left[indices], base, 0.0)
                )))
            route_deltas[speed].append(values)
    observed = np.asarray([
        values for speed in sorted(route_deltas) for values in route_deltas[speed]
    ], dtype=float)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(resamples):
        sampled = []
        for speed in sorted(route_deltas):
            values = np.asarray(route_deltas[speed], dtype=float)
            sampled.append(values[rng.integers(0, len(values), len(values))])
        draws.append(np.concatenate(sampled, axis=0).mean(axis=0))
    boot = np.asarray(draws)
    return {
        "definition": "right-minus-left action-conditioned local outcome",
        "routes": int(len(observed)), "stratified_by_speed": True,
        "resamples": resamples, "seed": seed,
        "mean": {field: float(observed[:, index].mean()) for index, field in enumerate(fields)},
        "ci95": {
            field: [float(np.percentile(boot[:, index], 2.5)), float(np.percentile(boot[:, index], 97.5))]
            for index, field in enumerate(fields)
        },
    }


def _rule_classification(
    samples: list[dict[str, Any]], labels: np.ndarray, features: np.ndarray,
    mask: np.ndarray, *, rule: str, names: tuple[str, ...],
) -> dict[str, Any]:
    # A deterministic rule is represented as a binary score for confusion/AP.
    scores = np.asarray([
        float(_rule_wait(feature, rule, names=names)) for feature in features
    ])
    return _classification_groups(samples, labels, scores, mask, 0.5)


def run_probe(
    *, root: Path, data_dir: Path, bank_dir: Path, output_dir: Path,
    model_seed: int, t4b_report_path: Path | None,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite T4C output: {output_dir}")
    split = _read_json(bank_dir / "route_split.json")
    bank_report = _read_json(bank_dir / "bank_report.json")
    if (
        split.get("schema_name") != "early_ho_post_ttt_train_route_split"
        or split.get("split") != "train"
        or split.get("validation_or_test_rows_read") != 0
        or bank_report.get("status") != "COMPLETE"
        or bank_report.get("validation_or_test_rows_read") != 0
    ):
        raise ValueError("T4C requires a complete post-TTT-only TRAIN bank.")
    folds = _folds_from_split(split)
    route_records = {
        (int(row["speed_kmh"]), int(row["trajectory_id"])): row
        for row in split["selected_routes"]
    }
    raw_anchors = _read_jsonl(bank_dir / "anchors.jsonl")
    if any(row.get("category") != "post_ttt" for row in raw_anchors):
        raise ValueError("T4C bank contains a non-post-TTT anchor.")
    anchors_by_fingerprint: dict[str, dict[str, Any]] = {}
    for row in sorted(raw_anchors, key=lambda item: (
        str(item["physical_state_fingerprint"]), str(item["state_id"])
    )):
        anchors_by_fingerprint.setdefault(str(row["physical_state_fingerprint"]), row)
    anchors = list(anchors_by_fingerprint.values())
    outcomes = _read_jsonl(bank_dir / "local_outcomes.jsonl")
    outcome_map = {
        (
            row["state_id"], float(row["horizon_seconds"]),
            row["branch"], row["requested_delay_ms"],
        ): row["physical"]
        for row in outcomes
    }
    config = load_protocol_config(str(data_dir))
    max_anchors = int(bank_report["selection"]["max_anchors_per_source_route"])
    separation_ticks = int(bank_report["selection"]["min_anchor_separation_ms"]) // int(config.delta_t_ms)
    names = feature_names(FEATURE_SCHEMA)
    trace_cache: dict[tuple[int, int], Any] = {}
    snapshot_by_fingerprint: dict[str, Any] = {}
    for route_index, route in enumerate(sorted(route_records), 1):
        record = route_records[route]
        pair = TracePair(
            route[0], route[1], str(record["rsrp_file"]), str(record["sinr_file"])
        )
        trace = load_radio_trace(
            config, str(data_dir), pair.rsrp_file, pair.sinr_file,
            speed_kmh=route[0],
        )
        trace_cache[route] = trace
        for source in SOURCES:
            replayed, _ = collect_post_ttt_source_anchors(
                config, trace, pair, source=source,
                sample_seed=int(split["sample_seed"]),
                max_anchors=max_anchors,
                min_separation_ticks=separation_ticks,
            )
            for item in replayed:
                fingerprint = str(item["record"]["physical_state_fingerprint"])
                snapshot_by_fingerprint.setdefault(fingerprint, item["engine"].snapshot)
        if route_index % 10 == 0:
            print(f"[T4C] replayed {route_index}/{len(route_records)} routes", flush=True)
    missing = sorted(set(anchors_by_fingerprint) - set(snapshot_by_fingerprint))
    if missing:
        raise RuntimeError(f"T4C replay missed {len(missing)} frozen physical states.")
    samples = []
    for anchor in anchors:
        route = (int(anchor["speed_kmh"]), int(anchor["trajectory_id"]))
        trace = trace_cache[route]
        now = outcome_map[(anchor["state_id"], 1.0, "timing_schedule", 0)]
        wait = outcome_map[(anchor["state_id"], 1.0, "timing_schedule", 20)]
        fingerprint = str(anchor["physical_state_fingerprint"])
        samples.append({
            "state_id": anchor["state_id"], "source": anchor["source"],
            "speed_kmh": route[0], "trajectory_id": route[1],
            "fold": folds[route],
            "feature": causal_features(
                trace, tic=int(anchor["tic"]),
                serving=int(anchor["serving_cell"]),
                target=int(anchor["target_cell"]),
                a3_start_tic=int(anchor["decision"]["a3_start_tic"]),
                speed_kmh=route[0], q_out_db=config.q_out_db,
                dt_ms=config.delta_t_ms,
                snapshot=snapshot_by_fingerprint[fingerprint],
                schema=FEATURE_SCHEMA,
            ),
            "label": timing_label(now, wait),
        })
    features = np.stack([row["feature"] for row in samples])
    labels = np.asarray([row["label"]["wait_useful"] for row in samples], dtype=bool)
    continuous = np.asarray([
        [
            row["label"]["delta_delivered_mbit"],
            *(row["label"]["risk_deltas"][name] for name in RISK_FIELDS),
        ]
        for row in samples
    ], dtype=np.float32)
    masks = {
        fold: np.asarray([row["fold"] == fold for row in samples])
        for fold in ("fit", "tuning", "confirmation")
    }
    if any(not np.any(mask) for mask in masks.values()):
        raise ValueError("T4C requires non-empty fit, tuning, and confirmation folds.")

    model_reports: dict[str, Any] = {}
    frozen: dict[str, Any] = {}
    arrays: dict[str, np.ndarray] = {}
    for index, architecture in enumerate(MODEL_NAMES):
        model, scores, mean, std, fit_config = _fit_probe(
            features, labels, continuous, masks["fit"],
            architecture=architecture, seed=model_seed + index,
        )
        tuning_rows = [row["label"] for row, keep in zip(samples, masks["tuning"]) if keep]
        threshold, threshold_report = _lock_threshold(
            tuning_rows, scores[masks["tuning"]]
        )
        confirmation_rows = [
            row["label"] for row, keep in zip(samples, masks["confirmation"]) if keep
        ]
        confirmation_samples = [
            row for row, keep in zip(samples, masks["confirmation"]) if keep
        ]
        choose = scores[masks["confirmation"]] >= threshold
        model_reports[architecture] = {
            "fit_config": fit_config,
            "threshold_selection_on_tuning": threshold_report,
            "confirmation_classification": _classification_groups(
                samples, labels, scores, masks["confirmation"], threshold
            ),
            "confirmation_selection": _selection(confirmation_rows, choose),
            "confirmation_route_bootstrap_gain": _route_bootstrap_gain(
                confirmation_samples, choose, seed=model_seed + 100 + index
            ),
        }
        frozen[architecture] = (model, mean, std, threshold, scores)
        arrays[f"{architecture}__feature_mean"] = mean
        arrays[f"{architecture}__feature_std"] = std
        arrays[f"{architecture}__threshold"] = np.asarray([threshold], dtype=np.float32)
        for name, tensor in model.state_dict().items():
            arrays[f"{architecture}__{name}"] = tensor.detach().cpu().numpy()

    rules = ("never_wait", "target_sinr_worse", "rsrp_delta_not_improving")
    tuning_rows = [row["label"] for row, keep in zip(samples, masks["tuning"]) if keep]
    tuning_features = features[masks["tuning"]]
    rule_tuning = {
        rule: _selection(tuning_rows, np.asarray([
            _rule_wait(feature, rule, names=names) for feature in tuning_features
        ]))
        for rule in rules
    }
    feasible = [rule for rule in rules if rule_tuning[rule]["reliability_not_worse"]]
    selected_rule = max(feasible, key=lambda rule: (
        rule_tuning[rule]["mean_delta_delivered_mbit"],
        -rule_tuning[rule]["false_wait_actions"],
        -rule_tuning[rule]["wait_actions"],
    ))
    confirmation_samples = [
        row for row, keep in zip(samples, masks["confirmation"]) if keep
    ]
    confirmation_labels = [row["label"] for row in confirmation_samples]
    confirmation_features = features[masks["confirmation"]]
    rule_choose = np.asarray([
        _rule_wait(feature, selected_rule, names=names)
        for feature in confirmation_features
    ])
    rule_confirmation = {
        "classification": _rule_classification(
            samples, labels, features, masks["confirmation"],
            rule=selected_rule, names=names,
        ),
        "selection": _selection(confirmation_labels, rule_choose),
        "route_bootstrap_gain": _route_bootstrap_gain(
            confirmation_samples, rule_choose, seed=model_seed + 200
        ),
    }
    mlp_scores = frozen["mlp_small"][4][masks["confirmation"]]
    mlp_threshold = float(frozen["mlp_small"][3])
    mlp_choose = mlp_scores >= mlp_threshold
    zero_choose = np.zeros(len(confirmation_samples), dtype=bool)
    mlp_vs_now = _paired_selection_bootstrap(
        confirmation_samples, zero_choose, mlp_choose,
        seed=model_seed + 300,
    )
    mlp_vs_rule = _paired_selection_bootstrap(
        confirmation_samples, rule_choose, mlp_choose,
        seed=model_seed + 301,
    )
    primary = model_reports["mlp_small"]
    classification = primary["confirmation_classification"]["overall"]
    local_gate = bool(
        classification["average_precision"] >= classification["prevalence"] + 0.05
        and primary["confirmation_selection"]["reliability_not_worse"]
        and primary["confirmation_route_bootstrap_gain"]["ci95"][0] > 0
    )
    beats_rule = bool(
        mlp_vs_rule["ci95"]["delta_delivered_mbit"][0] > 0
        and all(mlp_vs_rule["ci95"][field][1] <= 0 for field in RISK_FIELDS)
    )
    t4b_context = None
    if t4b_report_path is not None:
        old = _read_json(t4b_report_path)
        old_primary = old["models"]["mlp_small"]
        t4b_context = {
            "path": str(t4b_report_path.resolve()),
            "sha256": _sha256(t4b_report_path),
            "development_holdout_rows": old_primary["holdout_classification"]["overall"]["rows"],
            "development_holdout_positive_rows": old_primary["holdout_classification"]["overall"]["positive_rows"],
            "development_holdout_average_precision": old_primary["holdout_classification"]["overall"]["average_precision"],
            "development_holdout_precision": old_primary["holdout_classification"]["overall"]["precision"],
            "development_holdout_recall": old_primary["holdout_classification"]["overall"]["recall"],
            "warning": "Context only: T4B used different routes and is not a paired comparison.",
        }

    output_dir.mkdir(parents=True)
    np.savez_compressed(output_dir / "frozen_models.npz", **arrays)
    with (output_dir / "samples.jsonl").open("w", encoding="utf-8") as stream:
        for row in samples:
            serializable = {**row, "feature": row["feature"].tolist()}
            stream.write(json.dumps(serializable, allow_nan=False) + "\n")
    report = {
        "schema_name": SCHEMA_NAME, "schema_version": SCHEMA_VERSION,
        "status": "COMPLETE", "scope": "train_fit_tuning_confirmation_local_probe",
        "validation_or_test_rows_read": 0,
        "feature_contract": {
            "schema": FEATURE_SCHEMA, "names": list(names),
            "causal": True, "future_features": [],
            "target": "fixed_current_C_star",
        },
        "split_contract": {
            "fit_and_tuning_used_before_freeze": True,
            "confirmation_used_for_fit": False,
            "confirmation_used_for_threshold": False,
            "route_counts": dict(bank_report["fold_route_counts"]),
        },
        "sample_counts": {
            "raw": len(raw_anchors), "duplicate_physical_states_removed": len(raw_anchors) - len(samples),
            "total": len(samples),
            **{fold: int(mask.sum()) for fold, mask in masks.items()},
        },
        "positive_counts": {
            fold: int(labels[mask].sum()) for fold, mask in masks.items()
        },
        "models": model_reports,
        "simple_rules_tuning": rule_tuning,
        "selected_simple_rule": selected_rule,
        "simple_rule_confirmation": rule_confirmation,
        "paired_confirmation": {
            "mlp_minus_ho_now": mlp_vs_now,
            "mlp_minus_simple_rule": mlp_vs_rule,
        },
        "t4b_context": t4b_context,
        "gates": {
            "g_local_learnability": local_gate,
            "g_beats_simple_rule": beats_rule,
            "overall_t4c_probe": "PASSED" if local_gate and beats_rule else "NOT_PASSED",
            "next_gate": "full-route confirmation audit; PPO remains closed",
        },
    }
    _write_json(output_dir / "t4c_report.json", report)
    manifest = {
        "schema_name": SCHEMA_NAME, "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "validation_or_test_rows_read": 0, "model_seed": model_seed,
        "inputs": {
            name: {"path": str(path.resolve()), "sha256": _sha256(path)}
            for name, path in {
                "route_split": bank_dir / "route_split.json",
                "bank_report": bank_dir / "bank_report.json",
                "anchors": bank_dir / "anchors.jsonl",
                "local_outcomes": bank_dir / "local_outcomes.jsonl",
            }.items()
        },
        "t4b_context": t4b_context,
        "source_files": [
            {"path": str(path.relative_to(root)), "sha256": _sha256(path)}
            for path in (
                Path(__file__).resolve(),
                root / "src/ho_optim_drl/commands/audit_early_timing_learnability.py",
                root / "src/ho_optim_drl/commands/build_post_ttt_train_bank.py",
            )
        ],
        "git": _git_state(root),
    }
    _write_json(output_dir / "run_manifest.json", manifest)
    mlp = model_reports["mlp_small"]
    rule_overall = rule_confirmation["classification"]["overall"]
    lines = [
        "# T4C post-TTT probe", "",
        f"- Trạng thái: **{report['gates']['overall_t4c_probe']}**.",
        f"- Fit/tuning/confirmation: {report['sample_counts']['fit']}/{report['sample_counts']['tuning']}/{report['sample_counts']['confirmation']} state.",
        f"- Positive: {report['positive_counts']['fit']}/{report['positive_counts']['tuning']}/{report['positive_counts']['confirmation']}.",
        f"- MLP threshold khóa trên tuning: {mlp_threshold:.4f}.",
        f"- MLP confirmation AP/precision/recall: {classification['average_precision']:.4f}/{classification['precision']:.4f}/{classification['recall']:.4f}.",
        f"- Rule `{selected_rule}` confirmation AP/precision/recall: {rule_overall['average_precision']:.4f}/{rule_overall['precision']:.4f}/{rule_overall['recall']:.4f}.",
        f"- Local learnability gate: {local_gate}; beats-rule gate: {beats_rule}.",
        "", "Confirmation không được dùng để fit hoặc chọn threshold. Future chỉ tạo nhãn counterfactual, không đi vào observation.",
    ]
    (output_dir / "report_vi.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main(root_path: str, args: Namespace) -> int:
    root = Path(root_path).resolve()
    resolve = lambda value: (Path(value) if Path(value).is_absolute() else root / value).resolve()
    context_value = getattr(args, "t4b_report", None)
    report = run_probe(
        root=root, data_dir=resolve(args.data_dir), bank_dir=resolve(args.bank_dir),
        output_dir=resolve(args.output_dir), model_seed=int(args.model_seed),
        t4b_report_path=None if context_value is None else resolve(context_value),
    )
    primary = report["models"]["mlp_small"]["confirmation_classification"]["overall"]
    print(json.dumps({
        "status": report["status"], "gates": report["gates"],
        "mlp_confirmation": primary,
        "selected_simple_rule": report["selected_simple_rule"],
    }, indent=2), flush=True)
    return 0
