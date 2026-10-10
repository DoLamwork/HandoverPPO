"""Prepare leakage-safe D63-2 timing and target labels from D63-1 artifacts."""

from __future__ import annotations

from argparse import Namespace
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ho_optim_drl.commands.audit_policy_learning import _sha256_file


SCHEMA = "d63_2_counterfactual_labels_v1"
MAIN_HORIZON_SECONDS = 1.0
DELAYS_MS = (0, 100, 200, 300)
TARGET_SLOTS = (1, 2, 3)
FLOAT_EPS = 1e-9
RISK_METRICS = (
    "handover_failures",
    "rlf_starts",
    "pure_rlf_starts",
    "ping_pong_count",
    "recovery_seconds",
    "handover_interruption_seconds",
)
DELTA_METRICS = (
    "delivered_mbit",
    "relative_rate_percent",
    "connected_seconds",
    "handover_interruption_seconds",
    "recovery_seconds",
    "handover_attempts",
    "handover_successes",
    "handover_failures",
    "rlf_starts",
    "pure_rlf_starts",
    "ping_pong_count",
)


def _json_write(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _prepare_destination(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc


def build_route_split(
    routes: Iterable[tuple[int, int]], *, seed: int
) -> tuple[dict[tuple[int, int], str], dict[str, Any]]:
    """Create a deterministic 60/20/20 route split independently per speed."""
    grouped: dict[int, list[int]] = defaultdict(list)
    for speed, trajectory_id in sorted(set(routes)):
        grouped[int(speed)].append(int(trajectory_id))
    mapping: dict[tuple[int, int], str] = {}
    per_speed: dict[str, Any] = {}
    for speed, trajectory_ids in sorted(grouped.items()):
        values = np.asarray(sorted(set(trajectory_ids)), dtype=np.int64)
        if values.size < 5:
            raise ValueError(
                f"Need at least five routes at speed {speed}; found {values.size}."
            )
        rng = np.random.default_rng(np.random.SeedSequence((int(seed), int(speed))))
        shuffled = values[rng.permutation(values.size)].tolist()
        fit_count = int(np.floor(0.60 * len(shuffled)))
        tuning_count = int(np.floor(0.20 * len(shuffled)))
        holdout_count = len(shuffled) - fit_count - tuning_count
        if min(fit_count, tuning_count, holdout_count) < 1:
            raise ValueError(f"Empty route fold at speed {speed}.")
        assignments = {
            "fit": shuffled[:fit_count],
            "tuning": shuffled[fit_count : fit_count + tuning_count],
            "holdout": shuffled[fit_count + tuning_count :],
        }
        for fold, ids in assignments.items():
            for trajectory_id in ids:
                mapping[(speed, int(trajectory_id))] = fold
        per_speed[str(speed)] = {
            fold: sorted(int(value) for value in ids)
            for fold, ids in assignments.items()
        }
    return mapping, {
        "split_seed": int(seed),
        "split_unit": "(speed_kmh, trajectory_id)",
        "fractions": {"fit": 0.60, "tuning": 0.20, "holdout": 0.20},
        "per_speed": per_speed,
    }


def _requested_action_valid(row: dict[str, Any]) -> bool:
    execution = dict(row.get("execution") or {})
    return (
        execution.get("status") == "executed"
        and execution.get("fallback") is None
        and int(execution.get("executed_slot", -1)) == int(row["requested_slot"])
    )


def compare_branch(
    row: dict[str, Any], control: dict[str, Any]
) -> dict[str, Any]:
    """Return continuous benefit/risk deltas and non-collapsed labels."""
    physical = dict(row["physical"])
    control_physical = dict(control["physical"])
    deltas = {
        f"delta_{metric}": float(physical[metric]) - float(control_physical[metric])
        for metric in DELTA_METRICS
    }
    rate_gain = deltas["delta_delivered_mbit"] > FLOAT_EPS
    rate_loss = deltas["delta_delivered_mbit"] < -FLOAT_EPS
    risk_worse = any(deltas[f"delta_{metric}"] > FLOAT_EPS for metric in RISK_METRICS)
    risk_better = any(deltas[f"delta_{metric}"] < -FLOAT_EPS for metric in RISK_METRICS)
    valid = _requested_action_valid(row)
    is_control = bool(row.get("is_control", False))
    strict_useful = bool(valid and not is_control and rate_gain and not risk_worse)
    if is_control:
        pareto_class = "control"
    elif not valid:
        pareto_class = "unavailable_or_fallback"
    elif rate_gain and not risk_worse:
        pareto_class = "strict_dominant"
    elif rate_gain and risk_worse:
        pareto_class = "rate_risk_tradeoff"
    elif rate_loss and risk_better:
        pareto_class = "rate_loss_reliability_gain"
    elif (not rate_loss) and risk_better:
        pareto_class = "reliability_dominant"
    else:
        pareto_class = "dominated_or_equal"
    return {
        **deltas,
        "rate_positive": bool(rate_gain),
        "risk_worse": bool(risk_worse),
        "risk_better": bool(risk_better),
        "strict_useful": strict_useful,
        "pareto_class": pareto_class,
        "requested_action_valid": bool(valid),
    }


def _branch_rank(row: dict[str, Any]) -> tuple[float, ...]:
    """Physical lexicographic rank used only as an auxiliary teacher label."""
    p = row["physical"]
    return (
        -float(p["handover_failures"]),
        -float(p["rlf_starts"]),
        -float(p["pure_rlf_starts"]),
        -float(p["ping_pong_count"]),
        -float(p["recovery_seconds"]),
        -float(p["handover_interruption_seconds"]),
        float(p["delivered_mbit"]),
        -float(row["delay_ms"]),
        -float(row["requested_slot"]),
    )


def _family_rows(rows: list[dict[str, Any]], family: str) -> list[dict[str, Any]]:
    if family == "timing":
        return [row for row in rows if int(row["requested_slot"]) == 1]
    if family == "target":
        return [row for row in rows if int(row["delay_ms"]) == 0]
    if family == "joint":
        return list(rows)
    raise ValueError(family)


def _array(values: list[Any], dtype: Any | None = None) -> np.ndarray:
    return np.asarray(values, dtype=dtype) if dtype is not None else np.asarray(values)


def _write_family(
    *,
    output_path: Path,
    family: str,
    state_bank: dict[str, np.ndarray],
    state_indices: dict[str, int],
    rows_by_state: dict[str, list[dict[str, Any]]],
    route_folds: dict[tuple[int, int], str],
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for state_id in state_bank["state_id"].tolist():
        rows = _family_rows(rows_by_state[str(state_id)], family)
        control = next(
            row
            for row in rows_by_state[str(state_id)]
            if int(row["delay_ms"]) == 0 and int(row["requested_slot"]) == 1
        )
        best = max(rows, key=_branch_rank)
        index = state_indices[str(state_id)]
        speed = int(state_bank["speed_kmh"][index])
        trajectory_id = int(state_bank["trajectory_id"][index])
        for row in sorted(rows, key=lambda item: (int(item["delay_ms"]), int(item["requested_slot"]))):
            compared = compare_branch(row, control)
            execution = dict(row.get("execution") or {})
            records.append(
                {
                    "state_index": index,
                    "state_id": str(state_id),
                    "source": str(state_bank["source"][index]),
                    "speed_kmh": speed,
                    "trajectory_id": trajectory_id,
                    "fold": route_folds[(speed, trajectory_id)],
                    "planned_delay_ms": int(row["delay_ms"]),
                    "planned_slot": int(row["requested_slot"]),
                    "executed_slot": int(execution.get("executed_slot") or -1),
                    "horizon_censored": bool(row.get("horizon_censored", False)),
                    "is_control": bool(row.get("is_control", False)),
                    "is_best_physical_in_family": (
                        int(row["delay_ms"]) == int(best["delay_ms"])
                        and int(row["requested_slot"]) == int(best["requested_slot"])
                    ),
                    **compared,
                }
            )

    indices = _array([record["state_index"] for record in records], np.int32)
    arrays: dict[str, np.ndarray] = {
        "state_id": _array([record["state_id"] for record in records]),
        "source": _array([record["source"] for record in records]),
        "speed_kmh": _array([record["speed_kmh"] for record in records], np.int32),
        "trajectory_id": _array([record["trajectory_id"] for record in records], np.int32),
        "fold": _array([record["fold"] for record in records]),
        "observation": state_bank["observation"][indices].astype(np.float32, copy=False),
        "planned_delay_ms": _array([record["planned_delay_ms"] for record in records], np.int16),
        "planned_slot": _array([record["planned_slot"] for record in records], np.int8),
        "executed_slot": _array([record["executed_slot"] for record in records], np.int8),
        "horizon_censored": _array([record["horizon_censored"] for record in records], bool),
        "is_control": _array([record["is_control"] for record in records], bool),
        "is_best_physical_in_family": _array([record["is_best_physical_in_family"] for record in records], bool),
        "rate_positive": _array([record["rate_positive"] for record in records], bool),
        "risk_worse": _array([record["risk_worse"] for record in records], bool),
        "risk_better": _array([record["risk_better"] for record in records], bool),
        "strict_useful": _array([record["strict_useful"] for record in records], bool),
        "requested_action_valid": _array([record["requested_action_valid"] for record in records], bool),
        "pareto_class": _array([record["pareto_class"] for record in records]),
    }
    for metric in DELTA_METRICS:
        arrays[f"delta_{metric}"] = _array(
            [record[f"delta_{metric}"] for record in records], np.float32
        )
    np.savez_compressed(output_path, **arrays)

    by_fold: dict[str, Any] = {}
    for fold in ("fit", "tuning", "holdout"):
        subset = [record for record in records if record["fold"] == fold]
        positive_routes = {
            (record["speed_kmh"], record["trajectory_id"])
            for record in subset
            if record["strict_useful"]
        }
        by_fold[fold] = {
            "rows": len(subset),
            "states": len({record["state_id"] for record in subset}),
            "routes": len({(record["speed_kmh"], record["trajectory_id"]) for record in subset}),
            "strict_useful_rows": sum(bool(record["strict_useful"]) for record in subset),
            "strict_useful_routes": len(positive_routes),
            "rate_positive_rows": sum(bool(record["rate_positive"]) for record in subset),
            "risk_worse_rows": sum(bool(record["risk_worse"]) for record in subset),
        }
    return {
        "family": family,
        "path": output_path.name,
        "sha256": _sha256_file(output_path),
        "rows": len(records),
        "states": len({record["state_id"] for record in records}),
        "routes": len({(record["speed_kmh"], record["trajectory_id"]) for record in records}),
        "strict_useful_rows": sum(bool(record["strict_useful"]) for record in records),
        "strict_useful_routes": len({
            (record["speed_kmh"], record["trajectory_id"])
            for record in records if record["strict_useful"]
        }),
        "pareto_classes": dict(sorted(Counter(record["pareto_class"] for record in records).items())),
        "by_fold": by_fold,
    }


def prepare_labels(
    *, bank_dir: Path, main_audit_dir: Path | None, output_dir: Path, split_seed: int
) -> dict[str, Any]:
    _prepare_destination(output_dir)
    bank_path = bank_dir / "state_bank.npz"
    bank_manifest_path = bank_dir / "state_bank_manifest.json"
    outcomes_path = bank_dir / "timing_target_outcomes.jsonl"
    for path in (bank_path, bank_manifest_path, outcomes_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    bank_manifest = _read_json(bank_manifest_path)
    if bank_manifest.get("split") != "train" or int(bank_manifest.get("validation_or_test_rows_read", -1)) != 0:
        raise ValueError("D63-2 labels require a TRAIN-only D63 bank.")
    if int(bank_manifest.get("max_anchors_per_source_route", -1)) != 5:
        raise ValueError("Expected the frozen D63-1 bank with max 5 anchors/source/route.")

    with np.load(bank_path, allow_pickle=False) as loaded:
        state_bank = {name: loaded[name] for name in loaded.files}
    state_ids = [str(value) for value in state_bank["state_id"].tolist()]
    if len(state_ids) != len(set(state_ids)):
        raise ValueError("state_bank contains duplicate state_id values.")
    state_indices = {state_id: index for index, state_id in enumerate(state_ids)}
    routes = {
        (int(speed), int(route))
        for speed, route in zip(state_bank["speed_kmh"], state_bank["trajectory_id"])
    }
    route_folds, split_manifest = build_route_split(routes, seed=split_seed)

    rows_by_state: dict[str, list[dict[str, Any]]] = defaultdict(list)
    input_rows = 0
    for row in _iter_jsonl(outcomes_path):
        if abs(float(row["horizon_seconds"]) - MAIN_HORIZON_SECONDS) > 1e-12:
            continue
        state_id = str(row["state_id"])
        if state_id not in state_indices:
            raise ValueError(f"Outcome references unknown state_id {state_id}.")
        rows_by_state[state_id].append(row)
        input_rows += 1
    expected_pairs = {(delay, slot) for delay in DELAYS_MS for slot in TARGET_SLOTS}
    for state_id in state_ids:
        rows = rows_by_state.get(state_id, [])
        pairs = {(int(row["delay_ms"]), int(row["requested_slot"])) for row in rows}
        if len(rows) != len(expected_pairs) or pairs != expected_pairs:
            raise ValueError(
                f"State {state_id} does not contain exactly the registered 12 branches."
            )

    split_manifest.update(
        {
            "schema_name": "d63_2_route_split_v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "route_count": len(routes),
            "fold_route_counts": dict(sorted(Counter(route_folds.values()).items())),
            "assignments": [
                {"speed_kmh": speed, "trajectory_id": route, "fold": fold}
                for (speed, route), fold in sorted(route_folds.items())
            ],
            "leakage_check": {
                "route_overlap_between_folds": 0,
                "same_route_all_sources_and_anchors": True,
            },
        }
    )
    split_path = output_dir / "route_split_manifest.json"
    _json_write(split_path, split_manifest)

    family_reports = {}
    for family in ("timing", "target", "joint"):
        family_reports[family] = _write_family(
            output_path=output_dir / f"{family}_labels.npz",
            family=family,
            state_bank=state_bank,
            state_indices=state_indices,
            rows_by_state=rows_by_state,
            route_folds=route_folds,
        )

    inputs: dict[str, Any] = {
        "state_bank": {"path": str(bank_path.resolve()), "sha256": _sha256_file(bank_path)},
        "state_bank_manifest": {"path": str(bank_manifest_path.resolve()), "sha256": _sha256_file(bank_manifest_path)},
        "counterfactual_outcomes": {"path": str(outcomes_path.resolve()), "sha256": _sha256_file(outcomes_path)},
    }
    if main_audit_dir is not None:
        for name in ("main_report.json", "decision_gate.json"):
            path = main_audit_dir / name
            if not path.is_file():
                raise FileNotFoundError(path)
            inputs[name.removesuffix(".json")] = {
                "path": str(path.resolve()), "sha256": _sha256_file(path)
            }
    registry = {
        "schema_name": SCHEMA,
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_artifacts": inputs,
        "constraints": {
            "data_split": "train_only",
            "validation_test_zero_shot_used": False,
            "main_horizon_seconds": MAIN_HORIZON_SECONDS,
            "candidate_slots": list(TARGET_SLOTS),
            "delays_ms": list(DELAYS_MS),
            "actual_max_anchors_per_source_route": 5,
            "plan_quota_max_anchors_per_source_route": 10,
            "quota_difference_accepted_without_rerun": True,
        },
        "predictor_input_contract": {
            "allowed": ["observation", "planned_delay_ms", "planned_slot"],
            "metadata_not_allowed_as_input": ["source", "speed_kmh", "trajectory_id", "fold", "state_id"],
            "teacher_only": ["all delta_* fields", "rate_positive", "risk_worse", "risk_better", "strict_useful", "pareto_class", "is_best_physical_in_family"],
            "future_candidate_identity_in_input": False,
        },
    }
    registry_path = output_dir / "input_registry.json"
    _json_write(registry_path, registry)

    report = {
        "schema_name": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "state_count": len(state_ids),
        "route_count": len(routes),
        "source_counts": dict(sorted(Counter(str(value) for value in state_bank["source"]).items())),
        "speed_state_counts": dict(sorted(Counter(int(value) for value in state_bank["speed_kmh"]).items())),
        "selected_horizon_input_rows": input_rows,
        "route_split": {
            "path": split_path.name,
            "sha256": _sha256_file(split_path),
            "fold_route_counts": split_manifest["fold_route_counts"],
        },
        "families": family_reports,
        "interpretation": {
            "timing": "C1 at 0/100/200/300 ms, each compared with immediate C1.",
            "target": "C1/C2/C3 at delay 0, each compared with immediate C1.",
            "joint": "All 12 timing x target branches, each compared with immediate C1.",
            "labels_are_not_single_argmax": True,
            "continuous_gain_and_risk_are_preserved": True,
        },
    }
    report_path = output_dir / "label_report.json"
    _json_write(report_path, report)
    manifest = {
        "schema_name": "d63_2_label_manifest_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": [
            {"path": path.name, "size_bytes": path.stat().st_size, "sha256": _sha256_file(path)}
            for path in sorted(output_dir.iterdir()) if path.is_file()
        ],
    }
    _json_write(output_dir / "label_manifest.json", manifest)
    return report


def main(root_path: str, args: Namespace) -> int:
    root = Path(root_path)
    report = prepare_labels(
        bank_dir=(root / args.bank_dir).resolve(),
        main_audit_dir=(root / args.main_audit_dir).resolve() if args.main_audit_dir else None,
        output_dir=(root / args.output_dir).resolve(),
        split_seed=int(args.route_split_seed),
    )
    print(
        "D63-2 labels prepared: "
        f"{report['state_count']} states, {report['route_count']} routes, "
        f"timing/target/joint positives="
        f"{report['families']['timing']['strict_useful_rows']}/"
        f"{report['families']['target']['strict_useful_rows']}/"
        f"{report['families']['joint']['strict_useful_rows']}."
    )
    return 0
