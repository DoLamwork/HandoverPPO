"""Summarize completed PPO ablation arms without touching test data."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import pickle
from pathlib import Path
from typing import Any, Iterable

import torch


_MATCH_FIELDS = (
    "mean_return",
    "mean_estimated_rate_mbps",
    "mean_relative_rate_percent",
    "rlf_time_percent",
    "handover_failure_percent",
    "ping_pong_percent",
    "handover_fraction",
)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as source:
        value = json.load(source)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def _as_float(value: Any, default: float = 0.0) -> float:
    if value in (None, ""):
        return default
    return float(value)


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if value in (None, ""):
        return {}
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("Expected a serialized JSON object.")
    return parsed


def _load_validation_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    if not rows:
        raise ValueError(f"No validation records found: {path}")
    for row in rows:
        for name in ("action_counts", "action_fractions"):
            row[name] = json.loads(row.get(name) or "{}")
    return rows


def _same_metrics(row: dict[str, Any], selected: dict[str, Any]) -> bool:
    return all(
        math.isclose(
            _as_float(row.get(name)),
            _as_float(selected.get(name)),
            rel_tol=1e-10,
            abs_tol=1e-10,
        )
        for name in _MATCH_FIELDS
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _existing_checkpoint_path(root: Path, value: Any) -> Path | None:
    if not value:
        return None
    path = Path(str(value))
    candidates = [path]
    if not path.is_absolute():
        candidates.append(root / path)
    candidates.append(root / path.name)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _checkpoint_step(root: Path, report: dict[str, Any]) -> tuple[int | None, str]:
    """Resolve the selected step from immutable checkpoint evidence."""
    checkpoints = dict(report.get("checkpoints") or {})
    selected = _existing_checkpoint_path(root, checkpoints.get("selected_from"))
    if selected is None:
        selected = _existing_checkpoint_path(root, checkpoints.get("selected_export"))
    if selected is None:
        return None, "unavailable"

    if selected.name.startswith("step_"):
        try:
            return int(selected.stem.removeprefix("step_")), "checkpoint_filename"
        except ValueError:
            pass
    try:
        selected_hash = _sha256_file(selected)
        for candidate in sorted((root / "checkpoints" / "all").glob("step_*.pt")):
            if _sha256_file(candidate) == selected_hash:
                return (
                    int(candidate.stem.removeprefix("step_")),
                    "checkpoint_sha256",
                )
    except OSError:
        pass
    try:
        payload = torch.load(selected, map_location="cpu", weights_only=True)
        step = payload.get("total_macro_steps")
        if step is not None:
            return int(step), "checkpoint_counter"
    except (
        OSError,
        RuntimeError,
        ValueError,
        TypeError,
        EOFError,
        AttributeError,
        pickle.UnpicklingError,
    ):
        pass
    return None, "unavailable"


def _selected_validation_row(
    rows: list[dict[str, Any]],
    selected: dict[str, Any],
    *,
    selected_step: int | None = None,
) -> tuple[dict[str, Any], str]:
    if selected_step is not None:
        step_matches = [
            row for row in rows if int(row["macro_steps"]) == selected_step
        ]
        if len(step_matches) == 1:
            if not _same_metrics(step_matches[0], selected):
                raise ValueError(
                    "Selected checkpoint step exists, but its validation "
                    "metrics disagree with the final selected evaluation."
                )
            return step_matches[0], "explicit_step"
    matches = [row for row in rows if _same_metrics(row, selected)]
    if not matches:
        raise ValueError(
            "The selected validation metrics do not match any scheduled "
            "validation checkpoint."
        )
    if len(matches) == 1:
        return matches[0], "unique_metric_match"
    # Historical reports did not record the selected counter explicitly.
    # Keep them readable, but label this ambiguity instead of presenting the
    # first identical row as ground truth.
    return matches[0], "inferred_first_identical_metric_match"


def _last_training_step(root: Path, report: dict[str, Any]) -> int | None:
    metrics_path = root / "training_metrics.jsonl"
    last = None
    if metrics_path.is_file():
        with metrics_path.open("r", encoding="utf-8") as source:
            for line in source:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get("event") == "update_complete":
                    last = int(event["macro_steps"])
    if last is not None:
        return last
    history = report.get("history") or []
    return int(history[-1]["macro_steps"]) if history else None


def _policy_mode(action_fractions: dict[str, Any]) -> str:
    stay = _as_float(action_fractions.get("STAY"))
    c1 = _as_float(action_fractions.get("HO_C1"))
    alternative = sum(
        _as_float(value)
        for name, value in action_fractions.items()
        if name not in {"STAY", "HO_C1"}
    )
    if stay >= 0.99:
        return "STAY_COLLAPSE"
    if c1 >= 0.99:
        return "HO_C1_COLLAPSE_3GPP_EQUIVALENT"
    if alternative >= 0.01:
        return "USES_ALTERNATIVE_CANDIDATES"
    return "MIXED_STAY_C1"


def _validation_snapshot(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return {
        "macro_steps": int(row["macro_steps"]),
        "update": int(row.get("update") or 0),
        "episodes": int(row.get("episodes") or 0),
        "skipped_no_decision": int(row.get("skipped_no_decision") or 0),
        "qos_feasible": _as_bool(row.get("qos_feasible")),
        "mean_return": _as_float(row.get("mean_return")),
        "mean_estimated_rate_mbps": _as_float(
            row.get("mean_estimated_rate_mbps")
        ),
        "mean_relative_rate_percent": _as_float(
            row.get("mean_relative_rate_percent")
        ),
        "p10_relative_rate_percent": _as_float(
            row.get("p10_relative_rate_percent")
        ),
        "mean_relative_rate_percent_by_speed": _as_dict(
            row.get("mean_relative_rate_percent_by_speed")
        ),
        "p10_relative_rate_percent_by_speed": _as_dict(
            row.get("p10_relative_rate_percent_by_speed")
        ),
        "rlf_time_percent": _as_float(row.get("rlf_time_percent")),
        "handover_failure_percent": _as_float(
            row.get("handover_failure_percent")
        ),
        "ping_pong_percent": _as_float(row.get("ping_pong_percent")),
        "total_handover_attempts": int(row.get("total_handover_attempts") or 0),
        "total_rlf": int(row.get("total_rlf") or 0),
        "total_ping_pong": int(row.get("total_ping_pong") or 0),
        "action_counts": _as_dict(row.get("action_counts")),
        "action_fractions": _as_dict(row.get("action_fractions")),
    }


def summarize_arm(
    arm_dir: str | os.PathLike[str], *, expected_budget: int | None = None
) -> dict[str, Any]:
    """Validate and summarize one completed training arm."""

    root = Path(arm_dir)
    config_path = root / "experiment_config.json"
    report_path = root / "training_report.json"
    metrics_path = root / "checkpoints" / "validation_metrics.csv"
    for path in (config_path, report_path, metrics_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing required ablation artifact: {path}")

    config = _load_json(config_path)
    report = _load_json(report_path)
    experiment_id = str(config.get("experiment_id") or "")
    if not experiment_id or experiment_id != root.name:
        raise ValueError(
            f"Experiment ID mismatch for {root}: {experiment_id!r}"
        )
    report_config = report.get("experiment_config", {}).get("config", {})
    if report_config.get("experiment_id") != experiment_id:
        raise ValueError(f"Training report experiment ID mismatch: {root}")

    budget = int(config.get("budget", {}).get("total_macro_steps", 0))
    if expected_budget is not None and budget != expected_budget:
        raise ValueError(
            f"{experiment_id} has budget {budget}, expected {expected_budget}."
        )
    if bool(report.get("test_evaluated")):
        raise ValueError(f"{experiment_id} unexpectedly evaluated the test split.")

    checkpoint_paths = report.get("checkpoints", {})
    for name in ("latest", "selected_export"):
        path = Path(str(checkpoint_paths.get(name) or ""))
        if not path.is_file():
            raise FileNotFoundError(
                f"{experiment_id} is missing checkpoint {name}: {path}"
            )

    rows = _load_validation_rows(metrics_path)
    selected_metrics = dict(report.get("validation") or {})
    checkpoint_step, checkpoint_step_source = _checkpoint_step(root, report)
    selected_row, selected_row_resolution = _selected_validation_row(
        rows,
        selected_metrics,
        selected_step=checkpoint_step,
    )
    last_row = rows[-1]
    budget_end_row = (
        last_row if int(last_row["macro_steps"]) == budget else None
    )
    interval = int(
        config.get("budget", {}).get("validation_interval_macro_steps", 0)
    )
    scheduled_rows = [
        row
        for row in rows
        if interval <= 0 or int(row["macro_steps"]) % interval == 0
    ]
    last_scheduled_row = scheduled_rows[-1] if scheduled_rows else last_row
    action_fractions = dict(selected_metrics.get("action_fractions") or {})
    alternative_fraction = sum(
        _as_float(value)
        for name, value in action_fractions.items()
        if name not in {"STAY", "HO_C1"}
    )
    candidate = dict(config.get("candidate") or {})
    observation = dict(config.get("observation_schema") or {})
    checkpoint_selection = dict(report.get("checkpoint_selection") or {})

    return {
        "experiment_id": experiment_id,
        "parent_id": config.get("parent_id"),
        "changed_factor": config.get("changed_factor"),
        "candidate_count": int(candidate.get("K", 0)),
        "candidate_order": candidate.get("candidate_order_key"),
        "observation_size": int(observation.get("size", 0)),
        "action_size": len(config.get("action_schema") or []),
        "seed": int(config.get("seeds", {}).get("training", 0)),
        "budget_macro_steps": budget,
        "test_evaluated": False,
        "experiment_status": report.get("experiment_status"),
        "selection_reason": report.get("selection_reason"),
        "validation_checkpoints": int(
            checkpoint_selection.get("validation_checkpoints", len(rows))
        ),
        "qos_feasible_checkpoints": int(
            checkpoint_selection.get("qos_feasible_checkpoints", 0)
        ),
        "selected_step": int(selected_row["macro_steps"]),
        "selected_step_source": (
            checkpoint_step_source
            if checkpoint_step is not None
            else selected_row_resolution
        ),
        "selected_step_inferred": checkpoint_step is None,
        "last_scheduled_validation_step": int(
            last_scheduled_row["macro_steps"]
        ),
        "budget_end_validation_step": (
            int(budget_end_row["macro_steps"])
            if budget_end_row is not None
            else None
        ),
        "training_last_macro_steps": _last_training_step(root, report),
        "selected_validation": _validation_snapshot(selected_row),
        "last_scheduled_validation": _validation_snapshot(last_scheduled_row),
        "budget_end_validation": _validation_snapshot(budget_end_row),
        "selected_qos_feasible": _as_bool(selected_row.get("qos_feasible")),
        "selected_rate_mbps": _as_float(
            selected_metrics.get("mean_estimated_rate_mbps")
        ),
        "selected_relative_rate_percent": _as_float(
            selected_metrics.get("mean_relative_rate_percent")
        ),
        "selected_p10_relative_rate_percent": _as_float(
            selected_metrics.get("p10_relative_rate_percent")
        ),
        "selected_rlf_time_percent": _as_float(
            selected_metrics.get("rlf_time_percent")
        ),
        "selected_hof_percent": _as_float(
            selected_metrics.get("handover_failure_percent")
        ),
        "selected_ping_pong_percent": _as_float(
            selected_metrics.get("ping_pong_percent")
        ),
        "selected_stay_fraction": _as_float(action_fractions.get("STAY")),
        "selected_c1_fraction": _as_float(action_fractions.get("HO_C1")),
        "selected_alternative_fraction": alternative_fraction,
        "selected_policy_mode": _policy_mode(action_fractions),
        "selected_matches_3gpp": bool(
            math.isclose(
                _as_float(selected_row.get("aggregate_rate_ratio_to_3gpp")),
                1.0,
                abs_tol=1e-12,
            )
            and math.isclose(
                _as_float(
                    selected_row.get("aggregate_relative_rate_ratio_to_3gpp")
                ),
                1.0,
                abs_tol=1e-12,
            )
            and _as_float(action_fractions.get("HO_C1")) >= 0.99
        ),
        "last_validation_qos_feasible": _as_bool(last_row.get("qos_feasible")),
        "last_validation_relative_rate_percent": _as_float(
            last_row.get("mean_relative_rate_percent")
        ),
        "last_validation_stay_fraction": _as_float(
            last_row["action_fractions"].get("STAY")
        ),
        "last_validation_c1_fraction": _as_float(
            last_row["action_fractions"].get("HO_C1")
        ),
        "last_validation_policy_mode": _policy_mode(last_row["action_fractions"]),
        "dataset_manifest_sha256": (
            report.get("dataset_manifest", {}).get("sha256")
            or report.get("dataset_manifest", {}).get("manifest_sha256")
        ),
        "dataset_content_sha256": (
            report.get("dataset_manifest", {}).get("dataset_content_sha256")
            or report.get("experiment_config", {})
            .get("config", {})
            .get("dataset", {})
            .get("content_sha256")
        ),
        "source_manifest_sha256": (
            report.get("source_state", {}).get("source_manifest_sha256")
        ),
        "validation_reference": {
            "label": "3gpp_always_ho",
            "mean_estimated_rate_mbps": report.get(
                "validation_3gpp_reference", {}
            ).get("mean_estimated_rate_mbps"),
            "mean_relative_rate_percent": report.get(
                "validation_3gpp_reference", {}
            ).get("mean_relative_rate_percent"),
            "episodes": report.get("validation_3gpp_reference", {}).get(
                "episodes"
            ),
        },
    }


def _write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def _write_markdown(path: Path, records: list[dict[str, Any]]) -> None:
    lines = [
        "# PPO ablation screening registry",
        "",
        "Generated from validation artifacts only; the test split was not evaluated.",
        "",
        "| ID | K | Order | Status | Selected step | RR % | RLF-time % | HOF % | PP % | STAY % | C1 % | C2+ % | Mode |",
        "|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for record in records:
        lines.append(
            "| {experiment_id} | {candidate_count} | {candidate_order} | "
            "{experiment_status} | {selected_step} | "
            "{selected_relative_rate_percent:.3f} | "
            "{selected_rlf_time_percent:.3f} | {selected_hof_percent:.3f} | "
            "{selected_ping_pong_percent:.3f} | {stay:.1f} | {c1:.1f} | "
            "{alternative:.1f} | {selected_policy_mode} |".format(
                **record,
                stay=100.0 * record["selected_stay_fraction"],
                c1=100.0 * record["selected_c1_fraction"],
                alternative=100.0 * record["selected_alternative_fraction"],
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize_arms(
    input_dir: str | os.PathLike[str],
    experiment_ids: Iterable[str],
    output_dir: str | os.PathLike[str],
    *,
    expected_budget: int | None = None,
) -> list[dict[str, Any]]:
    records = [
        summarize_arm(
            Path(input_dir) / experiment_id,
            expected_budget=expected_budget,
        )
        for experiment_id in experiment_ids
    ]
    if not records:
        raise ValueError("At least one experiment ID is required.")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "ablation_registry.json").write_text(
        json.dumps(records, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _write_csv(destination / "ablation_registry.csv", records)
    _write_markdown(destination / "ablation_registry.md", records)
    return records


def main(root_path: str, args: Any) -> int:
    input_dir = os.path.abspath(os.path.join(root_path, args.input_dir))
    output_dir = os.path.abspath(os.path.join(root_path, args.output_dir))
    records = summarize_arms(
        input_dir,
        args.experiment_ids,
        output_dir,
        expected_budget=args.expected_budget,
    )
    print(f"Summarized {len(records)} ablation arms: {output_dir}")
    for record in records:
        print(
            f"[{record['experiment_id']}] {record['experiment_status']} | "
            f"step {record['selected_step']} | "
            f"RR {record['selected_relative_rate_percent']:.3f}% | "
            f"{record['selected_policy_mode']}"
        )
    return 0
