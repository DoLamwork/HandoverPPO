"""Summarize completed PPO ablation arms without touching test data."""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable


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


def _selected_validation_row(
    rows: list[dict[str, Any]], selected: dict[str, Any]
) -> dict[str, Any]:
    matches = [row for row in rows if _same_metrics(row, selected)]
    if not matches:
        raise ValueError(
            "The selected validation metrics do not match any scheduled "
            "validation checkpoint."
        )
    # Several deterministic checkpoints can have identical metrics. The first
    # match is the checkpoint that could first have established the saved best.
    return matches[0]


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
    selected_row = _selected_validation_row(rows, selected_metrics)
    last_row = rows[-1]
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
        "last_scheduled_validation_step": int(last_row["macro_steps"]),
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
