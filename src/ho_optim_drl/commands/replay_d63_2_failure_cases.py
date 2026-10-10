"""Replay frozen D63-2 target-MLP cases and audit causal radio dynamics."""

from __future__ import annotations

from argparse import Namespace
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np

from ho_optim_drl.commands.audit_policy_learning import _sha256_file
from ho_optim_drl.commands.audit_timing_target import (
    STAY_REDECISION_MS,
    _execute_requested_slot,
    _source_specs,
    replay_d63_anchors,
    simulate_timing_target,
)
from ho_optim_drl.commands.probe_d63_2 import choose_actions
from ho_optim_drl.protocol import ProtocolPhase


SCHEMA = "d63_2_failure_case_radio_replay_v1"
HORIZON_TICKS = 100
PRE_WINDOW_TICKS = 50
DIAGNOSTIC_THRESHOLD = 0.9


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


def _slope(values: np.ndarray, delta_t_seconds: float) -> float:
    if len(values) < 2:
        return 0.0
    x = np.arange(len(values), dtype=np.float64) * delta_t_seconds
    return float(np.polyfit(x, values.astype(np.float64), 1)[0])


def causal_radio_features(
    *, trace: Any, serving_cell: int, target_cell: int, anchor_tic: int,
    q_in_db: float, delta_t_ms: int,
) -> dict[str, float]:
    """Compute raw causal summaries using samples at or before the decision."""
    first = max(0, anchor_tic - PRE_WINDOW_TICKS)
    short_first = max(0, anchor_tic - 10)
    target_long = trace.sinr_rlm_db[target_cell, first : anchor_tic + 1]
    target_short = trace.sinr_rlm_db[target_cell, short_first : anchor_tic + 1]
    serving_long = trace.sinr_rlm_db[serving_cell, first : anchor_tic + 1]
    rsrp_gap_long = (
        trace.rsrp_a3_db[target_cell, first : anchor_tic + 1]
        - trace.rsrp_a3_db[serving_cell, first : anchor_tic + 1]
    )
    delta_t_seconds = delta_t_ms / 1000.0
    current_qin_gap = float(target_long[-1] - q_in_db)
    long_std = float(np.std(target_long))
    return {
        "target_qin_gap_now_db": current_qin_gap,
        "target_qin_gap_min_100ms_db": float(np.min(target_short) - q_in_db),
        "target_qin_gap_p10_500ms_db": float(np.percentile(target_long, 10) - q_in_db),
        "target_qin_suitable_fraction_500ms": float(np.mean(target_long > q_in_db)),
        "target_sinr_std_500ms_db": long_std,
        "target_sinr_slope_100ms_db_per_s": _slope(target_short, delta_t_seconds),
        "target_sinr_slope_500ms_db_per_s": _slope(target_long, delta_t_seconds),
        "serving_sinr_std_500ms_db": float(np.std(serving_long)),
        "target_rsrp_gap_now_db": float(rsrp_gap_long[-1]),
        "target_rsrp_gap_slope_500ms_db_per_s": _slope(rsrp_gap_long, delta_t_seconds),
        # Proposed D4 feature: a raw long-window lower-confidence margin.
        "execution_safety_margin_v1_db": current_qin_gap - long_std,
    }


def _radio_record(
    *, case_id: str, branch: str, offset_ticks: int, snapshot: Any,
    initial_serving: int, c1_cell: int, selected_cell: int,
    q_in_db: float, q_out_db: float, delivered_rate_mbps: float,
    best_rate_mbps: float, events: dict[str, bool],
) -> dict[str, Any]:
    def cell_values(cell: int) -> dict[str, float]:
        return {
            "rsrp_db": float(snapshot.rsrp_a3_db[cell]),
            "sinr_db": float(snapshot.sinr_rlm_db[cell]),
            "qin_gap_db": float(snapshot.sinr_rlm_db[cell] - q_in_db),
            "qout_gap_db": float(snapshot.sinr_rlm_db[cell] - q_out_db),
        }

    return {
        "state_id": case_id,
        "branch": branch,
        "offset_ticks": int(offset_ticks),
        "offset_ms": int(offset_ticks * 10),
        "tic": int(snapshot.tic),
        "phase": snapshot.phase.name,
        "serving_cell": None if snapshot.serving_cell is None else int(snapshot.serving_cell),
        "protocol_target_cell": None if snapshot.target_cell is None else int(snapshot.target_cell),
        "n310_progress": float(snapshot.n310_progress),
        "n311_progress": float(snapshot.n311_progress),
        "t310_active": bool(snapshot.t310_active),
        "t310_progress": float(snapshot.t310_progress),
        "initial_serving": cell_values(initial_serving),
        "c1": cell_values(c1_cell),
        "selected": cell_values(selected_cell),
        "delivered_rate_mbps": float(delivered_rate_mbps),
        "best_rate_mbps": float(best_rate_mbps),
        **events,
    }


def replay_branch_series(
    *, case_id: str, engine: Any, builder: Any, requested_slot: int,
    horizon_ticks: int = HORIZON_TICKS,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Replay an immediate target action plus registered Always-C1 continuation."""
    if engine.decision_point is None or engine.trace is None:
        raise RuntimeError("Detailed replay requires a pending D63 decision.")
    trace = engine.trace
    memo = {id(trace): trace}
    branch = copy.deepcopy(engine, memo)
    branch_builder = copy.deepcopy(builder, memo)
    decision = branch.decision_point
    assert decision is not None
    anchor_tic = int(decision.tic)
    branch_builder.build(branch.snapshot)
    candidates = tuple(int(cell) for cell in branch_builder.candidate_cells if cell is not None)
    initial_serving = int(decision.serving_cell)
    c1_cell = int(branch_builder.candidate_cells[0])
    selected_cell = int(branch_builder.candidate_cells[requested_slot - 1])
    execution = _execute_requested_slot(
        branch,
        branch_builder,
        requested_slot=requested_slot,
        requested_delay_ticks=0,
        anchor_tic=anchor_tic,
    )
    series: list[dict[str, Any]] = []
    event_tics: dict[str, list[int]] = {
        "handover_started": [], "handover_succeeded": [],
        "handover_failed": [], "rlf_started": [], "ping_pong": [],
    }
    redecision_ticks = STAY_REDECISION_MS // branch.config.delta_t_ms
    for offset in range(1, horizon_ticks + 1):
        tick = branch.advance_one_tick()
        branch_builder.observe(tick)
        events = {
            "handover_started": bool(tick.handover_started),
            "handover_succeeded": bool(tick.handover_succeeded),
            "handover_failed": bool(tick.handover_failed),
            "rlf_started": bool(tick.rlf_started),
            "ping_pong": bool(tick.ping_pong_detected),
        }
        for name, happened in events.items():
            if happened:
                event_tics[name].append(int(tick.snapshot.tic))
        series.append(
            _radio_record(
                case_id=case_id,
                branch=f"C{requested_slot}",
                offset_ticks=offset,
                snapshot=tick.snapshot,
                initial_serving=initial_serving,
                c1_cell=c1_cell,
                selected_cell=selected_cell,
                q_in_db=branch.config.q_in_db,
                q_out_db=branch.config.q_out_db,
                delivered_rate_mbps=tick.delivered_rate_mbps,
                best_rate_mbps=tick.best_rate_mbps,
                events=events,
            )
        )
        if tick.terminated:
            break
        if branch.decision_point is not None:
            branch_builder.build(branch.snapshot)
            target = branch_builder.candidate_cells[0]
            if target is None:
                branch.stay(redecision_delay_ticks=redecision_ticks)
            else:
                branch.start_handover(int(target))
    completion_rows = [
        row for row in series if row["handover_succeeded"] or row["handover_failed"]
    ]
    return series, {
        "requested_slot": requested_slot,
        "candidate_cells_at_anchor": list(candidates),
        "initial_serving_cell": initial_serving,
        "c1_cell": c1_cell,
        "selected_cell": selected_cell,
        "execution": execution,
        "event_tics": event_tics,
        "first_ho_completion": completion_rows[0] if completion_rows else None,
    }


def _pre_anchor_series(
    *, case_id: str, engine: Any, serving: int, c1: int, selected: int
) -> list[dict[str, Any]]:
    trace = engine.trace
    decision = engine.decision_point
    if trace is None or decision is None:
        raise RuntimeError("Pre-anchor extraction requires a pending decision.")
    rows = []
    first = max(0, decision.tic - PRE_WINDOW_TICKS)
    for tic in range(first, decision.tic + 1):
        def values(cell: int) -> dict[str, float]:
            return {
                "rsrp_db": float(trace.rsrp_a3_db[cell, tic]),
                "sinr_db": float(trace.sinr_rlm_db[cell, tic]),
                "qin_gap_db": float(trace.sinr_rlm_db[cell, tic] - engine.config.q_in_db),
                "qout_gap_db": float(trace.sinr_rlm_db[cell, tic] - engine.config.q_out_db),
            }
        rows.append(
            {
                "state_id": case_id,
                "branch": "pre_anchor_causal",
                "offset_ticks": tic - decision.tic,
                "offset_ms": (tic - decision.tic) * engine.config.delta_t_ms,
                "tic": tic,
                "initial_serving": values(serving),
                "c1": values(c1),
                "selected": values(selected),
            }
        )
    return rows


def run_replay(
    *, bank_dir: Path, labels_dir: Path, probe_dir: Path, output_dir: Path
) -> dict[str, Any]:
    _prepare_destination(output_dir)
    bank_path = bank_dir / "state_bank.npz"
    manifest_path = bank_dir / "state_bank_manifest.json"
    labels_path = labels_dir / "target_labels.npz"
    scores_path = probe_dir / "probe_scores.npz"
    for path in (bank_path, manifest_path, labels_path, scores_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    bank = _load_npz(bank_path)
    labels = _load_npz(labels_path)
    score_data = _load_npz(scores_path)
    scores = score_data["target__mlp_small__score"]
    holdout = labels["fold"] == "holdout"
    holdout_labels = {name: values[holdout] for name, values in labels.items()}
    holdout_scores = scores[holdout]
    chosen = choose_actions(holdout_labels, holdout_scores, DIAGNOSTIC_THRESHOLD)
    selected = chosen[~holdout_labels["is_control"][chosen]]
    selected_ids = [str(holdout_labels["state_id"][index]) for index in selected]
    selected_slots = {
        str(holdout_labels["state_id"][index]): int(holdout_labels["planned_slot"][index])
        for index in selected
    }
    selected_useful = {
        str(holdout_labels["state_id"][index]): bool(holdout_labels["strict_useful"][index])
        for index in selected
    }
    bank_indices = [int(np.flatnonzero(bank["state_id"] == state_id)[0]) for state_id in selected_ids]
    filtered = {name: values[bank_indices] for name, values in bank.items()}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    d61 = Path(manifest["sources"]["d61"]["checkpoint"])
    d62 = Path(manifest["sources"]["d62"]["checkpoint"])
    sources = _source_specs(d61, d62)

    summaries = []
    all_series = []
    for index, engine, builder in replay_d63_anchors(manifest, filtered, sources):
        state_id = str(filtered["state_id"][index])
        slot = selected_slots[state_id]
        builder.build(engine.snapshot)
        serving = int(filtered["serving_cell"][index])
        c1 = int(filtered["candidate_cells"][index, 0])
        selected_cell = int(filtered["candidate_cells"][index, slot - 1])
        all_series.extend(
            _pre_anchor_series(
                case_id=state_id, engine=engine, serving=serving,
                c1=c1, selected=selected_cell,
            )
        )
        features = causal_radio_features(
            trace=engine.trace,
            serving_cell=serving,
            target_cell=selected_cell,
            anchor_tic=int(filtered["tic"][index]),
            q_in_db=engine.config.q_in_db,
            delta_t_ms=engine.config.delta_t_ms,
        )
        control_series, control_events = replay_branch_series(
            case_id=state_id, engine=engine, builder=builder, requested_slot=1
        )
        action_series, action_events = replay_branch_series(
            case_id=state_id, engine=engine, builder=builder, requested_slot=slot
        )
        all_series.extend(control_series)
        all_series.extend(action_series)
        control_outcome, _ = simulate_timing_target(
            engine, builder, delay_ticks=0, requested_slot=1,
            horizon_ticks=(HORIZON_TICKS,),
        )
        action_outcome, _ = simulate_timing_target(
            engine, builder, delay_ticks=0, requested_slot=slot,
            horizon_ticks=(HORIZON_TICKS,),
        )
        diagnostics = {
            int(record["slot"]): record
            for record in builder.prediction_diagnostics(engine.snapshot)
        }
        completion = action_events["first_ho_completion"]
        summaries.append(
            {
                "state_id": state_id,
                "label": "true_positive" if selected_useful[state_id] else "false_positive",
                "speed_kmh": int(filtered["speed_kmh"][index]),
                "trajectory_id": int(filtered["trajectory_id"][index]),
                "anchor_tic": int(filtered["tic"][index]),
                "serving_cell": serving,
                "candidate_cells": filtered["candidate_cells"][index].tolist(),
                "selected_slot": slot,
                "selected_cell": selected_cell,
                "causal_features": features,
                "existing_projection_diagnostic": diagnostics[slot],
                "control": {
                    "events": control_events,
                    "physical": control_outcome[HORIZON_TICKS]["physical"],
                },
                "selected_action": {
                    "events": action_events,
                    "physical": action_outcome[HORIZON_TICKS]["physical"],
                    "target_qin_gap_at_first_ho_completion_db": (
                        None if completion is None else completion["selected"]["qin_gap_db"]
                    ),
                },
            }
        )
    summaries.sort(key=lambda row: (row["label"], row["state_id"]))
    fp = [row for row in summaries if row["label"] == "false_positive"]
    tp = [row for row in summaries if row["label"] == "true_positive"]
    feature_name = "execution_safety_margin_v1_db"
    report = {
        "schema_name": SCHEMA,
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "analysis_only_no_refit": True,
        "validation_or_test_rows_read": 0,
        "diagnostic_threshold": DIAGNOSTIC_THRESHOLD,
        "cases": summaries,
        "contrast": {
            "false_positive_count": len(fp),
            "true_positive_reference_count": len(tp),
            "proposed_feature": feature_name,
            "definition": "target SINR RLM gap to Qin now minus target SINR standard deviation over causal 500 ms",
            "causal_at_decision": True,
            "false_positive_values_db": [row["causal_features"][feature_name] for row in fp],
            "true_positive_reference_values_db": [row["causal_features"][feature_name] for row in tp],
            "interpretation_limit": "Four high-score holdout cases diagnose a hypothesis; they do not validate a threshold or generalization.",
        },
        "inputs": {
            "state_bank": {"path": str(bank_path.resolve()), "sha256": _sha256_file(bank_path)},
            "manifest": {"path": str(manifest_path.resolve()), "sha256": _sha256_file(manifest_path)},
            "target_labels": {"path": str(labels_path.resolve()), "sha256": _sha256_file(labels_path)},
            "probe_scores": {"path": str(scores_path.resolve()), "sha256": _sha256_file(scores_path)},
        },
    }
    _json_write(output_dir / "case_replay_report.json", report)
    with (output_dir / "radio_series.jsonl").open("w", encoding="utf-8") as handle:
        for row in all_series:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    _json_write(
        output_dir / "manifest.json",
        {
            "schema_name": "d63_2_failure_case_replay_manifest_v1",
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
    report = run_replay(
        bank_dir=(root / args.bank_dir).resolve(),
        labels_dir=(root / args.labels_dir).resolve(),
        probe_dir=(root / args.probe_dir).resolve(),
        output_dir=(root / args.output_dir).resolve(),
    )
    contrast = report["contrast"]
    print(
        "D63-2 case replay complete: "
        f"{contrast['false_positive_count']} FP and "
        f"{contrast['true_positive_reference_count']} TP references; "
        f"proposed causal feature={contrast['proposed_feature']}."
    )
    return 0
