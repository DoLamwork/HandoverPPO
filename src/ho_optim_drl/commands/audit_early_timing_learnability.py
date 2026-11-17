"""T4 causal learnability audit for post-A3 HO_NOW versus WAIT_20."""

from __future__ import annotations

from argparse import Namespace
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch import nn

from ho_optim_drl.commands.audit_baseline_parity import _git_state, _sha256
from ho_optim_drl.commands.audit_early_timing_references import _metrics
from ho_optim_drl.commands.audit_early_timing_bank import (
    SOURCES,
    collect_source_anchors,
)
from ho_optim_drl.commands.audit_early_timing_routes import (
    _extended_aggregate,
    paired_bootstrap,
)
from ho_optim_drl.commands.audit_policy_learning import _average_precision
from ho_optim_drl.data import TracePair, load_radio_trace
from ho_optim_drl.data.project import load_protocol_config
from ho_optim_drl.protocol import EarlyTimingHandoverEngine, ShadowA3DecisionPoint


SCHEMA_NAME = "early_ho_timing_t4_learnability"
SCHEMA_VERSION = 1
FEATURE_NAMES = (
    "speed_kmh_div_150", "route_progress", "a3_elapsed_s",
    "serving_rsrp_db_div_140", "target_rsrp_db_div_140", "rsrp_delta_db_div_20",
    "serving_sinr_rate_db_div_30", "target_sinr_rate_db_div_30",
    "serving_rlm_margin_qout_div_20", "target_rlm_margin_qout_div_20",
    "serving_rsrp_trend_20ms_div_10", "target_rsrp_trend_20ms_div_10",
    "delta_rsrp_trend_20ms_div_10", "serving_rsrp_trend_100ms_div_10",
    "target_rsrp_trend_100ms_div_10", "delta_rsrp_trend_100ms_div_10",
    "serving_rsrp_std_100ms_div_10", "target_rsrp_std_100ms_div_10",
    "delta_rsrp_std_100ms_div_10", "target_minus_serving_sinr_div_30",
)
T4B_EXTRA_FEATURE_NAMES = (
    "n310_progress", "n311_progress", "t310_active", "t310_progress",
    "serving_residence_s_div_10", "time_since_last_ho_s_div_10",
    "target_is_previous", "recent_ho_count_div_4",
    "candidate_best_fraction_200ms", "candidate_best_fraction_500ms",
    "candidate_persistence_500ms", "delta_rsrp_trend_200ms_div_10",
    "delta_rsrp_trend_500ms_div_10", "delta_sinr_trend_200ms_div_20",
    "delta_sinr_trend_500ms_div_20", "delta_rsrp_std_500ms_div_10",
    "serving_sinr_min_200ms_div_30", "target_sinr_min_200ms_div_30",
    "serving_rlm_min_margin_qout_div_20", "target_rlm_min_margin_qout_div_20",
)
RISK_FIELDS = (
    "handover_failures", "ping_pong_count", "rlf_starts",
    "recovery_seconds", "interruption_seconds",
)
MODEL_NAMES = ("linear", "mlp_small")
EPS = 1e-12


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def causal_features(
    trace: Any, *, tic: int, serving: int, target: int,
    a3_start_tic: int, speed_kmh: int, q_out_db: float, dt_ms: int,
    snapshot: Any | None = None, schema: str = "t4a_causal_v1",
) -> np.ndarray:
    """Build post-A3 features using samples at or before ``tic`` only."""
    if not 0 <= tic < trace.num_samples:
        raise ValueError("Feature tic is outside the trace.")
    if serving == target:
        raise ValueError("Serving and target cells must differ.")
    rsrp_s = trace.rsrp_a3_db[serving]
    rsrp_t = trace.rsrp_a3_db[target]
    sinr_s = trace.sinr_rate_db[serving]
    sinr_t = trace.sinr_rate_db[target]
    rlm_s = trace.sinr_rlm_db[serving]
    rlm_t = trace.sinr_rlm_db[target]

    def trend(values: np.ndarray, lag: int) -> float:
        return float(values[tic] - values[max(0, tic - lag)])

    start = max(0, tic - 9)
    delta_window = rsrp_t[start : tic + 1] - rsrp_s[start : tic + 1]
    values = np.asarray((
        speed_kmh / 150.0,
        tic / max(1, trace.num_samples - 1),
        max(0, tic - a3_start_tic) * dt_ms / 1000.0,
        float(rsrp_s[tic]) / 140.0,
        float(rsrp_t[tic]) / 140.0,
        float(rsrp_t[tic] - rsrp_s[tic]) / 20.0,
        float(sinr_s[tic]) / 30.0,
        float(sinr_t[tic]) / 30.0,
        float(rlm_s[tic] - q_out_db) / 20.0,
        float(rlm_t[tic] - q_out_db) / 20.0,
        trend(rsrp_s, 2) / 10.0,
        trend(rsrp_t, 2) / 10.0,
        (trend(rsrp_t, 2) - trend(rsrp_s, 2)) / 10.0,
        trend(rsrp_s, 10) / 10.0,
        trend(rsrp_t, 10) / 10.0,
        (trend(rsrp_t, 10) - trend(rsrp_s, 10)) / 10.0,
        float(np.std(rsrp_s[start : tic + 1])) / 10.0,
        float(np.std(rsrp_t[start : tic + 1])) / 10.0,
        float(np.std(delta_window)) / 10.0,
        float(sinr_t[tic] - sinr_s[tic]) / 30.0,
    ), dtype=np.float32)
    if schema == "t4b_protocol_history_v1":
        if snapshot is None:
            raise ValueError("T4B features require a replayed protocol snapshot.")
        values = np.concatenate((values, _t4b_extra_features(
            trace, snapshot=snapshot, tic=tic, serving=serving,
            target=target, q_out_db=q_out_db, dt_ms=dt_ms,
        ))).astype(np.float32)
    elif schema != "t4a_causal_v1":
        raise ValueError(f"Unknown timing feature schema: {schema}")
    expected = feature_names(schema)
    if len(values) != len(expected) or not np.all(np.isfinite(values)):
        raise ValueError("Causal timing feature vector is invalid.")
    return values


def feature_names(schema: str) -> tuple[str, ...]:
    if schema == "t4a_causal_v1":
        return FEATURE_NAMES
    if schema == "t4b_protocol_history_v1":
        return (*FEATURE_NAMES, *T4B_EXTRA_FEATURE_NAMES)
    raise ValueError(f"Unknown timing feature schema: {schema}")


def _t4b_extra_features(
    trace: Any, *, snapshot: Any, tic: int, serving: int, target: int,
    q_out_db: float, dt_ms: int,
) -> np.ndarray:
    """Protocol/history additions; every window ends at the decision tick."""
    if snapshot.tic != tic or snapshot.decision_point is None:
        raise ValueError("T4B snapshot must be the matching pending decision.")
    rsrp_s = trace.rsrp_a3_db[serving]
    rsrp_t = trace.rsrp_a3_db[target]
    sinr_s = trace.sinr_rate_db[serving]
    sinr_t = trace.sinr_rate_db[target]
    rlm_s = trace.sinr_rlm_db[serving]
    rlm_t = trace.sinr_rlm_db[target]

    def start_for(milliseconds: int) -> int:
        ticks = max(1, milliseconds // dt_ms)
        return max(0, tic - ticks)

    start_200 = start_for(200)
    start_500 = start_for(500)
    rsrp = trace.rsrp_a3_db[:, start_500 : tic + 1].copy()
    rsrp[serving, :] = -np.inf
    winners = np.argmax(rsrp, axis=0)
    persistence = 0
    for winner in winners[::-1]:
        if int(winner) != target:
            break
        persistence += 1
    delta_rsrp = rsrp_t - rsrp_s
    delta_sinr = sinr_t - sinr_s
    return np.asarray((
        float(snapshot.n310_progress), float(snapshot.n311_progress),
        float(snapshot.t310_active), float(snapshot.t310_progress),
        min(snapshot.serving_residence_ticks * dt_ms / 10000.0, 1.0),
        min(snapshot.time_since_last_ho_ticks * dt_ms / 10000.0, 1.0),
        float(snapshot.previous_cell == target),
        min(float(snapshot.recent_ho_count) / 4.0, 1.0),
        float(np.mean(winners[-(tic - start_200 + 1):] == target)),
        float(np.mean(winners == target)),
        persistence / max(1, len(winners)),
        float(delta_rsrp[tic] - delta_rsrp[start_200]) / 10.0,
        float(delta_rsrp[tic] - delta_rsrp[start_500]) / 10.0,
        float(delta_sinr[tic] - delta_sinr[start_200]) / 20.0,
        float(delta_sinr[tic] - delta_sinr[start_500]) / 20.0,
        float(np.std(delta_rsrp[start_500 : tic + 1])) / 10.0,
        float(np.min(sinr_s[start_200 : tic + 1])) / 30.0,
        float(np.min(sinr_t[start_200 : tic + 1])) / 30.0,
        float(np.min(rlm_s[start_200 : tic + 1]) - q_out_db) / 20.0,
        float(np.min(rlm_t[start_200 : tic + 1]) - q_out_db) / 20.0,
    ), dtype=np.float32)


def stratified_route_folds(
    routes: list[tuple[int, int]], *, seed: int,
) -> dict[tuple[int, int], str]:
    """Assign 60/20/20 route folds independently within each speed."""
    by_speed: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for route in routes:
        by_speed[int(route[0])].append(route)
    rng = np.random.default_rng(seed)
    result: dict[tuple[int, int], str] = {}
    for speed, group in sorted(by_speed.items()):
        ordered = sorted(set(group))
        shuffled = [ordered[int(index)] for index in rng.permutation(len(ordered))]
        fit_end = int(round(0.60 * len(shuffled)))
        tuning_end = fit_end + int(round(0.20 * len(shuffled)))
        for index, route in enumerate(shuffled):
            result[route] = "fit" if index < fit_end else "tuning" if index < tuning_end else "holdout"
    return result


def timing_label(now: dict[str, Any], wait: dict[str, Any]) -> dict[str, Any]:
    """Return WAIT-minus-HO outcomes without collapsing gain and risk."""
    risks = {name: float(wait[name]) - float(now[name]) for name in RISK_FIELDS}
    gain = float(wait["delivered_mbit"]) - float(now["delivered_mbit"])
    rr = float(wait["relative_rate_percent"]) - float(now["relative_rate_percent"])
    safe = all(value <= EPS for value in risks.values())
    return {
        "wait_useful": bool(gain > EPS and safe),
        "delta_delivered_mbit": gain,
        "delta_relative_rate_percent": rr,
        "risk_deltas": risks,
        "wait_safe": safe,
    }


class _Probe(nn.Module):
    def __init__(self, size: int, architecture: str) -> None:
        super().__init__()
        output = 2 + len(RISK_FIELDS)
        if architecture == "linear":
            self.network = nn.Linear(size, output)
        elif architecture == "mlp_small":
            self.network = nn.Sequential(nn.Linear(size, 32), nn.Tanh(), nn.Linear(32, output))
        else:
            raise ValueError(architecture)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value)


def _fit_probe(
    features: np.ndarray, labels: np.ndarray, continuous: np.ndarray,
    fit_mask: np.ndarray, *, architecture: str, seed: int,
) -> tuple[_Probe, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    torch.manual_seed(seed); torch.set_num_threads(1)
    x = torch.as_tensor(features, dtype=torch.float32)
    fit = torch.as_tensor(fit_mask, dtype=torch.bool)
    mean = x[fit].mean(0); std = x[fit].std(0, unbiased=False).clamp_min(1e-6)
    scaled = (x - mean) / std
    y = torch.as_tensor(labels, dtype=torch.float32)
    cont = torch.as_tensor(continuous, dtype=torch.float32)
    cmean = cont[fit].mean(0); cstd = cont[fit].std(0, unbiased=False).clamp_min(1e-6)
    positives = int(y[fit].sum()); negatives = int(fit.sum()) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("Fit fold requires useful and non-useful WAIT labels.")
    model = _Probe(features.shape[1], architecture)
    lr, epochs = ((0.02, 350) if architecture == "linear" else (0.005, 500))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    binary = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(negatives / positives))
    for _ in range(epochs):
        output = model(scaled[fit])
        loss = binary(output[:, 0], y[fit]) + 0.25 * torch.mean((output[:, 1:] - (cont[fit] - cmean) / cstd) ** 2)
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
    with torch.no_grad():
        output = model(scaled)
        scores = torch.sigmoid(output[:, 0]).cpu().numpy()
    return model, scores, mean.numpy(), std.numpy(), {
        "architecture": architecture, "seed": seed, "epochs": epochs,
        "learning_rate": lr, "positive_class_weight": negatives / positives,
    }


def _selection(
    labels: list[dict[str, Any]], choose_wait: np.ndarray,
) -> dict[str, Any]:
    gain = np.asarray([row["delta_delivered_mbit"] for row in labels])
    rr = np.asarray([row["delta_relative_rate_percent"] for row in labels])
    risks = {name: np.asarray([row["risk_deltas"][name] for row in labels]) for name in RISK_FIELDS}
    return {
        "states": len(labels), "wait_actions": int(np.sum(choose_wait)),
        "mean_delta_delivered_mbit": float(np.mean(np.where(choose_wait, gain, 0.0))),
        "total_delta_delivered_mbit": float(np.sum(np.where(choose_wait, gain, 0.0))),
        "mean_delta_relative_rate_percent": float(np.mean(np.where(choose_wait, rr, 0.0))),
        "risk_deltas": {name: float(np.mean(np.where(choose_wait, values, 0.0))) for name, values in risks.items()},
        "reliability_not_worse": all(float(np.mean(np.where(choose_wait, values, 0.0))) <= EPS for values in risks.values()),
        "useful_wait_actions": int(np.sum(choose_wait & np.asarray([row["wait_useful"] for row in labels], dtype=bool))),
        "false_wait_actions": int(np.sum(choose_wait & ~np.asarray([row["wait_useful"] for row in labels], dtype=bool))),
    }


def _lock_threshold(labels: list[dict[str, Any]], scores: np.ndarray) -> tuple[float, dict[str, Any]]:
    records = []
    for threshold in np.concatenate((np.linspace(0.05, 0.95, 19), [1.000001])):
        records.append({"threshold": float(threshold), **_selection(labels, scores >= threshold)})
    feasible = [row for row in records if row["reliability_not_worse"]]
    chosen = max(feasible, key=lambda row: (row["mean_delta_delivered_mbit"], -row["false_wait_actions"], -row["wait_actions"], row["threshold"]))
    return float(chosen["threshold"]), {"selected": chosen, "candidates": records}


def _classification(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    predicted = scores >= threshold
    tp = int(np.sum(labels & predicted)); fp = int(np.sum(~labels & predicted)); fn = int(np.sum(labels & ~predicted))
    return {
        "rows": len(labels), "positive_rows": int(np.sum(labels)),
        "prevalence": float(np.mean(labels)),
        "average_precision": _average_precision(labels.astype(np.int8), scores),
        "threshold": threshold, "predicted_wait_rows": int(np.sum(predicted)),
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "brier_score": float(np.mean((scores - labels.astype(float)) ** 2)),
    }


def _classification_groups(
    samples: list[dict[str, Any]], labels: np.ndarray, scores: np.ndarray,
    mask: np.ndarray, threshold: float,
) -> dict[str, Any]:
    groups = {"overall": _classification(labels[mask], scores[mask], threshold)}
    for field in ("speed_kmh", "source"):
        values = sorted({row[field] for row, keep in zip(samples, mask) if keep})
        for value in values:
            group = mask & np.asarray([row[field] == value for row in samples])
            groups[f"{field}:{value}"] = _classification(labels[group], scores[group], threshold)
    return groups


def _route_bootstrap_gain(
    samples: list[dict[str, Any]], choose_wait: np.ndarray, *, seed: int,
) -> dict[str, Any]:
    grouped: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, row in enumerate(samples): grouped[(row["speed_kmh"], row["trajectory_id"])].append(index)
    route_values = []
    for indices in grouped.values():
        values = [samples[i]["label"]["delta_delivered_mbit"] if choose_wait[i] else 0.0 for i in indices]
        route_values.append(float(np.mean(values)))
    array = np.asarray(route_values); rng = np.random.default_rng(seed)
    boot = [float(np.mean(array[rng.integers(0, len(array), len(array))])) for _ in range(2000)]
    return {"routes": len(array), "mean": float(np.mean(array)), "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))], "resamples": 2000, "seed": seed}


def _predict(model: _Probe, feature: np.ndarray, mean: np.ndarray, std: np.ndarray) -> float:
    with torch.no_grad():
        value = torch.as_tensor((feature - mean) / std, dtype=torch.float32)[None, :]
        return float(torch.sigmoid(model(value)[0, 0]).item())


def _run_online(
    trace: Any, config: Any, speed: int,
    selector: Callable[[np.ndarray], bool],
    *, feature_schema: str,
) -> tuple[dict[str, Any], dict[str, int]]:
    engine = EarlyTimingHandoverEngine(config, enable_early_decisions=False, enable_shadow_decisions=True)
    engine.reset(trace); decisions = waits = 0
    while True:
        if engine.decision_point is not None:
            decision = engine.decision_point
            if not isinstance(decision, ShadowA3DecisionPoint):
                raise RuntimeError("T4 post-A3 policy received an early decision.")
            feature = causal_features(
                trace, tic=decision.tic, serving=decision.serving_cell,
                target=decision.target_cell,
                a3_start_tic=decision.a3_start_tic, speed_kmh=speed,
                q_out_db=config.q_out_db, dt_ms=config.delta_t_ms,
                snapshot=engine.snapshot, schema=feature_schema,
            )
            decisions += 1
            if selector(feature):
                waits += 1; engine.wait_for_redecision(delay_ticks=20 // config.delta_t_ms)
            else:
                engine.start_handover()
        if engine.advance_one_tick().terminated: break
    return _metrics(engine, trace), {"decisions": decisions, "wait_actions": waits}


def _rule_wait(feature: np.ndarray, rule: str, *, names: tuple[str, ...]) -> bool:
    lookup = dict(zip(names, feature))
    if rule == "target_sinr_worse":
        return lookup["target_minus_serving_sinr_div_30"] < 0.0
    if rule == "rsrp_delta_not_improving":
        return lookup["delta_rsrp_trend_100ms_div_10"] <= 0.0
    if rule == "never_wait":
        return False
    raise ValueError(rule)


def _load_archived_routes(t3_dir: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = defaultdict(dict)
    for row in _read_jsonl(t3_dir / "route_outcomes.jsonl"):
        if row["policy"] in ("b0", "rpost"):
            result[str(row["trace_id"])][str(row["policy"])] = row["metrics"]
    return result


def run_audit(
    *, root: Path, data_dir: Path, t2_dir: Path, t3_dir: Path,
    output_dir: Path, split_seed: int, model_seed: int,
    feature_schema: str = "t4a_causal_v1",
    control_report_path: Path | None = None,
) -> dict[str, Any]:
    if output_dir.exists(): raise FileExistsError(f"Refusing to overwrite T4 output: {output_dir}")
    output_dir.mkdir(parents=True)
    split = _read_json(t2_dir / "route_split.json")
    if split.get("split") != "train" or split.get("validation_or_test_rows_read") != 0:
        raise ValueError("T4 accepts the frozen TRAIN route split only.")
    raw_anchors = [row for row in _read_jsonl(t2_dir / "anchors.jsonl") if row["category"] == "post_ttt"]
    # Identical physical states can be sampled by two source policies.  Keep one
    # deterministic copy so repeated states do not receive extra training weight.
    anchors_by_fingerprint: dict[str, dict[str, Any]] = {}
    for row in sorted(raw_anchors, key=lambda item: (str(item["physical_state_fingerprint"]), str(item["state_id"]))):
        anchors_by_fingerprint.setdefault(str(row["physical_state_fingerprint"]), row)
    anchors = list(anchors_by_fingerprint.values())
    outcomes = _read_jsonl(t2_dir / "local_outcomes.jsonl")
    outcome_map = {(row["state_id"], float(row["horizon_seconds"]), row["branch"], row["requested_delay_ms"]): row["physical"] for row in outcomes}
    routes = [(int(row["speed_kmh"]), int(row["trajectory_id"])) for row in split["selected_routes"]]
    folds = stratified_route_folds(routes, seed=split_seed)
    route_records = {(int(row["speed_kmh"]), int(row["trajectory_id"])): row for row in split["selected_routes"]}
    config = load_protocol_config(str(data_dir)); trace_cache: dict[tuple[int, int], Any] = {}
    names = feature_names(feature_schema)
    snapshot_by_fingerprint: dict[str, Any] = {}
    if feature_schema == "t4b_protocol_history_v1":
        max_anchors = int(_read_json(t2_dir / "bank_report.json")["selection"]["max_anchors_per_source_route"])
        for route_index, route in enumerate(sorted(route_records), 1):
            record = route_records[route]
            pair = TracePair(route[0], route[1], str(record["rsrp_file"]), str(record["sinr_file"]))
            trace = load_radio_trace(config, str(data_dir), pair.rsrp_file, pair.sinr_file, speed_kmh=route[0])
            trace_cache[route] = trace
            for source in SOURCES:
                replayed, _ = collect_source_anchors(
                    config, trace, pair, source=source,
                    sample_seed=int(split["sample_seed"]), max_anchors=max_anchors,
                )
                for item in replayed:
                    fingerprint = str(item["record"]["physical_state_fingerprint"])
                    snapshot_by_fingerprint.setdefault(fingerprint, item["engine"].snapshot)
            if route_index % 10 == 0:
                print(f"[T4B] replayed protocol state {route_index}/{len(route_records)} routes", flush=True)
    samples = []
    for anchor in anchors:
        route = (int(anchor["speed_kmh"]), int(anchor["trajectory_id"]))
        if route not in trace_cache:
            record = route_records[route]
            trace_cache[route] = load_radio_trace(config, str(data_dir), record["rsrp_file"], record["sinr_file"], speed_kmh=route[0])
        trace = trace_cache[route]
        now = outcome_map[(anchor["state_id"], 1.0, "timing_schedule", 0)]
        wait = outcome_map[(anchor["state_id"], 1.0, "timing_schedule", 20)]
        samples.append({
            "state_id": anchor["state_id"], "source": anchor["source"],
            "speed_kmh": route[0], "trajectory_id": route[1], "fold": folds[route],
            "feature": causal_features(
                trace, tic=int(anchor["tic"]),
                serving=int(anchor["serving_cell"]),
                target=int(anchor["target_cell"]),
                a3_start_tic=int(anchor["decision"]["a3_start_tic"]),
                speed_kmh=route[0], q_out_db=config.q_out_db,
                dt_ms=config.delta_t_ms,
                snapshot=snapshot_by_fingerprint.get(str(anchor["physical_state_fingerprint"])),
                schema=feature_schema,
            ),
            "label": timing_label(now, wait),
        })
    features = np.stack([row["feature"] for row in samples])
    labels = np.asarray([row["label"]["wait_useful"] for row in samples], dtype=bool)
    continuous = np.asarray([[row["label"]["delta_delivered_mbit"], *(row["label"]["risk_deltas"][name] for name in RISK_FIELDS)] for row in samples], dtype=np.float32)
    masks = {fold: np.asarray([row["fold"] == fold for row in samples]) for fold in ("fit", "tuning", "holdout")}
    model_reports = {}; models: dict[str, tuple[_Probe, np.ndarray, np.ndarray, float, np.ndarray]] = {}
    model_arrays: dict[str, np.ndarray] = {}
    for index, architecture in enumerate(MODEL_NAMES):
        model, scores, mean, std, fit_config = _fit_probe(features, labels, continuous, masks["fit"], architecture=architecture, seed=model_seed + index)
        tuning_rows = [row["label"] for row, keep in zip(samples, masks["tuning"]) if keep]
        threshold, threshold_report = _lock_threshold(tuning_rows, scores[masks["tuning"]])
        holdout_rows = [row["label"] for row, keep in zip(samples, masks["holdout"]) if keep]
        holdout_choose = scores[masks["holdout"]] >= threshold
        holdout_samples = [row for row, keep in zip(samples, masks["holdout"]) if keep]
        model_reports[architecture] = {
            "fit_config": fit_config, "threshold_selection_on_tuning": threshold_report,
            "holdout_classification": _classification_groups(samples, labels, scores, masks["holdout"], threshold),
            "holdout_selection": _selection(holdout_rows, holdout_choose),
            "holdout_route_bootstrap_gain": _route_bootstrap_gain(holdout_samples, holdout_choose, seed=model_seed + 100 + index),
        }
        models[architecture] = (model, mean, std, threshold, scores)
        model_arrays[f"{architecture}__feature_mean"] = mean
        model_arrays[f"{architecture}__feature_std"] = std
        model_arrays[f"{architecture}__threshold"] = np.asarray([threshold], dtype=np.float32)
        for name, tensor in model.state_dict().items(): model_arrays[f"{architecture}__{name}"] = tensor.detach().cpu().numpy()

    rules = ("never_wait", "target_sinr_worse", "rsrp_delta_not_improving")
    tuning_labels = [row["label"] for row, keep in zip(samples, masks["tuning"]) if keep]
    tuning_features = features[masks["tuning"]]
    rule_reports = {rule: _selection(tuning_labels, np.asarray([_rule_wait(feature, rule, names=names) for feature in tuning_features])) for rule in rules}
    feasible_rules = [rule for rule in rules if rule_reports[rule]["reliability_not_worse"]]
    selected_rule = max(feasible_rules, key=lambda rule: (rule_reports[rule]["mean_delta_delivered_mbit"], -rule_reports[rule]["false_wait_actions"], -rule_reports[rule]["wait_actions"]))

    archived = _load_archived_routes(t3_dir); full_rows = []
    holdout_routes = sorted(route for route, fold in folds.items() if fold == "holdout")
    for route_index, route in enumerate(holdout_routes, 1):
        record = route_records[route]
        trace = trace_cache.get(route) or load_radio_trace(config, str(data_dir), record["rsrp_file"], record["sinr_file"], speed_kmh=route[0])
        trace_id = f"{route[0]}:{route[1]}"
        row: dict[str, Any] = {"trace_id": trace_id, "speed_kmh": route[0], "trajectory_id": route[1], "b0": archived[trace_id]["b0"], "rpost": archived[trace_id]["rpost"]}
        rule_metrics, rule_actions = _run_online(
            trace, config, route[0],
            lambda feature, name=selected_rule: _rule_wait(feature, name, names=names),
            feature_schema=feature_schema,
        )
        row["simple_rule"] = rule_metrics; row["simple_rule_actions"] = rule_actions
        for architecture in MODEL_NAMES:
            model, mean, std, threshold, _ = models[architecture]
            metrics, actions = _run_online(
                trace, config, route[0],
                lambda feature, m=model, mu=mean, sigma=std, cut=threshold: _predict(m, feature, mu, sigma) >= cut,
                feature_schema=feature_schema,
            )
            row[architecture] = metrics; row[f"{architecture}_actions"] = actions
        full_rows.append(row); print(f"[T4] full-route {route_index}/{len(holdout_routes)} {trace_id}", flush=True)

    policies = ("b0", "rpost", "simple_rule", *MODEL_NAMES)
    aggregate = {policy: _extended_aggregate(full_rows, policy, config.delta_t_ms) for policy in policies}
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in full_rows: grouped[row["speed_kmh"]].append(row)
    per_speed = {str(speed): {policy: _extended_aggregate(group, policy, config.delta_t_ms) for policy in policies} for speed, group in sorted(grouped.items())}
    comparisons = {
        "mlp_minus_b0": paired_bootstrap(full_rows, left="b0", right="mlp_small", dt_ms=config.delta_t_ms, resamples=2000, seed=29),
        "mlp_minus_rule": paired_bootstrap(full_rows, left="simple_rule", right="mlp_small", dt_ms=config.delta_t_ms, resamples=2000, seed=30),
        "linear_minus_b0": paired_bootstrap(full_rows, left="b0", right="linear", dt_ms=config.delta_t_ms, resamples=2000, seed=31),
        "rule_minus_b0": paired_bootstrap(full_rows, left="b0", right="simple_rule", dt_ms=config.delta_t_ms, resamples=2000, seed=32),
    }
    primary = model_reports["mlp_small"]
    primary_classification = primary["holdout_classification"]["overall"]
    local_gate = bool(primary_classification["average_precision"] >= primary_classification["prevalence"] + 0.05 and primary["holdout_selection"]["reliability_not_worse"] and primary["holdout_route_bootstrap_gain"]["ci95"][0] > 0)
    delta_b0 = comparisons["mlp_minus_b0"]; delta_rule = comparisons["mlp_minus_rule"]
    physical_gate = bool(delta_b0["ci95"]["mean_route_relative_rate_percent"][0] > 0 and all(delta_b0["ci95"][name][1] <= 0 for name in ("handover_failure_percent", "ping_pong_percent", "rlf_time_percent")))
    rate_retention = 100 * aggregate["mlp_small"]["mean_route_estimated_rate_mbps"] / aggregate["b0"]["mean_route_estimated_rate_mbps"]
    rr_retention = 100 * aggregate["mlp_small"]["mean_route_relative_rate_percent"] / aggregate["b0"]["mean_route_relative_rate_percent"]
    qos_gate = bool(rate_retention >= 98 and rr_retention >= 98 and all(100 * item["mlp_small"]["mean_route_estimated_rate_mbps"] / item["b0"]["mean_route_estimated_rate_mbps"] >= 98 and 100 * item["mlp_small"]["mean_route_relative_rate_percent"] / item["b0"]["mean_route_relative_rate_percent"] >= 98 for item in per_speed.values()))
    rule_gate = bool(delta_rule["ci95"]["mean_route_relative_rate_percent"][0] > 0 and all(delta_rule["ci95"][name][1] <= 0 for name in ("handover_failure_percent", "ping_pong_percent", "rlf_time_percent")))
    gates = {"g_local_learnability": local_gate, "g_full_route_physical": physical_gate, "g_qos_guard": qos_gate, "g_beats_simple_rule": rule_gate, "overall_t4": "PASS" if local_gate and physical_gate and qos_gate and rule_gate else "NOT_PASSED"}

    np.savez_compressed(output_dir / "frozen_models.npz", **model_arrays)
    with (output_dir / "post_ttt_samples.jsonl").open("w", encoding="utf-8") as stream:
        for row in samples:
            serial = dict(row); serial["feature"] = row["feature"].tolist(); stream.write(json.dumps(serial, allow_nan=False) + "\n")
    with (output_dir / "holdout_route_outcomes.jsonl").open("w", encoding="utf-8") as stream:
        for row in full_rows: stream.write(json.dumps(row, allow_nan=False) + "\n")
    report = {
        "schema_name": SCHEMA_NAME, "schema_version": SCHEMA_VERSION, "status": "COMPLETE",
        "scope": "train_route_fit_tuning_holdout", "validation_or_test_rows_read": 0,
        "feature_contract": {"schema": feature_schema, "names": list(names), "causal": True, "future_features": [], "decision_scope": "post_a3_shadow_maturity_or_redecision_only", "target": "fixed_current_C_star"},
        "route_split": {fold: [f"{speed}:{route}" for (speed, route), value in sorted(folds.items()) if value == fold] for fold in ("fit", "tuning", "holdout")},
        "sample_counts": {"raw_post_ttt": len(raw_anchors), "duplicate_physical_states_removed": len(raw_anchors) - len(anchors), "total": len(samples), **{fold: int(np.sum(mask)) for fold, mask in masks.items()}},
        "positive_counts": {fold: int(np.sum(labels[mask])) for fold, mask in masks.items()},
        "models": model_reports, "simple_rules_tuning": rule_reports, "selected_simple_rule": selected_rule,
        "full_route_holdout": {"routes": len(full_rows), "aggregate": aggregate, "per_speed": per_speed, "paired_bootstrap": comparisons},
        "gates": gates,
        "interpretation": "T4 is TRAIN-development causal learnability evidence, not a PPO result or unseen test result.",
    }
    if control_report_path is not None:
        control = _read_json(control_report_path)
        control_classification = control["models"]["mlp_small"]["holdout_classification"]
        if "overall" in control_classification:
            control_classification = control_classification["overall"]
        report["observation_ablation_vs_t4a"] = {
            "control_report": str(control_report_path.resolve()),
            "control_sha256": _sha256(control_report_path),
            "feature_count_t4a": len(control["feature_contract"]["names"]),
            "feature_count_t4b": len(names),
            "holdout_ap_delta": primary_classification["average_precision"] - control_classification["average_precision"],
            "full_route_rr_delta_points": aggregate["mlp_small"]["mean_route_relative_rate_percent"] - control["full_route_holdout"]["aggregate"]["mlp_small"]["mean_route_relative_rate_percent"],
            "t4a_overall": control["gates"]["overall_t4"],
            "t4b_overall": gates["overall_t4"],
        }
    _write_json(output_dir / "t4_report.json", report); _write_json(output_dir / "decision_gate.json", gates)
    manifest = {
        "schema_name": SCHEMA_NAME, "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "validation_or_test_rows_read": 0,
        "inputs": {"t2_dir": str(t2_dir.resolve()), "t3_dir": str(t3_dir.resolve()), "t2_route_split_sha256": _sha256(t2_dir / "route_split.json"), "t2_anchors_sha256": _sha256(t2_dir / "anchors.jsonl"), "t2_outcomes_sha256": _sha256(t2_dir / "local_outcomes.jsonl"), "t3_outcomes_sha256": _sha256(t3_dir / "route_outcomes.jsonl"), "control_report": (None if control_report_path is None else {"path": str(control_report_path.resolve()), "sha256": _sha256(control_report_path)})},
        "feature_schema": feature_schema,
        "split_seed": split_seed, "model_seed": model_seed, "protocol_config": asdict(config), "git": _git_state(root),
        "source": {"path": str(Path(__file__).resolve().relative_to(root)), "sha256": _sha256(Path(__file__).resolve())},
    }
    _write_json(output_dir / "run_manifest.json", manifest)
    classification = model_reports["mlp_small"]["holdout_classification"]["overall"]
    local = model_reports["mlp_small"]["holdout_selection"]
    aggregate_mlp = aggregate["mlp_small"]
    title = (
        "T4B — observation ablation hậu-A3"
        if feature_schema == "t4b_protocol_history_v1"
        else "T4 — causal learnability hậu-A3"
    )
    lines = [
        f"# {title}",
        "",
        f"- Kết quả tổng: **{gates['overall_t4']}**",
        f"- Local learnability: **{'PASS' if local_gate else 'FAIL'}**",
        f"- Full-route physical: **{'PASS' if physical_gate else 'FAIL'}**",
        f"- QoS guard: **{'PASS' if qos_gate else 'FAIL'}**",
        f"- Vượt simple rule: **{'PASS' if rule_gate else 'FAIL'}**",
        "",
        "## Dữ liệu và probe",
        "",
        f"- {len(raw_anchors)} post-TTT anchors; loại {len(raw_anchors) - len(anchors)} fingerprint vật lý trùng; còn {len(samples)} state.",
        f"- MLP holdout AP {classification['average_precision']:.4f}; prevalence {classification['prevalence']:.4f}.",
        f"- MLP chọn WAIT ở {local['wait_actions']}/{local['states']} state; local gain {local['mean_delta_delivered_mbit']:+.6f} Mbit/state.",
        f"- Simple rule khóa trên tuning: `{selected_rule}`.",
    ]
    if "observation_ablation_vs_t4a" in report:
        ablation = report["observation_ablation_vs_t4a"]
        lines.extend([
            "",
            "## Đối chứng observation T4A → T4B",
            "",
            f"- Số feature: {ablation['feature_count_t4a']} → {ablation['feature_count_t4b']}.",
            f"- Holdout AP thay đổi {ablation['holdout_ap_delta']:+.4f}.",
            f"- Full-route RR thay đổi {ablation['full_route_rr_delta_points']:+.4f} điểm phần trăm.",
            f"- Gate tổng: {ablation['t4a_overall']} → {ablation['t4b_overall']}.",
        ])
    lines.extend([
        "",
        "## Full-route TRAIN-holdout",
        "",
        f"- B0 RR {aggregate['b0']['mean_route_relative_rate_percent']:.4f}%; Rpost RR {aggregate['rpost']['mean_route_relative_rate_percent']:.4f}%.",
        f"- Rule RR {aggregate['simple_rule']['mean_route_relative_rate_percent']:.4f}%; MLP RR {aggregate_mlp['mean_route_relative_rate_percent']:.4f}%.",
        f"- MLP HOF {aggregate_mlp['handover_failure_percent']:.4f}%, PP {aggregate_mlp['ping_pong_percent']:.4f}%, RLF-time {aggregate_mlp['rlf_time_percent']:.4f}%.",
        "",
        "## Quyết định",
        "",
        "Không mở PPO nếu T4 không PASS. Rpost vẫn chứng minh có dư địa timing, nhưng probe nhân quả phải vượt rule đơn giản và giữ rủi ro trước khi học tăng cường có ý nghĩa.",
    ])
    (output_dir / "report_vi.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main(root_path: str, args: Namespace) -> int:
    root = Path(root_path).resolve()
    def resolve(value: str) -> Path:
        path = Path(value); return (path if path.is_absolute() else root / path).resolve()
    control_value = getattr(args, "control_report", None)
    report = run_audit(root=root, data_dir=resolve(args.data_dir), t2_dir=resolve(args.t2_dir), t3_dir=resolve(args.t3_dir), output_dir=resolve(args.output_dir), split_seed=int(args.route_split_seed), model_seed=int(args.model_seed), feature_schema=str(getattr(args, "feature_schema", "t4a_causal_v1")), control_report_path=(None if control_value is None else resolve(control_value)))
    print(json.dumps({"status": report["status"], "gates": report["gates"]}, indent=2), flush=True)
    return 0
