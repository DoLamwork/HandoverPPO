"""TRAIN-only same-state audit for divergent T6A timing policies.

The audit replays two frozen timing-only checkpoints on the same locked TRAIN
routes, forms the union of policy-visited states, scores both policies on every
state, and summarizes the PPO learning diagnostics already emitted at train
time.  Validation, test, and zero-shot traces are deliberately out of scope.
"""

from __future__ import annotations

from argparse import Namespace
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Iterable, Sequence

import numpy as np
import torch

from ho_optim_drl.agents.ppo import SMDPPPO
from ho_optim_drl.commands.audit_baseline_parity import _git_state, _sha256
from ho_optim_drl.commands.audit_early_timing_routes import (
    _extended_aggregate,
    _load_route_split,
    paired_bootstrap,
)
from ho_optim_drl.commands.audit_t5a_benefit_sources import (
    _filter_pairs_per_speed,
    _objective_environment,
)
from ho_optim_drl.commands.audit_b1_lower_threshold import _route_metrics
from ho_optim_drl.data import TracePair
from ho_optim_drl.data.project import load_protocol_config
from ho_optim_drl.envs import HandoverSMDPEnv
from ho_optim_drl.smdp import HandoverAction


SCHEMA_NAME = "t6a_seed_divergence_train_audit"
SCHEMA_VERSION = 1
POLICY_LABELS = ("seed17", "seed29")


def _summary(values: Iterable[float]) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {"count": 0, "mean": None, "std": None, "p10": None,
                "p50": None, "p90": None, "min": None, "max": None}
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "p10": float(np.percentile(array, 10)),
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def comparison_summary(
    p_ho_seed17: np.ndarray,
    p_ho_seed29: np.ndarray,
    indices: np.ndarray | Sequence[int] | None = None,
) -> dict[str, Any]:
    """Summarize two binary policies on exactly the same state rows."""
    first = np.asarray(p_ho_seed17, dtype=np.float64)
    second = np.asarray(p_ho_seed29, dtype=np.float64)
    if first.shape != second.shape or first.ndim != 1:
        raise ValueError("Same-state probability vectors must be equal 1-D arrays.")
    selected = (
        np.arange(first.size, dtype=np.int64)
        if indices is None
        else np.asarray(indices, dtype=np.int64)
    )
    a = first[selected]
    b = second[selected]
    action_a = a > 0.5
    action_b = b > 0.5
    disagreement = action_a != action_b
    delta = b - a
    return {
        "states": int(selected.size),
        "seed17": {
            "p_ho": _summary(a),
            "deterministic_ho_percent": float(100.0 * np.mean(action_a))
            if a.size else None,
            "near_tie_percent_abs_margin_le_0_02": float(
                100.0 * np.mean(np.abs(2.0 * a - 1.0) <= 0.02)
            ) if a.size else None,
            "near_tie_percent_abs_margin_le_0_05": float(
                100.0 * np.mean(np.abs(2.0 * a - 1.0) <= 0.05)
            ) if a.size else None,
            "near_tie_percent_abs_margin_le_0_10": float(
                100.0 * np.mean(np.abs(2.0 * a - 1.0) <= 0.10)
            ) if a.size else None,
        },
        "seed29": {
            "p_ho": _summary(b),
            "deterministic_ho_percent": float(100.0 * np.mean(action_b))
            if b.size else None,
            "near_tie_percent_abs_margin_le_0_02": float(
                100.0 * np.mean(np.abs(2.0 * b - 1.0) <= 0.02)
            ) if b.size else None,
            "near_tie_percent_abs_margin_le_0_05": float(
                100.0 * np.mean(np.abs(2.0 * b - 1.0) <= 0.05)
            ) if b.size else None,
            "near_tie_percent_abs_margin_le_0_10": float(
                100.0 * np.mean(np.abs(2.0 * b - 1.0) <= 0.10)
            ) if b.size else None,
        },
        "p_ho_seed29_minus_seed17": _summary(delta),
        "deterministic_disagreement_percent": float(
            100.0 * np.mean(disagreement)
        ) if disagreement.size else None,
        "seed17_ho_seed29_stay_percent": float(
            100.0 * np.mean(action_a & ~action_b)
        ) if a.size else None,
        "seed17_stay_seed29_ho_percent": float(
            100.0 * np.mean(~action_a & action_b)
        ) if a.size else None,
        "p_ho_correlation": (
            float(np.corrcoef(a, b)[0, 1])
            if a.size > 1 and np.std(a) > 0 and np.std(b) > 0 else None
        ),
    }


def _state_fingerprint(
    *, observation: np.ndarray, mask: np.ndarray, speed: int,
    trajectory: int, tic: int, serving: int, target: int, stage: str,
) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(observation, dtype="<f4").tobytes())
    digest.update(np.asarray(mask, dtype=np.uint8).tobytes())
    digest.update(
        f"|{speed}|{trajectory}|{tic}|{serving}|{target}|{stage}".encode()
    )
    return digest.hexdigest()


def _collect_policy_rollout(
    *, env: HandoverSMDPEnv, agent: SMDPPPO,
    pairs: Sequence[TracePair], label: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    occurrences: list[dict[str, Any]] = []
    route_rows: list[dict[str, Any]] = []
    for trace_index, pair in enumerate(pairs):
        try:
            observation, _ = env.reset(options={"trace_index": trace_index})
            actionable = True
        except RuntimeError:
            observation = None
            actionable = False
        stays = 0
        handovers = 0
        while actionable and observation is not None:
            decision = env.engine.decision_point
            if decision is None:
                raise RuntimeError("T6A audit received observation without decision.")
            mask = env.action_mask().astype(np.bool_, copy=True)
            action, _, _, probabilities = agent.model.act_with_probabilities(
                observation,
                device=agent.device,
                deterministic=True,
                action_mask=mask,
            )
            stage = str(getattr(decision, "stage", "completed_a3"))
            fingerprint = _state_fingerprint(
                observation=observation,
                mask=mask,
                speed=int(pair.speed_kmh),
                trajectory=int(pair.trajectory_id),
                tic=int(decision.tic),
                serving=int(decision.serving_cell),
                target=int(decision.target_cell),
                stage=stage,
            )
            occurrences.append({
                "fingerprint": fingerprint,
                "source": label,
                "speed_kmh": int(pair.speed_kmh),
                "trajectory_id": int(pair.trajectory_id),
                "tic": int(decision.tic),
                "serving_cell": int(decision.serving_cell),
                "target_cell": int(decision.target_cell),
                "stage": stage,
                "a3_age_ms": int(
                    (int(decision.tic) - int(decision.a3_start_tic))
                    * env.config.delta_t_ms
                ),
                "source_action": HandoverAction(action).name,
                "source_p_ho": float(probabilities[1]),
                "observation": np.asarray(observation, dtype=np.float32).copy(),
                "action_mask": mask,
            })
            if action == int(HandoverAction.STAY):
                stays += 1
            else:
                handovers += 1
            observation, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
        trace = env.engine.trace
        if trace is None:
            raise RuntimeError("T6A audit environment did not retain the trace.")
        metrics = _route_metrics(env.engine, trace)
        metrics.update({
            "actionable": actionable,
            "stay_actions": stays,
            "handover_actions": handovers,
        })
        route_rows.append({
            "trace_id": f"{pair.speed_kmh}:{pair.trajectory_id}",
            "speed_kmh": int(pair.speed_kmh),
            "trajectory_id": int(pair.trajectory_id),
            label: metrics,
        })
        print(
            f"[T6A divergence:{label}] {trace_index + 1}/{len(pairs)} "
            f"{pair.speed_kmh}:{pair.trajectory_id}",
            flush=True,
        )
    return occurrences, route_rows


def _union_states(
    occurrences: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], np.ndarray]:
    unique: list[dict[str, Any]] = []
    by_fingerprint: dict[str, int] = {}
    occurrence_indices = np.empty(len(occurrences), dtype=np.int64)
    for occurrence_index, row in enumerate(occurrences):
        fingerprint = str(row["fingerprint"])
        index = by_fingerprint.get(fingerprint)
        source_bit = 1 if row["source"] == "seed17" else 2
        if index is None:
            index = len(unique)
            by_fingerprint[fingerprint] = index
            item = dict(row)
            item["source_mask"] = source_bit
            item["occurrence_count"] = 1
            unique.append(item)
        else:
            unique[index]["source_mask"] |= source_bit
            unique[index]["occurrence_count"] += 1
        occurrence_indices[occurrence_index] = index
    return unique, occurrence_indices


def _score_states(
    agent: SMDPPPO, observations: np.ndarray,
    masks: np.ndarray, batch_size: int = 4096,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    probabilities: list[np.ndarray] = []
    values: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, observations.shape[0], batch_size):
            stop = min(start + batch_size, observations.shape[0])
            tensor = torch.as_tensor(
                observations[start:stop], dtype=torch.float32, device=agent.device
            )
            mask_tensor = torch.as_tensor(
                masks[start:stop], dtype=torch.bool, device=agent.device
            )
            distribution, value = agent.model.distribution_and_value(
                tensor, mask_tensor
            )
            probabilities.append(distribution.probs[:, 1].cpu().numpy())
            values.append(value.cpu().numpy())
            actions.append(torch.argmax(distribution.logits, dim=-1).cpu().numpy())
    return (
        np.concatenate(probabilities).astype(np.float32, copy=False),
        np.concatenate(values).astype(np.float32, copy=False),
        np.concatenate(actions).astype(np.int8, copy=False),
    )


def _read_training_signals(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            event = json.loads(line)
            if event.get("event") != "update_complete":
                continue
            diagnostics = event.get("learning_diagnostics", {})
            action_update = diagnostics.get("action_update", {}).get("actions", {})
            if not action_update:
                continue
            row: dict[str, Any] = {
                "update": int(event["update"]),
                "macro_steps": int(event["macro_steps"]),
                "entropy": float(event["entropy"]),
                "normalized_entropy": float(event.get("normalized_entropy", 0.0)),
                "explained_variance": float(event["explained_variance"]),
            }
            for action_name in ("STAY", "HO_C1"):
                item = action_update[action_name]
                prefix = action_name.lower()
                row[f"{prefix}_chosen_fraction"] = float(item["chosen_fraction"])
                row[f"{prefix}_normalized_advantage_mean"] = float(
                    item["normalized_advantage"]["mean"]
                )
                row[f"{prefix}_raw_advantage_mean"] = float(
                    item["raw_advantage"]["mean"]
                )
                row[f"{prefix}_positive_advantage_fraction"] = float(
                    item["positive_advantage_fraction"]
                )
                row[f"{prefix}_probability_pre"] = float(
                    item["mean_probability_pre_when_available"]
                )
                row[f"{prefix}_probability_post"] = float(
                    item["mean_probability_post_when_available"]
                )
                row[f"{prefix}_probability_delta"] = float(
                    item["mean_probability_delta_when_available"]
                )
            rows.append(row)
    if not rows:
        raise ValueError(f"No PPO action-update diagnostics in {path}")
    return rows


def summarize_training_signals(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Training signal rows must not be empty.")

    def segment(group: Sequence[dict[str, Any]]) -> dict[str, Any]:
        fields = (
            "entropy", "normalized_entropy", "explained_variance",
            "stay_chosen_fraction", "ho_c1_chosen_fraction",
            "stay_normalized_advantage_mean", "ho_c1_normalized_advantage_mean",
            "stay_raw_advantage_mean", "ho_c1_raw_advantage_mean",
            "stay_positive_advantage_fraction", "ho_c1_positive_advantage_fraction",
            "stay_probability_pre", "ho_c1_probability_pre",
            "stay_probability_delta", "ho_c1_probability_delta",
        )
        return {
            "updates": len(group),
            "first_update": int(group[0]["update"]),
            "last_update": int(group[-1]["update"]),
            **{field: _summary(row[field] for row in group) for field in fields},
        }

    count = len(rows)
    quarter = max(1, count // 4)
    return {
        "all": segment(rows),
        "first_quarter": segment(rows[:quarter]),
        "last_quarter": segment(rows[-quarter:]),
        "last_10_updates": segment(rows[-min(10, count):]),
    }


def _indices_for(
    metadata: Sequence[dict[str, Any]], field: str, value: Any,
) -> np.ndarray:
    return np.asarray(
        [index for index, row in enumerate(metadata) if row[field] == value],
        dtype=np.int64,
    )


def run_audit(
    *, root: Path, data_dir: Path, route_split_path: Path,
    checkpoint17: Path, checkpoint29: Path,
    metrics17: Path, metrics29: Path, output_dir: Path,
    bootstrap_resamples: int, bootstrap_seed: int,
    max_routes_per_speed: int | None = None,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite T6A audit output: {output_dir}")
    output_dir.mkdir(parents=True)
    started = time.perf_counter()
    all_pairs, route_split = _load_route_split(route_split_path, data_dir)
    pairs = _filter_pairs_per_speed(all_pairs, max_routes_per_speed)
    config = load_protocol_config(str(data_dir))
    checkpoints = {"seed17": checkpoint17, "seed29": checkpoint29}
    agents: dict[str, SMDPPPO] = {}
    extras: dict[str, dict[str, Any]] = {}
    for label, path in checkpoints.items():
        agent, extra = SMDPPPO.from_checkpoint(path, device="cpu")
        experiment = extra.get("experiment_config", {})
        expected_seed = int(label.removeprefix("seed"))
        actual_seed = int(experiment.get("seeds", {}).get("training", -1))
        if actual_seed != expected_seed:
            raise ValueError(
                f"{label} checkpoint training seed is {actual_seed}, expected {expected_seed}."
            )
        objective = extra.get("objective", {})
        if objective.get("trigger") != "a3_entry_redecision_v1":
            raise ValueError(f"{label} is not a T6A A3-entry checkpoint.")
        if int(objective.get("candidate_count", -1)) != 1:
            raise ValueError("T6A divergence audit requires K=1 timing-only policies.")
        agents[label] = agent
        extras[label] = extra
    if extras["seed17"].get("objective") != extras["seed29"].get("objective"):
        raise ValueError("Seed checkpoints do not share the exact T6A objective.")

    occurrences: list[dict[str, Any]] = []
    route_rows_by_policy: dict[str, list[dict[str, Any]]] = {}
    for label in POLICY_LABELS:
        env = _objective_environment(
            data_dir=data_dir,
            pairs=pairs,
            agent=agents[label],
            extra=extras[label],
            protocol_config=config,
        )
        collected, route_rows = _collect_policy_rollout(
            env=env, agent=agents[label], pairs=pairs, label=label
        )
        occurrences.extend(collected)
        route_rows_by_policy[label] = route_rows

    union, occurrence_indices = _union_states(occurrences)
    observations = np.stack([row["observation"] for row in union]).astype(np.float32)
    masks = np.stack([row["action_mask"] for row in union]).astype(np.bool_)
    scores: dict[str, dict[str, np.ndarray]] = {}
    for label in POLICY_LABELS:
        p_ho, values, actions = _score_states(agents[label], observations, masks)
        scores[label] = {"p_ho": p_ho, "value": values, "action": actions}

    metadata = [{
        key: row[key]
        for key in (
            "fingerprint", "speed_kmh", "trajectory_id", "tic",
            "serving_cell", "target_cell", "stage", "a3_age_ms",
            "source_mask", "occurrence_count",
        )
    } for row in union]
    same_state: dict[str, Any] = {
        "unique_union": comparison_summary(
            scores["seed17"]["p_ho"], scores["seed29"]["p_ho"]
        ),
        "common_to_both_rollouts": comparison_summary(
            scores["seed17"]["p_ho"], scores["seed29"]["p_ho"],
            _indices_for(metadata, "source_mask", 3),
        ),
        "by_source_occupancy": {},
        "by_stage": {},
        "by_speed": {},
    }
    occurrence_sources = np.asarray([row["source"] for row in occurrences])
    for label in POLICY_LABELS:
        selected_occurrences = occurrence_indices[occurrence_sources == label]
        same_state["by_source_occupancy"][label] = comparison_summary(
            scores["seed17"]["p_ho"], scores["seed29"]["p_ho"],
            selected_occurrences,
        )
    for stage in sorted({str(row["stage"]) for row in metadata}):
        same_state["by_stage"][stage] = comparison_summary(
            scores["seed17"]["p_ho"], scores["seed29"]["p_ho"],
            _indices_for(metadata, "stage", stage),
        )
    for speed in sorted({int(row["speed_kmh"]) for row in metadata}):
        same_state["by_speed"][str(speed)] = comparison_summary(
            scores["seed17"]["p_ho"], scores["seed29"]["p_ho"],
            _indices_for(metadata, "speed_kmh", speed),
        )

    source_mask = np.asarray([row["source_mask"] for row in metadata], dtype=np.int8)
    npz_path = output_dir / "same_state_bank.npz"
    np.savez_compressed(
        npz_path,
        observation=observations,
        action_mask=masks,
        speed_kmh=np.asarray([row["speed_kmh"] for row in metadata], dtype=np.int16),
        trajectory_id=np.asarray([row["trajectory_id"] for row in metadata], dtype=np.int32),
        tic=np.asarray([row["tic"] for row in metadata], dtype=np.int32),
        serving_cell=np.asarray([row["serving_cell"] for row in metadata], dtype=np.int16),
        target_cell=np.asarray([row["target_cell"] for row in metadata], dtype=np.int16),
        a3_age_ms=np.asarray([row["a3_age_ms"] for row in metadata], dtype=np.int32),
        stage=np.asarray([row["stage"] for row in metadata], dtype="U16"),
        source_mask=source_mask,
        occurrence_count=np.asarray(
            [row["occurrence_count"] for row in metadata], dtype=np.int32
        ),
        p_ho_seed17=scores["seed17"]["p_ho"],
        p_ho_seed29=scores["seed29"]["p_ho"],
        value_seed17=scores["seed17"]["value"],
        value_seed29=scores["seed29"]["value"],
        action_seed17=scores["seed17"]["action"],
        action_seed29=scores["seed29"]["action"],
    )
    with np.load(npz_path, allow_pickle=False) as loaded:
        if set(loaded.files) == set():
            raise RuntimeError("T6A state bank unexpectedly empty.")

    feature_names = list(extras["seed17"]["objective"]["observation_features"])
    disagreement_path = output_dir / "disagreement_states.jsonl"
    disagreement = scores["seed17"]["action"] != scores["seed29"]["action"]
    with disagreement_path.open("w", encoding="utf-8") as stream:
        for index in np.flatnonzero(disagreement):
            row = metadata[int(index)]
            stream.write(json.dumps({
                **row,
                "p_ho_seed17": float(scores["seed17"]["p_ho"][index]),
                "p_ho_seed29": float(scores["seed29"]["p_ho"][index]),
                "delta_p_ho_seed29_minus_seed17": float(
                    scores["seed29"]["p_ho"][index]
                    - scores["seed17"]["p_ho"][index]
                ),
                "action_seed17": HandoverAction(
                    int(scores["seed17"]["action"][index])
                ).name,
                "action_seed29": HandoverAction(
                    int(scores["seed29"]["action"][index])
                ).name,
                "features": {
                    name: float(value)
                    for name, value in zip(
                        feature_names, observations[index], strict=True
                    )
                },
            }, allow_nan=False) + "\n")

    merged_routes: list[dict[str, Any]] = []
    for row17, row29 in zip(
        route_rows_by_policy["seed17"],
        route_rows_by_policy["seed29"],
        strict=True,
    ):
        if row17["trace_id"] != row29["trace_id"]:
            raise RuntimeError("T6A route alignment failed.")
        merged_routes.append({
            "trace_id": row17["trace_id"],
            "speed_kmh": row17["speed_kmh"],
            "trajectory_id": row17["trajectory_id"],
            "seed17": row17["seed17"],
            "seed29": row29["seed29"],
        })
    with (output_dir / "route_outcomes.jsonl").open("w", encoding="utf-8") as stream:
        for row in merged_routes:
            stream.write(json.dumps(row, allow_nan=False) + "\n")

    actual = {
        label: _extended_aggregate(merged_routes, label, config.delta_t_ms)
        for label in POLICY_LABELS
    }
    actual["paired_seed29_minus_seed17"] = paired_bootstrap(
        merged_routes,
        left="seed17",
        right="seed29",
        dt_ms=config.delta_t_ms,
        resamples=bootstrap_resamples,
        seed=bootstrap_seed,
    )

    signal_paths = {"seed17": metrics17, "seed29": metrics29}
    signals: dict[str, Any] = {}
    with (output_dir / "training_signal_rows.jsonl").open(
        "w", encoding="utf-8"
    ) as stream:
        for label, path in signal_paths.items():
            rows = _read_training_signals(path)
            signals[label] = summarize_training_signals(rows)
            for row in rows:
                stream.write(json.dumps({"policy": label, **row}, allow_nan=False) + "\n")

    union_summary = same_state["unique_union"]
    gates = {
        "g_same_state_divergence": {
            "pass": (
                float(union_summary["deterministic_disagreement_percent"] or 0.0)
                >= 10.0
            ),
            "criterion": "deterministic disagreement >= 10% on unique TRAIN union",
        },
        "g_near_tie_amplification": {
            "pass": max(
                float(union_summary[label]["near_tie_percent_abs_margin_le_0_10"] or 0.0)
                for label in POLICY_LABELS
            ) >= 50.0,
            "criterion": "at least one policy has >=50% states with |P(HO)-P(STAY)|<=0.1",
        },
        "g_seed29_under_handover": {
            "pass": (
                actual["seed29"]["handover_attempts"]
                < actual["seed17"]["handover_attempts"]
                and actual["seed29"]["mean_route_relative_rate_percent"]
                < actual["seed17"]["mean_route_relative_rate_percent"]
            ),
            "criterion": "seed29 has fewer HO attempts and lower RR than seed17",
        },
    }
    report = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "status": "COMPLETE",
        "scope": "train_only",
        "validation_or_test_rows_read": 0,
        "routes": len(pairs),
        "speeds_kmh": sorted({int(pair.speed_kmh) for pair in pairs}),
        "state_bank": {
            "occurrences": len(occurrences),
            "unique_states": len(union),
            "common_states": int(np.sum(source_mask == 3)),
            "observation_schema": extras["seed17"]["objective"].get(
                "observation_schema_key"
            ),
            "observation_features": feature_names,
            "npz": str(npz_path.resolve()),
            "npz_sha256": _sha256(npz_path),
        },
        "same_state_policy_comparison": same_state,
        "actual_train_full_route": actual,
        "training_signal_history": signals,
        "gates": gates,
        "interpretation_contract": [
            "All policy and state-bank rows come from the locked TRAIN split.",
            "Same-state scores are cross-policy inference, not counterfactual outcomes.",
            "Training advantages are on-policy rollout diagnostics and are not causal labels.",
            "No validation, test, or zero-shot trajectory was read by this audit.",
        ],
        "runtime_seconds": time.perf_counter() - started,
    }
    report_path = output_dir / "main_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "train_only",
        "validation_or_test_rows_read": 0,
        "data_dir": str(data_dir.resolve()),
        "route_split": {
            "path": str(route_split_path.resolve()),
            "sha256": _sha256(route_split_path),
            "routes": len(pairs),
            "source_routes": len(route_split["selected_routes"]),
        },
        "checkpoints": {
            label: {"path": str(path.resolve()), "sha256": _sha256(path)}
            for label, path in checkpoints.items()
        },
        "training_metrics": {
            label: {"path": str(path.resolve()), "sha256": _sha256(path)}
            for label, path in signal_paths.items()
        },
        "bootstrap_resamples": bootstrap_resamples,
        "bootstrap_seed": bootstrap_seed,
        "protocol_config": asdict(config),
        "git": _git_state(root),
        "source_files": [{
            "path": str(Path(__file__).resolve().relative_to(root)),
            "sha256": _sha256(Path(__file__).resolve()),
        }],
        "artifacts": {
            "main_report": {"path": str(report_path.resolve()), "sha256": _sha256(report_path)},
            "state_bank": {"path": str(npz_path.resolve()), "sha256": _sha256(npz_path)},
            "disagreement_states": {
                "path": str(disagreement_path.resolve()),
                "sha256": _sha256(disagreement_path),
            },
        },
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )

    delta = actual["paired_seed29_minus_seed17"]["delta_right_minus_left"]
    lines = [
        "# T6A seed 17 vs seed 29 — TRAIN-only divergence audit",
        "",
        f"- Routes: {len(pairs)}; validation/test/zero-shot rows read: 0.",
        f"- State occurrences: {len(occurrences)}; unique union: {len(union)}; "
        f"common to both rollouts: {int(np.sum(source_mask == 3))}.",
        f"- Same-state deterministic disagreement: "
        f"{union_summary['deterministic_disagreement_percent']:.2f}%.",
        f"- Seed17 HO / seed29 STAY: "
        f"{union_summary['seed17_ho_seed29_stay_percent']:.2f}% of unique states.",
        f"- Mean P(HO) seed29-seed17: "
        f"{union_summary['p_ho_seed29_minus_seed17']['mean']:+.4f}.",
        f"- Full-route seed29-seed17: ΔRR="
        f"{delta['mean_route_relative_rate_percent']:+.4f} pp; ΔHOF="
        f"{delta['handover_failure_percent']:+.4f} pp; ΔPP="
        f"{delta['ping_pong_percent']:+.4f} pp; ΔRLF-time="
        f"{delta['rlf_time_percent']:+.4f} pp.",
        "",
        "This is a TRAIN diagnostic. Same-state probabilities are policy scores; "
        "on-policy advantages are not counterfactual labels.",
    ]
    (output_dir / "report_vi.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main(root_path: str, args: Namespace) -> int:
    root = Path(root_path).resolve()

    def resolve(value: str) -> Path:
        path = Path(value)
        return (path if path.is_absolute() else root / path).resolve()

    report = run_audit(
        root=root,
        data_dir=resolve(args.data_dir),
        route_split_path=resolve(args.route_split),
        checkpoint17=resolve(args.seed17_checkpoint),
        checkpoint29=resolve(args.seed29_checkpoint),
        metrics17=resolve(args.seed17_metrics),
        metrics29=resolve(args.seed29_metrics),
        output_dir=resolve(args.output_dir),
        bootstrap_resamples=int(args.bootstrap_resamples),
        bootstrap_seed=int(args.bootstrap_seed),
        max_routes_per_speed=args.max_routes_per_speed,
    )
    print(json.dumps({
        "status": report["status"],
        "routes": report["routes"],
        "state_bank": report["state_bank"],
        "gates": report["gates"],
    }, indent=2, allow_nan=False), flush=True)
    return 0
