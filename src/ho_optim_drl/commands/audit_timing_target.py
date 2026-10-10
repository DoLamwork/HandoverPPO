"""D63 TRAIN-only timing x target counterfactual correctness audit."""

from __future__ import annotations

from argparse import Namespace
from collections import Counter, defaultdict
import copy
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Callable, Iterable, Iterator, Sequence

import numpy as np

from ho_optim_drl.agents.ppo import SMDPPPO
from ho_optim_drl.commands.audit_policy_learning import (
    _make_builders,
    _policy_spec,
    _selected_routes,
    _sha256_file,
)
from ho_optim_drl.data import TracePair, load_cell_topology, load_radio_trace
from ho_optim_drl.data.project import load_protocol_config
from ho_optim_drl.protocol import HandoverEngine, ProtocolPhase
from ho_optim_drl.smdp import (
    P3_E07_REWARD_SCHEMA,
    RewardBreakdown,
    TopologyCandidateObservationBuilder,
    build_reward_model,
    trajectory_reward_config,
)


SCHEMA = "d63_timing_target_pilot_v1"
MAIN_SCHEMA = "d63_timing_target_main_v1"
OBSERVATION_SCHEMA = "qos_full_v1"
CANDIDATE_COUNT = 3
CANDIDATE_ORDER = "ranked_v1"
STAY_MODE = "a3_redecision_v1"
STAY_REDECISION_MS = 100
REWARD_SCHEMA = P3_E07_REWARD_SCHEMA
LOCAL_METRICS = (
    "delivered_mbit",
    "relative_rate_percent",
    "handover_failures",
    "rlf_starts",
    "pure_rlf_starts",
    "ping_pong_count",
    "recovery_seconds",
    "handover_interruption_seconds",
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


def _run_git(root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ("git", *args),
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.rstrip("\n")


def _current_source_provenance(root: Path) -> dict[str, Any]:
    files = [root / "run.py", root / "pyproject.toml"]
    files.extend(sorted((root / "src" / "ho_optim_drl").rglob("*.py")))
    records = []
    digest = hashlib.sha256()
    for path in files:
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        file_hash = _sha256_file(path)
        records.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": file_hash,
            }
        )
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_hash.encode("ascii"))
        digest.update(b"\n")
    patch = _run_git(root, "diff", "--binary", "HEAD", "--", "run.py", "pyproject.toml", "src")
    status = (_run_git(root, "status", "--porcelain=v1", "--untracked-files=all") or "").splitlines()
    return {
        "git_revision": _run_git(root, "rev-parse", "HEAD"),
        "git_branch": _run_git(root, "branch", "--show-current"),
        "git_dirty": bool(status),
        "git_status_porcelain": status,
        "tracked_patch_sha256": (
            hashlib.sha256(patch.encode("utf-8")).hexdigest()
            if patch is not None
            else None
        ),
        "runtime_source_tree_sha256": digest.hexdigest(),
        "runtime_source_files": records,
    }


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _checkpoint_registry_record(label: str, checkpoint: Path) -> dict[str, Any]:
    agent, extra, spec = _policy_spec(str(checkpoint))
    objective = dict(extra.get("objective") or {})
    required = {
        "candidate_count": CANDIDATE_COUNT,
        "candidate_order_key": CANDIDATE_ORDER,
        "observation_schema_key": OBSERVATION_SCHEMA,
        "reward": REWARD_SCHEMA,
        "stay_mode": STAY_MODE,
        "stay_redecision_ms": STAY_REDECISION_MS,
    }
    missing = [name for name in required if name not in objective]
    if missing:
        raise ValueError(
            f"{label} checkpoint lacks explicit D63 objective fields: {missing}."
        )
    mismatches = {
        name: {"expected": expected, "actual": objective.get(name)}
        for name, expected in required.items()
        if objective.get(name) != expected
    }
    if mismatches:
        raise ValueError(f"{label} objective is incompatible with D63: {mismatches}")

    model_dir = checkpoint.parent
    report_path = model_dir / "training_report.json"
    config_path = model_dir / "experiment_config.json"
    reference_path = model_dir / "validation_3gpp_reference.json"
    for path in (report_path, config_path, reference_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    report = _load_json(report_path)
    experiment = _load_json(config_path)
    config = dict(experiment.get("config") or experiment)
    budget = dict(config.get("budget") or {})
    validation_records_path = model_dir / "checkpoints" / "validation_metrics.json"
    validation_records = (
        json.loads(validation_records_path.read_text(encoding="utf-8"))
        if validation_records_path.is_file()
        else []
    )
    budget_steps = int(budget.get("total_macro_steps", agent.total_macro_steps))
    budget_end = next(
        (
            record
            for record in validation_records
            if int(record.get("macro_steps", -1)) == budget_steps
        ),
        None,
    )
    return {
        "label": label,
        **spec,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": _sha256_file(checkpoint),
        "experiment_id": config.get("experiment_id"),
        "test_evaluated": bool(report.get("test_evaluated", False)),
        "budget": budget,
        "selected_macro_steps": int(agent.total_macro_steps),
        "selected_updates": int(agent.update_count),
        "budget_end_validation": budget_end,
        "selected_validation": report.get("validation"),
        "objective": objective,
        "network_architecture": agent.model.architecture,
        "action_decoder": "joint_argmax",
        "files": {
            "training_report": {
                "path": str(report_path.resolve()),
                "sha256": _sha256_file(report_path),
            },
            "experiment_config": {
                "path": str(config_path.resolve()),
                "sha256": _sha256_file(config_path),
            },
            "validation_3gpp_reference": {
                "path": str(reference_path.resolve()),
                "sha256": _sha256_file(reference_path),
            },
        },
        "training_source": config.get("source"),
        "dataset": config.get("dataset"),
        "validation_3gpp_reference": _load_json(reference_path),
    }


def build_registry(
    *,
    root: Path,
    d61_checkpoint: Path,
    d62_checkpoint: Path,
) -> dict[str, Any]:
    sources = [
        _checkpoint_registry_record("d61", d61_checkpoint),
        _checkpoint_registry_record("d62", d62_checkpoint),
    ]
    reference_hashes = {
        item["files"]["validation_3gpp_reference"]["sha256"] for item in sources
    }
    if len(reference_hashes) != 1:
        raise ValueError("D61 and D62 do not share the same 3GPP reference artifact.")
    if any(item["test_evaluated"] for item in sources):
        raise ValueError("D63 inputs must not use checkpoints with final test evaluation.")
    return {
        "schema_name": "d63_frozen_evidence_registry",
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "split": "train",
        "validation_or_test_rows_read_for_d63_labels": 0,
        "current_source": _current_source_provenance(root),
        "sources": sources,
        "reference_3gpp": sources[0]["validation_3gpp_reference"],
        "contract": {
            "candidate_count": CANDIDATE_COUNT,
            "candidate_order": CANDIDATE_ORDER,
            "observation_schema": OBSERVATION_SCHEMA,
            "reward_schema": REWARD_SCHEMA,
            "stay_mode": STAY_MODE,
            "stay_redecision_ms": STAY_REDECISION_MS,
            "trigger": "completed_a3_ttt",
            "action_decoder": "joint_argmax",
        },
    }


def _source_specs(
    d61_checkpoint: Path, d62_checkpoint: Path
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {
        "always_ho_c1": {
            "type": "deterministic_reference",
            "agent": None,
            "stay_mode": None,
            "stay_redecision_ms": None,
        }
    }
    for label, path in (("d61", d61_checkpoint), ("d62", d62_checkpoint)):
        agent, extra, spec = _policy_spec(str(path))
        objective = dict(extra.get("objective") or {})
        result[label] = {
            "type": "ppo_checkpoint",
            "agent": agent,
            "checkpoint": str(path.resolve()),
            "checkpoint_sha256": spec["checkpoint_sha256"],
            "observation_schema": spec["observation_schema"],
            "candidate_count": spec["candidate_count"],
            "candidate_order": spec["candidate_order"],
            "macro_steps": spec["macro_steps"],
            "updates": spec["updates"],
            "stay_mode": objective["stay_mode"],
            "stay_redecision_ms": int(objective["stay_redecision_ms"]),
            "architecture": agent.model.architecture,
        }
    return result


def _serializable_source_spec(spec: dict[str, Any]) -> dict[str, Any]:
    return {name: value for name, value in spec.items() if name != "agent"}


def _apply_source_action(
    engine: HandoverEngine,
    builder: TopologyCandidateObservationBuilder,
    observation: np.ndarray,
    source: dict[str, Any],
) -> tuple[int, tuple[int, int, int] | None]:
    agent: SMDPPPO | None = source["agent"]
    action = 1 if agent is None else int(
        agent.predict(
            observation,
            deterministic=True,
            action_mask=builder.action_mask(),
        )
    )
    decision = engine.decision_point
    if decision is None:
        raise RuntimeError("Source action requires a pending decision.")
    if action == 0:
        if source.get("stay_mode") != STAY_MODE:
            raise ValueError("D63 PPO source must explicitly use a3_redecision_v1.")
        ticks = int(source["stay_redecision_ms"]) // engine.config.delta_t_ms
        engine.stay(redecision_delay_ticks=ticks)
        return action, (
            decision.serving_cell,
            decision.target_cell,
            decision.a3_start_tic,
        )
    target = builder.candidate_cells[action - 1]
    if target is None:
        raise RuntimeError("Source policy selected an unavailable candidate.")
    engine.start_handover(int(target))
    return action, None


def _decision_stream(
    *,
    pair: TracePair,
    data_dir: str,
    protocol_config: Any,
    topology: Any,
    source: dict[str, Any],
) -> Iterator[tuple[dict[str, Any], HandoverEngine, TopologyCandidateObservationBuilder]]:
    trace = load_radio_trace(
        protocol_config,
        data_dir,
        pair.rsrp_file,
        pair.sinr_file,
        speed_kmh=pair.speed_kmh,
    )
    engine = HandoverEngine(protocol_config)
    engine.reset(trace)
    builder = _make_builders(
        protocol_config,
        topology,
        candidate_count=CANDIDATE_COUNT,
        candidate_order=CANDIDATE_ORDER,
        schemas=(OBSERVATION_SCHEMA,),
    )[OBSERVATION_SCHEMA]
    builder.reset()
    builder.start_episode(trace)
    previous_stay_event: tuple[int, int, int] | None = None
    decision_index = 0
    while True:
        tick = engine.advance_one_tick()
        builder.observe(tick)
        if tick.terminated:
            return
        decision = engine.decision_point
        if decision is None:
            continue
        observation = builder.build(engine.snapshot)
        event = (
            decision.serving_cell,
            decision.target_cell,
            decision.a3_start_tic,
        )
        row = {
            "decision_index": decision_index,
            "tic": int(decision.tic),
            "serving_cell": int(decision.serving_cell),
            "a3_target_cell": int(decision.target_cell),
            "a3_start_tic": int(decision.a3_start_tic),
            "a3_margin_db": float(decision.a3_margin_db),
            "is_redecision": previous_stay_event == event,
            "time_since_a3_ms": int(
                (decision.tic - decision.a3_start_tic) * protocol_config.delta_t_ms
            ),
            "time_since_last_ho_ms": int(
                engine.snapshot.time_since_last_ho_ticks * protocol_config.delta_t_ms
            ),
            "candidate_cells": tuple(builder.candidate_cells),
            "action_mask": builder.action_mask().copy(),
            "observation": observation.copy(),
        }
        yield row, engine, builder
        _, previous_stay_event = _apply_source_action(
            engine, builder, observation, source
        )
        decision_index += 1


def _distributed_indices(
    count: int,
    maximum: int,
    *,
    seed: int,
    source_index: int,
    speed: int,
    trajectory: int,
) -> tuple[int, ...]:
    if count <= maximum:
        return tuple(range(count))
    rng = np.random.default_rng(
        np.random.SeedSequence((seed, source_index, speed, trajectory))
    )
    edges = np.linspace(0, count, maximum + 1, dtype=int)
    chosen = []
    for start, stop in zip(edges[:-1], edges[1:]):
        if stop <= start:
            continue
        chosen.append(int(rng.integers(start, stop)))
    return tuple(sorted(set(chosen)))


def _states_to_arrays(states: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    candidate_cells = [
        [-1 if cell is None else int(cell) for cell in state["candidate_cells"]]
        for state in states
    ]
    return {
        "state_id": np.asarray([state["state_id"] for state in states]),
        "source": np.asarray([state["source"] for state in states]),
        "speed_kmh": np.asarray([state["speed_kmh"] for state in states], dtype=np.int32),
        "trajectory_id": np.asarray(
            [state["trajectory_id"] for state in states], dtype=np.int32
        ),
        "decision_index": np.asarray(
            [state["decision_index"] for state in states], dtype=np.int32
        ),
        "tic": np.asarray([state["tic"] for state in states], dtype=np.int32),
        "serving_cell": np.asarray(
            [state["serving_cell"] for state in states], dtype=np.int16
        ),
        "a3_target_cell": np.asarray(
            [state["a3_target_cell"] for state in states], dtype=np.int16
        ),
        "a3_start_tic": np.asarray(
            [state["a3_start_tic"] for state in states], dtype=np.int32
        ),
        "a3_margin_db": np.asarray(
            [state["a3_margin_db"] for state in states], dtype=np.float32
        ),
        "is_redecision": np.asarray(
            [state["is_redecision"] for state in states], dtype=np.bool_
        ),
        "time_since_a3_ms": np.asarray(
            [state["time_since_a3_ms"] for state in states], dtype=np.int32
        ),
        "time_since_last_ho_ms": np.asarray(
            [state["time_since_last_ho_ms"] for state in states], dtype=np.int32
        ),
        "candidate_cells": np.asarray(candidate_cells, dtype=np.int16),
        "action_mask": np.asarray(
            [state["action_mask"] for state in states], dtype=np.bool_
        ),
        "observation": np.asarray(
            [state["observation"] for state in states], dtype=np.float32
        ),
    }


def build_d63_state_bank(
    *,
    data_dir: str,
    output_dir: Path,
    speeds: list[int],
    routes_per_speed: int,
    max_anchors_per_source_route: int,
    split_seed: int,
    sample_seed: int,
    source_specs: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    protocol_config = load_protocol_config(data_dir)
    if STAY_REDECISION_MS % protocol_config.delta_t_ms:
        raise ValueError("100 ms redecision must align to dataset delta_t_ms.")
    topology = load_cell_topology(data_dir)
    pairs = _selected_routes(
        data_dir=data_dir,
        speeds=speeds,
        split_seed=split_seed,
        sample_seed=sample_seed,
        routes_per_speed=routes_per_speed,
        train_fraction=0.70,
        validation_fraction=0.15,
        test_fraction=0.15,
    )
    states: list[dict[str, Any]] = []
    route_decision_counts: list[dict[str, Any]] = []
    for source_index, (label, source) in enumerate(source_specs.items()):
        for pair in pairs:
            decisions = [
                dict(row)
                for row, _, _ in _decision_stream(
                    pair=pair,
                    data_dir=data_dir,
                    protocol_config=protocol_config,
                    topology=topology,
                    source=source,
                )
            ]
            selected = _distributed_indices(
                len(decisions),
                max_anchors_per_source_route,
                seed=sample_seed,
                source_index=source_index,
                speed=pair.speed_kmh,
                trajectory=pair.trajectory_id,
            )
            route_decision_counts.append(
                {
                    "source": label,
                    "speed_kmh": pair.speed_kmh,
                    "trajectory_id": pair.trajectory_id,
                    "decision_count": len(decisions),
                    "selected_decision_indices": list(selected),
                }
            )
            for decision_index in selected:
                row = decisions[decision_index]
                row.update(
                    {
                        "state_id": (
                            f"{label}:{pair.speed_kmh}:{pair.trajectory_id}:"
                            f"{row['decision_index']}"
                        ),
                        "source": label,
                        "speed_kmh": pair.speed_kmh,
                        "trajectory_id": pair.trajectory_id,
                    }
                )
                states.append(row)
    if not states:
        raise ValueError("D63 state bank contains no A3 decisions.")
    states.sort(
        key=lambda row: (
            row["source"],
            row["speed_kmh"],
            row["trajectory_id"],
            row["decision_index"],
        )
    )
    arrays = _states_to_arrays(states)
    bank_path = output_dir / "state_bank.npz"
    np.savez_compressed(bank_path, **arrays)
    with np.load(bank_path, allow_pickle=False) as saved:
        if any(saved[name].dtype == object for name in saved.files):
            raise RuntimeError("D63 state bank must remain pickle-free.")

    route_records = [
        {
            "speed_kmh": pair.speed_kmh,
            "trajectory_id": pair.trajectory_id,
            "rsrp_file": pair.rsrp_file,
            "sinr_file": pair.sinr_file,
            "rsrp_sha256": _sha256_file(Path(data_dir, pair.rsrp_file)),
            "sinr_sha256": _sha256_file(Path(data_dir, pair.sinr_file)),
        }
        for pair in pairs
    ]
    provenance_path = Path(data_dir, "dataset_provenance.json")
    manifest = {
        "schema_name": "d63_train_state_bank",
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "split": "train",
        "validation_or_test_rows_read": 0,
        "data_dir": str(Path(data_dir).resolve()),
        "dataset_provenance": (
            {
                "path": str(provenance_path.resolve()),
                "sha256": _sha256_file(provenance_path),
            }
            if provenance_path.is_file()
            else None
        ),
        "protocol_config": asdict(protocol_config),
        "split_seed": split_seed,
        "sample_seed": sample_seed,
        "split_fractions": {"train": 0.70, "validation": 0.15, "test": 0.15},
        "routes_per_speed_requested": routes_per_speed,
        "max_anchors_per_source_route": max_anchors_per_source_route,
        "anchor_selection": "one_seeded_sample_per_equal_decision_index_bin_v1",
        "selected_routes": route_records,
        "route_decision_counts": route_decision_counts,
        "sources": {
            label: _serializable_source_spec(spec)
            for label, spec in source_specs.items()
        },
        "candidate_count": CANDIDATE_COUNT,
        "candidate_order": CANDIDATE_ORDER,
        "observation_schema": OBSERVATION_SCHEMA,
        "stay_mode": STAY_MODE,
        "stay_redecision_ms": STAY_REDECISION_MS,
        "state_count": len(states),
        "state_count_by_source": dict(Counter(row["source"] for row in states)),
        "state_count_by_speed": {
            str(speed): count
            for speed, count in sorted(Counter(row["speed_kmh"] for row in states).items())
        },
        "redecision_count": int(sum(row["is_redecision"] for row in states)),
        "npz_file": bank_path.name,
        "npz_sha256": _sha256_file(bank_path),
        "npz_allow_pickle": False,
    }
    _json_write(output_dir / "state_bank_manifest.json", manifest)
    return manifest, arrays


def _route_lookup(manifest: dict[str, Any]) -> dict[tuple[int, int], TracePair]:
    return {
        (int(item["speed_kmh"]), int(item["trajectory_id"])): TracePair(
            int(item["speed_kmh"]),
            int(item["trajectory_id"]),
            str(item["rsrp_file"]),
            str(item["sinr_file"]),
        )
        for item in manifest["selected_routes"]
    }


def replay_d63_anchors(
    manifest: dict[str, Any],
    arrays: dict[str, np.ndarray],
    source_specs: dict[str, dict[str, Any]],
) -> Iterator[
    tuple[int, HandoverEngine, TopologyCandidateObservationBuilder]
]:
    data_dir = manifest["data_dir"]
    protocol_config = load_protocol_config(data_dir)
    topology = load_cell_topology(data_dir)
    pairs = _route_lookup(manifest)
    groups: dict[tuple[str, int, int], dict[int, int]] = defaultdict(dict)
    for bank_index in range(len(arrays["state_id"])):
        key = (
            str(arrays["source"][bank_index]),
            int(arrays["speed_kmh"][bank_index]),
            int(arrays["trajectory_id"][bank_index]),
        )
        groups[key][int(arrays["decision_index"][bank_index])] = bank_index
    for (label, speed, trajectory), requested in sorted(groups.items()):
        source = source_specs[label]
        for row, engine, builder in _decision_stream(
            pair=pairs[(speed, trajectory)],
            data_dir=data_dir,
            protocol_config=protocol_config,
            topology=topology,
            source=source,
        ):
            bank_index = requested.pop(int(row["decision_index"]), None)
            if bank_index is None:
                continue
            actual_cells = np.asarray(
                [-1 if cell is None else int(cell) for cell in row["candidate_cells"]],
                dtype=np.int16,
            )
            scalar_checks = (
                int(arrays["tic"][bank_index]) == int(row["tic"]),
                int(arrays["serving_cell"][bank_index]) == int(row["serving_cell"]),
                int(arrays["a3_target_cell"][bank_index]) == int(row["a3_target_cell"]),
                int(arrays["a3_start_tic"][bank_index]) == int(row["a3_start_tic"]),
                bool(arrays["is_redecision"][bank_index]) == bool(row["is_redecision"]),
            )
            if (
                not all(scalar_checks)
                or not np.array_equal(arrays["candidate_cells"][bank_index], actual_cells)
                or not np.array_equal(arrays["action_mask"][bank_index], row["action_mask"])
            ):
                raise RuntimeError("D63 replay anchor identity mismatch.")
            np.testing.assert_allclose(
                arrays["observation"][bank_index],
                row["observation"],
                rtol=0,
                atol=1e-7,
            )
            yield bank_index, engine, builder
            if not requested:
                break
        if requested:
            raise RuntimeError(f"Could not replay D63 anchors: {requested}")


def _execute_requested_slot(
    engine: HandoverEngine,
    builder: TopologyCandidateObservationBuilder,
    *,
    requested_slot: int,
    requested_delay_ticks: int,
    anchor_tic: int,
) -> dict[str, Any]:
    if engine.decision_point is None:
        raise RuntimeError("Timing-target execution requires a decision.")
    builder.build(engine.snapshot)
    candidates = tuple(builder.candidate_cells)
    requested_target = candidates[requested_slot - 1]
    fallback = None
    executed_slot: int | None = requested_slot
    executed_target = requested_target
    executed_action = f"C{requested_slot}"
    if requested_target is None:
        executed_slot = 1 if candidates[0] is not None else None
        executed_target = candidates[0]
        if executed_target is None:
            executed_action = "STAY"
            fallback = "stay_no_candidate"
            engine.stay(redecision_delay_ticks=STAY_REDECISION_MS // engine.config.delta_t_ms)
        else:
            executed_action = "C1"
            fallback = "c1_unavailable_requested_slot"
            engine.start_handover(int(executed_target))
    else:
        engine.start_handover(int(requested_target))
    return {
        "requested_slot": requested_slot,
        "requested_delay_ms": requested_delay_ticks * engine.config.delta_t_ms,
        "requested_target_cell": (
            None if requested_target is None else int(requested_target)
        ),
        "executed_action": executed_action,
        "executed_slot": executed_slot,
        "executed_target_cell": (
            None if executed_target is None else int(executed_target)
        ),
        "executed_tic": int(engine.snapshot.tic),
        "actual_delay_ms": int(
            (engine.snapshot.tic - anchor_tic) * engine.config.delta_t_ms
        ),
        "fallback": fallback,
        "status": "executed" if fallback is None else "executed_with_fallback",
        "candidate_cells_at_execution": [
            None if cell is None else int(cell) for cell in candidates
        ],
    }


def _physical_snapshot(
    totals: dict[str, float | int],
    *,
    elapsed_ticks: int,
    delta_t_seconds: float,
    terminated: bool,
) -> dict[str, Any]:
    return {
        "delivered_mbit": float(totals["delivered_mbit"]),
        "best_mbit": float(totals["best_mbit"]),
        "relative_rate_percent": (
            100.0 * float(totals["delivered_mbit"]) / float(totals["best_mbit"])
            if float(totals["best_mbit"]) > 0
            else 0.0
        ),
        "connected_seconds": float(totals["connected_ticks"]) * delta_t_seconds,
        "handover_interruption_seconds": (
            float(totals["handover_interruption_ticks"]) * delta_t_seconds
        ),
        "recovery_seconds": float(totals["recovery_ticks"]) * delta_t_seconds,
        "handover_attempts": int(totals["handover_attempts"]),
        "handover_successes": int(totals["handover_successes"]),
        "handover_failures": int(totals["handover_failures"]),
        "rlf_starts": int(totals["rlf_starts"]),
        "pure_rlf_starts": int(totals["pure_rlf_starts"]),
        "ping_pong_count": int(totals["ping_pong_count"]),
        "elapsed_seconds": elapsed_ticks * delta_t_seconds,
        "trace_terminated": bool(terminated),
    }


def simulate_timing_target(
    engine: HandoverEngine,
    builder: TopologyCandidateObservationBuilder,
    *,
    delay_ticks: int,
    requested_slot: int,
    horizon_ticks: tuple[int, ...],
    gamma_per_second: float = 0.99,
) -> tuple[dict[int, dict[str, Any]], int]:
    """Clone an A3 decision, delay via 100 ms redecisions, then select a slot."""
    if engine.decision_point is None or engine.trace is None:
        raise RuntimeError("D63 counterfactual requires a pending decision.")
    if delay_ticks < 0 or requested_slot not in (1, 2, 3):
        raise ValueError("Delay must be non-negative and slot must be C1/C2/C3.")
    if not horizon_ticks or min(horizon_ticks) <= 0:
        raise ValueError("D63 horizons must be positive.")
    trace = engine.trace
    memo = {id(trace): trace}
    branch = copy.deepcopy(engine, memo)
    branch_builder = copy.deepcopy(builder, memo)
    anchor_tic = int(branch.decision_point.tic)
    deadline_tic = anchor_tic + delay_ticks
    redecision_ticks = STAY_REDECISION_MS // branch.config.delta_t_ms
    execution: dict[str, Any] | None = None
    if delay_ticks == 0:
        execution = _execute_requested_slot(
            branch,
            branch_builder,
            requested_slot=requested_slot,
            requested_delay_ticks=delay_ticks,
            anchor_tic=anchor_tic,
        )
    else:
        branch.stay(redecision_delay_ticks=redecision_ticks)

    reward_model = build_reward_model(
        branch.config,
        REWARD_SCHEMA,
        trajectory_reward_config(REWARD_SCHEMA),
    )
    reward_model.reset(trace)
    reward_model.begin_rollout()
    reward_raw = RewardBreakdown()
    reward_discounted = RewardBreakdown()
    delta_t_seconds = branch.config.delta_t_ms / 1000.0
    totals: dict[str, float | int] = {
        "delivered_mbit": 0.0,
        "best_mbit": 0.0,
        "connected_ticks": 0,
        "handover_interruption_ticks": 0,
        "recovery_ticks": 0,
        "handover_attempts": 0,
        "handover_successes": 0,
        "handover_failures": 0,
        "rlf_starts": 0,
        "pure_rlf_starts": 0,
        "ping_pong_count": 0,
    }
    results: dict[int, dict[str, Any]] = {}
    elapsed_ticks = 0
    simulated_ticks = 0
    terminated = False
    while elapsed_ticks < max(horizon_ticks) and not terminated:
        tick = branch.advance_one_tick()
        simulated_ticks += 1
        elapsed_ticks += 1
        branch_builder.observe(tick)
        totals["delivered_mbit"] += tick.delivered_rate_mbps * delta_t_seconds
        totals["best_mbit"] += tick.best_rate_mbps * delta_t_seconds
        totals["connected_ticks"] += int(not np.isnan(branch.bs_idxs[-1]))
        totals["handover_interruption_ticks"] += int(
            tick.snapshot.phase
            in (ProtocolPhase.HO_PREPARATION, ProtocolPhase.HO_EXECUTION)
        )
        totals["recovery_ticks"] += int(
            tick.snapshot.phase is ProtocolPhase.RLF_RECOVERY
        )
        totals["handover_attempts"] += int(tick.handover_started)
        totals["handover_successes"] += int(tick.handover_succeeded)
        totals["handover_failures"] += int(tick.handover_failed)
        totals["rlf_starts"] += int(tick.rlf_started)
        totals["pure_rlf_starts"] += int(
            tick.rlf_started and not tick.handover_failed
        )
        totals["ping_pong_count"] += int(tick.ping_pong_detected)
        tick_reward = reward_model.compute(tick)
        reward_raw = reward_raw + tick_reward
        reward_discounted = reward_discounted + tick_reward.scaled(
            gamma_per_second ** ((elapsed_ticks - 1) * delta_t_seconds)
        )
        terminated = tick.terminated

        if not terminated and branch.decision_point is not None:
            if execution is None:
                if branch.decision_point.tic >= deadline_tic:
                    execution = _execute_requested_slot(
                        branch,
                        branch_builder,
                        requested_slot=requested_slot,
                        requested_delay_ticks=delay_ticks,
                        anchor_tic=anchor_tic,
                    )
                else:
                    branch.stay(redecision_delay_ticks=redecision_ticks)
            else:
                # Registered continuation: current A3 target / C1.
                branch_builder.build(branch.snapshot)
                target = branch_builder.candidate_cells[0]
                if target is None:
                    branch.stay(redecision_delay_ticks=redecision_ticks)
                else:
                    branch.start_handover(int(target))

        if elapsed_ticks in horizon_ticks:
            results[elapsed_ticks] = {
                "physical": _physical_snapshot(
                    totals,
                    elapsed_ticks=elapsed_ticks,
                    delta_t_seconds=delta_t_seconds,
                    terminated=terminated,
                ),
                "reward_raw": reward_raw.as_dict(),
                "reward_discounted": reward_discounted.as_dict(),
                "execution": (
                    execution
                    if execution is not None
                    else {
                        "requested_slot": requested_slot,
                        "requested_delay_ms": delay_ticks * branch.config.delta_t_ms,
                        "executed_action": None,
                        "executed_slot": None,
                        "executed_target_cell": None,
                        "executed_tic": None,
                        "actual_delay_ms": None,
                        "fallback": None,
                        "status": "no_execution_before_horizon",
                    }
                ),
                "horizon_censored": False,
            }

    for horizon in horizon_ticks:
        if horizon in results:
            continue
        record = {
            "physical": _physical_snapshot(
                totals,
                elapsed_ticks=elapsed_ticks,
                delta_t_seconds=delta_t_seconds,
                terminated=True,
            ),
            "reward_raw": reward_raw.as_dict(),
            "reward_discounted": reward_discounted.as_dict(),
            "execution": (
                execution
                if execution is not None
                else {
                    "requested_slot": requested_slot,
                    "requested_delay_ms": delay_ticks * branch.config.delta_t_ms,
                    "executed_action": None,
                    "executed_slot": None,
                    "executed_target_cell": None,
                    "executed_tic": None,
                    "actual_delay_ms": None,
                    "fallback": None,
                    "status": "trace_ended_before_execution",
                }
            ),
            "horizon_censored": True,
        }
        results[horizon] = record
    return results, simulated_ticks


def _no_worse(candidate: dict[str, Any], control: dict[str, Any]) -> bool:
    fields = (
        "handover_failures",
        "rlf_starts",
        "pure_rlf_starts",
        "ping_pong_count",
        "recovery_seconds",
        "handover_interruption_seconds",
    )
    return all(float(candidate[name]) <= float(control[name]) + 1e-12 for name in fields)


def _branch_key(row: dict[str, Any]) -> tuple[float, ...]:
    physical = row["physical"]
    return (
        float(physical["delivered_mbit"]),
        -float(physical["handover_failures"]),
        -float(physical["rlf_starts"]),
        -float(physical["ping_pong_count"]),
        -float(physical["recovery_seconds"]),
        -float(physical["handover_interruption_seconds"]),
    )


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["source"], row["horizon_seconds"])].append(row)
    summaries = []
    for (source, horizon), selected in sorted(groups.items()):
        by_state: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in selected:
            by_state[row["state_id"]].append(row)
        category_counts = Counter()
        category_state_counts = Counter()
        fallback_count = 0
        no_execution_count = 0
        positive_routes: set[tuple[int, int]] = set()
        for state_rows in by_state.values():
            control = next(
                row
                for row in state_rows
                if row["delay_ms"] == 0 and row["requested_slot"] == 1
            )
            useful = []
            useful_categories: set[str] = set()
            for row in state_rows:
                fallback_count += int(row["execution"].get("fallback") is not None)
                no_execution_count += int(row["execution"].get("executed_tic") is None)
                physical = row["physical"]
                is_useful = (
                    physical["delivered_mbit"]
                    > control["physical"]["delivered_mbit"] + 1e-9
                    and _no_worse(physical, control["physical"])
                )
                if is_useful:
                    useful.append(row)
                    positive_routes.add((row["speed_kmh"], row["trajectory_id"]))
                    if row["requested_slot"] == 1 and row["delay_ms"] > 0:
                        category_counts["timing_only"] += 1
                        useful_categories.add("timing_only")
                    elif (
                        row["requested_slot"] > 1
                        and row["delay_ms"] == 0
                        and row["execution"].get("executed_slot")
                        == row["requested_slot"]
                    ):
                        category_counts["target_only"] += 1
                        useful_categories.add("target_only")
                    elif (
                        row["requested_slot"] > 1
                        and row["delay_ms"] > 0
                        and row["execution"].get("executed_slot")
                        == row["requested_slot"]
                    ):
                        category_counts["joint"] += 1
                        useful_categories.add("joint")
            category_counts["positive_states"] += int(bool(useful))
            for category in useful_categories:
                category_state_counts[category] += 1
            best = max(state_rows, key=_branch_key)
            category_counts[f"best_delay_{best['delay_ms']}_c{best['requested_slot']}"] += 1
        summaries.append(
            {
                "source": source,
                "horizon_seconds": horizon,
                "states": len(by_state),
                "branches": len(selected),
                "strict_positive_states": category_counts["positive_states"],
                "strict_positive_state_fraction": (
                    category_counts["positive_states"] / len(by_state)
                ),
                "positive_routes": len(positive_routes),
                "strict_useful_branch_counts": {
                    name: category_counts[name]
                    for name in ("timing_only", "target_only", "joint")
                },
                "strict_positive_state_counts_by_category": {
                    name: category_state_counts[name]
                    for name in ("timing_only", "target_only", "joint")
                },
                "fallback_branches": fallback_count,
                "no_execution_branches": no_execution_count,
                "best_branch_counts": {
                    name.removeprefix("best_"): value
                    for name, value in category_counts.items()
                    if name.startswith("best_")
                },
            }
        )
    return {
        "schema_name": "d63_timing_target_counterfactual_summary",
        "schema_version": 1,
        "rows": len(rows),
        "summaries": summaries,
        "gate_g1": {
            "status": "NOT_EVALUATED_PILOT_CORRECTNESS_ONLY",
            "reason": (
                "The 5-route/speed pilot validates replay, branches and runtime; "
                "it is not the preregistered main-bank/full-route G1 analysis."
            ),
        },
    }


def audit_timing_target(
    *,
    manifest: dict[str, Any],
    arrays: dict[str, np.ndarray],
    source_specs: dict[str, dict[str, Any]],
    output_dir: Path,
    delays_ms: tuple[int, ...],
    horizons_seconds: tuple[float, ...],
    gamma_per_second: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    delta_t_ms = int(manifest["protocol_config"]["delta_t_ms"])
    if any(delay < 0 or delay % delta_t_ms for delay in delays_ms):
        raise ValueError("Every delay must be non-negative and align to delta_t_ms.")
    if 0 not in delays_ms:
        raise ValueError("D63 requires delay 0 as the C1 control.")
    horizon_ticks = tuple(
        sorted({int(round(seconds * 1000 / delta_t_ms)) for seconds in horizons_seconds})
    )
    if any(ticks <= 0 for ticks in horizon_ticks):
        raise ValueError("Every D63 horizon must be positive.")
    if {
        round(ticks * delta_t_ms / 1000.0, 12) for ticks in horizon_ticks
    } != {round(value, 12) for value in horizons_seconds}:
        raise ValueError("Every D63 horizon must align to delta_t_ms.")

    rows: list[dict[str, Any]] = []
    simulated_ticks = 0
    replayed = 0
    started = time.perf_counter()
    for bank_index, engine, builder in replay_d63_anchors(
        manifest, arrays, source_specs
    ):
        replayed += 1
        for delay_ms in sorted(set(delays_ms)):
            delay_ticks = delay_ms // delta_t_ms
            for slot in range(1, CANDIDATE_COUNT + 1):
                outcomes, ticks = simulate_timing_target(
                    engine,
                    builder,
                    delay_ticks=delay_ticks,
                    requested_slot=slot,
                    horizon_ticks=horizon_ticks,
                    gamma_per_second=gamma_per_second,
                )
                simulated_ticks += ticks
                for horizon, outcome in outcomes.items():
                    rows.append(
                        {
                            "state_id": str(arrays["state_id"][bank_index]),
                            "source": str(arrays["source"][bank_index]),
                            "speed_kmh": int(arrays["speed_kmh"][bank_index]),
                            "trajectory_id": int(arrays["trajectory_id"][bank_index]),
                            "decision_index": int(arrays["decision_index"][bank_index]),
                            "anchor_tic": int(arrays["tic"][bank_index]),
                            "anchor_is_redecision": bool(arrays["is_redecision"][bank_index]),
                            "delay_ms": delay_ms,
                            "requested_slot": slot,
                            "is_control": delay_ms == 0 and slot == 1,
                            "horizon_seconds": horizon * delta_t_ms / 1000.0,
                            **outcome,
                        }
                    )
    elapsed = time.perf_counter() - started
    expected = len(arrays["state_id"])
    if replayed != expected:
        raise RuntimeError(f"Replayed {replayed} D63 anchors, expected {expected}.")
    with (output_dir / "timing_target_outcomes.jsonl").open(
        "w", encoding="utf-8"
    ) as target:
        for row in rows:
            target.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    summary = summarize_rows(rows)
    summary.update(
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "split": "train",
            "validation_or_test_rows_read": 0,
            "state_bank_sha256": manifest["npz_sha256"],
            "delays_ms": list(sorted(set(delays_ms))),
            "horizons_seconds": list(sorted(set(horizons_seconds))),
            "continuation": "always_current_c1_after_registered_intervention",
            "reward_schema": REWARD_SCHEMA,
        }
    )
    runtime = {
        "schema_name": "d63_pilot_runtime",
        "elapsed_seconds": elapsed,
        "replayed_anchors": replayed,
        "branch_simulations": replayed * len(set(delays_ms)) * CANDIDATE_COUNT,
        "simulated_primitive_ticks": simulated_ticks,
        "simulated_ticks_per_second": (
            simulated_ticks / elapsed if elapsed > 0 else None
        ),
        "main_bank_linear_estimate_seconds": (
            elapsed * 4.0
            if int(manifest["routes_per_speed_requested"]) == 5
            else None
        ),
        "estimate_assumption": (
            "20 routes/speed is approximately four times the pilot routes; "
            "actual decisions, early termination and I/O may make scaling non-linear."
        ),
    }
    _json_write(output_dir / "counterfactual_summary.json", summary)
    _json_write(output_dir / "runtime.json", runtime)
    return summary, runtime


def run_pilot(
    *,
    root: Path,
    data_dir: str,
    output_dir: Path,
    d61_checkpoint: Path,
    d62_checkpoint: Path,
    speeds: list[int],
    routes_per_speed: int,
    max_anchors_per_source_route: int,
    split_seed: int,
    sample_seed: int,
    delays_ms: tuple[int, ...],
    horizons_seconds: tuple[float, ...],
    gamma_per_second: float,
) -> dict[str, Any]:
    _prepare_destination(output_dir)
    registry = build_registry(
        root=root,
        d61_checkpoint=d61_checkpoint,
        d62_checkpoint=d62_checkpoint,
    )
    _json_write(output_dir / "registry.json", registry)
    source_specs = _source_specs(d61_checkpoint, d62_checkpoint)
    manifest, arrays = build_d63_state_bank(
        data_dir=data_dir,
        output_dir=output_dir,
        speeds=speeds,
        routes_per_speed=routes_per_speed,
        max_anchors_per_source_route=max_anchors_per_source_route,
        split_seed=split_seed,
        sample_seed=sample_seed,
        source_specs=source_specs,
    )
    summary, runtime = audit_timing_target(
        manifest=manifest,
        arrays=arrays,
        source_specs=source_specs,
        output_dir=output_dir,
        delays_ms=delays_ms,
        horizons_seconds=horizons_seconds,
        gamma_per_second=gamma_per_second,
    )
    report = {
        "schema_name": SCHEMA,
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output_dir.resolve()),
        "state_bank": {
            "states": manifest["state_count"],
            "by_source": manifest["state_count_by_source"],
            "by_speed": manifest["state_count_by_speed"],
            "redecisions": manifest["redecision_count"],
        },
        "counterfactual": {
            "rows": summary["rows"],
            "gate_g1": summary["gate_g1"],
        },
        "runtime": runtime,
        "training_runs_started": 0,
        "test_or_zero_shot_rows_read": 0,
        "d63_2_implemented": False,
        "main_audit_started": False,
    }
    _json_write(output_dir / "pilot_report.json", report)
    return report


def main(root_path: str, args: Namespace) -> int:
    def resolve(value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else Path(root_path, path).resolve()

    report = run_pilot(
        root=Path(root_path).resolve(),
        data_dir=str(resolve(args.data_dir)),
        output_dir=resolve(args.output_dir),
        d61_checkpoint=resolve(args.d61_checkpoint),
        d62_checkpoint=resolve(args.d62_checkpoint),
        speeds=list(args.speeds),
        routes_per_speed=args.routes_per_speed,
        max_anchors_per_source_route=args.max_anchors_per_source_route,
        split_seed=args.split_seed,
        sample_seed=args.sample_seed,
        delays_ms=tuple(args.delays_ms),
        horizons_seconds=tuple(args.horizons),
        gamma_per_second=args.gamma_per_second,
    )
    print(
        f"D63 pilot: {report['state_bank']['states']} TRAIN anchors, "
        f"{report['counterfactual']['rows']} rows, "
        f"{report['runtime']['elapsed_seconds']:.1f}s. G1 not evaluated."
    )
    return 0
