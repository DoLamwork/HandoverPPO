"""Train-only D1/D2 policy-learning and counterfactual diagnostics."""

from __future__ import annotations

from argparse import Namespace
from collections import Counter, defaultdict
import copy
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn

from ho_optim_drl.agents.ppo import SMDPPPO, SMDPPPOConfig
from ho_optim_drl.commands.audit_candidates import (
    CounterfactualOutcome,
    _best_stay_or_c1,
    _is_useful,
    _select_train_routes,
)
from ho_optim_drl.data import (
    TracePair,
    discover_trace_pairs,
    filter_trace_pairs_by_speed,
    load_cell_topology,
    load_radio_trace,
    stratified_trajectory_split,
)
from ho_optim_drl.data.project import checkpoint_pairs, load_protocol_config
from ho_optim_drl.envs import HandoverSMDPEnv
from ho_optim_drl.protocol import HandoverEngine, ProtocolPhase
from ho_optim_drl.smdp import (
    LEGACY_REWARD_SCHEMA,
    P3_E07_REWARD_SCHEMA,
    P3_E08_REWARD_SCHEMA,
    P3_E09_REWARD_SCHEMA,
    RewardBreakdown,
    HandoverAction,
    TopologyCandidateObservationBuilder,
    build_reward_model,
    reward_config_from_dict,
    trajectory_reward_config,
)


_STATE_BANK_FILE = "state_bank.npz"
_MANIFEST_FILE = "state_bank_manifest.json"
_DEFAULT_SCHEMAS = ("rolling_v1", "monotonic_v1", "qos_full_v1")
_D2_REWARDS = (
    LEGACY_REWARD_SCHEMA,
    P3_E07_REWARD_SCHEMA,
    P3_E08_REWARD_SCHEMA,
    P3_E09_REWARD_SCHEMA,
)


def _sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(root: str, value: str) -> str:
    return value if os.path.isabs(value) else os.path.abspath(os.path.join(root, value))


def _parse_labelled_paths(
    root: str, specs: Iterable[str] | None
) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for spec in specs or ():
        if "=" not in spec:
            raise ValueError("Checkpoint specs must use LABEL=PATH.")
        label, raw_path = spec.split("=", 1)
        label = label.strip()
        if not label or label in parsed:
            raise ValueError(f"Invalid or duplicate checkpoint label: {label!r}.")
        path = _resolve(root, raw_path.strip())
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        parsed[label] = path
    return parsed


def _policy_spec(path: str) -> tuple[SMDPPPO, dict[str, Any], dict[str, Any]]:
    agent, extra = SMDPPPO.from_checkpoint(path, device="cpu")
    objective = dict(extra.get("objective") or {})
    spec = {
        "checkpoint": path,
        "checkpoint_sha256": _sha256_file(path),
        "observation_schema": str(
            objective.get("observation_schema_key", "legacy_v1")
        ),
        "candidate_count": int(
            objective.get("candidate_count", agent.model.action_size - 1)
        ),
        "candidate_order": str(
            objective.get("candidate_order_key", "legacy")
        ),
        "macro_steps": int(agent.total_macro_steps),
        "updates": int(agent.update_count),
    }
    return agent, extra, spec


def _selected_routes(
    *,
    data_dir: str,
    speeds: list[int] | None,
    split_seed: int,
    sample_seed: int,
    routes_per_speed: int,
    train_fraction: float,
    validation_fraction: float,
    test_fraction: float,
) -> tuple[TracePair, ...]:
    pairs = filter_trace_pairs_by_speed(
        discover_trace_pairs(data_dir), speeds
    )
    splits = stratified_trajectory_split(
        pairs,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
        seed=split_seed,
    )
    selected = _select_train_routes(
        splits.train, routes_per_speed=routes_per_speed, seed=sample_seed
    )
    if not selected:
        raise ValueError("The TRAIN state-bank route selection is empty.")
    selected_keys = {pair.key for pair in selected}
    if selected_keys & {pair.key for pair in splits.validation}:
        raise RuntimeError("State bank overlaps validation trajectories.")
    if selected_keys & {pair.key for pair in splits.test}:
        raise RuntimeError("State bank overlaps test trajectories.")
    return selected


def _make_builders(
    protocol_config: Any,
    topology: Any,
    *,
    candidate_count: int,
    candidate_order: str,
    schemas: Iterable[str],
) -> dict[str, TopologyCandidateObservationBuilder]:
    return {
        schema: TopologyCandidateObservationBuilder(
            protocol_config,
            topology=topology,
            candidate_count=candidate_count,
            candidate_order=candidate_order,
            observation_schema=schema,
        )
        for schema in schemas
    }


def _apply_policy(
    engine: HandoverEngine,
    builder: TopologyCandidateObservationBuilder,
    observation: np.ndarray,
    agent: SMDPPPO | None,
) -> int:
    action = 1 if agent is None else int(
        agent.predict(
            observation,
            deterministic=True,
            action_mask=builder.action_mask(),
        )
    )
    if action == 0:
        engine.stay()
    else:
        target = builder.candidate_cells[action - 1]
        if target is None:
            raise RuntimeError("Source policy selected an invalid candidate.")
        engine.start_handover(target)
    return action


def _collect_source_bank(
    *,
    source: str,
    source_agent: SMDPPPO | None,
    source_schema: str,
    pairs: tuple[TracePair, ...],
    data_dir: str,
    protocol_config: Any,
    topology: Any,
    candidate_count: int,
    candidate_order: str,
    schemas: tuple[str, ...],
    decision_indices: tuple[int, ...],
) -> list[dict[str, Any]]:
    if source_schema not in schemas:
        raise ValueError(
            f"Source policy requires {source_schema!r}, absent from state bank."
        )
    target_indices = set(decision_indices)
    maximum = max(decision_indices)
    states: list[dict[str, Any]] = []
    for pair in pairs:
        trace = load_radio_trace(
            protocol_config,
            data_dir,
            pair.rsrp_file,
            pair.sinr_file,
            speed_kmh=pair.speed_kmh,
        )
        engine = HandoverEngine(protocol_config)
        engine.reset(trace)
        builders = _make_builders(
            protocol_config,
            topology,
            candidate_count=candidate_count,
            candidate_order=candidate_order,
            schemas=schemas,
        )
        for builder in builders.values():
            builder.reset()
            builder.start_episode(trace)
        decision_index = 0
        while True:
            tick = engine.advance_one_tick()
            for builder in builders.values():
                builder.observe(tick)
            if tick.terminated:
                break
            if engine.decision_point is None:
                continue
            observations = {
                schema: builder.build(engine.snapshot)
                for schema, builder in builders.items()
            }
            reference_builder = builders[source_schema]
            candidate_cells = tuple(reference_builder.candidate_cells)
            for builder in builders.values():
                if tuple(builder.candidate_cells) != candidate_cells:
                    raise RuntimeError("Observation schemas changed candidate identity.")
            if decision_index in target_indices:
                decision = engine.decision_point
                states.append(
                    {
                        "state_id": (
                            f"{source}:{pair.speed_kmh}:{pair.trajectory_id}:"
                            f"{decision_index}"
                        ),
                        "source": source,
                        "speed_kmh": pair.speed_kmh,
                        "trajectory_id": pair.trajectory_id,
                        "decision_index": decision_index,
                        "tic": engine.snapshot.tic,
                        "serving_cell": decision.serving_cell,
                        "a3_target_cell": decision.target_cell,
                        "candidate_cells": candidate_cells,
                        "action_mask": reference_builder.action_mask(),
                        "observations": observations,
                    }
                )
            _apply_policy(
                engine,
                reference_builder,
                observations[source_schema],
                source_agent,
            )
            if decision_index >= maximum:
                break
            decision_index += 1
    return states


def _states_to_arrays(
    states: list[dict[str, Any]], schemas: tuple[str, ...]
) -> dict[str, np.ndarray]:
    if not states:
        raise ValueError("State bank contains no sampled decision states.")
    arrays: dict[str, np.ndarray] = {
        "state_id": np.asarray([state["state_id"] for state in states]),
        "source": np.asarray([state["source"] for state in states]),
        "speed_kmh": np.asarray(
            [state["speed_kmh"] for state in states], dtype=np.int32
        ),
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
        "candidate_cells": np.asarray(
            [
                [-1 if cell is None else int(cell) for cell in state["candidate_cells"]]
                for state in states
            ],
            dtype=np.int16,
        ),
        "action_mask": np.asarray(
            [state["action_mask"] for state in states], dtype=np.bool_
        ),
    }
    for schema in schemas:
        arrays[f"observation__{schema}"] = np.asarray(
            [state["observations"][schema] for state in states],
            dtype=np.float32,
        )
    return arrays


def build_state_bank(
    *,
    data_dir: str,
    output_dir: str,
    speeds: list[int] | None,
    split_seed: int,
    sample_seed: int,
    routes_per_speed: int,
    decision_indices: tuple[int, ...],
    candidate_count: int,
    candidate_order: str,
    schemas: tuple[str, ...],
    source_checkpoints: dict[str, str],
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
) -> dict[str, Any]:
    """Create a non-pickle TRAIN-only state bank with full provenance."""
    if not decision_indices or min(decision_indices) < 0:
        raise ValueError("Decision indices must be non-negative.")
    if len(set(decision_indices)) != len(decision_indices):
        raise ValueError("Decision indices must be unique.")
    if len(set(schemas)) != len(schemas):
        raise ValueError("Observation schemas must be unique.")
    pairs = _selected_routes(
        data_dir=data_dir,
        speeds=speeds,
        split_seed=split_seed,
        sample_seed=sample_seed,
        routes_per_speed=routes_per_speed,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
    )
    protocol_config = load_protocol_config(data_dir)
    topology = load_cell_topology(data_dir)
    states = _collect_source_bank(
        source="always_ho_c1",
        source_agent=None,
        source_schema=schemas[0],
        pairs=pairs,
        data_dir=data_dir,
        protocol_config=protocol_config,
        topology=topology,
        candidate_count=candidate_count,
        candidate_order=candidate_order,
        schemas=schemas,
        decision_indices=decision_indices,
    )
    source_specs: dict[str, Any] = {
        "always_ho_c1": {"type": "deterministic_reference"}
    }
    for label, path in source_checkpoints.items():
        agent, _, spec = _policy_spec(path)
        if spec["candidate_count"] != candidate_count:
            raise ValueError(f"{label} candidate count differs from state bank.")
        if spec["candidate_order"] != candidate_order:
            raise ValueError(f"{label} candidate order differs from state bank.")
        states.extend(
            _collect_source_bank(
                source=label,
                source_agent=agent,
                source_schema=spec["observation_schema"],
                pairs=pairs,
                data_dir=data_dir,
                protocol_config=protocol_config,
                topology=topology,
                candidate_count=candidate_count,
                candidate_order=candidate_order,
                schemas=schemas,
                decision_indices=decision_indices,
            )
        )
        source_specs[label] = {"type": "ppo_checkpoint", **spec}

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    arrays = _states_to_arrays(states, schemas)
    bank_path = destination / _STATE_BANK_FILE
    np.savez_compressed(bank_path, **arrays)
    # This load is deliberately strict: object arrays would require pickle.
    with np.load(bank_path, allow_pickle=False) as saved:
        if saved.files != list(arrays):
            raise RuntimeError("Saved state-bank field order/content changed.")

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
    feature_names = {}
    builders = _make_builders(
        protocol_config,
        topology,
        candidate_count=candidate_count,
        candidate_order=candidate_order,
        schemas=schemas,
    )
    for schema, builder in builders.items():
        feature_names[schema] = list(builder.feature_names)
    manifest = {
        "schema_name": "ppo_train_state_bank",
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "split": "train",
        "validation_or_test_rows_read": 0,
        "data_dir": os.path.abspath(data_dir),
        "dataset_provenance": (
            {
                "file": provenance_path.name,
                "sha256": _sha256_file(provenance_path),
            }
            if provenance_path.is_file()
            else None
        ),
        "protocol_config": asdict(protocol_config),
        "split_seed": split_seed,
        "sample_seed": sample_seed,
        "split_fractions": {
            "train": train_fraction,
            "validation": validation_fraction,
            "test": test_fraction,
        },
        "routes_per_speed_requested": routes_per_speed,
        "selected_routes": route_records,
        "candidate_count": candidate_count,
        "candidate_order": candidate_order,
        "decision_indices": list(decision_indices),
        "observation_schemas": list(schemas),
        "feature_names": feature_names,
        "sources": source_specs,
        "state_count": len(states),
        "state_count_by_source": dict(Counter(state["source"] for state in states)),
        "state_count_by_speed": {
            str(speed): count
            for speed, count in sorted(
                Counter(state["speed_kmh"] for state in states).items()
            )
        },
        "npz_file": bank_path.name,
        "npz_sha256": _sha256_file(bank_path),
        "npz_allow_pickle": False,
        "replay_contract": (
            "replay each source policy from route start and verify tic, serving, "
            "candidate identities, mask and observations"
        ),
    }
    (destination / _MANIFEST_FILE).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def load_state_bank(bank_dir: str) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    root = Path(bank_dir)
    manifest = json.loads((root / _MANIFEST_FILE).read_text(encoding="utf-8"))
    path = root / manifest["npz_file"]
    if _sha256_file(path) != manifest["npz_sha256"]:
        raise ValueError("State-bank SHA-256 does not match its manifest.")
    with np.load(path, allow_pickle=False) as source:
        arrays = {name: source[name].copy() for name in source.files}
    if len(arrays["state_id"]) != int(manifest["state_count"]):
        raise ValueError("State-bank manifest count does not match its arrays.")
    return manifest, arrays


def _summary(values: np.ndarray) -> dict[str, float | None]:
    if values.size == 0:
        return {name: None for name in ("mean", "std", "min", "p10", "median", "p90", "max")}
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "p10": float(np.percentile(values, 10)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "max": float(np.max(values)),
    }


def _forward_probe(
    agent: SMDPPPO, observations: np.ndarray, masks: np.ndarray
) -> dict[str, Any]:
    tensor = torch.as_tensor(observations, dtype=torch.float32, device=agent.device)
    mask_tensor = torch.as_tensor(masks, dtype=torch.bool, device=agent.device)
    current = tensor
    hidden = []
    with torch.no_grad():
        if agent.model.architecture in {
            "candidate_scorer_v1",
            "hierarchical_candidate_scorer_v1",
        }:
            distribution, values = agent.model.distribution_and_value(
                tensor, mask_tensor
            )
            probabilities = distribution.probs
            entropy = distribution.entropy()
            valid = mask_tensor.sum(-1).float()
            normalized_entropy = torch.where(
                valid > 1,
                entropy / torch.log(valid),
                torch.zeros_like(entropy),
            )
            probs = probabilities.cpu().numpy()
            sorted_probs = np.sort(probs, axis=1)
            return {
                "hidden": hidden,
                "probabilities": probs,
                "entropy": entropy.cpu().numpy(),
                "normalized_entropy": normalized_entropy.cpu().numpy(),
                "margin": sorted_probs[:, -1] - sorted_probs[:, -2],
                "argmax": np.argmax(probs, axis=1),
                "values": values.cpu().numpy(),
            }
        actor_encoder = (
            agent.model.encoder
            if agent.model.architecture == "shared_mlp_v1"
            else agent.model.actor_encoder
        )
        for layer in actor_encoder:
            current = layer(current)
            if isinstance(layer, nn.Tanh):
                hidden.append(current.cpu().numpy().copy())
        logits = agent.model.policy_head(current)
        _, values = agent.model(tensor)
        logits = logits.masked_fill(~mask_tensor, torch.finfo(logits.dtype).min)
        probabilities = torch.softmax(logits, dim=-1)
        entropy = -(probabilities * torch.log(probabilities.clamp_min(1e-12))).sum(-1)
        valid = mask_tensor.sum(-1).float()
        normalized_entropy = torch.where(
            valid > 1, entropy / torch.log(valid), torch.zeros_like(entropy)
        )
    probs = probabilities.cpu().numpy()
    sorted_probs = np.sort(probs, axis=1)
    return {
        "hidden": hidden,
        "probabilities": probs,
        "entropy": entropy.cpu().numpy(),
        "normalized_entropy": normalized_entropy.cpu().numpy(),
        "margin": sorted_probs[:, -1] - sorted_probs[:, -2],
        "argmax": np.argmax(probs, axis=1),
        "values": values.cpu().numpy(),
    }


_ACTION_SELECTION_MODES = (
    "joint_argmax",
    "hierarchical_argmax",
    "stochastic_joint",
)


def _select_action_from_probabilities(
    probabilities: np.ndarray,
    mode: str,
    *,
    rng: np.random.Generator | None = None,
) -> int:
    """Decode one masked flat distribution under an explicit audit rule."""
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("Action probabilities must be a one-dimensional vector.")
    if np.any(values < 0) or not np.isfinite(values).all():
        raise ValueError("Action probabilities must be finite and non-negative.")
    total = float(values.sum())
    if total <= 0:
        raise ValueError("Action probabilities must contain positive mass.")
    values = values / total
    if mode == "joint_argmax":
        return int(np.argmax(values))
    if mode == "hierarchical_argmax":
        stay_probability = float(values[0])
        handover_probability = float(values[1:].sum())
        if handover_probability <= stay_probability:
            return 0
        return 1 + int(np.argmax(values[1:]))
    if mode == "stochastic_joint":
        if rng is None:
            raise ValueError("stochastic_joint requires an explicit RNG.")
        return int(rng.choice(values.size, p=values))
    raise ValueError(f"Unknown action-selection mode: {mode!r}.")


def _count_actions(actions: np.ndarray, action_size: int) -> dict[str, int]:
    names = [action.name for action in list(HandoverAction)[:action_size]]
    counts = np.bincount(actions.astype(np.int64), minlength=action_size)
    return {name: int(counts[index]) for index, name in enumerate(names)}


def _selection_summary(
    probabilities: np.ndarray,
    *,
    stochastic_seeds: tuple[int, ...],
) -> dict[str, Any]:
    joint = np.argmax(probabilities, axis=1).astype(np.int64)
    handover_probability = probabilities[:, 1:].sum(axis=1)
    conditional = np.argmax(probabilities[:, 1:], axis=1).astype(np.int64) + 1
    hierarchical = np.where(
        handover_probability > probabilities[:, 0], conditional, 0
    ).astype(np.int64)
    stochastic = []
    for seed in stochastic_seeds:
        rng = np.random.default_rng(seed)
        actions = np.asarray(
            [
                _select_action_from_probabilities(
                    row, "stochastic_joint", rng=rng
                )
                for row in probabilities
            ],
            dtype=np.int64,
        )
        stochastic.append(
            {"seed": seed, "action_counts": _count_actions(actions, probabilities.shape[1])}
        )
    return {
        "states": int(len(probabilities)),
        "mean_action_probability": probabilities.mean(axis=0).tolist(),
        "handover_probability": _summary(handover_probability),
        "handover_probability_above_half_fraction": float(
            np.mean(handover_probability > 0.5)
        ),
        "joint_argmax_counts": _count_actions(joint, probabilities.shape[1]),
        "hierarchical_argmax_counts": _count_actions(
            hierarchical, probabilities.shape[1]
        ),
        "conditional_candidate_argmax_counts": _count_actions(
            conditional, probabilities.shape[1]
        ),
        "joint_vs_hierarchical_disagreement_fraction": float(
            np.mean(joint != hierarchical)
        ),
        "stochastic_joint": stochastic,
    }


def _useful_action_selection_summary(
    *,
    counterfactual_dir: str,
    state_ids: np.ndarray,
    probabilities: np.ndarray,
    horizon_seconds: float,
    expected_bank_sha256: str,
    stochastic_seeds: tuple[int, ...],
) -> dict[str, Any]:
    root = Path(counterfactual_dir)
    metadata = json.loads(
        (root / "counterfactual_summary.json").read_text(encoding="utf-8")
    )
    if metadata.get("state_bank_sha256") != expected_bank_sha256:
        raise ValueError("Counterfactual labels do not match the state bank.")
    useful_by_state: dict[str, tuple[int, ...]] = {}
    with (root / "counterfactual_rows.jsonl").open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            if np.isclose(float(row["horizon_seconds"]), horizon_seconds):
                useful_by_state[str(row["state_id"])] = tuple(
                    int(value)
                    for value in row["useful_slots_vs_best_stay_c1"]
                )
    if len(useful_by_state) != len(state_ids):
        raise ValueError("Counterfactual labels do not cover every state-bank row.")

    useful = [useful_by_state[str(state_id)] for state_id in state_ids]
    positive = np.asarray([bool(slots) for slots in useful], dtype=np.bool_)
    joint = np.argmax(probabilities, axis=1).astype(np.int64)
    handover_probability = probabilities[:, 1:].sum(axis=1)
    conditional = np.argmax(probabilities[:, 1:], axis=1).astype(np.int64) + 1
    hierarchical = np.where(
        handover_probability > probabilities[:, 0], conditional, 0
    ).astype(np.int64)

    def recall(actions: np.ndarray) -> float | None:
        if not positive.any():
            return None
        return float(
            np.mean(
                [
                    int(actions[index]) in useful[index]
                    for index in np.flatnonzero(positive)
                ]
            )
        )

    useful_mass = np.asarray(
        [sum(probabilities[index, slot] for slot in slots) for index, slots in enumerate(useful)],
        dtype=np.float64,
    )
    stochastic = []
    for seed in stochastic_seeds:
        rng = np.random.default_rng(seed)
        actions = np.asarray(
            [
                _select_action_from_probabilities(
                    row, "stochastic_joint", rng=rng
                )
                for row in probabilities
            ],
            dtype=np.int64,
        )
        stochastic.append({"seed": seed, "recall": recall(actions)})
    return {
        "horizon_seconds": horizon_seconds,
        "positive_states": int(positive.sum()),
        "mean_probability_on_useful_alternatives_given_positive_state": (
            float(useful_mass[positive].mean()) if positive.any() else None
        ),
        "joint_argmax_recall": recall(joint),
        "hierarchical_argmax_recall": recall(hierarchical),
        "conditional_candidate_argmax_recall": recall(conditional),
        "stochastic_joint_recall": stochastic,
    }


def _action_selection_environment(
    *,
    data_dir: str,
    checkpoint_path: str,
    agent: SMDPPPO,
    extra: dict[str, Any],
    speeds: list[int] | None,
    split_seed: int,
) -> tuple[HandoverSMDPEnv, tuple[TracePair, ...]]:
    objective = dict(extra.get("objective") or {})
    selected = checkpoint_pairs(
        data_dir,
        checkpoint_path,
        extra,
        split="validation",
        speeds=speeds,
        split_seed=split_seed,
    )
    reward_schema = str(objective.get("reward", LEGACY_REWARD_SCHEMA))
    env = HandoverSMDPEnv(
        load_protocol_config(data_dir, extra),
        data_dir=data_dir,
        trace_pairs=selected,
        gamma_per_second=agent.config.gamma_per_second,
        candidate_count=int(
            objective.get("candidate_count", agent.model.action_size - 1)
        ),
        candidate_order=str(objective.get("candidate_order_key", "legacy")),
        observation_schema=str(
            objective.get("observation_schema_key", "legacy_v1")
        ),
        decision_trigger=str(objective.get("trigger", "completed_a3_ttt")),
        reward_schema=reward_schema,
        reward_config=reward_config_from_dict(
            reward_schema, objective.get("reward_config")
        ),
        stay_mode=str(objective.get("stay_mode", "reject_event_v1")),
        stay_redecision_ms=int(objective.get("stay_redecision_ms", 100)),
    )
    env.load_reward_state(extra.get("reward_state"))
    env.validate_checkpoint_objective(extra)
    return env, selected


def _validation_selection_run(
    *,
    agent: SMDPPPO,
    env: HandoverSMDPEnv,
    mode: str,
    seed: int | None,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed) if seed is not None else None
    probabilities_seen: list[np.ndarray] = []
    selected_actions: list[int] = []

    def selector(observation: np.ndarray, action_mask: np.ndarray) -> int:
        _, _, _, probabilities = agent.model.act_with_probabilities(
            observation,
            device=agent.device,
            deterministic=True,
            action_mask=action_mask,
        )
        action = _select_action_from_probabilities(probabilities, mode, rng=rng)
        probabilities_seen.append(probabilities)
        selected_actions.append(action)
        return action

    metrics = agent.evaluate(env, action_selector=selector)
    probabilities = np.asarray(probabilities_seen, dtype=np.float64)
    selection = _selection_summary(
        probabilities,
        stochastic_seeds=(),
    )
    selection["selected_action_counts"] = _count_actions(
        np.asarray(selected_actions), agent.model.action_size
    )
    return {
        "mode": mode,
        "seed": seed,
        "metrics": asdict(metrics),
        "decision_diagnostics": selection,
    }


def audit_action_selection(
    *,
    data_dir: str,
    bank_dir: str,
    counterfactual_dir: str | None,
    output_dir: str,
    checkpoint_label: str,
    checkpoint_path: str,
    speeds: list[int] | None,
    split_seed: int,
    stochastic_seeds: tuple[int, ...],
    horizon_seconds: float,
) -> dict[str, Any]:
    if not stochastic_seeds:
        raise ValueError("At least one stochastic seed is required.")
    manifest, arrays = load_state_bank(bank_dir)
    agent, extra, spec = _policy_spec(checkpoint_path)
    if agent.model.architecture != "hierarchical_candidate_scorer_v1":
        raise ValueError("Action-selection audit requires a hierarchical checkpoint.")
    observation_key = f"observation__{spec['observation_schema']}"
    if observation_key not in arrays:
        raise ValueError("State bank lacks the checkpoint observation schema.")
    probe = _forward_probe(agent, arrays[observation_key], arrays["action_mask"])
    state_bank = _selection_summary(
        probe["probabilities"], stochastic_seeds=stochastic_seeds
    )
    if counterfactual_dir is not None:
        state_bank["useful_alternatives"] = _useful_action_selection_summary(
            counterfactual_dir=counterfactual_dir,
            state_ids=arrays["state_id"],
            probabilities=probe["probabilities"],
            horizon_seconds=horizon_seconds,
            expected_bank_sha256=manifest["npz_sha256"],
            stochastic_seeds=stochastic_seeds,
        )

    validation_runs = []
    for mode, seed in (
        ("joint_argmax", None),
        ("hierarchical_argmax", None),
        *[("stochastic_joint", value) for value in stochastic_seeds],
    ):
        env, selected = _action_selection_environment(
            data_dir=data_dir,
            checkpoint_path=checkpoint_path,
            agent=agent,
            extra=extra,
            speeds=speeds,
            split_seed=split_seed,
        )
        validation_runs.append(
            _validation_selection_run(
                agent=agent, env=env, mode=mode, seed=seed
            )
        )

    stochastic_metrics = [
        run["metrics"]
        for run in validation_runs
        if run["mode"] == "stochastic_joint"
    ]
    aggregate_fields = (
        "mean_return",
        "mean_estimated_rate_mbps",
        "mean_relative_rate_percent",
        "rlf_time_percent",
        "handover_failure_percent",
        "ping_pong_percent",
        "handover_fraction",
        "stay_fraction",
    )
    stochastic_aggregate = {
        name: {
            "mean": float(np.mean([row[name] for row in stochastic_metrics])),
            "std": float(np.std([row[name] for row in stochastic_metrics])),
        }
        for name in aggregate_fields
    }
    report = {
        "schema_name": "ppo_action_selection_audit",
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_label": checkpoint_label,
        "checkpoint": checkpoint_path,
        "checkpoint_sha256": spec["checkpoint_sha256"],
        "network_architecture": agent.model.architecture,
        "split": "validation",
        "split_seed": split_seed,
        "trace_ids": [
            f"{pair.speed_kmh}kmh_{pair.trajectory_id}" for pair in selected
        ],
        "test_or_zero_shot_rows_read": 0,
        "state_bank": state_bank,
        "validation_runs": validation_runs,
        "stochastic_validation_aggregate": stochastic_aggregate,
    }
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "action_selection_audit.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def _probe_group(
    outputs: dict[str, Any],
    observations: np.ndarray,
    indices: np.ndarray,
    feature_names: list[str],
) -> dict[str, Any]:
    obs = observations[indices]
    probabilities = outputs["probabilities"][indices]
    argmax = outputs["argmax"][indices]
    hidden = [values[indices] for values in outputs["hidden"]]
    return {
        "states": int(len(indices)),
        "input": {
            "fraction_abs_ge_0_95": float(np.mean(np.abs(obs) >= 0.95)),
            "fraction_abs_ge_0_99": float(np.mean(np.abs(obs) >= 0.99)),
            "constant_feature_fraction": float(np.mean(np.std(obs, axis=0) <= 1e-8)),
            "features": [
                {
                    "name": name,
                    "mean": float(np.mean(obs[:, feature_index])),
                    "std": float(np.std(obs[:, feature_index])),
                    "min": float(np.min(obs[:, feature_index])),
                    "max": float(np.max(obs[:, feature_index])),
                    "fraction_abs_ge_0_95": float(
                        np.mean(np.abs(obs[:, feature_index]) >= 0.95)
                    ),
                    "fraction_abs_ge_0_99": float(
                        np.mean(np.abs(obs[:, feature_index]) >= 0.99)
                    ),
                }
                for feature_index, name in enumerate(feature_names)
            ],
        },
        "hidden_layers": [
            {
                "layer": layer_index,
                "fraction_abs_ge_0_95": float(np.mean(np.abs(values) >= 0.95)),
                "fraction_abs_ge_0_99": float(np.mean(np.abs(values) >= 0.99)),
                "activation": _summary(values.reshape(-1)),
            }
            for layer_index, values in enumerate(hidden, start=1)
        ],
        "policy": {
            "entropy": _summary(outputs["entropy"][indices]),
            "normalized_entropy": _summary(outputs["normalized_entropy"][indices]),
            "top1_top2_margin": _summary(outputs["margin"][indices]),
            "argmax_counts": {
                str(action): int(count)
                for action, count in sorted(Counter(argmax.tolist()).items())
            },
            "mean_action_probability": probabilities.mean(axis=0).tolist(),
        },
        "value": _summary(outputs["values"][indices]),
    }


def probe_state_bank(
    *,
    bank_dir: str,
    output_dir: str,
    checkpoints: dict[str, str],
    include_fresh: bool,
    seed: int = 17,
) -> dict[str, Any]:
    manifest, arrays = load_state_bank(bank_dir)
    policies: list[tuple[str, SMDPPPO, dict[str, Any]]] = []
    for label, path in checkpoints.items():
        agent, _, spec = _policy_spec(path)
        policies.append((label, agent, spec))
    if include_fresh:
        for schema in manifest["observation_schemas"]:
            observation_size = arrays[f"observation__{schema}"].shape[1]
            agent = SMDPPPO(
                observation_size,
                int(manifest["candidate_count"]) + 1,
                SMDPPPOConfig(seed=seed),
            )
            policies.append(
                (
                    f"fresh_{schema}_s{seed}",
                    agent,
                    {
                        "checkpoint": None,
                        "checkpoint_sha256": None,
                        "observation_schema": schema,
                        "candidate_count": int(manifest["candidate_count"]),
                        "candidate_order": manifest["candidate_order"],
                        "macro_steps": 0,
                        "updates": 0,
                    },
                )
            )
    if not policies:
        raise ValueError("Probe requires at least one checkpoint or --include-fresh.")

    records = []
    for label, agent, spec in policies:
        schema = spec["observation_schema"]
        field = f"observation__{schema}"
        if field not in arrays:
            raise ValueError(f"State bank does not contain schema {schema!r}.")
        observations = arrays[field]
        if observations.shape[1] != agent.model.observation_size:
            raise ValueError(f"{label} observation size does not match state bank.")
        outputs = _forward_probe(agent, observations, arrays["action_mask"])
        feature_names = list(manifest["feature_names"][schema])
        groups: dict[str, np.ndarray] = {
            "overall:all": np.arange(len(observations))
        }
        for source in sorted(set(arrays["source"].tolist())):
            groups[f"source:{source}"] = np.flatnonzero(arrays["source"] == source)
        for speed in sorted(set(arrays["speed_kmh"].tolist())):
            groups[f"speed_kmh:{speed}"] = np.flatnonzero(
                arrays["speed_kmh"] == speed
            )
        records.append(
            {
                "label": label,
                **spec,
                "groups": {
                    group: _probe_group(
                        outputs, observations, indices, feature_names
                    )
                    for group, indices in groups.items()
                },
            }
        )

    report = {
        "schema_name": "ppo_state_bank_probe",
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "state_bank_manifest": str(Path(bank_dir, _MANIFEST_FILE).resolve()),
        "state_bank_sha256": manifest["npz_sha256"],
        "split": "train",
        "validation_or_test_rows_read": 0,
        "policies": records,
    }
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "probe_summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# D1 policy learning probe",
        "",
        "TRAIN-only fixed-state comparison; no validation/test/zero-shot rows were read.",
        "",
        "| Policy | Schema | step | hidden-2 |p|≥.99 | entropy norm. | STAY | C1 | C2+ |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for record in records:
        overall = record["groups"]["overall:all"]
        layers = overall["hidden_layers"]
        saturation = layers[-1]["fraction_abs_ge_0_99"] if layers else 0.0
        probabilities = overall["policy"]["mean_action_probability"]
        lines.append(
            f"| {record['label']} | {record['observation_schema']} | "
            f"{record['macro_steps']} | {100*saturation:.2f}% | "
            f"{overall['policy']['normalized_entropy']['mean']:.3f} | "
            f"{100*probabilities[0]:.2f}% | {100*probabilities[1]:.2f}% | "
            f"{100*sum(probabilities[2:]):.2f}% |"
        )
    (destination / "probe_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return report


def _average_precision(labels: np.ndarray, scores: np.ndarray) -> float | None:
    positives = int(labels.sum())
    if positives == 0:
        return None
    order = np.argsort(-scores, kind="stable")
    ranked = labels[order].astype(np.float64)
    precision = np.cumsum(ranked) / np.arange(1, len(ranked) + 1)
    return float(np.sum(precision * ranked) / positives)


def _binary_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    predictions = scores >= 0.5
    positives = labels.astype(bool)
    true_positive = int(np.sum(predictions & positives))
    predicted_positive = int(np.sum(predictions))
    actual_positive = int(np.sum(positives))
    return {
        "rows": int(len(labels)),
        "positive_rows": actual_positive,
        "prevalence": float(np.mean(positives)),
        "average_precision": _average_precision(labels, scores),
        "precision_at_0_5": (
            true_positive / predicted_positive if predicted_positive else None
        ),
        "recall_at_0_5": (
            true_positive / actual_positive if actual_positive else None
        ),
    }


def _route_bootstrap_average_precision(
    labels: np.ndarray,
    scores: np.ndarray,
    route_keys: np.ndarray,
    *,
    seed: int,
    samples: int = 1000,
) -> dict[str, Any]:
    routes = np.unique(route_keys)
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(samples):
        sampled_routes = routes[rng.integers(0, len(routes), size=len(routes))]
        sampled_indices = np.concatenate(
            [np.flatnonzero(route_keys == route) for route in sampled_routes]
        )
        estimate = _average_precision(labels[sampled_indices], scores[sampled_indices])
        if estimate is not None:
            estimates.append(estimate)
    values = np.asarray(estimates, dtype=np.float64)
    return {
        "unit": "route",
        "requested_samples": samples,
        "valid_samples": int(len(values)),
        "p2_5": float(np.percentile(values, 2.5)) if len(values) else None,
        "median": float(np.median(values)) if len(values) else None,
        "p97_5": float(np.percentile(values, 97.5)) if len(values) else None,
    }


def _route_folds(
    arrays: dict[str, np.ndarray], *, seed: int, holdout_fraction: float = 0.30
) -> tuple[np.ndarray, np.ndarray, dict[str, str]]:
    """Split unique TRAIN routes by speed; every source/state stays together."""
    rng = np.random.default_rng(seed)
    route_to_speed: dict[str, int] = {}
    for speed, trajectory in zip(arrays["speed_kmh"], arrays["trajectory_id"]):
        route_to_speed[f"{int(speed)}:{int(trajectory)}"] = int(speed)
    folds: dict[str, str] = {}
    for speed in sorted(set(route_to_speed.values())):
        routes = sorted(
            route for route, route_speed in route_to_speed.items()
            if route_speed == speed
        )
        shuffled = list(np.asarray(routes)[rng.permutation(len(routes))])
        holdout_count = max(1, int(round(len(routes) * holdout_fraction)))
        holdout_count = min(holdout_count, max(1, len(routes) - 1))
        holdout = set(shuffled[:holdout_count])
        folds.update(
            {route: ("holdout" if route in holdout else "fit") for route in routes}
        )
    row_routes = np.asarray(
        [
            f"{int(speed)}:{int(trajectory)}"
            for speed, trajectory in zip(
                arrays["speed_kmh"], arrays["trajectory_id"]
            )
        ]
    )
    fit = np.asarray([folds[route] == "fit" for route in row_routes])
    return fit, ~fit, folds


def _fit_linear_probe(
    features: np.ndarray,
    labels: np.ndarray,
    fit_mask: np.ndarray,
    *,
    classes: int,
    seed: int,
    epochs: int = 800,
) -> np.ndarray:
    """Fit a deterministic class-balanced linear probe and return probabilities."""
    torch.manual_seed(seed)
    x = torch.as_tensor(features, dtype=torch.float32)
    fit = torch.as_tensor(fit_mask, dtype=torch.bool)
    mean = x[fit].mean(dim=0)
    std = x[fit].std(dim=0, unbiased=False).clamp_min(1e-6)
    x = (x - mean) / std
    output_size = 1 if classes == 2 else classes
    model = nn.Linear(features.shape[1], output_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02, weight_decay=1e-4)
    y = torch.as_tensor(labels, dtype=torch.long)
    fit_labels = y[fit]
    if classes == 2:
        positives = int((fit_labels == 1).sum().item())
        negatives = int((fit_labels == 0).sum().item())
        if not positives or not negatives:
            raise ValueError("Binary probe fit fold must contain both classes.")
        loss_fn = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([negatives / positives], dtype=torch.float32)
        )
    else:
        counts = torch.bincount(fit_labels, minlength=classes).float()
        weights = torch.where(counts > 0, counts.sum() / counts, torch.zeros_like(counts))
        loss_fn = nn.CrossEntropyLoss(weight=weights)
    for _ in range(epochs):
        logits = model(x[fit])
        loss = (
            loss_fn(logits[:, 0], fit_labels.float())
            if classes == 2
            else loss_fn(logits, fit_labels)
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        logits = model(x)
        probabilities = (
            torch.sigmoid(logits[:, 0])
            if classes == 2
            else torch.softmax(logits, dim=1)
        )
    return probabilities.cpu().numpy()


def supervised_observation_probe(
    *,
    bank_dir: str,
    counterfactual_dir: str,
    output_dir: str,
    observation_schema: str,
    reward_schema: str,
    horizon_seconds: float,
    route_split_seed: int = 29,
) -> dict[str, Any]:
    """Test whether current observations linearly identify useful candidates."""
    manifest, arrays = load_state_bank(bank_dir)
    field = f"observation__{observation_schema}"
    if field not in arrays:
        raise ValueError(f"State bank does not contain {observation_schema!r}.")
    rows_path = Path(counterfactual_dir, "counterfactual_rows.jsonl")
    rows_by_id: dict[str, dict[str, Any]] = {}
    with rows_path.open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            if np.isclose(float(row["horizon_seconds"]), horizon_seconds):
                rows_by_id[row["state_id"]] = row
    selected_indices = np.asarray(
        [index for index, state_id in enumerate(arrays["state_id"]) if state_id in rows_by_id],
        dtype=np.int64,
    )
    if not len(selected_indices):
        raise ValueError("No state-bank rows match the requested horizon audit.")
    selected_arrays = {name: values[selected_indices] for name, values in arrays.items()}
    rows = [rows_by_id[str(state_id)] for state_id in selected_arrays["state_id"]]
    features = selected_arrays[field]
    fit_mask, holdout_mask, folds = _route_folds(
        selected_arrays, seed=route_split_seed
    )
    route_keys = np.asarray(
        [
            f"{int(speed)}:{int(trajectory)}"
            for speed, trajectory in zip(
                selected_arrays["speed_kmh"], selected_arrays["trajectory_id"]
            )
        ]
    )
    available_names = ["STAY"] + [
        f"C{index}" for index in range(1, int(manifest["candidate_count"]) + 1)
    ]
    best_actions = []
    for row in rows:
        rewards = {
            name: row["actions"][name]["reward_discounted"][reward_schema]["total"]
            for name in available_names if name in row["actions"]
        }
        best_actions.append(available_names.index(max(rewards, key=rewards.get)))
    multiclass_labels = np.asarray(best_actions, dtype=np.int64)
    multi_probabilities = _fit_linear_probe(
        features,
        multiclass_labels,
        fit_mask,
        classes=len(available_names),
        seed=route_split_seed,
    )
    binary_labels = {
        "any_useful_C2plus": np.asarray(
            [bool(row["useful_slots_vs_best_stay_c1"]) for row in rows], dtype=np.int64
        )
    }
    for action_index in range(2, int(manifest["candidate_count"]) + 1):
        binary_labels[f"C{action_index}_useful"] = np.asarray(
            [action_index in row["useful_slots_vs_best_stay_c1"] for row in rows],
            dtype=np.int64,
        )
    binary_reports: dict[str, Any] = {}
    for offset, (label, values) in enumerate(binary_labels.items()):
        fit_positives = int(values[fit_mask].sum())
        fit_negatives = int(fit_mask.sum()) - fit_positives
        if not fit_positives or not fit_negatives:
            binary_reports[label] = {
                "status": "insufficient_fit_classes",
                "fit_positive_rows": fit_positives,
            }
            continue
        scores = _fit_linear_probe(
            features,
            values,
            fit_mask,
            classes=2,
            seed=route_split_seed + offset + 1,
        )
        binary_reports[label] = {
            "status": "fit",
            "fit": _binary_metrics(values[fit_mask], scores[fit_mask]),
            "route_holdout": _binary_metrics(
                values[holdout_mask], scores[holdout_mask]
            ),
            "route_holdout_by_source": {
                str(source): _binary_metrics(
                    values[
                        holdout_mask & (selected_arrays["source"] == source)
                    ],
                    scores[
                        holdout_mask & (selected_arrays["source"] == source)
                    ],
                )
                for source in sorted(set(selected_arrays["source"].tolist()))
            },
            "route_bootstrap_average_precision_95_interval": (
                _route_bootstrap_average_precision(
                    values[holdout_mask],
                    scores[holdout_mask],
                    route_keys[holdout_mask],
                    seed=route_split_seed + offset + 101,
                )
            ),
        }
    per_class = {}
    for class_index, class_name in enumerate(available_names):
        labels = (multiclass_labels == class_index).astype(np.int64)
        per_class[class_name] = _binary_metrics(
            labels[holdout_mask], multi_probabilities[holdout_mask, class_index]
        )
    predicted = np.argmax(multi_probabilities[holdout_mask], axis=1)
    actual = multiclass_labels[holdout_mask]
    report = {
        "schema_name": "ppo_observation_supervised_probe",
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "split": "train_route_fit_holdout",
        "validation_or_test_rows_read": 0,
        "state_bank_sha256": manifest["npz_sha256"],
        "counterfactual_rows_sha256": _sha256_file(rows_path),
        "observation_schema": observation_schema,
        "reward_schema": reward_schema,
        "horizon_seconds": horizon_seconds,
        "route_split_seed": route_split_seed,
        "rows": int(len(rows)),
        "fit_rows": int(fit_mask.sum()),
        "holdout_rows": int(holdout_mask.sum()),
        "route_folds": folds,
        "multiclass_reward_best_action": {
            "classes": available_names,
            "holdout_accuracy": float(np.mean(predicted == actual)),
            "holdout_per_class": per_class,
        },
        "binary_useful_candidate": binary_reports,
    }
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "supervised_probe_summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# TRAIN route-split supervised observation probe",
        "",
        f"Schema: `{observation_schema}`; horizon: {horizon_seconds:.2f}s; reward: `{reward_schema}`.",
        "",
        "| Label | fit positives | holdout positives | holdout AP | precision@.5 | recall@.5 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, record in binary_reports.items():
        if record["status"] != "fit":
            lines.append(f"| {label} | {record['fit_positive_rows']} | — | — | — | — |")
            continue
        fit_metrics = record["fit"]
        holdout = record["route_holdout"]
        def cell(value: float | None) -> str:
            return "—" if value is None else f"{value:.3f}"
        lines.append(
            f"| {label} | {fit_metrics['positive_rows']} | {holdout['positive_rows']} | "
            f"{cell(holdout['average_precision'])} | {cell(holdout['precision_at_0_5'])} | "
            f"{cell(holdout['recall_at_0_5'])} |"
        )
    (destination / "supervised_probe_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return report


def _signal_group(
    values: np.ndarray,
    probabilities_pre: np.ndarray,
    probabilities_post: np.ndarray,
    *,
    indices: np.ndarray,
    action: int,
    include_advantage: bool = True,
) -> dict[str, Any]:
    if not len(indices):
        return {"rows": 0}
    delta = probabilities_post[indices, action] - probabilities_pre[indices, action]
    record: dict[str, Any] = {
        "rows": int(len(indices)),
        "mean_probability_pre": float(np.mean(probabilities_pre[indices, action])),
        "mean_probability_post": float(np.mean(probabilities_post[indices, action])),
        "mean_probability_delta": float(np.mean(delta)),
    }
    if include_advantage:
        selected = values[indices]
        record.update(
            normalized_advantage=_summary(selected),
            positive_advantage_fraction=float(np.mean(selected > 0)),
        )
    return record


def transition_signal_probe(
    *,
    data_dir: str,
    output_dir: str,
    checkpoints: dict[str, str],
    include_fresh: bool,
    speeds: list[int] | None,
    split_seed: int,
    rollout_steps: int,
    horizon_seconds: float,
) -> dict[str, Any]:
    """Join same-transition counterfactual labels, GAE and PPO probability deltas."""
    if not checkpoints:
        raise ValueError("Transition probe requires at least one checkpoint.")
    pairs = filter_trace_pairs_by_speed(discover_trace_pairs(data_dir), speeds)
    splits = stratified_trajectory_split(pairs, seed=split_seed)
    protocol_config = load_protocol_config(data_dir)
    horizon_ticks = int(round(horizon_seconds * 1000 / protocol_config.delta_t_ms))
    if not np.isclose(
        horizon_ticks * protocol_config.delta_t_ms / 1000.0, horizon_seconds
    ):
        raise ValueError("Transition horizon must align to delta_t_ms.")

    loaded: list[tuple[str, SMDPPPO, dict[str, Any], dict[str, Any]]] = []
    first_agent: SMDPPPO | None = None
    first_extra: dict[str, Any] | None = None
    for label, path in checkpoints.items():
        agent, extra = SMDPPPO.from_checkpoint(
            path, device="cpu", load_optimizer=True
        )
        spec = _policy_spec(path)[2]
        loaded.append((label, agent, extra, spec))
        if first_agent is None:
            first_agent, first_extra = agent, extra
    if include_fresh:
        assert first_agent is not None and first_extra is not None
        fresh_config = replace(
            first_agent.config,
            seed=17,
            rollout_steps=rollout_steps,
            learning_diagnostics=True,
        )
        fresh = SMDPPPO(
            first_agent.model.observation_size,
            first_agent.model.action_size,
            fresh_config,
        )
        objective = dict(first_extra.get("objective") or {})
        loaded.insert(
            0,
            (
                "fresh_s17",
                fresh,
                {"objective": objective, "reward_state": None},
                {
                    "checkpoint": None,
                    "checkpoint_sha256": None,
                    "observation_schema": objective["observation_schema_key"],
                    "candidate_count": int(objective["candidate_count"]),
                    "candidate_order": objective["candidate_order_key"],
                    "macro_steps": 0,
                    "updates": 0,
                },
            ),
        )

    policy_reports = []
    row_output: list[dict[str, Any]] = []
    for label, agent, extra, spec in loaded:
        objective = dict(extra.get("objective") or {})
        reward_schema = str(objective.get("reward", P3_E07_REWARD_SCHEMA))
        reward_config = (
            None
            if reward_schema == LEGACY_REWARD_SCHEMA
            else trajectory_reward_config(reward_schema)
        )
        stay_mode = str(objective.get("stay_mode", "reject_event_v1"))
        stay_redecision_ms = int(
            objective.get("stay_redecision_ms", 100)
        )
        env = HandoverSMDPEnv(
            protocol_config,
            data_dir=data_dir,
            trace_pairs=splits.train,
            gamma_per_second=agent.config.gamma_per_second,
            candidate_count=spec["candidate_count"],
            candidate_order=spec["candidate_order"],
            observation_schema=spec["observation_schema"],
            decision_trigger=str(
                objective.get("trigger", "completed_a3_ttt")
            ),
            reward_schema=reward_schema,
            reward_config=reward_config,
            stay_mode=stay_mode,
            stay_redecision_ms=stay_redecision_ms,
        )
        env.load_reward_state(extra.get("reward_state"))

        def capture(
            callback_env: HandoverSMDPEnv,
            _observation: np.ndarray,
            _mask: np.ndarray,
            _action: int,
            _probabilities: np.ndarray,
        ) -> dict[str, Any]:
            builder = callback_env.transition_builder.observation_builder
            targets = [None, *builder.candidate_cells]
            outcomes = {
                action: simulate_reward_counterfactual(
                    callback_env.engine,
                    initial_target=target,
                    horizon_ticks=(horizon_ticks,),
                    gamma_per_second=agent.config.gamma_per_second,
                    reward_schemas=(reward_schema,),
                    stay_redecision_ticks=(
                        stay_redecision_ms // protocol_config.delta_t_ms
                        if stay_mode == "a3_redecision_v1"
                        else None
                    ),
                )[horizon_ticks]
                for action, target in enumerate(targets)
                if action == 0 or target is not None
            }
            physical = {
                action: _as_outcome(result["physical"])
                for action, result in outcomes.items()
            }
            basic = _best_stay_or_c1(physical[0], physical[1])
            useful = [
                action
                for action in range(2, agent.model.action_size)
                if action in physical and _is_useful(physical[action], basic)
            ]
            return {
                "tic": int(callback_env.engine.snapshot.tic),
                "candidate_cells": [
                    -1 if cell is None else int(cell)
                    for cell in builder.candidate_cells
                ],
                "useful_slots": useful,
                "reward_best_action": int(
                    max(
                        outcomes,
                        key=lambda action: outcomes[action]["reward_discounted"]
                        [reward_schema]["total"],
                    )
                ),
            }

        buffer, _ = agent.collect_rollout(
            env,
            rollout_steps=rollout_steps,
            reset_seed=17,
            transition_diagnostic_callback=capture,
        )
        tensors = buffer.to_tensors(
            agent.device,
            normalize_advantage=agent.config.normalize_advantage,
        )
        with torch.no_grad():
            pre_distribution, _ = agent.model.distribution_and_value(
                tensors.observations, tensors.action_masks
            )
            probabilities_pre = pre_distribution.probs.cpu().numpy()
        normalized_advantages = tensors.advantages.cpu().numpy()
        actions = tensors.actions.cpu().numpy()
        agent.update(buffer)
        with torch.no_grad():
            post_distribution, _ = agent.model.distribution_and_value(
                tensors.observations, tensors.action_masks
            )
            probabilities_post = post_distribution.probs.cpu().numpy()

        metadata = list(buffer.transition_metadata[: buffer.size])
        useful_masks = {
            action: np.asarray(
                [action in item["useful_slots"] for item in metadata], dtype=bool
            )
            for action in range(2, agent.model.action_size)
        }
        action_reports = {}
        for action in range(2, agent.model.action_size):
            chosen = actions == action
            action_reports[f"C{action}"] = {
                "useful_chosen": _signal_group(
                    normalized_advantages,
                    probabilities_pre,
                    probabilities_post,
                    indices=np.flatnonzero(chosen & useful_masks[action]),
                    action=action,
                ),
                "nonuseful_chosen": _signal_group(
                    normalized_advantages,
                    probabilities_pre,
                    probabilities_post,
                    indices=np.flatnonzero(chosen & ~useful_masks[action]),
                    action=action,
                ),
                "useful_states_probability": _signal_group(
                    normalized_advantages,
                    probabilities_pre,
                    probabilities_post,
                    indices=np.flatnonzero(useful_masks[action]),
                    action=action,
                    include_advantage=False,
                ),
            }
        for index, item in enumerate(metadata):
            row_output.append(
                {
                    "policy": label,
                    "trace_id": str(buffer.trace_ids[index]),
                    **item,
                    "chosen_action": int(actions[index]),
                    "chosen_is_useful": bool(actions[index] in item["useful_slots"]),
                    "normalized_advantage": float(normalized_advantages[index]),
                    "probabilities_pre": probabilities_pre[index].tolist(),
                    "probabilities_post": probabilities_post[index].tolist(),
                }
            )
        policy_reports.append(
            {
                "label": label,
                **spec,
                "rollout_steps": rollout_steps,
                "positive_states": int(
                    sum(bool(item["useful_slots"]) for item in metadata)
                ),
                "actions": action_reports,
            }
        )

    report = {
        "schema_name": "ppo_same_transition_signal_probe",
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "split": "train",
        "validation_or_test_rows_read": 0,
        "horizon_seconds": horizon_seconds,
        "rollout_steps": rollout_steps,
        "policies": policy_reports,
    }
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "transition_signal_summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    with (destination / "transition_signal_rows.jsonl").open(
        "w", encoding="utf-8"
    ) as target:
        for row in row_output:
            target.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    return report


def _outcome_snapshot(
    totals: dict[str, float | int],
    *,
    elapsed_ticks: int,
    delta_t_seconds: float,
    terminated: bool,
) -> CounterfactualOutcome:
    return CounterfactualOutcome(
        delivered_mbit=float(totals["delivered_mbit"]),
        handover_failures=int(totals["handover_failures"]),
        rlf_count=int(totals["rlf_count"]),
        ping_pong_count=int(totals["ping_pong_count"]),
        recovery_seconds=float(totals["recovery_ticks"]) * delta_t_seconds,
        elapsed_seconds=elapsed_ticks * delta_t_seconds,
        trace_terminated=terminated,
    )


def simulate_reward_counterfactual(
    engine: HandoverEngine,
    *,
    initial_target: int | None,
    horizon_ticks: tuple[int, ...],
    reward_schemas: tuple[str, ...] = _D2_REWARDS,
    gamma_per_second: float = 0.99,
    stay_redecision_ticks: int | None = None,
) -> dict[int, dict[str, Any]]:
    """Clone one decision and audit physical plus reward outcomes.

    Reward models are reset with the immutable full trace, matching training.
    Both raw and SMDP-discounted sums start with exponent zero at the first
    primitive tick after the audited action.
    """
    if engine.decision_point is None or engine.trace is None:
        raise RuntimeError("Counterfactual requires a pending decision.")
    if not horizon_ticks or min(horizon_ticks) <= 0:
        raise ValueError("Counterfactual horizons must be positive ticks.")
    trace = engine.trace
    branch = copy.deepcopy(engine, {id(trace): trace})
    if initial_target is None:
        branch.stay(redecision_delay_ticks=stay_redecision_ticks)
    else:
        branch.start_handover(initial_target)
    models = {}
    for schema in reward_schemas:
        config = None if schema == LEGACY_REWARD_SCHEMA else trajectory_reward_config(schema)
        model = build_reward_model(branch.config, schema, config)
        model.reset(trace)
        model.begin_rollout()
        models[schema] = model

    delta_t_seconds = branch.config.delta_t_ms / 1000.0
    physical: dict[str, float | int] = {
        "delivered_mbit": 0.0,
        "handover_failures": 0,
        "rlf_count": 0,
        "ping_pong_count": 0,
        "recovery_ticks": 0,
    }
    raw = {schema: RewardBreakdown() for schema in reward_schemas}
    discounted = {schema: RewardBreakdown() for schema in reward_schemas}
    results: dict[int, dict[str, Any]] = {}
    elapsed_ticks = 0
    terminated = False
    next_decision_ticks: int | None = None
    while elapsed_ticks < max(horizon_ticks) and not terminated:
        if branch.decision_point is not None:
            if next_decision_ticks is None:
                next_decision_ticks = elapsed_ticks
            branch.start_handover()
        tick = branch.advance_one_tick()
        exponent_seconds = elapsed_ticks * delta_t_seconds
        discount = gamma_per_second**exponent_seconds
        elapsed_ticks += 1
        physical["delivered_mbit"] += tick.delivered_rate_mbps * delta_t_seconds
        physical["handover_failures"] += int(tick.handover_failed)
        physical["rlf_count"] += int(tick.rlf_started)
        physical["ping_pong_count"] += int(tick.ping_pong_detected)
        physical["recovery_ticks"] += int(
            tick.snapshot.phase is ProtocolPhase.RLF_RECOVERY
        )
        for schema, model in models.items():
            tick_reward = model.compute(tick)
            raw[schema] = raw[schema] + tick_reward
            discounted[schema] = discounted[schema] + tick_reward.scaled(discount)
        terminated = tick.terminated
        if elapsed_ticks in horizon_ticks:
            results[elapsed_ticks] = {
                "physical": asdict(
                    _outcome_snapshot(
                        physical,
                        elapsed_ticks=elapsed_ticks,
                        delta_t_seconds=delta_t_seconds,
                        terminated=terminated,
                    )
                ),
                "reward_raw": {
                    schema: value.as_dict() for schema, value in raw.items()
                },
                "reward_discounted": {
                    schema: value.as_dict()
                    for schema, value in discounted.items()
                },
                "time_to_next_decision_seconds": (
                    None
                    if next_decision_ticks is None
                    else next_decision_ticks * delta_t_seconds
                ),
            }
    for horizon in horizon_ticks:
        if horizon not in results:
            results[horizon] = {
                "physical": asdict(
                    _outcome_snapshot(
                        physical,
                        elapsed_ticks=elapsed_ticks,
                        delta_t_seconds=delta_t_seconds,
                        terminated=True,
                    )
                ),
                "reward_raw": {
                    schema: value.as_dict() for schema, value in raw.items()
                },
                "reward_discounted": {
                    schema: value.as_dict()
                    for schema, value in discounted.items()
                },
                "time_to_next_decision_seconds": (
                    None
                    if next_decision_ticks is None
                    else next_decision_ticks * delta_t_seconds
                ),
            }
    return results


def _as_outcome(record: dict[str, Any]) -> CounterfactualOutcome:
    return CounterfactualOutcome(**record)


def _replay_anchors(
    manifest: dict[str, Any], arrays: dict[str, np.ndarray]
) -> Iterable[tuple[int, HandoverEngine]]:
    data_dir = manifest["data_dir"]
    protocol_config = load_protocol_config(data_dir)
    topology = load_cell_topology(data_dir)
    schemas = tuple(manifest["observation_schemas"])
    route_lookup = {
        (int(item["speed_kmh"]), int(item["trajectory_id"])): item
        for item in manifest["selected_routes"]
    }
    groups: dict[tuple[str, int, int], list[int]] = defaultdict(list)
    for index in range(len(arrays["state_id"])):
        groups[
            (
                str(arrays["source"][index]),
                int(arrays["speed_kmh"][index]),
                int(arrays["trajectory_id"][index]),
            )
        ].append(index)
    for (source, speed, trajectory), indices in sorted(groups.items()):
        source_spec = manifest["sources"][source]
        agent = None
        source_schema = schemas[0]
        if source_spec["type"] == "ppo_checkpoint":
            agent, _, loaded = _policy_spec(source_spec["checkpoint"])
            if loaded["checkpoint_sha256"] != source_spec["checkpoint_sha256"]:
                raise ValueError("Source checkpoint changed after bank creation.")
            source_schema = loaded["observation_schema"]
        route = route_lookup[(speed, trajectory)]
        pair = TracePair(
            speed,
            trajectory,
            route["rsrp_file"],
            route["sinr_file"],
        )
        if _sha256_file(Path(data_dir, pair.rsrp_file)) != route["rsrp_sha256"]:
            raise ValueError("RSRP route changed after bank creation.")
        if _sha256_file(Path(data_dir, pair.sinr_file)) != route["sinr_sha256"]:
            raise ValueError("SINR route changed after bank creation.")
        trace = load_radio_trace(
            protocol_config,
            data_dir,
            pair.rsrp_file,
            pair.sinr_file,
            speed_kmh=speed,
        )
        engine = HandoverEngine(protocol_config)
        engine.reset(trace)
        builders = _make_builders(
            protocol_config,
            topology,
            candidate_count=int(manifest["candidate_count"]),
            candidate_order=manifest["candidate_order"],
            schemas=schemas,
        )
        for builder in builders.values():
            builder.reset()
            builder.start_episode(trace)
        by_decision = {int(arrays["decision_index"][i]): i for i in indices}
        decision_index = 0
        while by_decision:
            tick = engine.advance_one_tick()
            for builder in builders.values():
                builder.observe(tick)
            if tick.terminated:
                break
            if engine.decision_point is None:
                continue
            observations = {
                schema: builder.build(engine.snapshot)
                for schema, builder in builders.items()
            }
            bank_index = by_decision.pop(decision_index, None)
            if bank_index is not None:
                reference = builders[source_schema]
                expected_cells = arrays["candidate_cells"][bank_index]
                actual_cells = np.asarray(
                    [-1 if cell is None else cell for cell in reference.candidate_cells]
                )
                if (
                    int(arrays["tic"][bank_index]) != engine.snapshot.tic
                    or int(arrays["serving_cell"][bank_index])
                    != engine.decision_point.serving_cell
                    or not np.array_equal(expected_cells, actual_cells)
                    or not np.array_equal(
                        arrays["action_mask"][bank_index], reference.action_mask()
                    )
                ):
                    raise RuntimeError("State-bank replay anchor mismatch.")
                for schema, observation in observations.items():
                    np.testing.assert_allclose(
                        arrays[f"observation__{schema}"][bank_index],
                        observation,
                        rtol=0,
                        atol=1e-7,
                    )
                yield bank_index, engine
            _apply_policy(
                engine,
                builders[source_schema],
                observations[source_schema],
                agent,
            )
            decision_index += 1
        if by_decision:
            raise RuntimeError("Could not replay every state-bank anchor.")


def counterfactual_state_bank(
    *,
    bank_dir: str,
    output_dir: str,
    checkpoints: dict[str, str],
    horizons_seconds: tuple[float, ...],
    gamma_per_second: float = 0.99,
) -> dict[str, Any]:
    manifest, arrays = load_state_bank(bank_dir)
    delta_t_ms = int(manifest["protocol_config"]["delta_t_ms"])
    horizon_ticks = tuple(
        sorted({int(round(seconds * 1000 / delta_t_ms)) for seconds in horizons_seconds})
    )
    if any(
        not np.isclose(ticks * delta_t_ms / 1000.0, seconds)
        for ticks, seconds in zip(horizon_ticks, sorted(set(horizons_seconds)))
    ):
        raise ValueError("Every counterfactual horizon must align to delta_t_ms.")
    policies = {
        label: _policy_spec(path) for label, path in checkpoints.items()
    }
    rows = []
    for bank_index, engine in _replay_anchors(manifest, arrays):
        targets = [None] + [
            int(cell) if int(cell) >= 0 else None
            for cell in arrays["candidate_cells"][bank_index]
        ]
        outcomes = {
            action: simulate_reward_counterfactual(
                engine,
                initial_target=target,
                horizon_ticks=horizon_ticks,
                gamma_per_second=gamma_per_second,
            )
            for action, target in enumerate(targets)
            if action == 0 or target is not None
        }
        probabilities = {}
        for label, (agent, _, spec) in policies.items():
            observation = arrays[f"observation__{spec['observation_schema']}"][bank_index]
            _, _, value, probs = agent.model.act_with_probabilities(
                observation,
                device=agent.device,
                deterministic=True,
                action_mask=arrays["action_mask"][bank_index],
            )
            probabilities[label] = {
                "probabilities": probs.tolist(),
                "argmax": int(np.argmax(probs)),
                "value": value,
            }
        for horizon in horizon_ticks:
            physical = {
                action: _as_outcome(result[horizon]["physical"])
                for action, result in outcomes.items()
            }
            c1 = physical[1]
            best_basic = _best_stay_or_c1(physical[0], c1)
            useful_vs_c1 = [
                action
                for action in range(2, int(manifest["candidate_count"]) + 1)
                if action in physical and _is_useful(physical[action], c1)
            ]
            useful_vs_best = [
                action
                for action in range(2, int(manifest["candidate_count"]) + 1)
                if action in physical and _is_useful(physical[action], best_basic)
            ]
            row = {
                "state_id": str(arrays["state_id"][bank_index]),
                "source": str(arrays["source"][bank_index]),
                "speed_kmh": int(arrays["speed_kmh"][bank_index]),
                "trajectory_id": int(arrays["trajectory_id"][bank_index]),
                "decision_index": int(arrays["decision_index"][bank_index]),
                "tic": int(arrays["tic"][bank_index]),
                "horizon_seconds": horizon * delta_t_ms / 1000.0,
                "candidate_cells": arrays["candidate_cells"][bank_index].tolist(),
                "useful_slots_vs_c1": useful_vs_c1,
                "useful_slots_vs_best_stay_c1": useful_vs_best,
                "actions": {
                    ("STAY" if action == 0 else f"C{action}"): result[horizon]
                    for action, result in outcomes.items()
                },
                "policies": probabilities,
            }
            rows.append(row)

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / "counterfactual_rows.jsonl").open(
        "w", encoding="utf-8"
    ) as target:
        for row in rows:
            target.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    summary_rows = []
    for horizon in sorted({row["horizon_seconds"] for row in rows}):
        selected = [row for row in rows if row["horizon_seconds"] == horizon]
        record: dict[str, Any] = {
            "horizon_seconds": horizon,
            "states": len(selected),
            "useful_alternative_vs_c1_states": sum(
                bool(row["useful_slots_vs_c1"]) for row in selected
            ),
            "useful_alternative_vs_best_stay_c1_states": sum(
                bool(row["useful_slots_vs_best_stay_c1"]) for row in selected
            ),
            "policies": {},
            "reward_alignment": {},
        }
        for label in policies:
            useful_probability_all_states = []
            useful_probability_positive_states = []
            recall = []
            for row in selected:
                useful = row["useful_slots_vs_best_stay_c1"]
                probs = row["policies"][label]["probabilities"]
                probability = sum(probs[action] for action in useful)
                useful_probability_all_states.append(probability)
                if useful:
                    useful_probability_positive_states.append(probability)
                    recall.append(row["policies"][label]["argmax"] in useful)
            record["policies"][label] = {
                "mean_probability_on_useful_alternatives_all_states": float(
                    np.mean(useful_probability_all_states)
                ),
                "mean_probability_on_useful_alternatives_given_positive_state": (
                    float(np.mean(useful_probability_positive_states))
                    if useful_probability_positive_states
                    else None
                ),
                "argmax_recall_on_positive_states": (
                    float(np.mean(recall)) if recall else None
                ),
            }
        for schema in _D2_REWARDS:
            aligned = 0
            raw_discount_rank_changes = 0
            positive_states = 0
            reward_recognized_positive_states = 0
            useful_action_count = 0
            useful_action_reward_positive = 0
            false_alternative_argmax = 0
            for row in selected:
                action_records = row["actions"]
                action_names = list(action_records)
                physical_best = max(
                    action_names,
                    key=lambda name: (
                        action_records[name]["physical"]["delivered_mbit"],
                        -action_records[name]["physical"]["handover_failures"],
                        -action_records[name]["physical"]["rlf_count"],
                        -action_records[name]["physical"]["ping_pong_count"],
                        -action_records[name]["physical"]["recovery_seconds"],
                    ),
                )
                raw_best = max(
                    action_names,
                    key=lambda name: action_records[name]["reward_raw"][schema]["total"],
                )
                discounted_best = max(
                    action_names,
                    key=lambda name: action_records[name]["reward_discounted"][schema]["total"],
                )
                aligned += int(discounted_best == physical_best)
                raw_discount_rank_changes += int(raw_best != discounted_best)
                basic_reward = max(
                    action_records[name]["reward_discounted"][schema]["total"]
                    for name in ("STAY", "C1")
                )
                useful = row["useful_slots_vs_best_stay_c1"]
                if useful:
                    positive_states += 1
                    recognized = False
                    for action in useful:
                        useful_action_count += 1
                        reward_is_positive = (
                            action_records[f"C{action}"]["reward_discounted"][schema]["total"]
                            > basic_reward + 1e-12
                        )
                        useful_action_reward_positive += int(reward_is_positive)
                        recognized = recognized or reward_is_positive
                    reward_recognized_positive_states += int(recognized)
                if discounted_best.startswith("C") and discounted_best != "C1":
                    selected_action = int(discounted_best[1:])
                    false_alternative_argmax += int(selected_action not in useful)
            record["reward_alignment"][schema] = {
                "discounted_argmax_matches_physical": aligned / len(selected),
                "raw_vs_discounted_argmax_change": (
                    raw_discount_rank_changes / len(selected)
                ),
                "recognizes_any_useful_action_given_positive_state": (
                    reward_recognized_positive_states / positive_states
                    if positive_states
                    else None
                ),
                "useful_action_positive_reward_fraction": (
                    useful_action_reward_positive / useful_action_count
                    if useful_action_count
                    else None
                ),
                "false_alternative_argmax_fraction": (
                    false_alternative_argmax / len(selected)
                ),
            }
        stay_next = [
            row["actions"]["STAY"]["time_to_next_decision_seconds"]
            for row in selected
        ]
        c1_next = [
            row["actions"]["C1"]["time_to_next_decision_seconds"]
            for row in selected
        ]
        record["stay_suppression"] = {
            "stay_no_next_decision_fraction": float(
                np.mean([value is None for value in stay_next])
            ),
            "c1_no_next_decision_fraction": float(
                np.mean([value is None for value in c1_next])
            ),
            "stay_mean_time_to_next_decision_seconds": (
                float(np.mean([value for value in stay_next if value is not None]))
                if any(value is not None for value in stay_next)
                else None
            ),
            "c1_mean_time_to_next_decision_seconds": (
                float(np.mean([value for value in c1_next if value is not None]))
                if any(value is not None for value in c1_next)
                else None
            ),
        }
        summary_rows.append(record)
    report = {
        "schema_name": "ppo_same_state_counterfactual_audit",
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "split": "train",
        "validation_or_test_rows_read": 0,
        "state_bank_sha256": manifest["npz_sha256"],
        "gamma_per_second": gamma_per_second,
        "continuation_policy": "always_ho_c1_after_initial_action",
        "reward_schemas": list(_D2_REWARDS),
        "rows": len(rows),
        "summary": summary_rows,
    }
    (destination / "counterfactual_summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# D2 same-state counterfactual audit",
        "",
        "TRAIN-only replay with frozen candidates and Always-HO-C1 continuation.",
        "",
        "| Horizon | states | useful vs C1 | useful vs best(STAY,C1) |",
        "|---:|---:|---:|---:|",
    ]
    for record in summary_rows:
        lines.append(
            f"| {record['horizon_seconds']:.2f}s | {record['states']} | "
            f"{record['useful_alternative_vs_c1_states']} | "
            f"{record['useful_alternative_vs_best_stay_c1_states']} |"
        )
    (destination / "counterfactual_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return report


def main(root_path: str, args: Namespace) -> int:
    if args.mode == "build-bank":
        manifest = build_state_bank(
            data_dir=_resolve(root_path, args.data_dir),
            output_dir=_resolve(root_path, args.output_dir),
            speeds=args.speeds,
            split_seed=args.split_seed,
            sample_seed=args.sample_seed,
            routes_per_speed=args.routes_per_speed,
            decision_indices=tuple(sorted(args.decision_indices)),
            candidate_count=args.candidate_count,
            candidate_order=args.candidate_order,
            schemas=tuple(args.observation_schemas),
            source_checkpoints=_parse_labelled_paths(
                root_path, args.source_checkpoint
            ),
            train_fraction=args.train_fraction,
            validation_fraction=args.validation_fraction,
            test_fraction=args.test_fraction,
        )
        print(
            f"Built TRAIN state bank: {manifest['state_count']} states -> "
            f"{_resolve(root_path, args.output_dir)}"
        )
        return 0
    if args.mode == "probe":
        report = probe_state_bank(
            bank_dir=_resolve(root_path, args.bank_dir),
            output_dir=_resolve(root_path, args.output_dir),
            checkpoints=_parse_labelled_paths(root_path, args.checkpoint),
            include_fresh=args.include_fresh,
            seed=args.seed,
        )
        print(f"Probed {len(report['policies'])} policies on the TRAIN state bank.")
        return 0
    if args.mode == "counterfactual":
        report = counterfactual_state_bank(
            bank_dir=_resolve(root_path, args.bank_dir),
            output_dir=_resolve(root_path, args.output_dir),
            checkpoints=_parse_labelled_paths(root_path, args.checkpoint),
            horizons_seconds=tuple(args.horizons),
            gamma_per_second=args.gamma_per_second,
        )
        print(f"Audited {report['rows']} TRAIN state/horizon records.")
        return 0
    if args.mode == "supervised-probe":
        report = supervised_observation_probe(
            bank_dir=_resolve(root_path, args.bank_dir),
            counterfactual_dir=_resolve(root_path, args.counterfactual_dir),
            output_dir=_resolve(root_path, args.output_dir),
            observation_schema=args.probe_observation_schema,
            reward_schema=args.probe_reward_schema,
            horizon_seconds=args.probe_horizon,
            route_split_seed=args.route_split_seed,
        )
        print(
            f"Fit TRAIN route-split probes on {report['rows']} states "
            f"({report['holdout_rows']} route-holdout rows)."
        )
        return 0
    if args.mode == "transition-probe":
        report = transition_signal_probe(
            data_dir=_resolve(root_path, args.data_dir),
            output_dir=_resolve(root_path, args.output_dir),
            checkpoints=_parse_labelled_paths(root_path, args.checkpoint),
            include_fresh=args.include_fresh,
            speeds=args.speeds,
            split_seed=args.split_seed,
            rollout_steps=args.rollout_probe_steps,
            horizon_seconds=args.transition_horizon,
        )
        print(
            f"Audited same-transition signals for {len(report['policies'])} "
            "policies on TRAIN."
        )
        return 0
    if args.mode == "action-selection":
        checkpoints = _parse_labelled_paths(root_path, args.checkpoint)
        label, checkpoint = next(iter(checkpoints.items()))
        report = audit_action_selection(
            data_dir=_resolve(root_path, args.data_dir),
            bank_dir=_resolve(root_path, args.bank_dir),
            counterfactual_dir=(
                None
                if args.counterfactual_dir is None
                else _resolve(root_path, args.counterfactual_dir)
            ),
            output_dir=_resolve(root_path, args.output_dir),
            checkpoint_label=label,
            checkpoint_path=checkpoint,
            speeds=args.speeds,
            split_seed=args.split_seed,
            stochastic_seeds=tuple(args.stochastic_seeds),
            horizon_seconds=args.selection_horizon,
        )
        print(
            "Audited action selection on "
            f"{report['state_bank']['states']} TRAIN-bank states and "
            f"{len(report['trace_ids'])} validation trajectories."
        )
        return 0
    raise ValueError(f"Unknown audit mode: {args.mode!r}.")
