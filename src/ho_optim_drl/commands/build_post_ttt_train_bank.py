"""Build an outcome-blind, post-TTT-only bank from previously unused TRAIN routes."""

from __future__ import annotations

from argparse import Namespace
from collections import Counter, defaultdict
import copy
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Iterable

from ho_optim_drl.commands.audit_baseline_parity import _git_state, _sha256
from ho_optim_drl.commands.audit_early_timing_bank import (
    SOURCES,
    _anchor_record,
    _apply_source_action,
    _category,
    _score,
    _selected_route_records,
)
from ho_optim_drl.commands.audit_early_timing_references import simulate_schedule
from ho_optim_drl.data import (
    RadioTrace,
    TracePair,
    discover_trace_pairs,
    filter_trace_pairs_by_speed,
    load_radio_trace,
    stratified_trajectory_split,
)
from ho_optim_drl.data.project import load_protocol_config
from ho_optim_drl.protocol import EarlyTimingHandoverEngine


SCHEMA_NAME = "early_ho_post_ttt_train_bank"
SCHEMA_VERSION = 1
RISK_FIELDS = (
    "handover_failures",
    "ping_pong_count",
    "rlf_starts",
    "recovery_seconds",
    "interruption_seconds",
)


def lock_confirmation_route_split(
    *, root: Path, data_dir: Path, output_dir: Path, speeds: list[int],
    split_seed: int, route_seed: int, sample_seed: int,
    exclude_route_splits: list[Path], confirmation_per_speed: int,
) -> dict[str, Any]:
    """Freeze an outcome-blind TRAIN confirmation split without simulating outcomes."""
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite locked route split: {output_dir}")
    if confirmation_per_speed <= 0:
        raise ValueError("At least one confirmation route per speed is required.")
    excluded = excluded_route_keys(exclude_route_splits)
    pairs, folds, available = select_unused_train_routes(
        discover_trace_pairs(str(data_dir)), speeds=speeds, split_seed=split_seed,
        route_seed=route_seed, excluded=excluded, fit_per_speed=0,
        tuning_per_speed=0, confirmation_per_speed=confirmation_per_speed,
    )
    route_records = _selected_route_records(data_dir, pairs)
    for row in route_records:
        row["fold"] = folds[(int(row["speed_kmh"]), int(row["trajectory_id"]))]
    route_split = {
        "schema_name": "early_ho_post_ttt_train_route_split",
        "schema_version": 1,
        "split": "train",
        "validation_or_test_rows_read": 0,
        "split_seed": split_seed,
        "route_seed": route_seed,
        "sample_seed": sample_seed,
        "speeds": speeds,
        "fold_counts_per_speed": {
            "fit": 0, "tuning": 0,
            "confirmation": confirmation_per_speed,
        },
        "excluded_route_count": len(excluded),
        "unused_train_routes_available_by_speed": available,
        "selection_contract": (
            "Outcome-blind SHA-256 ordering; split frozen before full-route outcomes."
        ),
        "selected_routes": route_records,
    }
    output_dir.mkdir(parents=True)
    split_path = output_dir / "route_split.json"
    split_path.write_text(
        json.dumps(route_split, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema_name": "early_ho_locked_confirmation_split",
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).astimezone().isoformat(),
        "validation_or_test_rows_read": 0,
        "route_split": {"path": str(split_path.resolve()), "sha256": _sha256(split_path)},
        "excluded_route_splits": [
            {"path": str(path.resolve()), "sha256": _sha256(path)}
            for path in exclude_route_splits
        ],
        "selected_route_count": len(route_records),
        "selected_route_count_by_speed": dict(Counter(
            int(row["speed_kmh"]) for row in route_records
        )),
        "git": _git_state(root),
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (output_dir / "report_vi.md").write_text(
        "\n".join([
            "# Locked post-TTT TRAIN confirmation split", "",
            f"- Routes: **{len(route_records)}** ({confirmation_per_speed}/speed).",
            "- Validation/test rows read: **0**.",
            f"- Excluded unique routes: **{len(excluded)}**.",
            "- Selection is outcome-blind and was frozen before full-route audit.",
            "- This directory contains no simulated outcomes.",
        ]) + "\n",
        encoding="utf-8",
    )
    return manifest


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _route_key(pair: TracePair) -> tuple[int, int]:
    return int(pair.speed_kmh), int(pair.trajectory_id)


def _hash_rank(seed: int, speed: int, trajectory: int, purpose: str) -> int:
    value = f"{seed}|{purpose}|{speed}|{trajectory}"
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest(), 16)


def excluded_route_keys(paths: Iterable[Path]) -> set[tuple[int, int]]:
    excluded: set[tuple[int, int]] = set()
    for path in paths:
        payload = _read_json(path)
        if payload.get("split") != "train" or payload.get("validation_or_test_rows_read") != 0:
            raise ValueError(f"Excluded split is not TRAIN-only: {path}")
        for row in payload["selected_routes"]:
            excluded.add((int(row["speed_kmh"]), int(row["trajectory_id"])))
    return excluded


def select_unused_train_routes(
    pairs: Iterable[TracePair], *, speeds: list[int], split_seed: int,
    route_seed: int, excluded: set[tuple[int, int]], fit_per_speed: int,
    tuning_per_speed: int, confirmation_per_speed: int,
) -> tuple[list[TracePair], dict[tuple[int, int], str], dict[int, int]]:
    filtered = filter_trace_pairs_by_speed(tuple(pairs), speeds)
    splits = stratified_trajectory_split(
        filtered, train_fraction=0.70, validation_fraction=0.15,
        test_fraction=0.15, seed=split_seed,
    )
    requested = fit_per_speed + tuning_per_speed + confirmation_per_speed
    by_speed: dict[int, list[TracePair]] = defaultdict(list)
    for pair in splits.train:
        if _route_key(pair) not in excluded:
            by_speed[int(pair.speed_kmh)].append(pair)
    selected: list[TracePair] = []
    folds: dict[tuple[int, int], str] = {}
    available: dict[int, int] = {}
    for speed in speeds:
        group = sorted(
            by_speed.get(int(speed), []),
            key=lambda pair: (
                _hash_rank(route_seed, *_route_key(pair), "route-selection"),
                int(pair.trajectory_id),
            ),
        )
        available[int(speed)] = len(group)
        if len(group) < requested:
            raise ValueError(
                f"Speed {speed} has {len(group)} unused TRAIN routes; "
                f"{requested} are required."
            )
        chosen = group[:requested]
        for index, pair in enumerate(chosen):
            fold = (
                "fit" if index < fit_per_speed else
                "tuning" if index < fit_per_speed + tuning_per_speed else
                "confirmation"
            )
            folds[_route_key(pair)] = fold
        selected.extend(chosen)
    selected.sort(key=lambda pair: (int(pair.speed_kmh), int(pair.trajectory_id)))
    return selected, folds, available


def select_spaced_candidates(
    candidates: list[dict[str, Any]], *, max_anchors: int,
    min_separation_ticks: int,
) -> list[dict[str, Any]]:
    """Select outcome-blind hash-ranked anchors with temporal de-correlation."""
    selected: list[dict[str, Any]] = []
    for candidate in sorted(candidates, key=lambda item: int(item["score"])):
        tic = int(candidate["record"]["tic"])
        if all(
            abs(tic - int(existing["record"]["tic"])) >= min_separation_ticks
            for existing in selected
        ):
            selected.append(candidate)
            if len(selected) == max_anchors:
                break
    return sorted(selected, key=lambda item: int(item["record"]["tic"]))


def collect_post_ttt_source_anchors(
    config: Any, trace: RadioTrace, pair: TracePair, *, source: str,
    sample_seed: int, max_anchors: int, min_separation_ticks: int,
) -> tuple[list[dict[str, Any]], int]:
    engine = EarlyTimingHandoverEngine(
        config, enable_early_decisions=True, enable_shadow_decisions=True
    )
    engine.reset(trace)
    candidates: list[dict[str, Any]] = []
    available = 0
    while True:
        if engine.decision_point is not None:
            category = _category(engine)
            if category == "post_ttt":
                available += 1
                record = _anchor_record(
                    engine, trace, pair, source=source, category=category
                )
                candidates.append({
                    "record": record,
                    "engine": copy.deepcopy(engine, {id(trace): trace}),
                    "score": _score(sample_seed, record),
                })
            _apply_source_action(engine, source)
        if engine.advance_one_tick().terminated:
            break
    return select_spaced_candidates(
        candidates, max_anchors=max_anchors,
        min_separation_ticks=min_separation_ticks,
    ), available


def _strict_wait_label(now: dict[str, Any], wait: dict[str, Any]) -> dict[str, Any]:
    risks = {field: float(wait[field]) - float(now[field]) for field in RISK_FIELDS}
    gain = float(wait["delivered_mbit"]) - float(now["delivered_mbit"])
    return {
        "wait_useful": bool(gain > 1e-12 and all(value <= 1e-12 for value in risks.values())),
        "delta_delivered_mbit": gain,
        "risk_deltas": risks,
    }


def _positive_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    # Match T4 de-duplication: one deterministic state per physical fingerprint.
    unique: dict[str, dict[str, Any]] = {}
    for row in sorted(rows, key=lambda item: (item["fingerprint"], item["state_id"])):
        unique.setdefault(str(row["fingerprint"]), row)
    values = list(unique.values())
    result: dict[str, Any] = {
        "definition": "WAIT20-minus-HO_NOW at 1.0 s; data gain > 1e-12 and no risk increase",
        "raw_rows": len(rows),
        "duplicate_physical_states_removed": len(rows) - len(values),
        "unique_rows": len(values),
        "positive_rows": sum(bool(row["label"]["wait_useful"]) for row in values),
    }
    for field in ("fold", "speed_kmh", "source"):
        groups: dict[str, dict[str, int]] = {}
        for value in sorted({row[field] for row in values}, key=str):
            group = [row for row in values if row[field] == value]
            groups[str(value)] = {
                "rows": len(group),
                "positive_rows": sum(bool(row["label"]["wait_useful"]) for row in group),
            }
        result[f"by_{field}"] = groups
    return result


def run_bank(
    *, root: Path, data_dir: Path, output_dir: Path, speeds: list[int],
    split_seed: int, route_seed: int, sample_seed: int,
    exclude_route_splits: list[Path], fit_per_speed: int,
    tuning_per_speed: int, confirmation_per_speed: int,
    max_anchors_per_source_route: int, min_anchor_separation_ms: int,
    delays_ms: tuple[int, ...], horizons_seconds: tuple[float, ...],
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite post-TTT bank: {output_dir}")
    if min(fit_per_speed, tuning_per_speed, confirmation_per_speed) < 0:
        raise ValueError("Route fold counts cannot be negative.")
    if fit_per_speed + tuning_per_speed + confirmation_per_speed <= 0:
        raise ValueError("At least one route per speed is required.")
    config = load_protocol_config(str(data_dir))
    if min_anchor_separation_ms < 0 or min_anchor_separation_ms % config.delta_t_ms:
        raise ValueError("Anchor separation must be non-negative and align with dt.")
    if any(ms < 0 or ms % config.delta_t_ms for ms in delays_ms):
        raise ValueError("Delays must be non-negative and align with dt.")
    if 0 not in delays_ms or 20 not in delays_ms or 1.0 not in horizons_seconds:
        raise ValueError("Positive accounting requires delays 0/20 ms and horizon 1.0 s.")
    excluded = excluded_route_keys(exclude_route_splits)
    pairs, folds, available = select_unused_train_routes(
        discover_trace_pairs(str(data_dir)), speeds=speeds, split_seed=split_seed,
        route_seed=route_seed, excluded=excluded, fit_per_speed=fit_per_speed,
        tuning_per_speed=tuning_per_speed,
        confirmation_per_speed=confirmation_per_speed,
    )
    route_records = _selected_route_records(data_dir, pairs)
    for row in route_records:
        row["fold"] = folds[(int(row["speed_kmh"]), int(row["trajectory_id"]))]
    route_split = {
        "schema_name": "early_ho_post_ttt_train_route_split",
        "schema_version": 1,
        "split": "train",
        "validation_or_test_rows_read": 0,
        "split_seed": split_seed,
        "route_seed": route_seed,
        "sample_seed": sample_seed,
        "speeds": speeds,
        "fold_counts_per_speed": {
            "fit": fit_per_speed, "tuning": tuning_per_speed,
            "confirmation": confirmation_per_speed,
        },
        "excluded_route_count": len(excluded),
        "unused_train_routes_available_by_speed": available,
        "selected_routes": route_records,
    }
    output_dir.mkdir(parents=True)
    (output_dir / "route_split.json").write_text(
        json.dumps(route_split, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    horizon_ticks = tuple(round(value * 1000 / config.delta_t_ms) for value in horizons_seconds)
    delay_ticks = tuple(value // config.delta_t_ms for value in delays_ms)
    separation_ticks = min_anchor_separation_ms // config.delta_t_ms
    started = time.perf_counter()
    source_counts = Counter(); shortage_rows = []; fingerprints = Counter()
    anchor_count = 0; outcome_count = 0; simulated_ticks = 0
    label_rows: list[dict[str, Any]] = []
    with (
        (output_dir / "anchors.jsonl").open("w", encoding="utf-8") as anchor_stream,
        (output_dir / "local_outcomes.jsonl").open("w", encoding="utf-8") as outcome_stream,
    ):
        for route_index, pair in enumerate(pairs, 1):
            route = _route_key(pair); fold = folds[route]
            trace = load_radio_trace(
                config, str(data_dir), pair.rsrp_file, pair.sinr_file,
                speed_kmh=pair.speed_kmh,
            )
            for source in SOURCES:
                anchors, available_count = collect_post_ttt_source_anchors(
                    config, trace, pair, source=source, sample_seed=sample_seed,
                    max_anchors=max_anchors_per_source_route,
                    min_separation_ticks=separation_ticks,
                )
                shortage_rows.append({
                    "trace_id": f"{pair.speed_kmh}:{pair.trajectory_id}",
                    "fold": fold, "source": source,
                    "post_ttt_available": available_count,
                    "selected": len(anchors),
                    "shortage": max_anchors_per_source_route - len(anchors),
                })
                for item in anchors:
                    record = item["record"]; record["fold"] = fold
                    engine = item["engine"]
                    anchor_stream.write(json.dumps(record, allow_nan=False) + "\n")
                    anchor_count += 1; source_counts[source] += 1
                    fingerprints[str(record["physical_state_fingerprint"])] += 1
                    primary: dict[int, dict[str, Any]] = {}
                    for horizon, ticks in zip(horizons_seconds, horizon_ticks):
                        control, used = simulate_schedule(
                            engine, requested_delay_ticks=None,
                            horizon_ticks=ticks, mode="early",
                        )
                        simulated_ticks += used
                        base = {
                            "state_id": record["state_id"], "source": source,
                            "category": "post_ttt", "fold": fold,
                            "speed_kmh": record["speed_kmh"],
                            "trajectory_id": record["trajectory_id"],
                            "horizon_seconds": horizon,
                        }
                        outcome_stream.write(json.dumps({
                            **base, "branch": "baseline_continuation",
                            "requested_delay_ms": None, "physical": control,
                        }, allow_nan=False) + "\n")
                        outcome_count += 1
                        for delay_ms, delay in zip(delays_ms, delay_ticks):
                            outcome, used = simulate_schedule(
                                engine, requested_delay_ticks=delay,
                                horizon_ticks=ticks, mode="early",
                            )
                            simulated_ticks += used
                            outcome_stream.write(json.dumps({
                                **base, "branch": "timing_schedule",
                                "requested_delay_ms": delay_ms,
                                "physical": outcome,
                                "delta_vs_control": {
                                    name: float(outcome[name]) - float(control[name])
                                    for name in (
                                        "delivered_mbit", "relative_rate_percent",
                                        "handover_attempts", "handover_failures",
                                        "ping_pong_count", "rlf_starts",
                                        "recovery_seconds", "interruption_seconds",
                                    )
                                },
                            }, allow_nan=False) + "\n")
                            outcome_count += 1
                            if horizon == 1.0 and delay_ms in (0, 20):
                                primary[delay_ms] = outcome
                    label_rows.append({
                        "state_id": record["state_id"],
                        "fingerprint": record["physical_state_fingerprint"],
                        "fold": fold, "source": source,
                        "speed_kmh": int(pair.speed_kmh),
                        "label": _strict_wait_label(primary[0], primary[20]),
                    })
            print(
                f"[post-TTT] {route_index}/{len(pairs)} "
                f"{pair.speed_kmh}:{pair.trajectory_id} ({fold})", flush=True
            )
    positives = _positive_summary(label_rows)
    report = {
        "schema_name": SCHEMA_NAME, "schema_version": SCHEMA_VERSION,
        "status": "COMPLETE", "scope": "unused_train_routes_post_ttt_only",
        "validation_or_test_rows_read": 0, "routes": len(pairs),
        "fold_route_counts": dict(Counter(folds.values())),
        "sources": list(SOURCES), "anchors": anchor_count,
        "local_outcomes": outcome_count, "source_counts": dict(source_counts),
        "same_state_duplicate_fingerprints": sum(value > 1 for value in fingerprints.values()),
        "selection": {
            "category": "post_ttt", "outcome_blind": True,
            "max_anchors_per_source_route": max_anchors_per_source_route,
            "min_anchor_separation_ms": min_anchor_separation_ms,
            "method": "seeded SHA-256 rank with greedy temporal separation",
            "shortages": shortage_rows,
        },
        "counterfactual": {
            "delays_ms": list(delays_ms),
            "horizons_seconds": list(horizons_seconds),
            "primary_horizon_seconds": 1.0,
            "simulated_ticks": simulated_ticks,
        },
        "positive_summary": positives,
        "runtime_seconds": time.perf_counter() - started,
    }
    (output_dir / "bank_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    provenance = data_dir / "dataset_provenance.json"
    manifest = {
        "schema_name": SCHEMA_NAME, "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "validation_or_test_rows_read": 0,
        "data_dir": str(data_dir.resolve()),
        "dataset_provenance": ({
            "path": str(provenance.resolve()), "sha256": _sha256(provenance)
        } if provenance.is_file() else None),
        "excluded_route_splits": [
            {"path": str(path.resolve()), "sha256": _sha256(path)}
            for path in exclude_route_splits
        ],
        "route_split_sha256": _sha256(output_dir / "route_split.json"),
        "protocol_config": asdict(config), "git": _git_state(root),
        "source_files": [
            {"path": str(path.relative_to(root)), "sha256": _sha256(path)}
            for path in (
                Path(__file__).resolve(),
                root / "src/ho_optim_drl/commands/audit_early_timing_bank.py",
                root / "src/ho_optim_drl/commands/audit_early_timing_references.py",
                root / "src/ho_optim_drl/protocol/early_timing.py",
            )
        ],
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    report_lines = [
        "# Post-TTT-only TRAIN bank", "",
        f"- Route mới: **{report['routes']}**; không đọc validation/test.",
        f"- Anchor post-TTT: **{anchor_count}**; unique physical state: **{positives['unique_rows']}**.",
        f"- Strict WAIT-positive: **{positives['positive_rows']}**.",
        f"- Fold route: `{report['fold_route_counts']}`.",
        f"- Giãn cách anchor tối thiểu: {min_anchor_separation_ms} ms.",
        "", "Positive chỉ được tính sau khi route và anchor đã được khóa outcome-blind.",
    ]
    (output_dir / "report_vi.md").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )
    return report


def main(root_path: str, args: Namespace) -> int:
    root = Path(root_path).resolve()
    resolve = lambda value: (Path(value) if Path(value).is_absolute() else root / value).resolve()
    if bool(getattr(args, "split_only", False)):
        manifest = lock_confirmation_route_split(
            root=root, data_dir=resolve(args.data_dir),
            output_dir=resolve(args.output_dir), speeds=list(args.speeds),
            split_seed=int(args.split_seed), route_seed=int(args.route_seed),
            sample_seed=int(args.sample_seed),
            exclude_route_splits=[resolve(value) for value in args.exclude_route_split],
            confirmation_per_speed=int(args.confirmation_routes_per_speed),
        )
        print(json.dumps({
            "status": "LOCKED",
            "routes": manifest["selected_route_count"],
            "routes_by_speed": manifest["selected_route_count_by_speed"],
            "route_split": manifest["route_split"],
        }, indent=2), flush=True)
        return 0
    report = run_bank(
        root=root, data_dir=resolve(args.data_dir), output_dir=resolve(args.output_dir),
        speeds=list(args.speeds), split_seed=int(args.split_seed),
        route_seed=int(args.route_seed), sample_seed=int(args.sample_seed),
        exclude_route_splits=[resolve(value) for value in args.exclude_route_split],
        fit_per_speed=int(args.fit_routes_per_speed),
        tuning_per_speed=int(args.tuning_routes_per_speed),
        confirmation_per_speed=int(args.confirmation_routes_per_speed),
        max_anchors_per_source_route=int(args.max_anchors_per_source_route),
        min_anchor_separation_ms=int(args.min_anchor_separation_ms),
        delays_ms=tuple(args.delays_ms), horizons_seconds=tuple(args.horizons),
    )
    print(json.dumps({
        "status": report["status"], "routes": report["routes"],
        "anchors": report["anchors"],
        "positive_summary": report["positive_summary"],
        "runtime_seconds": report["runtime_seconds"],
    }, indent=2), flush=True)
    return 0
