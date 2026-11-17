"""TRAIN-only B0/B0A equivalence audit for the early-HO research branch."""

from __future__ import annotations

from argparse import Namespace
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Sequence

import numpy as np

from ho_optim_drl.agents import AlwaysHandoverPolicy, run_policy
from ho_optim_drl.commands.audit_policy_learning import _selected_routes
from ho_optim_drl.data import RadioTrace, TracePair, load_radio_trace
from ho_optim_drl.data.project import load_protocol_config
from ho_optim_drl.protocol import A3ReferenceSimulator, HandoverEngine


SCHEMA_NAME = "early_ho_b0_b0a_parity"
SCHEMA_VERSION = 1
COUNTERS = ("n310", "n311", "t310", "rlfr", "tttc", "ho_prep", "ho_exec")
COUNTER_FIELDS = (
    "start_idxs",
    "terminated_idxs",
    "done_idxs",
    "reset_idxs",
    "aborted_idxs",
    "failed_idxs",
    "error_idxs",
)
TIMELINES = (
    "bs_idxs",
    "ho_idxs",
    "rlfr_idxs",
    "rsrp_timeline",
    "sinr_timeline",
    "sinr_at_ho_exe_pcell",
    "sinr_after_ho_exe_tcell",
)


def _sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    return value


def _array_equal(left: Sequence[Any], right: Sequence[Any]) -> bool:
    a = np.asarray(left)
    b = np.asarray(right)
    if a.shape != b.shape:
        return False
    if a.dtype.kind in "fc" or b.dtype.kind in "fc":
        return bool(np.array_equal(a, b, equal_nan=True))
    return bool(np.array_equal(a, b))


def _first_difference(left: Sequence[Any], right: Sequence[Any]) -> dict[str, Any] | None:
    a = np.asarray(left)
    b = np.asarray(right)
    if a.shape != b.shape:
        return {"kind": "shape", "b0": list(a.shape), "b0a": list(b.shape)}
    if _array_equal(left, right):
        return None
    equal = (a == b) | (np.isnan(a) & np.isnan(b)) if (
        a.dtype.kind in "fc" or b.dtype.kind in "fc"
    ) else a == b
    flat_index = int(np.flatnonzero(~equal)[0])
    index = tuple(int(item) for item in np.unravel_index(flat_index, a.shape))
    return {
        "kind": "value",
        "index": list(index),
        "b0": _json_value(a[index]),
        "b0a": _json_value(b[index]),
    }


def compare_simulators(
    b0: A3ReferenceSimulator, b0a: HandoverEngine, trace: RadioTrace
) -> dict[str, Any]:
    """Compare every observable timeline, event counter, and route metric."""
    mismatches: dict[str, Any] = {}
    for name in TIMELINES:
        difference = _first_difference(getattr(b0, name), getattr(b0a, name))
        if difference is not None:
            mismatches[f"timeline.{name}"] = difference

    if b0.successful_ho_history != b0a.successful_ho_history:
        mismatches["events.successful_ho_history"] = {
            "b0": _json_value(b0.successful_ho_history),
            "b0a": _json_value(b0a.successful_ho_history),
        }
    if b0.ping_pong_idxs != b0a.ping_pong_idxs:
        mismatches["events.ping_pong_idxs"] = {
            "b0": list(b0.ping_pong_idxs),
            "b0a": list(b0a.ping_pong_idxs),
        }

    for counter_name in COUNTERS:
        left = b0.cntr[counter_name]
        right = b0a.cntr[counter_name]
        for field in COUNTER_FIELDS:
            left_value = list(getattr(left, field))
            right_value = list(getattr(right, field))
            if left_value != right_value:
                mismatches[f"counter.{counter_name}.{field}"] = {
                    "b0": left_value,
                    "b0a": right_value,
                }
        for field in ("cnt", "flag_is_reset", "flag_is_active", "flag_reached_max"):
            if getattr(left, field) != getattr(right, field):
                mismatches[f"counter.{counter_name}.{field}"] = {
                    "b0": _json_value(getattr(left, field)),
                    "b0a": _json_value(getattr(right, field)),
                }

    left_stats = b0.get_statistics(trace)
    right_stats = b0a.get_statistics(trace)
    if set(left_stats) != set(right_stats):
        mismatches["metrics.keys"] = {
            "b0_only": sorted(set(left_stats) - set(right_stats)),
            "b0a_only": sorted(set(right_stats) - set(left_stats)),
        }
    for name in sorted(set(left_stats) & set(right_stats)):
        left_value = left_stats[name]
        right_value = right_stats[name]
        equal = (
            bool(np.isclose(left_value, right_value, rtol=0.0, atol=0.0, equal_nan=True))
            if isinstance(left_value, (float, np.floating))
            or isinstance(right_value, (float, np.floating))
            else left_value == right_value
        )
        if not equal:
            mismatches[f"metric.{name}"] = {
                "b0": _json_value(left_value),
                "b0a": _json_value(right_value),
            }
    return {
        "passed": not mismatches,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "b0_metrics": _json_value(left_stats),
        "b0a_metrics": _json_value(right_stats),
    }


def run_pair(config: Any, trace: RadioTrace) -> tuple[dict[str, Any], float, float]:
    """Run B0 and B0A independently from the same immutable trace."""
    b0 = A3ReferenceSimulator(config)
    started = time.perf_counter()
    b0.predict(trace)
    b0_seconds = time.perf_counter() - started

    b0a = HandoverEngine(config)
    b0a.reset(trace)
    started = time.perf_counter()
    results = run_policy(b0a, AlwaysHandoverPolicy())
    b0a_seconds = time.perf_counter() - started
    if len(results) != trace.num_samples:
        raise RuntimeError("B0A did not emit exactly one result per trace sample.")
    return compare_simulators(b0, b0a, trace), b0_seconds, b0a_seconds


def _git_state(root: Path) -> dict[str, Any]:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=root, check=True,
            capture_output=True, text=True,
        ).stdout
        diff = subprocess.run(
            ["git", "diff", "--binary", "HEAD"], cwd=root, check=True,
            capture_output=True,
        ).stdout
        return {
            "commit": revision,
            "dirty": bool(status.strip()),
            "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def _prepare_output(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing parity audit: {path}")
    path.mkdir(parents=True)


def run_audit(
    *, root: Path, data_dir: Path, output_dir: Path, speeds: list[int],
    split_seed: int, sample_seed: int, routes_per_speed: int,
) -> dict[str, Any]:
    """Execute the reproducible TRAIN-only parity audit."""
    _prepare_output(output_dir)
    pairs = _selected_routes(
        data_dir=str(data_dir), speeds=speeds, split_seed=split_seed,
        sample_seed=sample_seed, routes_per_speed=routes_per_speed,
        train_fraction=0.70, validation_fraction=0.15, test_fraction=0.15,
    )
    config = load_protocol_config(str(data_dir))
    rows: list[dict[str, Any]] = []
    rows_path = output_dir / "route_parity.jsonl"
    with rows_path.open("w", encoding="utf-8") as stream:
        for index, pair in enumerate(pairs, 1):
            trace = load_radio_trace(
                config, str(data_dir), pair.rsrp_file, pair.sinr_file,
                speed_kmh=pair.speed_kmh,
            )
            comparison, b0_seconds, b0a_seconds = run_pair(config, trace)
            row = {
                "trace_id": f"{pair.speed_kmh}:{pair.trajectory_id}",
                "speed_kmh": pair.speed_kmh,
                "trajectory_id": pair.trajectory_id,
                "rsrp_file": pair.rsrp_file,
                "sinr_file": pair.sinr_file,
                "samples": trace.num_samples,
                "b0_seconds": b0_seconds,
                "b0a_seconds": b0a_seconds,
                **comparison,
            }
            rows.append(row)
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            print(
                f"[B0/B0A] {index}/{len(pairs)} {row['trace_id']} "
                f"{'PASS' if row['passed'] else 'FAIL'}"
            )

    selected_routes = [
        {
            "speed_kmh": pair.speed_kmh,
            "trajectory_id": pair.trajectory_id,
            "rsrp_file": pair.rsrp_file,
            "sinr_file": pair.sinr_file,
            "rsrp_sha256": _sha256(data_dir / pair.rsrp_file),
            "sinr_sha256": _sha256(data_dir / pair.sinr_file),
        }
        for pair in pairs
    ]
    provenance_path = data_dir / "dataset_provenance.json"
    source_paths = (
        Path(__file__).resolve(),
        root / "src/ho_optim_drl/protocol/reference.py",
        root / "src/ho_optim_drl/protocol/engine.py",
        root / "src/ho_optim_drl/agents/runner.py",
        root / "src/ho_optim_drl/config.py",
    )
    manifest = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "train_only",
        "validation_or_test_rows_read": 0,
        "policies": {
            "B0": "A3ReferenceSimulator.predict",
            "B0A": "HandoverEngine plus AlwaysHandoverPolicy at completed A3+TTT",
        },
        "comparison": {
            "numeric_tolerance": {"rtol": 0.0, "atol": 0.0, "equal_nan": True},
            "timelines": list(TIMELINES),
            "counters": list(COUNTERS),
            "counter_fields": list(COUNTER_FIELDS),
            "metrics": "all keys returned by get_statistics",
        },
        "data_dir": str(data_dir.resolve()),
        "dataset_provenance": (
            {"file": provenance_path.name, "sha256": _sha256(provenance_path)}
            if provenance_path.is_file() else None
        ),
        "protocol_config": asdict(config),
        "split": {"seed": split_seed, "fractions": [0.70, 0.15, 0.15]},
        "sampling": {"seed": sample_seed, "routes_per_speed": routes_per_speed},
        "speeds_kmh": speeds,
        "selected_routes": selected_routes,
        "git": _git_state(root),
        "source_files": [
            {"path": str(path.relative_to(root)), "sha256": _sha256(path)}
            for path in source_paths
        ],
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    passed = sum(bool(row["passed"]) for row in rows)
    attempts = sum(int(row["b0_metrics"]["num_ho_exe_started"]) for row in rows)
    successes = sum(int(row["b0_metrics"]["num_ho_exe_completed"]) for row in rows)
    failures = sum(int(row["b0_metrics"]["num_hof"]) for row in rows)
    ping_pongs = sum(int(row["b0_metrics"]["num_pp"]) for row in rows)
    recovery_samples = sum(
        int(row["b0_metrics"]["rlf_recovery_samples"]) for row in rows
    )
    total_samples = sum(int(row["b0_metrics"]["total_time"]) for row in rows)
    relative_rates = [
        100.0 * float(row["b0_metrics"]["estimated_rate_mbps"])
        / float(row["b0_metrics"]["max_estimated_rate_mbps"])
        for row in rows
        if float(row["b0_metrics"]["max_estimated_rate_mbps"]) > 0
    ]
    report = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "gate": "PASS" if passed == len(rows) else "FAIL",
        "routes": len(rows),
        "routes_passed": passed,
        "routes_failed": len(rows) - passed,
        "samples_compared": sum(int(row["samples"]) for row in rows),
        "total_b0_seconds": sum(float(row["b0_seconds"]) for row in rows),
        "total_b0a_seconds": sum(float(row["b0a_seconds"]) for row in rows),
        "failed_trace_ids": [row["trace_id"] for row in rows if not row["passed"]],
        "common_metrics": {
            "mean_route_relative_rate_percent": float(np.mean(relative_rates)),
            "handover_attempts": attempts,
            "handover_successes": successes,
            "handover_failures": failures,
            "handover_failure_percent": 100.0 * failures / attempts if attempts else None,
            "ping_pong_count": ping_pongs,
            "ping_pong_percent": 100.0 * ping_pongs / successes if successes else None,
            "rlf_recovery_samples": recovery_samples,
            "rlf_time_percent": (
                100.0 * recovery_samples / total_samples if total_samples else None
            ),
        },
        "conclusion": (
            "B0A is observationally equivalent to B0 on the selected TRAIN routes."
            if passed == len(rows)
            else "B0A is not equivalent to B0; inspect route_parity.jsonl."
        ),
    }
    (output_dir / "parity_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    report_md = (
        "# B0/B0A parity audit\n\n"
        f"Gate: **{report['gate']}**. TRAIN-only routes: {len(rows)}; "
        f"passed: {passed}; failed: {len(rows) - passed}.\n\n"
        f"Compared {report['samples_compared']} primitive ticks with exact "
        "equality (NaN aligned), every protocol counter/event and every "
        "`get_statistics` metric. B0A only accepts the completed A3+TTT "
        "decision and therefore does not yet enable early HO.\n"
    )
    (output_dir / "report_vi.md").write_text(report_md, encoding="utf-8")
    return report


def main(root_path: str, args: Namespace) -> int:
    root = Path(root_path).resolve()
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = root / data_dir
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    report = run_audit(
        root=root,
        data_dir=data_dir.resolve(),
        output_dir=output_dir.resolve(),
        speeds=list(args.speeds),
        split_seed=args.split_seed,
        sample_seed=args.sample_seed,
        routes_per_speed=args.routes_per_speed,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["gate"] == "PASS" else 1
