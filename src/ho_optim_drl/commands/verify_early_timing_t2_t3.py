"""Verify archived T2/T3 TRAIN artifacts and emit a corrected QoS gate.

This verifier is intentionally read-only with respect to the audited artifacts.  It
recomputes bookkeeping, deterministic T2 anchor selection, T3 aggregates,
bootstraps, and gates into a fresh output directory.
"""

from __future__ import annotations

from argparse import Namespace
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any

from ho_optim_drl.commands.audit_baseline_parity import _sha256
from ho_optim_drl.commands.audit_early_timing_bank import (
    SOURCES,
    collect_source_anchors,
)
from ho_optim_drl.commands.audit_early_timing_routes import (
    METRICS,
    POLICIES,
    _extended_aggregate,
    evaluate_gates,
    paired_bootstrap,
)
from ho_optim_drl.data import TracePair, load_radio_trace
from ho_optim_drl.data.project import load_protocol_config


SCHEMA_NAME = "early_ho_timing_t2_t3_verification"
SCHEMA_VERSION = 1


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _close(left: Any, right: Any, *, atol: float = 1e-10) -> bool:
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(
            _close(left[key], right[key], atol=atol) for key in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _close(a, b, atol=atol) for a, b in zip(left, right)
        )
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=1e-10, abs_tol=atol)
    return left == right


def _check(checks: list[dict[str, Any]], name: str, passed: bool, detail: Any) -> None:
    checks.append({"name": name, "passed": bool(passed), "detail": detail})


def _verify_t2(
    *, data_dir: Path, t2_dir: Path, replay_anchors: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    report = _read_json(t2_dir / "bank_report.json")
    manifest = _read_json(t2_dir / "run_manifest.json")
    split = _read_json(t2_dir / "route_split.json")
    anchors = _read_jsonl(t2_dir / "anchors.jsonl")
    outcomes = _read_jsonl(t2_dir / "local_outcomes.jsonl")

    selected = split["selected_routes"]
    speed_counts = Counter(int(row["speed_kmh"]) for row in selected)
    _check(checks, "train_only_scope", split.get("split") == "train" and split.get("validation_or_test_rows_read") == 0 and manifest.get("validation_or_test_rows_read") == 0, {"split": split.get("split"), "validation_or_test_rows_read": manifest.get("validation_or_test_rows_read")})
    _check(checks, "route_quota", len(selected) == 80 and speed_counts == Counter({30: 20, 50: 20, 70: 20, 90: 20}), dict(speed_counts))
    _check(checks, "route_split_hash", manifest.get("route_split_sha256") == _sha256(t2_dir / "route_split.json"), manifest.get("route_split_sha256"))

    bad_hashes: list[str] = []
    for row in selected:
        for kind in ("rsrp", "sinr"):
            path = data_dir / row[f"{kind}_file"]
            if not path.is_file() or _sha256(path) != row[f"{kind}_sha256"]:
                bad_hashes.append(str(path))
    _check(checks, "dataset_file_hashes", not bad_hashes, bad_hashes)

    state_ids = [str(row["state_id"]) for row in anchors]
    group_counts = Counter(
        (int(row["speed_kmh"]), int(row["trajectory_id"]), str(row["source"]))
        for row in anchors
    )
    shortages = sum(5 - count for count in group_counts.values())
    _check(checks, "anchor_counts", len(anchors) == report.get("anchors") == 1199 and len(set(state_ids)) == len(state_ids), {"anchors": len(anchors), "unique_state_ids": len(set(state_ids))})
    _check(checks, "anchor_quota", len(group_counts) == 80 * len(SOURCES) and max(group_counts.values(), default=0) <= 5 and shortages == 1, {"source_route_groups": len(group_counts), "max": max(group_counts.values(), default=0), "shortage": shortages})

    expected_delays = {0, 20, 40, 80, 160, 320}
    expected_horizons = {0.5, 1.0, 2.0}
    outcome_groups: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in outcomes:
        outcome_groups[(str(row["state_id"]), float(row["horizon_seconds"]))].append(row)
    grid_errors: list[str] = []
    arithmetic_errors: list[str] = []
    for state_id in state_ids:
        for horizon in expected_horizons:
            group = outcome_groups.get((state_id, horizon), [])
            controls = [row for row in group if row["branch"] == "baseline_continuation"]
            schedules = [row for row in group if row["branch"] == "timing_schedule"]
            if len(controls) != 1 or len(schedules) != 6 or {row["requested_delay_ms"] for row in schedules} != expected_delays:
                grid_errors.append(f"{state_id}@{horizon}")
                continue
            control = controls[0]["physical"]
            for scheduled in schedules:
                for key, value in scheduled["delta_vs_control"].items():
                    expected = float(scheduled["physical"][key]) - float(control[key])
                    if not math.isclose(float(value), expected, rel_tol=1e-10, abs_tol=1e-10):
                        arithmetic_errors.append(f"{state_id}@{horizon}:{scheduled['requested_delay_ms']}:{key}")
    _check(checks, "local_outcome_grid", len(outcomes) == report.get("local_outcomes") == len(anchors) * 21 and not grid_errors, {"rows": len(outcomes), "errors": grid_errors[:10]})
    _check(checks, "local_delta_arithmetic", not arithmetic_errors, arithmetic_errors[:10])

    replay_mismatches: list[dict[str, Any]] = []
    if replay_anchors:
        config = load_protocol_config(str(data_dir))
        archived = defaultdict(list)
        for row in anchors:
            archived[(int(row["speed_kmh"]), int(row["trajectory_id"]), str(row["source"]))].append(row)
        for route_index, row in enumerate(selected, 1):
            pair = TracePair(int(row["speed_kmh"]), int(row["trajectory_id"]), str(row["rsrp_file"]), str(row["sinr_file"]))
            trace = load_radio_trace(config, str(data_dir), pair.rsrp_file, pair.sinr_file, speed_kmh=pair.speed_kmh)
            for source in SOURCES:
                regenerated, _ = collect_source_anchors(
                    config, trace, pair, source=source,
                    sample_seed=int(split["sample_seed"]),
                    max_anchors=int(report["selection"]["max_anchors_per_source_route"]),
                )
                actual = [item["record"] for item in regenerated]
                expected = sorted(archived[(pair.speed_kmh, pair.trajectory_id, source)], key=lambda item: int(item["tic"]))
                if not _close(actual, expected):
                    replay_mismatches.append({"trace_id": f"{pair.speed_kmh}:{pair.trajectory_id}", "source": source, "expected": len(expected), "actual": len(actual)})
            if route_index % 10 == 0:
                print(f"[verify T2] replayed {route_index}/{len(selected)} routes", flush=True)
    _check(checks, "deterministic_anchor_replay", replay_anchors and not replay_mismatches, {"enabled": replay_anchors, "mismatches": replay_mismatches[:10]})
    return {"passed": all(item["passed"] for item in checks), "checks": checks}, split


def _verify_t3(*, t2_dir: Path, t3_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    report = _read_json(t3_dir / "main_report.json")
    archived_gate = _read_json(t3_dir / "decision_gate.json")
    manifest = _read_json(t3_dir / "run_manifest.json")
    flat = _read_jsonl(t3_dir / "route_outcomes.jsonl")
    split_path = t2_dir / "route_split.json"
    split = _read_json(split_path)

    rows_by_trace: dict[str, dict[str, Any]] = {}
    duplicates: list[str] = []
    for item in flat:
        trace_id = str(item["trace_id"])
        row = rows_by_trace.setdefault(trace_id, {"trace_id": trace_id, "speed_kmh": int(item["speed_kmh"]), "trajectory_id": int(item["trajectory_id"])})
        policy = str(item["policy"])
        if policy in row:
            duplicates.append(f"{trace_id}:{policy}")
        row[policy] = item["metrics"]
    rows = list(rows_by_trace.values())
    grid_ok = len(flat) == 560 and len(rows) == 80 and not duplicates and all(set(POLICIES).issubset(row) for row in rows)
    _check(checks, "policy_route_grid", grid_ok, {"flat_rows": len(flat), "routes": len(rows), "duplicates": duplicates[:10]})
    _check(checks, "train_only_scope", report.get("scope") == "train_only" and manifest.get("validation_or_test_rows_read") == 0, {"scope": report.get("scope"), "validation_or_test_rows_read": manifest.get("validation_or_test_rows_read")})
    _check(checks, "route_split_provenance", manifest.get("route_split", {}).get("sha256") == _sha256(split_path) and manifest.get("selected_routes") == split.get("selected_routes"), manifest.get("route_split"))
    parity_bad = [row["trace_id"] for row in rows if not _close(row["b0"], row["b0a"])]
    _check(checks, "b0_b0a_route_parity", not parity_bad, parity_bad[:10])

    dt_ms = int(manifest["protocol_config"]["delta_t_ms"])
    aggregate = {policy: _extended_aggregate(rows, policy, dt_ms) for policy in POLICIES}
    aggregate_match = all(_close(aggregate[policy], report["aggregate"][policy]) for policy in POLICIES)
    _check(checks, "aggregate_recomputation", aggregate_match, {"policies": list(POLICIES)})
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["speed_kmh"])].append(row)
    per_speed = {str(speed): {policy: _extended_aggregate(group, policy, dt_ms) for policy in POLICIES} for speed, group in sorted(grouped.items())}
    _check(checks, "per_speed_recomputation", _close(per_speed, report["per_speed"]), {"speeds": sorted(per_speed)})

    specs = (
        ("rpost_minus_b0", "b0", "rpost", 0),
        ("rearly_minus_b0", "b0", "rearly", 1),
        ("rearly_minus_rpost", "rpost", "rearly", 2),
        ("rearly_minus_b1", "b1", "rearly", 3),
        ("rearly_minus_b2", "b2", "rearly", 4),
        ("rearly_minus_b3", "b3", "rearly", 5),
    )
    resamples = int(manifest["bootstrap_resamples"])
    seed = int(manifest["bootstrap_seed"])
    recomputed_bootstrap = {
        name: paired_bootstrap(rows, left=left, right=right, dt_ms=dt_ms, resamples=resamples, seed=seed + offset)
        for name, left, right, offset in specs
    }
    bootstrap_match = all(_close(recomputed_bootstrap[name], report["paired_bootstrap"][name]) for name, *_ in specs)
    _check(checks, "paired_bootstrap_recomputation", bootstrap_match, {"resamples": resamples, "seed_range": [seed, seed + 5], "metrics": list(METRICS)})

    corrected_gate = evaluate_gates(aggregate, recomputed_bootstrap, per_speed)
    old_definition = archived_gate.get("g_qos_guard", {}).get("interpretation", "")
    _check(checks, "legacy_gate_identified", "absolute RR" in old_definition, old_definition)
    return {"passed": all(item["passed"] for item in checks), "checks": checks, "recomputed": {"aggregate": aggregate, "per_speed": per_speed, "paired_bootstrap": recomputed_bootstrap}}, corrected_gate


def _markdown(report: dict[str, Any]) -> str:
    gate = report["corrected_gate"]
    qos = gate["g_qos_guard"]
    lines = [
        "# Kiểm chứng T2/T3 và gate QoS hiệu chỉnh",
        "",
        f"- Trạng thái: **{report['status']}**",
        f"- T2: **{'PASS' if report['t2']['passed'] else 'FAIL'}**",
        f"- T3: **{'PASS' if report['t3']['passed'] else 'FAIL'}**",
        f"- G-physical: **{'PASS' if gate['g_physical_strong']['passed'] else 'FAIL'}**",
        f"- G-QoS (retention so với B0): **{'PASS' if qos['passed'] else 'FAIL'}**",
        f"- G-early: **{'PASS' if gate['g_early']['passed'] else 'FAIL'}**",
        f"- Gate nghiên cứu tổng: **{gate['overall_research_gate']}**",
        "",
        "## Retention QoS",
        "",
        f"- Mean rate: {qos['aggregate_rate_retention_percent']:.4f}%",
        f"- Relative rate: {qos['aggregate_relative_rate_retention_percent']:.4f}%",
        f"- Mean-rate theo speed: {json.dumps(qos['per_speed_rate_retention_percent'], ensure_ascii=False)}",
        f"- Relative-rate theo speed: {json.dumps(qos['per_speed_relative_rate_retention_percent'], ensure_ascii=False)}",
        "",
        "## Diễn giải",
        "",
        "Artifact T2/T3 cũ không bị sửa. Gate QoS cũ bị thay thế vì đã so RR tuyệt đối với 98%, trong khi tiêu chí đúng là giữ >=98% so với B0 ở cả aggregate và từng speed.",
        "G-early vẫn là phép thử riêng: lợi ích tăng thêm của Rearly so với Rpost chỉ đạt gate khi CI của RR dương và CI của mọi rủi ro không tăng.",
    ]
    return "\n".join(lines) + "\n"


def run_verification(
    *, root: Path, data_dir: Path, t2_dir: Path, t3_dir: Path,
    output_dir: Path, replay_anchors: bool,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite verification output: {output_dir}")
    output_dir.mkdir(parents=True)
    t2, split = _verify_t2(data_dir=data_dir, t2_dir=t2_dir, replay_anchors=replay_anchors)
    t3, gate = _verify_t3(t2_dir=t2_dir, t3_dir=t3_dir)
    status = "PASS" if t2["passed"] and t3["passed"] else "FAIL"
    result = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "scope": "train_only",
        "validation_or_test_rows_read": 0,
        "t2": t2,
        "t3": t3,
        "corrected_gate": gate,
        "archived_artifacts_modified": False,
        "limitations": [
            "TRAIN development evidence only; no validation/test/zero-shot data read.",
            "The corrected gate changes QoS interpretation only; it does not alter physical outcomes or bootstrap samples.",
        ],
    }
    (output_dir / "verification_report.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output_dir / "corrected_gate.json").write_text(json.dumps(gate, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output_dir / "report_vi.md").write_text(_markdown(result), encoding="utf-8")
    manifest = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "validation_or_test_rows_read": 0,
        "inputs": {
            "t2_dir": str(t2_dir.resolve()),
            "t3_dir": str(t3_dir.resolve()),
            "t2_files": {name: _sha256(t2_dir / name) for name in ("anchors.jsonl", "local_outcomes.jsonl", "bank_report.json", "route_split.json", "run_manifest.json")},
            "t3_files": {name: _sha256(t3_dir / name) for name in ("route_outcomes.jsonl", "main_report.json", "decision_gate.json", "run_manifest.json")},
            "selected_route_count": len(split["selected_routes"]),
        },
        "anchor_replay_enabled": replay_anchors,
        "verifier_source": {"path": str(Path(__file__).resolve().relative_to(root)), "sha256": _sha256(Path(__file__).resolve())},
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result


def main(root_path: str, args: Namespace) -> int:
    root = Path(root_path).resolve()
    def resolved(value: str) -> Path:
        path = Path(value)
        return (path if path.is_absolute() else root / path).resolve()
    report = run_verification(
        root=root,
        data_dir=resolved(args.data_dir),
        t2_dir=resolved(args.t2_dir),
        t3_dir=resolved(args.t3_dir),
        output_dir=resolved(args.output_dir),
        replay_anchors=not bool(args.skip_anchor_replay),
    )
    print(json.dumps({"status": report["status"], "corrected_gate": report["corrected_gate"]}, indent=2), flush=True)
    return 0 if report["status"] == "PASS" else 1
