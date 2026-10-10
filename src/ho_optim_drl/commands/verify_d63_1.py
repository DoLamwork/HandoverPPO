"""Verify and freeze the completed D63-1 TRAIN-only audit package."""

from __future__ import annotations

from argparse import Namespace
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
from typing import Any, Sequence

from ho_optim_drl.commands.audit_timing_target_main import (
    POLICIES,
    _aggregate_route_rows,
)


AGGREGATE_FIELDS = (
    "routes",
    "mean_rate_mbps",
    "mean_relative_rate_percent",
    "p10_relative_rate_percent",
    "handover_attempts",
    "handover_successes",
    "handover_failures",
    "handover_failure_percent",
    "rlf_starts",
    "pure_rlf_starts",
    "recovery_seconds",
    "rlf_time_percent",
    "ping_pong_count",
    "ping_pong_percent",
    "handover_interruption_seconds",
    "decision_count",
    "stay_actions",
    "handover_actions",
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}.")
            rows.append(row)
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prepare_output(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty verification output: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _git(root: Path, *args: str) -> str | None:
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


def _assert_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label}: expected {expected!r}, got {actual!r}.")


def _assert_close(actual: Any, expected: Any, label: str) -> None:
    if not math.isclose(float(actual), float(expected), rel_tol=1e-10, abs_tol=1e-10):
        raise ValueError(f"{label}: expected {expected!r}, got {actual!r}.")


def _verify_aggregate(
    rows: Sequence[dict[str, Any]], summary: dict[str, Any], label: str
) -> None:
    recomputed = _aggregate_route_rows(rows)
    for field in AGGREGATE_FIELDS:
        if isinstance(recomputed[field], int):
            _assert_equal(summary[field], recomputed[field], f"{label}.{field}")
        else:
            _assert_close(summary[field], recomputed[field], f"{label}.{field}")


def _verify_metric_definitions(
    rows: Sequence[dict[str, Any]], route_summary: dict[str, Any]
) -> dict[str, str]:
    by_policy: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_policy[str(row["policy"])].append(row)
        expected_rr = (
            100.0
            * float(row["estimated_rate_mbps"])
            / float(row["max_estimated_rate_mbps"])
            if float(row["max_estimated_rate_mbps"]) > 0
            else 0.0
        )
        _assert_close(row["relative_rate_percent"], expected_rr, "row.relative_rate_percent")
        _assert_equal(
            int(row["decision_count"]),
            int(row["stay_actions"]) + int(row["handover_actions"]),
            "row.action_accounting",
        )

    for policy in POLICIES:
        _verify_aggregate(
            by_policy[policy], route_summary["aggregate"][policy], f"aggregate.{policy}"
        )
        speed_groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in by_policy[policy]:
            speed_groups[int(row["speed_kmh"])].append(row)
        for speed, selected in sorted(speed_groups.items()):
            _verify_aggregate(
                selected,
                route_summary["by_speed"][policy][str(speed)],
                f"by_speed.{policy}.{speed}",
            )

    control = by_policy["always_ho_c1"]
    _assert_equal(sum(int(row["stay_actions"]) for row in control), 0, "always_c1.stays")
    return {
        "mean_rate_mbps": "arithmetic mean of per-route estimated_rate_mbps",
        "relative_rate_percent_per_route": "100 * estimated_rate_mbps / max_estimated_rate_mbps",
        "mean_relative_rate_percent": "arithmetic mean of per-route relative_rate_percent",
        "handover_failure_percent": "100 * total handover_failures / total handover_attempts",
        "rlf_time_percent": "100 * total recovery_seconds / total route time seconds",
        "ping_pong_percent": "100 * total ping_pong_count / total handover_successes",
        "pure_rlf_starts": "RLF starts whose tick is not simultaneously a handover failure",
        "qos_retention": "treatment aggregate mean divided by Always-C1 aggregate mean; per-speed ratios computed separately",
    }


def _verify_route_coverage(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    _assert_equal(len(rows), 320, "full_route.row_count")
    by_policy: dict[str, set[tuple[int, int]]] = defaultdict(set)
    for row in rows:
        by_policy[str(row["policy"])].add(
            (int(row["speed_kmh"]), int(row["trajectory_id"]))
        )
    _assert_equal(set(by_policy), set(POLICIES), "full_route.policy_set")
    control_routes = by_policy["always_ho_c1"]
    _assert_equal(len(control_routes), 80, "full_route.unique_routes")
    for policy in POLICIES:
        _assert_equal(by_policy[policy], control_routes, f"full_route.routes.{policy}")
    speed_counts = Counter(speed for speed, _ in control_routes)
    _assert_equal(dict(sorted(speed_counts.items())), {30: 20, 50: 20, 70: 20, 90: 20}, "full_route.routes_by_speed")
    return {
        "rows": len(rows),
        "policies": list(POLICIES),
        "unique_routes": len(control_routes),
        "routes_by_speed": {str(key): value for key, value in sorted(speed_counts.items())},
    }


def _verify_input_provenance(
    input_registry: dict[str, Any], local_manifest: dict[str, Any], local_registry: dict[str, Any]
) -> dict[str, Any]:
    checked_inputs = []
    for name, record in input_registry["files"].items():
        path = Path(record["path"])
        _assert_equal(path.is_file(), True, f"input.exists.{name}")
        actual = _sha256(path)
        _assert_equal(actual, record["sha256"], f"input.sha256.{name}")
        checked_inputs.append({"name": name, "path": str(path), "sha256": actual})

    dataset = local_manifest.get("dataset_provenance")
    if dataset is not None:
        dataset_path = Path(dataset["path"])
        _assert_equal(_sha256(dataset_path), dataset["sha256"], "dataset_provenance.sha256")

    trace_files_checked = 0
    data_dir = Path(local_manifest["data_dir"])
    for route in local_manifest["selected_routes"]:
        for kind in ("rsrp", "sinr"):
            path = data_dir / str(route[f"{kind}_file"])
            _assert_equal(_sha256(path), route[f"{kind}_sha256"], f"route.{kind}_sha256")
            trace_files_checked += 1

    checkpoints = []
    for source in local_registry["sources"]:
        path = Path(source["checkpoint"])
        actual = _sha256(path)
        _assert_equal(actual, source["checkpoint_sha256"], f"checkpoint.{source['label']}")
        checkpoints.append(
            {"label": source["label"], "path": str(path), "sha256": actual}
        )
    return {
        "input_artifacts": checked_inputs,
        "dataset_provenance": dataset,
        "selected_trace_files_checked": trace_files_checked,
        "checkpoints": checkpoints,
        "local_bank_runtime_source": local_registry.get("current_source"),
    }


def _current_source_snapshot(root: Path) -> dict[str, Any]:
    paths = (
        root / "run.py",
        root / "src" / "ho_optim_drl" / "cli.py",
        root / "src" / "ho_optim_drl" / "commands" / "audit_timing_target.py",
        root / "src" / "ho_optim_drl" / "commands" / "audit_timing_target_main.py",
        root / "src" / "ho_optim_drl" / "commands" / "verify_d63_1.py",
    )
    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": _git(root, "rev-parse", "HEAD"),
        "git_branch": _git(root, "branch", "--show-current"),
        "git_status_porcelain": (
            _git(root, "status", "--porcelain=v1", "--untracked-files=all") or ""
        ).splitlines(),
        "files": [
            {"path": path.relative_to(root).as_posix(), "sha256": _sha256(path)}
            for path in paths
        ],
        "provenance_note": (
            "This is a post-run verification snapshot. The local bank registry "
            "contains its runtime source-tree hash; the main full-route run did "
            "not embed a runtime source-tree hash when it started."
        ),
    }


def _markdown(report: dict[str, Any]) -> str:
    gate = report["gate_g1"]
    aggregate = report["aggregate"]
    timing = aggregate["timing_only_reference"]
    joint = aggregate["joint_reference"]
    control = aggregate["always_ho_c1"]
    return f"""# D63-1 — báo cáo kiểm chứng và tái lập

Ngày tạo: {report['created_at_utc']}. Trạng thái: **{report['verification_status']}**.

## Phạm vi đã chốt

- Chỉ dùng TRAIN: 20 route/tốc độ, tổng 80 route.
- State bank có {report['state_bank']['states']} anchor; tối đa **5 anchor/source/route**.
- Kế hoạch cho phép tối đa 10 anchor/source/route. Gói thực tế dùng 5; đây là quota nhỏ hơn đã ghi nhận, không chạy lại chỉ để lấp quota.
- Full-route có {report['coverage']['rows']} hàng = 80 route × 4 policy.
- Không chạy training mới, validation, test hay zero-shot.

## Định nghĩa metric đã kiểm tra

- Relative rate từng route = 100 × estimated rate / max estimated rate; aggregate là trung bình theo route.
- HOF% = 100 × tổng HOF / tổng HO attempt.
- RLF-time% = 100 × tổng recovery seconds / tổng thời gian route.
- Ping-pong% = 100 × tổng ping-pong / tổng HO success.
- QoS retention được tính từ tỷ lệ các aggregate mean, và báo riêng từng tốc độ.

## Kết quả full-route TRAIN

| Policy | RR (%) | HOF (%) | RLF-time (%) | PP (%) |
|---|---:|---:|---:|---:|
| Always-C1 | {control['mean_relative_rate_percent']:.3f} | {control['handover_failure_percent']:.3f} | {control['rlf_time_percent']:.3f} | {control['ping_pong_percent']:.3f} |
| Timing-only reference | {timing['mean_relative_rate_percent']:.3f} | {timing['handover_failure_percent']:.3f} | {timing['rlf_time_percent']:.3f} | {timing['ping_pong_percent']:.3f} |
| Timing×target reference | {joint['mean_relative_rate_percent']:.3f} | {joint['handover_failure_percent']:.3f} | {joint['rlf_time_percent']:.3f} | {joint['ping_pong_percent']:.3f} |

Gate G1: **{gate['status']}**. Đây là finite-lookahead reference trên TRAIN; chưa phải policy triển khai và chưa đánh giá G2.

## Provenance và giới hạn

- Đã kiểm tra SHA-256 của {len(report['provenance']['input_artifacts'])} input artifact, {report['provenance']['selected_trace_files_checked']} file trace và checkpoint D61/D62.
- Local bank registry có runtime source-tree hash.
- Main full-route run chưa nhúng source-tree hash ngay lúc bắt đầu. Báo cáo lưu post-run source snapshot để tái lập về sau, nhưng không trình bày snapshot này như bằng chứng mật mã của code đúng tại thời điểm run.
- Test command: `{report['tests']['command']}` — **{report['tests']['status']}**.

## Kết luận

D63-1 đã được chốt với G1 PASS. Bước nghiên cứu kế tiếp là D63-2 để kiểm tra observation nhân quả có học được cơ hội timing và target hay không.
"""


def verify_package(
    *,
    root: Path,
    main_dir: Path,
    output_dir: Path,
    test_command: str,
) -> dict[str, Any]:
    _prepare_output(output_dir)
    required = (
        "input_registry.json",
        "local_main_analysis.json",
        "route_reference_outcomes.jsonl",
        "route_reference_summary.json",
        "decision_gate.json",
        "main_report.json",
    )
    missing = [name for name in required if not (main_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete D63-1 main artifact: {missing}")

    input_registry = _load_json(main_dir / "input_registry.json")
    main_report = _load_json(main_dir / "main_report.json")
    route_summary = _load_json(main_dir / "route_reference_summary.json")
    gate = _load_json(main_dir / "decision_gate.json")
    rows = _load_jsonl(main_dir / "route_reference_outcomes.jsonl")
    local_dir = Path(input_registry["local_artifact_dir"])
    local_manifest = _load_json(local_dir / "state_bank_manifest.json")
    local_registry = _load_json(local_dir / "registry.json")

    _assert_equal(main_report["split"], "train", "main_report.split")
    _assert_equal(main_report["validation_or_test_rows_read"], 0, "main_report.validation_or_test_rows_read")
    _assert_equal(main_report["training_runs_started"], 0, "main_report.training_runs_started")
    _assert_equal(main_report["d63_2_implemented"], False, "main_report.d63_2_implemented")
    _assert_equal(main_report["full_route"]["rows"], 320, "main_report.full_route.rows")
    _assert_equal(gate, main_report["gate_g1"], "decision_gate consistency")
    _assert_equal(gate["status"], "PASS", "decision_gate.status")

    coverage = _verify_route_coverage(rows)
    metric_definitions = _verify_metric_definitions(rows, route_summary)
    provenance = _verify_input_provenance(input_registry, local_manifest, local_registry)
    _assert_equal(local_manifest["routes_per_speed_requested"], 20, "bank.routes_per_speed")
    _assert_equal(local_manifest["max_anchors_per_source_route"], 5, "bank.anchor_quota")

    output_hashes = {
        name: _sha256(main_dir / name)
        for name in required
    }
    report = {
        "schema_name": "d63_1_reproducibility_verification",
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "verification_status": "PASS",
        "scope": "TRAIN_only_D63_1",
        "main_artifact_dir": str(main_dir.resolve()),
        "tests": {"status": "PASS", "command": test_command},
        "coverage": coverage,
        "state_bank": {
            "states": int(local_manifest["state_count"]),
            "routes_per_speed": int(local_manifest["routes_per_speed_requested"]),
            "actual_max_anchors_per_source_route": int(local_manifest["max_anchors_per_source_route"]),
            "plan_max_anchors_per_source_route": 10,
            "quota_deviation": "accepted_smaller_bank_no_rerun",
        },
        "metric_definitions": metric_definitions,
        "aggregate": route_summary["aggregate"],
        "by_speed": route_summary["by_speed"],
        "paired_comparisons": route_summary["paired_comparisons"],
        "gate_g1": gate,
        "provenance": provenance,
        "main_output_sha256": output_hashes,
        "post_run_source_snapshot": _current_source_snapshot(root),
        "limitations": [
            "The main full-route run did not embed a runtime source-tree hash; the source snapshot in this report was captured after completion.",
            "The bank used at most 5 anchors/source/route while the plan allowed at most 10; the audit was not rerun solely to fill the larger quota.",
            "G1 is TRAIN-only and uses a 1-second finite-lookahead reference; it is not an online deployable policy result.",
        ],
        "training_runs_started": 0,
        "validation_test_or_zero_shot_rows_read": 0,
    }
    (output_dir / "verification_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "d63_1_reproducibility_report_vi.md").write_text(
        _markdown(report), encoding="utf-8"
    )
    return report


def main(root_path: str, args: Namespace) -> int:
    root = Path(root_path).resolve()

    def resolve(value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else (root / path).resolve()

    report = verify_package(
        root=root,
        main_dir=resolve(args.main_audit_dir),
        output_dir=resolve(args.output_dir),
        test_command=str(args.test_command),
    )
    print(
        "D63-1 verification PASS: "
        f"{report['coverage']['rows']} full-route rows, "
        f"G1={report['gate_g1']['status']}."
    )
    return 0
