"""TRAIN-only audit of T5A benefit sources and pre/post-A3 opportunity.

This command deliberately separates realised policy behaviour from oracle
counterfactual opportunity.  T5A is replayed deterministically on the frozen
T2 TRAIN routes, while the already materialised T2/T3 counterfactuals are
summarised without opening validation, test, or zero-shot data.
"""

from __future__ import annotations

from argparse import Namespace
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any, Iterable, Sequence

import numpy as np

from ho_optim_drl.agents.ppo import SMDPPPO
from ho_optim_drl.commands.audit_b1_lower_threshold import (
    _aggregate,
    _delta,
    _route_metrics,
)
from ho_optim_drl.commands.audit_baseline_parity import _git_state, _sha256
from ho_optim_drl.commands.audit_early_timing_routes import (
    _extended_aggregate,
    _load_route_split,
    paired_bootstrap,
)
from ho_optim_drl.data import TracePair, load_radio_trace
from ho_optim_drl.data.project import load_protocol_config
from ho_optim_drl.envs import HandoverSMDPEnv
from ho_optim_drl.protocol import A3ReferenceSimulator
from ho_optim_drl.smdp import HandoverAction, reward_config_from_dict


SCHEMA_NAME = "t5a_benefit_source_train_audit"
SCHEMA_VERSION = 1
RISK_FIELDS = (
    "handover_failures",
    "ping_pong_count",
    "rlf_starts",
    "recovery_seconds",
    "interruption_seconds",
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _policy_probabilities(
    agent: SMDPPPO, observation: np.ndarray, action_mask: np.ndarray
) -> tuple[int, float, np.ndarray]:
    action, _, value, probabilities = agent.model.act_with_probabilities(
        observation,
        device=agent.device,
        deterministic=True,
        action_mask=action_mask,
    )
    return action, value, probabilities


def _checkpoint_label(path: Path, extra: dict[str, Any]) -> str:
    experiment = extra.get("experiment_config", {})
    seed = (
        experiment.get("seeds", {}).get("training")
        if isinstance(experiment, dict)
        else None
    )
    if seed is None and isinstance(experiment, dict):
        seed = experiment.get("ppo", {}).get("seed")
    return f"t5a_s{seed}" if seed is not None else f"t5a_{path.parent.name}"


def _objective_environment(
    *, data_dir: Path, pairs: Sequence[TracePair], agent: SMDPPPO,
    extra: dict[str, Any], protocol_config: Any,
) -> HandoverSMDPEnv:
    objective = extra.get("objective", {})
    reward_schema = str(objective.get("reward", "trajectory_qos_recovery_reward_v1"))
    env = HandoverSMDPEnv(
        protocol_config,
        data_dir=str(data_dir),
        trace_pairs=pairs,
        gamma_per_second=agent.config.gamma_per_second,
        candidate_count=int(objective.get("candidate_count", 1)),
        candidate_order=str(objective.get("candidate_order_key", "ranked_v1")),
        observation_schema=str(objective.get("observation_schema_key", "t4d1_compact_v1")),
        decision_trigger=str(objective.get("trigger", "completed_a3_ttt")),
        reward_schema=reward_schema,
        reward_config=reward_config_from_dict(
            reward_schema, objective.get("reward_config")
        ),
        stay_mode=str(objective.get("stay_mode", "a3_redecision_v1")),
        stay_redecision_ms=int(objective.get("stay_redecision_ms", 20)),
    )
    env.load_reward_state(extra.get("reward_state"))
    env.validate_checkpoint_objective(extra)
    if env.action_space.n != 2:
        raise ValueError("T5A benefit audit requires timing-only STAY/HO_C1 checkpoints.")
    return env


def _source_classification(b0: dict[str, Any], policy: dict[str, Any]) -> str:
    rate_delta = float(policy["relative_rate_percent"]) - float(
        b0["relative_rate_percent"]
    )
    reliability = (
        int(policy["handover_failures"]) <= int(b0["handover_failures"])
        and int(policy["ping_pong_count"]) <= int(b0["ping_pong_count"])
        and int(policy["rlf_recovery_samples"]) <= int(b0["rlf_recovery_samples"])
    )
    strictly_better = (
        int(policy["handover_failures"]) < int(b0["handover_failures"])
        or int(policy["ping_pong_count"]) < int(b0["ping_pong_count"])
        or int(policy["rlf_recovery_samples"]) < int(b0["rlf_recovery_samples"])
    )
    if rate_delta >= -1e-12 and reliability and (strictly_better or rate_delta > 1e-12):
        return "pareto_improvement"
    if rate_delta < -1e-12 and reliability and strictly_better:
        return "reliability_gain_rate_tradeoff"
    if rate_delta > 1e-12 and not reliability:
        return "rate_gain_reliability_tradeoff"
    if abs(rate_delta) <= 1e-12 and not strictly_better:
        return "same_as_b0"
    return "regression_or_mixed"


def _evaluate_policy(
    *, env: HandoverSMDPEnv, agent: SMDPPPO,
    pairs: Sequence[TracePair], label: str,
    decision_stream: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    decision_counts = Counter()
    stay_run_lengths: list[int] = []
    margins: list[float] = []
    probabilities_stay: list[float] = []
    for index, pair in enumerate(pairs):
        try:
            observation, _ = env.reset(options={"trace_index": index})
            actionable = True
        except RuntimeError:
            observation = None
            actionable = False
        route_stays = 0
        route_handover = 0
        current_stay_run = 0
        max_stay_run = 0
        while actionable and observation is not None:
            decision = env.engine.decision_point
            if decision is None:
                raise RuntimeError("T5A environment exposed an observation without a decision.")
            mask = env.action_mask()
            action, value, probs = _policy_probabilities(agent, observation, mask)
            selected = HandoverAction(action)
            p_stay = float(probs[0])
            p_ho = float(probs[1])
            margin = p_ho - p_stay
            probabilities_stay.append(p_stay)
            margins.append(margin)
            decision_counts[selected.name] += 1
            if selected is HandoverAction.STAY:
                route_stays += 1
                current_stay_run += 1
                max_stay_run = max(max_stay_run, current_stay_run)
            else:
                route_handover += 1
                if current_stay_run:
                    stay_run_lengths.append(current_stay_run)
                current_stay_run = 0
            record = {
                "policy": label,
                "speed_kmh": int(pair.speed_kmh),
                "trajectory_id": int(pair.trajectory_id),
                "tic": int(decision.tic),
                "serving_cell": int(decision.serving_cell),
                "target_cell": int(decision.target_cell),
                "a3_start_tic": int(decision.a3_start_tic),
                "a3_age_ms": int(
                    (decision.tic - decision.a3_start_tic) * env.config.delta_t_ms
                ),
                "action": selected.name,
                "p_stay": p_stay,
                "p_ho_c1": p_ho,
                "policy_margin_ho_minus_stay": margin,
                "value": float(value),
                "stay_run_before_action": current_stay_run,
            }
            decision_stream.write(json.dumps(record, allow_nan=False) + "\n")
            observation, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                if current_stay_run:
                    stay_run_lengths.append(current_stay_run)
                break
        trace = env.engine.trace
        if trace is None:
            raise RuntimeError("T5A environment did not retain the audited trace.")
        metrics = _route_metrics(env.engine, trace)
        metrics.update({
            "actionable": actionable,
            "stay_actions": route_stays,
            "handover_actions": route_handover,
            "max_consecutive_stays": max_stay_run,
        })
        rows.append({
            "trace_id": f"{pair.speed_kmh}:{pair.trajectory_id}",
            "speed_kmh": int(pair.speed_kmh),
            "trajectory_id": int(pair.trajectory_id),
            label: metrics,
        })
    total_actions = sum(decision_counts.values())
    diagnostics = {
        "decisions": total_actions,
        "stay_actions": int(decision_counts["STAY"]),
        "handover_actions": int(decision_counts["HO_C1"]),
        "stay_percent": (
            100.0 * decision_counts["STAY"] / total_actions if total_actions else 0.0
        ),
        "mean_p_stay": float(np.mean(probabilities_stay)) if probabilities_stay else None,
        "mean_abs_policy_margin": float(np.mean(np.abs(margins))) if margins else None,
        "near_tie_percent_abs_margin_le_0_1": (
            100.0 * np.mean(np.abs(margins) <= 0.1) if margins else None
        ),
        "mean_completed_stay_run": (
            float(np.mean(stay_run_lengths)) if stay_run_lengths else 0.0
        ),
        "max_completed_stay_run": max(stay_run_lengths, default=0),
    }
    return rows, diagnostics


def _safe_gain(control: dict[str, Any], candidate: dict[str, Any]) -> bool:
    return (
        float(candidate["delivered_mbit"]) > float(control["delivered_mbit"]) + 1e-12
        and all(
            float(candidate[field]) <= float(control[field]) + 1e-12
            for field in RISK_FIELDS
        )
    )


def _opportunity_records(
    anchors: Sequence[dict[str, Any]], outcomes: Sequence[dict[str, Any]],
    *, horizon_seconds: float,
) -> list[dict[str, Any]]:
    anchor_by_id = {row["state_id"]: row for row in anchors}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in outcomes:
        if np.isclose(float(row["horizon_seconds"]), horizon_seconds):
            grouped[str(row["state_id"])].append(row)
    records: list[dict[str, Any]] = []
    for state_id, rows in grouped.items():
        anchor = anchor_by_id.get(state_id)
        if anchor is None:
            raise ValueError(f"Outcome has no anchor: {state_id}")
        controls = [row for row in rows if row["branch"] == "baseline_continuation"]
        schedules = [row for row in rows if row["branch"] == "timing_schedule"]
        if len(controls) != 1 or not schedules:
            raise ValueError(f"Incomplete counterfactual branches for {state_id}")
        control = controls[0]["physical"]
        safe = [row for row in schedules if _safe_gain(control, row["physical"])]
        best = max(
            safe,
            key=lambda row: (
                float(row["physical"]["delivered_mbit"]),
                -int(row["requested_delay_ms"]),
            ),
            default=None,
        )
        records.append({
            "state_id": state_id,
            "physical_state_fingerprint": anchor["physical_state_fingerprint"],
            "source": anchor["source"],
            "category": anchor["category"],
            "speed_kmh": int(anchor["speed_kmh"]),
            "trajectory_id": int(anchor["trajectory_id"]),
            "tic": int(anchor["tic"]),
            "has_safe_positive_opportunity": best is not None,
            "best_delay_ms": (
                int(best["requested_delay_ms"]) if best is not None else None
            ),
            "best_delivered_mbit_gain": (
                float(best["physical"]["delivered_mbit"])
                - float(control["delivered_mbit"])
                if best is not None else 0.0
            ),
            "best_relative_rate_gain_pp": (
                float(best["physical"]["relative_rate_percent"])
                - float(control["relative_rate_percent"])
                if best is not None else 0.0
            ),
        })
    return records


def _opportunity_summary(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_category[str(row["category"])].append(row)
    result: dict[str, Any] = {}
    for category, group in sorted(by_category.items()):
        positives = [row for row in group if row["has_safe_positive_opportunity"]]
        result[category] = {
            "anchors": len(group),
            "unique_physical_states": len(
                {row["physical_state_fingerprint"] for row in group}
            ),
            "positive_opportunities": len(positives),
            "positive_percent": 100.0 * len(positives) / len(group),
            "mean_best_delivered_mbit_gain_positive": (
                float(np.mean([row["best_delivered_mbit_gain"] for row in positives]))
                if positives else 0.0
            ),
            "mean_best_relative_rate_gain_pp_positive": (
                float(np.mean([row["best_relative_rate_gain_pp"] for row in positives]))
                if positives else 0.0
            ),
            "best_delay_ms_counts": dict(
                sorted(Counter(row["best_delay_ms"] for row in positives).items())
            ),
        }
    pre = [
        row for row in records
        if row["category"] in {"early_pre_a3", "a3_pending_ttt"}
    ]
    post = [row for row in records if row["category"] == "post_ttt"]
    for name, group in (("pre_a3_combined", pre), ("post_ttt_combined", post)):
        positives = [row for row in group if row["has_safe_positive_opportunity"]]
        result[name] = {
            "anchors": len(group),
            "positive_opportunities": len(positives),
            "positive_percent": 100.0 * len(positives) / len(group) if group else 0.0,
            "mean_gain_mbit_all_anchors": (
                float(np.mean([row["best_delivered_mbit_gain"] for row in group]))
                if group else 0.0
            ),
        }
    return result


def _opportunity_bootstrap(
    records: Sequence[dict[str, Any]], *, resamples: int, seed: int
) -> dict[str, Any]:
    """Paired route-cluster bootstrap of pre-A3 minus post-TTT opportunity."""
    if resamples <= 0:
        raise ValueError("bootstrap resamples must be positive")
    routes_by_speed: dict[int, dict[tuple[int, int], list[dict[str, Any]]]] = (
        defaultdict(lambda: defaultdict(list))
    )
    for row in records:
        key = (int(row["speed_kmh"]), int(row["trajectory_id"]))
        routes_by_speed[key[0]][key].append(row)

    def statistics(rows: Sequence[dict[str, Any]]) -> dict[str, float]:
        pre = [
            row for row in rows
            if row["category"] in {"early_pre_a3", "a3_pending_ttt"}
        ]
        post = [row for row in rows if row["category"] == "post_ttt"]

        def positive_percent(group: Sequence[dict[str, Any]]) -> float:
            return (
                100.0
                * sum(bool(row["has_safe_positive_opportunity"]) for row in group)
                / len(group)
                if group else 0.0
            )

        def mean_gain(group: Sequence[dict[str, Any]]) -> float:
            return (
                float(np.mean([row["best_delivered_mbit_gain"] for row in group]))
                if group else 0.0
            )

        pre_positive = positive_percent(pre)
        post_positive = positive_percent(post)
        pre_gain = mean_gain(pre)
        post_gain = mean_gain(post)
        return {
            "pre_positive_percent": pre_positive,
            "post_positive_percent": post_positive,
            "delta_pre_minus_post_positive_pp": pre_positive - post_positive,
            "pre_mean_gain_mbit_all_anchors": pre_gain,
            "post_mean_gain_mbit_all_anchors": post_gain,
            "delta_pre_minus_post_mean_gain_mbit": pre_gain - post_gain,
        }

    point = statistics(records)
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = defaultdict(list)
    for _ in range(resamples):
        sampled: list[dict[str, Any]] = []
        for speed in sorted(routes_by_speed):
            route_groups = list(routes_by_speed[speed].values())
            indices = rng.integers(0, len(route_groups), size=len(route_groups))
            for index in indices:
                sampled.extend(route_groups[int(index)])
        for name, value in statistics(sampled).items():
            samples[name].append(value)
    return {
        "point": point,
        "ci95": {
            name: [
                float(np.percentile(values, 2.5)),
                float(np.percentile(values, 97.5)),
            ]
            for name, values in samples.items()
        },
        "resamples": resamples,
        "seed": seed,
        "cluster": "trajectory_within_speed",
    }


def _filter_pairs_per_speed(
    pairs: Sequence[TracePair], limit: int | None
) -> list[TracePair]:
    if limit is None:
        return list(pairs)
    counts: Counter[int] = Counter()
    selected: list[TracePair] = []
    for pair in pairs:
        if counts[int(pair.speed_kmh)] < limit:
            selected.append(pair)
            counts[int(pair.speed_kmh)] += 1
    return selected


def run_audit(
    *, root: Path, data_dir: Path, route_split_path: Path,
    t2_dir: Path, t3_dir: Path, checkpoints: Sequence[Path],
    output_dir: Path, horizon_seconds: float,
    bootstrap_resamples: int, bootstrap_seed: int,
    max_routes_per_speed: int | None,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite T5A audit output: {output_dir}")
    output_dir.mkdir(parents=True)
    all_pairs, route_split = _load_route_split(route_split_path, data_dir)
    pairs = _filter_pairs_per_speed(all_pairs, max_routes_per_speed)
    config = load_protocol_config(str(data_dir))
    anchors_path = t2_dir / "anchors.jsonl"
    outcomes_path = t2_dir / "local_outcomes.jsonl"
    t3_report_path = t3_dir / "main_report.json"
    anchors = _read_jsonl(anchors_path)
    outcomes = _read_jsonl(outcomes_path)
    t3_report = json.loads(t3_report_path.read_text(encoding="utf-8"))
    if t3_report.get("scope") != "train_only":
        raise ValueError("T5A audit accepts only TRAIN T3 evidence.")
    opportunity = _opportunity_records(
        anchors, outcomes, horizon_seconds=horizon_seconds
    )
    opportunity_summary = _opportunity_summary(opportunity)
    opportunity_bootstrap = _opportunity_bootstrap(
        opportunity, resamples=bootstrap_resamples, seed=bootstrap_seed + 1000
    )
    with (output_dir / "opportunity_states.jsonl").open("w", encoding="utf-8") as stream:
        for row in opportunity:
            stream.write(json.dumps(row, allow_nan=False) + "\n")

    started = time.perf_counter()
    baseline_by_key: dict[tuple[int, int], dict[str, Any]] = {}
    for index, pair in enumerate(pairs, 1):
        trace = load_radio_trace(
            config,
            str(data_dir),
            pair.rsrp_file,
            pair.sinr_file,
            speed_kmh=pair.speed_kmh,
        )
        simulator = A3ReferenceSimulator(config)
        simulator.predict(trace)
        baseline_by_key[pair.key] = _route_metrics(simulator, trace)
        print(f"[T5A source audit:B0] {index}/{len(pairs)} {pair.key}", flush=True)

    policy_rows: dict[str, list[dict[str, Any]]] = {}
    policy_diagnostics: dict[str, Any] = {}
    checkpoint_manifest: list[dict[str, Any]] = []
    decision_path = output_dir / "t5a_decisions.jsonl"
    with decision_path.open("w", encoding="utf-8") as decision_stream:
        for checkpoint in checkpoints:
            agent, extra = SMDPPPO.from_checkpoint(checkpoint, device="cpu")
            label = _checkpoint_label(checkpoint, extra)
            if label in policy_rows:
                raise ValueError(f"Duplicate T5A audit label: {label}")
            env = _objective_environment(
                data_dir=data_dir, pairs=pairs, agent=agent,
                extra=extra, protocol_config=config,
            )
            rows, diagnostics = _evaluate_policy(
                env=env, agent=agent, pairs=pairs, label=label,
                decision_stream=decision_stream,
            )
            for row, pair in zip(rows, pairs, strict=True):
                row["b0"] = baseline_by_key[pair.key]
                row["benefit_source"] = _source_classification(
                    row["b0"], row[label]
                )
            policy_rows[label] = rows
            policy_diagnostics[label] = diagnostics
            checkpoint_manifest.append({
                "label": label,
                "path": str(checkpoint.resolve()),
                "sha256": _sha256(checkpoint),
                "experiment_id": extra.get("experiment_config", {}).get(
                    "experiment_id", extra.get("experiment_id")
                ),
                "objective": extra.get("objective"),
            })

    route_path = output_dir / "route_outcomes.jsonl"
    with route_path.open("w", encoding="utf-8") as stream:
        for label, rows in policy_rows.items():
            for row in rows:
                stream.write(json.dumps({
                    "trace_id": row["trace_id"],
                    "speed_kmh": row["speed_kmh"],
                    "trajectory_id": row["trajectory_id"],
                    "policy": label,
                    "b0": row["b0"],
                    "metrics": row[label],
                    "benefit_source": row["benefit_source"],
                }, allow_nan=False) + "\n")

    actual: dict[str, Any] = {}
    for offset, (label, rows) in enumerate(policy_rows.items()):
        aggregate_b0 = _extended_aggregate(rows, "b0", config.delta_t_ms)
        aggregate_policy = _extended_aggregate(rows, label, config.delta_t_ms)
        grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[int(row["speed_kmh"])].append(row)
        actual[label] = {
            "b0": aggregate_b0,
            "policy": aggregate_policy,
            "delta_policy_minus_b0": _delta(aggregate_b0, aggregate_policy),
            "benefit_source_routes": dict(sorted(Counter(
                row["benefit_source"] for row in rows
            ).items())),
            "per_speed": {
                str(speed): {
                    "b0": _extended_aggregate(group, "b0", config.delta_t_ms),
                    "policy": _extended_aggregate(group, label, config.delta_t_ms),
                }
                for speed, group in sorted(grouped.items())
            },
            "paired_bootstrap": paired_bootstrap(
                rows, left="b0", right=label, dt_ms=config.delta_t_ms,
                resamples=bootstrap_resamples, seed=bootstrap_seed + offset,
            ),
            "decision_diagnostics": policy_diagnostics[label],
        }

    report = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "status": "COMPLETE",
        "scope": "train_only",
        "validation_or_test_rows_read": 0,
        "routes": len(pairs),
        "speeds_kmh": sorted({int(pair.speed_kmh) for pair in pairs}),
        "actual_t5a_vs_b0": actual,
        "local_counterfactual_opportunity": {
            "horizon_seconds": horizon_seconds,
            "summary": opportunity_summary,
            "paired_route_cluster_bootstrap": opportunity_bootstrap,
            "semantics": {
                "early_pre_a3_and_pending_ttt": (
                    "safe positive gain from executing before standard A3+TTT "
                    "versus continuing the standard trigger"
                ),
                "post_ttt": (
                    "safe positive gain from delaying a completed-A3 decision "
                    "versus immediate HO"
                ),
            },
        },
        "full_route_oracle": {
            "rpost": t3_report["aggregate"]["rpost"],
            "rearly": t3_report["aggregate"]["rearly"],
            "b0": t3_report["aggregate"]["b0"],
            "rearly_minus_rpost": t3_report["paired_bootstrap"][
                "rearly_minus_rpost"
            ],
            "rpost_minus_b0": t3_report["paired_bootstrap"]["rpost_minus_b0"],
            "g_early": t3_report["gates"]["g_early"],
        },
        "interpretation_contract": [
            "Actual T5A rows measure realised deterministic policy behaviour.",
            "T2/T3 opportunities use future only to create offline oracle labels.",
            "Oracle opportunity is not evidence that PPO learned or can observe it.",
            "This TRAIN audit must not be reported as held-out performance.",
        ],
        "runtime_seconds": time.perf_counter() - started,
    }
    (output_dir / "main_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "validation_or_test_rows_read": 0,
        "data_dir": str(data_dir.resolve()),
        "route_split": {
            "path": str(route_split_path.resolve()),
            "sha256": _sha256(route_split_path),
            "selected_routes": len(pairs),
            "source_routes": len(route_split["selected_routes"]),
        },
        "t2": {
            "anchors": {"path": str(anchors_path.resolve()), "sha256": _sha256(anchors_path)},
            "outcomes": {"path": str(outcomes_path.resolve()), "sha256": _sha256(outcomes_path)},
        },
        "t3": {"path": str(t3_report_path.resolve()), "sha256": _sha256(t3_report_path)},
        "checkpoints": checkpoint_manifest,
        "horizon_seconds": horizon_seconds,
        "bootstrap_resamples": bootstrap_resamples,
        "bootstrap_seed": bootstrap_seed,
        "protocol_config": asdict(config),
        "git": _git_state(root),
        "source_files": [{
            "path": str(Path(__file__).resolve().relative_to(root)),
            "sha256": _sha256(Path(__file__).resolve()),
        }],
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    report_lines = [
        "# T5A benefit-source TRAIN audit",
        "",
        f"- Routes: {len(pairs)}; validation/test rows read: 0.",
        f"- Local counterfactual horizon: {horizon_seconds:g} s.",
    ]
    for label, item in actual.items():
        delta = item["delta_policy_minus_b0"]
        report_lines.append(
            f"- {label}: ΔRR={delta['mean_route_relative_rate_percent']:+.4f} pp; "
            f"ΔHOF={delta['handover_failure_percent']:+.4f} pp; "
            f"ΔPP={delta['ping_pong_percent']:+.4f} pp; "
            f"ΔRLF-time={delta['rlf_time_percent']:+.4f} pp."
        )
    pre = opportunity_summary["pre_a3_combined"]
    post = opportunity_summary["post_ttt"]
    report_lines.extend([
        f"- Safe-positive opportunity: pre-A3={pre['positive_percent']:.2f}% "
        f"vs post-TTT={post['positive_percent']:.2f}% of sampled anchors.",
        "",
        "TRAIN diagnostic only; oracle labels use future and are not policy inputs.",
    ])
    (output_dir / "report_vi.md").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )
    return report


def main(root_path: str, args: Namespace) -> int:
    root = Path(root_path).resolve()
    resolve = lambda value: (Path(value) if Path(value).is_absolute() else root / value).resolve()
    report = run_audit(
        root=root,
        data_dir=resolve(args.data_dir),
        route_split_path=resolve(args.route_split),
        t2_dir=resolve(args.t2_dir),
        t3_dir=resolve(args.t3_dir),
        checkpoints=[resolve(value) for value in args.checkpoints],
        output_dir=resolve(args.output_dir),
        horizon_seconds=float(args.horizon),
        bootstrap_resamples=int(args.bootstrap_resamples),
        bootstrap_seed=int(args.bootstrap_seed),
        max_routes_per_speed=args.max_routes_per_speed,
    )
    print(json.dumps({
        "status": report["status"],
        "routes": report["routes"],
        "actual": report["actual_t5a_vs_b0"],
        "opportunity": report["local_counterfactual_opportunity"]["summary"],
    }, indent=2, allow_nan=False), flush=True)
    return 0
