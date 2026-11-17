"""Classify large adjacent-sample RSRP jumps in the complete P0 dataset."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from uav_simulation.validation.rsrp_factor_audit import _manifest_config
from uav_simulation.simulation import CellularNetworkReceivedPower


COMPONENTS = ("Gtx", "FSPL", "LoS/NLoS", "shadow", "fast fading")


def _make_sim(parameters: dict[str, Any], speed_kmh: int, route_id: int) -> CellularNetworkReceivedPower:
    return CellularNetworkReceivedPower(
        num_ues=1,
        rect_len_m=float(parameters["width_m"]),
        rect_wid_m=float(parameters["height_m"]),
        grid_spacing_m=float(parameters["grid_spacing_m"]),
        uav_ue_altitude_m=float(parameters["altitude_m"]),
        uav_ue_speed_kmh=float(speed_kmh),
        seed=speed_kmh * 1000 + route_id,
        headless=True,
        train_ready_output=True,
    )


def _gain_cache(sim: CellularNetworkReceivedPower, steps: int) -> np.ndarray:
    """Generate the route once, then evaluate each 3D pattern in a vectorized call."""
    positions = np.empty((steps, 2), dtype=float)
    for frame in range(steps):
        sim.current_frame = frame
        sim.mobility.step_manhattan_on_grid(
            0, float(sim.ue_speeds_ms[0] * sim.time_per_step)
        )
        positions[frame] = sim.ue_positions[0]

    gains = np.empty((steps, len(sim.bs_positions)), dtype=float)
    for bs_idx, (bs_x, bs_y) in enumerate(sim.bs_positions):
        if not sim.use_3gpp_3d_antenna:
            gains[:, bs_idx] = float(sim.gtx)
            continue
        dx = positions[:, 0] - float(bs_x)
        dy = positions[:, 1] - float(bs_y)
        dz = float(sim.h_ut - sim.bs_heights[bs_idx])
        distance = np.sqrt(dx * dx + dy * dy + dz * dz)
        theta = np.rad2deg(np.arccos(np.clip(dz / distance, -1.0, 1.0)))
        phi = np.rad2deg(np.arctan2(dy, dx))
        gains[:, bs_idx] = sim.bs_antennas[bs_idx].power_gain_db_gcs(theta, phi)
    return gains


def _route_stats(payload: tuple[str, dict[str, Any], dict[str, Any], float]) -> dict[str, Any]:
    data_dir_raw, manifest, record, threshold_db = payload
    data_dir = Path(data_dir_raw)
    parameters = manifest["parameters"]
    speed = int(record["speed_kmh"])
    route_id = int(record["trajectory_id"])
    source = data_dir / str(record["rsrp_file"])
    stored = pd.read_csv(source).iloc[:, 2:].to_numpy(dtype=float)
    steps, cells = stored.shape
    if steps != int(parameters["steps"]):
        raise ValueError(f"Unexpected sample count in {source}")
    event_mask = np.abs(np.diff(stored, axis=0)) > threshold_db

    primary_count = Counter()
    primary_with_los_switch = Counter()
    absolute_contribution_sum = Counter()
    signed_contribution_sum = Counter()
    top_events: list[dict[str, Any]] = []
    event_count = 0
    los_switch_events = 0
    max_replay_error = 0.0

    with _manifest_config(manifest):
        geometry_sim = _make_sim(parameters, speed, route_id)
        gains = _gain_cache(geometry_sim, steps)
        sim = _make_sim(parameters, speed, route_id)
        sim.radio_model.calculate_bs_antenna_gain_idx = (
            lambda ue_x, ue_y, bs_idx: float(gains[sim.current_frame, bs_idx])
        )
        previous: dict[str, np.ndarray] | None = None
        for frame in range(steps):
            sim.current_frame = frame
            sim.mobility.step_manhattan_on_grid(
                0, float(sim.ue_speeds_ms[0] * sim.time_per_step)
            )
            rsrp, _ = sim.radio_model.calculate_all_bs_measurements(0)
            current = sim.radio_model.last_measurement_components
            max_replay_error = max(
                max_replay_error,
                float(np.max(np.abs(np.round(rsrp, 2) - stored[frame]))),
            )
            if previous is not None:
                selected = np.flatnonzero(event_mask[frame - 1])
                if selected.size:
                    contributions = np.stack((
                        current["antenna_gain_db"] - previous["antenna_gain_db"],
                        -(current["fspl_db"] - previous["fspl_db"]),
                        -(current["excess_loss_db"] - previous["excess_loss_db"]),
                        -(current["shadow_loss_db"] - previous["shadow_loss_db"]),
                        current["fast_fading_gain_db"] - previous["fast_fading_gain_db"],
                    ))
                    selected_contributions = contributions[:, selected]
                    primary = np.argmax(np.abs(selected_contributions), axis=0)
                    los_changed = (
                        current["is_los"][selected] != previous["is_los"][selected]
                    )
                    event_count += int(selected.size)
                    los_switch_events += int(np.count_nonzero(los_changed))
                    for index, component in enumerate(COMPONENTS):
                        values = selected_contributions[index]
                        absolute_contribution_sum[component] += float(np.sum(np.abs(values)))
                        signed_contribution_sum[component] += float(np.sum(values))
                    for primary_index, changed in zip(primary, los_changed):
                        component = COMPONENTS[int(primary_index)]
                        primary_count[component] += 1
                        if changed:
                            primary_with_los_switch[component] += 1
                    for sector in selected:
                        jump_db = float(stored[frame, sector] - stored[frame - 1, sector])
                        record_event = {
                            "abs_rsrp_delta_db": abs(jump_db),
                            "signed_rsrp_delta_db": jump_db,
                            "speed_kmh": speed,
                            "route_id": route_id,
                            "sector_id": int(sector),
                            "from_sample": frame - 1,
                            "to_sample": frame,
                            "is_los_before": bool(previous["is_los"][sector]),
                            "is_los_after": bool(current["is_los"][sector]),
                            "signed_component_contributions_db": {
                                component: float(contributions[index, sector])
                                for index, component in enumerate(COMPONENTS)
                            },
                        }
                        top_events.append(record_event)
            previous = {
                name: np.asarray(current[name], dtype=float).copy()
                for name in (
                    "antenna_gain_db", "fspl_db", "excess_loss_db",
                    "shadow_loss_db", "fast_fading_gain_db", "is_los",
                )
            }

    top_events.sort(key=lambda item: item["abs_rsrp_delta_db"], reverse=True)
    return {
        "speed_kmh": speed,
        "route_id": route_id,
        "events": event_count,
        "primary_count": dict(primary_count),
        "primary_with_los_switch": dict(primary_with_los_switch),
        "los_switch_events": los_switch_events,
        "absolute_contribution_sum": dict(absolute_contribution_sum),
        "signed_contribution_sum": dict(signed_contribution_sum),
        "max_replay_error_db": max_replay_error,
        "top_events": top_events[:10],
        "sector_step_transitions": (steps - 1) * cells,
    }


def _combine(route_stats: list[dict[str, Any]], threshold_db: float) -> dict[str, Any]:
    primary_count = Counter()
    primary_with_los_switch = Counter()
    absolute_contribution_sum = Counter()
    signed_contribution_sum = Counter()
    by_speed: dict[int, dict[str, Any]] = {}
    top_events: list[dict[str, Any]] = []
    for route in route_stats:
        primary_count.update(route["primary_count"])
        primary_with_los_switch.update(route["primary_with_los_switch"])
        absolute_contribution_sum.update(route["absolute_contribution_sum"])
        signed_contribution_sum.update(route["signed_contribution_sum"])
        top_events.extend(route["top_events"])
        speed = int(route["speed_kmh"])
        bucket = by_speed.setdefault(speed, {"events": 0, "transitions": 0, "primary": Counter()})
        bucket["events"] += int(route["events"])
        bucket["transitions"] += int(route["sector_step_transitions"])
        bucket["primary"].update(route["primary_count"])

    events = sum(int(route["events"]) for route in route_stats)
    transitions = sum(int(route["sector_step_transitions"]) for route in route_stats)
    top_events.sort(key=lambda item: item["abs_rsrp_delta_db"], reverse=True)
    primary_table = []
    for component in COMPONENTS:
        count = int(primary_count[component])
        primary_table.append({
            "component": component,
            "events_primary": count,
            "share_percent": 100.0 * count / events if events else 0.0,
            "events_with_los_state_change": int(primary_with_los_switch[component]),
            "mean_abs_component_contribution_db_per_event": (
                float(absolute_contribution_sum[component] / events) if events else 0.0
            ),
            "net_signed_component_contribution_db": float(signed_contribution_sum[component]),
        })
    return {
        "status": "COMPLETE",
        "threshold_abs_rsrp_delta_db": threshold_db,
        "routes": len(route_stats),
        "sector_step_transitions": transitions,
        "events": events,
        "event_rate_percent": 100.0 * events / transitions if transitions else 0.0,
        "max_replay_error_db_after_csv_rounding": max(
            float(route["max_replay_error_db"]) for route in route_stats
        ),
        "primary_cause": primary_table,
        "events_with_los_state_change": sum(int(route["los_switch_events"]) for route in route_stats),
        "by_speed": {
            str(speed): {
                "events": item["events"],
                "event_rate_percent": 100.0 * item["events"] / item["transitions"],
                "primary_cause": dict(item["primary"]),
            }
            for speed, item in sorted(by_speed.items())
        },
        "largest_events": top_events[:20],
        "method": {
            "primary_cause": "The component with the largest absolute signed contribution to that event's RSRP delta.",
            "los_switch": "Counts events where the realized LoS state changes between the two samples.",
            "caution": "Multiple components can contribute to one event; primary-cause counts are a classification, not an additive causal partition.",
        },
    }


def _write_markdown(report: dict[str, Any], target: Path) -> None:
    lines = [
        f"# P0: nguyên nhân các bước nhảy RSRP lớn hơn {report['threshold_abs_rsrp_delta_db']:.0f} dB",
        "",
        f"Phạm vi: **{report['routes']} route**, {report['sector_step_transitions']:,} cặp sector–timestep. Một sự kiện thỏa `abs(RSRP[t] - RSRP[t-1]) > {report['threshold_abs_rsrp_delta_db']:.0f} dB`.",
        "",
        f"Có **{report['events']:,} sự kiện** ({report['event_rate_percent']:.3f}% số cặp). Mỗi sự kiện được gán nguyên nhân chính là thành phần có đóng góp tuyệt đối lớn nhất vào delta RSRP; các thành phần còn lại vẫn có thể cùng gây ra bước nhảy.",
        "",
        "| Nguyên nhân chính | Sự kiện | Tỷ lệ | Có chuyển LoS/NLoS | Mean đóng góp tuyệt đối / sự kiện (dB) |",
        "|---|---:|---:|---:|---:|",
    ]
    for item in report["primary_cause"]:
        lines.append(
            f"| {item['component']} | {item['events_primary']:,} | {item['share_percent']:.2f}% "
            f"| {item['events_with_los_state_change']:,} | {item['mean_abs_component_contribution_db_per_event']:.2f} |"
        )
    lines.extend([
        "",
        f"Số sự kiện có thay đổi trạng thái LoS/NLoS: **{report['events_with_los_state_change']:,}**.",
        "",
        "## Theo tốc độ",
        "",
        "| Tốc độ (km/h) | Sự kiện | Tỷ lệ cặp sector–timestep | Nguyên nhân chính nhiều nhất |",
        "|---:|---:|---:|---|",
    ])
    for speed, item in report["by_speed"].items():
        primary = max(item["primary_cause"].items(), key=lambda pair: pair[1])[0]
        lines.append(
            f"| {speed} | {item['events']:,} | {item['event_rate_percent']:.3f}% | {primary} |"
        )
    lines.extend([
        "",
        "## Các sự kiện lớn nhất",
        "",
        "| abs(δRSRP) | Tốc độ | Route | Sector | Sample | LoS trước → sau | Đóng góp có dấu: Gtx / FSPL / LoS / shadow / fast (dB) |",
        "|---:|---:|---:|---:|---:|---|---|",
    ])
    for item in report["largest_events"][:10]:
        values = item["signed_component_contributions_db"]
        lines.append(
            f"| {item['abs_rsrp_delta_db']:.2f} | {item['speed_kmh']} | {item['route_id']} "
            f"| bs_{item['sector_id']} | {item['from_sample']}→{item['to_sample']} "
            f"| {item['is_los_before']} → {item['is_los_after']} "
            f"| {values['Gtx']:.2f} / {values['FSPL']:.2f} / {values['LoS/NLoS']:.2f} / {values['shadow']:.2f} / {values['fast fading']:.2f} |"
        )
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold-db", type=float, default=10.0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--speeds", type=int, nargs="+", help="Optional speed filter.")
    parser.add_argument("--route-ids", type=int, nargs="+", help="Optional route-ID filter.")
    args = parser.parse_args()
    if args.threshold_db <= 0.0:
        raise ValueError("--threshold-db must be positive.")
    if args.workers <= 0:
        raise ValueError("--workers must be positive.")
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output: {args.output_dir}")
    manifest = json.loads((args.data_dir / "simulation_manifest.json").read_text(encoding="utf-8"))
    metadata = json.loads((args.data_dir / "dataset_metadata.json").read_text(encoding="utf-8"))
    if manifest["parameters"].get("profile") != "p0_canonical_v1":
        raise ValueError("This analysis accepts only P0 canonical data.")
    records = metadata.get("trajectories")
    if not isinstance(records, list) or not records:
        raise ValueError("No dataset trajectories found.")
    selected_speeds = None if args.speeds is None else set(args.speeds)
    selected_route_ids = None if args.route_ids is None else set(args.route_ids)
    records = [
        record for record in records
        if (selected_speeds is None or int(record["speed_kmh"]) in selected_speeds)
        and (selected_route_ids is None or int(record["trajectory_id"]) in selected_route_ids)
    ]
    if not records:
        raise ValueError("No trajectories match the filters.")
    args.output_dir.mkdir(parents=True)
    payloads = [(str(args.data_dir.resolve()), manifest, record, args.threshold_db) for record in records]
    route_stats: list[dict[str, Any]] = []
    if args.workers == 1:
        iterator = map(_route_stats, payloads)
    else:
        pool = get_context("spawn").Pool(processes=args.workers)
        iterator = pool.imap_unordered(_route_stats, payloads)
    try:
        for index, item in enumerate(iterator, 1):
            route_stats.append(item)
            print(f"[RSRP jump analysis] {index}/{len(payloads)} speed={item['speed_kmh']} route={item['route_id']}", flush=True)
    finally:
        if args.workers != 1:
            pool.close()
            pool.join()
    report = _combine(route_stats, args.threshold_db)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_markdown(report, args.output_dir / "report_vi.md")
    print(json.dumps({"status": report["status"], "events": report["events"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
