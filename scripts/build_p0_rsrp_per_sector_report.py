"""Expand a P0 factor audit into one row per route and sector."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd


FACTOR_FIELDS = {
    "antenna_gain_db": "gtx_range_db",
    "fspl_db": "fspl_range_db",
    "excess_loss_db": "los_nlos_range_db",
    "shadow_loss_db": "shadow_range_db",
    "fast_fading_gain_db": "fast_range_db",
}

FACTOR_LABELS = {
    "gtx_range_db": "Gtx",
    "fspl_range_db": "FSPL",
    "los_nlos_range_db": "LoS/NLoS",
    "shadow_range_db": "shadow",
    "fast_range_db": "fast fading",
}

STEP_FACTOR_FIELDS = {
    "antenna_gain_db": "gtx",
    "fspl_db": "fspl",
    "excess_loss_db": "los_nlos",
    "shadow_loss_db": "shadow",
    "fast_fading_gain_db": "fast",
}


def _sector_label(row: dict[str, float | int]) -> str:
    return f"`bs_{row['sector_id']}` (site {row['site_id']}, {row['bearing_deg']:.0f}°)"


def _factor_breakdown(row: dict[str, float | int]) -> str:
    factors = sorted(
        ((float(row[field]), label) for field, label in FACTOR_LABELS.items()),
        reverse=True,
    )
    return ", ".join(f"Δ{label} {value:.2f} dB" for value, label in factors)


def _format_sector_shares(rows: list[dict[str, float | int]], field: str) -> str:
    return ", ".join(
        f"`bs_{row['sector_id']}` {row[field]:.1f}%"
        for row in sorted(rows, key=lambda item: float(item[field]), reverse=True)
    )


def _route_evaluation_lines(group: list[dict[str, float | int]]) -> list[str]:
    """Write a route-specific interpretation without pooling sectors."""
    frequent = [row for row in group if float(row["top3_percent"]) >= 10.0]
    top1 = [row for row in group if float(row["top1_percent"]) > 0.0]
    top1_switches = int(group[0]["measurement_best_switches"])
    unique_top1 = int(group[0]["unique_measurement_best_sectors"])
    highest_range = max(group, key=lambda row: float(row["rsrp_range_db"]))
    highest_candidate = max(frequent, key=lambda row: float(row["rsrp_range_db"]))
    widest_geometry = max(frequent, key=lambda row: float(row["fspl_range_db"]))
    strongest_component = max(
        frequent,
        key=lambda row: max(float(row[field]) for field in FACTOR_LABELS),
    )
    strongest_component_field = max(FACTOR_LABELS, key=lambda field: float(strongest_component[field]))
    noisiest_candidate = max(
        frequent, key=lambda row: float(row["rsrp_mean_abs_step_delta_db"])
    )
    burstiest_candidate = max(
        frequent, key=lambda row: float(row["rsrp_p95_abs_step_delta_db"])
    )
    candidate_sites = sorted({int(row["site_id"]) for row in frequent})
    candidate_share_text = _format_sector_shares(frequent, "top3_percent")
    top1_share_text = _format_sector_shares(top1, "top1_percent")

    lines = [
        "### Đánh giá route",
        "",
        f"- **Các candidate đo đạc:** {len(frequent)} sector ở {len(candidate_sites)} site có mặt trong top 3 ít nhất 10% thời gian: {candidate_share_text}.",
        f"- **Sector mạnh nhất theo từng mẫu:** có {unique_top1} sector từng đứng thứ nhất; tỉ lệ đứng thứ nhất là {top1_share_text}. Identity của sector có RSRP lớn nhất đổi {top1_switches} lần trong 5.999 chuyển tiếp mẫu. Đây là thay đổi của phép đo RSRP, không phải số handover mà Agent phải thực hiện.",
        f"- **Candidate có RSRP biến thiên lớn nhất:** {_sector_label(highest_candidate)} đi từ {highest_candidate['rsrp_min_dbm']:.2f} đến {highest_candidate['rsrp_max_dbm']:.2f} dBm, ΔRSRP = {highest_candidate['rsrp_range_db']:.2f} dB. Các biên độ thành phần riêng: {_factor_breakdown(highest_candidate)}.",
        f"- **Hình học của candidate:** {_sector_label(widest_geometry)} có ΔFSPL lớn nhất trong nhóm candidate, {widest_geometry['fspl_range_db']:.2f} dB, khi khoảng cách 3D đổi từ {widest_geometry['distance_3d_min_m']:.1f} đến {widest_geometry['distance_3d_max_m']:.1f} m. Đây là phần biến thiên xác định trực tiếp từ chuyển động; Gtx/shadow/fading còn phụ thuộc hướng hoặc trạng thái kênh.",
        f"- **Thành phần dao động riêng lớn nhất trong nhóm candidate:** {_sector_label(strongest_component)} có Δ{FACTOR_LABELS[strongest_component_field]} = {strongest_component[strongest_component_field]:.2f} dB. Không cộng các Δ trong bảng để suy ra ΔRSRP, vì cực đại và cực tiểu của các thành phần thường nằm ở các thời điểm khác nhau.",
        f"- **Biến thiên giữa timestep:** {_sector_label(noisiest_candidate)} có trung bình |δRSRP| lớn nhất trong nhóm candidate, {noisiest_candidate['rsrp_mean_abs_step_delta_db']:.2f} dB/10 ms. Bước nhảy P95 lớn nhất thuộc {_sector_label(burstiest_candidate)}, {burstiest_candidate['rsrp_p95_abs_step_delta_db']:.2f} dB/10 ms. Các số này đo tín hiệu thay đổi cục bộ giữa hai mẫu liền kề, khác với ΔRSRP là biên độ cả route.",
    ]
    if float(highest_range["top3_percent"]) < 10.0:
        lines.append(
            f"- **Cực trị ngoài vùng cạnh tranh:** {_sector_label(highest_range)} có ΔRSRP lớn nhất toàn bảng ({highest_range['rsrp_range_db']:.2f} dB) nhưng chỉ vào top 3 {highest_range['top3_percent']:.1f}% thời gian. Nó hữu ích để kiểm tra mô hình kênh, nhưng không nên được dùng một mình để đánh giá khả năng chọn target HO của Agent."
        )
    else:
        lines.append(
            f"- **Cực trị cũng là candidate:** {_sector_label(highest_range)} vừa có ΔRSRP lớn nhất toàn bảng ({highest_range['rsrp_range_db']:.2f} dB), vừa nằm trong top 3 {highest_range['top3_percent']:.1f}% thời gian. Đây là sector cần ưu tiên khi phân tích quyết định HO của route này."
        )
    return lines


def build_rows(audit_dir: Path, data_dir: Path) -> list[dict[str, float | int]]:
    audit = json.loads((audit_dir / "factor_report.json").read_text(encoding="utf-8"))
    if not audit["replay_validation"]["passes_exact_csv_replay"]:
        raise ValueError("Factor audit did not reproduce stored RSRP.")
    summaries = [
        json.loads(line)
        for line in (audit_dir / "route_factor_summary.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    if len(summaries) != audit["dataset"]["selected_routes"]:
        raise ValueError("Route count differs from audit metadata.")

    rows: list[dict[str, float | int]] = []
    for summary in summaries:
        speed = int(summary["speed_kmh"])
        route = int(summary["trajectory_id"])
        sectors = int(summary["cells"])
        csv_path = data_dir / f"rsrp_{speed}kmh_{route}.csv"
        frame = pd.read_csv(csv_path)
        columns = [f"bs_{sector}" for sector in range(sectors)]
        if list(frame.columns[2:]) != columns:
            raise ValueError(f"Unexpected sector columns in {csv_path}")
        rsrp = frame[columns].to_numpy(dtype=float)
        if rsrp.shape[0] != 6000:
            raise ValueError(f"Expected 6000 samples in {csv_path}")
        ranking = np.argsort(-rsrp, axis=1, kind="stable")
        top1_count = np.bincount(ranking[:, 0], minlength=sectors)
        top3_count = np.bincount(ranking[:, :3].ravel(), minlength=sectors)
        top1_switches = int(np.count_nonzero(np.diff(ranking[:, 0])))
        unique_top1_sectors = int(np.count_nonzero(top1_count))
        rsrp_abs_step_delta = np.abs(np.diff(rsrp, axis=0))

        for sector in range(sectors):
            factor_ranges = {
                output_name: float(summary["same_sector_route_range_db"][input_name][sector])
                for input_name, output_name in FACTOR_FIELDS.items()
            }
            factor_step_delta = {
                f"{prefix}_{metric}_abs_step_delta_db": float(
                    summary["same_sector_step_abs_delta_db"][input_name][
                        f"{metric}_abs_by_sector"
                    ][sector]
                )
                for input_name, prefix in STEP_FACTOR_FIELDS.items()
                for metric in ("mean", "p95", "p99", "max")
            }
            replay_range = float(
                summary["same_sector_route_range_db"]["rsrp_dbm"][sector]
            )
            csv_range = float(np.ptp(rsrp[:, sector]))
            if abs(replay_range - csv_range) > 0.02:
                raise ValueError(
                    f"RSRP range disagrees after CSV rounding: {csv_path}, bs_{sector}"
                )
            rows.append({
                "speed_kmh": speed,
                "route_id": route,
                "sector_id": sector,
                "site_id": sector // 3,
                "site_sector_id": sector % 3,
                "bearing_deg": float((sector % 3) * 120),
                "travel_distance_m": float(summary["travel_distance_m"]),
                "distance_3d_min_m": float(
                    summary["distance_3d_m"]["minimum_by_sector"][sector]
                ),
                "distance_3d_max_m": float(
                    summary["distance_3d_m"]["maximum_by_sector"][sector]
                ),
                "distance_3d_range_m": float(
                    summary["same_sector_distance_route_range_m"][sector]
                ),
                **factor_ranges,
                "rsrp_min_dbm": float(np.min(rsrp[:, sector])),
                "rsrp_max_dbm": float(np.max(rsrp[:, sector])),
                "rsrp_range_db": csv_range,
                "rsrp_mean_abs_step_delta_db": float(
                    np.mean(rsrp_abs_step_delta[:, sector])
                ),
                "rsrp_p95_abs_step_delta_db": float(
                    np.percentile(rsrp_abs_step_delta[:, sector], 95)
                ),
                "rsrp_p99_abs_step_delta_db": float(
                    np.percentile(rsrp_abs_step_delta[:, sector], 99)
                ),
                "rsrp_max_abs_step_delta_db": float(
                    np.max(rsrp_abs_step_delta[:, sector])
                ),
                **factor_step_delta,
                "top1_steps": int(top1_count[sector]),
                "top3_steps": int(top3_count[sector]),
                "top1_percent": float(100 * top1_count[sector] / len(rsrp)),
                "top3_percent": float(100 * top3_count[sector] / len(rsrp)),
                "measurement_best_switches": top1_switches,
                "unique_measurement_best_sectors": unique_top1_sectors,
            })
    return sorted(rows, key=lambda row: (row["speed_kmh"], row["route_id"], row["sector_id"]))


def write_report(rows: list[dict[str, float | int]], csv_path: Path, markdown_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# P0: từng sector trên từng route",
        "",
        "Mỗi hàng cố định một `bs_i` trong một route 60 giây (6.000 mẫu). Các cột Δ là `max − min` theo thời gian của **chính sector đó**. Không lấy cực trị sau khi gộp 21 sector. Khoảng cách là khoảng cách 3D từ UAV tới site phát. Gtx, FSPL, LoS/NLoS, shadow, fast và RSRP có đơn vị dB; khoảng cách có đơn vị mét.",
        "",
        "Biến thiên giữa hai timestep liên tiếp dùng ký hiệu `|δ| = |x[t] − x[t−1]|`, với khoảng lấy mẫu 10 ms. `mean |δ|` là thay đổi điển hình mỗi bước; P95 là mức chỉ vượt ở 5% bước; P99 và max cho thấy các bước nhảy hiếm. Các chỉ số `|δRSRP|` tính trực tiếp từ CSV; các chỉ số thành phần tính từ phép phát lại generator.",
        "",
        "Có 7 site, mỗi site có 3 sector hướng 0°, 120°, 240° (`bs_0`–`bs_2` cùng site 0, v.v.). Ba sector cùng site chia sẻ vị trí, nên khoảng cách và FSPL giống nhau; cấu hình P0 còn chia sẻ trạng thái LoS và shadow theo site. Gtx khác nhau vì hướng anten khác nhau; fast fading có trạng thái riêng theo sector. Vì vậy cần đọc cả `bs_i`, không chỉ site.",
        "",
        "`Top 3 %` là tỉ lệ bước mà sector nằm trong ba sector có RSRP cao nhất, tính từ CSV đã làm tròn. Biên độ Δ vẫn dùng **toàn bộ route**, kể cả khi sector không nằm trong top 3. CSV chi tiết có thêm RSRP nhỏ nhất/lớn nhất và tỉ lệ đứng thứ nhất.",
        "",
        "Ngay dưới mỗi bảng có phần đánh giá riêng cho route. “Candidate” trong phần này chỉ là sector nằm trong top 3 ít nhất 10% thời gian; đây là tiêu chí đọc dữ liệu, không phải luật HO. “Sector mạnh nhất theo mẫu” cũng chỉ là argmax RSRP tức thời, chưa áp dụng A3, hysteresis hay TTT.",
        "",
        "ΔRSRP, RSRP nhỏ nhất/lớn nhất và thứ hạng được tính trực tiếp từ CSV dataset (đã làm tròn 0,01 dB); biên độ các thành phần được tính từ phép phát lại generator. Chênh lệch biên độ RSRP giữa hai nguồn được kiểm tra không quá 0,02 dB.",
        "",
        f"Tổng cộng **{len(rows)} hàng sector–route**. Dấu chấm trong bảng là dấu thập phân.",
        "",
    ]
    keys = sorted({(int(row["speed_kmh"]), int(row["route_id"])) for row in rows})
    for speed, route in keys:
        group = [row for row in rows if row["speed_kmh"] == speed and row["route_id"] == route]
        highest_range = max(group, key=lambda row: row["rsrp_range_db"])
        most_often_top3 = max(group, key=lambda row: row["top3_percent"])
        frequent_candidates = [row for row in group if row["top3_percent"] >= 10.0]
        highest_candidate_range = max(
            frequent_candidates, key=lambda row: row["rsrp_range_db"]
        )
        lines.extend([
            f"## {speed} km/h · route {route}",
            "",
            f"UAV đi khoảng **{group[0]['travel_distance_m']:.1f} m**. Biên độ RSRP lớn nhất thuộc `bs_{highest_range['sector_id']}`: **{highest_range['rsrp_range_db']:.2f} dB**, xuất hiện trong top 3 **{highest_range['top3_percent']:.1f}%** số bước. Sector xuất hiện trong top 3 nhiều nhất là `bs_{most_often_top3['sector_id']}` (**{most_often_top3['top3_percent']:.1f}%**), với biên độ RSRP **{most_often_top3['rsrp_range_db']:.2f} dB**.",
            "",
            f"Nếu chỉ đánh dấu các sector nằm trong top 3 ít nhất 10% thời gian, route này có **{len(frequent_candidates)}/21 sector**. Trong nhóm đó, `bs_{highest_candidate_range['sector_id']}` có biên độ RSRP lớn nhất (**{highest_candidate_range['rsrp_range_db']:.2f} dB**). Ngưỡng 10% chỉ giúp đọc bảng; mọi Δ trong bảng vẫn tính trên toàn route.",
            "",
            "| BS (site/hướng) | d3 gần → xa (m) | ΔGtx | ΔFSPL | ΔLoS | ΔShadow | ΔFast | ΔRSRP | mean abs(δRSRP) | P95 abs(δRSRP) | Top 3 % |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for row in group:
            lines.append(
                f"| `bs_{row['sector_id']}` ({row['site_id']}/{row['bearing_deg']:.0f}°) "
                f"| {row['distance_3d_min_m']:.1f} → {row['distance_3d_max_m']:.1f} "
                f"| {row['gtx_range_db']:.2f} | {row['fspl_range_db']:.2f} "
                f"| {row['los_nlos_range_db']:.0f} | {row['shadow_range_db']:.2f} "
                f"| {row['fast_range_db']:.2f} | {row['rsrp_range_db']:.2f} "
                f"| {row['rsrp_mean_abs_step_delta_db']:.2f} "
                f"| {row['rsrp_p95_abs_step_delta_db']:.2f} "
                f"| {row['top3_percent']:.1f} |"
            )
        lines.extend([
            "",
            "### Biến thiên từng thành phần giữa timestep 10 ms của candidate",
            "",
            "Mỗi giá trị là `abs(x[t] − x[t−1])`, đơn vị dB/10 ms. Bảng đầu cho mức thay đổi điển hình; bảng sau cho bước nhảy P95. P99 và max của mọi thành phần vẫn có trong CSV chi tiết.",
            "",
            "| BS | mean abs(δGtx) | mean abs(δFSPL) | mean abs(δLoS) | mean abs(δShadow) | mean abs(δFast) | mean abs(δRSRP) |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ])
        for row in sorted(frequent_candidates, key=lambda item: float(item["top3_percent"]), reverse=True):
            lines.append(
                f"| `bs_{row['sector_id']}` "
                f"| {row['gtx_mean_abs_step_delta_db']:.3f} "
                f"| {row['fspl_mean_abs_step_delta_db']:.3f} "
                f"| {row['los_nlos_mean_abs_step_delta_db']:.3f} "
                f"| {row['shadow_mean_abs_step_delta_db']:.3f} "
                f"| {row['fast_mean_abs_step_delta_db']:.3f} "
                f"| {row['rsrp_mean_abs_step_delta_db']:.3f} |"
            )
        lines.extend([
            "",
            "| BS | P95 abs(δGtx) | P95 abs(δFSPL) | P95 abs(δLoS) | P95 abs(δShadow) | P95 abs(δFast) | P95 abs(δRSRP) |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ])
        for row in sorted(frequent_candidates, key=lambda item: float(item["top3_percent"]), reverse=True):
            lines.append(
                f"| `bs_{row['sector_id']}` "
                f"| {row['gtx_p95_abs_step_delta_db']:.3f} "
                f"| {row['fspl_p95_abs_step_delta_db']:.3f} "
                f"| {row['los_nlos_p95_abs_step_delta_db']:.3f} "
                f"| {row['shadow_p95_abs_step_delta_db']:.3f} "
                f"| {row['fast_p95_abs_step_delta_db']:.3f} "
                f"| {row['rsrp_p95_abs_step_delta_db']:.3f} |"
            )
        lines.extend(["", *_route_evaluation_lines(group), ""])
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    rows = build_rows(args.audit_dir, args.data_dir)
    csv_path = args.audit_dir / "sector_route_metrics.csv"
    write_report(rows, csv_path, args.report)
    print(f"Wrote {len(rows)} sector-route rows to {csv_path} and {args.report}")


if __name__ == "__main__":
    main()
