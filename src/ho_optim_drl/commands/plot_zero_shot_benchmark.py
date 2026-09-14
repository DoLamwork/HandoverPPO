"""Plot a benchmark-by-speed comparison for zero-shot UAV evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


POLICIES = ("3gpp_always_ho", "ppo")
POLICY_LABELS = {
    "3gpp_always_ho": "3GPP Always-HO",
    "ppo": "PPO",
}
POLICY_COLORS = {
    "3gpp_always_ho": "#D55E00",
    "ppo": "#0072B2",
}
POLICY_MARKERS = {
    "3gpp_always_ho": "s",
    "ppo": "o",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path(
            "results/benchmark/uav_120_150_zero_shot/summary.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/benchmark/uav_120_150_zero_shot/"
            "uav_120_150_benchmark_by_speed.png"
        ),
    )
    parser.add_argument("--speeds", type=int, nargs="+", default=[120, 150])
    return parser.parse_args()


def _load_rows(summary_path: Path) -> list[dict[str, object]]:
    with summary_path.open(encoding="utf-8") as summary_file:
        summary = json.load(summary_file)
    return list(summary["by_speed"])


def _metric_values(
    rows: list[dict[str, object]],
    speeds: list[int],
    policy: str,
    field: str,
    *,
    scale: float = 1.0,
) -> list[float]:
    indexed = {
        (int(row["speed_kmh"]), str(row["policy"])): row for row in rows
    }
    values = []
    for speed in speeds:
        key = (speed, policy)
        if key not in indexed:
            raise ValueError(f"Missing {policy} result at {speed} km/h.")
        values.append(float(indexed[key][field]) * scale)
    return values


def main() -> int:
    args = _parse_args()
    rows = _load_rows(args.summary)
    speeds = list(dict.fromkeys(args.speeds))

    metrics = (
        (
            "Relative rate (%)",
            "mean_relative_rate_percent",
            1.0,
            (70.0, 100.0),
        ),
        (
            "Handover failure rate (%)",
            "handover_failure_percent",
            1.0,
            None,
        ),
        ("RLF recovery time (%)", "rlf_time_percent", 1.0, None),
        ("Ping-pong rate (%)", "ping_pong_percent", 1.0, None),
    )

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "axes.titleweight": "bold",
        }
    )
    figure, axes = plt.subplots(2, 2, figsize=(14, 10))
    figure.subplots_adjust(
        left=0.07,
        right=0.98,
        bottom=0.08,
        top=0.89,
        hspace=0.31,
        wspace=0.08,
    )

    for axis, (title, field, scale, fixed_limits) in zip(axes.flat, metrics):
        all_values: list[float] = []
        for policy in ("ppo", "3gpp_always_ho"):
            values = _metric_values(
                rows,
                speeds,
                policy,
                field,
                scale=scale,
            )
            all_values.extend(values)
            axis.plot(
                speeds,
                values,
                color=POLICY_COLORS[policy],
                label=POLICY_LABELS[policy],
                marker=POLICY_MARKERS[policy],
                markersize=8,
                linewidth=2.4,
            )

        axis.set_title(title)
        axis.set_xlabel("UAV speed (km/h)")
        axis.set_xticks(speeds)
        if fixed_limits is not None:
            axis.set_ylim(*fixed_limits)
        else:
            axis.set_ylim(0.0, max(all_values) * 1.12 if all_values else 1.0)
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.8)
        axis.set_axisbelow(True)
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.975),
        ncol=2,
        frameon=False,
        fontsize=13,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=300)
    figure.savefig(args.output.with_suffix(".svg"))
    plt.close(figure)
    print(args.output.resolve())
    print(args.output.with_suffix(".svg").resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
