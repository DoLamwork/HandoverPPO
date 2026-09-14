"""Export the publication-ready relative-rate benchmark as a PDF."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


POLICIES = ("3gpp_always_ho", "ppo")
POLICY_LABELS = {"3gpp_always_ho": "3GPP", "ppo": "PPO"}
POLICY_STYLES = {
    "3gpp_always_ho": {
        "color": "#D55E00",
        "marker": "s",
        "linestyle": "--",
    },
    "ppo": {
        "color": "#0072B2",
        "marker": "o",
        "linestyle": "-",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "results/benchmark/uav_100m_grid100m_all_speeds_percent/"
            "by_speed.csv"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "output/pdf/uav_100m_grid100m/relative_rate_by_speed.pdf"
        ),
    )
    parser.add_argument(
        "--speeds",
        type=int,
        nargs="+",
        default=[30, 50, 70, 90, 120, 150],
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    with args.input.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream, skipinitialspace=True))
    indexed = {
        (row["policy"].strip(), int(row["speed_kmh"])): row
        for row in rows
        if row["policy"].strip() in POLICIES
    }

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
        }
    )
    figure, axis = plt.subplots(figsize=(3.48, 2.65))
    for policy in POLICIES:
        values = [
            float(indexed[(policy, speed)]["mean_relative_rate_percent"])
            for speed in args.speeds
        ]
        style = POLICY_STYLES[policy]
        axis.plot(
            args.speeds,
            values,
            label=POLICY_LABELS[policy],
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=1.5,
            markersize=4.5,
            markerfacecolor="white",
            markeredgewidth=1.1,
        )

    axis.axvspan(105, 155, color="#808080", alpha=0.07, linewidth=0)
    axis.axvline(105, color="#777777", linewidth=0.8, linestyle=":")
    axis.text(
        130,
        0.98,
        "Unseen speeds",
        transform=axis.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=7.5,
        color="#555555",
    )
    axis.set_xlim(25, 155)
    axis.set_ylim(70, 100)
    axis.set_xticks(args.speeds)
    axis.set_xlabel("UAV speed (km/h)")
    axis.set_ylabel("Relative rate (%)")
    axis.grid(axis="y", color="#D0D0D0", linewidth=0.6, linestyle="--")
    axis.set_axisbelow(True)
    axis.legend(loc="lower left", frameon=False, ncol=2, handlelength=2.0)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    figure.tight_layout(pad=0.35)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, format="pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
