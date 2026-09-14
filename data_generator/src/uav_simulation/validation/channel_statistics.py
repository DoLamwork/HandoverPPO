"""Statistical validation suite for the UAV-to-ground A2G channel model."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

from uav_simulation.paths import VALIDATION_DIR
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from uav_simulation import config
from uav_simulation.radio_models import RadioModel


class ValidationReport:
    def __init__(self):
        self.metrics = []

    def add(self, name, observed, expected, tolerance, unit=""):
        observed = float(observed)
        expected = float(expected)
        tolerance = float(tolerance)
        passed = bool(abs(observed - expected) <= tolerance)
        self.metrics.append({
            "name": name,
            "observed": observed,
            "expected": expected,
            "tolerance": tolerance,
            "unit": unit,
            "passed": passed,
        })
        return passed

    def add_upper_bound(self, name, observed, upper_bound, unit=""):
        observed = float(observed)
        upper_bound = float(upper_bound)
        passed = bool(observed <= upper_bound)
        self.metrics.append({
            "name": name,
            "observed": observed,
            "expected": upper_bound,
            "tolerance": None,
            "unit": unit,
            "comparison": "<=",
            "passed": passed,
        })
        return passed

    def add_boolean(self, name, passed):
        self.metrics.append({
            "name": name,
            "observed": bool(passed),
            "expected": True,
            "tolerance": None,
            "unit": "",
            "passed": bool(passed),
        })
        return bool(passed)


def build_radio_harness(
    num_ues,
    seed,
    altitude_m=None,
    speed_kmh=0.0,
    direction=0,
):
    altitude_m = float(config.H_UT if altitude_m is None else altitude_m)
    speed_m_s = float(speed_kmh) / 3.6
    sim = SimpleNamespace(
        bs_positions=[(0.0, 0.0)],
        bs_heights=[float(config.H_BS)],
        bs_site_ids=[0],
        h_ut=altitude_m,
        fc=float(config.FC),
        building_area_ratio=float(config.BUILDING_AREA_RATIO),
        building_density_km2=float(config.BUILDING_DENSITY_KM2),
        building_height_rayleigh_m=float(config.BUILDING_HEIGHT_RAYLEIGH_M),
        excess_loss_los_db=float(config.EXCESS_LOSS_LOS_DB),
        excess_loss_nlos_db=float(config.EXCESS_LOSS_NLOS_DB),
        pathloss_mode="correlated_state",
        los_state_correlation_distance_m=float(
            config.LOS_STATE_CORRELATION_DISTANCE_M
        ),
        los_state_shared_per_site=True,
        los_state_cache={},
        sf_cache={},
        sf_sigma=dict(config.SF_SIGMA),
        sf_decorr=50.0,
        enable_small_scale_fading=True,
        rician_k_factor_db=float(config.RICIAN_K_FACTOR_DB),
        fading_num_sinusoids=int(config.FADING_NUM_SINUSOIDS),
        fading_substep_s=float(config.FADING_SUBSTEP_MS) / 1000.0,
        speed_of_light_m_s=float(config.SPEED_OF_LIGHT_M_S),
        time_per_step=float(config.DELTA_T_MS) / 1000.0,
        current_frame=0,
        fast_fading_cache={},
        ue_speeds_ms=np.full(int(num_ues), speed_m_s, dtype=float),
        ue_dir=[int(direction)] * int(num_ues),
        rng=np.random.default_rng(int(seed)),
    )
    return RadioModel(sim), sim


def empirical_cdf(values):
    values = np.sort(np.asarray(values, dtype=float))
    probability = (np.arange(values.size, dtype=float) + 0.5) / values.size
    return values, probability


def validate_pathloss_and_los(report, num_trials):
    distances_m = np.linspace(1.0, 2000.0, 241)
    altitudes_m = (50.0, 100.0, 150.0)
    p_los_curves = {}

    for altitude_m in altitudes_m:
        radio, _ = build_radio_harness(1, 1000 + int(altitude_m), altitude_m)
        p_los_curves[altitude_m] = np.array([
            radio.calculate_los_probability_idx(distance_m, 0.0, 0)
            for distance_m in distances_m
        ])

    all_probabilities = np.concatenate(list(p_los_curves.values()))
    report.add_boolean(
        "P_LOS remains in [0, 1]",
        np.all((all_probabilities >= 0.0) & (all_probabilities <= 1.0)),
    )

    radio, sim = build_radio_harness(1, 1200, float(config.H_UT))
    components = np.array([
        radio.calculate_path_loss_components_idx(distance_m, 0.0, 0)
        for distance_m in distances_m
    ])
    pl_los_db = components[:, 0]
    pl_nlos_db = components[:, 1]
    expected_gap_db = float(
        config.EXCESS_LOSS_NLOS_DB - config.EXCESS_LOSS_LOS_DB
    )
    observed_gap_db = pl_nlos_db - pl_los_db
    report.add_upper_bound(
        "LoS/NLoS pathloss-gap maximum error",
        np.max(np.abs(observed_gap_db - expected_gap_db)),
        1e-10,
        "dB",
    )
    report.add_boolean(
        "LoS pathloss is monotonic with horizontal distance",
        np.all(np.diff(pl_los_db) >= -1e-10),
    )
    report.add_boolean(
        "NLoS pathloss is monotonic with horizontal distance",
        np.all(np.diff(pl_nlos_db) >= -1e-10),
    )

    selected_distances_m = (250.0, 500.0, 1000.0, 1500.0)
    empirical_probabilities = []
    theoretical_probabilities = []
    radio, sim = build_radio_harness(num_trials, 1300, float(config.H_UT))
    for distance_m in selected_distances_m:
        sim.los_state_cache = {}
        probability = radio.calculate_los_probability_idx(distance_m, 0.0, 0)
        states = np.array([
            radio.update_correlated_los_state(
                ue_idx, 0, probability, (distance_m, 0.0)
            )
            for ue_idx in range(num_trials)
        ], dtype=float)
        empirical = float(np.mean(states))
        standard_error = math.sqrt(
            max(probability * (1.0 - probability), 1e-12) / num_trials
        )
        tolerance = max(0.015, 3.5 * standard_error)
        report.add(
            f"LoS realization probability at {distance_m:g} m",
            empirical,
            probability,
            tolerance,
        )
        empirical_probabilities.append(empirical)
        theoretical_probabilities.append(probability)

    return {
        "distances_m": distances_m,
        "p_los_curves": p_los_curves,
        "pl_los_db": pl_los_db,
        "pl_nlos_db": pl_nlos_db,
        "selected_distances_m": np.array(selected_distances_m),
        "empirical_probabilities": np.array(empirical_probabilities),
        "theoretical_probabilities": np.array(theoretical_probabilities),
    }


def validate_los_spatial_correlation(report, num_pairs):
    lags_m = np.array((5.0, 10.0, 25.0, 50.0, 100.0, 200.0))
    empirical = []
    theoretical = np.exp(-lags_m / float(config.LOS_STATE_CORRELATION_DISTANCE_M))

    for lag_m, expected in zip(lags_m, theoretical):
        radio, sim = build_radio_harness(
            num_pairs, 2000 + int(lag_m * 10), float(config.H_UT)
        )
        initial_z = np.empty(num_pairs, dtype=float)
        displaced_z = np.empty(num_pairs, dtype=float)
        for ue_idx in range(num_pairs):
            radio.update_correlated_los_state(ue_idx, 0, 0.5, (0.0, 0.0))
            initial_z[ue_idx] = sim.los_state_cache[(ue_idx, 0)]["z"]
            radio.update_correlated_los_state(ue_idx, 0, 0.5, (lag_m, 0.0))
            displaced_z[ue_idx] = sim.los_state_cache[(ue_idx, 0)]["z"]
        observed = float(np.corrcoef(initial_z, displaced_z)[0, 1])
        empirical.append(observed)
        report.add(
            f"Latent LoS spatial correlation at {lag_m:g} m",
            observed,
            expected,
            0.045,
        )

    return {
        "lags_m": lags_m,
        "empirical": np.array(empirical),
        "theoretical": theoretical,
    }


def validate_shadow_fading(report, num_pairs):
    lags_m = np.array((5.0, 10.0, 25.0, 50.0, 100.0, 200.0))
    results = {}

    for branch_index, (branch, is_los) in enumerate((("LOS", True), ("NLOS", False))):
        sigma = float(config.SF_SIGMA[branch])
        radio, _ = build_radio_harness(num_pairs, 3000 + branch_index)
        samples = np.array([
            radio.shadow_fading(
                ue_idx, 0, 1.0 if is_los else 0.0, (0.0, 0.0), is_los
            )
            for ue_idx in range(num_pairs)
        ])
        report.add(
            f"{branch} shadow-fading mean",
            np.mean(samples),
            0.0,
            max(0.15, 3.5 * sigma / math.sqrt(num_pairs)),
            "dB",
        )
        report.add(
            f"{branch} shadow-fading standard deviation",
            np.std(samples),
            sigma,
            0.05 * sigma,
            "dB",
        )

        empirical_correlation = []
        theoretical_correlation = np.exp(-lags_m / 50.0)
        for lag_m, expected in zip(lags_m, theoretical_correlation):
            radio, sim = build_radio_harness(
                num_pairs,
                3100 + branch_index * 1000 + int(lag_m * 10),
            )
            initial = np.empty(num_pairs, dtype=float)
            displaced = np.empty(num_pairs, dtype=float)
            for ue_idx in range(num_pairs):
                initial[ue_idx] = radio.shadow_fading(
                    ue_idx,
                    0,
                    1.0 if is_los else 0.0,
                    (0.0, 0.0),
                    is_los,
                )
                displaced[ue_idx] = radio.shadow_fading(
                    ue_idx,
                    0,
                    1.0 if is_los else 0.0,
                    (lag_m, 0.0),
                    is_los,
                )
            observed = float(np.corrcoef(initial, displaced)[0, 1])
            empirical_correlation.append(observed)
            report.add(
                f"{branch} shadow correlation at {lag_m:g} m",
                observed,
                expected,
                0.055,
            )

        results[branch] = {
            "samples": samples,
            "empirical_correlation": np.array(empirical_correlation),
            "theoretical_correlation": theoretical_correlation,
        }

    results["lags_m"] = lags_m
    return results


def validate_small_scale_fading(report, num_links):
    rayleigh_radio, _ = build_radio_harness(
        num_links, 4000, speed_kmh=0.0
    )
    rician_radio, _ = build_radio_harness(
        num_links, 4001, speed_kmh=0.0
    )
    rayleigh_power = np.array([
        10.0 ** (
            rayleigh_radio.small_scale_fading_db(
                ue_idx, 0, False, (1000.0, 0.0)
            ) / 10.0
        )
        for ue_idx in range(num_links)
    ])
    rician_power = np.array([
        10.0 ** (
            rician_radio.small_scale_fading_db(
                ue_idx, 0, True, (1000.0, 0.0)
            ) / 10.0
        )
        for ue_idx in range(num_links)
    ])

    num_waves = float(config.FADING_NUM_SINUSOIDS)
    rayleigh_variance = 1.0 - 1.0 / num_waves
    k_linear = 10.0 ** (float(config.RICIAN_K_FACTOR_DB) / 10.0)
    rician_variance = (
        2.0 * k_linear + 1.0 - 1.0 / num_waves
    ) / (k_linear + 1.0) ** 2

    report.add("Rayleigh mean power", np.mean(rayleigh_power), 1.0, 0.05)
    report.add(
        "Rayleigh power variance",
        np.var(rayleigh_power),
        rayleigh_variance,
        0.12,
    )
    report.add("Rician mean power", np.mean(rician_power), 1.0, 0.04)
    report.add(
        "Rician power variance",
        np.var(rician_power),
        rician_variance,
        0.045,
    )

    rayleigh_sorted, rayleigh_ecdf = empirical_cdf(rayleigh_power)
    rayleigh_theoretical_cdf = 1.0 - np.exp(-rayleigh_sorted)
    ks_distance = float(np.max(np.abs(
        rayleigh_ecdf - rayleigh_theoretical_cdf
    )))
    report.add_upper_bound(
        "Rayleigh power CDF maximum distance",
        ks_distance,
        0.035,
    )

    return {
        "rayleigh_power": rayleigh_power,
        "rician_power": rician_power,
        "rayleigh_variance_expected": rayleigh_variance,
        "rician_variance_expected": rician_variance,
        "rayleigh_ks_distance": ks_distance,
    }


def validate_doppler(report):
    speeds_kmh = np.array((0.0, 30.0, 60.0, 90.0, 120.0))
    measured_hz = []
    expected_hz = []
    direct_hz = []

    for speed_kmh in speeds_kmh:
        radio, sim = build_radio_harness(
            1, 5000 + int(speed_kmh), speed_kmh=speed_kmh, direction=0
        )
        direct, maximum = radio.calculate_doppler_hz(
            0, 0, (1000.0, 0.0)
        )
        expected = (
            (speed_kmh / 3.6) * float(config.FC) * 1e9
            / float(config.SPEED_OF_LIGHT_M_S)
        )
        measured_hz.append(maximum)
        expected_hz.append(expected)
        direct_hz.append(direct)

    measured_hz = np.array(measured_hz)
    expected_hz = np.array(expected_hz)
    direct_hz = np.array(direct_hz)
    report.add_upper_bound(
        "Maximum Doppler formula error",
        np.max(np.abs(measured_hz - expected_hz)),
        1e-10,
        "Hz",
    )

    tangential_radio, _ = build_radio_harness(
        1, 5500, speed_kmh=120.0, direction=1
    )
    tangential_hz, _ = tangential_radio.calculate_doppler_hz(
        0, 0, (1000.0, 0.0)
    )
    report.add("Tangential direct-path Doppler", tangential_hz, 0.0, 1e-10, "Hz")

    return {
        "speeds_kmh": speeds_kmh,
        "measured_hz": measured_hz,
        "expected_hz": expected_hz,
        "direct_hz": direct_hz,
    }


def validate_end_to_end_traces(report, trace_steps):
    from uav_simulation.simulation import CellularNetworkReceivedPower

    speeds_kmh = (30.0, 60.0, 90.0, 120.0)
    traces = {}
    for speed_kmh in speeds_kmh:
        sim = CellularNetworkReceivedPower(
            num_ues=1,
            rect_len_m=1000.0,
            rect_wid_m=1000.0,
            grid_spacing_m=200.0,
            uav_ue_speed_kmh=speed_kmh,
            headless=True,
            seed=20260730,
        )
        for frame in range(trace_steps):
            sim.update(frame)

        rsrp = np.column_stack([
            sim.data_log[f"ue0_bs{bs_idx}_rsrp_dbm"]
            for bs_idx in range(len(sim.bs_positions))
        ]).astype(float)
        sinr = np.column_stack([
            sim.data_log[f"ue0_bs{bs_idx}_sinr_db"]
            for bs_idx in range(len(sim.bs_positions))
        ]).astype(float)
        best_bs = np.argmax(rsrp, axis=1)
        row_index = np.arange(trace_steps)
        best_rsrp = rsrp[row_index, best_bs]
        corresponding_sinr = sinr[row_index, best_bs]

        report.add_boolean(
            f"Finite end-to-end RSRP/SINR at {speed_kmh:g} km/h",
            np.all(np.isfinite(rsrp)) and np.all(np.isfinite(sinr)),
        )
        report.add_boolean(
            f"One large-scale state per site at {speed_kmh:g} km/h",
            len(sim.los_state_cache) == len(sim.hexes)
            and len(sim.sf_cache) == len(sim.hexes),
        )
        report.add_boolean(
            f"One small-scale process per sector at {speed_kmh:g} km/h",
            len(sim.fast_fading_cache) == len(sim.bs_positions),
        )
        traces[speed_kmh] = {
            "best_rsrp_dbm": best_rsrp,
            "corresponding_sinr_db": corresponding_sinr,
        }
        plt.close(sim.fig)

    return traces


def create_channel_figure(output_path, pathloss, los_corr, shadow, fading, doppler):
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    colors = plt.get_cmap("tab10").colors

    ax = axes[0, 0]
    for index, (altitude, probability) in enumerate(pathloss["p_los_curves"].items()):
        ax.plot(pathloss["distances_m"], probability, color=colors[index], label=f"h={altitude:g} m")
    ax.scatter(
        pathloss["selected_distances_m"],
        pathloss["empirical_probabilities"],
        color="black",
        marker="x",
        label="Empirical at h=100 m",
    )
    ax.set(title="LoS probability", xlabel="Horizontal distance (m)", ylabel="Probability")
    ax.set_ylim(-0.03, 1.03)
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.plot(pathloss["distances_m"], pathloss["pl_los_db"], label="LoS")
    ax.plot(pathloss["distances_m"], pathloss["pl_nlos_db"], label="NLoS")
    ax.set(title=f"Pathloss at h={config.H_UT:g} m", xlabel="Horizontal distance (m)", ylabel="Pathloss (dB)")
    ax.legend(fontsize=8)

    ax = axes[0, 2]
    ax.plot(los_corr["lags_m"], los_corr["theoretical"], label="exp(-d/dcorr)")
    ax.scatter(los_corr["lags_m"], los_corr["empirical"], marker="x", color="black", label="Empirical latent z")
    ax.set(title="LoS-state spatial correlation", xlabel="Displacement (m)", ylabel="Correlation")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.plot(shadow["lags_m"], shadow["LOS"]["theoretical_correlation"], color="black", label="Theory")
    ax.scatter(shadow["lags_m"], shadow["LOS"]["empirical_correlation"], marker="o", label="LoS")
    ax.scatter(shadow["lags_m"], shadow["NLOS"]["empirical_correlation"], marker="x", label="NLoS")
    ax.set(title="Shadow-fading spatial correlation", xlabel="Displacement (m)", ylabel="Correlation")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    rayleigh_x, rayleigh_y = empirical_cdf(fading["rayleigh_power"])
    rician_x, rician_y = empirical_cdf(fading["rician_power"])
    theory_x = np.linspace(0.0, max(5.0, float(rayleigh_x[-1])), 500)
    ax.plot(theory_x, 1.0 - np.exp(-theory_x), color="black", linestyle="--", label="Ideal Rayleigh")
    ax.plot(rayleigh_x, rayleigh_y, label="Rayleigh empirical")
    ax.plot(rician_x, rician_y, label=f"Rician empirical, K={config.RICIAN_K_FACTOR_DB:g} dB")
    ax.set(title="Small-scale power CDF", xlabel="Normalized linear power", ylabel="CDF", xlim=(0.0, 5.0), ylim=(0.0, 1.0))
    ax.legend(fontsize=8)

    ax = axes[1, 2]
    ax.plot(doppler["speeds_kmh"], doppler["expected_hz"], label="v fc / c")
    ax.scatter(doppler["speeds_kmh"], doppler["measured_hz"], marker="x", color="black", label="Implementation")
    ax.plot(doppler["speeds_kmh"], np.abs(doppler["direct_hz"]), linestyle="--", label="Direct path at 1 km")
    ax.set(title=f"Doppler at {config.FC:g} GHz", xlabel="UAV speed (km/h)", ylabel="Doppler magnitude (Hz)")
    ax.legend(fontsize=8)

    for ax in axes.flat:
        ax.grid(True, alpha=0.25)
    fig.suptitle("A2G channel statistical validation")
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def create_trace_figure(output_path, traces):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for speed_kmh, trace in traces.items():
        rsrp_x, rsrp_y = empirical_cdf(trace["best_rsrp_dbm"])
        sinr_x, sinr_y = empirical_cdf(trace["corresponding_sinr_db"])
        axes[0].plot(rsrp_x, rsrp_y, label=f"{speed_kmh:g} km/h")
        axes[1].plot(sinr_x, sinr_y, label=f"{speed_kmh:g} km/h")
    axes[0].set(title="Best-BS RSRP CDF", xlabel="RSRP (dBm)", ylabel="CDF")
    axes[1].set(title="SINR of best-RSRP BS", xlabel="SINR (dB)", ylabel="CDF")
    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def format_value(value, unit=""):
    if isinstance(value, bool):
        return str(value)
    suffix = f" {unit}" if unit else ""
    return f"{float(value):.6g}{suffix}"


def write_report(output_path, summary):
    lines = [
        "# A2G channel statistical validation",
        "",
        f"Generated: {summary['generated_at_utc']}",
        "",
        f"Overall result: **{'PASS' if summary['overall_pass'] else 'FAIL'}** ",
        f"({summary['passed_metrics']}/{summary['total_metrics']} checks passed)",
        "",
        "## Validation metrics",
        "",
        "| Metric | Observed | Expected/limit | Tolerance | Result |",
        "|---|---:|---:|---:|:---:|",
    ]
    for metric in summary["metrics"]:
        unit = metric.get("unit", "")
        expected_prefix = metric.get("comparison", "")
        expected = format_value(metric["expected"], unit)
        if expected_prefix:
            expected = f"{expected_prefix} {expected}"
        tolerance = metric.get("tolerance")
        tolerance_text = "-" if tolerance is None else format_value(tolerance, unit)
        lines.append(
            f"| {metric['name']} | {format_value(metric['observed'], unit)} "
            f"| {expected} | {tolerance_text} | {'PASS' if metric['passed'] else 'FAIL'} |"
        )
    lines.extend([
        "",
        "## Figures",
        "",
        "![Channel statistics](channel_statistics_validation.png)",
        "",
        "![End-to-end radio traces](radio_trace_validation.png)",
        "",
        "## Interpretation boundaries",
        "",
        "- These checks validate implementation statistics against the configured equations; they do not calibrate the model against a measured urban UAV campaign.",
        (
            f"- Small-scale fading is flat across {config.BANDWIDTH_MHZ:g} MHz "
            f"and averaged over {int(round(config.DELTA_T_MS / config.FADING_SUBSTEP_MS))} "
            f"internal {config.FADING_SUBSTEP_MS:g} ms samples per logged "
            f"{config.DELTA_T_MS:g} ms step."
        ),
        "- LoS and shadow consistency are trajectory-correlated, not generated from a fixed two-dimensional obstruction map.",
        "- End-to-end SINR assumes full-band, reuse-1, full-load interference.",
    ])
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=VALIDATION_DIR,
        help="Directory for PNG, JSON, and Markdown outputs.",
    )
    parser.add_argument("--full", action="store_true", help="Use larger Monte Carlo samples and 1000-step traces.")
    parser.add_argument("--trace-steps", type=int, default=None, help="Override the number of end-to-end steps per speed.")
    args = parser.parse_args(argv)

    if args.full:
        num_trials, num_pairs, num_links = 10000, 6000, 10000
        trace_steps = 1000 if args.trace_steps is None else args.trace_steps
    else:
        num_trials, num_pairs, num_links = 4000, 3000, 4000
        trace_steps = 200 if args.trace_steps is None else args.trace_steps
    if trace_steps <= 0:
        raise ValueError("trace_steps must be positive")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report = ValidationReport()

    pathloss = validate_pathloss_and_los(report, num_trials)
    los_corr = validate_los_spatial_correlation(report, num_pairs)
    shadow = validate_shadow_fading(report, num_pairs)
    fading = validate_small_scale_fading(report, num_links)
    doppler = validate_doppler(report)
    traces = validate_end_to_end_traces(report, trace_steps)

    channel_figure = output_dir / "channel_statistics_validation.png"
    trace_figure = output_dir / "radio_trace_validation.png"
    create_channel_figure(channel_figure, pathloss, los_corr, shadow, fading, doppler)
    create_trace_figure(trace_figure, traces)

    passed_metrics = sum(metric["passed"] for metric in report.metrics)
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "full" if args.full else "standard",
        "sample_sizes": {
            "los_probability_trials": num_trials,
            "correlation_pairs": num_pairs,
            "small_scale_links": num_links,
            "trace_steps_per_speed": trace_steps,
        },
        "configuration": {
            "carrier_ghz": float(config.FC),
            "bandwidth_mhz": float(config.BANDWIDTH_MHZ),
            "delta_t_ms": float(config.DELTA_T_MS),
            "uav_altitude_m": float(config.H_UT),
            "los_correlation_distance_m": float(config.LOS_STATE_CORRELATION_DISTANCE_M),
            "shadow_sigma_db": dict(config.SF_SIGMA),
            "rician_k_factor_db": float(config.RICIAN_K_FACTOR_DB),
            "fading_sinusoids": int(config.FADING_NUM_SINUSOIDS),
            "fading_substep_ms": float(config.FADING_SUBSTEP_MS),
        },
        "overall_pass": bool(passed_metrics == len(report.metrics)),
        "passed_metrics": int(passed_metrics),
        "total_metrics": len(report.metrics),
        "failed_metrics": [
            metric["name"] for metric in report.metrics if not metric["passed"]
        ],
        "metrics": report.metrics,
        "artifacts": {
            "channel_statistics_figure": channel_figure.name,
            "radio_trace_figure": trace_figure.name,
            "markdown_report": "validation_report.md",
        },
    }

    json_path = output_dir / "validation_summary.json"
    report_path = output_dir / "validation_report.md"
    json_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_report(report_path, summary)

    print(
        f"A2G statistical validation: "
        f"{'PASS' if summary['overall_pass'] else 'FAIL'} "
        f"({passed_metrics}/{len(report.metrics)})"
    )
    print(f"Report: {report_path}")
    print(f"Summary: {json_path}")
    print(f"Figures: {channel_figure}, {trace_figure}")
    if not summary["overall_pass"]:
        print("Failed metrics:")
        for name in summary["failed_metrics"]:
            print(f"- {name}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
