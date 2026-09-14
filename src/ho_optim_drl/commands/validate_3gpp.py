"""Validate the A3-based 3GPP reference on the handover environment."""

import os
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from ho_optim_drl.data.project import load_protocol_config
from ho_optim_drl.data import discover_trace_pairs, load_radio_trace
from ho_optim_drl.protocol import A3ReferenceSimulator
import ho_optim_drl.utils as ut


def main(root_path: str, args) -> int:
    """Main function."""

    data_dir = args.data_dir
    if not os.path.isabs(data_dir):
        data_dir = os.path.join(root_path, data_dir)

    config = load_protocol_config(data_dir)
    output_dir = os.path.abspath(os.path.join(root_path, args.output_dir))
    trace_pairs = discover_trace_pairs(data_dir)
    if args.speeds is not None:
        allowed_speeds = set(args.speeds)
        trace_pairs = [
            pair for pair in trace_pairs if pair.speed_kmh in allowed_speeds
        ]
    if not trace_pairs:
        raise ValueError(
            f"No complete trace pairs found in {data_dir!r} for the requested speeds."
        )
    speeds = [pair.speed_kmh for pair in trace_pairs]

    # Result containers
    simulator = A3ReferenceSimulator(config)
    aggregated_stats = {key: [] for key in simulator.get_stats_dict()}
    result_container = ut.get_result_container(speeds)

    # Run the 3GPP handover protocol
    for i, pair in enumerate(trace_pairs):
        speed = pair.speed_kmh
        print(f"Testing 3GPP HO protocol with dataset {i:3d}/{len(trace_pairs)}.")
        trace = load_radio_trace(
            config,
            data_dir,
            pair.rsrp_file,
            pair.sinr_file,
        )

        # Handover protocol
        simulator = A3ReferenceSimulator(config)

        # Simulate handover
        simulator.predict(trace)

        # Save statistics
        stats = simulator.get_statistics()
        for key, val in stats.items():
            aggregated_stats[key].append(val)

        # Save additional statistics
        result_container["bs_idxs"][speed].append(
            np.array(simulator.bs_idxs).tolist()
        )
        result_container["ho_idxs"][speed].append(
            np.array(simulator.ho_idxs).tolist()
        )
        result_container["rlf_idxs"][speed].append(
            np.array(simulator.rlfr_idxs).tolist()
        )
        result_container["sinr_connected"][speed].extend(
            simulator.sinr_timeline
        )
        result_container["sinr_max"][speed].extend(
            list(np.max(trace.sinr_rate_db, axis=0))
        )
        result_container["sinr_at_ho_exe_pcell"].extend(
            simulator.sinr_at_ho_exe_pcell
        )
        result_container["sinr_after_ho_exe_tcell"].extend(
            simulator.sinr_after_ho_exe_tcell
        )
        result_container["n_ho"][speed].append(stats["num_ho_exe_started"])
        result_container["n_ho_success"][speed].append(
            stats["num_ho_exe_completed"]
        )
        result_container["n_hof"][speed].append(stats["num_hof"])
        result_container["n_pp"][speed].append(stats["num_pp"])
        result_container["n_rlf"][speed].append(stats["num_rlf"])
        result_container["duration_s"][speed].append(stats["total_time_s"])

    # SINR (all speeds combined)
    sinr_at_ho_exe_pcell_db = np.array(result_container["sinr_at_ho_exe_pcell"])
    sinr_after_ho_exe_tcell_db = np.array(result_container["sinr_after_ho_exe_tcell"])
    sinr_at_ho_exe_pcell_db[np.isnan(sinr_at_ho_exe_pcell_db)] = -np.inf
    sinr_after_ho_exe_tcell_db[np.isnan(sinr_after_ho_exe_tcell_db)] = -np.inf
    ut.write_to_csv(
        file_path=os.path.join(output_dir, "ho_sinr", "ref_pre_ho_sinr.csv"),
        data_dict={"pre_ho_sinr_db": sinr_at_ho_exe_pcell_db.tolist()},
    )
    ut.write_to_csv(
        file_path=os.path.join(output_dir, "ho_sinr", "ref_post_ho_sinr.csv"),
        data_dict={"post_ho_sinr_db": sinr_after_ho_exe_tcell_db.tolist()},
    )

    # Results (all speeds individually)
    relative_rate_percent = []
    pp_percent = []
    hof_percent = []
    rlf_time_percent = []
    for speed in np.unique(speeds):
        sinr_connected_db = np.array(result_container["sinr_connected"][speed])
        sinr_connected_lin = 10 ** (sinr_connected_db / 10)
        sinr_connected_lin[np.isnan(sinr_connected_lin)] = 0

        sinr_max_lin = 10 ** (np.array(result_container["sinr_max"][speed]) / 10)
        sinr_max_lin[np.isnan(sinr_max_lin)] = 0

        r_mean = np.mean(config.bw * np.log2(1 + sinr_connected_lin))
        r_max = np.mean(config.bw * np.log2(1 + sinr_max_lin))

        relative_rate_percent.append(100 * r_mean / r_max if r_max else 0.0)
        num_ho_success = np.sum(result_container["n_ho_success"][speed])
        num_ho_attempts = np.sum(result_container["n_ho"][speed])
        pp_percent.append(
            100 * np.sum(result_container["n_pp"][speed]) / num_ho_success
            if num_ho_success
            else 0.0
        )
        hof_percent.append(
            100 * np.sum(result_container["n_hof"][speed]) / num_ho_attempts
            if num_ho_attempts
            else 0.0
        )
        rlf_timelines = result_container["rlf_idxs"][speed]
        total_samples = sum(len(timeline) for timeline in rlf_timelines)
        rlf_recovery_samples = sum(
            int(np.count_nonzero(timeline)) for timeline in rlf_timelines
        )
        rlf_time_percent.append(
            100 * rlf_recovery_samples / total_samples
            if total_samples
            else 0.0
        )

    # Print aggregated statistics
    agg_metrics = {
        "speed": np.unique(speeds).tolist(),
        "relative_rate_percent": relative_rate_percent,
        "pp_percent": pp_percent,
        "hof_percent": hof_percent,
        "rlf_time_percent": rlf_time_percent,
    }
    aggregated_stats.update(agg_metrics)
    ut.print_aggregated_stats(aggregated_stats)

    # Log result metrics to csv file
    metrics_path = os.path.join(output_dir, "metrics", "ref_metrics.csv")
    ut.write_to_csv(metrics_path, agg_metrics)
    summary = {
        "data_dir": data_dir, "protocol_config": asdict(config),
        "trajectories": len(trace_pairs), "metrics": agg_metrics,
        "reference": "A3 reference with TS 38.331 rules and documented RLM/HO simulation assumptions",
    }
    (Path(output_dir) / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8",
    )

    return 0
