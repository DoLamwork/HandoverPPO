"""Project command-line entry point."""

import argparse
import os
import sys

DEFAULT_DATA_DIR = "data/uav_generated"
DEFAULT_MODEL_DIR = "results/models/uav_generated_ppo"


def run(argv=None) -> int:
    root_path = os.getcwd()
    argv = list(sys.argv[1:] if argv is None else argv)
    # The parent owns the workflow; the simulator is only a data-generation tool.
    if argv and argv[0] == "generate_data":
        from uav_simulation.batch.generate_dataset import main
        return main(argv[1:])
    if argv and argv[0] == "prepare_data":
        from ho_optim_drl.commands.prepare_uav_data import main
        return main(argv[1:])
    parser = argparse.ArgumentParser(
        description="Trace-driven adaptive handover research framework."
    )
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("generate_data", add_help=False, help="Generate radio data with the data_generator component.")
    subparsers.add_parser("prepare_data", add_help=False, help="Convert legacy UAV CSV pairs to the PPO schema.")
    validate_parser = subparsers.add_parser(
        "validate_3gpp",
        help="Run the audited A3-based reference on all CSV traces.",
    )
    validate_parser.add_argument(
        "--data-dir",
        default=DEFAULT_DATA_DIR,
        help="Directory containing train-ready RSRP/SINR CSV trace pairs.",
    )
    validate_parser.add_argument(
        "--speeds",
        type=int,
        nargs="+",
        default=None,
        help="Optional speed(s) in km/h to validate; defaults to every discovered speed.",
    )
    validate_parser.add_argument("--output-dir", default="results/3gpp/uav_generated")
    parity_parser = subparsers.add_parser(
        "audit_baseline_parity",
        help="Check TRAIN-only B0 reference parity with the B0A action adapter.",
    )
    parity_parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parity_parser.add_argument(
        "--speeds", type=int, nargs="+", default=[30, 50, 70, 90]
    )
    parity_parser.add_argument("--split-seed", type=int, default=17)
    parity_parser.add_argument("--sample-seed", type=int, default=17)
    parity_parser.add_argument("--routes-per-speed", type=int, default=2)
    parity_parser.add_argument(
        "--output-dir",
        default="results/analysis/early_ho_timing/b0_b0a_pilot_v2",
    )
    b1_parser = subparsers.add_parser(
        "audit_b1_lower_threshold",
        help="Run the TRAIN-only B1 lower-A3-offset paired pilot.",
    )
    b1_parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    b1_parser.add_argument(
        "--baseline-manifest",
        default="results/analysis/early_ho_timing/b0_b0a_pilot_v2/run_manifest.json",
    )
    b1_parser.add_argument(
        "--output-dir",
        default="results/analysis/early_ho_timing/b1_lower_threshold_pilot_v2",
    )
    b23_parser = subparsers.add_parser(
        "audit_b2_b3_early_controls",
        help="Run paired TRAIN-only B2 immediate and B3 delayed early-HO controls.",
    )
    b23_parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    b23_parser.add_argument(
        "--baseline-manifest",
        default="results/analysis/early_ho_timing/b0_b0a_pilot_v2/run_manifest.json",
    )
    b23_parser.add_argument(
        "--output-dir",
        default="results/analysis/early_ho_timing/b2_b3_controls_pilot_v1",
    )
    timing_reference_parser = subparsers.add_parser(
        "audit_early_timing_references",
        help="Run TRAIN-only shadow-A3 Rpost/Rearly correctness and cost pilot.",
    )
    timing_reference_parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    timing_reference_parser.add_argument(
        "--baseline-manifest",
        default="results/analysis/early_ho_timing/b0_b0a_pilot_v2/run_manifest.json",
    )
    timing_reference_parser.add_argument(
        "--delays-ms", type=int, nargs="+", default=[0, 20, 40, 80, 160, 320]
    )
    timing_reference_parser.add_argument("--reference-horizon", type=float, default=1.0)
    timing_reference_parser.add_argument(
        "--output-dir",
        default="results/analysis/early_ho_timing/rpost_rearly_pilot_v1",
    )
    t2_parser = subparsers.add_parser(
        "audit_early_timing_bank",
        help="Build the T2 TRAIN anchor bank and local timing counterfactuals.",
    )
    t2_parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    t2_parser.add_argument("--speeds", type=int, nargs="+", default=[30, 50, 70, 90])
    t2_parser.add_argument("--split-seed", type=int, default=17)
    t2_parser.add_argument("--sample-seed", type=int, default=17)
    t2_parser.add_argument("--routes-per-speed", type=int, default=20)
    t2_parser.add_argument("--max-anchors-per-source-route", type=int, default=5)
    t2_parser.add_argument("--delays-ms", type=int, nargs="+", default=[0, 20, 40, 80, 160, 320])
    t2_parser.add_argument("--horizons", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    t2_parser.add_argument(
        "--output-dir", default="results/analysis/early_ho_timing/t2_bank_main_v1"
    )
    t3_parser = subparsers.add_parser(
        "audit_early_timing_routes",
        help="Run T3 seven-policy TRAIN full-route audit and paired bootstrap.",
    )
    t3_parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    t3_parser.add_argument(
        "--route-split",
        default="results/analysis/early_ho_timing/t2_bank_main_v1/route_split.json",
    )
    t3_parser.add_argument("--delays-ms", type=int, nargs="+", default=[0, 20, 40, 80, 160, 320])
    t3_parser.add_argument("--reference-horizon", type=float, default=1.0)
    t3_parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    t3_parser.add_argument("--bootstrap-seed", type=int, default=29)
    t3_parser.add_argument(
        "--output-dir", default="results/analysis/early_ho_timing/t3_route_main_v1"
    )
    t5a_source_parser = subparsers.add_parser(
        "audit_t5a_benefit_sources",
        help=(
            "Audit realised T5A benefit sources and compare TRAIN-only "
            "pre-A3 versus post-TTT counterfactual opportunity."
        ),
    )
    t5a_source_parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    t5a_source_parser.add_argument(
        "--route-split",
        default="results/analysis/early_ho_timing/t2_bank_main_v1/route_split.json",
    )
    t5a_source_parser.add_argument(
        "--t2-dir",
        default="results/analysis/early_ho_timing/t2_bank_main_v1",
    )
    t5a_source_parser.add_argument(
        "--t3-dir",
        default="results/analysis/early_ho_timing/t3_route_main_v1",
    )
    t5a_source_parser.add_argument(
        "--checkpoints",
        nargs="+",
        default=[
            "results/models/early_ho_timing/"
            "T5A_timing_ppo_fixed_scale_b100000_s17/smdp_ppo_selected.pt",
            "results/models/early_ho_timing/"
            "T5A_timing_ppo_fixed_scale_b100000_s29/smdp_ppo_selected.pt",
            "results/models/early_ho_timing/"
            "T5A_timing_ppo_fixed_scale_b100000_s43/smdp_ppo_selected.pt",
        ],
    )
    t5a_source_parser.add_argument("--horizon", type=float, default=1.0)
    t5a_source_parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    t5a_source_parser.add_argument("--bootstrap-seed", type=int, default=71)
    t5a_source_parser.add_argument(
        "--max-routes-per-speed",
        type=int,
        help="Optional correctness-pilot cap; omit for the frozen 80-route audit.",
    )
    t5a_source_parser.add_argument(
        "--output-dir",
        default="results/analysis/early_ho_timing/t5a_benefit_source_main_v1",
    )
    t6a_divergence_parser = subparsers.add_parser(
        "audit_t6a_seed_divergence",
        help=(
            "Compare T6A seed 17 and 29 on a shared TRAIN state union and "
            "their recorded PPO learning signals."
        ),
    )
    t6a_divergence_parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    t6a_divergence_parser.add_argument(
        "--route-split",
        default="results/analysis/early_ho_timing/t2_bank_main_v1/route_split.json",
    )
    t6a_divergence_parser.add_argument(
        "--seed17-checkpoint",
        default=(
            "results/models/early_ho_timing/"
            "T6A_a3_entry_timing_b100000_s17/smdp_ppo_selected.pt"
        ),
    )
    t6a_divergence_parser.add_argument(
        "--seed29-checkpoint",
        default=(
            "results/models/early_ho_timing/"
            "T6A_a3_entry_timing_b100000_s29/smdp_ppo_selected.pt"
        ),
    )
    t6a_divergence_parser.add_argument(
        "--seed17-metrics",
        default=(
            "results/models/early_ho_timing/"
            "T6A_a3_entry_timing_b100000_s17/training_metrics.jsonl"
        ),
    )
    t6a_divergence_parser.add_argument(
        "--seed29-metrics",
        default=(
            "results/models/early_ho_timing/"
            "T6A_a3_entry_timing_b100000_s29/training_metrics.jsonl"
        ),
    )
    t6a_divergence_parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    t6a_divergence_parser.add_argument("--bootstrap-seed", type=int, default=83)
    t6a_divergence_parser.add_argument(
        "--max-routes-per-speed",
        type=int,
        help="Optional correctness-pilot cap; omit for the locked 80-route audit.",
    )
    t6a_divergence_parser.add_argument(
        "--output-dir",
        default="results/analysis/early_ho_timing/t6a_seed17_vs_seed29_v1",
    )
    verify_t23_parser = subparsers.add_parser(
        "verify_early_timing_t2_t3",
        help="Verify archived T2/T3 TRAIN artifacts and recompute the corrected QoS gate.",
    )
    verify_t23_parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    verify_t23_parser.add_argument(
        "--t2-dir", default="results/analysis/early_ho_timing/t2_bank_main_v1"
    )
    verify_t23_parser.add_argument(
        "--t3-dir", default="results/analysis/early_ho_timing/t3_route_main_v1"
    )
    verify_t23_parser.add_argument(
        "--output-dir",
        default="results/analysis/early_ho_timing/t2_t3_verification_v1",
    )
    verify_t23_parser.add_argument(
        "--skip-anchor-replay", action="store_true",
        help="Skip deterministic T2 anchor replay (bookkeeping checks still run).",
    )
    t4_parser = subparsers.add_parser(
        "audit_early_timing_learnability",
        help="Run T4 post-A3 causal HO_NOW/WAIT_20 learnability audit.",
    )
    t4_parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    t4_parser.add_argument("--t2-dir", default="results/analysis/early_ho_timing/t2_bank_main_v1")
    t4_parser.add_argument("--t3-dir", default="results/analysis/early_ho_timing/t3_route_main_v1")
    t4_parser.add_argument("--route-split-seed", type=int, default=29)
    t4_parser.add_argument("--model-seed", type=int, default=29)
    t4_parser.add_argument("--output-dir", default="results/analysis/early_ho_timing/t4_learnability_v1")
    t4_parser.add_argument("--feature-schema", choices=["t4a_causal_v1", "t4b_protocol_history_v1"], default="t4a_causal_v1")
    t4_parser.add_argument("--control-report", default=None)
    t4b_parser = subparsers.add_parser(
        "audit_early_timing_observation_ablation",
        help="Run T4B post-A3 protocol/history observation ablation against T4A.",
    )
    t4b_parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    t4b_parser.add_argument("--t2-dir", default="results/analysis/early_ho_timing/t2_bank_main_v1")
    t4b_parser.add_argument("--t3-dir", default="results/analysis/early_ho_timing/t3_route_main_v1")
    t4b_parser.add_argument("--route-split-seed", type=int, default=29)
    t4b_parser.add_argument("--model-seed", type=int, default=29)
    t4b_parser.add_argument("--feature-schema", default="t4b_protocol_history_v1")
    t4b_parser.add_argument("--control-report", default="results/analysis/early_ho_timing/t4_learnability_v3/t4_report.json")
    t4b_parser.add_argument("--output-dir", default="results/analysis/early_ho_timing/t4b_observation_v1")
    t4b_error_parser = subparsers.add_parser(
        "analyze_t4b_errors",
        help="Audit frozen T4B false-positive/false-negative decisions without refit.",
    )
    t4b_error_parser.add_argument("--t4b-dir", default="results/analysis/early_ho_timing/t4b_observation_v1")
    t4b_error_parser.add_argument("--seed", type=int, default=29)
    t4b_error_parser.add_argument("--output-dir", default="results/analysis/early_ho_timing/t4b_error_audit_v1")
    post_ttt_parser = subparsers.add_parser(
        "build_post_ttt_train_bank",
        help="Build a post-TTT-only bank from previously unused TRAIN routes.",
    )
    post_ttt_parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    post_ttt_parser.add_argument("--speeds", type=int, nargs="+", default=[30, 50, 70, 90])
    post_ttt_parser.add_argument("--split-seed", type=int, default=17)
    post_ttt_parser.add_argument("--route-seed", type=int, default=43)
    post_ttt_parser.add_argument("--sample-seed", type=int, default=17)
    post_ttt_parser.add_argument(
        "--exclude-route-split", action="append",
        default=["results/analysis/early_ho_timing/t2_bank_main_v1/route_split.json"],
    )
    post_ttt_parser.add_argument("--fit-routes-per-speed", type=int, default=40)
    post_ttt_parser.add_argument("--tuning-routes-per-speed", type=int, default=10)
    post_ttt_parser.add_argument("--confirmation-routes-per-speed", type=int, default=10)
    post_ttt_parser.add_argument(
        "--split-only", action="store_true",
        help="Freeze only an outcome-blind confirmation route split; do not simulate anchors/outcomes.",
    )
    post_ttt_parser.add_argument("--max-anchors-per-source-route", type=int, default=5)
    post_ttt_parser.add_argument("--min-anchor-separation-ms", type=int, default=500)
    post_ttt_parser.add_argument("--delays-ms", type=int, nargs="+", default=[0, 20, 40, 80, 160, 320])
    post_ttt_parser.add_argument("--horizons", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    post_ttt_parser.add_argument(
        "--output-dir",
        default="results/analysis/early_ho_timing/post_ttt_train_bank_v1",
    )
    t4c_parser = subparsers.add_parser(
        "audit_t4c_post_ttt_probe",
        help="Fit/tune/freeze and confirm T4C on the expanded post-TTT TRAIN bank.",
    )
    t4c_parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    t4c_parser.add_argument(
        "--bank-dir",
        default="results/analysis/early_ho_timing/post_ttt_train_bank_v1",
    )
    t4c_parser.add_argument("--model-seed", type=int, default=29)
    t4c_parser.add_argument(
        "--t4b-report",
        default="results/analysis/early_ho_timing/t4b_observation_v1/t4_report.json",
    )
    t4c_parser.add_argument(
        "--output-dir",
        default="results/analysis/early_ho_timing/t4c_probe_v1",
    )
    t4d_parser = subparsers.add_parser(
        "audit_t4d_compact_causal_probe",
        help="Fit separate compact causal QoS/risk heads on frozen T4C TRAIN samples.",
    )
    t4d_parser.add_argument(
        "--t4c-dir",
        default="results/analysis/early_ho_timing/t4c_probe_v1",
    )
    t4d_parser.add_argument("--model-seed", type=int, default=29)
    t4d_parser.add_argument(
        "--minimum-meaningful-gain-mbit", type=float, default=0.05,
    )
    t4d_parser.add_argument(
        "--output-dir",
        default="results/analysis/early_ho_timing/t4d_compact_probe_v1",
    )
    t4d1_parser = subparsers.add_parser(
        "audit_t4d1_ranked_probe",
        help="Run T4D.1 material-gain ranking with calibrated risk on frozen TRAIN states.",
    )
    t4d1_parser.add_argument(
        "--t4c-dir",
        default="results/analysis/early_ho_timing/t4c_probe_v1",
    )
    t4d1_parser.add_argument(
        "--t4d-report",
        default="results/analysis/early_ho_timing/t4d_compact_probe_v1/t4d_report.json",
    )
    t4d1_parser.add_argument("--model-seed", type=int, default=29)
    t4d1_parser.add_argument("--minimum-gain-mbit", type=float, default=0.05)
    t4d1_parser.add_argument("--minimum-tuning-wait-actions", type=int, default=10)
    t4d1_parser.add_argument(
        "--output-dir",
        default="results/analysis/early_ho_timing/t4d1_ranked_probe_v1",
    )
    sinr_full_parser = subparsers.add_parser(
        "audit_sinr_rule_fullroute",
        help="Audit the target-SINR-worse rule on 40 post-TTT TRAIN confirmation routes.",
    )
    sinr_full_parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    sinr_full_parser.add_argument(
        "--route-split",
        default="results/analysis/early_ho_timing/post_ttt_train_bank_v1/route_split.json",
    )
    sinr_full_parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    sinr_full_parser.add_argument("--bootstrap-seed", type=int, default=43)
    sinr_full_parser.add_argument(
        "--output-dir",
        default="results/analysis/early_ho_timing/sinr_rule_fullroute_confirmation_v2",
    )
    audit_parser = subparsers.add_parser(
        "audit_candidates",
        help="Run the P1 train-only counterfactual candidate audit.",
    )
    audit_parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    audit_parser.add_argument("--speeds", type=int, nargs="+", default=None)
    audit_parser.add_argument("--train-fraction", type=float, default=0.70)
    audit_parser.add_argument("--validation-fraction", type=float, default=0.15)
    audit_parser.add_argument("--test-fraction", type=float, default=0.15)
    audit_parser.add_argument("--split-seed", type=int, default=17)
    audit_parser.add_argument("--sample-seed", type=int, default=17)
    audit_parser.add_argument("--routes-per-speed", type=int, default=20)
    audit_parser.add_argument("--max-decisions-per-route", type=int, default=4)
    audit_parser.add_argument("--decision-stride", type=int, default=10)
    audit_parser.add_argument(
        "--horizons", type=float, nargs="+", default=[0.5, 1.0, 2.0]
    )
    audit_parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional existing PPO checkpoint used as a second state source.",
    )
    audit_parser.add_argument(
        "--output-dir", default="results/audit/p1_candidates_s17"
    )
    observation_audit_parser = subparsers.add_parser(
        "audit_observations",
        help="Audit P2 observation saturation and causal prediction error.",
    )
    observation_audit_parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    observation_audit_parser.add_argument("--speeds", type=int, nargs="+", default=None)
    observation_audit_parser.add_argument(
        "--split", choices=("train", "validation"), default="train"
    )
    observation_audit_parser.add_argument("--split-seed", type=int, default=17)
    observation_audit_parser.add_argument("--sample-seed", type=int, default=17)
    observation_audit_parser.add_argument("--routes-per-speed", type=int, default=20)
    observation_audit_parser.add_argument(
        "--max-decisions-per-route", type=int, default=20
    )
    observation_audit_parser.add_argument(
        "--candidate-count", type=int, choices=(3, 5, 10), default=3
    )
    observation_audit_parser.add_argument(
        "--candidate-order", choices=("legacy", "ranked_v1"), default="ranked_v1"
    )
    observation_audit_parser.add_argument(
        "--observation-schema",
        choices=("rolling_v1", "monotonic_v1", "qos_full_v1"),
        default="qos_full_v1",
    )
    observation_audit_parser.add_argument(
        "--output-dir", default="results/audit/p2_observations"
    )
    learning_audit_parser = subparsers.add_parser(
        "audit_policy_learning",
        help="Build/probe the D1 TRAIN state bank or run D2 counterfactuals.",
    )
    learning_audit_parser.add_argument(
        "mode",
        choices=(
            "build-bank",
            "probe",
            "counterfactual",
            "supervised-probe",
            "transition-probe",
            "action-selection",
        ),
    )
    learning_audit_parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    learning_audit_parser.add_argument("--bank-dir", default=None)
    learning_audit_parser.add_argument("--speeds", type=int, nargs="+", default=None)
    learning_audit_parser.add_argument("--train-fraction", type=float, default=0.70)
    learning_audit_parser.add_argument(
        "--validation-fraction", type=float, default=0.15
    )
    learning_audit_parser.add_argument("--test-fraction", type=float, default=0.15)
    learning_audit_parser.add_argument("--split-seed", type=int, default=17)
    learning_audit_parser.add_argument("--sample-seed", type=int, default=17)
    learning_audit_parser.add_argument("--routes-per-speed", type=int, default=5)
    learning_audit_parser.add_argument(
        "--decision-indices", type=int, nargs="+", default=[0, 1, 2, 3]
    )
    learning_audit_parser.add_argument(
        "--candidate-count", type=int, choices=(3, 5, 10), default=3
    )
    learning_audit_parser.add_argument(
        "--candidate-order", choices=("legacy", "ranked_v1"), default="ranked_v1"
    )
    learning_audit_parser.add_argument(
        "--observation-schemas",
        nargs="+",
        choices=("legacy_v1", "rolling_v1", "monotonic_v1", "qos_full_v1"),
        default=["rolling_v1", "monotonic_v1", "qos_full_v1"],
    )
    learning_audit_parser.add_argument(
        "--source-checkpoint",
        action="append",
        default=[],
        metavar="LABEL=PATH",
    )
    learning_audit_parser.add_argument(
        "--checkpoint", action="append", default=[], metavar="LABEL=PATH"
    )
    learning_audit_parser.add_argument("--include-fresh", action="store_true")
    learning_audit_parser.add_argument("--seed", type=int, default=17)
    learning_audit_parser.add_argument(
        "--horizons", type=float, nargs="+", default=[0.5, 1.0, 2.0]
    )
    learning_audit_parser.add_argument("--gamma-per-second", type=float, default=0.99)
    learning_audit_parser.add_argument("--counterfactual-dir", default=None)
    learning_audit_parser.add_argument(
        "--probe-observation-schema",
        choices=("legacy_v1", "rolling_v1", "monotonic_v1", "qos_full_v1"),
        default="qos_full_v1",
    )
    learning_audit_parser.add_argument(
        "--probe-reward-schema", default="trajectory_qos_reward_v1"
    )
    learning_audit_parser.add_argument("--probe-horizon", type=float, default=1.0)
    learning_audit_parser.add_argument("--route-split-seed", type=int, default=29)
    learning_audit_parser.add_argument(
        "--rollout-probe-steps", type=int, default=1024
    )
    learning_audit_parser.add_argument(
        "--transition-horizon", type=float, default=1.0
    )
    learning_audit_parser.add_argument(
        "--stochastic-seeds", type=int, nargs="+", default=[17, 29, 43]
    )
    learning_audit_parser.add_argument(
        "--selection-horizon", type=float, default=1.0
    )
    learning_audit_parser.add_argument(
        "--output-dir", default="results/analysis/p23_diagnostics/state_bank_pilot"
    )
    d63_parser = subparsers.add_parser(
        "audit_timing_target",
        help="Run the D63 TRAIN-only timing x target correctness pilot.",
    )
    d63_parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    d63_parser.add_argument(
        "--d61-checkpoint",
        required=True,
        help="Frozen D61 selected checkpoint using a3_redecision_v1.",
    )
    d63_parser.add_argument(
        "--d62-checkpoint",
        required=True,
        help="Frozen D62 selected checkpoint using a3_redecision_v1.",
    )
    d63_parser.add_argument(
        "--speeds", type=int, nargs="+", default=[30, 50, 70, 90]
    )
    d63_parser.add_argument("--split-seed", type=int, default=17)
    d63_parser.add_argument("--sample-seed", type=int, default=17)
    d63_parser.add_argument("--routes-per-speed", type=int, default=5)
    d63_parser.add_argument(
        "--max-anchors-per-source-route", type=int, default=5
    )
    d63_parser.add_argument(
        "--delays-ms", type=int, nargs="+", default=[0, 100, 200, 300]
    )
    d63_parser.add_argument(
        "--horizons", type=float, nargs="+", default=[0.5, 1.0, 2.0]
    )
    d63_parser.add_argument("--gamma-per-second", type=float, default=0.99)
    d63_parser.add_argument(
        "--output-dir",
        default="results/analysis/d63_timing_target_pilot_v1",
    )
    d63_main_parser = subparsers.add_parser(
        "audit_timing_target_main",
        help=(
            "Analyze a completed 20-route/speed D63 TRAIN bank with route "
            "bootstrap and finite-lookahead full-route references."
        ),
    )
    d63_main_parser.add_argument("--local-audit-dir", required=True)
    d63_main_parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    d63_main_parser.add_argument("--bootstrap-seed", type=int, default=29)
    d63_main_parser.add_argument(
        "--delays-ms", type=int, nargs="+", default=[0, 100, 200, 300]
    )
    d63_main_parser.add_argument("--reference-horizon", type=float, default=1.0)
    d63_main_parser.add_argument(
        "--output-dir",
        default="results/analysis/d63_timing_target_main_analysis_v1",
    )
    d63_verify_parser = subparsers.add_parser(
        "verify_d63_1",
        help="Verify and freeze the completed TRAIN-only D63-1 package.",
    )
    d63_verify_parser.add_argument(
        "--main-audit-dir",
        default="results/analysis/d63_timing_target_main_analysis_v1",
    )
    d63_verify_parser.add_argument(
        "--output-dir",
        default="results/analysis/d63_1_verification_v1",
    )
    d63_verify_parser.add_argument(
        "--test-command",
        default=(
            ".venv\\Scripts\\python.exe -m unittest "
            "tests.test_d63_timing_target_audit tests.test_reference"
        ),
    )
    d63_labels_parser = subparsers.add_parser(
        "prepare_d63_2_labels",
        help="Create leakage-safe TRAIN timing/target labels from D63-1.",
    )
    d63_labels_parser.add_argument(
        "--bank-dir",
        default="results/analysis/d63_timing_target_main_local_v1",
    )
    d63_labels_parser.add_argument(
        "--main-audit-dir",
        default="results/analysis/d63_timing_target_main_analysis_v1",
    )
    d63_labels_parser.add_argument("--route-split-seed", type=int, default=29)
    d63_labels_parser.add_argument(
        "--output-dir", default="results/analysis/d63_2_labels_v1"
    )
    d63_probe_parser = subparsers.add_parser(
        "probe_d63_2",
        help="Fit fixed linear and small-MLP probes on D63-2 TRAIN labels.",
    )
    d63_probe_parser.add_argument(
        "--labels-dir", default="results/analysis/d63_2_labels_v1"
    )
    d63_probe_parser.add_argument("--seed", type=int, default=29)
    d63_probe_parser.add_argument(
        "--output-dir", default="results/analysis/d63_2_probe_v1"
    )
    d63_fp_parser = subparsers.add_parser(
        "analyze_d63_2_false_positives",
        help="Analyze frozen target-MLP false positives without refitting.",
    )
    d63_fp_parser.add_argument(
        "--labels-dir", default="results/analysis/d63_2_labels_v1"
    )
    d63_fp_parser.add_argument(
        "--probe-dir", default="results/analysis/d63_2_probe_v1"
    )
    d63_fp_parser.add_argument(
        "--output-dir", default="results/analysis/d63_2_target_mlp_fp_v1"
    )
    d63_replay_parser = subparsers.add_parser(
        "replay_d63_2_failure_cases",
        help="Replay frozen target-MLP FP cases and audit radio/RLM dynamics.",
    )
    d63_replay_parser.add_argument(
        "--bank-dir", default="results/analysis/d63_timing_target_main_local_v1"
    )
    d63_replay_parser.add_argument(
        "--labels-dir", default="results/analysis/d63_2_labels_v1"
    )
    d63_replay_parser.add_argument(
        "--probe-dir", default="results/analysis/d63_2_probe_v1"
    )
    d63_replay_parser.add_argument(
        "--output-dir", default="results/analysis/d63_2_failure_case_replay_v1"
    )
    d63_feature_parser = subparsers.add_parser(
        "validate_d63_3_feature",
        help="Validate the frozen execution-safety feature on TRAIN fit/tuning.",
    )
    d63_feature_parser.add_argument(
        "--bank-dir", default="results/analysis/d63_timing_target_main_local_v1"
    )
    d63_feature_parser.add_argument(
        "--labels-dir", default="results/analysis/d63_2_labels_v1"
    )
    d63_feature_parser.add_argument(
        "--output-dir", default="results/analysis/d63_3_execution_safety_v1"
    )
    summary_parser = subparsers.add_parser(
        "summarize_ablation",
        help="Validate and summarize completed PPO screening arms.",
    )
    summary_parser.add_argument(
        "--input-dir", default="results/models/ablation_v2"
    )
    summary_parser.add_argument(
        "--experiment-ids", nargs="+", required=True
    )
    summary_parser.add_argument("--expected-budget", type=int, default=None)
    summary_parser.add_argument(
        "--output-dir", default="results/analysis/ablation_summary"
    )
    train_parser = subparsers.add_parser(
        "train_ppo",
        help="Train duration-aware PPO on the SMDP environment.",
    )
    train_parser.add_argument(
        "--data-dir",
        default=DEFAULT_DATA_DIR,
        help="Directory containing train-ready RSRP/SINR CSV trace pairs.",
    )
    train_parser.add_argument(
        "--speeds",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Optional speed(s) in km/h used for the train/validation/test "
            "split; defaults to every discovered speed."
        ),
    )
    train_parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.70,
        help="Fraction of trajectories per speed used for training.",
    )
    train_parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.15,
        help="Fraction of trajectories per speed used for validation.",
    )
    train_parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.15,
        help="Fraction of trajectories per speed reserved for final testing.",
    )
    train_parser.add_argument("--total-macro-steps", type=int, default=100_000)
    train_parser.add_argument(
        "--candidate-count",
        type=int,
        choices=(1, 3, 5, 10),
        default=10,
        help="Number K of frozen handover candidates exposed to PPO.",
    )
    train_parser.add_argument(
        "--candidate-order",
        choices=("legacy", "ranked_v1"),
        default="legacy",
        help=(
            "legacy keeps the topology pool order; ranked_v1 promotes the "
            "best-current and causal +90 ms projected alternatives."
        ),
    )
    train_parser.add_argument(
        "--observation-schema",
        choices=(
            "legacy_v1",
            "rolling_v1",
            "monotonic_v1",
            "qos_full_v1",
            "t4d1_compact_v1",
        ),
        default="legacy_v1",
        help=(
            "Observation ablation: legacy, cell-ID rolling history, "
            "monotonic normalization, or the full QoS schema."
        ),
    )
    train_parser.add_argument(
        "--reward-schema",
        choices=(
            "legacy_handover_reward_v1",
            "trajectory_qos_reward_v1",
            "trajectory_qos_events_reward_v1",
            "trajectory_qos_recovery_reward_v1",
            "trajectory_qos_dual_reward_v1",
        ),
        default="legacy_handover_reward_v1",
        help=(
            "Reward ablation: legacy, trajectory-normalized QoS utility, "
            "rescaled events, recovery-time cost, or per-speed dual QoS."
        ),
    )
    train_parser.add_argument(
        "--stay-mode",
        choices=("reject_event_v1", "a3_redecision_v1"),
        default="reject_event_v1",
        help=(
            "STAY either suppresses the completed A3 event until rearm or "
            "offers the same still-valid event again after a fixed delay."
        ),
    )
    train_parser.add_argument(
        "--decision-trigger",
        choices=("completed_a3_ttt", "a3_entry_redecision_v1"),
        default="completed_a3_ttt",
        help=(
            "completed_a3_ttt preserves the legacy/T5A trigger; "
            "a3_entry_redecision_v1 exposes STAY/HO_C1 immediately when "
            "the standard A3 entering inequality first becomes true while "
            "retaining the 40 ms TTT as an observational shadow."
        ),
    )
    train_parser.add_argument(
        "--stay-redecision-ms",
        type=int,
        default=100,
        help="Delay before re-offering a still-valid A3 event in redecision mode.",
    )
    train_parser.add_argument(
        "--a3-offset-db",
        type=float,
        default=None,
        help=(
            "Optional A3 offset used by the PPO environment. The dataset/default "
            "value is retained when omitted."
        ),
    )
    train_parser.add_argument(
        "--reference-a3-offset-db",
        type=float,
        default=None,
        help=(
            "Optional A3 offset used only by the 3GPP Always-HO validation "
            "reference. Omit to use the PPO environment offset."
        ),
    )
    train_parser.add_argument("--dual-target-ratio", type=float, default=0.98)
    train_parser.add_argument("--dual-learning-rate", type=float, default=0.5)
    train_parser.add_argument(
        "--dual-max-coefficient", type=float, default=10.0
    )
    train_parser.add_argument(
        "--dual-min-episodes-per-speed", type=int, default=4
    )
    train_parser.add_argument("--rollout-steps", type=int, default=1024)
    train_parser.add_argument("--batch-size", type=int, default=128)
    train_parser.add_argument("--update-epochs", type=int, default=4)
    train_parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[64, 64])
    train_parser.add_argument(
        "--network-architecture",
        choices=(
            "shared_mlp_v1",
            "separate_mlp_v1",
            "candidate_scorer_v1",
            "hierarchical_candidate_scorer_v1",
        ),
        default="shared_mlp_v1",
        help=(
            "Actor architecture: flat MLP/scorer or D62 timing plus "
            "conditional candidate heads."
        ),
    )
    train_parser.add_argument(
        "--value-normalization",
        choices=("none", "fixed_scale_v1"),
        default="none",
        help="D3-B critic target scaling; rewards and advantages remain raw.",
    )
    train_parser.add_argument(
        "--value-target-scale",
        type=float,
        default=None,
        help=(
            "Fixed critic scale. Omit with fixed_scale_v1 to fit it from a "
            "frozen TRAIN Always-HO-C1 calibration subset."
        ),
    )
    train_parser.add_argument(
        "--value-scale-calibration-routes-per-speed",
        type=int,
        default=20,
        help="TRAIN routes per speed used when auto-fitting D3-B scale.",
    )
    train_parser.add_argument("--learning-rate", type=float, default=5e-5)
    train_parser.add_argument("--clip-range", type=float, default=0.2)
    train_parser.add_argument("--value-coefficient", type=float, default=0.5)
    train_parser.add_argument("--entropy-coefficient", type=float, default=0.01)
    train_parser.add_argument("--max-grad-norm", type=float, default=0.5)
    train_parser.add_argument("--gamma-per-second", type=float, default=0.99)
    train_parser.add_argument("--lambda-per-second", type=float, default=0.95)
    train_parser.add_argument("--target-kl", type=float, default=0.03)
    train_parser.add_argument(
        "--learning-diagnostics",
        action="store_true",
        help=(
            "Log first-minibatch actor/critic/entropy gradient diagnostics; "
            "disabled by default and compatible with legacy checkpoints."
        ),
    )
    train_parser.add_argument("--evaluation-interval", type=int, default=10_240)
    train_parser.add_argument(
        "--stochastic-validation-seeds",
        type=int,
        nargs="*",
        default=[],
        help=(
            "Optional sampled-action validation seeds evaluated beside each "
            "deterministic validation. These diagnostics never select a checkpoint."
        ),
    )
    train_parser.add_argument(
        "--qos-rate-ratio",
        type=float,
        default=0.98,
        help=(
            "Minimum validation mean-rate ratio to the 3GPP Always-HO "
            "reference for a best-QoS checkpoint."
        ),
    )
    train_parser.add_argument(
        "--qos-rate-ratio-per-speed",
        type=float,
        default=0.98,
        help=(
            "Minimum validation rate ratio to 3GPP at every selected speed."
        ),
    )
    train_parser.add_argument(
        "--qos-relative-rate-ratio",
        type=float,
        default=0.98,
        help="Minimum aggregate trajectory-relative-rate ratio to 3GPP.",
    )
    train_parser.add_argument(
        "--qos-relative-rate-ratio-per-speed",
        type=float,
        default=0.98,
        help="Minimum relative-rate ratio to 3GPP at every selected speed.",
    )
    train_parser.add_argument(
        "--progress-interval",
        type=int,
        default=50,
        help="Print collection progress every N macro-steps.",
    )
    train_parser.add_argument("--seed", type=int, default=17)
    train_parser.add_argument("--split-seed", type=int, default=17)
    train_parser.add_argument(
        "--experiment-id",
        default=None,
        help="Stable ablation ID; defaults to the output directory name.",
    )
    train_parser.add_argument(
        "--parent-id",
        default=None,
        help="Parent experiment ID used for the one-change comparison.",
    )
    train_parser.add_argument(
        "--changed-factor",
        default="baseline_contract",
        help="Single preregistered factor changed from the parent.",
    )
    train_parser.add_argument(
        "--skip-final-test",
        action="store_true",
        help=(
            "Do not instantiate or evaluate the held-out test split; use for "
            "development ablations."
        ),
    )
    train_parser.add_argument("--device", default="cpu")
    train_parser.add_argument(
        "--output-dir",
        default=DEFAULT_MODEL_DIR,
    )

    evaluate_parser = subparsers.add_parser(
        "evaluate_ppo",
        help="Evaluate a saved SMDP PPO checkpoint.",
    )
    evaluate_parser.add_argument(
        "--checkpoint",
        default=os.path.join(DEFAULT_MODEL_DIR, "smdp_ppo_final.pt"),
    )
    evaluate_parser.add_argument(
        "--split",
        choices=("validation", "test"),
        default="test",
    )
    evaluate_parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    evaluate_parser.add_argument("--speeds", nargs="+", type=int, default=None)
    evaluate_parser.add_argument("--output", default=None, help="Optional JSON report path.")
    evaluate_parser.add_argument("--split-seed", type=int, default=None)
    evaluate_parser.add_argument("--device", default="cpu")
    benchmark_parser = subparsers.add_parser(
        "benchmark",
        help="Compare PPO with 3GPP Always-HO.",
    )
    benchmark_parser.add_argument(
        "--data-dir",
        default=DEFAULT_DATA_DIR,
        help="Directory containing train-ready RSRP/SINR CSV trace pairs.",
    )
    benchmark_parser.add_argument(
        "--checkpoint",
        default=os.path.join(DEFAULT_MODEL_DIR, "smdp_ppo_final.pt"),
    )
    benchmark_parser.add_argument(
        "--split",
        choices=("validation", "test", "all"),
        default="test",
        help=(
            "Trajectory subset to benchmark. Use 'all' for zero-shot "
            "evaluation on every selected trajectory."
        ),
    )
    benchmark_parser.add_argument("--split-seed", type=int, default=None)
    benchmark_parser.add_argument(
        "--train-fraction",
        type=float,
        default=None,
        help=(
            "Optional split override; defaults to the fraction stored in "
            "the PPO checkpoint."
        ),
    )
    benchmark_parser.add_argument(
        "--validation-fraction",
        type=float,
        default=None,
        help=(
            "Optional split override; defaults to the fraction stored in "
            "the PPO checkpoint."
        ),
    )
    benchmark_parser.add_argument(
        "--test-fraction",
        type=float,
        default=None,
        help=(
            "Optional split override; defaults to the fraction stored in "
            "the PPO checkpoint."
        ),
    )
    benchmark_parser.add_argument(
        "--speeds",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Optional speed(s) in km/h to benchmark; defaults to every "
            "discovered speed."
        ),
    )
    benchmark_parser.add_argument(
        "--zero-shot-speeds",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Optional speed(s) evaluated on every trajectory and combined "
            "with the requested validation/test split. These speeds must "
            "not overlap --speeds."
        ),
    )
    benchmark_parser.add_argument("--device", default="cpu")
    benchmark_parser.add_argument(
        "--output-dir",
        default=os.path.join("results", "benchmark", "uav_generated_ppo"),
    )

    args = parser.parse_args(argv)
    if args.command == "validate_3gpp":
        from ho_optim_drl.commands import validate_3gpp

        return validate_3gpp.main(root_path, args)
    if args.command == "audit_baseline_parity":
        from ho_optim_drl.commands import audit_baseline_parity

        return audit_baseline_parity.main(root_path, args)
    if args.command == "audit_b1_lower_threshold":
        from ho_optim_drl.commands import audit_b1_lower_threshold

        return audit_b1_lower_threshold.main(root_path, args)
    if args.command == "audit_b2_b3_early_controls":
        from ho_optim_drl.commands import audit_b2_b3_early_controls

        return audit_b2_b3_early_controls.main(root_path, args)
    if args.command == "audit_early_timing_references":
        from ho_optim_drl.commands import audit_early_timing_references

        return audit_early_timing_references.main(root_path, args)
    if args.command == "audit_early_timing_bank":
        from ho_optim_drl.commands import audit_early_timing_bank

        return audit_early_timing_bank.main(root_path, args)
    if args.command == "audit_early_timing_routes":
        from ho_optim_drl.commands import audit_early_timing_routes

        return audit_early_timing_routes.main(root_path, args)
    if args.command == "audit_t5a_benefit_sources":
        from ho_optim_drl.commands import audit_t5a_benefit_sources

        return audit_t5a_benefit_sources.main(root_path, args)
    if args.command == "audit_t6a_seed_divergence":
        from ho_optim_drl.commands import audit_t6a_seed_divergence

        return audit_t6a_seed_divergence.main(root_path, args)
    if args.command == "verify_early_timing_t2_t3":
        from ho_optim_drl.commands import verify_early_timing_t2_t3

        return verify_early_timing_t2_t3.main(root_path, args)
    if args.command == "audit_early_timing_learnability":
        from ho_optim_drl.commands import audit_early_timing_learnability

        return audit_early_timing_learnability.main(root_path, args)
    if args.command == "audit_early_timing_observation_ablation":
        from ho_optim_drl.commands import audit_early_timing_learnability

        return audit_early_timing_learnability.main(root_path, args)
    if args.command == "analyze_t4b_errors":
        from ho_optim_drl.commands import analyze_t4b_errors

        return analyze_t4b_errors.main(root_path, args)
    if args.command == "build_post_ttt_train_bank":
        from ho_optim_drl.commands import build_post_ttt_train_bank

        return build_post_ttt_train_bank.main(root_path, args)
    if args.command == "audit_t4c_post_ttt_probe":
        from ho_optim_drl.commands import audit_t4c_post_ttt_probe

        return audit_t4c_post_ttt_probe.main(root_path, args)
    if args.command == "audit_t4d_compact_causal_probe":
        from ho_optim_drl.commands import audit_t4d_compact_causal_probe

        return audit_t4d_compact_causal_probe.main(root_path, args)
    if args.command == "audit_t4d1_ranked_probe":
        from ho_optim_drl.commands import audit_t4d1_ranked_probe

        return audit_t4d1_ranked_probe.main(root_path, args)
    if args.command == "audit_sinr_rule_fullroute":
        from ho_optim_drl.commands import audit_sinr_rule_fullroute

        return audit_sinr_rule_fullroute.main(root_path, args)
    if args.command == "audit_candidates":
        from ho_optim_drl.commands import audit_candidates

        return audit_candidates.main(root_path, args)
    if args.command == "audit_observations":
        from ho_optim_drl.commands import audit_observations

        return audit_observations.main(root_path, args)
    if args.command == "audit_policy_learning":
        from ho_optim_drl.commands import audit_policy_learning

        if args.mode in {
            "probe",
            "counterfactual",
            "supervised-probe",
            "action-selection",
        } and not args.bank_dir:
            parser.error(f"audit_policy_learning {args.mode} requires --bank-dir")
        if args.mode == "supervised-probe" and not args.counterfactual_dir:
            parser.error(
                "audit_policy_learning supervised-probe requires --counterfactual-dir"
            )
        if args.mode == "action-selection" and len(args.checkpoint) != 1:
            parser.error(
                "audit_policy_learning action-selection requires exactly one "
                "--checkpoint LABEL=PATH"
            )
        return audit_policy_learning.main(root_path, args)
    if args.command == "audit_timing_target":
        from ho_optim_drl.commands import audit_timing_target

        return audit_timing_target.main(root_path, args)
    if args.command == "audit_timing_target_main":
        from ho_optim_drl.commands import audit_timing_target_main

        return audit_timing_target_main.main(root_path, args)
    if args.command == "verify_d63_1":
        from ho_optim_drl.commands import verify_d63_1

        return verify_d63_1.main(root_path, args)
    if args.command == "prepare_d63_2_labels":
        from ho_optim_drl.commands import prepare_d63_2_labels

        return prepare_d63_2_labels.main(root_path, args)
    if args.command == "probe_d63_2":
        from ho_optim_drl.commands import probe_d63_2

        return probe_d63_2.main(root_path, args)
    if args.command == "analyze_d63_2_false_positives":
        from ho_optim_drl.commands import analyze_d63_2_false_positives

        return analyze_d63_2_false_positives.main(root_path, args)
    if args.command == "replay_d63_2_failure_cases":
        from ho_optim_drl.commands import replay_d63_2_failure_cases

        return replay_d63_2_failure_cases.main(root_path, args)
    if args.command == "validate_d63_3_feature":
        from ho_optim_drl.commands import validate_d63_3_feature

        return validate_d63_3_feature.main(root_path, args)
    if args.command == "summarize_ablation":
        from ho_optim_drl.commands import summarize_ablation

        return summarize_ablation.main(root_path, args)
    if args.command == "train_ppo":
        from ho_optim_drl.commands import train_smdp_ppo

        return train_smdp_ppo.main(root_path, args)
    if args.command == "evaluate_ppo":
        from ho_optim_drl.commands import evaluate_smdp_ppo

        return evaluate_smdp_ppo.main(root_path, args)
    if args.command == "benchmark":
        from ho_optim_drl.commands import benchmark_policies

        return benchmark_policies.main(root_path, args)

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(run())
