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
    train_parser.add_argument("--rollout-steps", type=int, default=512)
    train_parser.add_argument("--batch-size", type=int, default=64)
    train_parser.add_argument("--update-epochs", type=int, default=10)
    train_parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[64, 64])
    train_parser.add_argument("--learning-rate", type=float, default=3e-4)
    train_parser.add_argument("--clip-range", type=float, default=0.2)
    train_parser.add_argument("--value-coefficient", type=float, default=0.5)
    train_parser.add_argument("--entropy-coefficient", type=float, default=0.01)
    train_parser.add_argument("--max-grad-norm", type=float, default=0.5)
    train_parser.add_argument("--gamma-per-second", type=float, default=0.99)
    train_parser.add_argument("--lambda-per-second", type=float, default=0.95)
    train_parser.add_argument("--target-kl", type=float, default=0.03)
    train_parser.add_argument("--evaluation-interval", type=int, default=5_000)
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
        "--progress-interval",
        type=int,
        default=50,
        help="Print collection progress every N macro-steps.",
    )
    train_parser.add_argument("--seed", type=int, default=17)
    train_parser.add_argument("--split-seed", type=int, default=17)
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
