"""Project command-line entry point."""

import argparse
import os
import sys

THIS_PATH = os.path.dirname(os.path.abspath(__file__))


def run() -> int:
    parser = argparse.ArgumentParser(
        description="Trace-driven adaptive handover research framework."
    )
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser(
        "validate_3gpp",
        help="Run the audited A3-based reference on all CSV traces.",
    )
    train_parser = subparsers.add_parser(
        "train_ppo",
        help="Train duration-aware PPO on the SMDP environment.",
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
        default=os.path.join("results", "models", "smdp_ppo"),
    )

    evaluate_parser = subparsers.add_parser(
        "evaluate_ppo",
        help="Evaluate a saved SMDP PPO checkpoint.",
    )
    evaluate_parser.add_argument(
        "--checkpoint",
        default=os.path.join(
            "results",
            "models",
            "smdp_ppo",
            "smdp_ppo_final.pt",
        ),
    )
    evaluate_parser.add_argument(
        "--split",
        choices=("validation", "test"),
        default="test",
    )
    evaluate_parser.add_argument("--split-seed", type=int, default=17)
    evaluate_parser.add_argument("--device", default="cpu")

    args = parser.parse_args()
    if args.command == "validate_3gpp":
        from scripts import validate_3gpp

        return validate_3gpp.main(THIS_PATH)
    if args.command == "train_ppo":
        from scripts import train_smdp_ppo

        return train_smdp_ppo.main(THIS_PATH, args)
    if args.command == "evaluate_ppo":
        from scripts import evaluate_smdp_ppo

        return evaluate_smdp_ppo.main(THIS_PATH, args)

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(run())
