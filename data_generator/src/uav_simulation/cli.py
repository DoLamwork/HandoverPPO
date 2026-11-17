"""Command dispatch with lazy imports so help does not initialize plotting."""

import argparse
from importlib import import_module
import sys


COMMANDS = {
    "generate": ("batch.generate_dataset", "Generate train-ready radio traces for HandoverOptimDRL"),
    "simulate": ("interactive", "Run a simulation with interactive parameter prompts"),
    "batch": ("batch.generate_batch", "Generate the 100 m-altitude dataset"),
    "batch-300m": ("batch.generate_batch_300m", "Generate the 300 m-altitude dataset"),
    "batch-grid100": ("batch.generate_batch_grid100_h100", "Generate the 100 m-grid dataset"),
    "route": ("visualization.route_viewer", "Display one seeded route"),
    "heatmap": ("visualization.heatmap_viewer", "Display or save a route occupancy heatmap"),
    "plot-antenna": ("visualization.plot_antenna_pattern", "Export 3D antenna patterns"),
    "plot-system": ("visualization.plot_system_model", "Export the system-model figure"),
    "validate": ("validation.channel_statistics", "Validate channel statistics"),
    "audit-rsrp-factors": (
        "validation.rsrp_factor_audit",
        "Replay a manifested P0 dataset and rank RSRP variation factors",
    ),
}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        prog="uav-sim",
        description="UAV-to-ground cellular network simulation tools.",
        epilog="Run uav-sim COMMAND --help for command-specific options.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command, (_, description) in COMMANDS.items():
        subparsers.add_parser(command, help=description, add_help=False)

    # Let each command own its arguments, including --help.
    args = parser.parse_args(argv[:1])
    module_name, _ = COMMANDS[args.command]
    module = import_module(f"uav_simulation.{module_name}")
    return module.main(argv[1:])
