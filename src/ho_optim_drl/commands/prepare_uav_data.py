"""Prepare simulation CSV pairs and preserve radio configuration and provenance."""

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path

from ho_optim_drl.commands.convert_uav_csv import convert_directory, _discover_source_pairs, _select_source_pairs
from ho_optim_drl.config import Config
from ho_optim_drl.data.project import DEFAULT_DATA_DIR, file_sha256


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("data/UAV_data_100m"))
    parser.add_argument("--output-dir", type=Path, default=Path(DEFAULT_DATA_DIR))
    parser.add_argument("--speeds", nargs="+", type=int)
    parser.add_argument("--source-trace-ids", nargs="+", type=int)
    parser.add_argument("--carrier-ghz", type=float, help="Declared carrier for legacy traces without a generation manifest")
    parser.add_argument("--bandwidth-mhz", type=float, help="Declared bandwidth for legacy traces without a generation manifest")
    parser.add_argument("--delta-t-ms", type=int, help="Declared sample interval for legacy traces without a generation manifest")
    args = parser.parse_args(argv)
    source, destination = args.input_dir.resolve(), args.output_dir.resolve()
    if source == destination:
        raise ValueError("Raw and prepared data directories must differ.")
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError("Prepared output directory must be empty; use a new dataset directory to preserve split identities.")
    manifest_path = source / "simulation_manifest.json"
    generation = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None
    declared = {
        "carrier_hz": None if args.carrier_ghz is None else args.carrier_ghz * 1e9,
        "bandwidth_hz": None if args.bandwidth_mhz is None else args.bandwidth_mhz * 1e6,
        "delta_t_ms": args.delta_t_ms,
    }
    protocol_defaults = Config()
    radio = dict(generation["radio"]) if generation else {
        "carrier_hz": protocol_defaults.fc,
        "bandwidth_hz": protocol_defaults.bw,
        "delta_t_ms": protocol_defaults.delta_t_ms,
    }
    for key, value in declared.items():
        if value is not None:
            if generation and radio[key] != value:
                raise ValueError(f"{key} conflicts with the generation manifest.")
            radio[key] = value
    if any(not math.isfinite(value) or value <= 0 for value in radio.values()):
        raise ValueError("Radio parameters must be positive finite values.")
    Config(fc=radio["carrier_hz"], bw=radio["bandwidth_hz"], delta_t_ms=radio["delta_t_ms"])
    pairs = _select_source_pairs(_discover_source_pairs(source), speeds=args.speeds, source_trace_ids=args.source_trace_ids)
    # Keep route IDs stable across conversions and speed subsets. Pipeline
    # generation uses one UAV; multiple UEs sharing IDs are rejected by converter.
    count = convert_directory(source, destination, radio["delta_t_ms"], False,
                              speeds=args.speeds, source_trace_ids=args.source_trace_ids,
                              preserve_source_trace_ids=True)
    records = []
    for pair in pairs:
        record = {"ue": pair.ue, "speed_kmh": pair.speed_kmh, "source_trace_id": pair.source_trace_id}
        for kind in ("rsrp", "sinr"):
            raw_path = getattr(pair, f"{kind}_path")
            output = destination / f"{kind}_{pair.speed_kmh}kmh_{pair.source_trace_id}.csv"
            record[f"source_{kind}_file"] = raw_path.name
            record[f"source_{kind}_sha256"] = file_sha256(raw_path)
            record[f"{kind}_file"] = output.name
            record[f"{kind}_sha256"] = file_sha256(output)
        records.append(record)
    metadata = {
        "schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_dir": str(source), "radio": radio,
        "radio_provenance": "generation_manifest" if generation else "declared_legacy_parameters",
        "generation": generation, "trajectory_count": count, "trajectories": records,
        "csv_schema": "sample,time_ms,bs_0,...", "split_unit": "whole trajectory",
    }
    (destination / "dataset_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Prepared {count} trajectories: {destination}")
    print(f"Radio configuration ({metadata['radio_provenance']}): {radio}")
    return 0
