"""Dataset radio configuration and portable, checkpoint-bound held-out splits."""

import hashlib
import json
import math
from pathlib import Path

from ho_optim_drl.config import Config
from .split import TracePair, discover_trace_pairs, filter_trace_pairs_by_speed, stratified_trajectory_split


DEFAULT_DATA_DIR = "data/uav_generated"
DEFAULT_MODEL_DIR = "results/models/uav_generated_ppo"


def file_sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_protocol_config(data_dir, checkpoint_extra=None):
    """Use dataset radio parameters and retain the checkpoint's protocol rules.

    Legacy datasets without metadata retain the original Config defaults.
    """
    saved = (checkpoint_extra or {}).get("protocol_config")
    config = Config(**saved) if saved else Config()
    directory = Path(data_dir)
    metadata_path = directory / "dataset_metadata.json"
    generation_path = directory / "simulation_manifest.json"
    if metadata_path.is_file() and generation_path.is_file():
        metadata_radio = json.loads(
            metadata_path.read_text(encoding="utf-8")
        )["radio"]
        generation_radio = json.loads(
            generation_path.read_text(encoding="utf-8")
        )["radio"]
        if metadata_radio != generation_radio:
            raise ValueError(
                "dataset_metadata.json and simulation_manifest.json declare "
                "different radio configurations."
            )
    declaration_path = metadata_path if metadata_path.is_file() else generation_path
    if declaration_path.is_file():
        declaration = json.loads(declaration_path.read_text(encoding="utf-8"))
        radio = declaration["radio"]
        values = {"fc": radio["carrier_hz"], "bw": radio["bandwidth_hz"], "delta_t_ms": radio["delta_t_ms"]}
        for name, value in values.items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"Invalid dataset radio parameter {name}: {value}")
            if saved and getattr(config, name) != value:
                raise ValueError(f"Dataset {name}={value} differs from checkpoint {getattr(config, name)}.")
        config.update(values)
    return config


def checkpoint_pairs(data_dir, checkpoint_path, extra, *, split, speeds=None, split_seed=None, fractions=None):
    """Recover exact held-out files, validating hashes after a project move.

    Old checkpoints without a manifest fall back to their recorded split rules.
    Explicit all-trace/zero-shot evaluation is handled by the benchmark command.
    """
    if split not in {"validation", "test"}:
        raise ValueError("Only validation and test are held-out splits.")
    saved_seed = extra.get("split_seed", 17)
    saved_fractions = extra.get("split_fractions", {"train": 0.70, "validation": 0.15, "test": 0.15})
    manifest_name = extra.get("dataset_manifest_file")
    manifest_path = Path(checkpoint_path).parent / manifest_name if manifest_name else None
    if manifest_path is not None:
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Checkpoint split manifest is missing: {manifest_path}")
        expected_hash = extra.get("dataset_manifest_sha256")
        if expected_hash and file_sha256(manifest_path) != expected_hash:
            raise ValueError("Checkpoint split manifest fingerprint does not match.")
        if split_seed is not None and split_seed != saved_seed:
            raise ValueError("split_seed differs from the checkpoint's saved split.")
        if fractions and any(value is not None and value != saved_fractions[name] for name, value in fractions.items()):
            raise ValueError("Split fractions differ from the checkpoint's saved split.")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        provenance = manifest.get("dataset_provenance")
        if provenance is not None:
            provenance_path = Path(data_dir) / provenance["file"]
            if not provenance_path.is_file():
                raise FileNotFoundError(
                    "Checkpoint topology provenance is missing: "
                    f"{provenance_path}"
                )
            if file_sha256(provenance_path) != provenance["sha256"]:
                raise ValueError(
                    "Checkpoint topology provenance fingerprint changed: "
                    f"{provenance_path}"
                )
        records = manifest["splits"][split]
        pairs = [TracePair(row["speed_kmh"], row["trajectory_id"], row["rsrp_file"], row["sinr_file"]) for row in records]
        selected = filter_trace_pairs_by_speed(pairs, speeds)
        by_key = {(row["speed_kmh"], row["trajectory_id"]): row for row in records}
        for pair in selected:
            record = by_key[pair.key]
            for kind in ("rsrp", "sinr"):
                path = Path(data_dir) / record[f"{kind}_file"]
                if file_sha256(path) != record[f"{kind}_sha256"]:
                    raise ValueError(f"Held-out dataset fingerprint changed: {path}")
    else:
        pairs = filter_trace_pairs_by_speed(discover_trace_pairs(data_dir), extra.get("selected_speeds_kmh"))
        resolved = {key: (fractions or {}).get(key) if (fractions or {}).get(key) is not None else value for key, value in saved_fractions.items()}
        splits = stratified_trajectory_split(pairs, seed=saved_seed if split_seed is None else split_seed, **{f"{key}_fraction": value for key, value in resolved.items()})
        selected = filter_trace_pairs_by_speed(list(getattr(splits, split)), speeds)
    if not selected:
        raise ValueError(f"The checkpoint's {split} split is empty.")
    return tuple(selected)
