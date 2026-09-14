"""Reproducible provenance manifests for UAV radio-trace datasets."""

from __future__ import annotations

from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Mapping

from uav_simulation import config


_TRACE_PATTERN = re.compile(r"(rsrp|sinr)_(\d+)kmh_(\d+)\.csv")


def _distribution_version() -> str:
    try:
        return version("handover-ppo")
    except PackageNotFoundError:
        return "source-tree"


def _source_fingerprints() -> tuple[str, dict[str, str]]:
    """Hash the installed generator source so a dataset pins exact code bytes."""
    package_root = Path(__file__).resolve().parent
    file_hashes: dict[str, str] = {}
    tree_hash = sha256()
    for path in sorted(package_root.rglob("*.py")):
        relative = path.relative_to(package_root).as_posix()
        digest = sha256(path.read_bytes()).hexdigest()
        file_hashes[relative] = digest
        tree_hash.update(relative.encode("utf-8"))
        tree_hash.update(b"\0")
        tree_hash.update(bytes.fromhex(digest))
    return tree_hash.hexdigest(), file_hashes


def build_generation_manifest(
    parameters: Mapping[str, object],
    *,
    entrypoint: str,
) -> dict[str, object]:
    """Build a deterministic, JSON-compatible generator manifest."""
    source_tree_sha256, source_files = _source_fingerprints()
    manifest = {
        "schema_version": 3,
        "generator": "uav_simulation",
        "generator_provenance": {
            "distribution": "handover-ppo",
            "distribution_version": _distribution_version(),
            "entrypoint": entrypoint,
            "source_tree_sha256": source_tree_sha256,
            "source_files_sha256": source_files,
        },
        "parameters": dict(parameters),
        "seed_rule": "speed_kmh * 1000 + route_id",
        "output_contract": "handover_ppo_train_ready_v1",
        "csv_schema": "sample,time_ms,bs_0,...",
        "split_unit": "whole trajectory",
        "radio": {
            "carrier_hz": config.FC * 1e9,
            "bandwidth_hz": config.BANDWIDTH_MHZ * 1e6,
            "delta_t_ms": config.DELTA_T_MS,
        },
        "simulation_config": {
            name: value
            for name, value in vars(config).items()
            if name.isupper() and not name.startswith("_")
        },
    }
    # Normalize tuples and mapping keys before comparing manifests on resume.
    return json.loads(json.dumps(manifest))


def ensure_generation_manifest(
    output_dir: str | Path,
    manifest: Mapping[str, object],
) -> Path:
    """Write a new manifest or verify an exact match before resuming output."""
    directory = Path(output_dir).resolve()
    manifest_path = directory / "simulation_manifest.json"
    normalized = json.loads(json.dumps(dict(manifest)))
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing != normalized:
            raise ValueError(
                "Existing dataset has a different generator, source hash, or "
                "generation configuration; choose a new output directory."
            )
        return manifest_path
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(
            "Existing traces have no generation manifest and cannot be assigned "
            "unverified provenance. Generate into a new output directory."
        )
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(normalized, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def train_ready_csv_is_complete(path: str | Path, expected_rows: int) -> bool:
    """Return whether one CSV satisfies the PPO loader's schema and row count."""
    csv_path = Path(path)
    if not csv_path.is_file():
        return False
    try:
        with csv_path.open("r", encoding="utf-8-sig") as stream:
            columns = stream.readline().strip().split(",")
            rows = sum(1 for _ in stream)
    except OSError:
        return False
    expected_bs = [f"bs_{index}" for index in range(max(0, len(columns) - 2))]
    return (
        rows == expected_rows
        and len(columns) >= 3
        and columns[:2] == ["sample", "time_ms"]
        and columns[2:] == expected_bs
    )


def write_train_ready_dataset_metadata(
    output_dir: str | Path,
    manifest: Mapping[str, object],
    *,
    expected_keys: set[tuple[int, int]] | None = None,
) -> Path:
    """Validate a completed generated dataset and write its training metadata."""
    directory = Path(output_dir).resolve()
    discovered: dict[tuple[int, int], dict[str, Path]] = {}
    unrecognized = []
    for path in sorted(directory.glob("*.csv")):
        match = _TRACE_PATTERN.fullmatch(path.name)
        if match is None:
            unrecognized.append(path.name)
            continue
        kind, speed, trajectory_id = match.groups()
        discovered.setdefault((int(speed), int(trajectory_id)), {})[kind] = path
    if unrecognized:
        raise ValueError(f"CSV files outside the train-ready contract: {unrecognized}")

    incomplete = {
        key: sorted({"rsrp", "sinr"} - set(files))
        for key, files in discovered.items()
        if set(files) != {"rsrp", "sinr"}
    }
    if incomplete:
        raise ValueError(f"Incomplete generated trace pairs: {incomplete}")
    if expected_keys is not None and set(discovered) != expected_keys:
        missing = sorted(expected_keys - set(discovered))
        unexpected = sorted(set(discovered) - expected_keys)
        raise ValueError(
            f"Generated trajectory set mismatch: missing={missing}, "
            f"unexpected={unexpected}"
        )
    if not discovered:
        raise ValueError("Generated dataset contains no trace pairs.")

    parameters = dict(manifest.get("parameters", {}))
    expected_rows = parameters.get("steps")
    records = []
    for (speed, trajectory_id), files in sorted(discovered.items()):
        headers = []
        record = {"speed_kmh": speed, "trajectory_id": trajectory_id}
        for kind in ("rsrp", "sinr"):
            path = files[kind]
            if expected_rows is not None and not train_ready_csv_is_complete(
                path, int(expected_rows)
            ):
                raise ValueError(
                    f"{path} does not satisfy the train-ready CSV schema or "
                    f"expected {expected_rows} rows."
                )
            with path.open("r", encoding="utf-8-sig") as stream:
                headers.append(stream.readline().strip())
            record[f"{kind}_file"] = path.name
            record[f"{kind}_sha256"] = sha256(path.read_bytes()).hexdigest()
        if headers[0] != headers[1]:
            raise ValueError(
                f"RSRP/SINR columns differ for trajectory {(speed, trajectory_id)}."
            )
        records.append(record)

    metadata_path = directory / "dataset_metadata.json"
    created_at = datetime.now(timezone.utc).isoformat()
    if metadata_path.is_file():
        try:
            created_at = json.loads(
                metadata_path.read_text(encoding="utf-8")
            ).get("created_at_utc", created_at)
        except (OSError, json.JSONDecodeError):
            pass
    metadata = {
        "schema_version": 1,
        "created_at_utc": created_at,
        "dataset_format": "handover_ppo_train_ready_v1",
        "radio": dict(manifest["radio"]),
        "radio_provenance": "generation_manifest",
        "generation": dict(manifest),
        "trajectory_count": len(records),
        "trajectories": records,
        "csv_schema": "sample,time_ms,bs_0,...",
        "split_unit": "whole trajectory",
    }
    temporary_path = metadata_path.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(metadata_path)
    return metadata_path
