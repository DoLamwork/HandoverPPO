"""Trajectory-level discovery and deterministic stratified splitting."""

from dataclasses import dataclass
import os
import re

import numpy as np


@dataclass(frozen=True, order=True)
class TracePair:
    """One indivisible RSRP/SINR trajectory pair."""

    speed_kmh: int
    trajectory_id: int
    rsrp_file: str
    sinr_file: str

    @property
    def key(self) -> tuple[int, int]:
        return self.speed_kmh, self.trajectory_id


@dataclass(frozen=True)
class TraceSplits:
    """Disjoint train, validation, and test trajectory sets."""

    train: tuple[TracePair, ...]
    validation: tuple[TracePair, ...]
    test: tuple[TracePair, ...]

    def validate(self) -> None:
        groups = {
            "train": {pair.key for pair in self.train},
            "validation": {pair.key for pair in self.validation},
            "test": {pair.key for pair in self.test},
        }
        if groups["train"] & groups["validation"]:
            raise ValueError("Train and validation splits overlap.")
        if groups["train"] & groups["test"]:
            raise ValueError("Train and test splits overlap.")
        if groups["validation"] & groups["test"]:
            raise ValueError("Validation and test splits overlap.")


_TRACE_PATTERN = re.compile(r"(rsrp|sinr)_(\d+)kmh_(\d+)\.csv")


def discover_trace_pairs(data_dir: str) -> list[TracePair]:
    """Discover complete RSRP/SINR pairs and reject incomplete trajectories."""
    discovered: dict[tuple[int, int], dict[str, str]] = {}
    for filename in os.listdir(data_dir):
        match = _TRACE_PATTERN.fullmatch(filename)
        if match is None:
            continue
        kind, speed, trajectory_id = match.groups()
        discovered.setdefault((int(speed), int(trajectory_id)), {})[kind] = filename

    incomplete = {
        key: sorted({"rsrp", "sinr"} - set(files))
        for key, files in discovered.items()
        if set(files) != {"rsrp", "sinr"}
    }
    if incomplete:
        raise ValueError(f"Incomplete radio trace pairs: {incomplete}")

    return [
        TracePair(speed, trajectory_id, files["rsrp"], files["sinr"])
        for (speed, trajectory_id), files in sorted(discovered.items())
    ]


def stratified_trajectory_split(
    pairs: list[TracePair],
    *,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 0,
) -> TraceSplits:
    """Split whole trajectories independently within each UE-speed stratum."""
    fractions = np.array(
        [train_fraction, validation_fraction, test_fraction],
        dtype=np.float64,
    )
    if np.any(fractions < 0) or not np.isclose(np.sum(fractions), 1.0):
        raise ValueError("Split fractions must be non-negative and sum to one.")
    if not pairs:
        raise ValueError("At least one trace pair is required.")

    rng = np.random.default_rng(seed)
    by_speed: dict[int, list[TracePair]] = {}
    for pair in pairs:
        by_speed.setdefault(pair.speed_kmh, []).append(pair)

    split_lists: list[list[TracePair]] = [[], [], []]
    for speed in sorted(by_speed):
        group = sorted(by_speed[speed])
        permutation = rng.permutation(len(group))
        shuffled = [group[index] for index in permutation]
        counts = _allocate_counts(len(group), fractions)

        offset = 0
        for destination, count in zip(split_lists, counts):
            destination.extend(shuffled[offset : offset + count])
            offset += count

    splits = TraceSplits(
        train=tuple(sorted(split_lists[0])),
        validation=tuple(sorted(split_lists[1])),
        test=tuple(sorted(split_lists[2])),
    )
    splits.validate()
    return splits


def _allocate_counts(n_items: int, fractions: np.ndarray) -> np.ndarray:
    raw = fractions * n_items
    counts = np.floor(raw).astype(int)
    remainder = n_items - int(np.sum(counts))
    order = np.argsort(-(raw - counts), kind="stable")
    for index in order[:remainder]:
        counts[index] += 1
    return counts
