"""Convert the Vienna RSRP/SINR MATLAB traces to row-oriented CSV files."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import scipy.io


def _load_single_matrix(mat_path: Path) -> np.ndarray:
    mat = scipy.io.loadmat(mat_path, squeeze_me=True)
    variables = {
        key: np.asarray(value)
        for key, value in mat.items()
        if not key.startswith("__")
    }
    if len(variables) != 1:
        raise ValueError(
            f"{mat_path} must contain exactly one data variable; "
            f"found {list(variables)}"
        )

    matrix = next(iter(variables.values()))
    if matrix.ndim != 2:
        raise ValueError(f"{mat_path} must contain a 2-D matrix, got {matrix.shape}")
    return matrix.astype(np.float64, copy=False)


def convert_file(
    mat_path: Path,
    output_dir: Path,
    delta_t_ms: int,
    overwrite: bool,
) -> Path:
    """Convert one BS x timestep MATLAB matrix to a timestep x column CSV."""
    output_path = output_dir / f"{mat_path.stem}.csv"
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"{output_path} already exists. Pass --overwrite to replace it."
        )

    matrix = _load_single_matrix(mat_path)
    n_base_stations, n_samples = matrix.shape
    sample = np.arange(n_samples, dtype=np.int64)
    time_ms = sample * delta_t_ms
    csv_data = np.column_stack((sample, time_ms, matrix.T))
    header = ",".join(
        ["sample", "time_ms"]
        + [f"bs_{index}" for index in range(n_base_stations)]
    )
    formats = ["%d", "%d"] + ["%.17g"] * n_base_stations

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        output_path,
        csv_data,
        delimiter=",",
        header=header,
        comments="",
        fmt=formats,
    )
    return output_path


def convert_directory(
    input_dir: Path,
    output_dir: Path,
    delta_t_ms: int,
    overwrite: bool,
) -> list[Path]:
    """Convert all MATLAB traces in a directory."""
    mat_paths = sorted(input_dir.glob("*.mat"))
    if not mat_paths:
        raise FileNotFoundError(f"No .mat files found in {input_dir}")

    outputs = []
    for index, mat_path in enumerate(mat_paths, start=1):
        output = convert_file(mat_path, output_dir, delta_t_ms, overwrite)
        outputs.append(output)
        print(f"[{index:3d}/{len(mat_paths)}] {mat_path.name} -> {output.name}")
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory containing the MATLAB traces.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/from_mat"),
        help="Directory in which CSV traces are written.",
    )
    parser.add_argument(
        "--delta-t-ms",
        type=int,
        default=10,
        help="Sampling interval used to populate the time_ms column.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace CSV files that already exist.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    outputs = convert_directory(
        args.input_dir,
        args.output_dir,
        args.delta_t_ms,
        args.overwrite,
    )
    print(f"Converted {len(outputs)} MATLAB files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
