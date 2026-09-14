from types import SimpleNamespace

import numpy as np

from uav_simulation import config
from uav_simulation.mobility import GridMobility


def route_seed(speed_kmh: float, route_id: int) -> int:
    """Return the seed convention used by both batch generators."""
    speed_kmh = float(speed_kmh)
    route_id = int(route_id)
    if not np.isfinite(speed_kmh) or speed_kmh < 0.0:
        raise ValueError('Speed must be a non-negative finite number.')
    if route_id < 0:
        raise ValueError('Route ID must be a non-negative integer.')
    return int(speed_kmh * 1000.0 + route_id)


def reconstruct_seeded_grid_route(
    *,
    speed_kmh: float,
    route_id: int,
    x_lines: np.ndarray,
    y_lines: np.ndarray,
    rect_xmin: float,
    rect_xmax: float,
    rect_ymin: float,
    rect_ymax: float,
    steps: int = 6000,
    time_per_step_s: float = config.DELTA_T_MS / 1000.0,
) -> tuple[np.ndarray, int]:
    """Recreate a batch route using mobility only, with no radio evaluation."""
    seed = route_seed(speed_kmh, route_id)
    steps = int(steps)
    time_per_step_s = float(time_per_step_s)
    if steps < 0:
        raise ValueError('Steps must be non-negative.')
    if not np.isfinite(time_per_step_s) or time_per_step_s <= 0.0:
        raise ValueError('time_per_step_s must be positive and finite.')

    # RandomState reproduces np.random.seed(...)+np.random.choice(...) exactly,
    # while remaining isolated from any live simulation RNG state.
    rng = np.random.RandomState(seed)
    replay = SimpleNamespace(
        x_lines=np.asarray(x_lines, dtype=float),
        y_lines=np.asarray(y_lines, dtype=float),
        rect_xmin=float(rect_xmin),
        rect_xmax=float(rect_xmax),
        rect_ymin=float(rect_ymin),
        rect_ymax=float(rect_ymax),
        ue_positions=[],
        ue_dir=[],
        mobility_rng=rng,
    )
    mobility = GridMobility(replay)

    x = float(rng.choice(replay.x_lines))
    y = float(rng.choice(replay.y_lines))
    replay.ue_positions.append([x, y])
    allowed = mobility.allowed_dirs_at_intersection(x, y)
    replay.ue_dir.append(
        int(rng.choice(allowed)) if allowed else int(rng.randint(0, 4))
    )

    points = np.empty((steps + 1, 2), dtype=float)
    points[0] = replay.ue_positions[0]
    distance_per_step_m = (
        float(speed_kmh) * (1000.0 / 3600.0) * time_per_step_s
    )
    for step in range(steps):
        mobility.step_manhattan_on_grid(0, distance_per_step_m)
        points[step + 1] = replay.ue_positions[0]

    return points, seed
