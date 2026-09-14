from pathlib import Path

import numpy as np
import pandas as pd

from uav_simulation.paths import DATA_DIR

class SimulationLogger:
    def __init__(self, sim):
        self.sim = sim

    def setup_log(self):
        self.sim.data_log = {'Step': []}
        for ue_idx in range(self.sim.num_ues):
            for bs_idx in range(len(self.sim.bs_positions)):
                self.sim.data_log[f'ue{ue_idx}_bs{bs_idx}_rsrp_dbm'] = []
                self.sim.data_log[f'ue{ue_idx}_bs{bs_idx}_sinr_db'] = []

    def log_step(self, frame):
        self.sim.data_log['Step'].append(frame)

    def log_all_bs_data(self, ue_idx, rsrp_dbm, sinr_db):
        num_bs = len(self.sim.bs_positions)
        if len(rsrp_dbm) != num_bs or len(sinr_db) != num_bs:
            raise ValueError("RSRP/SINR arrays must contain one value per ground BS.")

        for bs_idx in range(num_bs):
            self.sim.data_log[f'ue{ue_idx}_bs{bs_idx}_rsrp_dbm'].append(float(rsrp_dbm[bs_idx]))
            self.sim.data_log[f'ue{ue_idx}_bs{bs_idx}_sinr_db'].append(float(sinr_db[bs_idx]))

    def save_data_to_csv(self):
        data_dir = Path(getattr(self.sim, "output_data_dir", DATA_DIR))
        data_dir.mkdir(parents=True, exist_ok=True)
        max_len = max((len(v) for v in self.sim.data_log.values()), default=0)
        for k in self.sim.data_log:
            while len(self.sim.data_log[k]) < max_len:
                self.sim.data_log[k].append(None)

        def speed_tag(ue_idx):
            speed = float(self.sim.ue_speeds[ue_idx])
            if getattr(self.sim, "train_ready_output", False):
                if not speed.is_integer() or speed <= 0:
                    raise ValueError(
                        "Train-ready filenames require a positive integer speed in km/h."
                    )
                return str(int(speed))
            value = f"{speed:g}"
            return value.replace('.', 'p').replace('-', 'minus')

        def output_paths(run_idx):
            paths = []
            for ue_idx in range(self.sim.num_ues):
                tag = speed_tag(ue_idx)
                if getattr(self.sim, "train_ready_output", False):
                    paths.extend([
                        data_dir / f"rsrp_{tag}kmh_{run_idx}.csv",
                        data_dir / f"sinr_{tag}kmh_{run_idx}.csv",
                    ])
                else:
                    paths.extend([
                        data_dir / f"ue{ue_idx}_rsrp_{tag}kmh_{run_idx}.csv",
                        data_dir / f"ue{ue_idx}_sinr_{tag}kmh_{run_idx}.csv",
                    ])
            return paths

        configured_run_idx = getattr(self.sim, 'output_run_idx', None)
        if configured_run_idx is None:
            run_idx = 0
            paths = output_paths(run_idx)
            while any(path.exists() for path in paths):
                run_idx += 1
                paths = output_paths(run_idx)
        else:
            run_idx = int(configured_run_idx)
            paths = output_paths(run_idx)
            existing = [path for path in paths if path.exists()]
            if existing:
                raise FileExistsError(f"Refusing to overwrite output files: {existing}")

        saved_paths = []
        train_ready = getattr(self.sim, "train_ready_output", False)
        for ue_idx in range(self.sim.num_ues):
            tag = speed_tag(ue_idx)
            for metric, suffix in (('rsrp', 'rsrp_dbm'), ('sinr', 'sinr_db')):
                if train_ready:
                    samples = np.asarray(self.sim.data_log['Step'], dtype=np.int64)
                    delta_t_ms = self.sim.time_per_step * 1000.0
                    if float(delta_t_ms).is_integer():
                        time_ms = samples * int(delta_t_ms)
                    else:
                        time_ms = samples * delta_t_ms
                    output = {'sample': samples, 'time_ms': time_ms}
                else:
                    output = {'Step': self.sim.data_log['Step']}
                for bs_idx in range(len(self.sim.bs_positions)):
                    key = f'ue{ue_idx}_bs{bs_idx}_{suffix}'
                    column = f'bs_{bs_idx}' if train_ready else f'bs{bs_idx}'
                    output[column] = pd.to_numeric(
                        self.sim.data_log[key], errors='coerce'
                    ).round(2)

                if train_ready:
                    filepath = data_dir / f"{metric}_{tag}kmh_{run_idx}.csv"
                else:
                    filepath = data_dir / f"ue{ue_idx}_{metric}_{tag}kmh_{run_idx}.csv"
                pd.DataFrame(output).to_csv(
                    filepath,
                    index=False,
                    sep=',',
                    decimal='.',
                    encoding='utf-8' if train_ready else 'utf-8-sig',
                    float_format='%.2f',
                )
                saved_paths.append(filepath)
                print(f"Saved {metric.upper()} for UAV-UE {ue_idx}: {filepath}")

        return saved_paths
