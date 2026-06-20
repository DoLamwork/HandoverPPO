# Trace data

The runtime reads the RSRP and SINR traces in `processed` from CSV files.
Each file has the following row-oriented schema:

```text
sample,time_ms,bs_0,bs_1,bs_2,bs_3,bs_4
```

- `sample` is a zero-based, contiguous sample index.
- `time_ms` is the elapsed trace time using the configured 10 ms interval.
- `bs_N` is the RSRP in dBm or SINR in dB for base station `N`.

The processed CSV files are versioned so training works after a normal Git
clone. MATLAB source files are excluded from Git to keep the repository
compact. If they are available separately, conversion can be reproduced with:

```bash
python scripts/convert_mat_to_csv.py \
  --input-dir path/to/mat/traces \
  --output-dir data/processed \
  --overwrite
```

The loader validates matching shapes, finite values, and the native 10 ms
sampling interval. It returns a `RadioTrace` with three explicit
`(base_station, timestep)` streams:

- `rsrp_a3_db`: L3-filtered RSRP used only by A3 event detection.
- `sinr_rate_db`: raw SINR used for spectral efficiency and estimated rate.
- `sinr_rlm_db`: SINR used by Qin/Qout, N310/T310, and recovery.

In version 1, `sinr_rlm_db` is the same raw SINR data as `sinr_rate_db`. No
resampling or SINR filtering is applied, so protocol timers keep their native
sample timing and rate peaks are not shifted by preprocessing.

Train, validation, and test sets are split by complete trajectory and
stratified by UE speed:

```python
from ho_optim_drl.data import (
    discover_trace_pairs,
    stratified_trajectory_split,
)

pairs = discover_trace_pairs("data/processed")
splits = stratified_trajectory_split(pairs, seed=17)
```

This prevents samples from the same trajectory leaking across datasets.
