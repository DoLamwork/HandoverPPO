# UAV simulation as a HandoverOptimDRL component

The authoritative project is `C:\Project\HandoverOptimDRL`. UAV simulation is
contained in its `data_generator/` folder; PPO code remains in `src/ho_optim_drl`.
The root `run.py` exposes `generate_data`, `prepare_data`, `train_ppo`,
`evaluate_ppo`, `benchmark`, and `validate_3gpp`.

The generator is strictly a radio-trace producer. It emits per-cell RSRP and
SINR without selecting a serving cell or implementing connection, A3/TTT,
handover, HOF, or RLF state. All such protocol behavior belongs exclusively to
`src/ho_optim_drl/`.

## Source and output locations

| Component | Location |
| --- | --- |
| Simulation source | `data_generator/src/uav_simulation/` |
| Simulation docs and tests | `data_generator/docs/`, `data_generator/tests/` |
| Handover/PPO source and installed commands | `src/ho_optim_drl/` |
| Compatibility script names | `scripts/` |
| Existing legacy UAV CSV pairs | `data/UAV_data_100m/`, `data/UAV_data_300m/`, `data/UAV_data_100m_grid100m/` |
| New train-ready generated data | user-selected `--output-dir`, for example `data/uav_100m_grid100m_v3/` |
| New checkpoints | `results/models/uav_generated_ppo/` |
| New benchmark/reference outputs | `results/benchmark/uav_generated_ppo/`, `results/3gpp/uav_generated/` |

`setup.py` installs both source locations as part of the existing `handover-ppo`
distribution. This is one project, with no separate trainer inside the generator.
Existing handover datasets, model files, and reports retain their original paths.

Batch generation now writes the PPO contract directly: filenames
`{rsrp|sinr}_{speed}kmh_{trajectory}.csv` and columns
`sample,time_ms,bs_0,...`. Generator and handover environment share the defaults
2.1 GHz, 10 MHz, and 10 ms from `ho_optim_drl.config`. `train_ppo --data-dir` may
therefore point at the completed generator output with no conversion step.

All batch generators write a schema-v3
`simulation_manifest.json`. It identifies the Python entrypoint, all generation
parameters, route seed rule, output contract, radio/simulation configuration,
installed distribution version, a source-tree SHA-256, and per-source-file hashes.
After all expected trajectories pass schema and row-count validation, the generator
writes `dataset_metadata.json` with that manifest and SHA-256 for each output CSV.

The current unmanifested `data/UAV_data_100m_grid100m` directory is historical and
is not backfilled: doing so would claim that today's code/config generated old CSVs.
Generate the reproducible train-ready replacement in a new directory such as
`data/uav_100m_grid100m_v3`. `prepare_data` remains only as the compatibility path
for confirmed legacy traces named `ue0_*` with columns `Step,bs0,...`.

A checkpoint stores the protocol config and a hashed split manifest. Held-out
evaluation uses that exact manifest. Keep it next to the checkpoint after moving
model files. Legacy data without metadata retain the original handover defaults.

## Migration safety

The earlier combination in `C:\Project\UAV simulation` is retained as a source
copy. The active workflow is now HandoverOptimDRL; its scripts have no runtime
reference to the old project directory. Existing raw files already in the parent
are reused after SHA-256 comparison. The previously prepared dataset is copied
into the new `data/uav_generated` folder so metadata does not alter old experiments.

Files changed in the principal project are backed up under
`results/integration/uav_component/before/`. The migration manifest records each
changed file and data transfer. The old Git repositories are not nested or merged,
and the parent's existing virtual environment is reused.
