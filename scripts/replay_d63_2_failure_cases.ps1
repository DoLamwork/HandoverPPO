$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Root ".venv\Scripts\python.exe"

& $Python (Join-Path $Root "run.py") replay_d63_2_failure_cases `
    --bank-dir results/analysis/d63_timing_target_main_local_v1 `
    --labels-dir results/analysis/d63_2_labels_v1 `
    --probe-dir results/analysis/d63_2_probe_v1 `
    --output-dir results/analysis/d63_2_failure_case_replay_v1

if ($LASTEXITCODE -ne 0) {
    throw "D63-2 failure-case replay failed with exit code $LASTEXITCODE"
}
