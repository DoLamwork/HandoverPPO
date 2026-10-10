$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Root ".venv\Scripts\python.exe"

& $Python (Join-Path $Root "run.py") prepare_d63_2_labels `
    --bank-dir results/analysis/d63_timing_target_main_local_v1 `
    --main-audit-dir results/analysis/d63_timing_target_main_analysis_v1 `
    --route-split-seed 29 `
    --output-dir results/analysis/d63_2_labels_v1

if ($LASTEXITCODE -ne 0) {
    throw "D63-2 label preparation failed with exit code $LASTEXITCODE"
}
