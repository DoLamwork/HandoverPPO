$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Root ".venv\Scripts\python.exe"

& $Python (Join-Path $Root "run.py") probe_d63_2 `
    --labels-dir results/analysis/d63_2_labels_v1 `
    --seed 29 `
    --output-dir results/analysis/d63_2_probe_v1

if ($LASTEXITCODE -ne 0) {
    throw "D63-2 probe failed with exit code $LASTEXITCODE"
}
