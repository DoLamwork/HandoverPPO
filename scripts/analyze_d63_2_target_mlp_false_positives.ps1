$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Root ".venv\Scripts\python.exe"

& $Python (Join-Path $Root "run.py") analyze_d63_2_false_positives `
    --labels-dir results/analysis/d63_2_labels_v1 `
    --probe-dir results/analysis/d63_2_probe_v1 `
    --output-dir results/analysis/d63_2_target_mlp_fp_v1

if ($LASTEXITCODE -ne 0) {
    throw "D63-2 target-MLP false-positive analysis failed with exit code $LASTEXITCODE"
}
