param(
    [string]$Python = ".\.venv\Scripts\python.exe",
    [string]$OutputDir = "results\analysis\d63_3_execution_safety_v1"
)

$Root = Split-Path -Parent $PSScriptRoot
& $Python (Join-Path $Root "run.py") validate_d63_3_feature `
    --bank-dir results/analysis/d63_timing_target_main_local_v1 `
    --labels-dir results/analysis/d63_2_labels_v1 `
    --output-dir $OutputDir
if ($LASTEXITCODE -ne 0) {
    throw "D63-3 feature validation failed with exit code $LASTEXITCODE"
}
