param(
    [string]$OutputDir = "results/analysis/early_ho_timing/rpost_rearly_pilot_v3"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Root ".venv\Scripts\python.exe"

if (Test-Path -LiteralPath (Join-Path $Root $OutputDir)) {
    throw "Refusing to overwrite existing timing-reference output: $OutputDir"
}

& $Python (Join-Path $Root "run.py") audit_early_timing_references `
    --data-dir (Join-Path $Root "data/uav_100m_grid100m_dev_processed") `
    --baseline-manifest (Join-Path $Root "results/analysis/early_ho_timing/b0_b0a_pilot_v2/run_manifest.json") `
    --delays-ms 0 20 40 80 160 320 `
    --reference-horizon 1.0 `
    --output-dir (Join-Path $Root $OutputDir)

exit $LASTEXITCODE
