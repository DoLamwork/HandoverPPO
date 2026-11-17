param(
    [string]$OutputDir = "results/analysis/early_ho_timing/b2_b3_controls_pilot_v1"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv/Scripts/python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Project Python was not found: $Python"
}

Push-Location $ProjectRoot
try {
    & $Python run.py audit_b2_b3_early_controls `
        --data-dir data/uav_100m_grid100m_dev_processed `
        --baseline-manifest results/analysis/early_ho_timing/b0_b0a_pilot_v2/run_manifest.json `
        --output-dir $OutputDir
    if ($LASTEXITCODE -ne 0) {
        throw "B2/B3 pilot failed with exit code $LASTEXITCODE."
    }
}
finally {
    Pop-Location
}
