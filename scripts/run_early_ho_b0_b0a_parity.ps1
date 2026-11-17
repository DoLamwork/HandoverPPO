param(
    [string]$OutputDir = "results/analysis/early_ho_timing/b0_b0a_pilot_v2"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv/Scripts/python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Project Python was not found: $Python"
}

Push-Location $ProjectRoot
try {
    & $Python run.py audit_baseline_parity `
        --data-dir data/uav_100m_grid100m_dev_processed `
        --speeds 30 50 70 90 `
        --split-seed 17 `
        --sample-seed 17 `
        --routes-per-speed 2 `
        --output-dir $OutputDir
    if ($LASTEXITCODE -ne 0) {
        throw "B0/B0A parity audit failed with exit code $LASTEXITCODE."
    }
}
finally {
    Pop-Location
}
