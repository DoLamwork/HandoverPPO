param(
    [int]$Workers = 4,
    [string]$OutputDir = "data/uav_p0_canonical_v1_pilot"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$RunPy = Join-Path $ProjectRoot "run.py"
$ResolvedOutput = Join-Path $ProjectRoot $OutputDir

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python environment not found: $Python"
}
if ($Workers -le 0) {
    throw "Workers must be positive."
}

Write-Host "P0 canonical pilot"
Write-Host "  output: $ResolvedOutput"
Write-Host "  radio: 2.1 GHz, 10 MHz, dt=10 ms (project canonical defaults)"
Write-Host "  geometry: 2000 x 1667 m, altitude=100 m, road grid=100 m"
Write-Host "  routes: 20 per speed at 30/50/70/90/120/150 km/h"
Write-Host "  duration: 6000 samples = 60 s per route"

Push-Location $ProjectRoot
try {
    & $Python $RunPy generate_data `
        --output-dir $OutputDir `
        --profile p0_canonical_v1 `
        --speeds 30 50 70 90 120 150 `
        --routes-per-speed 20 `
        --start-route-id 0 `
        --steps 6000 `
        --altitude 100 `
        --grid-spacing 100 `
        --width 2000 `
        --height 1667 `
        --workers $Workers

    if ($LASTEXITCODE -ne 0) {
        throw "P0 generator exited with code $LASTEXITCODE."
    }
}
finally {
    Pop-Location
}

