param(
    [int]$Workers = 4,
    [string]$OutputDir = "data/uav_p4_combined_harsh_v1_pilot"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$RunPy = Join-Path $ProjectRoot "run.py"

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python environment not found: $Python"
}
if ($Workers -le 0) { throw "Workers must be positive." }

Write-Host "P4 combined-harsh pilot: dense urban + high mobility + fast channel"
Push-Location $ProjectRoot
try {
    & $Python $RunPy generate_data `
        --output-dir $OutputDir `
        --profile p4_combined_harsh_v1 `
        --speeds 90 120 150 180 210 240 `
        --routes-per-speed 20 `
        --start-route-id 0 `
        --steps 6000 `
        --altitude 100 `
        --grid-spacing 100 `
        --width 2000 `
        --height 1667 `
        --workers $Workers
    if ($LASTEXITCODE -ne 0) { throw "P4 generator exited with code $LASTEXITCODE." }
}
finally { Pop-Location }
