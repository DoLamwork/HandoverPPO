param(
    [string]$T2Output = "results/analysis/early_ho_timing/t2_bank_main_v1",
    [string]$T3Output = "results/analysis/early_ho_timing/t3_route_main_v1"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$StatusDir = Join-Path $Root "results/automation/early_ho_t2_t3"
$StatusFile = Join-Path $StatusDir "runner_status.jsonl"
New-Item -ItemType Directory -Force -Path $StatusDir | Out-Null

foreach ($relative in @($T2Output, $T3Output)) {
    if (Test-Path -LiteralPath (Join-Path $Root $relative)) {
        throw "Refusing to overwrite existing T2/T3 output: $relative"
    }
}

function Write-Status([string]$Phase, [string]$Status, [int]$ExitCode = 0) {
    $record = [ordered]@{
        timestamp = (Get-Date).ToUniversalTime().ToString("o")
        phase = $Phase
        status = $Status
        exit_code = $ExitCode
    }
    Add-Content -LiteralPath $StatusFile -Value ($record | ConvertTo-Json -Compress)
}

Write-Status "T2" "STARTED"
& $Python (Join-Path $Root "run.py") audit_early_timing_bank `
    --data-dir (Join-Path $Root "data/uav_100m_grid100m_dev_processed") `
    --speeds 30 50 70 90 `
    --split-seed 17 `
    --sample-seed 17 `
    --routes-per-speed 20 `
    --max-anchors-per-source-route 5 `
    --delays-ms 0 20 40 80 160 320 `
    --horizons 0.5 1.0 2.0 `
    --output-dir (Join-Path $Root $T2Output)
$Code = $LASTEXITCODE
if ($Code -ne 0) {
    Write-Status "T2" "FAILED" $Code
    exit $Code
}
Write-Status "T2" "COMPLETE"

Write-Status "T3" "STARTED"
& $Python (Join-Path $Root "run.py") audit_early_timing_routes `
    --data-dir (Join-Path $Root "data/uav_100m_grid100m_dev_processed") `
    --route-split (Join-Path $Root "$T2Output/route_split.json") `
    --delays-ms 0 20 40 80 160 320 `
    --reference-horizon 1.0 `
    --bootstrap-resamples 2000 `
    --bootstrap-seed 29 `
    --output-dir (Join-Path $Root $T3Output)
$Code = $LASTEXITCODE
if ($Code -ne 0) {
    Write-Status "T3" "FAILED" $Code
    exit $Code
}
Write-Status "T3" "COMPLETE"
exit 0
