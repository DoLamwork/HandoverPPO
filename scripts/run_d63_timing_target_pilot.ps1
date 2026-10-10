$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$output = Join-Path $projectRoot "results\analysis\d63_timing_target_pilot_v2"
$d61 = Join-Path $projectRoot "results\models\d6_pilots\D61_a3_redecision_pilot_b10240_s17\smdp_ppo_selected.pt"
$d62 = Join-Path $projectRoot "results\models\d6_pilots\D62_hierarchical_pilot_b10240_s17\smdp_ppo_selected.pt"

if (Test-Path -LiteralPath $output) {
    $items = @(Get-ChildItem -LiteralPath $output -Force)
    if ($items.Count -gt 0) {
        throw "Refusing to overwrite existing D63 pilot output: $output"
    }
}

& $python (Join-Path $projectRoot "run.py") audit_timing_target `
    --data-dir (Join-Path $projectRoot "data\uav_100m_grid100m_dev_processed") `
    --d61-checkpoint $d61 `
    --d62-checkpoint $d62 `
    --speeds 30 50 70 90 `
    --split-seed 17 `
    --sample-seed 17 `
    --routes-per-speed 5 `
    --max-anchors-per-source-route 5 `
    --delays-ms 0 100 200 300 `
    --horizons 0.5 1.0 2.0 `
    --gamma-per-second 0.99 `
    --output-dir $output

exit $LASTEXITCODE
