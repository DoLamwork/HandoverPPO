param(
    [string]$ProjectRoot = "C:\Project\HandoverOptimDRL",
    [string]$OutputDir = "results/analysis/early_ho_timing/t6a_seed17_vs_seed29_v1"
)

$ErrorActionPreference = "Stop"
$absoluteOutput = Join-Path $ProjectRoot $OutputDir
if (Test-Path -LiteralPath $absoluteOutput) {
    throw "Refusing to overwrite existing T6A divergence audit: $absoluteOutput"
}

$python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    throw "Python environment not found: $python"
}

Push-Location $ProjectRoot
try {
    & $python run.py audit_t6a_seed_divergence `
        --data-dir data/uav_100m_grid100m_dev_processed `
        --route-split results/analysis/early_ho_timing/t2_bank_main_v1/route_split.json `
        --seed17-checkpoint results/models/early_ho_timing/T6A_a3_entry_timing_b100000_s17/smdp_ppo_selected.pt `
        --seed29-checkpoint results/models/early_ho_timing/T6A_a3_entry_timing_b100000_s29/smdp_ppo_selected.pt `
        --seed17-metrics results/models/early_ho_timing/T6A_a3_entry_timing_b100000_s17/training_metrics.jsonl `
        --seed29-metrics results/models/early_ho_timing/T6A_a3_entry_timing_b100000_s29/training_metrics.jsonl `
        --bootstrap-resamples 2000 `
        --bootstrap-seed 83 `
        --output-dir $OutputDir
    if ($LASTEXITCODE -ne 0) {
        throw "T6A divergence audit failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
