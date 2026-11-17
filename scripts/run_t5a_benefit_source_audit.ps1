param(
    [string]$ProjectRoot = "C:\Project\HandoverOptimDRL",
    [string]$OutputDir = "results/analysis/early_ho_timing/t5a_benefit_source_main_v1"
)

$ErrorActionPreference = "Stop"
$absoluteOutput = Join-Path $ProjectRoot $OutputDir
if (Test-Path -LiteralPath $absoluteOutput) {
    throw "Refusing to overwrite existing T5A benefit-source audit: $absoluteOutput"
}

$python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    throw "Python environment not found: $python"
}

Push-Location $ProjectRoot
try {
    & $python run.py audit_t5a_benefit_sources `
        --data-dir data/uav_100m_grid100m_dev_processed `
        --route-split results/analysis/early_ho_timing/t2_bank_main_v1/route_split.json `
        --t2-dir results/analysis/early_ho_timing/t2_bank_main_v1 `
        --t3-dir results/analysis/early_ho_timing/t3_route_main_v1 `
        --checkpoints `
            results/models/early_ho_timing/T5A_timing_ppo_fixed_scale_b100000_s17/smdp_ppo_selected.pt `
            results/models/early_ho_timing/T5A_timing_ppo_fixed_scale_b100000_s29/smdp_ppo_selected.pt `
            results/models/early_ho_timing/T5A_timing_ppo_fixed_scale_b100000_s43/smdp_ppo_selected.pt `
        --horizon 1.0 `
        --bootstrap-resamples 2000 `
        --bootstrap-seed 71 `
        --output-dir $OutputDir
    if ($LASTEXITCODE -ne 0) {
        throw "T5A benefit-source audit failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
