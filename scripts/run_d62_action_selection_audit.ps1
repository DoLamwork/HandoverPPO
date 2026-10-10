param(
    [string]$DataDir = "data/uav_100m_grid100m_dev_processed",
    [string]$Checkpoint = "results/models/d6_pilots/D62_hierarchical_pilot_b10240_s17/smdp_ppo_selected.pt",
    [string]$StateBank = "results/analysis/p23_diagnostics/state_bank_train_v2",
    [string]$Counterfactual = "results/analysis/p23_diagnostics/counterfactual_train_horizons_v1",
    [string]$OutputDir = "results/analysis/p23_diagnostics/d62_action_selection_audit_v1"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$RunPy = Join-Path $RepoRoot "run.py"
$ResolvedOutput = Join-Path $RepoRoot $OutputDir

foreach ($Required in @($Python, $RunPy, (Join-Path $RepoRoot $Checkpoint), (Join-Path $RepoRoot $StateBank), (Join-Path $RepoRoot $Counterfactual))) {
    if (-not (Test-Path -LiteralPath $Required)) {
        throw "Required action-selection audit input was not found: $Required"
    }
}
if (Test-Path -LiteralPath $ResolvedOutput) {
    throw "Refusing to overwrite action-selection audit output: $ResolvedOutput"
}

Push-Location $RepoRoot
try {
    & $Python $RunPy audit_policy_learning action-selection `
        --data-dir $DataDir `
        --bank-dir $StateBank `
        --counterfactual-dir $Counterfactual `
        --checkpoint "D62=$Checkpoint" `
        --speeds 30 50 70 90 `
        --split-seed 17 `
        --stochastic-seeds 17 29 43 `
        --selection-horizon 1.0 `
        --output-dir $OutputDir
    if ($LASTEXITCODE -ne 0) {
        throw "D62 action-selection audit failed."
    }
}
finally {
    Pop-Location
}
