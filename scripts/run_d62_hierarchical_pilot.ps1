param(
    [string]$DataDir = "data/uav_100m_grid100m_dev_processed",
    [string]$OutputRoot = "results/models/d6_pilots",
    [string]$AnalysisRoot = "results/analysis/p23_diagnostics/d62_pilot_v1",
    [string]$StateBank = "results/analysis/p23_diagnostics/state_bank_train_v2",
    [int]$Seed = 17
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$RunPy = Join-Path $RepoRoot "run.py"
$ExperimentId = "D62_hierarchical_pilot_b10240_s$Seed"
$PilotOutput = Join-Path (Join-Path $RepoRoot $OutputRoot) $ExperimentId
$AnalysisOutput = Join-Path $RepoRoot $AnalysisRoot
$ProbeOutput = Join-Path $AnalysisOutput "fixed_bank_probe"
$TransitionOutput = Join-Path $AnalysisOutput "transition_signal"
$ResolvedBank = Join-Path $RepoRoot $StateBank
$D61 = Join-Path $RepoRoot "results/models/d6_pilots/D61_a3_redecision_pilot_b10240_s17/smdp_ppo_selected.pt"

foreach ($Required in @($Python, $RunPy, $ResolvedBank, $D61)) {
    if (-not (Test-Path -LiteralPath $Required)) {
        throw "Required D62 input was not found: $Required"
    }
}

function Test-CompletedPilot {
    $ReportPath = Join-Path $PilotOutput "training_report.json"
    if (-not (Test-Path -LiteralPath $ReportPath -PathType Leaf)) {
        return $false
    }
    try {
        $Report = Get-Content -LiteralPath $ReportPath -Raw | ConvertFrom-Json
        return (
            $Report.experiment_config.config.experiment_id -eq $ExperimentId -and
            [int]$Report.experiment_config.config.budget.total_macro_steps -eq 10240 -and
            $Report.experiment_config.config.model.architecture -eq "hierarchical_candidate_scorer_v1" -and
            $Report.experiment_config.config.stay_mode -eq "a3_redecision_v1" -and
            [int]$Report.experiment_config.config.stay_redecision_ms -eq 100 -and
            $Report.test_evaluated -eq $false -and
            (Test-Path -LiteralPath $Report.checkpoints.selected_export -PathType Leaf)
        )
    }
    catch {
        return $false
    }
}

Push-Location $RepoRoot
try {
    if (-not (Test-CompletedPilot)) {
        if (Test-Path -LiteralPath $PilotOutput) {
            $Existing = @(Get-ChildItem -LiteralPath $PilotOutput -Force)
            if ($Existing.Count -gt 0) {
                throw "Refusing to overwrite incomplete D62 pilot: $PilotOutput"
            }
        }
        & $Python $RunPy train_ppo `
            --data-dir $DataDir `
            --speeds 30 50 70 90 `
            --candidate-count 3 `
            --candidate-order ranked_v1 `
            --observation-schema qos_full_v1 `
            --reward-schema trajectory_qos_reward_v1 `
            --network-architecture hierarchical_candidate_scorer_v1 `
            --stay-mode a3_redecision_v1 `
            --stay-redecision-ms 100 `
            --total-macro-steps 10240 `
            --rollout-steps 1024 `
            --batch-size 128 `
            --update-epochs 4 `
            --learning-rate 0.00005 `
            --evaluation-interval 10240 `
            --learning-diagnostics `
            --seed $Seed `
            --split-seed $Seed `
            --skip-final-test `
            --experiment-id $ExperimentId `
            --parent-id D61_a3_redecision_pilot_b10240_s17 `
            --changed-factor flat_to_hierarchical_actor `
            --output-dir $PilotOutput
        if ($LASTEXITCODE -ne 0) { throw "D62 pilot failed." }
    }
    else {
        Write-Host "Skipping completed D62 pilot: $ExperimentId"
    }

    $D62 = Join-Path $PilotOutput "smdp_ppo_selected.pt"
    if (-not (Test-Path -LiteralPath $ProbeOutput)) {
        & $Python $RunPy audit_policy_learning probe `
            --bank-dir $ResolvedBank `
            --checkpoint "D61=$D61" `
            --checkpoint "D62=$D62" `
            --output-dir $ProbeOutput
        if ($LASTEXITCODE -ne 0) { throw "D62 fixed-bank probe failed." }
    }
    if (-not (Test-Path -LiteralPath $TransitionOutput)) {
        & $Python $RunPy audit_policy_learning transition-probe `
            --data-dir $DataDir `
            --speeds 30 50 70 90 `
            --checkpoint "D62=$D62" `
            --include-fresh `
            --rollout-probe-steps 1024 `
            --transition-horizon 1.0 `
            --split-seed $Seed `
            --output-dir $TransitionOutput
        if ($LASTEXITCODE -ne 0) { throw "D62 transition audit failed." }
    }
}
finally {
    Pop-Location
}
