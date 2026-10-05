param(
    [string]$DataDir = "data/uav_100m_grid100m_dev_processed",
    [string]$OutputRoot = "results/models/d6_pilots_repro",
    [string]$AnalysisRoot = "results/analysis/p23_diagnostics/d6_repro",
    [string]$StateBank = "results/analysis/p23_diagnostics/state_bank_train_v2",
    [string]$ControlCheckpoint = "results/models/d3_pilots/D30_e07_control_pilot_b10240_s17/smdp_ppo_selected.pt",
    [int]$Seed = 17
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$RunPy = Join-Path $RepoRoot "run.py"
$ExperimentId = "D60_candidate_scorer_pilot_b10240_repro_s$Seed"
$PilotOutput = Join-Path (Join-Path $RepoRoot $OutputRoot) $ExperimentId
$ResolvedBank = Join-Path $RepoRoot $StateBank
$ResolvedControl = Join-Path $RepoRoot $ControlCheckpoint
$ResolvedAnalysis = Join-Path $RepoRoot $AnalysisRoot
$ProbeOutput = Join-Path $ResolvedAnalysis "probe"
$CounterfactualOutput = Join-Path $ResolvedAnalysis "counterfactual"
$TransitionOutput = Join-Path $ResolvedAnalysis "transition_signal"

foreach ($RequiredFile in @($Python, $RunPy, $ResolvedControl)) {
    if (-not (Test-Path -LiteralPath $RequiredFile -PathType Leaf)) {
        throw "Required file was not found: $RequiredFile"
    }
}
if (-not (Test-Path -LiteralPath $ResolvedBank -PathType Container)) {
    throw "TRAIN state bank was not found: $ResolvedBank"
}
foreach ($Output in @($PilotOutput, $ProbeOutput, $CounterfactualOutput, $TransitionOutput)) {
    if (Test-Path -LiteralPath $Output) {
        throw "Refusing to overwrite D6 reproduction output: $Output"
    }
}

Push-Location $RepoRoot
try {
    & $Python $RunPy train_ppo `
        --data-dir $DataDir `
        --speeds 30 50 70 90 `
        --candidate-count 3 `
        --candidate-order ranked_v1 `
        --observation-schema qos_full_v1 `
        --reward-schema trajectory_qos_reward_v1 `
        --network-architecture candidate_scorer_v1 `
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
        --parent-id D30_e07_control_pilot_b10240_s17 `
        --changed-factor network_architecture_shared_to_candidate_scorer `
        --output-dir $PilotOutput
    if ($LASTEXITCODE -ne 0) { throw "D6 pilot failed." }

    $CandidateCheckpoint = Join-Path $PilotOutput "smdp_ppo_selected.pt"
    & $Python $RunPy audit_policy_learning probe `
        --bank-dir $ResolvedBank `
        --checkpoint "D30=$ResolvedControl" `
        --checkpoint "D60=$CandidateCheckpoint" `
        --output-dir $ProbeOutput
    if ($LASTEXITCODE -ne 0) { throw "D6 fixed-bank probe failed." }

    & $Python $RunPy audit_policy_learning counterfactual `
        --bank-dir $ResolvedBank `
        --checkpoint "D30=$ResolvedControl" `
        --checkpoint "D60=$CandidateCheckpoint" `
        --horizons 1.0 `
        --output-dir $CounterfactualOutput
    if ($LASTEXITCODE -ne 0) { throw "D6 counterfactual audit failed." }

    & $Python $RunPy audit_policy_learning transition-probe `
        --data-dir $DataDir `
        --speeds 30 50 70 90 `
        --checkpoint "D30=$ResolvedControl" `
        --checkpoint "D60=$CandidateCheckpoint" `
        --include-fresh `
        --rollout-probe-steps 1024 `
        --transition-horizon 1.0 `
        --split-seed $Seed `
        --output-dir $TransitionOutput
    if ($LASTEXITCODE -ne 0) { throw "D6 transition-signal audit failed." }
}
finally {
    Pop-Location
}
