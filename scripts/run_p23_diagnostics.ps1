param(
    [string]$DataDir = "data/uav_100m_grid100m_dev_processed",
    [string]$OutputRoot = "results/analysis/p23_diagnostics_reproduction",
    [switch]$RunGradientPilot
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$RunPy = Join-Path $RepoRoot "run.py"
$ModelRoot = Join-Path $RepoRoot "results\models\ablation_v2"
$Output = Join-Path $RepoRoot $OutputRoot

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Project Python was not found: $Python"
}
if (Test-Path -LiteralPath $Output) {
    throw "Refusing to overwrite an existing diagnostic output: $Output"
}

$Registry = Join-Path $Output "registry_v1"
$Bank = Join-Path $Output "state_bank_pilot_v1"
$Probe = Join-Path $Bank "probe_v1"
$Counterfactual = Join-Path $Bank "counterfactual_v1"

Push-Location $RepoRoot
try {
    & $Python $RunPy summarize_ablation `
        --input-dir results/models/ablation_v2 `
        --experiment-ids `
            E04_rolling_k3_s17 E05_monotonic_k3_s17 E06_qos_full_k3_s17 `
            E07_qos_rate_k3_s17 E08_qos_events_k3_s17 `
            E09_qos_recovery_k3_s17 E10_qos_dual_k3_s17 `
        --expected-budget 100000 `
        --output-dir $Registry
    if ($LASTEXITCODE -ne 0) { throw "D0 registry failed." }

    & $Python $RunPy audit_policy_learning build-bank `
        --data-dir $DataDir --speeds 30 50 70 90 `
        --routes-per-speed 5 --decision-indices 0 1 2 3 `
        --candidate-count 3 --candidate-order ranked_v1 `
        --observation-schemas rolling_v1 monotonic_v1 qos_full_v1 `
        --split-seed 17 --sample-seed 17 --output-dir $Bank
    if ($LASTEXITCODE -ne 0) { throw "D1 state-bank build failed." }

    $ProbeCheckpoints = @(
        "E04_latest=$(Join-Path $ModelRoot 'E04_rolling_k3_s17\smdp_ppo_latest.pt')",
        "E05_latest=$(Join-Path $ModelRoot 'E05_monotonic_k3_s17\smdp_ppo_latest.pt')",
        "E06_latest=$(Join-Path $ModelRoot 'E06_qos_full_k3_s17\smdp_ppo_latest.pt')",
        "E07_selected=$(Join-Path $ModelRoot 'E07_qos_rate_k3_s17\smdp_ppo_selected.pt')",
        "E07_latest=$(Join-Path $ModelRoot 'E07_qos_rate_k3_s17\smdp_ppo_latest.pt')",
        "E08_latest=$(Join-Path $ModelRoot 'E08_qos_events_k3_s17\smdp_ppo_latest.pt')",
        "E09_latest=$(Join-Path $ModelRoot 'E09_qos_recovery_k3_s17\smdp_ppo_latest.pt')",
        "E10_selected=$(Join-Path $ModelRoot 'E10_qos_dual_k3_s17\smdp_ppo_selected.pt')",
        "E10_latest=$(Join-Path $ModelRoot 'E10_qos_dual_k3_s17\smdp_ppo_latest.pt')"
    )
    $ProbeArgs = @(
        $RunPy, "audit_policy_learning", "probe", "--bank-dir", $Bank,
        "--include-fresh", "--seed", "17", "--output-dir", $Probe
    )
    foreach ($Checkpoint in $ProbeCheckpoints) {
        $ProbeArgs += @("--checkpoint", $Checkpoint)
    }
    & $Python @ProbeArgs
    if ($LASTEXITCODE -ne 0) { throw "D1 checkpoint probe failed." }

    $D2Labels = @("E07_selected", "E07_latest", "E08_latest", "E09_latest", "E10_selected", "E10_latest")
    $D2Args = @(
        $RunPy, "audit_policy_learning", "counterfactual", "--bank-dir", $Bank,
        "--horizons", "0.5", "1.0", "2.0", "--gamma-per-second", "0.99",
        "--output-dir", $Counterfactual
    )
    foreach ($Label in $D2Labels) {
        $Checkpoint = $ProbeCheckpoints | Where-Object { $_ -like "$Label=*" }
        $D2Args += @("--checkpoint", $Checkpoint)
    }
    & $Python @D2Args
    if ($LASTEXITCODE -ne 0) { throw "D2 counterfactual audit failed." }

    if ($RunGradientPilot) {
        $PilotOutput = Join-Path $RepoRoot "results\models\diagnostics_reproduction\D01_gradient_pilot_qos_rate_k3_s17"
        if (Test-Path -LiteralPath $PilotOutput) {
            throw "Refusing to overwrite gradient pilot output: $PilotOutput"
        }
        & $Python $RunPy train_ppo `
            --data-dir $DataDir --speeds 30 50 70 90 `
            --candidate-count 3 --candidate-order ranked_v1 `
            --observation-schema qos_full_v1 `
            --reward-schema trajectory_qos_reward_v1 `
            --experiment-id D01_gradient_pilot_qos_rate_k3_s17 `
            --parent-id E07_qos_rate_k3_s17 `
            --changed-factor learning_diagnostics_pilot `
            --total-macro-steps 2048 --rollout-steps 1024 `
            --batch-size 128 --update-epochs 4 --learning-rate 0.00005 `
            --evaluation-interval 2048 --learning-diagnostics `
            --seed 17 --split-seed 17 --skip-final-test `
            --output-dir $PilotOutput
        if ($LASTEXITCODE -ne 0) { throw "D1 gradient pilot failed." }
    }
}
finally {
    Pop-Location
}
