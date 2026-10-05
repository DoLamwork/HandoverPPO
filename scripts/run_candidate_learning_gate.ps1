param(
    [string]$Tag = "train_v2",
    [string]$Checkpoint = "results/models/d3_screening/D30_e07_control_b100k_s17/smdp_ppo_selected.pt"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$Bank = Join-Path $Root "results\analysis\p23_diagnostics\state_bank_$Tag"
$Counterfactual = Join-Path $Root "results\analysis\p23_diagnostics\counterfactual_$Tag"
$Probe = Join-Path $Root "results\analysis\p23_diagnostics\supervised_probe_$Tag"

foreach ($Path in @($Bank, $Counterfactual, $Probe)) {
    if (Test-Path -LiteralPath $Path) {
        throw "Refusing to overwrite diagnostic output: $Path"
    }
}

Push-Location $Root
try {
    & $Python run.py audit_policy_learning build-bank `
        --data-dir data/uav_100m_grid100m_dev_processed `
        --speeds 30 50 70 90 `
        --routes-per-speed 20 `
        --decision-indices (0..19) `
        --candidate-count 3 `
        --candidate-order ranked_v1 `
        --observation-schemas qos_full_v1 `
        --source-checkpoint "D30_selected=$Checkpoint" `
        --split-seed 17 `
        --sample-seed 17 `
        --output-dir $Bank
    if ($LASTEXITCODE -ne 0) { throw "State-bank build failed." }

    & $Python run.py audit_policy_learning counterfactual `
        --bank-dir $Bank `
        --checkpoint "D30_selected=$Checkpoint" `
        --horizons 1.0 `
        --output-dir $Counterfactual
    if ($LASTEXITCODE -ne 0) { throw "Counterfactual audit failed." }

    & $Python run.py audit_policy_learning supervised-probe `
        --bank-dir $Bank `
        --counterfactual-dir $Counterfactual `
        --probe-observation-schema qos_full_v1 `
        --probe-reward-schema trajectory_qos_reward_v1 `
        --probe-horizon 1.0 `
        --route-split-seed 29 `
        --output-dir $Probe
    if ($LASTEXITCODE -ne 0) { throw "Supervised probe failed." }
}
finally {
    Pop-Location
}
