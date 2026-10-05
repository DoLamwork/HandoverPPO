param(
    [string]$DataDir = "data/uav_100m_grid100m_dev_processed",
    [string]$OutputRoot = "results/models/d3_pilots",
    [int]$Seed = 17
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$RunPy = Join-Path $RepoRoot "run.py"
$Output = Join-Path $RepoRoot $OutputRoot

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Project Python was not found: $Python"
}

$Common = @(
    $RunPy, "train_ppo",
    "--data-dir", $DataDir,
    "--speeds", "30", "50", "70", "90",
    "--candidate-count", "3",
    "--candidate-order", "ranked_v1",
    "--observation-schema", "qos_full_v1",
    "--reward-schema", "trajectory_qos_reward_v1",
    "--total-macro-steps", "10240",
    "--rollout-steps", "1024",
    "--batch-size", "128",
    "--update-epochs", "4",
    "--learning-rate", "0.00005",
    "--evaluation-interval", "10240",
    "--learning-diagnostics",
    "--seed", "$Seed",
    "--split-seed", "$Seed",
    "--skip-final-test"
)

$Arms = @(
    @{
        Id = "D30_e07_control_pilot_b10240_s$Seed"
        Parent = "E07_qos_rate_k3_s17"
        Factor = "d3_instrumented_control"
        Extra = @("--network-architecture", "shared_mlp_v1")
    },
    @{
        Id = "D31_e07_separate_pilot_b10240_s$Seed"
        Parent = "D30_e07_control_pilot_b10240_s$Seed"
        Factor = "network_architecture_shared_to_separate"
        Extra = @("--network-architecture", "separate_mlp_v1")
    },
    @{
        Id = "D32_e07_vscale_pilot_b10240_s$Seed"
        Parent = "D30_e07_control_pilot_b10240_s$Seed"
        Factor = "value_normalization_fixed_scale"
        Extra = @(
            "--network-architecture", "shared_mlp_v1",
            "--value-normalization", "fixed_scale_v1",
            "--value-scale-calibration-routes-per-speed", "20"
        )
    }
)

function Test-CompletedArm {
    param(
        [string]$ArmOutput,
        [string]$ExpectedId
    )
    $ReportPath = Join-Path $ArmOutput "training_report.json"
    if (-not (Test-Path -LiteralPath $ReportPath -PathType Leaf)) {
        return $false
    }
    try {
        $Report = Get-Content -LiteralPath $ReportPath -Raw | ConvertFrom-Json
        $Config = $Report.experiment_config.config
        $LastHistory = $Report.history | Select-Object -Last 1
        $Latest = [string]$Report.checkpoints.latest
        $Selected = [string]$Report.checkpoints.selected_export
        return (
            $Config.experiment_id -eq $ExpectedId -and
            [int]$Config.budget.total_macro_steps -eq 10240 -and
            $Report.test_evaluated -eq $false -and
            [int]$LastHistory.macro_steps -eq 10240 -and
            (Test-Path -LiteralPath $Latest -PathType Leaf) -and
            (Test-Path -LiteralPath $Selected -PathType Leaf)
        )
    }
    catch {
        return $false
    }
}

Push-Location $RepoRoot
try {
    foreach ($Arm in $Arms) {
        $ArmOutput = Join-Path $Output $Arm.Id
        if (Test-Path -LiteralPath $ArmOutput) {
            $ExistingItems = @(
                Get-ChildItem -LiteralPath $ArmOutput -Force
            )
            if (Test-CompletedArm -ArmOutput $ArmOutput -ExpectedId $Arm.Id) {
                Write-Host "Skipping completed D3 pilot: $($Arm.Id)"
                continue
            }
            if ($ExistingItems.Count -gt 0) {
                throw (
                    "Refusing to overwrite incomplete D3 pilot output: " +
                    $ArmOutput
                )
            }
        }
        $Arguments = @($Common) + @(
            "--experiment-id", $Arm.Id,
            "--parent-id", $Arm.Parent,
            "--changed-factor", $Arm.Factor,
            "--output-dir", $ArmOutput
        ) + @($Arm.Extra)
        & $Python @Arguments
        if ($LASTEXITCODE -ne 0) {
            throw "D3 pilot failed: $($Arm.Id)"
        }
    }
}
finally {
    Pop-Location
}
