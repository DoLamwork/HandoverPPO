param(
    [string]$DataDir = "data/uav_100m_grid100m_dev_processed",
    [string]$OutputRoot = "results/models/d3_screening",
    [string]$LogRoot = "results/automation/d3_screening",
    [int]$Seed = 17
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$RunPy = Join-Path $RepoRoot "run.py"
$Output = Join-Path $RepoRoot $OutputRoot
$Logs = Join-Path $RepoRoot $LogRoot
$StatusPath = Join-Path $Logs "runner_status.jsonl"

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Project Python was not found: $Python"
}
New-Item -ItemType Directory -Force -Path $Output, $Logs | Out-Null

function Write-RunnerStatus {
    param([string]$Event, [string]$Arm, [string]$Message = "")
    [ordered]@{
        timestamp = [DateTimeOffset]::Now.ToString("o")
        event = $Event
        arm = $Arm
        message = $Message
    } | ConvertTo-Json -Compress | Add-Content -LiteralPath $StatusPath
}

function Test-CompletedArm {
    param(
        [string]$ArmOutput,
        [string]$ExpectedId,
        [string]$ExpectedValueNormalization
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
            [int]$Config.budget.total_macro_steps -eq 100000 -and
            $Config.candidate.K -eq 3 -and
            $Config.candidate.candidate_order_key -eq "ranked_v1" -and
            $Config.observation_schema.key -eq "qos_full_v1" -and
            $Config.reward_schema -eq "trajectory_qos_reward_v1" -and
            $Config.model.architecture -eq "shared_mlp_v1" -and
            $Config.model.value_normalization -eq $ExpectedValueNormalization -and
            $Report.test_evaluated -eq $false -and
            [int]$LastHistory.macro_steps -eq 100000 -and
            (Test-Path -LiteralPath $Latest -PathType Leaf) -and
            (Test-Path -LiteralPath $Selected -PathType Leaf)
        )
    }
    catch {
        return $false
    }
}

$Common = @(
    $RunPy, "train_ppo",
    "--data-dir", $DataDir,
    "--speeds", "30", "50", "70", "90",
    "--candidate-count", "3",
    "--candidate-order", "ranked_v1",
    "--observation-schema", "qos_full_v1",
    "--reward-schema", "trajectory_qos_reward_v1",
    "--network-architecture", "shared_mlp_v1",
    "--total-macro-steps", "100000",
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
        Id = "D30_e07_control_b100k_s$Seed"
        Parent = "E07_qos_rate_k3_s17"
        Factor = "d3_instrumented_control"
        ValueNormalization = "none"
        Extra = @()
    },
    @{
        Id = "D32_e07_vscale_b100k_s$Seed"
        Parent = "D30_e07_control_b100k_s$Seed"
        Factor = "value_normalization_fixed_scale"
        ValueNormalization = "fixed_scale_v1"
        Extra = @(
            "--value-normalization", "fixed_scale_v1",
            "--value-scale-calibration-routes-per-speed", "20"
        )
    }
)

Push-Location $RepoRoot
try {
    Write-RunnerStatus -Event "runner_started" -Arm ""
    foreach ($Arm in $Arms) {
        $ArmOutput = Join-Path $Output $Arm.Id
        if (Test-Path -LiteralPath $ArmOutput) {
            if (
                Test-CompletedArm `
                    -ArmOutput $ArmOutput `
                    -ExpectedId $Arm.Id `
                    -ExpectedValueNormalization $Arm.ValueNormalization
            ) {
                Write-RunnerStatus -Event "arm_skipped_complete" -Arm $Arm.Id
                continue
            }
            if (@(Get-ChildItem -LiteralPath $ArmOutput -Force).Count -gt 0) {
                throw "Refusing to overwrite incomplete arm: $ArmOutput"
            }
        }

        $Stdout = Join-Path $Logs "$($Arm.Id).stdout.log"
        $Stderr = Join-Path $Logs "$($Arm.Id).stderr.log"
        if ((Test-Path $Stdout) -or (Test-Path $Stderr)) {
            throw "Refusing to overwrite existing arm logs for $($Arm.Id)."
        }
        $Arguments = @($Common) + @(
            "--experiment-id", $Arm.Id,
            "--parent-id", $Arm.Parent,
            "--changed-factor", $Arm.Factor,
            "--output-dir", $ArmOutput
        ) + @($Arm.Extra)

        Write-RunnerStatus -Event "arm_started" -Arm $Arm.Id
        & $Python @Arguments 1> $Stdout 2> $Stderr
        if ($LASTEXITCODE -ne 0) {
            throw "D3 screening arm failed with exit ${LASTEXITCODE}: $($Arm.Id)"
        }
        if (-not (
            Test-CompletedArm `
                -ArmOutput $ArmOutput `
                -ExpectedId $Arm.Id `
                -ExpectedValueNormalization $Arm.ValueNormalization
        )) {
            throw "D3 arm exited without a valid completion report: $($Arm.Id)"
        }
        Write-RunnerStatus -Event "arm_completed" -Arm $Arm.Id
    }
    Write-RunnerStatus -Event "runner_completed" -Arm ""
}
catch {
    Write-RunnerStatus -Event "runner_failed" -Arm "" -Message $_.Exception.Message
    throw
}
finally {
    Pop-Location
}
