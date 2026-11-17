param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot),
    [ValidateSet('cpu', 'cuda')]
    [string]$Device = 'cuda',
    [switch]$WithStochasticValidation
)

$ErrorActionPreference = 'Stop'
$python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
$logDir = Join-Path $ProjectRoot 'results\automation\t6a_100k_multiseed'
$seeds = @(29, 43)

function Model-Dir([int]$Seed) {
    Join-Path $ProjectRoot "results\models\early_ho_timing\T6A_a3_entry_timing_b100000_s$Seed"
}

function Write-Status([string]$EventName, [hashtable]$Fields) {
    $record = @{
        timestamp_utc = [DateTimeOffset]::UtcNow.ToString('o')
        event = $EventName
    }
    foreach ($key in $Fields.Keys) {
        $record[$key] = $Fields[$key]
    }
    Add-Content `
        -LiteralPath (Join-Path $logDir 'runner_status.jsonl') `
        -Value ($record | ConvertTo-Json -Compress -Depth 8) `
        -Encoding utf8
}

if (-not (Test-Path -LiteralPath $python)) {
    throw "Python environment not found: $python"
}

foreach ($seed in $seeds) {
    $output = Model-Dir $seed
    if (Test-Path -LiteralPath $output) {
        throw "Refusing to overwrite existing T6A output: $output"
    }
}

if (Test-Path -LiteralPath $logDir) {
    throw "Refusing to overwrite existing T6A runner logs: $logDir"
}

$active = @(
    Get-CimInstance Win32_Process |
        Where-Object {
            $_.Name -match '^python' -and
            $_.CommandLine -match 'run\.py\s+train_ppo'
        }
)
if ($active.Count -gt 0) {
    throw 'Another PPO training process is already running; refusing concurrent training.'
}

New-Item -ItemType Directory -Path $logDir | Out-Null
Write-Status 'runner_started' @{
    seeds = $seeds
    budget = 100000
    device = $Device
    split_seed = 17
}

Push-Location -LiteralPath $ProjectRoot
try {
    foreach ($seed in $seeds) {
        $experimentId = "T6A_a3_entry_timing_b100000_s$seed"
        $output = Model-Dir $seed
        $arguments = @(
            '-u', 'run.py', 'train_ppo',
            '--data-dir', 'data/uav_100m_grid100m_dev_processed',
            '--speeds', '30', '50', '70', '90',
            '--candidate-count', '1',
            '--candidate-order', 'ranked_v1',
            '--observation-schema', 't4d1_compact_v1',
            '--reward-schema', 'trajectory_qos_recovery_reward_v1',
            '--decision-trigger', 'a3_entry_redecision_v1',
            '--stay-mode', 'a3_redecision_v1',
            '--stay-redecision-ms', '20',
            '--network-architecture', 'shared_mlp_v1',
            '--value-normalization', 'fixed_scale_v1',
            '--value-scale-calibration-routes-per-speed', '20',
            '--total-macro-steps', '100000',
            '--rollout-steps', '1024',
            '--batch-size', '128',
            '--update-epochs', '4',
            '--learning-rate', '0.00005',
            '--evaluation-interval', '10240',
            '--learning-diagnostics',
            '--qos-rate-ratio', '0.98',
            '--qos-rate-ratio-per-speed', '0.98',
            '--qos-relative-rate-ratio', '0.98',
            '--qos-relative-rate-ratio-per-speed', '0.98',
            '--seed', "$seed",
            '--split-seed', '17',
            '--device', $Device,
            '--skip-final-test',
            '--experiment-id', $experimentId,
            '--parent-id', 'T6A_a3_entry_timing_b100000_s17',
            '--changed-factor', "training_seed_17_to_$seed",
            '--output-dir', "results/models/early_ho_timing/$experimentId"
        )

        if ($WithStochasticValidation) {
            $arguments += @(
                '--stochastic-validation-seeds', '17', '29', '43'
            )
        }

        $stdout = Join-Path $logDir "seed_$seed.stdout.log"
        $stderr = Join-Path $logDir "seed_$seed.stderr.log"
        Write-Status 'training_started' @{
            seed = $seed
            experiment_id = $experimentId
            output_dir = $output
        }

        $process = Start-Process `
            -FilePath $python `
            -ArgumentList $arguments `
            -WorkingDirectory $ProjectRoot `
            -WindowStyle Hidden `
            -RedirectStandardOutput $stdout `
            -RedirectStandardError $stderr `
            -PassThru `
            -Wait

        if ($process.ExitCode -ne 0) {
            throw "T6A seed $seed exited with code $($process.ExitCode). See $stderr"
        }

        $reportPath = Join-Path $output 'training_report.json'
        $selectedPath = Join-Path $output 'smdp_ppo_selected.pt'
        $latestPath = Join-Path $output 'smdp_ppo_latest.pt'
        if (-not (Test-Path -LiteralPath $reportPath)) {
            throw "Missing training report for seed ${seed}: $reportPath"
        }
        if (-not (Test-Path -LiteralPath $selectedPath)) {
            throw "Missing selected checkpoint for seed ${seed}: $selectedPath"
        }
        if (-not (Test-Path -LiteralPath $latestPath)) {
            throw "Missing latest checkpoint for seed ${seed}: $latestPath"
        }

        $report = Get-Content -LiteralPath $reportPath -Raw | ConvertFrom-Json
        $config = $report.experiment_config.config
        if (
            $config.experiment_id -ne $experimentId -or
            $config.budget.total_macro_steps -ne 100000 -or
            $config.seeds.training -ne $seed -or
            $config.seeds.split -ne 17 -or
            $config.ppo.device -ne $Device -or
            $report.objective.trigger -ne 'a3_entry_redecision_v1' -or
            $report.test_evaluated -ne $false
        ) {
            throw "Completed report violates the frozen T6A contract for seed $seed."
        }

        Write-Status 'training_completed' @{
            seed = $seed
            experiment_id = $experimentId
            experiment_status = $report.experiment_status
            selection_reason = $report.selection_reason
            selected_checkpoint = $selectedPath
        }
    }

    Write-Status 'runner_completed' @{ seeds = $seeds }
}
catch {
    Write-Status 'runner_failed' @{ error = $_.ToString() }
    throw
}
finally {
    Pop-Location
}
