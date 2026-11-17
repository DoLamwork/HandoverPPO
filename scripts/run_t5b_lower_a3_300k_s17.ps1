param([string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot))

$ErrorActionPreference = 'Stop'
$python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
$experimentId = 'T5B_lower_a3_timing_ppo_qos_first_b300000_s17'
$modelDir = Join-Path $ProjectRoot "results\models\early_ho_timing\$experimentId"
$benchmarkDir = Join-Path $ProjectRoot 'results\benchmark\early_ho_timing\T5B_lower_a3_qos_first_b300000_s17_all_speeds'
$logDir = Join-Path $ProjectRoot 'results\automation\t5b_lower_a3_300k_s17'

foreach ($path in @($modelDir, $benchmarkDir, $logDir)) {
    if (Test-Path -LiteralPath $path) {
        throw "Refusing to overwrite existing output: $path"
    }
}
New-Item -ItemType Directory -Path $logDir | Out-Null

function Write-Status([string]$Event, [hashtable]$Fields) {
    $record = [ordered]@{
        timestamp_utc = [DateTimeOffset]::UtcNow.ToString('o')
        event = $Event
    }
    foreach ($key in $Fields.Keys) { $record[$key] = $Fields[$key] }
    Add-Content -LiteralPath (Join-Path $logDir 'runner_status.jsonl') `
        -Value ($record | ConvertTo-Json -Compress -Depth 8) -Encoding utf8
}

function Run-Process([string]$Name, [string[]]$Arguments) {
    Write-Status 'job_started' @{ name = $Name; arguments = $Arguments }
    $process = Start-Process -FilePath $python -ArgumentList $Arguments `
        -WorkingDirectory $ProjectRoot -WindowStyle Hidden -Wait -PassThru `
        -RedirectStandardOutput (Join-Path $logDir "$Name.stdout.log") `
        -RedirectStandardError (Join-Path $logDir "$Name.stderr.log")
    if ($process.ExitCode -ne 0) {
        throw "$Name failed with exit code $($process.ExitCode)."
    }
}

try {
    Write-Status 'runner_started' @{
        experiment_id = $experimentId
        budget = 300000
        ppo_a3_offset_db = 0.0
        reference_a3_offset_db = 1.0
        a3_hysteresis_db = 1.0
        a3_ttt_ms = 40
        checkpoint_selection = 'qos_first_v2'
    }
    $trainArgs = @(
        '-u', 'run.py', 'train_ppo',
        '--data-dir', 'data/uav_100m_grid100m_dev_processed',
        '--speeds', '30', '50', '70', '90',
        '--candidate-count', '1', '--candidate-order', 'ranked_v1',
        '--observation-schema', 't4d1_compact_v1',
        '--reward-schema', 'trajectory_qos_recovery_reward_v1',
        '--stay-mode', 'a3_redecision_v1', '--stay-redecision-ms', '20',
        '--a3-offset-db', '0', '--reference-a3-offset-db', '1',
        '--network-architecture', 'shared_mlp_v1',
        '--value-normalization', 'fixed_scale_v1',
        '--value-scale-calibration-routes-per-speed', '20',
        '--total-macro-steps', '300000', '--rollout-steps', '1024',
        '--batch-size', '128', '--update-epochs', '4',
        '--learning-rate', '0.00005', '--evaluation-interval', '10240',
        '--stochastic-validation-seeds', '17', '29', '43',
        '--learning-diagnostics',
        '--qos-rate-ratio', '0.98', '--qos-rate-ratio-per-speed', '0.98',
        '--qos-relative-rate-ratio', '0.98',
        '--qos-relative-rate-ratio-per-speed', '0.98',
        '--seed', '17', '--split-seed', '17', '--device', 'cpu',
        '--skip-final-test', '--experiment-id', $experimentId,
        '--parent-id', 'T5A_timing_ppo_fixed_scale_b300000_s17',
        '--changed-factor', 'ppo_a3_offset_1_to_0_reference_a3_offset_stays_1',
        '--output-dir', $modelDir
    )
    Run-Process 'train' $trainArgs

    $report = Get-Content -LiteralPath (Join-Path $modelDir 'training_report.json') -Raw | ConvertFrom-Json
    $config = $report.experiment_config.config
    if ($config.experiment_id -ne $experimentId -or
        [int]$config.budget.total_macro_steps -ne 300000 -or
        [double]$config.protocol_config.a3_off -ne 0.0 -or
        [double]$config.reference_protocol_config.a3_off -ne 1.0 -or
        [int]$config.protocol_config.a3_ttt_ms -ne 40 -or
        [bool]$report.test_evaluated) {
        throw 'Completed training report violates the frozen T5B contract.'
    }
    $checkpoint = Join-Path $modelDir 'smdp_ppo_selected.pt'
    if (-not (Test-Path -LiteralPath $checkpoint -PathType Leaf)) {
        throw "Missing selected checkpoint: $checkpoint"
    }
    Write-Status 'training_completed' @{
        checkpoint = $checkpoint
        selection_reason = $report.selection_reason
    }

    $benchmarkArgs = @(
        '-u', 'run.py', 'benchmark',
        '--data-dir', 'data/uav_100m_grid100m_dev_processed',
        '--checkpoint', $checkpoint,
        '--split', 'test', '--split-seed', '17',
        '--speeds', '30', '50', '70', '90',
        '--zero-shot-speeds', '120', '150', '--device', 'cpu',
        '--output-dir', $benchmarkDir
    )
    Run-Process 'benchmark' $benchmarkArgs
    if (-not (Test-Path -LiteralPath (Join-Path $benchmarkDir 'summary.json'))) {
        throw 'Benchmark completed without summary.json.'
    }
    Write-Status 'runner_completed' @{
        model_dir = $modelDir
        benchmark_dir = $benchmarkDir
    }
}
catch {
    Write-Status 'runner_failed' @{ error = $_.ToString() }
    throw
}
