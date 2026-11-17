param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot)
)

$ErrorActionPreference = 'Stop'

$python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
$dataDir = 'data/uav_100m_grid100m_dev_processed'
$resultRoot = Join-Path $ProjectRoot 'results\benchmark\early_ho_timing\T5A_100k_multiseed_final_test_v1'
$automationRoot = Join-Path $ProjectRoot 'results\automation\t5a_final_test'
$statusPath = Join-Path $automationRoot 'runner_status.jsonl'
$lockPath = Join-Path $resultRoot 'final_test_lock.json'

if (Test-Path -LiteralPath $resultRoot) {
    throw "Refusing to overwrite existing final-test output: $resultRoot"
}
if (Test-Path -LiteralPath $automationRoot) {
    throw "Refusing to overwrite existing final-test automation logs: $automationRoot"
}

New-Item -ItemType Directory -Path $resultRoot | Out-Null
New-Item -ItemType Directory -Path $automationRoot | Out-Null

function Get-Sha256 {
    param([string]$Path)

    $stream = [System.IO.File]::OpenRead($Path)
    $sha256 = [System.Security.Cryptography.SHA256]::Create()
    try {
        $digest = $sha256.ComputeHash($stream)
        return ([System.BitConverter]::ToString($digest)).Replace('-', '').ToLowerInvariant()
    }
    finally {
        $sha256.Dispose()
        $stream.Dispose()
    }
}

$runs = @(
    [pscustomobject]@{
        Seed = 17
        ExperimentId = 'T5A_timing_ppo_fixed_scale_b100000_s17'
    },
    [pscustomobject]@{
        Seed = 29
        ExperimentId = 'T5A_timing_ppo_fixed_scale_b100000_s29'
    },
    [pscustomobject]@{
        Seed = 43
        ExperimentId = 'T5A_timing_ppo_fixed_scale_b100000_s43'
    }
)

$lockRuns = @()
foreach ($run in $runs) {
    $modelDir = Join-Path $ProjectRoot "results\models\early_ho_timing\$($run.ExperimentId)"
    $reportPath = Join-Path $modelDir 'training_report.json'
    $checkpointPath = Join-Path $modelDir 'smdp_ppo_selected.pt'
    if (-not (Test-Path -LiteralPath $reportPath -PathType Leaf)) {
        throw "Missing training report: $reportPath"
    }
    if (-not (Test-Path -LiteralPath $checkpointPath -PathType Leaf)) {
        throw "Missing selected checkpoint: $checkpointPath"
    }

    $report = Get-Content -LiteralPath $reportPath -Raw | ConvertFrom-Json
    $config = $report.experiment_config.config
    if ([string]$config.experiment_id -ne $run.ExperimentId) {
        throw "Experiment ID mismatch for seed $($run.Seed)."
    }
    if ([int]$config.seeds.training -ne $run.Seed) {
        throw "Training seed mismatch for $($run.ExperimentId)."
    }
    if ([int]$config.seeds.split -ne 17) {
        throw "Split seed is not 17 for $($run.ExperimentId)."
    }
    if ([int]$config.budget.total_macro_steps -ne 100000) {
        throw "Training budget is not 100000 for $($run.ExperimentId)."
    }
    if ([string]$config.ppo.device -ne 'cpu') {
        throw "Training device is not CPU for $($run.ExperimentId)."
    }
    if ([bool]$report.test_evaluated) {
        throw "Held-out test was already evaluated for $($run.ExperimentId)."
    }
    if ([string]$report.selection_reason -ne 'best_qos') {
        throw "Selected checkpoint is not best_qos for $($run.ExperimentId)."
    }

    $lockRuns += [ordered]@{
        seed = $run.Seed
        experiment_id = $run.ExperimentId
        training_budget_macro_steps = 100000
        split_seed = 17
        training_device = 'cpu'
        selection_reason = 'best_qos'
        checkpoint = $checkpointPath
        checkpoint_sha256 = Get-Sha256 -Path $checkpointPath
        training_report = $reportPath
        training_report_sha256 = Get-Sha256 -Path $reportPath
        dataset_content_sha256 = [string]$config.dataset.content_sha256
        dataset_topology_sha256 = [string]$config.dataset.topology_sha256
    }
}

$datasetHashes = @($lockRuns | ForEach-Object { $_.dataset_content_sha256 } | Sort-Object -Unique)
$topologyHashes = @($lockRuns | ForEach-Object { $_.dataset_topology_sha256 } | Sort-Object -Unique)
if ($datasetHashes.Count -ne 1 -or $topologyHashes.Count -ne 1) {
    throw 'The three runs do not share one frozen dataset/topology provenance.'
}

$lock = [ordered]@{
    schema_name = 't5a_100k_multiseed_final_test_lock'
    schema_version = 1
    created_at_utc = [DateTimeOffset]::UtcNow.ToString('o')
    decision = 'Freeze 100k multi-seed and open held-out test exactly once.'
    split = 'test'
    split_seed = 17
    split_fractions = [ordered]@{ train = 0.70; validation = 0.15; test = 0.15 }
    speeds_kmh = @(30, 50, 70, 90)
    expected_test_routes = 150
    zero_shot_enabled = $false
    benchmark_device = 'cpu'
    runs = $lockRuns
}
$lock | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $lockPath -Encoding utf8

function Write-Status {
    param(
        [string]$Event,
        [hashtable]$Fields
    )
    $record = [ordered]@{
        timestamp_utc = [DateTimeOffset]::UtcNow.ToString('o')
        event = $Event
    }
    foreach ($key in $Fields.Keys) {
        $record[$key] = $Fields[$key]
    }
    Add-Content -LiteralPath $statusPath -Value ($record | ConvertTo-Json -Compress -Depth 6) -Encoding utf8
}

Write-Status -Event 'runner_started' -Fields @{
    seeds = @(17, 29, 43)
    result_root = $resultRoot
    lock_file = $lockPath
}

foreach ($run in $runs) {
    $modelDir = Join-Path $ProjectRoot "results\models\early_ho_timing\$($run.ExperimentId)"
    $checkpointPath = Join-Path $modelDir 'smdp_ppo_selected.pt'
    $outputDir = Join-Path $resultRoot "seed_$($run.Seed)"
    $stdoutPath = Join-Path $automationRoot "seed_$($run.Seed).stdout.log"
    $stderrPath = Join-Path $automationRoot "seed_$($run.Seed).stderr.log"
    if (Test-Path -LiteralPath $outputDir) {
        throw "Refusing to overwrite seed output: $outputDir"
    }

    Write-Status -Event 'seed_started' -Fields @{
        seed = $run.Seed
        experiment_id = $run.ExperimentId
        output_dir = $outputDir
    }

    $arguments = @(
        'run.py', 'benchmark',
        '--data-dir', $dataDir,
        '--checkpoint', $checkpointPath,
        '--split', 'test',
        '--split-seed', '17',
        '--speeds', '30', '50', '70', '90',
        '--device', 'cpu',
        '--output-dir', $outputDir
    )

    $process = Start-Process `
        -FilePath $python `
        -ArgumentList $arguments `
        -WorkingDirectory $ProjectRoot `
        -WindowStyle Hidden `
        -RedirectStandardOutput $stdoutPath `
        -RedirectStandardError $stderrPath `
        -Wait `
        -PassThru

    if ($process.ExitCode -ne 0) {
        Write-Status -Event 'seed_failed' -Fields @{
            seed = $run.Seed
            exit_code = $process.ExitCode
            stderr = $stderrPath
        }
        throw "Final held-out benchmark failed for seed $($run.Seed) with exit code $($process.ExitCode)."
    }

    $summaryPath = Join-Path $outputDir 'summary.json'
    if (-not (Test-Path -LiteralPath $summaryPath -PathType Leaf)) {
        Write-Status -Event 'seed_failed' -Fields @{
            seed = $run.Seed
            reason = 'missing_benchmark_summary'
        }
        throw "Missing benchmark summary for seed $($run.Seed)."
    }

    Write-Status -Event 'seed_completed' -Fields @{
        seed = $run.Seed
        output_dir = $outputDir
        summary = $summaryPath
    }
}

Write-Status -Event 'runner_completed' -Fields @{
    seeds = @(17, 29, 43)
    result_root = $resultRoot
}

