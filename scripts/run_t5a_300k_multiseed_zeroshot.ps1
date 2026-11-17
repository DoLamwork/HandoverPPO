param([string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot))
$ErrorActionPreference = 'Stop'
$python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
$logs = Join-Path $ProjectRoot 'results\automation\t5a_300k_zeroshot_v1'
$bench = Join-Path $ProjectRoot 'results\benchmark\early_ho_timing\T5A_300k_multiseed_zeroshot_v1'
$seeds = @(17,29,43)
function Model-Dir($seed) { Join-Path $ProjectRoot "results\models\early_ho_timing\T5A_timing_ppo_fixed_scale_b300000_s$seed" }
foreach ($path in @($logs,$bench) + @($seeds | ForEach-Object { Model-Dir $_ })) {
    if (Test-Path -LiteralPath $path) { throw "Refusing existing output: $path" }
}
$active = @(Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python' -and $_.CommandLine -match 'run.py\s+(train_ppo|benchmark)' })
if ($active.Count) { throw 'Existing training/benchmark process; refusing concurrent run.' }
New-Item -ItemType Directory -Path $logs,$bench | Out-Null
function Status($event, $fields) {
    $record = @{ timestamp_utc=[DateTimeOffset]::UtcNow.ToString('o'); event=$event }
    foreach ($key in $fields.Keys) { $record[$key] = $fields[$key] }
    Add-Content -LiteralPath (Join-Path $logs 'runner_status.jsonl') -Value ($record | ConvertTo-Json -Compress -Depth 8) -Encoding utf8
}
function Hash($path) {
    $stream = [IO.File]::OpenRead($path)
    $sha = [Security.Cryptography.SHA256]::Create()
    try { ([BitConverter]::ToString($sha.ComputeHash($stream))).Replace('-','').ToLowerInvariant() }
    finally { $sha.Dispose(); $stream.Dispose() }
}
function Run-Job($label, $arguments) {
    Status 'job_started' @{job=$label; arguments=$arguments}
    $p = Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory $ProjectRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $logs "$label.stdout.log") -RedirectStandardError (Join-Path $logs "$label.stderr.log") -PassThru -Wait
    if ($p.ExitCode -ne 0) { throw "$label exited with code $($p.ExitCode)" }
}
try {
    Status 'runner_started' @{ seeds=$seeds; budget=300000; initialization='fresh'; device='cpu'; benchmark_speeds=@(120,150) }
    $locked = @()
    foreach ($seed in $seeds) {
        $id = "T5A_timing_ppo_fixed_scale_b300000_s$seed"
        $out = Model-Dir $seed
        $arguments = @('-u','run.py','train_ppo','--data-dir','data/uav_100m_grid100m_dev_processed',
            '--speeds','30','50','70','90','--candidate-count','1','--candidate-order','ranked_v1',
            '--observation-schema','t4d1_compact_v1','--reward-schema','trajectory_qos_recovery_reward_v1',
            '--stay-mode','a3_redecision_v1','--stay-redecision-ms','20','--network-architecture','shared_mlp_v1',
            '--value-normalization','fixed_scale_v1','--value-scale-calibration-routes-per-speed','20',
            '--total-macro-steps','300000','--rollout-steps','1024','--batch-size','128','--update-epochs','4',
            '--learning-rate','0.00005','--evaluation-interval','10240','--stochastic-validation-seeds','17','29','43',
            '--learning-diagnostics','--qos-rate-ratio','0.98','--qos-rate-ratio-per-speed','0.98',
            '--qos-relative-rate-ratio','0.98','--qos-relative-rate-ratio-per-speed','0.98',
            '--seed',"$seed",'--split-seed','17','--device','cpu','--skip-final-test',
            '--experiment-id',$id,'--parent-id',"T5A_timing_ppo_fixed_scale_b100000_s$seed",
            '--changed-factor','training_budget_100000_to_300000','--output-dir',$out)
        Run-Job "train_seed_$seed" $arguments
        $reportPath = Join-Path $out 'training_report.json'
        $r = Get-Content -LiteralPath $reportPath -Raw | ConvertFrom-Json
        $c = $r.experiment_config.config
        if ($c.experiment_id -ne $id -or $c.budget.total_macro_steps -ne 300000 -or $r.test_evaluated -ne $false -or $c.seeds.training -ne $seed -or $c.seeds.split -ne 17 -or $c.ppo.device -ne 'cpu') { throw "Invalid report: $id" }
        $checkpoint = Join-Path $out 'smdp_ppo_selected.pt'
        if (-not (Test-Path (Join-Path $out 'smdp_ppo_latest.pt'))) { throw "Missing latest checkpoint: $id" }
        $locked += @{seed=$seed; checkpoint=$checkpoint; checkpoint_sha256=(Hash $checkpoint); report_sha256=(Hash $reportPath); selection_reason=$r.selection_reason}
        Status 'training_completed' @{seed=$seed; checkpoint=$checkpoint; selection_reason=$r.selection_reason}
    }
    @{created_at_utc=[DateTimeOffset]::UtcNow.ToString('o'); runs=$locked; train_speeds=@(30,50,70,90); evaluation_speeds=@(120,150); split='all'; deterministic=$true; note='Zero-shot-only selection via --split all --speeds 120 150; no seen-speed test.'} | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $bench 'zeroshot_lock.json') -Encoding utf8
    foreach ($run in $locked) {
        if ((Hash $run.checkpoint) -ne $run.checkpoint_sha256) { throw 'Checkpoint changed after lock' }
        $out = Join-Path $bench "seed_$($run.seed)"
        Run-Job "zeroshot_seed_$($run.seed)" @('-u','run.py','benchmark','--data-dir','data/uav_100m_grid100m_dev_processed','--checkpoint',$run.checkpoint,'--split','all','--split-seed','17','--speeds','120','150','--device','cpu','--output-dir',$out)
        $summary = Get-Content -LiteralPath (Join-Path $out 'summary.json') -Raw | ConvertFrom-Json
        if ($summary.metadata.split -ne 'all' -or ($summary.metadata.speeds -join ',') -ne '120,150' -or $summary.metadata.checkpoint -ne $run.checkpoint -or $summary.metadata.trajectories -le 0) { throw 'Invalid zero-shot summary' }
        Status 'zeroshot_completed' @{seed=$run.seed; trajectories=$summary.metadata.trajectories; output=$out}
    }
    Status 'runner_completed' @{seeds=$seeds}
} catch {
    Status 'runner_failed' @{error=$_.ToString()}
    throw
}
