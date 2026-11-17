param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot)
)

$ErrorActionPreference = 'Stop'
$python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
$output = Join-Path $ProjectRoot 'results\models\early_ho_timing\T5_timing_ppo_pilot_b10240_s17'

if (Test-Path -LiteralPath $output) {
    throw "Refusing to overwrite existing T5 pilot output: $output"
}

$arguments = @(
    'run.py', 'train_ppo',
    '--data-dir', 'data/uav_100m_grid100m_dev_processed',
    '--speeds', '30', '50', '70', '90',
    '--candidate-count', '1',
    '--candidate-order', 'ranked_v1',
    '--observation-schema', 't4d1_compact_v1',
    '--reward-schema', 'trajectory_qos_recovery_reward_v1',
    '--stay-mode', 'a3_redecision_v1',
    '--stay-redecision-ms', '20',
    '--total-macro-steps', '10240',
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
    '--seed', '17',
    '--split-seed', '17',
    '--skip-final-test',
    '--experiment-id', 'T5_timing_ppo_pilot_b10240_s17',
    '--parent-id', 'sinr_rule_fullroute_fresh_confirmation_v1',
    '--changed-factor', 'ppo_timing_policy_ho_now_wait20',
    '--output-dir', 'results/models/early_ho_timing/T5_timing_ppo_pilot_b10240_s17'
)

& $python @arguments
exit $LASTEXITCODE
