param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot),
    [ValidateSet('cpu', 'cuda')]
    [string]$Device = 'cuda',
    [switch]$WithStochasticValidation
)

$ErrorActionPreference = 'Stop'
$Python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
$ExperimentId = 'P1_T6A_dense_urban_b100000_s17'
$Output = Join-Path $ProjectRoot "results\models\p1_dense_urban\$ExperimentId"

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python environment not found: $Python"
}
if (-not (Test-Path -LiteralPath (Join-Path $ProjectRoot 'data\uav_p1_dense_urban_v1_pilot\dataset_metadata.json') -PathType Leaf)) {
    throw 'P1 dense-urban dataset is missing or incomplete.'
}
if (Test-Path -LiteralPath $Output) {
    throw "Refusing to overwrite existing P1 training output: $Output"
}

$Arguments = @(
    'run.py', 'train_ppo',
    '--data-dir', 'data/uav_p1_dense_urban_v1_pilot',
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
    '--seed', '17',
    '--split-seed', '17',
    '--device', $Device,
    '--skip-final-test',
    '--experiment-id', $ExperimentId,
    '--parent-id', 'P0_T6A_canonical_b100000_s17',
    '--changed-factor', 'dataset_p0_canonical_to_p1_dense_urban',
    '--output-dir', "results/models/p1_dense_urban/$ExperimentId"
)

if ($WithStochasticValidation) {
    $Arguments += @(
        '--stochastic-validation-seeds', '17', '29', '43'
    )
}

Push-Location -LiteralPath $ProjectRoot
try {
    & $Python @Arguments
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
