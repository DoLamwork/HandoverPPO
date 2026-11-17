param(
    [Parameter(Mandatory = $true)]
    [string]$ModelDir,
    [string]$OutputName = 'smdp_ppo_selected_qos_first_v2.pt'
)

$ErrorActionPreference = 'Stop'
$resolvedModelDir = (Resolve-Path -LiteralPath $ModelDir).Path
$metricsPath = Join-Path $resolvedModelDir 'checkpoints\validation_metrics.json'
$outputPath = Join-Path $resolvedModelDir $OutputName
$manifestPath = Join-Path $resolvedModelDir 'qos_first_v2_selection.json'

foreach ($path in @($outputPath, $manifestPath)) {
    if (Test-Path -LiteralPath $path) {
        throw "Refusing to overwrite existing artifact: $path"
    }
}
if (-not (Test-Path -LiteralPath $metricsPath -PathType Leaf)) {
    throw "Missing validation metrics: $metricsPath"
}

$records = Get-Content -LiteralPath $metricsPath -Raw | ConvertFrom-Json
$feasible = @()
foreach ($record in $records) {
    if ([bool]$record.qos_feasible) {
        $feasible += $record
    }
}
if ($feasible.Count -eq 0) {
    throw 'No QoS-feasible validation checkpoint exists.'
}

$sortProperties = @(
    @{ Expression = { -([double]$_.validation.mean_relative_rate_percent) } }
    @{ Expression = { [double]($_.validation.handover_failure_percent) } }
    @{ Expression = { [double]($_.validation.ping_pong_percent) } }
    @{ Expression = { [double]($_.validation.rlf_time_percent) } }
    @{ Expression = { [double]($_.validation.total_handover_attempts) } }
)
$ranked = @($feasible | Sort-Object -Property $sortProperties)
$selected = $ranked[0]
$sourcePath = [string]$selected.checkpoint_path
if (-not (Test-Path -LiteralPath $sourcePath -PathType Leaf)) {
    throw "Selected checkpoint is missing: $sourcePath"
}

Copy-Item -LiteralPath $sourcePath -Destination $outputPath
$manifest = [ordered]@{
    schema_name = 'qos_first_checkpoint_selection'
    schema_version = 1
    created_at_utc = [DateTimeOffset]::UtcNow.ToString('o')
    selection_policy = 'qos_first_v2'
    priority = @(
        'mean_relative_rate_percent:max'
        'handover_failure_percent:min'
        'ping_pong_percent:min'
        'rlf_time_percent:min'
        'total_handover_attempts:min'
    )
    feasible_checkpoint_count = $feasible.Count
    source_checkpoint = $sourcePath
    exported_checkpoint = $outputPath
    macro_steps = [int]$selected.macro_steps
    validation = $selected.validation
}
$manifest | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $manifestPath -Encoding utf8

$manifest | ConvertTo-Json -Depth 4
