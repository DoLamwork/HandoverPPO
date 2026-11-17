param([int]$Workers = 4)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$StatusDir = Join-Path $ProjectRoot "results\automation\p2_p4_data_pipeline"
$StatusFile = Join-Path $StatusDir "runner_status.jsonl"
New-Item -ItemType Directory -Force -Path $StatusDir | Out-Null

function Write-Status([string]$EventName, [string]$Arm, [string]$Message) {
    [ordered]@{
        timestamp = [DateTimeOffset]::Now.ToString("o")
        event = $EventName
        arm = $Arm
        message = $Message
    } | ConvertTo-Json -Compress | Add-Content -LiteralPath $StatusFile -Encoding UTF8
}

$Arms = @(
    @{ Id = "P2"; Script = "run_p2_high_mobility_pilot.ps1" },
    @{ Id = "P3"; Script = "run_p3_fast_channel_pilot.ps1" },
    @{ Id = "P4"; Script = "run_p4_combined_harsh_pilot.ps1" }
)

Write-Status "runner_started" "" "Sequential P2-P4 data generation started with $Workers workers."
foreach ($Arm in $Arms) {
    $ScriptPath = Join-Path $PSScriptRoot $Arm.Script
    Write-Status "arm_started" $Arm.Id $ScriptPath
    try {
        & $ScriptPath -Workers $Workers
        if ($LASTEXITCODE -ne 0) { throw "$($Arm.Id) exited with code $LASTEXITCODE." }
        Write-Status "arm_completed" $Arm.Id "Dataset metadata written."
    }
    catch {
        Write-Status "arm_failed" $Arm.Id $_.Exception.Message
        throw
    }
}
Write-Status "runner_completed" "" "P2, P3 and P4 datasets completed."
