param(
    [string]$MainAuditDir = "results\analysis\d63_timing_target_main_analysis_v1",
    [string]$OutputDir = "results\analysis\d63_1_verification_v1"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$mainPath = Join-Path $projectRoot $MainAuditDir
$outputPath = Join-Path $projectRoot $OutputDir
$testCommand = ".venv\Scripts\python.exe -m unittest tests.test_d63_timing_target_audit tests.test_reference"

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Python environment not found: $python"
}
if (-not (Test-Path -LiteralPath $mainPath -PathType Container)) {
    throw "D63-1 main artifact not found: $mainPath"
}
if (Test-Path -LiteralPath $outputPath) {
    $existing = @(Get-ChildItem -LiteralPath $outputPath -Force)
    if ($existing.Count -gt 0) {
        throw "Refusing to overwrite existing D63-1 verification output: $outputPath"
    }
}

Push-Location $projectRoot
try {
    Write-Host "[D63-1] Compiling verification and audit modules..."
    & $python -m py_compile `
        "src\ho_optim_drl\commands\audit_timing_target.py" `
        "src\ho_optim_drl\commands\audit_timing_target_main.py" `
        "src\ho_optim_drl\commands\verify_d63_1.py" `
        "tests\test_d63_timing_target_audit.py"
    if ($LASTEXITCODE -ne 0) {
        throw "D63-1 py_compile failed with exit code $LASTEXITCODE"
    }

    Write-Host "[D63-1] Running full-route and protocol metric tests..."
    & $python -m unittest `
        tests.test_d63_timing_target_audit `
        tests.test_reference
    if ($LASTEXITCODE -ne 0) {
        throw "D63-1 tests failed with exit code $LASTEXITCODE"
    }

    Write-Host "[D63-1] Verifying rows, metric definitions and provenance..."
    & $python run.py verify_d63_1 `
        --main-audit-dir $mainPath `
        --output-dir $outputPath `
        --test-command $testCommand
    if ($LASTEXITCODE -ne 0) {
        throw "D63-1 verification failed with exit code $LASTEXITCODE"
    }

    Write-Host "[D63-1] Reproducibility package: $outputPath"
}
finally {
    Pop-Location
}
