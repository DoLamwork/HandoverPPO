param(
    [string]$Python = ".\.venv\Scripts\python.exe"
)

$Root = Split-Path -Parent $PSScriptRoot
Push-Location $Root
try {
    & $Python -m py_compile `
        src\ho_optim_drl\commands\audit_timing_target.py `
        src\ho_optim_drl\commands\audit_timing_target_main.py `
        src\ho_optim_drl\commands\verify_d63_1.py `
        src\ho_optim_drl\commands\prepare_d63_2_labels.py `
        src\ho_optim_drl\commands\probe_d63_2.py `
        src\ho_optim_drl\commands\analyze_d63_2_false_positives.py `
        src\ho_optim_drl\commands\replay_d63_2_failure_cases.py `
        src\ho_optim_drl\commands\validate_d63_3_feature.py
    if ($LASTEXITCODE -ne 0) { throw "D63 py_compile failed." }

    & $Python -m unittest `
        tests.test_d63_timing_target_audit `
        tests.test_d63_2_labels `
        tests.test_d63_2_probe `
        tests.test_d63_2_false_positive `
        tests.test_d63_2_failure_replay `
        tests.test_d63_3_feature_validation
    if ($LASTEXITCODE -ne 0) { throw "D63 test suite failed." }

    $required = @(
        "results\analysis\d63_timing_target_main_analysis_v1\decision_gate.json",
        "results\analysis\d63_1_verification_v1\verification_report.json",
        "results\analysis\d63_2_labels_v1\label_report.json",
        "results\analysis\d63_2_probe_v1\probe_report.json",
        "results\analysis\d63_2_target_mlp_fp_v1\false_positive_report.json",
        "results\analysis\d63_2_failure_case_replay_v1\case_replay_report.json",
        "results\analysis\d63_3_execution_safety_v1\feature_validation_report.json",
        "results\analysis\d63_3_execution_safety_v1\decision_gate.json"
    )
    foreach ($item in $required) {
        if (-not (Test-Path -LiteralPath $item -PathType Leaf)) {
            throw "Missing D63 artifact: $item"
        }
    }

    $g1 = Get-Content -LiteralPath $required[0] -Raw | ConvertFrom-Json
    $g3 = Get-Content -LiteralPath $required[7] -Raw | ConvertFrom-Json
    if ($g1.status -ne "PASS") { throw "Unexpected D63 G1 status: $($g1.status)" }
    if ($g3.status -ne "CLOSE_D63_NO_GENERALIZING_FEATURE_EVIDENCE") {
        throw "Unexpected D63-3 status: $($g3.status)"
    }
    Write-Host "D63 finalized: G1=PASS, G2=FAIL, D63-3=$($g3.status)."
}
finally {
    Pop-Location
}
