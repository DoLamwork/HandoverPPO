$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot

Push-Location $ProjectRoot
try {
    & .\.venv\Scripts\python.exe -m ho_optim_drl.commands.audit_p0_quality `
        --data-dir data/uav_p0_canonical_v1_pilot `
        --output-dir results/analysis/p0_canonical_v1_quality_audit_v2
    if ($LASTEXITCODE -ne 0) {
        throw "P0 quality audit exited with code $LASTEXITCODE."
    }
}
finally {
    Pop-Location
}

