$ErrorActionPreference = 'Stop'
$workspaceDir = Split-Path -Parent $PSScriptRoot
Push-Location $workspaceDir
try {
    if (-not (Test-Path -LiteralPath '.venv/Scripts/python.exe')) {
        python -m venv .venv
        if ($LASTEXITCODE -ne 0) { throw 'Python 3.11+ is required.' }
    }
    & .venv/Scripts/python.exe -m pip install -e '.[dev]'
    if ($LASTEXITCODE -ne 0) { throw 'Python installation failed.' }
    Push-Location monitoring/web
    try {
        npm ci
        if ($LASTEXITCODE -ne 0) { throw 'npm ci failed. Node.js 22+ is required.' }
        npm run build
        if ($LASTEXITCODE -ne 0) { throw 'Web build failed.' }
    } finally { Pop-Location }
    & .venv/Scripts/python.exe -m monitoring.kneeops import
    if ($LASTEXITCODE -ne 0) { Write-Warning 'Some local records were skipped; inspect import errors.' }
    Write-Host 'Ready. Double-click KneeOps.cmd. Close its console to stop the local app.'
} finally { Pop-Location }
