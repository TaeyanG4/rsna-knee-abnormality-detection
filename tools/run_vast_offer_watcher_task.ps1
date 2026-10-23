param([ValidateSet('Observe','Auto','Supervise')][string]$Mode = 'Observe')

$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$configPath = Join-Path $projectRoot '.local\vast-retry-20261008\watcher-config.json'
$pythonPath = Join-Path $projectRoot '.local\vast-retry-20261008\venv\Scripts\python.exe'
$watcherPath = Join-Path $projectRoot 'tools\vast_offer_watcher.py'
$stateDir = Join-Path $projectRoot '.local\vast-retry-20261008\watcher-state'
New-Item -ItemType Directory -Force -Path $stateDir | Out-Null
$argument = if ($Mode -eq 'Auto') { '--once' } elseif ($Mode -eq 'Supervise') { '--supervise-only' } else { '--observe' }
$output = (& $pythonPath $watcherPath --config $configPath $argument 2>&1 | Out-String).Trim()
$code = $LASTEXITCODE
$record = [ordered]@{ at_utc = [DateTime]::UtcNow.ToString('o'); mode = $Mode; exit_code = $code; result = $output }
$record | ConvertTo-Json -Compress | Set-Content -LiteralPath (Join-Path $stateDir 'scheduled-last.json') -Encoding UTF8
if ($code -ne 0) {
    $errorPath = Join-Path $stateDir 'scheduled-error.json'
    $previous = if (Test-Path -LiteralPath $errorPath) { Get-Content -LiteralPath $errorPath -Raw | ConvertFrom-Json } else { $null }
    $record | ConvertTo-Json -Compress | Set-Content -LiteralPath $errorPath -Encoding UTF8
    if ($null -eq $previous -or $previous.result -ne $output) {
        & msg.exe $env:USERNAME /TIME:30 'RSNA Vast watcher failed. Check scheduled-error.json.' 2>$null
    }
    exit $code
}
