param(
    [ValidateSet('Install', 'InstallSupervisor', 'InstallReadOnly', 'Status', 'Uninstall')]
    [string]$Mode = 'Status'
)

$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$configPath = Join-Path $projectRoot '.local\vast-retry-20261008\watcher-config.json'
$watcherPath = Join-Path $projectRoot 'tools\vast_offer_watcher.py'
$pythonPath = Join-Path $projectRoot '.local\vast-retry-20261008\venv\Scripts\python.exe'
$runnerPath = Join-Path $projectRoot 'tools\run_vast_offer_watcher_task.ps1'
$hiddenRunnerPath = Join-Path $projectRoot 'tools\run_vast_offer_watcher_hidden.vbs'
$taskName = 'RSNA-Vast-KRJP-3090-Watcher'

if ($Mode -eq 'Status') {
    Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue | Select-Object TaskName, State
    exit 0
}

$config = Get-Content -LiteralPath $configPath -Raw | ConvertFrom-Json
$statePath = Join-Path $projectRoot '.local\vast-retry-20261008\watcher-state\state.json'

if ($Mode -eq 'InstallReadOnly' -and (Test-Path -LiteralPath $statePath)) {
    $state = Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
    if ($state.status -notin @('watching','destroyed')) {
        throw "A lease may still be billable ($($state.status)); read-only mode would remove its supervisor."
    }
}

if ($Mode -eq 'Uninstall') {
    if (Test-Path -LiteralPath $statePath) {
        $state = Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
        if ($state.status -notin @('watching','destroyed')) {
            throw "A lease may still be billable ($($state.status)). Confirm and stop it before removing supervision."
        }
    }
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction SilentlyContinue
    Write-Output 'Watcher task removed.'
    exit 0
}

if ($Mode -in @('Install', 'InstallSupervisor')) {
    if (-not $config.enabled) { throw 'Watcher config is disabled.' }
    & $pythonPath $watcherPath --config $configPath --check-config
    if ($LASTEXITCODE -ne 0) { throw 'Watcher configuration validation failed.' }
    $runnerMode = if ($Mode -eq 'InstallSupervisor') { 'Supervise' } else { 'Auto' }
} else {
    & $pythonPath $watcherPath --config $configPath --dry-run
    if ($LASTEXITCODE -ne 0) { throw 'Public inventory test failed.' }
    $runnerMode = 'Observe'
}

$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) `
    -RepetitionInterval (New-TimeSpan -Minutes 1) -RepetitionDuration (New-TimeSpan -Days 3650)
$action = New-ScheduledTaskAction -Execute 'wscript.exe' `
    -Argument ('//B //Nologo "{0}" {1}' -f $hiddenRunnerPath, $runnerMode) `
    -WorkingDirectory $projectRoot
$settings = New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 10) -StartWhenAvailable
$principal = New-ScheduledTaskPrincipal -UserId ([Security.Principal.WindowsIdentity]::GetCurrent().Name) `
    -LogonType Interactive -RunLevel Limited
Register-ScheduledTask -TaskName $taskName -Trigger $trigger -Action $action `
    -Settings $settings -Principal $principal -Force | Out-Null
Write-Output "Watcher task installed: $taskName ($runnerMode, every minute while this user is signed in)."
