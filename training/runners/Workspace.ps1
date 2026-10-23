function Get-KneeWorkspaceRoot {
    if ($env:KNEE_WORKSPACE_ROOT) {
        $candidate = [IO.Path]::GetFullPath($env:KNEE_WORKSPACE_ROOT)
    } else {
        $candidate = $PSScriptRoot
    }
    while ($candidate) {
        if (Test-Path -LiteralPath (Join-Path $candidate 'config\workspace.toml')) { return $candidate }
        $parent = Split-Path -Parent $candidate
        if ($parent -eq $candidate) { break }
        $candidate = $parent
    }
    throw 'Workspace config missing; set KNEE_WORKSPACE_ROOT.'
}

function Resolve-KneePath {
    param([string]$Root, [string]$Path)
    if (-not $Root) { $Root = Get-KneeWorkspaceRoot }
    $result = & python (Join-Path $Root 'tools\workspace_path.py') --root $Root $Path
    if ($LASTEXITCODE -ne 0) { throw "Workspace path resolution failed: $Path" }
    return ($result | Out-String).Trim()
}
