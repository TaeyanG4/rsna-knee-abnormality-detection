# HB151 Vast seed43 transfer v1.0.0. Run from the project root; no automatic fit.
param([Parameter(Mandatory=$true)][ValidateSet('Copy','Pull')][string]$Action)
$ErrorActionPreference = 'Stop'
$work = Join-Path (Split-Path $PSScriptRoot -Parent) '.local\scratch\slot-b\H-B-156'
$remote = 'root@<remote-host>'
function Check-Exit { if ($LASTEXITCODE -ne 0) { throw "Command failed: $LASTEXITCODE" } }
if ($Action -eq 'Copy') {
    $archive = Join-Path $work 'hb156-convnext-s42-v1.tar'
    $pin = Join-Path $work 'hb156-convnext-s42-v1.tar.sha256'
    $expected = (Get-Content -LiteralPath $pin -Raw).Split(' ')[0].Trim()
    if ((Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLower() -ne $expected) { throw 'Archive SHA256 mismatch' }
    & scp <remote-host> $archive $pin "${remote}:/workspace/"
    Check-Exit
    & ssh -p 19960 $remote 'cd /workspace && sha256sum -c hb156-convnext-s42-v1.tar.sha256 && test ! -e hb156-convnext-s42-v1 && tar -xf hb156-convnext-s42-v1.tar && cd hb156-convnext-s42-v1 && sha256sum -c SHA256SUMS'
    Check-Exit
} else {
    $dest = Join-Path $work 'pulled'
    New-Item -ItemType Directory -Force -Path $dest | Out-Null
    $stem = 'exp143-v2-e14-e11-e12-bn'
    foreach ($ext in @('pt','sha256','json')) {
        & scp <remote-host> "${remote}:/workspace/hb156-convnext-s42-v1/stage-convnext-s42-v1/export/${stem}.${ext}" $dest
        Check-Exit
    }
    $expected = (Get-Content -LiteralPath (Join-Path $dest "$stem.sha256") -Raw).Split(' ')[0].Trim()
    $actual = (Get-FileHash -LiteralPath (Join-Path $dest "$stem.pt") -Algorithm SHA256).Hash.ToLower()
    $meta = Get-Content -LiteralPath (Join-Path $dest "$stem.json") -Raw | ConvertFrom-Json
    $protocol = (Get-FileHash -LiteralPath (Join-Path $work 'package\stage-convnext-s42-v1\manifest.json') -Algorithm SHA256).Hash.ToLower()
    if ($actual -ne $expected -or $actual -ne $meta.sha256 -or $meta.protocol_sha256 -ne $protocol -or $meta.clean_oof -ne $false -or $meta.exposure -ne 'EXPOSED') { throw 'Export SHA256/seed43 provenance mismatch' }
    Write-Output "VERIFIED $actual $dest\$stem.pt"
}
