# H-B-156 L2 seed43 transfer v3.0.0. Changelog: isolated seed43 upload, deployment and Pull paths.
param([Parameter(Mandatory=$true)][ValidateSet('Copy','Pull')][string]$Action)
$ErrorActionPreference = 'Stop'
$work = Join-Path (Split-Path $PSScriptRoot -Parent) '.local\scratch\hb156-l2-s43\H-B-156-L2-S43'
$remote = 'root@<remote-host>'
function Check-Exit { if ($LASTEXITCODE -ne 0) { throw "Command failed: $LASTEXITCODE" } }
if ($Action -eq 'Copy') {
    $archive = Join-Path $work 'hb156-convnext-l2-s43-v1.tar'
    $pin = Join-Path $work 'hb156-convnext-l2-s43-v1.tar.sha256'
    $expected = (Get-Content -LiteralPath $pin -Raw).Split(' ')[0].Trim()
    if ((Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLower() -ne $expected) { throw 'Archive SHA256 mismatch' }
    # Read-only prerequisites first. No corpus upload or access to another run's output.
    & ssh -p 19960 $remote 'test ! -e /workspace/hb156-convnext-l2-s43-v1 && cd /workspace/cache/EXP143_RAPTOR96_384_C140_S02 && echo "9c521caa39c51cee0834176c980be63ffeaf4d22b072215dd74dc6e0c97c8b4c  manifest.json" | sha256sum -c - && echo "137ed33c53858ee785cebc7ded0a6511cbb2e309705d54992baea07f646a00cf  build-registration.json" | sha256sum -c -'
    Check-Exit
    & ssh -p 19960 $remote 'mkdir /workspace/hb156-l2-s43-upload-v1'
    Check-Exit
    & scp <remote-host> $archive $pin "${remote}:/workspace/hb156-l2-s43-upload-v1/"
    Check-Exit
    & ssh -p 19960 $remote 'cd /workspace/hb156-l2-s43-upload-v1 && sha256sum -c hb156-convnext-l2-s43-v1.tar.sha256 && mkdir /workspace/hb156-convnext-l2-s43-v1 && tar -xf hb156-convnext-l2-s43-v1.tar -C /workspace/hb156-convnext-l2-s43-v1 --strip-components=1 && cd /workspace/hb156-convnext-l2-s43-v1 && sha256sum -c SHA256SUMS'
    Check-Exit
} else {
    $dest = Join-Path $work 'pulled'
    New-Item -ItemType Directory -Force -Path $dest | Out-Null
    $stem = 'exp143-v2-e14-e11-e12-bn'
    foreach ($ext in @('pt','sha256','json')) {
        & scp <remote-host> "${remote}:/workspace/hb156-convnext-l2-s43-v1/stage-convnext-l2-s43-v1/export/${stem}.${ext}" $dest
        Check-Exit
    }
    $expected = (Get-Content -LiteralPath (Join-Path $dest "$stem.sha256") -Raw).Split(' ')[0].Trim()
    $actual = (Get-FileHash -LiteralPath (Join-Path $dest "$stem.pt") -Algorithm SHA256).Hash.ToLower()
    $meta = Get-Content -LiteralPath (Join-Path $dest "$stem.json") -Raw | ConvertFrom-Json
    $protocol = (Get-FileHash -LiteralPath (Join-Path $work 'package\stage-convnext-l2-s43-v1\manifest.json') -Algorithm SHA256).Hash.ToLower()
    if ($actual -ne $expected -or $actual -ne $meta.sha256 -or $meta.protocol_sha256 -ne $protocol -or $meta.clean_oof -ne $false -or $meta.exposure -ne 'EXPOSED') { throw 'Export SHA256/L2 provenance mismatch' }
    Write-Output "VERIFIED $actual $dest\$stem.pt"
}
