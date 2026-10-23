param(
    [ValidateSet('rad5','coat5rad5')][string]$Own = 'rad5',
    [switch]$PlaneFlipTta,
    [string]$Out = '.local/exp126-anchor-blend/rad5-v8-r9',
    [double]$AnchorSeconds = 26404
)
$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$outPath = [System.IO.Path]::GetFullPath((Join-Path $root $Out))
if (Test-Path -LiteralPath $outPath) { throw "Fresh output directory required: $outPath" }
$utf8 = New-Object System.Text.UTF8Encoding($false)
function ReadText($p) { [System.IO.File]::ReadAllText($p, [System.Text.Encoding]::UTF8) }
function WriteText($p,$s) { [System.IO.File]::WriteAllText($p,$s,$utf8) }
function Sha($p) { (Get-FileHash -LiteralPath $p -Algorithm SHA256).Hash.ToLowerInvariant() }
function J($x) { ConvertTo-Json -InputObject $x -Depth 100 -Compress }
$baseDir = Join-Path $root '.local/hb15-combined/knee-exp105-anchor-own-v1'
$baseMeta = (ReadText (Join-Path $baseDir 'kernel-metadata.json') | ConvertFrom-Json)
$baseNotebook = (ReadText (Join-Path $baseDir $baseMeta.code_file) | ConvertFrom-Json)
$baseManifestPath = Join-Path $root '.local/exp119-release-v7/coat5rad5/release-manifest.json'
$manifest = ReadText $baseManifestPath | ConvertFrom-Json
if ($manifest.schema -ne 'exp119-test-release-v7' -or (Sha (Join-Path $root 'tools/exp119_release_v7.py')) -ne $manifest.release_source_sha256) {
    throw 'Pinned v7 source/manifest changed'
}
$names = if ($Own -eq 'rad5') { @('rad5') } else { @('coat5','rad5') }
$manifest.arms = @($manifest.arms | Where-Object { $names -contains $_.name })
if ($manifest.arms.Count -ne $names.Count) { throw 'Missing selected own arm' }
foreach ($a in $manifest.arms) { $a.weight = 1.0 }
if ($Own -eq 'rad5') { $manifest.arm_assets.PSObject.Properties.Remove('coat5') }
$manifest.arm_specs = @($manifest.arm_specs | Where-Object { $names -contains $_.Split('=')[0] })
$manifest.combination = if ($names.Count -eq 1) { 'single arm: raw mean_prob probabilities' } else {
    'per-target mean of average-tie percentile ranks, equal arm weights (EXP098)' }
$manifest.rank_weights = New-Object PSObject
foreach ($name in $names) { $manifest.rank_weights | Add-Member -NotePropertyName $name -NotePropertyValue (1.0 / $names.Count) }
$manifest.release = 'exp126-' + $Own + $(if ($PlaneFlipTta) { '-tta' } else { '' })
New-Item -ItemType Directory -Path $outPath | Out-Null
$manifestPath = Join-Path $outPath 'exp126-own-manifest.json'
WriteText $manifestPath (((ConvertTo-Json -InputObject $manifest -Depth 100).Replace("`r`n", "`n")) + "`n")
$manifestSha = Sha $manifestPath
$config = [ordered]@{own=$Own;plane_flip_tta=[bool]$PlaneFlipTta;manifest_sha256=$manifestSha;
    anchor_projection_seconds=$AnchorSeconds;final_weights=@(0.7,0.3)}
$configPath = Join-Path $outPath 'exp126-config.json'
WriteText $configPath ((J $config) + "`n")
$setup = $baseNotebook.cells[0].source -join ''
$insert = New-Object System.Collections.Generic.List[string]
foreach ($item in @(
        @('exp119_release_v7.py', (Join-Path $root 'tools/exp119_release_v7.py')),
        @('exp111_release_v6.py', (Join-Path $root 'tools/exp111_release_v6.py')),
        @('exp126_anchor_blend_v8.py', (Join-Path $root 'tools/exp126_anchor_blend_v8.py')),
        @('exp126-own-manifest.json', $manifestPath),
        @('exp126-config.json', $configPath))) {
    $name = $item[0]; $path = $item[1]; $raw = ReadText $path
    if (-not $raw.EndsWith("`n") -or $raw.Contains("`r")) { throw "Embedded source must end LF only: $name" }
    $lines = $raw.Substring(0,$raw.Length-1).Split([char]10)
    $insert.Add("_HB15_SOURCES[" + (J $name) + "] = (" + (J (Sha $path)) + ", [")
    foreach ($line in $lines) { $insert.Add(' ' + (J $line) + ',') }
    $insert.Add('])')
}
$marker = "`n`ndef _hb15_setup():"
if (-not $setup.Contains($marker)) { throw 'EXP105 setup insertion point changed' }
$setup = $setup.Replace($marker, "`n" + ($insert -join "`n") + $marker)
$baseNotebook.cells[0].source = @($setup -split '(?<=\n)') | Where-Object { $_ -ne '' }
$post = @'
import importlib.util
from pathlib import Path
try:
    _spec126 = importlib.util.spec_from_file_location('exp126', Path(_HB15_SETUP['src']) / 'exp126_anchor_blend_v8.py')
    _module126 = importlib.util.module_from_spec(_spec126)
    _spec126.loader.exec_module(_module126)
    _EXP126 = _module126.post_anchor(globals())
    print(_EXP126['path'], _EXP126['reason'])
except BaseException as _exc126:
    print('EXP126 anchor-only:', type(_exc126).__name__, str(_exc126))
'@
$baseNotebook.cells[-1].source = @($post.Replace("`r",'') -split '(?<=\n)') | Where-Object { $_ -ne '' }
$slug = 'knee-exp126-' + $Own + $(if ($PlaneFlipTta) { '-tta' } else { '' }) + '-v8'
$baseMeta.id = 'taeyangg4/' + $slug
$baseMeta.title = $slug
$baseMeta.code_file = 'release.ipynb'
$baseMeta.is_private = $true
$baseMeta.enable_internet = $false
$baseMeta.dataset_sources = @($baseMeta.dataset_sources + @('taeyangg4/knee-release-assets-v8') | Select-Object -Unique)
$notebookPath = Join-Path $outPath 'release.ipynb'
$metadataPath = Join-Path $outPath 'kernel-metadata.json'
WriteText $notebookPath ((J $baseNotebook) + "`n")
WriteText $metadataPath ((J $baseMeta) + "`n")
$n = 1300.0
$rad = 1.3 * 5 * 0.25 * $n * $(if ($PlaneFlipTta) { 2 } else { 1 })
$coat = if ($Own -eq 'coat5rad5') { 1.3 * 5 * 0.35 * $n } else { 0 }
$total = $AnchorSeconds + $rad + $coat + 600
$runtime = [ordered]@{studies=1300;own=$Own;plane_flip_tta=[bool]$PlaneFlipTta;anchor_scenario_seconds=$AnchorSeconds;
    own_model_seconds=[ordered]@{rad5=$rad;coat5=$coat};fallback_decode_seconds=0;fixed_seconds=600;
    total_seconds=$total;margin_seconds=(32400-$total);minimum_margin_seconds=3600;
    fits_one_hour_margin=($total -le 28800);
    scope='Arithmetic screen scenario; actual launch checks elapsed anchor and usable store count';
    gpu_schedule='Anchor unchanged on GPU0+GPU1; then one v7 shared worker with all own arms on GPU0; GPU1 idle'}
$plan = [ordered]@{schema='exp126-anchor-blend-v8';state='STAGED_ONLY_NOT_PUSHED';kernel=$baseMeta.id;
    anchor_notebook_sha256=(Sha (Join-Path $root 'kaggle/packages/KNEE_ANCHOR_0943/rsna-knee-fast-2xt4-inference.ipynb'));
    anchor_cells_unchanged=$true;manifest_sha256=$manifestSha;notebook_sha256=(Sha $notebookPath);
    metadata_sha256=(Sha $metadataPath);runtime=$runtime;
    generated_by='tools/build_exp126_fallback.ps1 (Python launcher unavailable)'}
WriteText (Join-Path $outPath 'execution-plan.json') ((J $plan) + "`n")
Write-Output ("STAGED_ONLY_NOT_PUSHED $($baseMeta.id) margin_seconds=$($runtime.margin_seconds) $outPath")
