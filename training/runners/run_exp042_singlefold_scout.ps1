param(
  [string]$Root = ''
)
. (Join-Path $PSScriptRoot "Workspace.ps1")
if (-not $Root) { $Root = Get-KneeWorkspaceRoot }

$ErrorActionPreference='Stop'
$env:PYTHONUTF8='1'; $env:PYTHONIOENCODING='utf-8'
Set-Location $Root
$kernel='taeyangg4/rsna-knee-exp034-adjacent-cache-4x9x288'
$cache=Resolve-KneePath -Root $Root -Path 'cache\EXP034_ADJACENT_4X9X288'
$logs=Resolve-KneePath -Root $Root -Path 'logs\EXP042_SINGLEFOLD_SCOUT'
New-Item -ItemType Directory -Force -Path $cache,$logs | Out-Null

function Alarm([string]$msg) {
  try { [console]::beep(1200,500); Start-Sleep -Milliseconds 200; [console]::beep(1500,700) } catch {}
  try { Add-Type -AssemblyName System.Windows.Forms; [System.Windows.Forms.MessageBox]::Show($msg,'RSNA Knee Training') | Out-Null } catch {}
}

Write-Host '[1/5] Waiting for Kaggle adjacent cache...'
while ($true) {
  $st = (& kaggle kernels status $kernel 2>&1 | Out-String)
  Write-Host ((Get-Date -Format 'yyyy-MM-dd HH:mm:ss') + ' ' + $st.Trim())
  if ($st -match 'COMPLETE') { break }
  if ($st -match 'ERROR|CANCEL') { Alarm 'EXP034 cache failed'; throw $st }
  Start-Sleep -Seconds 60
}

if (-not (Test-Path -LiteralPath (Join-Path $cache 'adjacent_cache_manifest.json'))) {
  Write-Host '[2/5] Downloading cache...'
  & kaggle kernels output $kernel -p $cache
  if ($LASTEXITCODE -ne 0) { Alarm 'Cache download failed'; exit $LASTEXITCODE }
} else { Write-Host '[2/5] Cache already present; skipping download.' }

Write-Host '[3/5] Verifying hashes...'
& python .\training\runners\verify_adjacent_cache.py --cache-dir $cache 2>&1 | Tee-Object -FilePath (Join-Path $logs 'verify_cache.log')
if ($LASTEXITCODE -ne 0) { Alarm 'Cache verification failed'; exit $LASTEXITCODE }

$jobs = @(
  @{Name='convnext_tiny'; Img=288; Batch=4; Out='experiments\EXP042A_CONVNEXT_T_FOLD0'},
  @{Name='vit_small_patch14_dinov2'; Img=280; Batch=4; Out='experiments\EXP042B_DINOV2S_FOLD0'}
)
$i=0
foreach ($j in $jobs) {
  $i++
  $out=Resolve-KneePath -Root $Root -Path $j.Out; New-Item -ItemType Directory -Force -Path $out | Out-Null
  if (Test-Path -LiteralPath (Join-Path $out 'summary.json')) { Write-Host "[4/5] $($j.Name) already complete; skip."; continue }
  $log=Join-Path $logs ($j.Name + '.log')
  Write-Host "[4/5] Training $($j.Name) fold0 20 epochs..."
  & python .\training\src\train_adjacent_2p5d_clean.py --cache-dir $cache --backbone $j.Name --img $j.Img --fold 0 --epochs 20 --batch $j.Batch --workers 0 --head-lr 2e-4 --backbone-lr 2e-5 --wd 0.02 --min-conf 0.5 --seed 20260913 --out $out --resume --save-every 5 2>&1 | Tee-Object -FilePath $log
  if ($LASTEXITCODE -ne 0) { Alarm "$($j.Name) training failed; rerun the same command to resume"; exit $LASTEXITCODE }
}

Write-Host '[5/5] Scout complete.'
Get-Content (Resolve-KneePath -Root $Root -Path "experiments\EXP042A_CONVNEXT_T_FOLD0\summary.json") -ErrorAction SilentlyContinue
Get-Content (Resolve-KneePath -Root $Root -Path "experiments\EXP042B_DINOV2S_FOLD0\summary.json") -ErrorAction SilentlyContinue
Set-Content -LiteralPath (Join-Path $logs 'DONE.txt') -Value (Get-Date -Format o) -Encoding ascii
Alarm 'RSNA EXP042 single-fold scout COMPLETE. Please report back to ChatGPT.'
