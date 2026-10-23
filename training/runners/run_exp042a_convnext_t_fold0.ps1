param([switch]$PreflightOnly)
. (Join-Path $PSScriptRoot "Workspace.ps1")
if (-not $Root) { $Root = Get-KneeWorkspaceRoot }

$ErrorActionPreference = 'Stop'
$root = (Get-KneeWorkspaceRoot)
$cache = Resolve-KneePath -Root $root -Path 'cache\EXP034_ADJACENT_4X9X288_TEST'
$out = Resolve-KneePath -Root $root -Path 'experiments\EXP042A_CONVNEXT_T_FOLD0'
$logDir = Resolve-KneePath -Root $root -Path 'logs\EXP042A_CONVNEXT_T_FOLD0'
New-Item -ItemType Directory -Force -Path $out,$logDir | Out-Null
$log = Join-Path $logDir 'train.log'
$success = Join-Path $logDir 'SUCCESS.txt'
$failed = Join-Path $logDir 'FAILED.txt'
Remove-Item -LiteralPath $success,$failed -Force -ErrorAction SilentlyContinue
$env:PYTHONUTF8='1'
$env:PYTHONIOENCODING='utf-8'
$env:HF_HUB_DISABLE_SYMLINKS_WARNING='1'
"[$(Get-Date -Format o)] WRAPPER BOOT" | Tee-Object -FilePath $log -Append
try {
  Add-Type @"
using System;
using System.Runtime.InteropServices;
public static class AwakeState {
  [DllImport("kernel32.dll", CharSet = CharSet.Auto, SetLastError = true)]
  public static extern uint SetThreadExecutionState(uint esFlags);
}
"@ -ErrorAction SilentlyContinue
  $esContinuous = [Convert]::ToUInt32('80000000',16)
  $esSystemRequired = [Convert]::ToUInt32('00000001',16)
  [AwakeState]::SetThreadExecutionState($esContinuous -bor $esSystemRequired) | Out-Null
  Set-Location $root
  $manifest = Join-Path $cache 'adjacent_cache_manifest.json'
  if (-not (Test-Path -LiteralPath $manifest)) { throw "Cache manifest missing: $manifest" }
  $m = Get-Content -LiteralPath $manifest -Raw | ConvertFrom-Json
  if ([int]$m.failure_count -ne 0 -or [int]$m.studies -ne 4407 -or $m.shards.Count -ne 8) { throw 'Cache manifest contract check failed' }
  foreach ($s in $m.shards) {
    $cp = Join-Path $cache $s.file
    if (-not (Test-Path -LiteralPath $cp)) { throw "Missing cache shard: $cp" }
    if ((Get-Item -LiteralPath $cp).Length -ne [int64]$s.bytes) { throw "Cache shard size mismatch: $cp" }
  }
  "[$(Get-Date -Format o)] CACHE CONTRACT PASS" | Tee-Object -FilePath $log -Append
  if ($PreflightOnly) {
    & python -c "import torch,timm; assert torch.cuda.is_available(); m=timm.create_model('convnext_tiny',pretrained=True,num_classes=0,global_pool='avg').cuda().eval(); x=torch.zeros(1,3,288,288,device='cuda'); y=m(x); print('PREFLIGHT_OK',torch.cuda.get_device_name(0),tuple(y.shape))" 2>&1 | Tee-Object -FilePath $log -Append
    if ($LASTEXITCODE -ne 0) { throw "Preflight Python exit code $LASTEXITCODE" }
    "[$(Get-Date -Format o)] PREFLIGHT SUCCESS" | Tee-Object -FilePath $log -Append
    exit 0
  }
  "[$(Get-Date -Format o)] EXP042A TRAIN START" | Tee-Object -FilePath $log -Append
  & python .\training\src\train_adjacent_2p5d_clean.py `
    --cache-dir $cache `
    --backbone convnext_tiny `
    --img 288 `
    --fold 0 `
    --epochs 20 `
    --batch 4 `
    --workers 4 `
    --head-lr 0.0002 `
    --backbone-lr 0.00002 `
    --wd 0.02 `
    --min-conf 0.5 `
    --seed 20260913 `
    --out $out `
    --resume `
    --save-every 5 2>&1 | Tee-Object -FilePath $log -Append
  $code = $LASTEXITCODE
  if ($code -ne 0) { throw "Training process exit code $code" }
  $summary = Join-Path $out 'summary.json'
  if (-not (Test-Path -LiteralPath $summary)) { throw 'Training exited but summary.json is missing' }
  "SUCCESS $(Get-Date -Format o)" | Set-Content -LiteralPath $success -Encoding ASCII
  [console]::Beep(880,220); Start-Sleep -Milliseconds 100
  [console]::Beep(1047,220); Start-Sleep -Milliseconds 100
  [console]::Beep(1319,350)
  Add-Type -AssemblyName System.Windows.Forms
  [System.Windows.Forms.MessageBox]::Show('EXP042A ConvNeXt-T fold0 training completed successfully. Report COMPLETE to ChatGPT.','RSNA Knee - SUCCESS') | Out-Null
}
catch {
  $msg = $_.Exception.Message
  "FAILED $(Get-Date -Format o) $msg" | Set-Content -LiteralPath $failed -Encoding UTF8
  "[$(Get-Date -Format o)] FAILED: $msg" | Tee-Object -FilePath $log -Append
  [console]::Beep(300,1200)
  try {
    Add-Type -AssemblyName System.Windows.Forms
    [System.Windows.Forms.MessageBox]::Show("EXP042A FAILED.`n$msg`nCheck: $log",'RSNA Knee - FAILED') | Out-Null
  } catch {}
  exit 1
}
finally {
  try {
    $esContinuous = [Convert]::ToUInt32('80000000',16)
    [AwakeState]::SetThreadExecutionState($esContinuous) | Out-Null
  } catch {}
}
