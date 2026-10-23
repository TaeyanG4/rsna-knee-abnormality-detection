param([switch]$PreflightOnly)
. (Join-Path $PSScriptRoot "Workspace.ps1")
if (-not $Root) { $Root = Get-KneeWorkspaceRoot }

$ErrorActionPreference='Stop'
$root=(Get-KneeWorkspaceRoot)
$cache=Resolve-KneePath -Root $root -Path 'cache\EXP034_ADJACENT_4X9X288_TEST'
$logDir=Resolve-KneePath -Root $root -Path 'logs\EXP050_CONVNEXT_BASE_CENTER9_FOLD0'
$out=Resolve-KneePath -Root $root -Path 'experiments\EXP050_CONVNEXT_BASE_CENTER9_FOLD0'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$success=Join-Path $logDir 'SUCCESS.txt'; $failed=Join-Path $logDir 'FAILED.txt'
Remove-Item -LiteralPath $success,$failed -Force -ErrorAction SilentlyContinue
$env:PYTHONUTF8='1'; $env:PYTHONIOENCODING='utf-8'; $env:HF_HUB_DISABLE_SYMLINKS_WARNING='1'
try {
  "[$(Get-Date -Format o)] WRAPPER BOOT" | Tee-Object -FilePath (Join-Path $logDir 'runner.log') -Append
  Add-Type @"
using System;
using System.Runtime.InteropServices;
public static class AwakeState50 { [DllImport("kernel32.dll", CharSet=CharSet.Auto, SetLastError=true)] public static extern uint SetThreadExecutionState(uint esFlags); }
"@
  $continuous=[Convert]::ToUInt32('80000000',16); $system=[Convert]::ToUInt32('00000001',16); [AwakeState50]::SetThreadExecutionState($continuous -bor $system) | Out-Null
  $manifest=Join-Path $cache 'adjacent_cache_manifest.json'; if(-not(Test-Path -LiteralPath $manifest)){throw "missing cache manifest: $manifest"}
  $m=Get-Content -LiteralPath $manifest -Raw | ConvertFrom-Json; if([int]$m.failure_count -ne 0 -or [int]$m.studies -ne 4407 -or $m.shards.Count -ne 8){throw 'cache contract failed'}
  Set-Location $root
  python -c "import torch,timm; print('PREFLIGHT',torch.cuda.is_available(),torch.cuda.get_device_name(0),timm.__version__); m=timm.create_model('convnext_base',pretrained=True,num_classes=0,global_pool='avg'); print('BASE_OK',m.num_features)"
  if($LASTEXITCODE -ne 0){throw 'preflight failed'}
  if($PreflightOnly){Write-Host 'PREFLIGHT SUCCESS'; exit 0}
  Write-Host "[EXP050] TRAIN START $(Get-Date -Format o)"
  $log=Join-Path $logDir 'train.log'; $old=$ErrorActionPreference; $ErrorActionPreference='Continue'
  & python .\training\src\train_adjacent_2p5d_clean.py --cache-dir $cache --backbone convnext_base --img 288 --fold 0 --epochs 10 --batch 2 --workers 4 --head-lr 0.00010 --backbone-lr 0.000005 --wd 0.05 --min-conf 0.5 --seed 20260914 --out $out --augment --patience 4 --min-delta 0.0002 --resume --save-every 2 2>&1 | Tee-Object -FilePath $log -Append
  $code=$LASTEXITCODE; $ErrorActionPreference=$old
  if($code -ne 0){throw "training exit code $code"}
  $summary=Join-Path $out 'summary.json'; if(-not(Test-Path -LiteralPath $summary)){throw 'missing summary.json'}
  "SUCCESS $(Get-Date -Format o)" | Set-Content -LiteralPath $success -Encoding ASCII
  [console]::Beep(880,220); Start-Sleep -Milliseconds 100; [console]::Beep(1047,220); Start-Sleep -Milliseconds 100; [console]::Beep(1319,350)
  Add-Type -AssemblyName System.Windows.Forms; [System.Windows.Forms.MessageBox]::Show('EXP050 ConvNeXt-Base fold0 completed. Report COMPLETE to ChatGPT.','RSNA Knee - SUCCESS') | Out-Null
}
catch {
  $msg=$_.Exception.ToString(); [IO.File]::WriteAllText($failed,$msg,(New-Object Text.UTF8Encoding($false))); Write-Host $msg; [console]::Beep(300,1200)
  try { Add-Type -AssemblyName System.Windows.Forms; [System.Windows.Forms.MessageBox]::Show("EXP050 FAILED.`nCheck FAILED.txt",'RSNA Knee - FAILED') | Out-Null } catch {}
  exit 1
}
finally { try { if('AwakeState50' -as [type]){[AwakeState50]::SetThreadExecutionState([Convert]::ToUInt32('80000000',16)) | Out-Null} } catch {} }