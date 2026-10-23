param([switch]$PreflightOnly)
. (Join-Path $PSScriptRoot "Workspace.ps1")
if (-not $Root) { $Root = Get-KneeWorkspaceRoot }

$ErrorActionPreference='Stop'
$root=(Get-KneeWorkspaceRoot)
$cache=Resolve-KneePath -Root $root -Path 'cache\EXP034_ADJACENT_4X9X288_TEST'
$logDir=Resolve-KneePath -Root $root -Path 'logs\EXP043_CONVNEXT_REG_SCOUT'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$success=Join-Path $logDir 'SUCCESS.txt'
$failed=Join-Path $logDir 'FAILED.txt'
Remove-Item -LiteralPath $success,$failed -Force -ErrorAction SilentlyContinue
$env:PYTHONUTF8='1'; $env:PYTHONIOENCODING='utf-8'; $env:HF_HUB_DISABLE_SYMLINKS_WARNING='1'
try {
  "[$(Get-Date -Format o)] WRAPPER BOOT" | Tee-Object -FilePath (Join-Path $logDir 'runner.log') -Append
  Add-Type @"
using System;
using System.Runtime.InteropServices;
public static class AwakeState2 {
 [DllImport("kernel32.dll", CharSet=CharSet.Auto, SetLastError=true)]
 public static extern uint SetThreadExecutionState(uint esFlags);
}
"@
  $continuous=[Convert]::ToUInt32('80000000',16); $system=[Convert]::ToUInt32('00000001',16)
  [AwakeState2]::SetThreadExecutionState($continuous -bor $system) | Out-Null
  $manifest=Join-Path $cache 'adjacent_cache_manifest.json'
  if(-not (Test-Path -LiteralPath $manifest)){throw "missing cache manifest: $manifest"}
  $m=Get-Content -LiteralPath $manifest -Raw | ConvertFrom-Json
  if([int]$m.failure_count -ne 0 -or [int]$m.studies -ne 4407 -or $m.shards.Count -ne 8){throw 'cache contract failed'}
  foreach($s in $m.shards){$p=Join-Path $cache $s.file; if(-not(Test-Path -LiteralPath $p)){throw "missing shard $p"}; if((Get-Item -LiteralPath $p).Length -ne [int64]$s.bytes){throw "bad shard size $p"}}
  Set-Location $root
  python -c "import torch; print('PREFLIGHT',torch.cuda.is_available(),torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO_GPU')"
  if($LASTEXITCODE -ne 0){throw 'python/cuda preflight failed'}
  if($PreflightOnly){Write-Host 'PREFLIGHT SUCCESS'; exit 0}

  $jobs=@(
    @{Name='TINY'; Out='.\experiments\EXP043A_CONVNEXT_T_REG_FOLD0'; Backbone='convnext_tiny'; Batch='4'; HeadLR='0.00015'; BBLR='0.0000075'},
    @{Name='SMALL'; Out='.\experiments\EXP043B_CONVNEXT_S_REG_FOLD0'; Backbone='convnext_small'; Batch='4'; HeadLR='0.00015'; BBLR='0.0000075'}
  )
  foreach($j in $jobs){
    $summary=Resolve-KneePath -Root $root -Path (($j.Out.TrimStart('.\'))+'\summary.json')
    if(Test-Path -LiteralPath $summary){Write-Host "[$($j.Name)] summary exists; skipping"; continue}
    $log=Join-Path $logDir ("train_"+$j.Name.ToLower()+'.log')
    Write-Host "[$($j.Name)] TRAIN START $(Get-Date -Format o)"
    $oldEap=$ErrorActionPreference; $ErrorActionPreference='Continue'
    & python .\training\src\train_adjacent_2p5d_clean.py --cache-dir $cache --backbone $j.Backbone --img 288 --fold 0 --epochs 12 --batch $j.Batch --workers 4 --head-lr $j.HeadLR --backbone-lr $j.BBLR --wd 0.05 --min-conf 0.5 --seed 20260913 --out $j.Out --augment --patience 4 --min-delta 0.0002 --resume --save-every 3 2>&1 | Tee-Object -FilePath $log -Append
    $code=$LASTEXITCODE; $ErrorActionPreference=$oldEap
    if($code -ne 0){throw "$($j.Name) training exit code $code"}
    if(-not(Test-Path -LiteralPath $summary)){throw "$($j.Name) ended without summary.json"}
    Write-Host "[$($j.Name)] COMPLETE $(Get-Date -Format o)"
  }
  "SUCCESS $(Get-Date -Format o)" | Set-Content -LiteralPath $success -Encoding ASCII
  [console]::Beep(880,220); Start-Sleep -Milliseconds 100; [console]::Beep(1047,220); Start-Sleep -Milliseconds 100; [console]::Beep(1319,350)
  Add-Type -AssemblyName System.Windows.Forms
  [System.Windows.Forms.MessageBox]::Show('EXP043 ConvNeXt regularized scout completed. Report COMPLETE to ChatGPT.','RSNA Knee - SUCCESS') | Out-Null
}
catch {
  $msg=$_.Exception.ToString(); $msg | Set-Content -LiteralPath $failed -Encoding UTF8; Write-Host $msg
  [console]::Beep(300,1200)
  try { Add-Type -AssemblyName System.Windows.Forms; [System.Windows.Forms.MessageBox]::Show("EXP043 FAILED.`nCheck FAILED.txt",'RSNA Knee - FAILED') | Out-Null } catch {}
  exit 1
}
finally {
  try { if('AwakeState2' -as [type]){[AwakeState2]::SetThreadExecutionState([Convert]::ToUInt32('80000000',16)) | Out-Null} } catch {}
}
