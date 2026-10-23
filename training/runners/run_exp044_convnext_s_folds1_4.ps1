param([switch]$PreflightOnly)
. (Join-Path $PSScriptRoot "Workspace.ps1")
if (-not $Root) { $Root = Get-KneeWorkspaceRoot }

$ErrorActionPreference='Stop'
$root=(Get-KneeWorkspaceRoot)
$cache=Resolve-KneePath -Root $root -Path 'cache\EXP034_ADJACENT_4X9X288_TEST'
$logDir=Resolve-KneePath -Root $root -Path 'logs\EXP044_CONVNEXT_S_FOLDS1_4'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$success=Join-Path $logDir 'SUCCESS.txt'; $failed=Join-Path $logDir 'FAILED.txt'; Remove-Item -LiteralPath $success,$failed -Force -ErrorAction SilentlyContinue
$env:PYTHONUTF8='1'; $env:PYTHONIOENCODING='utf-8'; $env:HF_HUB_DISABLE_SYMLINKS_WARNING='1'
try {
  "[$(Get-Date -Format o)] WRAPPER BOOT" | Tee-Object -FilePath (Join-Path $logDir 'runner.log') -Append
  Add-Type @"
using System;
using System.Runtime.InteropServices;
public static class AwakeState44 { [DllImport("kernel32.dll", CharSet=CharSet.Auto, SetLastError=true)] public static extern uint SetThreadExecutionState(uint esFlags); }
"@
  $continuous=[Convert]::ToUInt32('80000000',16); $system=[Convert]::ToUInt32('00000001',16); [AwakeState44]::SetThreadExecutionState($continuous -bor $system) | Out-Null
  $manifest=Join-Path $cache 'adjacent_cache_manifest.json'; if(-not(Test-Path -LiteralPath $manifest)){throw "missing cache manifest $manifest"}
  $m=Get-Content -LiteralPath $manifest -Raw | ConvertFrom-Json; if([int]$m.failure_count -ne 0 -or [int]$m.studies -ne 4407 -or $m.shards.Count -ne 8){throw 'cache contract failed'}
  foreach($s in $m.shards){$p=Join-Path $cache $s.file; if(-not(Test-Path -LiteralPath $p)){throw "missing shard $p"}; if((Get-Item -LiteralPath $p).Length -ne [int64]$s.bytes){throw "bad shard size $p"}}
  Set-Location $root
  python -c "import torch,timm; m=timm.create_model('convnext_small',pretrained=True,num_classes=0,global_pool='avg').cuda().eval(); print('PREFLIGHT',torch.cuda.is_available(),torch.cuda.get_device_name(0),m.num_features)"
  if($LASTEXITCODE -ne 0){throw 'CUDA/model preflight failed'}
  if($PreflightOnly){Write-Host 'PREFLIGHT SUCCESS'; exit 0}
  foreach($fold in 1..4){
    $out=".\experiments\EXP044_CONVNEXT_S_FOLD$fold"; $summary=Resolve-KneePath -Root $root -Path ("experiments\EXP044_CONVNEXT_S_FOLD$fold\summary.json")
    if(Test-Path -LiteralPath $summary){Write-Host "[FOLD $fold] summary exists; skipping"; continue}
    $log=Join-Path $logDir ("fold$fold.log"); $seed=20260913+$fold; Write-Host "[FOLD $fold] TRAIN START $(Get-Date -Format o)"
    $oldEap=$ErrorActionPreference; $ErrorActionPreference='Continue'
    & python .\training\src\train_adjacent_2p5d_clean.py --cache-dir $cache --backbone convnext_small --img 288 --fold $fold --epochs 12 --batch 4 --workers 4 --head-lr 0.00015 --backbone-lr 0.0000075 --wd 0.05 --min-conf 0.5 --seed $seed --out $out --augment --patience 4 --min-delta 0.0002 --resume --save-every 3 2>&1 | Tee-Object -FilePath $log -Append
    $code=$LASTEXITCODE; $ErrorActionPreference=$oldEap
    if($code -ne 0){throw "fold $fold training exit code $code"}; if(-not(Test-Path -LiteralPath $summary)){throw "fold $fold ended without summary"}
    Write-Host "[FOLD $fold] COMPLETE $(Get-Date -Format o)"
  }
  $aggLog=Join-Path $logDir 'aggregate.log'; $oldEap=$ErrorActionPreference; $ErrorActionPreference='Continue'; & python .\training\src\summarize_exp044_convnext_s_5fold.py 2>&1 | Tee-Object -FilePath $aggLog -Append; $code=$LASTEXITCODE; $ErrorActionPreference=$oldEap; if($code -ne 0){throw "5-fold aggregation exit code $code"}
  "SUCCESS $(Get-Date -Format o)" | Set-Content -LiteralPath $success -Encoding ASCII
  [console]::Beep(880,220); Start-Sleep -Milliseconds 100; [console]::Beep(1047,220); Start-Sleep -Milliseconds 100; [console]::Beep(1319,350)
  Add-Type -AssemblyName System.Windows.Forms; [System.Windows.Forms.MessageBox]::Show('EXP044 ConvNeXt-S folds 1-4 and 5-fold OOF aggregation completed. Report COMPLETE to ChatGPT.','RSNA Knee - SUCCESS') | Out-Null
} catch {
  $msg=$_.Exception.ToString(); $msg | Set-Content -LiteralPath $failed -Encoding UTF8; Write-Host $msg; [console]::Beep(300,1200)
  try{Add-Type -AssemblyName System.Windows.Forms;[System.Windows.Forms.MessageBox]::Show("EXP044 FAILED.`nCheck FAILED.txt",'RSNA Knee - FAILED')|Out-Null}catch{}; exit 1
} finally { try{if('AwakeState44' -as [type]){[AwakeState44]::SetThreadExecutionState([Convert]::ToUInt32('80000000',16))|Out-Null}}catch{} }
