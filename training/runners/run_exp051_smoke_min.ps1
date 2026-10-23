. (Join-Path $PSScriptRoot "Workspace.ps1")
Set-Location (Get-KneeWorkspaceRoot) 
$mask = python -c 'from training.src.train_adjacent_2p5d_clean import TARGETS;print(chr(44).join(TARGETS[i] for i in (3,5,6,8)))' 
