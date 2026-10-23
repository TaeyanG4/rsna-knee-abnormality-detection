@echo off
setlocal
set "KNEE_WORKSPACE_ROOT=%~dp0"
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo KneeOps is not installed in this folder.
  pause
  exit /b 1
)
echo Stopping KneeOps. Active downloads will pause; partial files are kept.
".venv\Scripts\python.exe" -X utf8 -m monitoring.kneeops.desktop stop
if errorlevel 1 pause
