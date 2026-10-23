@echo off
setlocal
set "KNEE_WORKSPACE_ROOT=%~dp0"
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo First run: powershell -ExecutionPolicy Bypass -File tools\Setup-KneeOps.ps1
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -X utf8 -m monitoring.kneeops.desktop start
if errorlevel 1 pause
