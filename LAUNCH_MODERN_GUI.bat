@echo off
setlocal
cd /d "%~dp0"
if exist "venv\Scripts\python.exe" (
  "venv\Scripts\python.exe" -m modern_gui.server
) else (
  python -m modern_gui.server
)
if errorlevel 1 pause
