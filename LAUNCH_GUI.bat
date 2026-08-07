@echo off
setlocal
cd /d "%~dp0"
title Musubi Tuner - Classic GUI

call tools\BOOTSTRAP_WINDOWS.bat
if errorlevel 1 (
  echo.
  echo Setup failed. Read logs\setup.log for details.
  pause
  exit /b 1
)

"venv\Scripts\python.exe" musubi_tuner_gui.py
set "EXIT_CODE=%ERRORLEVEL%"
if not "%EXIT_CODE%"=="0" pause
exit /b %EXIT_CODE%
