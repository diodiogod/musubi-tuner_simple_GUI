@echo off
setlocal
cd /d "%~dp0\.."

if exist "venv\Scripts\python.exe" (
  "venv\Scripts\python.exe" tools\bootstrap_environment.py
  if errorlevel 1 exit /b 1
  exit /b 0
)

python -c "import sys; raise SystemExit(0 if (3, 10) <= sys.version_info[:2] < (3, 13) else 1)" >nul 2>nul
if not errorlevel 1 (
  python tools\bootstrap_environment.py
  if errorlevel 1 exit /b 1
  exit /b 0
)

py -3.12 -c "import sys" >nul 2>nul
if not errorlevel 1 (
  py -3.12 tools\bootstrap_environment.py
  if errorlevel 1 exit /b 1
  exit /b 0
)

py -3.11 -c "import sys" >nul 2>nul
if not errorlevel 1 (
  py -3.11 tools\bootstrap_environment.py
  if errorlevel 1 exit /b 1
  exit /b 0
)

py -3.10 -c "import sys" >nul 2>nul
if not errorlevel 1 (
  py -3.10 tools\bootstrap_environment.py
  if errorlevel 1 exit /b 1
  exit /b 0
)

echo ERROR: Python 3.10, 3.11, or 3.12 was not found.
echo Install a supported Python version and run this launcher again.
exit /b 1
