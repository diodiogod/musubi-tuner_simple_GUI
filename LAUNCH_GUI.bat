@echo off
TITLE Musubi Tuner GUI Launcher

REM Get the directory where the batch file is located
SET "BATCH_DIR=%~dp0"

ECHO Looking for virtual environment in: "%BATCH_DIR%venv"
ECHO.

REM Check if the activate script exists
IF NOT EXIST "%BATCH_DIR%venv\Scripts\activate.bat" (
    ECHO ERROR: Virtual environment not found.
    ECHO Please make sure the 'venv' folder exists in the same directory as this script.
    ECHO.
    PAUSE
    EXIT /B 1
)

ECHO --- Activating Virtual Environment ---
CALL "%BATCH_DIR%venv\Scripts\activate.bat"
ECHO.

ECHO --- Starting Musubi Tuner GUI ---
ECHO Please wait for the application to load...
ECHO.
python "%BATCH_DIR%musubi_tuner_gui.py"
SET "EXIT_CODE=%ERRORLEVEL%"

IF NOT "%EXIT_CODE%"=="0" (
    ECHO.
    ECHO --- GUI exited with error code %EXIT_CODE%. ---
    PAUSE
)

EXIT /B %EXIT_CODE%
