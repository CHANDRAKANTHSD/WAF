@echo off
echo ================================================================================
echo                    DDoS Detection System - Training
echo ================================================================================
echo.

echo Step 1: Checking setup...
python test_setup.py
if errorlevel 1 (
    echo.
    echo ERROR: Setup verification failed!
    echo Please fix the issues above before training.
    pause
    exit /b 1
)

echo.
echo ================================================================================
echo Step 2: Starting model training...
echo This may take 10-30 minutes depending on your hardware.
echo ================================================================================
echo.

python ddos_detection.py

if errorlevel 1 (
    echo.
    echo ERROR: Training failed!
    pause
    exit /b 1
)

echo.
echo ================================================================================
echo Training completed successfully!
echo ================================================================================
echo.
echo Model files have been saved.
echo You can now run the API server with: run_api.bat
echo.
pause
