@echo off
echo ================================================================================
echo                    DDoS Detection API Server
echo ================================================================================
echo.

echo Checking if model files exist...
if not exist "*.joblib" (
    echo ERROR: Model files not found!
    echo Please run training first: run_training.bat
    pause
    exit /b 1
)

echo Starting API server on http://localhost:5000
echo Press Ctrl+C to stop the server
echo.
echo ================================================================================
echo.

python api.py

pause
