@echo off
echo ================================================================================
echo                    Installing Dependencies
echo ================================================================================
echo.

echo Installing Python packages from requirements.txt...
echo This may take a few minutes...
echo.

pip install -r requirements.txt

if errorlevel 1 (
    echo.
    echo ERROR: Installation failed!
    echo Please check your internet connection and try again.
    pause
    exit /b 1
)

echo.
echo ================================================================================
echo Installation completed successfully!
echo ================================================================================
echo.
echo Next step: Run setup verification with: python test_setup.py
echo Or use: run_training.bat to start training
echo.
pause
