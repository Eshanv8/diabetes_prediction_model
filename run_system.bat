@echo off
echo ========================================
echo  Advanced Diabetes Prediction System
echo ========================================
echo.
echo Step 1: Training Advanced Models...
echo This will take 2-5 minutes. Please wait...
echo.

python train_advanced_model.py

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ========================================
    echo  Training Complete!
    echo ========================================
    echo.
    echo Step 2: Launching GUI...
    echo.
    python gui_advanced.py
) else (
    echo.
    echo ========================================
    echo  Training Failed!
    echo ========================================
    echo Please check the error messages above.
    pause
)
