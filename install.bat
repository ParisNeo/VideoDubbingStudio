@echo off
REM VoiceDub Pro - Windows Installation Script (Batch Wrapper)
REM This simply calls the PowerShell script with proper execution policy

setlocal EnableDelayedExpansion

echo VoiceDub Pro Installer for Windows
echo ===================================
echo.

REM Check if PowerShell is available
powershell -Command "Get-Host" >nul 2>&1
if errorlevel 1 (
    echo ERROR: PowerShell is not available on this system.
    echo Please install PowerShell or run install.ps1 manually.
    exit /b 1
)

REM Run the PowerShell installer with bypass execution policy
echo Starting PowerShell installer...
powershell -ExecutionPolicy Bypass -File "install.ps1" %*

if errorlevel 1 (
    echo.
    echo Installation failed. See errors above.
    exit /b 1
)

echo.
echo Installation script completed.
pause
