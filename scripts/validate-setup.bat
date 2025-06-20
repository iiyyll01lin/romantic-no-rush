@echo off
REM Validation and Setup Script for RomanticNoRush DEAAP
REM This batch file calls the corresponding bash script

REM Check if Git Bash or WSL is available
where bash >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    bash "%~dp0validate-setup.sh"
    exit /b %ERRORLEVEL%
)

REM Try WSL
where wsl >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    wsl bash "%~dp0validate-setup.sh"
    exit /b %ERRORLEVEL%
)

echo Error: No bash interpreter found. Please install Git Bash or WSL.
echo You can download Git Bash from: https://git-scm.com/download/win
pause
exit /b 1
