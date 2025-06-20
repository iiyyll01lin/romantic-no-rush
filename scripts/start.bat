@echo off
echo ===============================================
echo    RomanticNoRush DEAAP - Windows Startup
echo ===============================================
echo.

REM Check Docker Desktop
docker info >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Error: Docker Desktop is not running or not installed.
    echo Please start Docker Desktop and try again.
    echo.
    pause
    exit /b 1
)

echo Docker Desktop is running...
echo.

REM Ask user for startup options
echo Startup Options:
echo 1. Quick start (default)
echo 2. Full validation + start
echo 3. Development mode
echo 4. Production mode
echo.
set /p "choice=Choose option (1-4, default=1): "

if "%choice%"=="" set choice=1

if "%choice%"=="1" (
    echo Starting with default settings...
    call "%~dp0startup.bat"
) else if "%choice%"=="2" (
    echo Running validation first...
    call "%~dp0validate-setup.bat"
    if %ERRORLEVEL% EQU 0 (
        call "%~dp0startup.bat" --validate
    )
) else if "%choice%"=="3" (
    echo Starting in development mode...
    call "%~dp0startup.bat" --dev
) else if "%choice%"=="4" (
    echo Starting in production mode...
    call "%~dp0startup.bat" --production
) else (
    echo Invalid choice. Using default startup...
    call "%~dp0startup.bat"
)

echo.
echo Startup completed. You can now:
echo - Check system health: scripts\health-check.bat
echo - View logs: docker compose logs -f [service]
echo - Stop system: docker compose down
echo.
pause
