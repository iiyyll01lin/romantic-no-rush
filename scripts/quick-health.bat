@echo off
echo ===============================================
echo    RomanticNoRush DEAAP - Quick Health Check
echo ===============================================
echo.

REM Check Docker
docker info >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Docker Desktop is not running
    goto :end
)
echo [OK] Docker Desktop is running

REM Check if services are running
echo.
echo Service Status:
docker compose ps

echo.
echo Quick connectivity test:
timeout /t 2 /nobreak >nul
curl -s http://localhost:8080/health >nul 2>&1 && echo [OK] API Gateway responding || echo [WARNING] API Gateway not responding
curl -s http://localhost:3001/health >nul 2>&1 && echo [OK] Auth Service responding || echo [WARNING] Auth Service not responding
curl -s http://localhost:8001/health >nul 2>&1 && echo [OK] LLM Runtime responding || echo [WARNING] LLM Runtime not responding

echo.
echo For detailed health check, run: scripts\health-check.bat

:end
echo.
pause
