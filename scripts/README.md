# RomanticNoRush DEAAP Scripts

This directory contains operational scripts for the RomanticNoRush Decentralized Enterprise AI Agent Platform.

## Windows Users

Use the .bat files for Windows compatibility:

- start.bat - Interactive startup script with options
- validate-setup.bat - System validation
- startup.bat - Full system startup
- health-check.bat - System health monitoring
- quick-health.bat - Quick status check

## Linux/macOS Users

Use the .sh files directly:

- validate-setup.sh - System validation
- startup.sh - Full system startup  
- health-check.sh - System health monitoring

## Requirements

### Windows
- Docker Desktop for Windows
- Git Bash (recommended) or WSL (Windows Subsystem for Linux)

### Linux/macOS
- Docker and Docker Compose
- Bash shell
- curl (for health checks)

## Quick Start

1. **Windows**: Double-click scripts\start.bat
2. **Linux/macOS**: Run ./scripts/startup.sh

## Script Options

### validate-setup
- --help - Show help
- Validates Docker, dependencies, and configuration

### startup
- --help - Show help
- --validate - Run validation before startup
- --quick - Quick startup without health checks
- --dev - Development mode
- --production - Production mode
- --services <list> - Start specific services only

### health-check
- --help - Show help
- --quick - Quick health check
- --detailed - Detailed metrics
- --continuous - Continuous monitoring
- --service <name> - Check specific service
- --json - JSON output format

## Troubleshooting

1. **Docker not running**: Start Docker Desktop
2. **Permission denied**: Use Git Bash or WSL on Windows
3. **Services not responding**: Check logs with docker compose logs [service]
4. **Port conflicts**: Stop other services using the same ports

## Logs

All script outputs are logged to the logs/ directory:
- validation.log - Validation results
- startup.log - Startup process
- health-check.log - Health check results
