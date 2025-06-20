#!/bin/bash
# Validation and Setup Script for RomanticNoRush DEAAP
# This script validates dependencies and system requirements

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== RomanticNoRush DEAAP Setup Validation ===${NC}"
echo "Validating system dependencies and configuration..."
echo ""

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_FILE="$PROJECT_ROOT/logs/validation.log"

# Create logs directory
mkdir -p "$PROJECT_ROOT/logs"

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1" | tee -a "$LOG_FILE"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "$LOG_FILE"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
}

# Validation functions
validate_docker() {
    log_info "Validating Docker installation..."
    
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install Docker Desktop."
        return 1
    fi
    
    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running. Please start Docker Desktop."
        return 1
    fi
    
    docker_version=$(docker --version | cut -d' ' -f3 | cut -d',' -f1)
    log_success "Docker installed: $docker_version"
    
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        log_error "Docker Compose is not available."
        return 1
    fi
    
    log_success "Docker Compose is available"
    return 0
}

validate_python() {
    log_info "Validating Python installation..."
    
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is not installed."
        return 1
    fi
    
    python_version=$(python3 --version 2>&1 | cut -d' ' -f2)
    log_success "Python installed: $python_version"
    
    if ! command -v pip3 &> /dev/null; then
        log_error "pip3 is not installed."
        return 1
    fi
    
    log_success "pip3 is available"
    return 0
}

validate_node() {
    log_info "Validating Node.js installation..."
    
    if ! command -v node &> /dev/null; then
        log_error "Node.js is not installed."
        return 1
    fi
    
    node_version=$(node --version)
    log_success "Node.js installed: $node_version"
    
    if ! command -v npm &> /dev/null; then
        log_error "npm is not installed."
        return 1
    fi
    
    npm_version=$(npm --version)
    log_success "npm installed: $npm_version"
    return 0
}

validate_environment_files() {
    log_info "Validating environment configuration..."
    
    cd "$PROJECT_ROOT"
    
    if [ ! -f ".env" ]; then
        log_warning ".env file not found. Creating from template..."
        if [ -f ".env.example" ]; then
            cp .env.example .env
            log_success "Created .env from template"
        else
            log_error ".env.example template not found"
            return 1
        fi
    else
        log_success ".env file exists"
    fi
    
    # Check required environment variables
    required_vars=("OPENAI_API_KEY" "ANTHROPIC_API_KEY" "HUGGINGFACE_API_KEY" "POSTGRES_PASSWORD")
    
    for var in "${required_vars[@]}"; do
        if grep -q "^${var}=" .env && [ -n "$(grep "^${var}=" .env | cut -d'=' -f2-)" ]; then
            log_success "Environment variable $var is set"
        else
            log_warning "Environment variable $var is not set or empty"
        fi
    done
    
    return 0
}

validate_docker_compose() {
    log_info "Validating Docker Compose configuration..."
    
    cd "$PROJECT_ROOT"
    
    if [ ! -f "docker-compose.yml" ]; then
        log_error "docker-compose.yml not found"
        return 1
    fi
    
    if docker compose config &> /dev/null; then
        log_success "docker-compose.yml is valid"
    else
        log_error "docker-compose.yml has syntax errors"
        return 1
    fi
    
    return 0
}

validate_services() {
    log_info "Validating service configurations..."
    
    cd "$PROJECT_ROOT"
    
    # Check critical service directories
    services=("auth-service" "llm-runtime" "vector-database" "blockchain-service" "api-gateway")
    
    for service in "${services[@]}"; do
        if [ -d "services/$service" ]; then
            log_success "Service directory found: $service"
            
            # Check for Dockerfile
            if [ -f "services/$service/Dockerfile" ]; then
                log_success "  Dockerfile found for $service"
            else
                log_warning "  Dockerfile missing for $service"
            fi
            
            # Check for requirements/package files
            if [ -f "services/$service/requirements.txt" ] || [ -f "services/$service/package.json" ]; then
                log_success "  Dependencies file found for $service"
            else
                log_warning "  Dependencies file missing for $service"
            fi
        else
            log_warning "Service directory missing: $service"
        fi
    done
    
    return 0
}

validate_network_connectivity() {
    log_info "Validating network connectivity..."
    
    # Test basic internet connectivity
    if ping -c 3 8.8.8.8 &> /dev/null; then
        log_success "Basic internet connectivity working"
    else
        log_warning "Basic internet connectivity issues detected"
    fi
    
    # Test Docker registry connectivity
    if curl -s --max-time 10 https://registry-1.docker.io/ &> /dev/null; then
        log_success "Docker registry accessible"
    else
        log_warning "Docker registry connectivity issues"
    fi
    
    # Test npm registry connectivity
    if curl -s --max-time 10 https://registry.npmjs.org/ &> /dev/null; then
        log_success "npm registry accessible"
    else
        log_warning "npm registry connectivity issues"
    fi
    
    return 0
}

check_system_resources() {
    log_info "Checking system resources..."
    
    # Check available disk space (need at least 10GB)
    available_space=$(df . | awk 'NR==2 {print $4}')
    available_gb=$((available_space / 1024 / 1024))
    
    if [ $available_gb -gt 10 ]; then
        log_success "Sufficient disk space: ${available_gb}GB available"
    else
        log_warning "Low disk space: ${available_gb}GB available (recommend 10GB+)"
    fi
    
    # Check available memory
    if command -v free &> /dev/null; then
        available_mem=$(free -g | awk 'NR==2{print $7}')
        if [ $available_mem -gt 4 ]; then
            log_success "Sufficient memory: ${available_mem}GB available"
        else
            log_warning "Low memory: ${available_mem}GB available (recommend 8GB+)"
        fi
    fi
    
    return 0
}

# Main validation function
main() {
    echo "Starting validation at $(date)" > "$LOG_FILE"
    
    validation_failed=false
    
    # Run all validations
    validate_docker || validation_failed=true
    echo ""
    
    validate_python || validation_failed=true
    echo ""
    
    validate_node || validation_failed=true
    echo ""
    
    validate_environment_files || validation_failed=true
    echo ""
    
    validate_docker_compose || validation_failed=true
    echo ""
    
    validate_services || validation_failed=true
    echo ""
    
    validate_network_connectivity || validation_failed=true
    echo ""
    
    check_system_resources || validation_failed=true
    echo ""
    
    # Final result
    if [ "$validation_failed" = true ]; then
        log_error "Validation completed with warnings/errors. Check logs for details."
        echo ""
        log_info "To fix common issues:"
        echo "  1. Install missing dependencies"
        echo "  2. Configure environment variables in .env"
        echo "  3. Ensure Docker Desktop is running"
        echo "  4. Check network connectivity"
        echo ""
        exit 1
    else
        log_success "All validations passed! System is ready for deployment."
        echo ""
        log_info "Next steps:"
        echo "  1. Run: ./scripts/startup.sh"
        echo "  2. Monitor: ./scripts/health-check.sh"
        echo ""
        exit 0
    fi
}

# Run main function
main "$@"
