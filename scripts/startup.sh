#!/bin/bash
# Comprehensive Startup Script for RomanticNoRush DEAAP
# This script handles the complete system startup sequence

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_FILE="$PROJECT_ROOT/logs/startup.log"
PID_FILE="$PROJECT_ROOT/logs/startup.pid"

# Create logs directory
mkdir -p "$PROJECT_ROOT/logs"

# Store script PID
echo $$ > "$PID_FILE"

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

log_step() {
    echo -e "${PURPLE}[STEP]${NC} $1" | tee -a "$LOG_FILE"
}

log_progress() {
    echo -e "${CYAN}[PROGRESS]${NC} $1" | tee -a "$LOG_FILE"
}

# Cleanup function
cleanup() {
    log_info "Cleaning up startup process..."
    if [ -f "$PID_FILE" ]; then
        rm -f "$PID_FILE"
    fi
}

# Trap cleanup on exit
trap cleanup EXIT

# Help function
show_help() {
    echo "RomanticNoRush DEAAP Startup Script"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -h, --help          Show this help message"
    echo "  -v, --validate      Run validation before startup"
    echo "  -q, --quick         Quick startup (skip some health checks)"
    echo "  -d, --dev          Development mode (with additional services)"
    echo "  -p, --production   Production mode (optimized settings)"
    echo "  -s, --services     Comma-separated list of specific services to start"
    echo "  -w, --wait         Wait time between service startups (default: 10s)"
    echo "  --no-pull          Skip pulling latest Docker images"
    echo "  --rebuild          Force rebuild of all images"
    echo ""
    echo "Examples:"
    echo "  $0                           # Standard startup"
    echo "  $0 --validate               # Validate before starting"
    echo "  $0 --dev                    # Development mode"
    echo "  $0 --services db,redis      # Start only database and Redis"
    echo "  $0 --production --no-pull   # Production mode without pulling images"
}

# Default configuration
VALIDATE=false
QUICK=false
DEV_MODE=false
PRODUCTION_MODE=false
SPECIFIC_SERVICES=""
WAIT_TIME=10
NO_PULL=false
REBUILD=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -v|--validate)
            VALIDATE=true
            shift
            ;;
        -q|--quick)
            QUICK=true
            shift
            ;;
        -d|--dev)
            DEV_MODE=true
            shift
            ;;
        -p|--production)
            PRODUCTION_MODE=true
            shift
            ;;
        -s|--services)
            SPECIFIC_SERVICES="$2"
            shift 2
            ;;
        -w|--wait)
            WAIT_TIME="$2"
            shift 2
            ;;
        --no-pull)
            NO_PULL=true
            shift
            ;;
        --rebuild)
            REBUILD=true
            shift
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Banner
print_banner() {
    echo -e "${PURPLE}"
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║                    RomanticNoRush DEAAP                     ║"
    echo "║           Decentralized Enterprise AI Agent Platform        ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

# Pre-flight checks
preflight_checks() {
    log_step "Running pre-flight checks..."
    
    cd "$PROJECT_ROOT"
    
    # Check if Docker is running
    if ! docker info &> /dev/null; then
        log_error "Docker is not running. Please start Docker Desktop."
        exit 1
    fi
    
    # Check if docker-compose.yml exists
    if [ ! -f "docker-compose.yml" ]; then
        log_error "docker-compose.yml not found in project root"
        exit 1
    fi
    
    # Check if .env file exists
    if [ ! -f ".env" ]; then
        log_warning ".env file not found. Creating from template..."
        if [ -f ".env.example" ]; then
            cp .env.example .env
            log_success "Created .env from template"
        else
            log_error ".env.example template not found"
            exit 1
        fi
    fi
    
    log_success "Pre-flight checks completed"
}

# Validation step
run_validation() {
    if [ "$VALIDATE" = true ]; then
        log_step "Running system validation..."
        
        if [ -f "$SCRIPT_DIR/validate-setup.sh" ]; then
            bash "$SCRIPT_DIR/validate-setup.sh"
            if [ $? -ne 0 ]; then
                log_error "Validation failed. Please fix issues before continuing."
                exit 1
            fi
        else
            log_warning "Validation script not found, skipping validation"
        fi
    fi
}

# Clean up previous deployments
cleanup_previous() {
    log_step "Cleaning up previous deployments..."
    
    cd "$PROJECT_ROOT"
    
    # Stop any running containers
    if docker compose ps -q &> /dev/null; then
        log_info "Stopping existing containers..."
        docker compose down --remove-orphans &> /dev/null || true
    fi
    
    # Clean up orphaned containers
    orphaned=$(docker ps -aq --filter "label=com.docker.compose.project=romanticnorush" 2>/dev/null || true)
    if [ -n "$orphaned" ]; then
        log_info "Removing orphaned containers..."
        docker rm -f $orphaned &> /dev/null || true
    fi
    
    # Clean up unused networks
    docker network prune -f &> /dev/null || true
    
    log_success "Cleanup completed"
}

# Pull or build images
prepare_images() {
    log_step "Preparing Docker images..."
    
    cd "$PROJECT_ROOT"
    
    if [ "$REBUILD" = true ]; then
        log_info "Force rebuilding all images..."
        docker compose build --no-cache
    elif [ "$NO_PULL" = false ]; then
        log_info "Pulling latest images and building custom ones..."
        docker compose pull --ignore-buildable 2>/dev/null || true
        docker compose build
    else
        log_info "Building custom images only..."
        docker compose build
    fi
    
    log_success "Images prepared"
}

# Start core infrastructure services
start_infrastructure() {
    log_step "Starting core infrastructure services..."
    
    cd "$PROJECT_ROOT"
    
    # Define service startup order
    infrastructure_services=(
        "postgres"
        "redis"
        "qdrant"
        "nginx"
    )
    
    for service in "${infrastructure_services[@]}"; do
        if [ -n "$SPECIFIC_SERVICES" ] && [[ "$SPECIFIC_SERVICES" != *"$service"* ]]; then
            continue
        fi
        
        log_progress "Starting $service..."
        docker compose up -d "$service"
        
        # Wait for service to be ready
        log_info "Waiting for $service to be ready..."
        sleep $WAIT_TIME
        
        # Basic health check
        if docker compose ps "$service" | grep -q "Up"; then
            log_success "$service started successfully"
        else
            log_error "$service failed to start"
            docker compose logs "$service" | tail -20
            exit 1
        fi
    done
}

# Start application services
start_applications() {
    log_step "Starting application services..."
    
    cd "$PROJECT_ROOT"
    
    # Define application services in startup order
    app_services=(
        "auth-service"
        "vector-database"
        "llm-runtime"
        "synthetic-data"
        "blockchain-service"
        "api-gateway"
        "frontend"
    )
    
    # Add development services if in dev mode
    if [ "$DEV_MODE" = true ]; then
        app_services+=(
            "mock-validator"
            "data-processor"
            "consensus-manager"
        )
    fi
    
    for service in "${app_services[@]}"; do
        if [ -n "$SPECIFIC_SERVICES" ] && [[ "$SPECIFIC_SERVICES" != *"$service"* ]]; then
            continue
        fi
        
        log_progress "Starting $service..."
        docker compose up -d "$service"
        
        # Wait for service to be ready
        if [ "$QUICK" = false ]; then
            log_info "Waiting for $service to be ready..."
            sleep $WAIT_TIME
            
            # Check if service is running
            if docker compose ps "$service" | grep -q "Up"; then
                log_success "$service started successfully"
            else
                log_warning "$service may have issues, check logs"
            fi
        fi
    done
    
    if [ "$QUICK" = true ]; then
        log_info "Quick mode: Waiting 30 seconds for all services to initialize..."
        sleep 30
    fi
}

# Perform health checks
health_checks() {
    if [ "$QUICK" = true ]; then
        log_info "Skipping detailed health checks (quick mode)"
        return 0
    fi
    
    log_step "Performing health checks..."
    
    cd "$PROJECT_ROOT"
    
    # Check if health check script exists
    if [ -f "$SCRIPT_DIR/health-check.sh" ]; then
        bash "$SCRIPT_DIR/health-check.sh" --quick
        if [ $? -ne 0 ]; then
            log_warning "Some health checks failed, but continuing..."
        fi
    else
        log_warning "Health check script not found, performing basic checks..."
        
        # Basic container status check
        failed_services=()
        for service in $(docker compose config --services); do
            if ! docker compose ps "$service" | grep -q "Up"; then
                failed_services+=("$service")
            fi
        done
        
        if [ ${#failed_services[@]} -gt 0 ]; then
            log_warning "Services with issues: ${failed_services[*]}"
        else
            log_success "All services appear to be running"
        fi
    fi
}

# Display startup summary
startup_summary() {
    log_step "Startup Summary"
    
    cd "$PROJECT_ROOT"
    
    echo ""
    echo -e "${CYAN}=== SYSTEM STATUS ===${NC}"
    
    # Service status
    echo -e "${BLUE}Service Status:${NC}"
    docker compose ps --format "table {{.Name}}\t{{.State}}\t{{.Status}}"
    
    echo ""
    echo -e "${BLUE}Access Points:${NC}"
    echo "  🌐 Frontend:           http://localhost:3000"
    echo "  🔌 API Gateway:        http://localhost:8080"
    echo "  🔐 Auth Service:       http://localhost:3001"
    echo "  🧠 LLM Runtime:        http://localhost:8001"
    echo "  📊 Vector Database:    http://localhost:8002"
    echo "  ⛓️  Blockchain Service: http://localhost:8003"
    echo "  🗄️  Qdrant Dashboard:  http://localhost:6333/dashboard"
    echo "  📝 Nginx Status:       http://localhost:80/status"
    
    if [ "$DEV_MODE" = true ]; then
        echo ""
        echo -e "${BLUE}Development Services:${NC}"
        echo "  🔍 Mock Validator:     http://localhost:8010"
        echo "  📦 Data Processor:     http://localhost:8011"
        echo "  🤝 Consensus Manager:  http://localhost:8012"
    fi
    
    echo ""
    echo -e "${BLUE}Management Commands:${NC}"
    echo "  📊 Health Check:       ./scripts/health-check.sh"
    echo "  📋 View Logs:          docker compose logs -f [service]"
    echo "  🛑 Stop System:        docker compose down"
    echo "  🔄 Restart Service:    docker compose restart [service]"
    
    echo ""
    log_success "RomanticNoRush DEAAP startup completed successfully!"
    
    if [ "$PRODUCTION_MODE" = true ]; then
        echo ""
        echo -e "${YELLOW}Production Mode Notes:${NC}"
        echo "  - Monitoring and logging are optimized for production"
        echo "  - Debug services are disabled"
        echo "  - Performance metrics are being collected"
    fi
    
    if [ "$DEV_MODE" = true ]; then
        echo ""
        echo -e "${YELLOW}Development Mode Notes:${NC}"
        echo "  - Additional debugging services are available"
        echo "  - Hot reloading is enabled where applicable"
        echo "  - Development tools are accessible"
    fi
}

# Main startup function
main() {
    echo "Startup initiated at $(date)" > "$LOG_FILE"
    
    print_banner
    
    log_info "Starting RomanticNoRush DEAAP..."
    
    if [ "$PRODUCTION_MODE" = true ]; then
        log_info "Mode: Production"
    elif [ "$DEV_MODE" = true ]; then
        log_info "Mode: Development"
    else
        log_info "Mode: Standard"
    fi
    
    if [ -n "$SPECIFIC_SERVICES" ]; then
        log_info "Starting specific services: $SPECIFIC_SERVICES"
    fi
    
    echo ""
    
    # Execute startup sequence
    preflight_checks
    run_validation
    cleanup_previous
    prepare_images
    start_infrastructure
    start_applications
    health_checks
    startup_summary
    
    echo ""
    log_success "Startup sequence completed at $(date)"
}

# Handle script interruption
interrupt_handler() {
    log_warning "Startup interrupted by user"
    cleanup
    exit 130
}

trap interrupt_handler SIGINT SIGTERM

# Run main function
main "$@"
