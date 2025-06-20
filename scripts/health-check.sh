#!/bin/bash
# Comprehensive Health Check Script for RomanticNoRush DEAAP
# This script monitors system health and service status

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
LOG_FILE="$PROJECT_ROOT/logs/health-check.log"

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

log_step() {
    echo -e "${PURPLE}[CHECK]${NC} $1" | tee -a "$LOG_FILE"
}

# Configuration
QUICK_MODE=false
DETAILED_MODE=false
CONTINUOUS_MODE=false
SPECIFIC_SERVICE=""
ALERT_THRESHOLD=80  # CPU/Memory threshold for alerts
CHECK_INTERVAL=30   # Interval for continuous mode (seconds)

# Health check results
TOTAL_CHECKS=0
PASSED_CHECKS=0
FAILED_CHECKS=0
WARNING_CHECKS=0

# Help function
show_help() {
    echo "RomanticNoRush DEAAP Health Check Script"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -h, --help          Show this help message"
    echo "  -q, --quick         Quick health check (basic status only)"
    echo "  -d, --detailed      Detailed health check with performance metrics"
    echo "  -c, --continuous    Continuous monitoring mode"
    echo "  -s, --service       Check specific service only"
    echo "  -i, --interval      Check interval for continuous mode (default: 30s)"
    echo "  -t, --threshold     Alert threshold for CPU/Memory (default: 80%)"
    echo "  --json             Output results in JSON format"
    echo "  --export           Export health report to file"
    echo ""
    echo "Examples:"
    echo "  $0                          # Standard health check"
    echo "  $0 --quick                  # Quick check"
    echo "  $0 --detailed               # Detailed metrics"
    echo "  $0 --continuous             # Monitor continuously"
    echo "  $0 --service api-gateway    # Check specific service"
    echo "  $0 --json --export          # Export JSON report"
}

# Parse command line arguments
JSON_OUTPUT=false
EXPORT_REPORT=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -q|--quick)
            QUICK_MODE=true
            shift
            ;;
        -d|--detailed)
            DETAILED_MODE=true
            shift
            ;;
        -c|--continuous)
            CONTINUOUS_MODE=true
            shift
            ;;
        -s|--service)
            SPECIFIC_SERVICE="$2"
            shift 2
            ;;
        -i|--interval)
            CHECK_INTERVAL="$2"
            shift 2
            ;;
        -t|--threshold)
            ALERT_THRESHOLD="$2"
            shift 2
            ;;
        --json)
            JSON_OUTPUT=true
            shift
            ;;
        --export)
            EXPORT_REPORT=true
            shift
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Utility functions
increment_total() { ((TOTAL_CHECKS++)); }
increment_passed() { ((PASSED_CHECKS++)); }
increment_failed() { ((FAILED_CHECKS++)); }
increment_warning() { ((WARNING_CHECKS++)); }

# Docker service status check
check_docker_status() {
    log_step "Checking Docker daemon status"
    increment_total
    
    if docker info &> /dev/null; then
        log_success "Docker daemon is running"
        increment_passed
        return 0
    else
        log_error "Docker daemon is not running"
        increment_failed
        return 1
    fi
}

# Container status checks
check_container_status() {
    log_step "Checking container status"
    
    cd "$PROJECT_ROOT"
    
    if [ ! -f "docker-compose.yml" ]; then
        log_error "docker-compose.yml not found"
        return 1
    fi
    
    local services
    if [ -n "$SPECIFIC_SERVICE" ]; then
        services="$SPECIFIC_SERVICE"
    else
        services=$(docker compose config --services 2>/dev/null || echo "")
    fi
    
    if [ -z "$services" ]; then
        log_warning "No services found in docker-compose.yml"
        return 1
    fi
    
    local failed_services=()
    local healthy_services=()
    local unhealthy_services=()
    
    for service in $services; do
        increment_total
        
        # Check if container exists and is running
        if docker compose ps "$service" | grep -q "Up"; then
            # Check health status if available
            health_status=$(docker compose ps "$service" --format "{{.Status}}" | grep -o "(healthy\|unhealthy\|starting)" || echo "unknown")
            
            case "$health_status" in
                "*healthy*")
                    log_success "$service: Running and healthy"
                    healthy_services+=("$service")
                    increment_passed
                    ;;
                "*unhealthy*")
                    log_warning "$service: Running but unhealthy"
                    unhealthy_services+=("$service")
                    increment_warning
                    ;;
                "*starting*")
                    log_info "$service: Starting up"
                    increment_warning
                    ;;
                *)
                    log_success "$service: Running"
                    healthy_services+=("$service")
                    increment_passed
                    ;;
            esac
        else
            log_error "$service: Not running"
            failed_services+=("$service")
            increment_failed
        fi
    done
    
    # Summary
    if [ ${#failed_services[@]} -gt 0 ]; then
        log_error "Failed services: ${failed_services[*]}"
    fi
    
    if [ ${#unhealthy_services[@]} -gt 0 ]; then
        log_warning "Unhealthy services: ${unhealthy_services[*]}"
    fi
    
    return ${#failed_services[@]}
}

# Network connectivity checks
check_network_connectivity() {
    log_step "Checking network connectivity"
    
    cd "$PROJECT_ROOT"
    
    # Internal service connectivity
    local services=("api-gateway:8080" "auth-service:3001" "llm-runtime:8001")
    
    for service_endpoint in "${services[@]}"; do
        increment_total
        
        local service_name=$(echo "$service_endpoint" | cut -d: -f1)
        local port=$(echo "$service_endpoint" | cut -d: -f2)
        
        # Check if service is responding
        if curl -s --max-time 5 "http://localhost:$port/health" &> /dev/null || \
           curl -s --max-time 5 "http://localhost:$port/" &> /dev/null; then
            log_success "$service_name: Network connectivity OK"
            increment_passed
        else
            # Check if container is running first
            if docker compose ps "$service_name" | grep -q "Up"; then
                log_warning "$service_name: Container running but not responding on port $port"
                increment_warning
            else
                log_error "$service_name: Container not running"
                increment_failed
            fi
        fi
    done
    
    # External connectivity (if not in quick mode)
    if [ "$QUICK_MODE" = false ]; then
        increment_total
        if ping -c 3 8.8.8.8 &> /dev/null; then
            log_success "External network connectivity OK"
            increment_passed
        else
            log_warning "External network connectivity issues"
            increment_warning
        fi
    fi
}

# Database connectivity checks
check_database_connectivity() {
    log_step "Checking database connectivity"
    
    cd "$PROJECT_ROOT"
    
    # PostgreSQL check
    increment_total
    if docker compose exec -T postgres pg_isready -U postgres &> /dev/null; then
        log_success "PostgreSQL: Connected and ready"
        increment_passed
    else
        log_error "PostgreSQL: Connection failed"
        increment_failed
    fi
    
    # Redis check
    increment_total
    if docker compose exec -T redis redis-cli ping &> /dev/null; then
        log_success "Redis: Connected and responding"
        increment_passed
    else
        log_error "Redis: Connection failed"
        increment_failed
    fi
    
    # Qdrant check
    increment_total
    if curl -s --max-time 5 "http://localhost:6333/health" | grep -q "ok" 2>/dev/null; then
        log_success "Qdrant: Connected and healthy"
        increment_passed
    else
        log_warning "Qdrant: Health check failed"
        increment_warning
    fi
}

# Resource usage checks
check_resource_usage() {
    if [ "$QUICK_MODE" = true ]; then
        return 0
    fi
    
    log_step "Checking resource usage"
    
    cd "$PROJECT_ROOT"
    
    # System resources
    increment_total
    if command -v free &> /dev/null; then
        memory_usage=$(free | awk 'FNR==2{printf "%.0f", $3/$2*100}')
        if [ "$memory_usage" -gt "$ALERT_THRESHOLD" ]; then
            log_warning "High memory usage: ${memory_usage}%"
            increment_warning
        else
            log_success "Memory usage: ${memory_usage}%"
            increment_passed
        fi
    else
        log_info "Memory usage check not available on this system"
        increment_passed
    fi
    
    # Disk usage
    increment_total
    disk_usage=$(df "$PROJECT_ROOT" | awk 'NR==2 {print $5}' | sed 's/%//')
    if [ "$disk_usage" -gt "$ALERT_THRESHOLD" ]; then
        log_warning "High disk usage: ${disk_usage}%"
        increment_warning
    else
        log_success "Disk usage: ${disk_usage}%"
        increment_passed
    fi
    
    # Docker resource usage
    if [ "$DETAILED_MODE" = true ]; then
        log_info "Docker container resource usage:"
        docker compose ps --format "table {{.Name}}\t{{.Status}}" | head -20
    fi
}

# API endpoint health checks
check_api_endpoints() {
    if [ "$QUICK_MODE" = true ]; then
        return 0
    fi
    
    log_step "Checking API endpoint health"
    
    # Define critical endpoints
    local endpoints=(
        "http://localhost:8080/health:API Gateway"
        "http://localhost:3001/health:Auth Service"
        "http://localhost:8001/health:LLM Runtime"
        "http://localhost:8002/health:Vector Database"
        "http://localhost:8003/health:Blockchain Service"
    )
    
    for endpoint_info in "${endpoints[@]}"; do
        increment_total
        
        local url=$(echo "$endpoint_info" | cut -d: -f1-2)
        local name=$(echo "$endpoint_info" | cut -d: -f3)
        
        local response_code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 "$url" 2>/dev/null || echo "000")
        
        if [ "$response_code" = "200" ]; then
            log_success "$name: API endpoint healthy (HTTP $response_code)"
            increment_passed
        elif [ "$response_code" = "000" ]; then
            log_error "$name: API endpoint unreachable"
            increment_failed
        else
            log_warning "$name: API endpoint responding with HTTP $response_code"
            increment_warning
        fi
    done
}

# Log analysis
check_recent_errors() {
    if [ "$QUICK_MODE" = true ]; then
        return 0
    fi
    
    log_step "Checking for recent errors in logs"
    
    cd "$PROJECT_ROOT"
    
    increment_total
    
    # Check for error patterns in recent logs
    error_count=$(docker compose logs --since="10m" 2>/dev/null | grep -i "error\|exception\|fatal\|critical" | wc -l || echo "0")
    
    if [ "$error_count" -eq 0 ]; then
        log_success "No recent errors found in logs"
        increment_passed
    elif [ "$error_count" -lt 10 ]; then
        log_warning "Found $error_count recent errors in logs"
        increment_warning
    else
        log_error "Found $error_count recent errors in logs (high error rate)"
        increment_failed
    fi
    
    # Show recent critical errors if in detailed mode
    if [ "$DETAILED_MODE" = true ] && [ "$error_count" -gt 0 ]; then
        log_info "Recent error samples:"
        docker compose logs --since="10m" 2>/dev/null | grep -i "error\|exception\|fatal\|critical" | tail -5 | while read -r line; do
            echo "  $line"
        done
    fi
}

# Generate health report
generate_report() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    local score=$(( (PASSED_CHECKS * 100) / (TOTAL_CHECKS > 0 ? TOTAL_CHECKS : 1) ))
    
    if [ "$JSON_OUTPUT" = true ]; then
        cat << EOF
{
  "timestamp": "$timestamp",
  "overall_score": $score,
  "total_checks": $TOTAL_CHECKS,
  "passed_checks": $PASSED_CHECKS,
  "failed_checks": $FAILED_CHECKS,
  "warning_checks": $WARNING_CHECKS,
  "status": "$([ $FAILED_CHECKS -eq 0 ] && echo "healthy" || echo "unhealthy")"
}
EOF
    else
        echo ""
        echo -e "${CYAN}=== HEALTH CHECK SUMMARY ===${NC}"
        echo "Timestamp: $timestamp"
        echo "Overall Score: $score%"
        echo "Total Checks: $TOTAL_CHECKS"
        echo "✅ Passed: $PASSED_CHECKS"
        echo "⚠️  Warnings: $WARNING_CHECKS"
        echo "❌ Failed: $FAILED_CHECKS"
        echo ""
        
        if [ $FAILED_CHECKS -eq 0 ]; then
            if [ $WARNING_CHECKS -eq 0 ]; then
                log_success "System is healthy! All checks passed."
            else
                log_warning "System is mostly healthy with some warnings."
            fi
        else
            log_error "System has health issues that need attention."
        fi
    fi
    
    # Export report if requested
    if [ "$EXPORT_REPORT" = true ]; then
        local report_file="$PROJECT_ROOT/logs/health-report-$(date '+%Y%m%d-%H%M%S').json"
        generate_report > "$report_file"
        log_info "Health report exported to: $report_file"
    fi
}

# Single health check run
run_health_check() {
    local start_time=$(date '+%Y-%m-%d %H:%M:%S')
    
    if [ "$JSON_OUTPUT" = false ]; then
        echo -e "${PURPLE}=== RomanticNoRush DEAAP Health Check ===${NC}"
        echo "Started at: $start_time"
        echo ""
    fi
    
    # Reset counters
    TOTAL_CHECKS=0
    PASSED_CHECKS=0
    FAILED_CHECKS=0
    WARNING_CHECKS=0
    
    # Run checks
    check_docker_status
    check_container_status
    check_network_connectivity
    check_database_connectivity
    check_resource_usage
    check_api_endpoints
    check_recent_errors
    
    # Generate report
    generate_report
    
    # Return appropriate exit code
    return $FAILED_CHECKS
}

# Continuous monitoring mode
continuous_monitoring() {
    log_info "Starting continuous monitoring (interval: ${CHECK_INTERVAL}s)"
    log_info "Press Ctrl+C to stop monitoring"
    echo ""
    
    local iteration=1
    
    while true; do
        echo -e "${BLUE}=== Health Check #$iteration ===${NC}"
        
        run_health_check
        
        echo ""
        log_info "Next check in ${CHECK_INTERVAL} seconds..."
        sleep "$CHECK_INTERVAL"
        
        ((iteration++))
        echo ""
    done
}

# Main function
main() {
    cd "$PROJECT_ROOT"
    
    echo "Health check initiated at $(date)" >> "$LOG_FILE"
    
    if [ "$CONTINUOUS_MODE" = true ]; then
        continuous_monitoring
    else
        run_health_check
        exit $?
    fi
}

# Handle script interruption
interrupt_handler() {
    if [ "$CONTINUOUS_MODE" = true ]; then
        echo ""
        log_info "Continuous monitoring stopped by user"
    fi
    exit 130
}

trap interrupt_handler SIGINT SIGTERM

# Run main function
main "$@"
