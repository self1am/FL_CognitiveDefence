#!/bin/bash
# run_production_experiments.sh
# Automated script to run all production-scale experiments (100 clients, 30-50 rounds)
# Usage: ./scripts/run_production_experiments.sh [experiment_name]
# If no argument provided, runs all experiments in sequence

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs/campaign_$TIMESTAMP"
RESULTS_DIR="experiments/results"
CAMPAIGN_LOG="$LOG_DIR/campaign.log"

# Create directories
mkdir -p "$LOG_DIR"
mkdir -p "$RESULTS_DIR"

# Function to log with timestamp
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$CAMPAIGN_LOG"
}

log_success() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] ✓ $1${NC}" | tee -a "$CAMPAIGN_LOG"
}

log_error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ✗ $1${NC}" | tee -a "$CAMPAIGN_LOG"
}

log_warning() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] ⚠ $1${NC}" | tee -a "$CAMPAIGN_LOG"
}

# Function to monitor resources
monitor_resources() {
    local interval=${1:-5}
    
    while true; do
        clear
        echo -e "${BLUE}=== System Resources at $(date +'%Y-%m-%d %H:%M:%S') ===${NC}"
        echo ""
        
        # Memory
        echo -e "${YELLOW}Memory:${NC}"
        free -h | grep Mem
        echo ""
        
        # CPU
        echo -e "${YELLOW}CPU Load:${NC}"
        uptime
        echo ""
        
        # Python processes
        echo -e "${YELLOW}Python Processes:${NC}"
        ps aux | grep "python" | grep -v grep | wc -l
        echo ""
        
        # Disk usage
        echo -e "${YELLOW}Disk Usage (/):${NC}"
        df -h / | tail -1
        echo ""
        
        # Logs size
        echo -e "${YELLOW}Logs Size:${NC}"
        du -sh logs 2>/dev/null || echo "No logs yet"
        
        echo ""
        echo -e "${BLUE}(Press Ctrl+C to exit monitoring)${NC}"
        sleep $interval
    done
}

# Function to run single experiment
run_experiment() {
    local config_file=$1
    local config_name=$(basename "$config_file" .yaml)
    local exp_log="$LOG_DIR/${config_name}.log"
    
    log "Starting experiment: ${config_name}"
    log "Config file: ${config_file}"
    
    # Check if config file exists
    if [ ! -f "$config_file" ]; then
        log_error "Config file not found: $config_file"
        return 1
    fi
    
    # Check system resources before starting
    local available_mem=$(free -h | grep Mem | awk '{print $7}' | sed 's/G//')
    if (( $(echo "$available_mem < 10" | bc -l) )); then
        log_warning "Low available memory: ${available_mem}GB. Proceeding anyway..."
    fi
    
    # Run experiment
    local start_time=$(date +%s)
    
    python -m src.orchestration.experiment_runner \
        --config "$config_file" 2>&1 | tee -a "$exp_log"
    
    local exit_code=$?
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    local duration_minutes=$((duration / 60))
    
    if [ $exit_code -eq 0 ]; then
        log_success "Experiment completed: ${config_name} (Duration: ${duration_minutes} minutes)"
    else
        log_error "Experiment failed: ${config_name} (Exit code: $exit_code)"
        return 1
    fi
    
    return 0
}

# Function to run cleanup between experiments
cleanup_between_experiments() {
    local cooldown=${1:-300}  # Default 5 minutes
    
    log "Cooling down system for $((cooldown / 60)) minutes..."
    
    # Kill any zombie processes
    pkill -9 -f "python.*client" 2>/dev/null || true
    
    # Clear caches
    sync
    
    # Wait
    sleep "$cooldown"
    
    log "Cooldown complete"
}

# Function to collect results summary
summarize_results() {
    log "Collecting experiment results..."
    
    cat > "$LOG_DIR/results_summary.txt" << 'EOF'
PRODUCTION EXPERIMENT CAMPAIGN SUMMARY
=====================================
EOF
    
    # Summarize each experiment
    for log_file in "$LOG_DIR"/*.log; do
        if [ -f "$log_file" ] && [ "$(basename "$log_file")" != "campaign.log" ]; then
            echo "" >> "$LOG_DIR/results_summary.txt"
            echo "File: $(basename "$log_file")" >> "$LOG_DIR/results_summary.txt"
            
            # Extract key metrics (adjust based on your logging format)
            grep "Experiment completed\|successfully spawned\|CENTRALIZED EVALUATION" "$log_file" | tail -20 >> "$LOG_DIR/results_summary.txt" || true
        fi
    done
    
    log_success "Results summary saved to $LOG_DIR/results_summary.txt"
}

# Function to show help
show_help() {
    cat << EOF
Usage: ./scripts/run_production_experiments.sh [OPTIONS]

Options:
  -h, --help                  Show this help message
  -m, --monitor              Run in monitoring mode (watch resources)
  -c, --config CONFIG_FILE   Run single experiment
  -a, --all                  Run all production experiments (default)
  -s, --skip-cleanup         Skip cleanup between experiments
  
Examples:
  # Run all experiments
  ./scripts/run_production_experiments.sh --all
  
  # Run single experiment
  ./scripts/run_production_experiments.sh --config experiments/configs/production_100_clients_cognitive.yaml
  
  # Monitor resources in separate terminal
  ./scripts/run_production_experiments.sh --monitor

EOF
}

# Main script
main() {
    local mode="all"
    local skip_cleanup=false
    local config_file=""
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -m|--monitor)
                mode="monitor"
                shift
                ;;
            -c|--config)
                mode="single"
                config_file="$2"
                shift 2
                ;;
            -a|--all)
                mode="all"
                shift
                ;;
            -s|--skip-cleanup)
                skip_cleanup=true
                shift
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    # Run based on mode
    case $mode in
        monitor)
            echo "Starting resource monitoring..."
            monitor_resources 2
            ;;
            
        single)
            log "========================================="
            log "Starting Single Experiment"
            log "========================================="
            run_experiment "$config_file"
            exit_code=$?
            summarize_results
            exit $exit_code
            ;;
            
        all)
            log "========================================="
            log "Starting Production Experiment Campaign"
            log "========================================="
            log "Timestamp: $TIMESTAMP"
            log "Log directory: $LOG_DIR"
            log ""
            
            # Array of experiments to run
            declare -a experiments=(
                "experiments/configs/production_100_clients_cognitive.yaml"
                "experiments/configs/production_100_clients_adaptive.yaml"
            )
            
            local total=${#experiments[@]}
            local completed=0
            local failed=0
            
            for i in "${!experiments[@]}"; do
                local config="${experiments[$i]}"
                local exp_num=$((i + 1))
                
                log ""
                log "========================================="
                log "Experiment $exp_num/$total: $(basename $config)"
                log "========================================="
                
                if run_experiment "$config"; then
                    ((completed++))
                else
                    ((failed++))
                    if [ "$skip_cleanup" = false ]; then
                        log_warning "Experiment failed. Attempting recovery..."
                    fi
                fi
                
                # Cleanup between experiments (unless last one)
                if [ $((i + 1)) -lt $total ] && [ "$skip_cleanup" = false ]; then
                    cleanup_between_experiments 300
                fi
            done
            
            # Final summary
            log ""
            log "========================================="
            log "CAMPAIGN SUMMARY"
            log "========================================="
            log "Total experiments: $total"
            log "Completed: $completed"
            log "Failed: $failed"
            log "Timestamp: $TIMESTAMP"
            log "Log directory: $LOG_DIR"
            log "========================================="
            
            summarize_results
            
            # Archive results
            log "Archiving results..."
            tar -czf "$RESULTS_DIR/campaign_${TIMESTAMP}.tar.gz" "$LOG_DIR" 2>/dev/null || true
            log_success "Results archived to $RESULTS_DIR/campaign_${TIMESTAMP}.tar.gz"
            
            if [ $failed -eq 0 ]; then
                log_success "All experiments completed successfully!"
                exit 0
            else
                log_error "$failed experiment(s) failed"
                exit 1
            fi
            ;;
    esac
}

# Run main function
main "$@"
