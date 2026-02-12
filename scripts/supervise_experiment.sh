#!/bin/bash
# Process supervisor with auto-restart and failure recovery
# Usage: ./scripts/supervise_experiment.sh experiments/configs/baseline_100_clients.yaml

set -e

# Configuration
CONFIG_FILE="${1:-experiments/configs/baseline_100_clients.yaml}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${PROJECT_DIR}/logs/experiments"
SUPERVISOR_LOG="${LOG_DIR}/supervisor_$(date +"%Y%m%d_%H%M%S").log"
MAX_RETRIES=3
RETRY_DELAY=60  # seconds
CHECK_INTERVAL=30  # seconds

mkdir -p "${LOG_DIR}"

# Log function
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*" | tee -a "${SUPERVISOR_LOG}"
}

log "=========================================="
log "Experiment Supervisor Started"
log "=========================================="
log "Config: ${CONFIG_FILE}"
log "Max retries: ${MAX_RETRIES}"
log "Retry delay: ${RETRY_DELAY}s"
log "Check interval: ${CHECK_INTERVAL}s"
log ""

# Cleanup function
cleanup() {
    log "Supervisor received termination signal. Cleaning up..."
    if [ -n "${EXPERIMENT_PID}" ] && kill -0 "${EXPERIMENT_PID}" 2>/dev/null; then
        log "Stopping experiment process ${EXPERIMENT_PID}..."
        kill "${EXPERIMENT_PID}" 2>/dev/null || true
        wait "${EXPERIMENT_PID}" 2>/dev/null || true
    fi
    log "Supervisor terminated."
    exit 0
}

trap cleanup SIGINT SIGTERM

# Check if experiment completed successfully
check_completion() {
    local log_file="$1"
    if [ -f "${log_file}" ]; then
        # Check for completion markers
        if grep -q "Experiment completed successfully" "${log_file}" 2>/dev/null || \
           grep -q "FL finished in" "${log_file}" 2>/dev/null; then
            return 0  # Success
        fi
        
        # Check for critical errors
        if grep -q "OOM\|Out of memory\|Killed\|MemoryError" "${log_file}" 2>/dev/null; then
            return 2  # OOM error
        fi
    fi
    return 1  # Still running or unknown
}

# Main loop
retry_count=0
while [ ${retry_count} -lt ${MAX_RETRIES} ]; do
    log "Starting experiment (attempt $((retry_count + 1))/${MAX_RETRIES})..."
    
    # Generate timestamp for this attempt
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    EXPERIMENT_NAME=$(basename "${CONFIG_FILE}" .yaml)
    ATTEMPT_LOG="${LOG_DIR}/${EXPERIMENT_NAME}_${TIMESTAMP}_attempt${retry_count}.log"
    
    # Start experiment in background
    cd "${PROJECT_DIR}"
    
    # Activate virtual environment
    if [ -d "${PROJECT_DIR}/fl_env" ]; then
        source "${PROJECT_DIR}/fl_env/bin/activate"
    fi
    
    # Run experiment
    python -u -m src.orchestration.simulation_runner \
        --config "${CONFIG_FILE}" \
        >> "${ATTEMPT_LOG}" 2>&1 &
    
    EXPERIMENT_PID=$!
    log "Experiment started with PID: ${EXPERIMENT_PID}"
    log "Log file: ${ATTEMPT_LOG}"
    
    # Monitor the process
    while kill -0 "${EXPERIMENT_PID}" 2>/dev/null; do
        sleep ${CHECK_INTERVAL}
        
        # Check for completion
        check_completion "${ATTEMPT_LOG}"
        case $? in
            0)  # Success
                log "Experiment completed successfully!"
                wait "${EXPERIMENT_PID}"
                exit 0
                ;;
            2)  # OOM error
                log "ERROR: Out of memory detected. Killing process..."
                kill -9 "${EXPERIMENT_PID}" 2>/dev/null || true
                break
                ;;
        esac
    done
    
    # Process terminated
    wait "${EXPERIMENT_PID}" 2>/dev/null || true
    EXIT_CODE=$?
    
    log "Experiment process terminated with exit code: ${EXIT_CODE}"
    
    # Check if completed successfully
    check_completion "${ATTEMPT_LOG}"
    if [ $? -eq 0 ]; then
        log "Experiment completed successfully!"
        exit 0
    fi
    
    # Increment retry counter
    retry_count=$((retry_count + 1))
    
    if [ ${retry_count} -lt ${MAX_RETRIES} ]; then
        log "Experiment failed. Retrying in ${RETRY_DELAY} seconds..."
        log "Analyzing failure..."
        
        # Tail last 50 lines of log for debugging
        if [ -f "${ATTEMPT_LOG}" ]; then
            log "Last 50 lines of experiment log:"
            tail -n 50 "${ATTEMPT_LOG}" | while read line; do
                log "  | ${line}"
            done
        fi
        
        sleep ${RETRY_DELAY}
    else
        log "ERROR: Maximum retry attempts (${MAX_RETRIES}) reached. Giving up."
        log "Please check the logs at: ${ATTEMPT_LOG}"
        exit 1
    fi
done

log "Supervisor exiting."
