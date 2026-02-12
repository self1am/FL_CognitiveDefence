#!/bin/bash
# Robust experiment runner with comprehensive logging and monitoring
# Usage: ./scripts/run_experiment_robust.sh experiments/configs/baseline_100_clients.yaml

set -e  # Exit on error

# Configuration
CONFIG_FILE="${1:-experiments/configs/baseline_100_clients.yaml}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${PROJECT_DIR}/logs/experiments"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
EXPERIMENT_NAME=$(basename "${CONFIG_FILE}" .yaml)
LOG_FILE="${LOG_DIR}/${EXPERIMENT_NAME}_${TIMESTAMP}.log"
MONITOR_LOG="${LOG_DIR}/${EXPERIMENT_NAME}_${TIMESTAMP}_monitor.log"
PID_FILE="${LOG_DIR}/${EXPERIMENT_NAME}.pid"
STATUS_FILE="${LOG_DIR}/${EXPERIMENT_NAME}_status.json"

# Create log directory
mkdir -p "${LOG_DIR}"

# Print usage
usage() {
    echo "Usage: $0 <config_file>"
    echo "Example: $0 experiments/configs/baseline_100_clients.yaml"
    exit 1
}

# Check if config file exists
if [ ! -f "${CONFIG_FILE}" ]; then
    echo "Error: Config file not found: ${CONFIG_FILE}"
    usage
fi

# Log function
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*" | tee -a "${LOG_FILE}"
}

# Monitor function (runs in background)
monitor_resources() {
    local pid=$1
    log "Starting resource monitoring for PID: ${pid}"
    
    while kill -0 "${pid}" 2>/dev/null; do
        # Get memory usage
        if command -v ps &> /dev/null; then
            local mem_usage=$(ps -p "${pid}" -o rss= 2>/dev/null || echo "0")
            local mem_mb=$((mem_usage / 1024))
            
            # Get CPU usage
            local cpu_usage=$(ps -p "${pid}" -o %cpu= 2>/dev/null || echo "0")
            
            # Log to monitor file
            echo "[$(date +'%Y-%m-%d %H:%M:%S')] PID: ${pid} | Memory: ${mem_mb} MB | CPU: ${cpu_usage}%" >> "${MONITOR_LOG}"
            
            # Update status file
            cat > "${STATUS_FILE}" <<EOF
{
    "pid": ${pid},
    "status": "running",
    "memory_mb": ${mem_mb},
    "cpu_percent": ${cpu_usage},
    "last_update": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
    "log_file": "${LOG_FILE}",
    "config_file": "${CONFIG_FILE}"
}
EOF
        fi
        
        sleep 30  # Monitor every 30 seconds
    done
    
    log "Process ${pid} has terminated"
    
    # Update status to completed
    cat > "${STATUS_FILE}" <<EOF
{
    "pid": ${pid},
    "status": "completed",
    "end_time": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
    "log_file": "${LOG_FILE}",
    "config_file": "${CONFIG_FILE}"
}
EOF
}

# Main execution
log "=========================================="
log "Starting Robust Experiment Runner"
log "=========================================="
log "Config: ${CONFIG_FILE}"
log "Log file: ${LOG_FILE}"
log "Monitor log: ${MONITOR_LOG}"
log "PID file: ${PID_FILE}"
log "Status file: ${STATUS_FILE}"
log ""

# Check if experiment is already running
if [ -f "${PID_FILE}" ]; then
    OLD_PID=$(cat "${PID_FILE}")
    if kill -0 "${OLD_PID}" 2>/dev/null; then
        log "WARNING: Experiment is already running with PID: ${OLD_PID}"
        log "If you want to stop it, run: kill ${OLD_PID}"
        exit 1
    else
        log "Removing stale PID file"
        rm "${PID_FILE}"
    fi
fi

# Activate virtual environment
log "Activating virtual environment..."
if [ -d "${PROJECT_DIR}/fl_env" ]; then
    source "${PROJECT_DIR}/fl_env/bin/activate"
elif [ -d "${PROJECT_DIR}/venv" ]; then
    source "${PROJECT_DIR}/venv/bin/activate"
else
    log "WARNING: No virtual environment found. Using system Python."
fi

# Print system info
log "System Information:"
log "  Hostname: $(hostname)"
log "  Python: $(python --version 2>&1)"
log "  Available CPU cores: $(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 'unknown')"
log "  Total Memory: $(free -h 2>/dev/null | grep Mem | awk '{print $2}' || echo 'unknown')"
log "  Available Memory: $(free -h 2>/dev/null | grep Mem | awk '{print $7}' || echo 'unknown')"
log ""

# Check system resources before starting
AVAILABLE_MEM_MB=$(free -m 2>/dev/null | grep Mem | awk '{print $7}' || echo "0")
if [ "${AVAILABLE_MEM_MB}" -lt 10000 ] && [ "${AVAILABLE_MEM_MB}" -gt 0 ]; then
    log "WARNING: Low available memory: ${AVAILABLE_MEM_MB} MB"
    log "Consider stopping other processes or reducing experiment scale."
fi

# Start the experiment
log "Starting experiment..."
cd "${PROJECT_DIR}"

# Run with nohup and redirect output
nohup python -u -m src.orchestration.simulation_runner \
    --config "${CONFIG_FILE}" \
    >> "${LOG_FILE}" 2>&1 &

EXPERIMENT_PID=$!
echo "${EXPERIMENT_PID}" > "${PID_FILE}"

log "Experiment started with PID: ${EXPERIMENT_PID}"
log "Log file: ${LOG_FILE}"
log "Monitor log: ${MONITOR_LOG}"
log ""
log "To monitor progress in real-time, run:"
log "  tail -f ${LOG_FILE}"
log ""
log "To check resource usage, run:"
log "  tail -f ${MONITOR_LOG}"
log ""
log "To stop the experiment, run:"
log "  kill ${EXPERIMENT_PID}"
log ""

# Start monitoring in background
monitor_resources "${EXPERIMENT_PID}" &
MONITOR_PID=$!

log "Resource monitoring started with PID: ${MONITOR_PID}"
log "Experiment is now running in the background."
log "=========================================="

# Wait a moment to check if process started successfully
sleep 5
if ! kill -0 "${EXPERIMENT_PID}" 2>/dev/null; then
    log "ERROR: Experiment process terminated immediately. Check logs for errors."
    rm -f "${PID_FILE}"
    exit 1
fi

log "Experiment is running successfully. Detaching..."
