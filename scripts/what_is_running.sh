#!/bin/bash
# Quick status check - what's running right now?
# Usage: ./scripts/what_is_running.sh

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${PROJECT_DIR}/logs/experiments"

echo "=========================================="
echo "🔍 FL Experiments Status Check"
echo "=========================================="
echo ""

# Check for Python simulation processes
echo "📊 Active Python Processes:"
PYTHON_PROCS=$(ps aux | grep "simulation_runner" | grep -v grep)
if [ -z "$PYTHON_PROCS" ]; then
    echo "  ❌ No simulation_runner processes found"
else
    echo "$PYTHON_PROCS" | while read line; do
        PID=$(echo $line | awk '{print $2}')
        CPU=$(echo $line | awk '{print $3}')
        MEM=$(echo $line | awk '{print $4}')
        RUNTIME=$(echo $line | awk '{print $10}')
        echo "  ✅ PID $PID | CPU: ${CPU}% | MEM: ${MEM}% | Runtime: ${RUNTIME}"
    done
fi
echo ""

# Check Ray processes
echo "☢️  Ray Processes:"
RAY_PROCS=$(ps aux | grep ray | grep -v grep | wc -l)
if [ "$RAY_PROCS" -gt 0 ]; then
    echo "  ✅ $RAY_PROCS Ray processes running"
else
    echo "  ❌ No Ray processes found"
fi
echo ""

# Check PID files
echo "📝 Experiments with PID Files:"
if [ -d "$LOG_DIR" ]; then
    PID_FILES=$(find "$LOG_DIR" -name "*.pid" 2>/dev/null)
    if [ -z "$PID_FILES" ]; then
        echo "  ❌ No PID files found"
    else
        echo "$PID_FILES" | while read pidfile; do
            EXPERIMENT=$(basename "$pidfile" .pid)
            PID=$(cat "$pidfile" 2>/dev/null)
            if kill -0 "$PID" 2>/dev/null; then
                echo "  ✅ $EXPERIMENT (PID: $PID) - RUNNING"
            else
                echo "  🔴 $EXPERIMENT (PID: $PID) - STOPPED"
            fi
        done
    fi
else
    echo "  ❌ Log directory not found: $LOG_DIR"
fi
echo ""

# Check status files
echo "📋 Status Files:"
if [ -d "$LOG_DIR" ]; then
    STATUS_FILES=$(find "$LOG_DIR" -name "*_status.json" 2>/dev/null)
    if [ -z "$STATUS_FILES" ]; then
        echo "  ❌ No status files found"
    else
        echo "$STATUS_FILES" | while read statusfile; do
            if [ -f "$statusfile" ]; then
                EXPERIMENT=$(basename "$statusfile" _status.json)
                STATUS=$(grep -o '"status"[[:space:]]*:[[:space:]]*"[^"]*"' "$statusfile" | cut -d'"' -f4)
                PID=$(grep -o '"pid"[[:space:]]*:[[:space:]]*[0-9]*' "$statusfile" | awk '{print $2}')
                echo "  📊 $EXPERIMENT: $STATUS (PID: $PID)"
            fi
        done
    fi
fi
echo ""

# Recent log files
echo "📄 Recent Log Files (last 5):"
if [ -d "$LOG_DIR" ]; then
    ls -lt "$LOG_DIR"/*.log 2>/dev/null | head -n 5 | while read line; do
        FILE=$(echo $line | awk '{print $NF}')
        SIZE=$(echo $line | awk '{print $5}')
        DATE=$(echo $line | awk '{print $6, $7, $8}')
        echo "  📝 $(basename $FILE) - ${SIZE} bytes - $DATE"
    done
else
    echo "  ❌ No log files found"
fi
echo ""

# Memory status
echo "💾 Memory Status:"
if command -v free &> /dev/null; then
    free -h | grep -E "Mem|Swap" | while read line; do
        echo "  $line"
    done
else
    echo "  ⚠️  'free' command not available"
fi
echo ""

# Disk space
echo "💿 Disk Space:"
df -h / | tail -1 | awk '{print "  Total: " $2 " | Used: " $3 " | Available: " $4 " | " $5 " full"}'
echo ""

# tmux sessions
echo "🖥️  tmux Sessions:"
if command -v tmux &> /dev/null; then
    SESSIONS=$(tmux ls 2>/dev/null)
    if [ -z "$SESSIONS" ]; then
        echo "  ❌ No tmux sessions found"
    else
        echo "$SESSIONS" | while read line; do
            echo "  ✅ $line"
        done
    fi
else
    echo "  ⚠️  tmux not installed"
fi
echo ""

# Actionable commands
echo "=========================================="
echo "🎯 Suggested Actions:"
echo "=========================================="

# Check if anything is running
if ps aux | grep -q "simulation_runner" | grep -v grep; then
    echo "✅ Experiments are running!"
    echo ""
    echo "To monitor:"
    echo "  python scripts/monitor_experiment.py --list"
    echo "  tail -f $LOG_DIR/<experiment>_*.log"
    echo ""
    echo "To stop:"
    PID=$(ps aux | grep "simulation_runner" | grep -v grep | awk '{print $2}' | head -1)
    echo "  kill $PID"
else
    echo "❌ No experiments running"
    echo ""
    echo "To start an experiment:"
    echo "  ./scripts/run_experiment_robust.sh experiments/configs/baseline_100_clients_optimized.yaml"
    echo ""
    echo "Or with auto-restart:"
    echo "  ./scripts/supervise_experiment.sh experiments/configs/baseline_100_clients_optimized.yaml"
fi

echo ""
echo "To check for OOM kills:"
echo "  ./scripts/check_system_resources.sh"
echo ""
echo "To start web dashboard:"
echo "  python scripts/monitoring_api.py"
echo ""
echo "=========================================="
