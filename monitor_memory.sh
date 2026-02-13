#!/usr/bin/env bash
# Add this to your experiment script or run separately in another tab

# Monitor memory and kill if necessary
check_memory() {
    while true; do
        mem_percent=$(free | awk '/^Mem:/ {printf("%.0f", $3/$2 * 100)}')
        swap_percent=$(free | awk '/^Swap:/ {printf("%.0f", $3/$2 * 100)}')
        
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Memory: ${mem_percent}% | Swap: ${swap_percent}%"
        
        # CRITICAL: If memory > 95%, clean up Ray
        if [ "$mem_percent" -gt 95 ]; then
            echo "[WARNING] Memory critical (${mem_percent}%) - cleaning Ray cache"
            python3 -c "import ray; ray.init(ignore_reinit_error=True); ray.shutdown()" 2>/dev/null || true
        fi
        
        # Swap > 80%
        if [ "$swap_percent" -gt 80 ]; then
            echo "[ERROR] Swap critical (${swap_percent}%) - might need to restart"
        fi
        
        sleep 30
    done
}

check_memory
