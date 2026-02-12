#!/bin/bash
# Script to check system resources and detect OOM kills
# Run this on your GCP VM to diagnose issues

echo "=========================================="
echo "System Resource Check"
echo "=========================================="
echo ""

# Memory information
echo "📊 Memory Status:"
free -h
echo ""

# Check for OOM kills in system logs
echo "🔍 Checking for OOM (Out of Memory) kills in the last 24 hours:"
if [ -f /var/log/syslog ]; then
    sudo grep -i "out of memory\|oom\|killed process" /var/log/syslog | tail -n 20
elif [ -f /var/log/messages ]; then
    sudo grep -i "out of memory\|oom\|killed process" /var/log/messages | tail -n 20
else
    journalctl --since "24 hours ago" | grep -i "out of memory\|oom\|killed process" | tail -n 20
fi
echo ""

# Check for Python processes
echo "🐍 Python processes:"
ps aux | grep python | grep -v grep
echo ""

# Disk usage
echo "💾 Disk Usage:"
df -h
echo ""

# Check if Ray is running
echo "☢️  Ray processes:"
ps aux | grep ray | grep -v grep
echo ""

# Memory usage by top processes
echo "📈 Top 10 memory-consuming processes:"
ps aux --sort=-%mem | head -n 11
echo ""

# Swap usage
echo "💿 Swap Usage:"
swapon --show
echo ""

# Check ulimit
echo "⚙️  Process limits (ulimit):"
ulimit -a
echo ""

echo "=========================================="
echo "Analysis Complete"
echo "=========================================="
echo ""
echo "💡 Tips:"
echo "1. If you see OOM kills, reduce experiment scale or add swap"
echo "2. If Ray processes are stuck, consider restarting Ray"
echo "3. Monitor memory usage during experiments with: watch -n 5 free -h"
echo "4. Check specific experiment logs in logs/experiments/ directory"
