#!/bin/bash
# optimize_gcp_instance.sh
# Optimize GCP instance for production FL experiments (100 clients, 64GB memory)

set -e

echo "========================================="
echo "GCP Instance Optimization for FL Experiments"
echo "========================================="
echo ""

# Check if running as root for certain operations
if [ "$EUID" -ne 0 ]; then 
    echo "⚠️  Some optimizations require sudo. Running with sudo for those..."
fi

# 1. System Memory Check
echo "1. Checking system memory..."
total_mem=$(free -h | grep Mem | awk '{print $2}')
available_mem=$(free -h | grep Mem | awk '{print $7}')
echo "   Total Memory: $total_mem"
echo "   Available Memory: $available_mem"
echo ""

# 2. Increase file descriptors
echo "2. Increasing file descriptors limit..."
if [ -w /etc/security/limits.conf ]; then
    sudo tee -a /etc/security/limits.conf > /dev/null << EOF
* soft nofile 65536
* hard nofile 65536
* soft nproc 32768
* hard nproc 32768
EOF
    echo "   ✓ File descriptors increased"
else
    echo "   ⚠️  Cannot modify /etc/security/limits.conf (requires sudo)"
fi
echo ""

# 3. TCP optimizations
echo "3. Optimizing TCP settings..."
if [ -w /etc/sysctl.conf ] || [ -w /etc/sysctl.d/ ]; then
    sudo tee /etc/sysctl.d/99-fl-optimization.conf > /dev/null << EOF
# TCP optimizations for FL experiments
net.core.rmem_max=134217728
net.core.wmem_max=134217728
net.ipv4.tcp_rmem=4096 87380 67108864
net.ipv4.tcp_wmem=4096 65536 67108864
net.core.netdev_max_backlog=5000
net.ipv4.tcp_max_syn_backlog=5000
net.ipv4.tcp_tw_reuse=1
EOF
    sudo sysctl -p /etc/sysctl.d/99-fl-optimization.conf > /dev/null 2>&1 || true
    echo "   ✓ TCP settings optimized"
else
    echo "   ⚠️  Cannot modify sysctl (requires sudo)"
fi
echo ""

# 4. CPU governor
echo "4. Setting CPU governor to performance..."
if command -v cpupower &> /dev/null; then
    sudo cpupower frequency-set -g performance 2>/dev/null || true
    echo "   ✓ CPU governor set to performance"
else
    echo "   ⚠️  cpupower not available, skipping CPU governor optimization"
fi
echo ""

# 5. Python environment check
echo "5. Checking Python environment..."
if command -v python3 &> /dev/null; then
    python_version=$(python3 --version)
    echo "   Python: $python_version"
else
    echo "   ✗ Python3 not found!"
    exit 1
fi

if python3 -c "import torch; print(f'   PyTorch version: {torch.__version__}'); print(f'   CUDA available: {torch.cuda.is_available()}')" 2>/dev/null; then
    :
else
    echo "   ⚠️  PyTorch not installed or CUDA check failed"
fi
echo ""

# 6. Virtual environment setup
echo "6. Setting up Python virtual environment..."
if [ ! -d "fl_env" ]; then
    python3 -m venv fl_env
    echo "   ✓ Virtual environment created at ./fl_env"
else
    echo "   ℹ Virtual environment already exists at ./fl_env"
fi
echo ""

# 7. Disk space check
echo "7. Checking disk space..."
disk_usage=$(df -h / | tail -1 | awk '{print $5}')
disk_available=$(df -h / | tail -1 | awk '{print $4}')
echo "   Disk usage: $disk_usage"
echo "   Available: $disk_available"

if [ "${disk_usage%\%}" -gt 80 ]; then
    echo "   ⚠️  Disk usage is high (>80%). Consider cleaning up old logs."
fi
echo ""

# 8. Swap configuration
echo "8. Checking swap configuration..."
swap_size=$(free -h | grep Swap | awk '{print $2}')
if [ "$swap_size" == "0B" ]; then
    echo "   ℹ No swap available (OK for 64GB system)"
else
    echo "   Swap size: $swap_size"
fi
echo ""

# 9. Environment variables for optimization
echo "9. Recommended environment variables for experiments:"
cat << 'EOF'
   Add these to your shell before running experiments:
   
   export OMP_NUM_THREADS=8
   export MKL_NUM_THREADS=8
   export CUDA_LAUNCH_BLOCKING=0
   export TORCH_NUM_THREADS=8
   export MALLOC_MMAP_THRESHOLD_=131072
   
   Or add to .bashrc/.zshrc:
   
   # FL Optimization
   export OMP_NUM_THREADS=8
   export MKL_NUM_THREADS=8
   export CUDA_LAUNCH_BLOCKING=0
   export TORCH_NUM_THREADS=8
   export MALLOC_MMAP_THRESHOLD_=131072

EOF

# 10. Create optimization profile
echo "10. Creating optimization profile..."
cat > ~/.fl_optimization.sh << 'EOF'
#!/bin/bash
# FL Optimization Profile
# Source this file before running experiments: source ~/.fl_optimization.sh

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export CUDA_LAUNCH_BLOCKING=0
export TORCH_NUM_THREADS=8
export MALLOC_MMAP_THRESHOLD_=131072

# Increase file descriptors
ulimit -n 65536
ulimit -u 32768

echo "✓ FL optimization profile loaded"
EOF

chmod +x ~/.fl_optimization.sh
echo "   ✓ Optimization profile created at ~/.fl_optimization.sh"
echo ""

echo "========================================="
echo "✓ Optimization Complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo "1. Source the optimization profile:"
echo "   source ~/.fl_optimization.sh"
echo ""
echo "2. Install dependencies:"
echo "   source fl_env/bin/activate"
echo "   pip install -r requirements.txt"
echo ""
echo "3. Run experiments:"
echo "   ./scripts/run_production_experiments.sh --all"
echo ""
