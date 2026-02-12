#!/bin/bash
# Quick setup script for GCP VM
# Run this once after deploying to a new VM

set -e

echo "=========================================="
echo "Setting up FL Cognitive Defence on VM"
echo "=========================================="
echo ""

# Get project directory
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"

# Make scripts executable
echo "📝 Making scripts executable..."
chmod +x scripts/*.sh

# Install Python dependencies
echo "📦 Installing Python dependencies..."
if [ -d "fl_env" ]; then
    source fl_env/bin/activate
elif [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "Creating virtual environment..."
    python3 -m venv fl_env
    source fl_env/bin/activate
fi

pip install -r requirements.txt

# Create log directories
echo "📁 Creating log directories..."
mkdir -p logs/experiments
mkdir -p /tmp/ray_spill

# Check system resources
echo ""
echo "📊 System Resources:"
echo "  CPU cores: $(nproc)"
echo "  Total RAM: $(free -h | grep Mem | awk '{print $2}')"
echo "  Available RAM: $(free -h | grep Mem | awk '{print $7}')"
echo "  Disk space: $(df -h / | tail -1 | awk '{print $4}') available"
echo ""

# Test if Ray can start
echo "🧪 Testing Ray initialization..."
python -c "import ray; ray.init(num_cpus=2, _memory=1000000000); print('✅ Ray OK'); ray.shutdown()"
echo ""

# Print instructions
echo "=========================================="
echo "✅ Setup Complete!"
echo "=========================================="
echo ""
echo "🚀 To run an experiment:"
echo "  ./scripts/run_experiment_robust.sh experiments/configs/baseline_100_clients_optimized.yaml"
echo ""
echo "📊 To monitor experiments:"
echo "  python scripts/monitor_experiment.py --list"
echo "  python scripts/monitor_experiment.py <experiment_name>"
echo ""
echo "🌐 To start web dashboard:"
echo "  python scripts/monitoring_api.py"
echo "  Then setup port forwarding: ssh -L 5000:localhost:5000 user@vm-ip"
echo ""
echo "📖 For more information, see PRODUCTION_RUNNING_GUIDE.md"
echo ""
