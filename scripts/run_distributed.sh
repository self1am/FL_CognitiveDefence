#!/bin/bash
# scripts/run_distributed.sh
# Script for running experiments across multiple machines

set -e

# Configuration
SERVER_VM="your-server-vm-ip"
CLIENT_VMS=("your-client-vm1-ip" "your-client-vm2-ip")
CONFIG_FILE="experiments/configs/basic_cognitive_defense.yaml"

echo "Setting up distributed federated learning experiment..."

# Function to run command on remote host
run_remote() {
    local host=$1
    local command=$2
    echo "Running on $host: $command"
    ssh "$host" "$command"
}

# Function to copy files to remote host
copy_to_remote() {
    local host=$1
    local local_path=$2
    local remote_path=$3
    echo "Copying $local_path to $host:$remote_path"
    scp -r "$local_path" "$host:$remote_path"
}

# Setup server VM
echo "Setting up server VM: $SERVER_VM"
copy_to_remote "$SERVER_VM" "." "/tmp/fl-project"
run_remote "$SERVER_VM" "cd /tmp/fl-project && chmod +x scripts/setup_project.sh && ./scripts/setup_project.sh"

# Setup client VMs
for vm in "${CLIENT_VMS[@]}"; do
    echo "Setting up client VM: $vm"
    copy_to_remote "$vm" "." "/tmp/fl-project"
    run_remote "$vm" "cd /tmp/fl-project && chmod +x scripts/setup_project.sh && ./scripts/setup_project.sh"
done

echo "All VMs setup completed. Ready to run distributed experiment."

# Start server
echo "Starting server on $SERVER_VM..."
run_remote "$SERVER_VM" "cd /tmp/fl-project && nohup python -m src.orchestration.experiment_runner --config $CONFIG_FILE > server.log 2>&1 &"

echo "Server started. Check server.log on $SERVER_VM for progress."
echo "Clients will be automatically spawned by the orchestrator."
