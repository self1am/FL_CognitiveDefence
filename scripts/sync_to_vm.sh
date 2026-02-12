#!/bin/bash
# Upload updated scripts and configs to GCP VM
# Usage: ./scripts/sync_to_vm.sh [vm_ip] [vm_user]

VM_IP="${1}"
VM_USER="${2:-miraahanafee}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ -z "$VM_IP" ]; then
    echo "Usage: $0 <vm_ip> [vm_user]"
    echo "Example: $0 35.123.456.789 miraahanafee"
    exit 1
fi

echo "=========================================="
echo "Syncing to VM: ${VM_USER}@${VM_IP}"
echo "=========================================="
echo ""

# Files and directories to sync
SYNC_ITEMS=(
    "scripts/"
    "experiments/configs/baseline_100_clients_optimized.yaml"
    "requirements.txt"
    "VM_COMMANDS.md"
    "VM_QUICK_START.md"
    "PRODUCTION_RUNNING_GUIDE.md"
)

for item in "${SYNC_ITEMS[@]}"; do
    echo "📤 Syncing: $item"
    if [ -d "${PROJECT_DIR}/${item}" ]; then
        # Directory
        rsync -avz --progress "${PROJECT_DIR}/${item}" "${VM_USER}@${VM_IP}:~/FL_CognitiveDefence/${item%/*}/"
    else
        # File
        rsync -avz --progress "${PROJECT_DIR}/${item}" "${VM_USER}@${VM_IP}:~/FL_CognitiveDefence/${item}"
    fi
done

echo ""
echo "✅ Sync complete!"
echo ""
echo "Next steps on VM:"
echo "  ssh ${VM_USER}@${VM_IP}"
echo "  cd FL_CognitiveDefence"
echo "  chmod +x scripts/*.sh"
echo "  ./scripts/what_is_running.sh"
