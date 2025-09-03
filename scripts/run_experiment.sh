#!/bin/bash
# scripts/run_experiment.sh
set -e

# Default values
CONFIG_FILE="experiments/configs/basic_cognitive_defense.yaml"
SERVER_HOST="localhost"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --server-host)
            SERVER_HOST="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [--config CONFIG_FILE] [--server-host HOST]"
            echo "  --config: Path to experiment configuration YAML (default: experiments/configs/basic_cognitive_defense.yaml)"
            echo "  --server-host: Server hostname/IP (default: localhost)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "Running experiment with config: $CONFIG_FILE"
echo "Server host: $SERVER_HOST"

# Update server address in config if needed
if [ "$SERVER_HOST" != "localhost" ]; then
    # Create temporary config with updated server address
    TEMP_CONFIG=$(mktemp)
    sed "s/0.0.0.0:8080/${SERVER_HOST}:8080/g" "$CONFIG_FILE" > "$TEMP_CONFIG"
    CONFIG_FILE="$TEMP_CONFIG"
    echo "Updated server address to: ${SERVER_HOST}:8080"
fi

# Run experiment
python -m src.orchestration.experiment_runner --config "$CONFIG_FILE"

# Cleanup temporary config if created
if [[ "$CONFIG_FILE" == /tmp/* ]]; then
    rm "$CONFIG_FILE"
fi

echo "Experiment completed!"
