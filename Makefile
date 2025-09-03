# Makefile
.PHONY: setup install test clean run-basic run-adaptive run-large help

# Default target
help:
	@echo "Available targets:"
	@echo "  setup       - Setup project structure and dependencies"
	@echo "  install     - Install Python dependencies"
	@echo "  test        - Run tests"
	@echo "  clean       - Clean generated files"
	@echo "  run-basic   - Run basic cognitive defense experiment"
	@echo "  run-adaptive - Run adaptive attack scenario"
	@echo "  run-large   - Run large scale test"

# Setup project
setup:
	chmod +x scripts/setup_project.sh
	./scripts/setup_project.sh

# Install dependencies
install:
	pip install -r requirements.txt
	pip install -e .

# Run tests
test:
	python -m pytest tests/ -v

# Clean generated files
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf build/ dist/ *.egg-info/
	rm -rf logs/*.log experiments/results/*.json

# Run experiments
run-basic:
	python -m src.orchestration.experiment_runner --config experiments/configs/basic_cognitive_defense.yaml

run-adaptive:
	python -m src.orchestration.experiment_runner --config experiments/configs/adaptive_attack_scenario.yaml

run-large:
	python -m src.orchestration.experiment_runner --config experiments/configs/large_scale_test.yaml

# Development targets
dev-install: install
	pip install pytest black flake8

format:
	black src/ tests/

lint:
	flake8 src/ tests/
