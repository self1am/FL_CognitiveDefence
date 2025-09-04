# !/bin/bash
# scripts/setup_project.sh
set -e

echo "Setting up Federated Cognitive defence project..."

# Create directory structure
echo "Creating directory structure..."
mkdir -p src/{attacks,defences,datasets,models,clients,server,orchestration,utils}
mkdir -p experiments/{configs,scripts,results}
mkdir -p tests/{unit,integration}
mkdir -p logs
mkdir -p data

# Create __init__.py files
find src -type d -exec touch {}/__init__.py \;

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Create git repository
if [ ! -d ".git" ]; then
    echo "Initializing git repository..."
    git init
    
    # Create .gitignore
    cat > .gitignore << EOF
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Virtual environments
venv/
env/
ENV/

# IDEs
.vscode/
.idea/
*.swp
*.swo

# Data and logs
data/
logs/
*.log
*.json

# Experiments
experiments/results/
*.pkl
*.pth

# OS
.DS_Store
Thumbs.db
EOF

    git add .
    git commit -m "Initial project structure"
    echo "Git repository initialized"
fi

echo "Project setup completed!"
