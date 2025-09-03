# setup.py
from setuptools import setup, find_packages

setup(
    name="federated-cognitive-defense",
    version="0.1.0",
    description="Federated Learning with Cognitive Defense Mechanisms",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "flwr>=1.0.0",
        "torch>=1.12.0",
        "torchvision>=0.13.0",
        "numpy>=1.21.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "pyyaml>=6.0",
        "tqdm>=4.64.0",
        "psutil>=5.9.0",  # For resource monitoring
        "scikit-learn>=1.1.0",
        "pandas>=1.4.0",
    ],
    extras_require={
        "quantum": ["pennylane>=0.28.0", "pennylane-lightning>=0.28.0"],
        "dev": ["pytest>=7.0.0", "black>=22.0.0", "flake8>=4.0.0"],
    },
)