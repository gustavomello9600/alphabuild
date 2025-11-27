#!/bin/bash

# AlphaBuilder Setup Script for Ubuntu 24.04 (WSL)
# Usage: ./setup_wsl.sh

set -e  # Exit on error

echo "=================================================="
echo "AlphaBuilder Setup - Ubuntu 24.04 (WSL)"
echo "=================================================="

# 1. System Update & Dependencies
echo "[1/4] Updating System and Installing Base Dependencies..."
sudo apt-get update && sudo apt-get upgrade -y
sudo apt-get install -y wget curl git build-essential

# 2. Install Miniforge (if not present)
if ! command -v conda &> /dev/null; then
    echo "[2/4] Installing Miniforge (Conda)..."
    wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh" -O Miniforge3.sh
    bash Miniforge3.sh -b -p "$HOME/miniforge3"
    rm Miniforge3.sh
    
    # Initialize conda for bash
    eval "$("$HOME/miniforge3/bin/conda" shell.bash hook)"
    conda init bash
    
    echo "Miniforge installed. You may need to restart your shell later."
else
    echo "[2/4] Conda already installed."
fi

# 3. Create Conda Environment
echo "[3/4] Creating 'alphabuilder' environment from environment.yml..."
if conda env list | grep -q "alphabuilder"; then
    echo "Environment 'alphabuilder' already exists. Updating..."
    conda env update -f environment.yml --prune
else
    conda env create -f environment.yml
fi

# 4. Final Instructions
echo "=================================================="
echo "Setup Complete!"
echo "=================================================="
echo "To activate the environment, run:"
echo "  conda activate alphabuilder"
echo ""
echo "To verify installation:"
echo "  python -c 'import dolfinx; import torch; print(\"Setup OK\")'"
echo "=================================================="
