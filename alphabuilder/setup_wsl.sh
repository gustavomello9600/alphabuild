#!/bin/bash
set -e

echo "=== AlphaBuilder WSL Setup (Conda Version) ==="
echo "This script installs Miniforge (Conda) and sets up the environment with FEniCSx."

# 1. Install Miniforge if Conda is not found
if ! command -v conda &> /dev/null; then
    echo "Conda not found. Installing Miniforge..."
    wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh" -O Miniforge3.sh
    bash Miniforge3.sh -b -p $HOME/miniforge3
    
    # Initialize conda for bash
    eval "$($HOME/miniforge3/bin/conda shell.bash hook)"
    conda init
    
    echo "Miniforge installed."
else
    echo "Conda detected."
fi

# 2. Create Conda Environment
ENV_NAME="alphabuilder"

if conda info --envs | grep -q "$ENV_NAME"; then
    echo "Environment '$ENV_NAME' already exists."
else
    echo "Creating environment '$ENV_NAME'..."
    # Install FEniCSx, PyTorch, and other deps from conda-forge
    conda create -n $ENV_NAME -c conda-forge -y \
        python=3.10 \
        fenics-dolfinx mpich pyvista \
        numpy scipy matplotlib \
        tqdm
fi

# 3. Activate and Install Pip Deps
echo "Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME

echo "Installing remaining Python packages via pip..."
pip install torch monai einops

echo "=== Setup Complete ==="
echo "To run the harvester:"
echo "1. conda activate $ENV_NAME"
echo "2. python run_data_harvest.py --episodes 50"
