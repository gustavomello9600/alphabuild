#!/bin/bash
# setup_arch_intel.sh

echo "Starting setup in $(pwd)"

# Ensure we are in the project root
# If the terminal opens in user home, we try to navigate
if [ ! -f "environment_arch_intel.yml" ]; then
    echo "environment_arch_intel.yml not found in current directory."
    echo "Attempting to find project directory..."
    # Try typical WSL mount path
    if [ -d "/mnt/c/Users/admin/Projetos/alphabuild" ]; then
        cd /mnt/c/Users/admin/Projetos/alphabuild
        echo "Changed directory to $(pwd)"
    else
        echo "Could not find project directory. Please execute this script from c:\Users\admin\Projetos\alphabuild"
        exit 1
    fi
fi

# Create environment
echo "Creating Conda environment 'alphabuilder_intel'..."
mamba env create -f environment_arch_intel.yml || { echo "Environment creation failed (or already exists)"; }

# Activate and Run Verification
echo "Activating environment and running verification..."
# Need to use source for activation in script
source $(conda info --base)/etc/profile.d/conda.sh
conda activate alphabuilder_intel

echo "Verifying PyTorch and IPEX..."
python -c "import torch; import intel_extension_for_pytorch as ipex; print(f'Torch: {torch.__version__}'); print(f'XPU Available: {torch.xpu.is_available()}'); print(f'Device Name: {torch.xpu.get_device_name(0) if torch.xpu.is_available() else 'None'}')"

echo "Running Self Play Test..."
python run_selfplay.py --episodes 1 --max-steps 10 --device xpu

echo "Done!"
