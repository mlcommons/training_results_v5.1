#!/bin/bash

# MLPerf Training v5.1 - Llama 3.1 8B
# One-time setup script for B200 8x8 system

set -euo pipefail

echo "Setting up environment for Llama 3.1 8B training on B200 8x8..."

# ==============================================================================
# Prerequisites - NGC Registry Setup
# ==============================================================================
echo "Step 1: NGC Container Registry Setup"
echo "--------------------------------------"
echo "Please ensure you have completed the following manual steps:"
echo "  1. Docker login to NGC registry:"
echo "     docker login nvcr.io"
echo "     Username: \$oauthtoken"
echo "     Password: <your nvidia ngc token>"
echo ""
echo "  2. For Enroot NGC login (if using Slurm/Enroot):"
echo "     Create /mnt/local_disk/enroot/config/.credentials with:"
echo "     machine nvcr.io login \$oauthtoken password <nvidia ngc token>"
echo ""

# ==============================================================================
# Container Setup
# ==============================================================================
echo "Step 2: Pull NGC Container"
echo "--------------------------------------"
export CONT=nvcr.io/nvdlfwea/mlperftv51/llama31_8b-amd:20251006

echo "Container: ${CONT}"
echo "Container documentation: https://registry.ngc.nvidia.com/orgs/nvdlfwea/teams/mlperftv51/containers/llama31_8b-amd"

# Uncomment if you want to pull the container automatically
# docker pull ${CONT}
echo "To pull container manually, run: docker pull ${CONT}"

# ==============================================================================
# Extract LLM Code from Container (Optional)
# ==============================================================================
echo ""
echo "Step 3: Extract LLM Code from Container (if needed)"
echo "--------------------------------------"
echo "To copy llm folder from container:"
echo "  1. docker run -it -d --gpus all --network=host --ipc=host ${CONT} bash"
echo "  2. docker cp <CONTAINER_ID>:/workspace/llm ."
echo ""

# ==============================================================================
# Install Python Dependencies
# ==============================================================================
echo "Step 4: Install Python Dependencies"
echo "--------------------------------------"
if [ -f requirements.txt ]; then
    echo "Installing Python requirements..."
    pip install -r requirements.txt
else
    echo "No requirements.txt found, skipping pip install"
fi

# ==============================================================================
# Apply Patches (if needed)
# ==============================================================================
echo ""
echo "Step 5: Apply Patches"
echo "--------------------------------------"
if [ -f pytorch_ckpt.patch ]; then
    echo "PyTorch checkpoint patch available: pytorch_ckpt.patch"
    echo "Apply manually if needed"
fi

if [ -f te.patch ]; then
    echo "TransformerEngine patch available: te.patch"
    echo "Apply manually if needed"
fi

# ==============================================================================
# Verify GPU/System
# ==============================================================================
echo ""
echo "Step 6: Verify GPU Availability"
echo "--------------------------------------"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
else
    echo "nvidia-smi not found - GPU verification skipped"
fi

# ==============================================================================
# System Configuration Notes
# ==============================================================================
echo ""
echo "=============================================="
echo "Setup Complete!"
echo "=============================================="
echo ""
echo "System Configuration:"
echo "  - Container: nvcr.io/nvdlfwea/mlperftv51/llama31_8b-amd:20251006"
echo "  - Config file: config_DGXB200_8x8x1xtp1pp1cp2_8b.sh"
echo "  - System: 8 nodes × 8 GPUs = 64 GPUs total"
echo "  - Global Batch Size: 32"
echo "  - Learning Rate: 0.0008"
echo ""
echo "Configuration file format: config_<SYSTEM>_<NODES>x<GPUS/NODE>x<BATCH/GPU>xtpXppYcpZ.sh"
echo "  - TP (X): Tensor Parallel"
echo "  - PP (Y): Pipeline Parallel"
echo "  - CP (Z): Context Parallel"
echo ""
echo "Next Steps:"
echo "  1. Initialize datasets with: ./init_datasets.sh"
echo "  2. Configure environment variables (DATADIR, LOGDIR, CONT)"
echo "  3. Run training with: ./run_and_time.sh"
echo ""