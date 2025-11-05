#!/bin/bash

# MLPerf Training v5.1 - Llama 3.1 8B
# One-time dataset initialization script

set -euo pipefail

echo "==========================================="
echo "Initializing c4/en/3.0.1 dataset for Llama 3.1 8B"
echo "==========================================="
echo ""

# ==============================================================================
# Configuration
# ==============================================================================
# Set paths where datasets will be downloaded
# Default to current directory, but can be overridden with environment variables
TOKENIZER_PATH="${TOKENIZER_PATH:-.}"
PREPROCESSED_PATH="${PREPROCESSED_PATH:-.}"

echo "Configuration:"
echo "  - Tokenizer path: ${TOKENIZER_PATH}"
echo "  - Preprocessed dataset path: ${PREPROCESSED_PATH}"
echo ""

# ==============================================================================
# Download Llama 3.1 8B Tokenizer
# ==============================================================================
echo "Step 1: Downloading Llama 3.1 8B Tokenizer"
echo "--------------------------------------"
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) \
    -d llama3_1_8b_tokenizer \
    https://training.mlcommons-storage.org/metadata/llama-3-1-8b-tokenizer.uri

echo ""
echo "Tokenizer download complete!"
echo ""

# ==============================================================================
# Download Preprocessed C4 Dataset
# ==============================================================================
echo "Step 2: Downloading Preprocessed C4 Dataset"
echo "--------------------------------------"
echo "NOTE: This is a large dataset and may take significant time to download"
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) \
    -d llama3_1_8b_preprocessed_c4_dataset \
    https://training.mlcommons-storage.org/metadata/llama-3-1-8b-preprocessed-c4-dataset.uri

echo ""
echo "Dataset download complete!"
echo ""

# ==============================================================================
# Organize Dataset Structure
# ==============================================================================
echo "Step 3: Organizing Dataset Structure"
echo "--------------------------------------"

# Create 8b directory structure
if [ ! -d "8b" ]; then
    mkdir -p 8b
    echo "Created 8b directory"
fi

# Move preprocessed dataset
if [ -d "llama3_1_8b_preprocessed_c4_dataset" ]; then
    echo "Moving preprocessed dataset to 8b/"
    mv llama3_1_8b_preprocessed_c4_dataset/* 8b/ 2>/dev/null || true
    rmdir llama3_1_8b_preprocessed_c4_dataset 2>/dev/null || true
fi

# Move tokenizer
if [ -d "llama3_1_8b_tokenizer" ]; then
    echo "Moving tokenizer to 8b/tokenizer/"
    mkdir -p 8b/tokenizer
    mv llama3_1_8b_tokenizer/* 8b/tokenizer/ 2>/dev/null || true
    rmdir llama3_1_8b_tokenizer 2>/dev/null || true
fi

# ==============================================================================
# Verify Dataset
# ==============================================================================
echo ""
echo "Step 4: Verifying Dataset"
echo "--------------------------------------"
if [ -d "8b" ]; then
    echo "Dataset directory structure:"
    ls -lh 8b/ | head -20
    echo ""
    if [ -d "8b/tokenizer" ]; then
        echo "Tokenizer files:"
        ls -lh 8b/tokenizer/
    fi
else
    echo "WARNING: 8b directory not found!"
fi

# ==============================================================================
# Complete
# ==============================================================================
echo ""
echo "==========================================="
echo "Dataset Initialization Complete!"
echo "==========================================="
echo ""
echo "Dataset location: $(pwd)/8b"
echo ""
echo "Next steps:"
echo "  1. Verify dataset integrity"
echo "  2. Update DATADIR environment variable to point to this location"
echo "  3. Run training with: ./run_and_time.sh"
echo ""