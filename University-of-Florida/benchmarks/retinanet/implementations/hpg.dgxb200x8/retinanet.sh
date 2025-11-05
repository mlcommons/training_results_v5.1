#!/bin/bash

export NEXP=7
export DGXNNODES=${DGXNNODES:-8}

export NCCL_LIB_DIR="<IMPLEMENTATION_PATH>/nccl/build/lib"
export DATADIR="<DATASET_PATH>/ssd_dataset/open-images-v6"
export BACKBONE_DIR="<MODEL_PATH>/backbone"
export LOGDIR="<RESULTS_PATH>/results/hpg.dgxb200x8.n${DGXNNODES}/retinanet"
mkdir -p "${LOGDIR}"
export CONT="<CONTAINER_PATH>/retinanet.sif"

# Load appropriate config based on number of nodes
if [ "$DGXNNODES" -eq 1 ]; then
    source $(dirname ${BASH_SOURCE[0]})/config_DGXB200_001x08x032.sh
elif [ "$DGXNNODES" -eq 2 ]; then
    source $(dirname ${BASH_SOURCE[0]})/config_DGXB200_002x08x016.sh
elif [ "$DGXNNODES" -eq 4 ]; then
    source $(dirname ${BASH_SOURCE[0]})/config_DGXB200_004x08x008.sh
elif [ "$DGXNNODES" -eq 8 ]; then
    source $(dirname ${BASH_SOURCE[0]})/config_DGXB200_008x08x004.sh
elif [ "$DGXNNODES" -eq 16 ]; then
    source $(dirname ${BASH_SOURCE[0]})/config_DGXB200_016x08x002.sh
else
    echo "Error: Unsupported DGXNNODES value: ${DGXNNODES}"
    echo "Supported values: 1, 2, 4, 8, 16"
    exit 1
fi

source $(dirname ${BASH_SOURCE[0]})/config_hpg.sh
sbatch -A rc-rse  -p hpg-b200 --gpus-per-node 8 --reservation=mlperf-training -N $DGXNNODES -t $((60 + $WALLTIME)) retinanet.sub
