#!/bin/bash

export NEXP=12
export DGXNNODES=${DGXNNODES:-8}

export NCCL_LIB_DIR="<IMPLEMENTATION_PATH>/nccl/build/lib"
export DATAROOT="<DATASET_PATH>/energon"
export LOGDIR="<RESULTS_PATH>/results/hpg.dgxb200x8.n${DGXNNODES}/flux1"
mkdir -p "${LOGDIR}"
export CONT="<CONTAINER_PATH>/flux.sif"

# Load appropriate config based on number of nodes
if [ "$DGXNNODES" -eq 2 ]; then
    source $(dirname ${BASH_SOURCE[0]})/config_DGXB200_02x08x32.sh
elif [ "$DGXNNODES" -eq 9 ]; then
    source $(dirname ${BASH_SOURCE[0]})/config_DGXB200_09x08x32.sh
else
    source $(dirname ${BASH_SOURCE[0]})/config_DGXB200_scale.sh
fi

source $(dirname ${BASH_SOURCE[0]})/config_hpg.sh
sbatch -A rc-rse  -p hpg-b200 --gpus-per-node 8 --reservation=mlperf-training -N $DGXNNODES -t $((60 + $WALLTIME)) flux.sub
