#!/bin/bash

source $(dirname ${BASH_SOURCE[0]})/config_common.sh
source $(dirname ${BASH_SOURCE[0]})/config_common_fp4.sh

# hyperparameters
export MAX_STEPS=700
export LR=0.0006
export MINIBS=2
export CP=2
export MCORE_CUDA_GRAPH=1
export NUM_WORKERS=4

export HEALING_ITER=350

# system parameters
export DGXNNODES=1
export DGXNGPU=8
export WALLTIME_RUNANDTIME=35
export WALLTIME=$((5 + ${NEXP:-1} * ($WALLTIME_RUNANDTIME + 5)))



