#!/bin/bash

source $(dirname ${BASH_SOURCE[0]})/config_common.sh

# hyperparameters
export MAX_STEPS=800
export LR=0.0005
export MINIBS=2
export CP=1
export MCORE_CUDA_GRAPH=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export NUM_WORKERS=4

# system parameters
export VBOOST_VALUE=0
export DGXNNODES=1
export DGXNGPU=4
export WALLTIME_RUNANDTIME=35
export WALLTIME=$((5 + ${NEXP:-1} * ($WALLTIME_RUNANDTIME + 5)))
