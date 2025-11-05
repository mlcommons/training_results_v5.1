#!/bin/bash

source $(dirname ${BASH_SOURCE[0]})/config_common.sh
source $(dirname ${BASH_SOURCE[0]})/config_common_fp4.sh

# hyperparameters
export MAX_STEPS=800
export LR=0.0006
export MINIBS=1
export CP=4
export FP8_ACT=1
export MCORE_CUDA_GRAPH=1
export HEALING_ITER=350

# system parameters
export VBOOST_VALUE=0
export DGXNNODES=4
export DGXNGPU=8
export WALLTIME_RUNANDTIME=20
export WALLTIME=$((5 + ${NEXP:-1} * ($WALLTIME_RUNANDTIME + 5)))
