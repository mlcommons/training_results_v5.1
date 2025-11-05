#!/bin/bash

source $(dirname ${BASH_SOURCE[0]})/config_common.sh

# hyperparameters
export MAX_STEPS=1150
export LR=0.0004
export MINIBS=8
export CP=1
export FP8_ACT=True
export MCORE_CUDA_GRAPH=1

# system parameters
export VBOOST_VALUE=0
export DGXNNODES=1
export DGXNGPU=4
export WALLTIME_RUNANDTIME=40
export WALLTIME=$((5 + ${NEXP:-1} * ($WALLTIME_RUNANDTIME + 5)))

