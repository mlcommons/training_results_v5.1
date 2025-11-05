#!/bin/bash

source $(dirname ${BASH_SOURCE[0]})/config_common.sh

# hyperparameters
export MAX_STEPS=896
export LR=0.0005
export MINIBS=2
export CP=1
export FP8_ACT=1
export NCCL_NVLS_ENABLE=0

# system parameters
export VBOOST_VALUE=0
export DGXNNODES=1
export DGXNGPU=8
export WALLTIME_RUNANDTIME=UNLIMITED
export WALLTIME=UNLIMITED
