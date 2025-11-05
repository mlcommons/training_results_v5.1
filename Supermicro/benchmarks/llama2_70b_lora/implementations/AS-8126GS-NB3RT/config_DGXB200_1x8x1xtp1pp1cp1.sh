#!/bin/bash

source $(dirname ${BASH_SOURCE[0]})/config_common.sh

# hyperparameters
export MAX_STEPS=800
export LR=0.00055
export MINIBS=1
export CP=1
export FP8_ACT=1
export NCCL_NVLS_ENABLE=0

# system parameters
export VBOOST_VALUE=0
export DGXNNODES=1
export DGXNGPU=8
export WALLTIME_RUNANDTIME=25
export WALLTIME=$((5 + ${NEXP:-1} * ($WALLTIME_RUNANDTIME + 5)))



unset NCCL_SHARP_GROUP_SIZE_THRESH 
unset NCCL_CFG_PATH 
export NCCL_IB_DISABLE=1 
export NCCL_NET=Socket 
export NCCL_P2P_LEVEL=NVL 
export NCCL_SHARP_DISABLE=1 
export FULL_CUDA_GRAPH=0 
export MCORE_CUDA_GRAPH=0