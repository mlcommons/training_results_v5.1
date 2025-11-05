#!/bin/bash

source $(dirname ${BASH_SOURCE[0]})/config_common.sh

# hyperparameters
export MAX_STEPS=600
export LR=0.0006
export MINIBS=1
export CP=8
export MCORE_CUDA_GRAPH=1
export NUM_WORKERS=2
#export NCCL_DEBUG=INFO
#export NCCL_DEBUG_SUBSYS=COLL,TIME

# Have to disable MCore CG and use our implementation, otherwise if using  with CP_EVAL, it fails with
# AssertionError: Tried replaying a cudagraph with different arguments than what if was created with!
export LAYER_CUDA_GRAPH=0
export MCORE_CUDA_GRAPH=1

#alpha release
export BINDCMD="bindpcie --cpu=node"


# system parameters
export VBOOST_VALUE=0
export DGXNNODES=144
export DGXNGPU=4
export WALLTIME_RUNANDTIME=10000000 #10
export WALLTIME=$((5 + ${NEXP:-1} * ($WALLTIME_RUNANDTIME + 5)))
