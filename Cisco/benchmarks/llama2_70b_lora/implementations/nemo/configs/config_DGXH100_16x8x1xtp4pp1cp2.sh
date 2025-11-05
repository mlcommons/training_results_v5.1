#!/bin/bash

source $(dirname ${BASH_SOURCE[0]})/config_common.sh

# hyperparameters
export MAX_STEPS=1024
export MINIBS=1
export LR=0.00035
export TP=4
export CP=2
export TP_COMM_OVERLAP=1
export FP8_DPA=0
export NVTE_FP8_DPA_BWD=0
export SKIP_EVALS=4
export LAYER_CUDA_GRAPH=1
export MC_TP_OVERLAP_RS_DGRAD=False

# system parameters
export DGXNNODES=16
export WALLTIME_RUNANDTIME=20
export WALLTIME=$((5 + ${NEXP:-1} * ($WALLTIME_RUNANDTIME + 5)))
