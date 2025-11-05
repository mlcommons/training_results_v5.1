#!/bin/bash

export DATADIR="/data/llama31_8b/"
export LOGDIR="/data/llama31_8b/logs/"
export CONT="/mnt/nfs_server/ml_commons/images/llama31_8b_20251007.sqsh"

export NCCL_NET_PLUGIN=none
export HANG_MONITOR_TIMEOUT=0
export NEXP=10

source config_DGXB200_1x8x2xtp1pp1cp1_8b.sh  
sbatch --job-name=llama3 --exclusive --gpus-per-node=${DGXNGPU} -N ${DGXNNODES} --time=${WALLTIME} run.sub