#!/bin/bash

export DATADIR="/run/tmpfs/dataset/llama2_70b_lora/gov_report"
export MODEL="/run/tmpfs/dataset/llama2_70b_lora/model"
export LOGDIR="/mnt/nfs_server/ml_commons/results/llama2_70b_lora"
export CONT="/mnt/nfs_server/ml_commons/images/llama2_70b_lora_20251007.sqsh"

export NCCL_NET_PLUGIN=none
export NCCL_TEST=0
export NEXP=10

source config_DGXB200_1x8x1xtp1pp1cp1.sh 

sbatch --job-name=lora --exclusive --gpus-per-node=${DGXNGPU} -N ${DGXNNODES} --time=${WALLTIME} run.sub  