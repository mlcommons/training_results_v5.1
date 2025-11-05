#!/bin/bash
export WARMUP=True
export DGXNGPU=8
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )

export DGXNNODES=2
export WALLTIME_MINUTES=40
export WALLTIME=$(( (${NEXP:-1} * WALLTIME_MINUTES) ))

export NCCL_SOCKET_IFNAME=ens11np0,ens12np0,ens21np0,ens22np0,ens31np0,ens32np0,ens41np0,ens42np0

export LORA_A2A=1
export POSSIBLE_USER_WARNINGS=0
export CUDNN_FRONTEND_ATTN_DP_WORKSPACE_LIMIT=0

export MAX_STEPS=700
export TRAINING_LOSS_LOG_FREQ=10
export TP=1
export PP=1
export CP=2
export SP=False
export VBOOST_VALUE=1
export MBS=1
export LR=0.0005
export MINIBS=1
export SKIP_EVALS=3
export VAL_CHECK_INTERVAL=384
export HYDRA_FULL_ERROR=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

export SEED=$RANDOM

export FP8_DPA=0    
export FP8=True
export FP8_AMAX_ALGO=most_recent
export FP8_REDUCE_AMAX=False
export FP8_AMAX_HISTORY=4
export FP8_ACTIVATION=True

export ACG=full && export ACM=block && export ACL=21
export FUSED_SOFTMAX=0
export RMSNORM_CAST=0


export USE_HIPBLASLT=1
export TORCH_BLAS_PREFER_HIPBLASLT=1
export NVTE_USE_RMSNORM_TRITON=1
export ENABLE_TRANSPOSE_CACHE=0

export MLPERF_SUBMISSION_ORG="MangoBoost"
export MLPERF_SUBMISSION_PLATFORM="MI300X"
export NVIDIA_PYTORCH_VERSION=25.04
