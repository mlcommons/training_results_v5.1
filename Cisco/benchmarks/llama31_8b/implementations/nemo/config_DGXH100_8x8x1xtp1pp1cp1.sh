source $(dirname ${BASH_SOURCE[0]})/config_common.sh

export MICRO_BATCH_SIZE=1
export MINIBS=1

export LR=0.0008
export MIN_LR=0.00008

export TENSOR_MODEL_PARALLEL=1
export PIPELINE_MODEL_PARALLEL=1
export INTERLEAVED_PIPELINE=1
export CONTEXT_PARALLEL=1
export TP_COMM_OVERLAP=True

export DGXNNODES=8
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )

export WALLTIME_RUNANDTIME=420
export WALLTIME=$((5 + ${NEXP:-1} * ($WALLTIME_RUNANDTIME + 5)))

export MODEL_SIZE="8b"

export MAX_STEPS=1200000
export WARMUP_STEPS=192

# 1_200_000 - 192 = 1_199_808
export MAX_STEPS_FOR_LR_SCHED=1199808

# #fix OOM
export BUCKET_CAP_MB=125
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"
export CUBLAS_WORKSPACE_CONFIG=":16:8"
export MCORE_CUDA_GRAPH=False
