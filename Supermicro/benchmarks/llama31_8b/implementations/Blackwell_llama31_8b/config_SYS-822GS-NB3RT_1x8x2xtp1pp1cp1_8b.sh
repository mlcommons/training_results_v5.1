source $(dirname ${BASH_SOURCE[0]})/config_common.sh
source $(dirname ${BASH_SOURCE[0]})/config_common_8b.sh
source $(dirname ${BASH_SOURCE[0]})/config_common_cg.sh

export MINIBS=2
export TENSOR_MODEL_PARALLEL=1
export SEQ_PARALLEL=False

export PIPELINE_MODEL_PARALLEL=1
export INTERLEAVED_PIPELINE=null
export CONTEXT_PARALLEL=1

export TP_COMM_OVERLAP=False
export MICRO_BATCH_SIZE=2
export USE_TE_OPS=True
export CE_FUSION_IMPL=te

export LR=0.0004
export WARMUP_STEPS=16
export VAL_CHECK_INTERVAL=768

export DGXNNODES=1
export DGXNGPU=8
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )

export WALLTIME_RUNANDTIME=140
export WALLTIME=$((5 + ${NEXP:-1} * ($WALLTIME_RUNANDTIME + 5)))

unset NCCL_SHARP_GROUP_SIZE_THRESH
unset NCCL_CFG_PATH
export NCCL_IB_DISABLE=1
export NCCL_NET=Socket
export NCCL_P2P_LEVEL=NVL
export NCCL_SHARP_DISABLE=1
export FULL_CUDA_GRAPH=0
export MCORE_CUDA_GRAPH=0
