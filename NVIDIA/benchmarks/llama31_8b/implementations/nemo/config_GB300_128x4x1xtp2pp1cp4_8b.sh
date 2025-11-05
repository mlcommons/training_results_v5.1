source $(dirname ${BASH_SOURCE[0]})/config_common.sh
source $(dirname ${BASH_SOURCE[0]})/config_common_cg.sh
source $(dirname ${BASH_SOURCE[0]})/config_common_8b.sh

export MINIBS=1
export TENSOR_MODEL_PARALLEL=2
export SEQ_PARALLEL=True
export PIPELINE_MODEL_PARALLEL=1
export INTERLEAVED_PIPELINE=null
export CONTEXT_PARALLEL=4

export TP_COMM_OVERLAP=True
export MICRO_BATCH_SIZE=1
export NVTE_DPA_FP8_RECIPE="F16"
export BUCKET_SIZE=768000000
export USE_TE_OPS=False

export LR=0.0008
export WARMUP_STEPS=64
export VAL_CHECK_INTERVAL=192


export DGXNNODES=128
export DGXNGPU=4
export SEGMENT=16
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )

export WALLTIME_RUNANDTIME=20
export WALLTIME=$((5 + ${NEXP:-1} * ($WALLTIME_RUNANDTIME + 5)))
