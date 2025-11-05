source $(dirname ${BASH_SOURCE[0]})/config_common.sh
source $(dirname ${BASH_SOURCE[0]})/config_XE9685Lx8B200-SXM-180GB_common_8b.sh
source $(dirname ${BASH_SOURCE[0]})/config_common_cg.sh

export MINIBS=4
export TENSOR_MODEL_PARALLEL=1
export SEQ_PARALLEL=False

export PIPELINE_MODEL_PARALLEL=1
export INTERLEAVED_PIPELINE=null
export CONTEXT_PARALLEL=1

export TP_COMM_OVERLAP=False
export MICRO_BATCH_SIZE=2

export USE_TE_OPS=True
export CE_FUSION_IMPL=te
export WARMUP_VALIDATION_STEPS=0

export WARMUP_STEPS=192
export VAL_CHECK_INTERVAL=384

export LR=0.0008

export DGXNNODES=1
export DGXNGPU=8
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )

export WALLTIME_RUNANDTIME=140
export WALLTIME=$((5 + ${NEXP:-1} * ($WALLTIME_RUNANDTIME + 5)))
