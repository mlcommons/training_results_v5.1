source $(dirname ${BASH_SOURCE[0]})/config_common.sh
source $(dirname ${BASH_SOURCE[0]})/config_common_fp4.sh
source $(dirname ${BASH_SOURCE[0]})/config_common_405b.sh
source $(dirname ${BASH_SOURCE[0]})/config_common_cg.sh

export MINIBS=24
export TENSOR_MODEL_PARALLEL=4
export PIPELINE_MODEL_PARALLEL=8
export INTERLEAVED_PIPELINE=8
export CONTEXT_PARALLEL=2

export MICRO_BATCH_SIZE=1

export TP_COMM_OVERLAP=False
export ASYM_PP_EMBED=True
export ASYM_PP_LOSS=True

export MAX_STEPS=600

export DGXNNODES=640
export DGXNGPU=4
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )

export WALLTIME_RUNANDTIME=60
export WALLTIME=$((5 + ${NEXP:-1} * ($WALLTIME_RUNANDTIME + 5)))

# disable coredump due to hang at teardown
export ATTEMPT_CUDA_GDB_CORE_DUMP=0
export CUDA_ENABLE_COREDUMP_ON_EXCEPTION=0
export CUDA_ENABLE_LIGHTWEIGHT_COREDUMP=0
export CUDA_COREDUMP_SHOW_PROGRESS=0
export SEGMENT=$(( (TENSOR_MODEL_PARALLEL * PIPELINE_MODEL_PARALLEL * CONTEXT_PARALLEL) / DGXNGPU ))

export NCCL_NVLS_ENABLE=0
export UB_SKIPMC=1
export QKV_DGRAD_SM=16
export QKV_WGRAD_SM=16
export FC1_DGRAD_SM=16
export FC1_WGRAD_SM=16
