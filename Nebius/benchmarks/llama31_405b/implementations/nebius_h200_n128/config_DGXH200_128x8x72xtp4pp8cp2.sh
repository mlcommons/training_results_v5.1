source $(dirname ${BASH_SOURCE[0]})/config_common.sh

# MINIBS is the number of gradient accumulation steps, it can be adjusted in
# steps of 8 (the same as the PIPELINE_MODEL_PARALLELISM amount)
export MINIBS=72

export TENSOR_MODEL_PARALLEL=4
export PIPELINE_MODEL_PARALLEL=8
export INTERLEAVED_PIPELINE=8
export CONTEXT_PARALLEL=2

export TP_COMM_OVERLAP=True

# DGXNNODES is the number of nodes (each with 8 gpus), it can be adjusted in
# increments of 16 nodes
# (TENSOR_MODEL_PARALLEL*PIPELINE_MODEL_PARALLEL*CONTEXT_PARALLEL/8)
# global batch size will be DGXNNODES*MINIBS/16
export DGXNNODES=128

export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )

export WALLTIME_RUNANDTIME=200
export WALLTIME=$((5 + ${NEXP:-1} * ($WALLTIME_RUNANDTIME + 5)))

export MODEL_SIZE="405b"

export MICRO_BATCH_SIZE=1

export MAX_STEPS=300

export ASYM_PP_EMBED=True
export ASYM_PP_LOSS=True
