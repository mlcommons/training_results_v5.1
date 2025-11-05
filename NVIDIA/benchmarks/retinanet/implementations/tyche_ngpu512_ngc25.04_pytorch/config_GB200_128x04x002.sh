#!/bin/bash
source $(dirname ${BASH_SOURCE[0]})/config_common.sh

## DL params
export BATCHSIZE=${BATCHSIZE:-2}
export NUMEPOCHS=${NUMEPOCHS:-10}
export LR=${LR:-0.0001}
export WARMUP_EPOCHS=${WARMUP_EPOCHS:-1}
export EXTRA_PARAMS=${EXTRA_PARAMS:-'--jit --frozen-bn-opt --frozen-bn-fp16 --apex-adam --apex-focal-loss --apex-head-fusion --disable-ddp-broadcast-buffers --reg-head-pad --cls-head-pad --cuda-graphs --dali --dali-matched-idxs --dali-eval --cuda-graphs-syn --async-coco --master-weights --eval-batch-size=32'}

## System run params
export DGXNNODES=128
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
export WALLTIME_RUNANDTIME=7
export WALLTIME=$((9 + ${NEXP:-1} * ($WALLTIME_RUNANDTIME + 3)))

## System config params
export DGXNGPU=4
export DGXSOCKETCORES=64
export DGXNSOCKET=2
export DGXHT=2  # HT is on is 2, HT off is 1