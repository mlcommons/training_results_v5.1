
source $(dirname ${BASH_SOURCE[0]})/config_common.sh
export SUBMISSION_ORG=Dell
export MLPERF_SUBMITTER=Dell
export MLPERF_SYSTEM_NAME="1xXE9680Lx8H200-SXM-141GB"
export MLPERF_STATUS="Available on-premise"
export MLPERF_DIVISION=closed
export MLPERF_NUM_NODES=1
export MLPERF_CLUSTER_NAME=Dell

## DL params
export RUN_SCRIPT="train.py"
export BATCHSIZE=65536
export BATCHSIZE_EVAL=524288
export LEARNING_RATE=0.004
export USE_MIXED_PRECISION=true
export SCALER=8192
export SHARDING_PLAN=auto
export MEM_COMM_BW_RATIO=8
export GEN_LOSS_SUMMARY=true
export MINIMUM_TRAINING_TIME=10
export DP_SHARDING_THRESHOLD=0.01

## System run parms
export DGXNNODES=1
export DGXNGPU=8
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
export WALLTIME_RUNANDTIME=10

## Set clocks and walltime for maxQ and minEDP runs
if [[ "${SET_MAXQ_CLK:-0}" == "1" ]]; then
  export MAXQ_CLK=1320
  WALLTIME_RUNANDTIME=$(expr ${WALLTIME_RUNANDTIME} + ${WALLTIME_RUNANDTIME} / 2) # 50% longer walltime
elif [[ "${SET_MINEDP_CLK:-0}" == "1" ]]; then
  export MINEDP_CLK=1665
  WALLTIME_RUNANDTIME=$(expr ${WALLTIME_RUNANDTIME} + ${WALLTIME_RUNANDTIME} / 3) # 33% longer walltime
fi
export WALLTIME=$((5 + ${NEXP:-1} * ($WALLTIME_RUNANDTIME + 5)))
