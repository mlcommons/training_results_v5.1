#!/bin/bash

set -euo pipefail

#################### MODIFY THESE ####################
export RUN_DATETIME=$(date +%Y%m%d-%H%M%S)
export DGXSYSTEM=MI325X
export DATADIR=/data/mlperf_llama2/data
export MODELDIR=/data/mlperf_llama2/model
export IMAGE_NAME=mangollm/mb-llmboost-training:mlperf-5.1-prod-20251009-v1
export NCCL_SOCKET_IFNAME=$(rdma link | grep 'LINK_UP' | cut -d' ' -f 8 | head -n 16 | tr '\n' ',' | sed 's/,$//')
export GLOO_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME
export MASTER_ADDR=mi325x-01
export MASTER_PORT=29500
export WORLD_SIZE=1
export RANK=$([[ "$MASTER_ADDR" == "$(hostname)" ]] && echo 0 || echo 1)
export NCCL_DEBUG="" #"INFO"
export CONTAINER=mb_llmboost_llama2_${RUN_DATETIME}_${RANK}

################# CHECK REQUIRED ENVS #################
required_vars=(
    RUN_DATETIME DGXSYSTEM DATADIR MODELDIR IMAGE_NAME
    NCCL_SOCKET_IFNAME GLOO_SOCKET_IFNAME MASTER_ADDR MASTER_PORT
    WORLD_SIZE RANK NCCL_DEBUG CONTAINER
)

missing=()
for v in "${required_vars[@]}"; do
    # use parameter expansion with default to avoid unbound variable error under set -u
    if [ -z "${!v:-}" ]; then
        missing+=("$v")
    fi
done

if [ "${#missing[@]}" -ne 0 ]; then
    echo "ERROR: the following required environment variables are not set or are empty: ${missing[*]}" >&2
    exit 1
fi
#######################################################

docker pull $IMAGE_NAME
mkdir -p mlperf-logs/$WORLD_SIZE/

# Stop all other containers
echo "Stopping all existing containers ..."
docker ps -q | xargs -r docker stop

# Start the container
docker run --rm --init --detach \
        --network host --ipc host --uts host \
        --cap-add SYS_PTRACE \
        --security-opt seccomp=unconfined \
        --group-add video \
        --device /dev/dri:/dev/dri \
        --device /dev/kfd:/dev/kfd \
        -v $DATADIR:/data \
        -v $MODELDIR:/ckpt \
        -w /workspace \
        -e RUN_DATETIME="$RUN_DATETIME" \
        -e DGXSYSTEM="$DGXSYSTEM" \
        -e DATADIR="$DATADIR" \
        -e MODELDIR="$MODELDIR" \
        -e IMAGE_NAME="$IMAGE_NAME" \
        -e NCCL_SOCKET_IFNAME="$NCCL_SOCKET_IFNAME" \
        -e GLOO_SOCKET_IFNAME="$GLOO_SOCKET_IFNAME" \
        -e MASTER_ADDR="$MASTER_ADDR" \
        -e MASTER_PORT="$MASTER_PORT" \
        -e WORLD_SIZE="$WORLD_SIZE" \
        -e RANK="$RANK" \
        -e NCCL_DEBUG="$NCCL_DEBUG" \
        -e CONTAINER="$CONTAINER" \
        --name "${CONTAINER}" \
        $IMAGE_NAME sleep infinity

# Wait for container to complete starting
sleep 5
# Verify env
echo RUN_DATETIME="$RUN_DATETIME"
echo DGXSYSTEM="$DGXSYSTEM"
echo DATADIR="$DATADIR"
echo MODELDIR="$MODELDIR"
echo IMAGE_NAME="$IMAGE_NAME"
echo NCCL_SOCKET_IFNAME="$NCCL_SOCKET_IFNAME"
echo GLOO_SOCKET_IFNAME="$GLOO_SOCKET_IFNAME"
echo MASTER_ADDR="$MASTER_ADDR"
echo MASTER_PORT="$MASTER_PORT"
echo WORLD_SIZE="$WORLD_SIZE"
echo RANK="$RANK"
echo NCCL_DEBUG="$NCCL_DEBUG"
echo CONTAINER="$CONTAINER"

docker exec "${CONTAINER}" true
docker exec -it ${CONTAINER} cp /workspace/conf/config_MI325X_1x8x1.sh /workspace/conf/config_MI325X_2x8x1.sh

export HOST_LOG_DIR="mlperf-logs/$WORLD_SIZE"
export HOST_LOG_FILE="${HOST_LOG_DIR}/${RUN_DATETIME}-rank_${RANK}_of_${WORLD_SIZE}.log"
mkdir -p $HOST_LOG_DIR

echo "=== Node $(hostname) starting ==="
echo "MASTER_ADDR:MASTER_PORT $MASTER_ADDR:$MASTER_PORT"
echo "NCCL SOCKETS: $NCCL_SOCKET_IFNAME"
echo "GLOO SOCKETS: $GLOO_SOCKET_IFNAME"

# Start run
docker exec \
    -e NCCL_SOCKET_IFNAME="$NCCL_SOCKET_IFNAME" \
    -e GLOO_SOCKET_IFNAME="$GLOO_SOCKET_IFNAME" \
    -e MASTER_ADDR="$MASTER_ADDR" \
    -e MASTER_PORT="$MASTER_PORT" \
    -e MODELDIR="$MODELDIR" \
    -e DATADIR="$DATADIR" \
    -e DGXSYSTEM="$DGXSYSTEM" \
    -e IMAGE_NAME="$IMAGE_NAME" \
    -e NCCL_DEBUG="$NCCL_DEBUG" \
    "${CONTAINER}" \
    llmboost mlperf \
      --MASTER_ADDR "${MASTER_ADDR}" \
      --RANK        "${RANK}" \
      --config_sh   conf/config_${DGXSYSTEM}_${WORLD_SIZE}x8x1.sh \
      2>&1 | tee $HOST_LOG_FILE
