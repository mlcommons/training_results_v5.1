#!/bin/bash

# Copyright (c) 2024-2025, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

###########################################################################
# This script is invoked inside the container, and a copy is launched on every
# rank.
#
# This script MUST NOT have any SLURM dependences (no use of SLURM envvars)
#
# If this is a pytorch benchmark this script assumes that it is invoked with
# something (torchrun, srun+enroot, srun+slurm2pytorch, or
# mpirun+slurm2pytorch) that correctly sets the variables described at
# https://pytorch.org/docs/stable/elastic/run.html#environment-variables RANK,
# LOCAL_RANK, WORLD_SIZE, LOCAL_WORLD_SIZE, MASTER_ADDR, MASTER_PORT
#
# To run this script interactively you must invoke it with torchrun
#
# if you need NODE_RANK you can derive it by
# NODE_RANK=$(( RANK / LOCAL_WORLD_SIZE ))
#
# If this script is for a framework that assumes mpirun (or srun --mpi), the
# envvars the envvars described at
# https://docs.open-mpi.org/en/v5.0.x/tuning-apps/environment-var.html
# OMPI_COMM_WORLD_SIZE, OMPI_COMM_WORLD_RANK, OMPI_COMM_WORLD_LOCAL_SIZE,
# OMPI_COMM_WORLD_LOCAL_RANK, OMPI_COMM_WORLD_NODE_RANK
###########################################################################

# vars that should be set by the launcher (Pytorch)
: "${RANK:?RANK not set}"
: "${LOCAL_RANK:?LOCAL_RANK not set}"
: "${WORLD_SIZE:?WORLD_SIZE not set}"
: "${LOCAL_WORLD_SIZE:?LOCAL_WORLD_SIZE not set}"
: "${MASTER_ADDR:?MASTER_ADDR not set}"
: "${MASTER_PORT:?MASTER_PORT not set}"
: "${NSYS_METRICS:=0}"

readonly NODE_RANK=$(( RANK / LOCAL_WORLD_SIZE ))

: "${LOGGER:=""}"
if [[ -n "${APILOG_DIR:-}" ]]; then
    if [[ "$RANK" -eq 0 ]]; then
      LOGGER="apiLog.sh -p MLPerf/${MODEL_NAME} -v ${FRAMEWORK}/train/${DGXSYSTEM}"
    fi
fi

NSYS_OUT="/results/${NSYS_PREFIX:="lora"}_${SLURM_JOBID}_n${NODE_RANK}_p${LOCAL_RANK}"
NSYSCMD=""
if [ "${NVTX_FLAG:-0}" -eq 1 ]
then
    NSYSCMD="nsys profile --sample=cpu --cuda-graph-trace=node --cpuctxsw=none --trace=cuda,nvtx -f true --stats true -o ${NSYS_OUT}"
    if [ ${NSYS_METRICS} -gt 0 ]; then
        NSYSCMD="${NSYSCMD} --gpu-metrics-devices=${LOCAL_RANK} --gpu-metrics-set=${NSYS_METRICS_SET} --gpu-metrics-frequency=50000"
    fi
fi

declare -a CMD
if [[ ${LOCAL_WORLD_SIZE} -gt 1 ]]; then
    # Mode 1: Slurm launched a task for each GPU and set some envvars
    CMD=( ${NSYSCMD} 'python' '-u')
else
    # interactive run on single node, no need to bind
    CMD=( ${NSYSCMD} 'torchrun' "--nproc_per_node=${DGXNGPU}" )
fi
${LOGGER:-} ${BINDCMD:-} ${CMD[@]} train.py; ret_code=$?

if [[ $ret_code != 0 ]]; then exit $ret_code; fi
