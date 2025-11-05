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

readonly node_rank="${SLURM_NODEID:-0}"
readonly local_rank="${LOCAL_RANK:=${SLURM_LOCALID:=${OMPI_COMM_WORLD_LOCAL_RANK:-0}}}"

: "${LOGGER:=""}"
if [[ -n "${APILOG_DIR:-}" ]]; then
    if [ "$node_rank" -eq 0 ] && [ "$local_rank" -eq 0 ]; then
      LOGGER="apiLog.sh -p MLPerf/${MODEL_NAME} -v ${FRAMEWORK}/train/${DGXSYSTEM}"
    fi
fi

NSYS_OUT="/results/${NSYS_PREFIX:="lora"}_${SLURM_JOBID}_n${node_rank}_p${local_rank}"
NSYSCMD=""
if [ "${NVTX_FLAG:-0}" -eq 1 ]
then
    NSYSCMD="nsys profile --sample=cpu --cuda-graph-trace=node --cpuctxsw=none --trace=cuda,nvtx -f true --stats true -o ${NSYS_OUT}"
fi

declare -a CMD
#if [[ -n "${SLURM_LOCALID-}" ]] && [[ "${SLURM_NTASKS}" -gt "${SLURM_JOB_NUM_NODES}" ]]; then
if [[ ${LOCAL_WORLD_SIZE} -gt 1 ]]; then
    # Mode 1: Slurm launched a task for each GPU and set some envvars
    CMD=( ${NSYSCMD} 'python' '-u')
else
    # interactive run on single node, no need to bind
    CMD=( ${NSYSCMD} 'torchrun' "--nproc_per_node=${DGXNGPU}" )
fi
${LOGGER:-} ${BINDCMD:-} ${CMD[@]} train.py; ret_code=$?

if [[ $ret_code != 0 ]]; then exit $ret_code; fi
