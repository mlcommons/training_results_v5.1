#!/bin/bash

# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

source $(dirname ${BASH_SOURCE[0]})/config_XD670_common.sh

# hyperparameters
export NCCL_TEST=0
export LR=0.0005
export MAX_STEPS=896
export MINIBS=1
export TP=4
export PP=1
export CP=1
export SP=1
export TP_COMM_OVERLAP=1 
export VBOOST_VALUE=1 
export GPU_ARCH="h"

export FP8=True
export FP8_AMAX_ALGO=max
export FP8_REDUCE_AMAX=True
export FP8_AMAX_HISTORY=32

export SKIP_EVALS=3
export HYDRA_FULL_ERROR=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

# system parameters
export DGXNNODES=4
export SHARP=True
export WALLTIME_RUNANDTIME=500
export WALLTIME=$((5 + ${NEXP:-10} * ($WALLTIME_RUNANDTIME + 5)))
export MLPERF_NUM_NODES=$DGXNNODES
