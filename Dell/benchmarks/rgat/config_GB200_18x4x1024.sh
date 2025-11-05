# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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


source $(dirname ${BASH_SOURCE[0]})/config_common.sh

## DL params

# tunable HPs
# This configuration generally converges using .75-.85 epochs
# if we hit 1 full epoch then we need to abort
export EPOCHS="2"
export BATCHSIZE="1024" # local batch size
export LEARNING_RATE="0.003"

# WG related
export WG_SHARDING_LOCATION="cuda"
export WG_SHARDING_PARTITION="global"
export WG_SHARDING_TYPE="continuous"
export SAMPLING_DEVICE="cuda"
export GRAPH_DEVICE="cuda"
export GRAPH_SHARDING_PARTITION="global"
export NUM_SAMPLING_THREADS="1"
export NUM_WORKERS="0"

# Knobs
export TRAIN_OVERLAP="1"
export EVAL_OVERLAP="1"
export HIGH_PRIORITY_EMBED_STREAM="1"
export PAD_NODE_COUNT_TO="3072"
export GC_THRESHOLD_MULTIPLIER="2"

# model configs not fixed on reference branch for now
# need to remove them after the reference branch is fixed.
export FAN_OUT="5,10,15"
export HIDDEN_DIM="512"
export NUM_HEADS="4"
export AMP="1"

# CUDA Graph related
export USE_CUDA_GRAPH="1"
export CUDA_GRAPH_ESTIMATION_BATCHES="20"
export CUDA_GRAPH_PADDING_SIGMA="5"

# debugging
export TIMETAG="1"
export DEBUG="1"

# training related
export EVAL_FREQUENCY="0.05"
export VALIDATION_BATCHSIZE="2048"

# Binding
export BINDCMD="bindpcie --cpu=node"

## System run params
export DGXNNODES=18
export DGXNGPU=4
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )

# Disable warnings from torch
export TORCH_CPP_LOG_LEVEL=ERROR

export WALLTIME_RUNANDTIME=15  # measured: run_and_time.sh takes between 3-4 minutes.
export WALLTIME=$((10 + ${NEXP:-1} * ($WALLTIME_RUNANDTIME + 5)))
