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
# This configuration was created for GB200 pipecleaning.
# High chance of being suboptimal.
export EPOCHS="1"
export BATCHSIZE="4096" # local batch size
export VALIDATION_BATCHSIZE="2048"
export LEARNING_RATE="0.002"

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
export HIGH_PRIORITY_EMBED_STREAM="0"
export USE_CONCAT_EMBEDDING="0"
export PAD_NODE_COUNT_TO="3072"

export AMP="1"
export DIST_ADAM="1"

# CUDA Graph related
export USE_CUDA_GRAPH="1"
export CUDA_GRAPH_ESTIMATION_BATCHES="20"
export CUDA_GRAPH_PADDING_SIGMA="5"

# Configs that should not change
export FAN_OUT="5,10,15"
export HIDDEN_DIM="512"
export NUM_HEADS="4"
export EVAL_FREQUENCY="0.05"

# Binding
export BINDCMD="bindpcie --cpu=node"

# debugging
export TIMETAG="1"
export DEBUG="1"

## System run params
export DGXNNODES=2
export DGXNGPU=4
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )

# Disable warnings from torch
export TORCH_CPP_LOG_LEVEL=ERROR

export WALLTIME_RUNANDTIME=30  # measured: run_and_time.sh takes up to 20 minutes
export WALLTIME=$((10 + ${NEXP:-1} * ($WALLTIME_RUNANDTIME + 5)))
