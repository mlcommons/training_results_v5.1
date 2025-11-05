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

# Load default settings
source $(dirname ${BASH_SOURCE[0]})/config_common.sh
# Load universal parameters regardless of batchsize
source $(dirname ${BASH_SOURCE[0]})/config_common_hpgs.sh

# Training parameters
export MBS=32  # constant microbatch size
export DGXNGPU=8  # constant number of GPUs per node
export DGXNNODES=${DGXNNODES:-8}  # get from environment variable or default to 8
export BATCHSIZE=$((DGXNNODES * DGXNGPU * MBS))  # batchsize = nnodes * ngpu * mbs
export FP8_RECIPE="mxfp8"  # Use FP8 on Blackwell if mbs >= 4

# System parameters
export SEGMENT=${DGXNNODES}

# Walltime calculation: 400/nnodes/0.5 (assume linear scaling with 50% penalty, 2 node time is 200 mins)
export WALLTIME_RUNANDTIME=$((20 + 400 / DGXNNODES / 1 * 2))  # divided by 0.5 is same as multiply by 2
export WALLTIME=$((5 + ${NEXP:-1} * ($WALLTIME_RUNANDTIME + 5)))
