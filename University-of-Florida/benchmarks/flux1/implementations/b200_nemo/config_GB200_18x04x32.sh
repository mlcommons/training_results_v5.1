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
# Training parameters
export BATCHSIZE=2304
source $(dirname ${BASH_SOURCE[0]})/config_common_gbs_${BATCHSIZE}.sh
export MBS=32
export FP8_RECIPE="mxfp8"  # Use FP8 on Blackwell if mbs >= 4
# System parameters
export DGXNNODES=18
export DGXNGPU=4

export SEGMENT=18
# Walltime
export WALLTIME_RUNANDTIME=80
export WALLTIME=$((5 + ${NEXP:-1} * ($WALLTIME_RUNANDTIME + 5)))
