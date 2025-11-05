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

export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[1]}) | sed 's/^config_//' | sed 's/\.sh$//' )
export HYDRA_FULL_ERROR=1
export TQDM_DISABLE=True
export DISABLE_PRINT=${DISABLE_PRINT:-True} # filter out nemo prints. Set to False to debug.

export MODEL=${MODEL:-schnell}
export DATA=${DATA:-cc12m}
export TARGET_ACCURACY=${TARGET_ACCURACY:-0.586}
export EXP_NAME=${EXP_NAME:-flux-train-$(date +%y%m%d%H%M%S%N)}
# Validation interval in samples
export VAL_CHECK_INTERVAL=${VAL_CHECK_INTERVAL:-262144}

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export GRAD_REDUCE_IN_FP32=${GRAD_REDUCE_IN_FP32:-False}

export WARMUP_ENABLED=${WARMUP_ENABLED:-True}
export WARMUP_TRAIN_STEPS=${WARMUP_TRAIN_STEPS:-2}
export WARMUP_VALIDATION_STEPS=${WARMUP_VALIDATION_STEPS:-2}

export HF_HUB_OFFLINE=1 # disable network requests for HF transfers

export NCCL_NVLS_ENABLE=1
export NCCL_GRAPH_REGISTER=0
export NCCL_LOCAL_REGISTER=0

export MODEL_TFLOP_PER_SAMPLE=20.3

#---------------------- OCI Extras -------------------
set -eux
export PMI_DEBUG=1
export OMPI_MCA_pml=ucx
export OMPI_MCA_btl=^openib
export OMPI_MCA_btl_tcp_if_include="10.224.0.0/12"
export PMIX_MCA_gds="^ds12" \
      NCCL_SOCKET_NTHREADS=16 \
      NCCL_DEBUG=WARN \
      NCCL_CUMEM_ENABLE=0 \
      NCCL_IB_SPLIT_DATA_ON_QPS=0 \
      NCCL_IB_QPS_PER_CONNECTION=1 \
      NCCL_IB_GID_INDEX=3 \
      NCCL_IB_TC=41 \
      NCCL_IB_SL=0 \
      NCCL_IB_TIMEOUT=22 \
      NCCL_NET_PLUGIN=none \
      NCCL_SOCKET_IFNAME=eth0 \
      NCCL_IGNORE_CPU_AFFINITY=1 \
      RX_QUEUE_LEN=8192 \
      IB_RX_QUEUE_LEN=8192 \
      UCX_NET_DEVICES=eth0 \
      UCX_TLS=tcp \
      HCOLL_ENABLE_MCAST_ALL=0 \
      coll_hcoll_enable=0 \
      NCCL_IB_HCA='=mlx5_0,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_9,mlx5_10,mlx5_11'
