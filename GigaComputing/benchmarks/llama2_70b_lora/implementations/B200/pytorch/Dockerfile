# Copyright (c) 2024-2025, NVIDIA CORPORATION.  All rights reserved.
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

ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:25.09-py3
FROM ${FROM_IMAGE_NAME}
WORKDIR /workspace/ft-llm

ARG GIT_COMMIT_ID
ENV GIT_COMMIT_ID=$GIT_COMMIT_ID

RUN git config --global user.name "a" && \
    git config --global user.email "a"

ENV PIP_CONSTRAINT=""

RUN NVARCH=$(uname -m |grep x86_64 2>/dev/null || echo sbsa) && \
    curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/${NVARCH}/3bf863cc.pub > /etc/apt/trusted.gpg.d/cuda.asc && \
    echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/${NVARCH} /" > /etc/apt/sources.list.d/cuda.list && \
    apt-get update && apt-get install -y libcudnn9-cuda-13=9.13.1.26-1 && rm -rf /var/lib/apt/lists/*
ENV CUDNN_VERSION=9.13.1.26
RUN pip install nvidia-mathdx==25.1.1

## NeMo
ARG NEMO_REVISION=25.09-alpha
RUN git clone https://github.com/NVIDIA/NeMo && \
    cd NeMo && \
    git checkout ${NEMO_REVISION} && \
    git remote add tim https://github.com/timmoon10/NeMo.git && \
    git fetch tim lora-disable-fused-norm-quantize && \
    git cherry-pick b6de95956756309064abaf8bd13b5bfb997f512b && \
    echo NEMO_COMMIT_HASH=$(git rev-parse HEAD) && \
    echo $(git rev-parse HEAD) > /NEMO_COMMIT_HASH.env && \
    pip install -e ".[llm]"

## Megatron-core
ARG MCORE_REVISION=25.09-alpha
RUN pip uninstall -y megatron-core && \
    git clone https://github.com/NVIDIA/Megatron-LM Megatron-LM && \
    cd Megatron-LM && \
    git checkout ${MCORE_REVISION} && \
    echo MCORE_COMMIT_HASH=$(git rev-parse HEAD) && \
    echo $(git rev-parse HEAD) > /MCORE_COMMIT_HASH.env && \
    pip install -e . && \
    cd megatron/core/datasets && \
    make
ENV PYTHONPATH "${PYTHONPATH}:/workspace/ft-llm/Megatron-LM"

## TransformerEngine
ARG TE_REVISION=3f5b47549567d13db76470073c8f0467c23d4fca
RUN if [ "${TE_REVISION}" != SKIP ]; then \
    pip uninstall -y transformer_engine && \
    git clone https://github.com/NVIDIA/TransformerEngine TransformerEngine && \
    cd TransformerEngine && \
    git checkout ${TE_REVISION} && \
    echo TE_COMMIT_HASH=$(git rev-parse HEAD) && \
    echo $(git rev-parse HEAD) > /TE_COMMIT_HASH.env && \
    git submodule init && git submodule update && \
    git -C 3rdparty/cudnn-frontend checkout 1a7b4b78db44712fb9707d21cd2e3179f1fd88b8 && \
    git -C 3rdparty/googletest checkout f8d7d77c06936315286eb55f8de22cd23c188571 && \
    git -C 3rdparty/cutlass checkout 57e3cfb47a2d9e0d46eb6335c3dc411498efa198 && \
    NVTE_CUDA_ARCHS="100a;103a" NVTE_BUILD_ACTIVATION_WITH_FAST_MATH=1 NVTE_UB_WITH_MPI=1 NVTE_FRAMEWORK=pytorch NVTE_WITH_USERBUFFERS=1 MPI_HOME=/usr/local/mpi pip install --no-build-isolation . \
    ; fi

# Install NCCL v2.28.3-1:
ARG NCCL_REVISION=v2.28.3-1
RUN export DEBIAN_FRONTEND=noninteractive &&        \
    apt-get update &&                               \
    apt-get remove -y libnccl2 &&                   \
    git clone https://github.com/NVIDIA/nccl.git && \
    cd nccl &&                                      \
    echo "NCCL_REVISION=${NCCL_REVISION}" &&      \
    git checkout "${NCCL_REVISION}" &&             \
    make -j24 install
ENV NCCL_VERSION "${NCCL_REVISION}"

## fix opencc
RUN apt-get update && apt-get install -y --no-install-recommends libopencc-dev

ADD . /workspace/ft-llm


RUN pip install -r requirements.txt
RUN pip uninstall -y bitsandbytes
