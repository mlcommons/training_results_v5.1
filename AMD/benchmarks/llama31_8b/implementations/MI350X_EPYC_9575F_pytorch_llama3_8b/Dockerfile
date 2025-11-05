ARG BASE_IMAGE=rocm/7.0:rocm7.0_ubuntu22.04_py3.10_pytorch_release_2.8.0_rc1
FROM ${BASE_IMAGE}

WORKDIR /workspace

RUN pip install pybind11
RUN pip install ninja
RUN pip install packaging
RUN /usr/bin/python3 -m pip install pyYAML

# Install library dependencies
WORKDIR /workspace/deps

ADD patches /workspace/deps/patches

# Megatron-core
ARG MEGATRON_COMMIT=5c34c4cfe41dc304ff51c686358cf039621c7cb1
RUN git clone --recursive https://github.com/ROCm/Megatron-LM.git megatron_lm
RUN pip uninstall -y megatron-core
# dev branch commit
RUN cd megatron_lm && git checkout ${MEGATRON_COMMIT}   \
    && pip install -e . && cd megatron/core/datasets && make

ENV PYTHONPATH "${PYTHONPATH}:/workspace/deps/megatron_lm"

# NeMo
RUN git clone https://github.com/NVIDIA/NeMo nemo \
    && cd nemo && git checkout v2.5.0rc0 \
    && git apply /workspace/deps/patches/nemo_v2.5.0rc0.patch \
    && sed -i '12d' requirements/requirements_nlp.txt
RUN cd /workspace/deps/nemo \
    && pip install --no-build-isolation -e ".[nlp]"

# NeMo-Run
RUN pip install git+https://github.com/NVIDIA/NeMo-Run.git@v0.4.0

# Python deps
# Important this should be done after NeMo, otherwise the pinned transformers==4.40.2 version will be overwritten
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

# Transformer Engine
ARG TE_COMMIT=f3692b67613849d27f0e04d6087dedf9402e208c
RUN git clone --recursive https://github.com/ROCm/TransformerEngine.git \
    && cd TransformerEngine && git checkout ${TE_COMMIT} \
    && git submodule sync \
    && git submodule update --init --recursive \
    && NVTE_FUSED_ATTN_AOTRITON=0 NVTE_ROCM_ARCH="gfx942;gfx950" GPU_ARCHS="gfx942;gfx950" NVTE_FRAMEWORK='pytorch' NVTE_USE_HIPBLASLT=1 MAX_JOBS=128 PYTORCH_ROCM_ARCH="gfx942;gfx950" CU_NUM=304 pip install . -v

WORKDIR /workspace/code

# Copy the current state of the code inside the image
COPY . .

# Uninstall bitsandbytes to remove NVIDIA related warnings
RUN pip uninstall -y bitsandbytes
