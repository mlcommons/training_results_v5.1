FROM rocm/pytorch:rocm6.4_ubuntu22.04_py3.10_pytorch_release_2.6.0

WORKDIR /workspace

RUN pip install pybind11
RUN pip install ninja
RUN pip install packaging
RUN /usr/bin/python3 -m pip install pyYAML

# Install library dependencies
WORKDIR /workspace/deps

# FlashAttention
RUN git clone https://github.com/ROCm/flash-attention/ flash_attention \
    # latest stable commit of ck_tile/fa3 branch
    && cd flash_attention && git checkout cace3592812640486b04196a209bb85d12267b4c \
    && git submodule update --init --recursive \
    && PYTORCH_ROCM_ARCH='gfx942' GPU_ARCHS="gfx942" MAX_JOBS=64 pip install --no-build-isolation -e .

ADD patches /workspace/deps/patches

# Megatron-core
RUN git clone --recursive https://github.com/ROCm/Megatron-LM.git megatron_lm
RUN pip uninstall -y megatron-core
# dev branch commit
RUN cd megatron_lm && git checkout megatron_190213a_mlperf   \
    && pip install -e . && cd megatron/core/datasets && make

ENV PYTHONPATH "${PYTHONPATH}:/workspace/deps/megatron_lm"

# NeMo
RUN git clone https://github.com/NVIDIA/NeMo nemo \
    && cd nemo && git checkout v2.0.0.rc0.beta && sed -i '13d' requirements/requirements_nlp.txt
RUN cd /workspace/deps/nemo \
    && git apply /workspace/deps/patches/nemo_attention.patch \
    && git apply /workspace/deps/patches/nemo_utils.patch \
    && pip install --no-build-isolation -e ".[nlp]"

# Python deps
# Important this should be done after NeMo, otherwise the pinned transformers==4.40.2 version will be overwritten
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

# Transformer Engine
ARG TE_COMMIT=te_v1.9_mlperf_llama2
RUN git clone --recursive https://github.com/ROCm/TransformerEngine.git \
    # dev branch commit
    && cd TransformerEngine && git checkout $TE_COMMIT && git submodule update --init --recursive \
    # Workaround logging debug info to the console
    && sed -i 's/self.logger.info/self.logger.debug/g' /workspace/deps/TransformerEngine/transformer_engine/pytorch/attention.py \
    && sed -i 's/warnings.warn/if False: warnings.warn/g' /workspace/deps/TransformerEngine/transformer_engine/pytorch/attention.py \
    && sed -i '/.*\"window_size should be.*/d' /workspace/deps/TransformerEngine/transformer_engine/common/fused_attn_rocm/fused_attn.cpp \
    && NVTE_FUSED_ATTN_AOTRITON=0 NVTE_ROCM_ARCH='gfx942' NVTE_FRAMEWORK='pytorch' NVTE_USE_HIPBLASLT=1 MAX_JOBS=128 PYTORCH_ROCM_ARCH='gfx942' GPU_ARCHS='gfx942' pip install  -e .

# Install hipBLASLt (FP8 tuned gemms - second round)
 RUN git clone https://github.com/ROCm/hipBLASLt.git \
     && cd hipBLASLt && git checkout ebc770851dfb99a1bbb8ef2e5873c753f8011a47 \
     && sudo apt-get update \
     && apt install -y python3.10-venv \
     && ./install.sh -idc -a gfx942

WORKDIR /workspace/code

# Copy the current state of the code inside the image
COPY . .
