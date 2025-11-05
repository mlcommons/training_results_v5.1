ARG BASE_IMAGE=rocm/7.0:rocm7.0_ubuntu22.04_py3.10_pytorch_release_2.8.0_rc1
FROM ${BASE_IMAGE}

WORKDIR /workspace

RUN pip install pybind11
RUN pip install ninja
RUN pip install packaging
RUN /usr/bin/python3 -m pip install pyYAML

# Install library dependencies
WORKDIR /workspace/deps

# Megatron-core
ARG MEGATRON_COMMIT=bc0be960c2f1fb2c4a57ecd7dec6235c79f85412
RUN git clone --recursive https://github.com/ROCm/Megatron-LM.git megatron_lm
RUN pip uninstall -y megatron-core
RUN cd megatron_lm && git checkout ${MEGATRON_COMMIT}   \
    && pip install -e . && cd megatron/core/datasets && make

ENV PYTHONPATH "${PYTHONPATH}:/workspace/deps/megatron_lm"

# NeMo
RUN git clone https://github.com/NVIDIA/NeMo nemo \
    && cd nemo && git checkout v2.3.0 && sed -i '12d' requirements/requirements_nlp.txt
RUN cd /workspace/deps/nemo \
    && pip install --no-build-isolation -e ".[nlp]"

# Python deps
# Important this should be done after NeMo, otherwise the pinned transformers==4.40.2 version will be overwritten
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

RUN python -m ensurepip || apt-get install -y python3-venv \
&& pip install cmake==3.28.3

# Transformer Engine
ARG TE_COMMIT=d0ef413b5bfe9ea33b1c8626e99fb46967a60712
RUN git clone --recursive https://github.com/ROCm/TransformerEngine.git \
    && cd TransformerEngine && git checkout ${TE_COMMIT} \
    && git submodule sync \
    && git submodule update --init --recursive \
    && NVTE_FUSED_ATTN_AOTRITON=0 NVTE_ROCM_ARCH="gfx942;gfx950" GPU_ARCHS="gfx942;gfx950" NVTE_FRAMEWORK='pytorch' NVTE_USE_HIPBLASLT=1 MAX_JOBS=128 PYTORCH_ROCM_ARCH="gfx942;gfx950" CU_NUM=304 pip install . -v --no-build-isolation

WORKDIR /workspace/code

# Copy the current state of the code inside the image
COPY . .
