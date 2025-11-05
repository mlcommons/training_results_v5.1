# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

_cont_mounts="${DATADIR}:/data:ro,${MODEL}:/ckpt:ro,${LOGDIR}:/results:rw"
mounts_to_verify="DATADIR:/data MODEL:/ckpt"

if [[ "${DEBUG_IN_CWD:-}" ]]; then
    echo "mounting $(pwd) as workdir"
    WORK_DIR="$(pwd)"
    _cont_mounts+=",$(pwd):$(pwd)"
fi

if [ ${NVTX_FLAG:-0} -gt 0 ]; then
    if [[ "$LOGBASE" == *'_'* ]];then
        LOGBASE="${LOGBASE}_nsys"
    else
        LOGBASE="${SPREFIX}_nsys"
    fi
fi


# Cleanup data index dir
rm -rf "${LOGDIR}/data_index/train" "${LOGDIR}/data_index/val"
mkdir -p "${LOGDIR}/data_index/train" "${LOGDIR}/data_index/val"

if [[ -n "${APILOG_DIR:-}" ]]; then
    APILOG_DIR=${APILOG_DIR}/${MODEL_FRAMEWORK}/${MODEL_NAME}/${DGXSYSTEM}
    mkdir -p ${APILOG_DIR}
    _cont_mounts="${_cont_mounts},${APILOG_DIR}:/logs"

    # Create JSON file for cuDNN
    JSON_MODEL_NAME="MLPERF_${MODEL_NAME}_${MODEL_FRAMEWORK}_train"
    JSON_README_LINK="${README_PREFIX}/${MODEL_NAME}/${MODEL_FRAMEWORK}/README.md"
    JSON_FMT='{model_name: $mn, readme_link: $rl, configs: {($dt): [$bs]}, sweep: {($dt): [$bs]}}'
    JSON_OUTPUT="${JSON_MODEL_NAME}.cudnn.json"
    jq -n --indent 4 --arg mn "${JSON_MODEL_NAME}" --arg rl "${JSON_README_LINK}" --arg dt "${APILOG_PRECISION}" --arg bs "${BATCHSIZE}" "$JSON_FMT" > "${APILOG_DIR}/${JSON_OUTPUT}"
fi

if [[ "${JET:-0}" -eq 1 ]]; then
    _cont_mounts="${_cont_mounts},${JET_DIR}:/root/.jet"
fi
