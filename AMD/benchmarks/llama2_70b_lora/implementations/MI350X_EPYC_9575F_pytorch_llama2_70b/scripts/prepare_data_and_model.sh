#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"

# Install Dependencies
pip install huggingface-hub==0.30.0
rm -rf /workspace/deps/NeMo && pip uninstall -y nemo-toolkit \
    && git clone https://github.com/NVIDIA/NeMo.git && cd NeMo && git checkout 25.04-alpha.rc1 && pip install -e ".[nlp]" && cd ..

rm -rf /workspace/deps/megatron_lm && pip uninstall -y megatron-core \
    && git clone https://github.com/NVIDIA/Megatron-LM.git Megatron-LM && cd Megatron-LM && git checkout 25.04-alpha.rc1 \
    && pip install . && cd megatron/core/datasets && make && cd ../../../.. \
    && export PYTHONPATH="${PYTHONPATH}:/workspace/ft-llm/Megatron-LM"

pip install git+https://github.com/NVIDIA-NeMo/Run.git

cd /workspace/code/

python ${SCRIPT_DIR}/download_dataset.py --data_dir /data/gov_report  # download dataset
python ${SCRIPT_DIR}/convert_dataset.py --data_dir /data/gov_report
python ${SCRIPT_DIR}/convert_model.py --output_path=/ckpt/

# Organize files for finetuning
mv /data/gov_report /data/data   
mv /ckpt/weights/ /ckpt/model_weights/ 
cd /ckpt/model_weights/
for d in module.*/; do mv -- "$d" "model.${d#module.}"; done  
cp /ckpt/context/nemo_tokenizer/tokenizer.model /ckpt/tokenizer.model
cp /workspace/code/scripts/model_config.yaml /ckpt/model_config.yaml

mv /ckpt /data/model/

echo "✅ Data and model preparation completed successfully!"
exit 0