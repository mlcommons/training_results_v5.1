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

from nemo.lightning import teardown
from nemo.collections import llm
from nemo.collections.llm.gpt.model.llama import HFLlamaImporter
from huggingface_hub import snapshot_download
from pathlib import Path
import argparse
import shutil

def apply(self, output_path: Path) -> Path:
    """ modify to save ckpt in zarr format """
    from transformers import LlamaForCausalLM
    source = LlamaForCausalLM.from_pretrained(str(self), torch_dtype='auto')
    target = self.init()
    trainer = self.nemo_setup(target, save_ckpt_format='zarr', async_save=False, parallel_save=False)
    trainer.strategy.async_save = False
    trainer.strategy.parallel_save = False
    self.convert_state(source, target)
    self.nemo_save(output_path, trainer)
    print(f"Converted Llama model to Nemo, model saved to {output_path} in {source.dtype}.")
    teardown(trainer, target)
    del trainer, target
    return output_path

HFLlamaImporter.apply = apply

# This script requires one of the following setups:
# 1. Set the HF_HOME environment variable to the directory where the model is downloaded:
#    export HF_HOME=/path/to/hf_home
#    Ensure the directory structure is as follows:
#    $HF_HOME/
#      hub/
#        models--meta-llama--Llama-2-70B-hf/
#          blobs/
#          refs/
#          snapshots/
#
# 2. Set the HF_TOKEN environment variable with your Hugging Face access token:
#    export HF_TOKEN=<your_token>
#    The script will then download the model to $HF_HOME/hub
#
# Note: You must have access to the meta-llama/Llama-2-70B-hf model on
# Hugging Face (https://huggingface.co/meta-llama/Llama-2-70b-hf) to download it.

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, default="hf://meta-llama/Llama-2-70B-hf", help='Path to the checkpoint')
    parser.add_argument('--output_path', type=str, default="/ckpt", help='Path to save the imported checkpoint')

    args = parser.parse_args()

    snapshot_download(
        "meta-llama/Llama-2-70B-hf",
        max_workers=32,
        allow_patterns=["*.safetensors"],
    )

    model = llm.LlamaModel(llm.Llama2Config70B())
    # This could take around 10-20 minutes to complete
    llm.import_ckpt(model, args.checkpoint_path, Path(args.output_path) / "model", True)
    shutil.move(Path(args.output_path) / "model" / "context", Path(args.output_path))
    shutil.move(Path(args.output_path) / "model" / "weights", Path(args.output_path))
    shutil.rmtree(Path(args.output_path) / "model")
