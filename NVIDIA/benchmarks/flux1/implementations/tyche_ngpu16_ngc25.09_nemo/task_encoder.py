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

import io
import json

import numpy as np
import torch
from megatron.energon import Cooker, basic_sample_keys, CrudeSample, DefaultTaskEncoder
from megatron.energon.task_encoder.base import stateless


def deserialize_numpy_array(data: bytes) -> np.ndarray:
    """Deserialize numpy array from bytes.

    Args:
        data: Serialized bytes
    """
    buffer = io.BytesIO(data)

    # Load uint16 view and convert back to bf16
    uint16_data = np.load(buffer)
    tensor = torch.from_numpy(uint16_data).view(torch.bfloat16)
    return tensor


@stateless
def deserialize_preprocessed_example(example):
    """
    Utility function to deserialize arrays from preprocessed dataset during training.

    Usage:
        dataset = load_from_disk("/path/to/preprocessed/dataset")
        dataset = dataset.map(deserialize_preprocessed_example)
    """
    sample = {}
    sample["prompt_embeds"] = deserialize_numpy_array(example["t5.bytes"])
    sample["pooled_prompt_embeds"] = deserialize_numpy_array(example["clip.bytes"])
    sample["mean"] = deserialize_numpy_array(example["mean.bytes"])
    sample["logvar"] = deserialize_numpy_array(example["logvar.bytes"])
    if "json" in example:
        decoded = json.loads(example["json"])
        if "timestep" in decoded:
            sample["timestep"] = torch.tensor(decoded["timestep"])
    return CrudeSample(**basic_sample_keys(example), **sample)


class FluxTaskEncoder(DefaultTaskEncoder):
    decoder = None

    cookers = [Cooker(deserialize_preprocessed_example)]
