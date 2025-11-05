# Copyright (c) 2024-2025, NVIDIA CORPORATION. All rights reserved.
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

import math
import torch
from transformer_engine.pytorch.tensor.float8_tensor import Float8Quantizer
import transformer_engine_torch as tex

class BlockWiseRoundRobinSharder():

    def __init__(self, block_size, num_bucket, num_embedding):
        self.block_size = block_size
        self.num_bucket = num_bucket
        # padding
        self.num_embedding = math.ceil(num_embedding / (block_size*num_bucket)) \
            * block_size*num_bucket
        self.block_per_bucket = self.num_embedding // block_size // num_bucket
    
    def get_num_embedding_w_padding(self):
        return self.num_embedding
    
    def map(self, x):
        block_id = x // self.block_size
        bucket_id = block_id % self.num_bucket
        block_offset = block_id // self.num_bucket
        y = (bucket_id*self.block_per_bucket + block_offset) * self.block_size + x % self.block_size
        return y
    

class FP8Helper:
    def __init__(self, device, fp8_format='e4m3', scale=1.0):
        self.device = device
        self.fp8_format = fp8_format
        dtype = tex.DType.kFloat8E4M3 if fp8_format == "e4m3" else tex.DType.kFloat8E5M2
        fp8_scale_inv = torch.ones(1, dtype=torch.float32, device=device) / scale
        amax_history = torch.zeros(1, 1, dtype=torch.float32, device=device)

        self.quantizer = Float8Quantizer(
            scale=fp8_scale_inv, 
            fp8_dtype=dtype, 
            amax=amax_history,
            
            # Need to tune these knobs
            rowwise=True, columnwise=False,
        )
    
    def fp8_to_fp16(self, x):
        x = x.view(dtype=torch.uint8)

        if x.shape[0] == 0:
            return torch.tensor([], dtype=torch.float16).view(x.shape).to(x.device)
        
        # Need to check the perf impact between these two solutions
        x_fp16 = self.quantizer.create_tensor_from_data(x.to(self.device), fake_dtype=torch.float16, internal=True).dequantize(dtype=torch.float16)
        # x_fp16 = self.quantizer.create_tensor_from_data(x.to(self.device), fake_dtype = torch.float16) # if we use this, then the output defaults to FP16. 
        return x_fp16
    
    def cast_np_to_fp8(self, np_in):
        torch_in = torch.from_numpy(np_in).to(self.device)
        fp8_filler = self.quantizer.make_empty(torch_in.shape, dtype=torch_in.dtype, device=self.device)
        self.quantizer.update_quantized(torch_in, fp8_filler)

        # get_data_tensors returns two values, one is fp8 matrix other is transposed matrix
        fp8_out = fp8_filler.get_data_tensors()[0].cpu().numpy()
        del fp8_filler
        del torch_in
        return fp8_out
    
def get_offsets_indices(blocks):
    # extracts the offsets and indices for the given blocks
    # to avoid repeatedly calling block[etype].adj_tensors("csc")
    offsets_indices = []
    for block in blocks:
        offset_index = {}
        for rel in block.canonical_etypes:
            offset, index, _ = block[rel].adj_tensors("csc")
            offset_index[rel] = (offset, index)

        offsets_indices.append(offset_index)
    return offsets_indices