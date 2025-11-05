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

import torch
import triton
import triton.language as tl

@triton.jit
def multi_tensor_copy_pad_kernel(
    x_ptrs,
    y_ptrs,
    sizes_x,
    sizes_y,
    BLOCK_SIZE: tl.constexpr,
    DATA_TYPE: tl.constexpr,
):
    tensor_id = tl.program_id(0)
    block_id = tl.program_id(1)

    len_x = tl.load(sizes_x + tensor_id)
    len_y = tl.load(sizes_y + tensor_id)

    x_ptr = tl.cast(tl.load(x_ptrs + tensor_id), tl.pointer_type(DATA_TYPE))
    y_ptr = tl.cast(tl.load(y_ptrs + tensor_id), tl.pointer_type(DATA_TYPE))

    block_start = block_id * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    mask_y = offsets < len_y
    mask_cp = offsets < len_x

    # handling empty input tensor
    if len_x == 0:
        tl.store(y_ptr + offsets, 0, mask=mask_y)
        return
    x_last = tl.load(x_ptr + len_x - 1)
    x_data = tl.load(x_ptr + offsets, mask=mask_cp)
    result = tl.where(mask_cp, x_data, x_last)
    tl.store(y_ptr + offsets, result, mask=mask_y)


def multi_tensor_copy_pad(
    list_x,
    list_y,
    BLOCK_SIZE: int = 256,
):
    num_tensors = len(list_x)
    device = list_x[0].device
    x_ptrs = torch.tensor([x.data_ptr() for x in list_x], device=device, dtype=torch.int64)
    y_ptrs = torch.tensor([y.data_ptr() for y in list_y], device=device, dtype=torch.int64)

    sizes_x_tensor = torch.tensor([x.numel() for x in list_x], device=device, dtype=torch.int64)
    sizes_y = [y.numel() for y in list_y]
    sizes_y_tensor = torch.tensor(sizes_y, device=device, dtype=torch.int64)
    num_blocks = (max(sizes_y) + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = (num_tensors, num_blocks)

    DATA_TYPE = tl.int64
    multi_tensor_copy_pad_kernel[grid](
        x_ptrs,
        y_ptrs,
        sizes_x_tensor,
        sizes_y_tensor,
        BLOCK_SIZE,
        DATA_TYPE)