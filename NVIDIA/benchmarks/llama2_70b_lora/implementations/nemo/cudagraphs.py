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
from megatron.core import parallel_state
from megatron.core.distributed import DistributedDataParallel as McoreDDP
from megatron.core.num_microbatches_calculator import get_num_microbatches
from megatron.core.transformer.custom_layers.transformer_engine import TEDelayedScaling
from megatron.core.transformer.module import Float16Module as MCoreFloat16Module
from nemo.collections.nlp.modules.common.megatron.module import Float16Module
from transformer_engine.common import recipe
from transformer_engine.pytorch import make_graphed_callables
from nemo.lightning.megatron_parallel import MegatronParallel
from nemo.collections.llm.gpt.model import LlamaModel
from nemo.lightning.pytorch.callbacks.megatron_comm_overlap import MegatronCommOverlapCallback


def extract_module(model):
    if isinstance(model, list):
        return extract_module(model[0])
    elif isinstance(model, (McoreDDP, Float16Module, MCoreFloat16Module, MegatronParallel, LlamaModel)):
        return extract_module(model.module)
    return model

def get_microbatch_schedule(num_microbatches, num_model_chunks):
    schedule = []
    pipeline_model_parallel_size = parallel_state.get_pipeline_model_parallel_world_size()
    pipeline_parallel_rank = parallel_state.get_pipeline_model_parallel_rank()

    def get_model_chunk_id(microbatch_id, forward):
        """Helper method to get the model chunk ID given the iteration number."""
        microbatch_id_in_group = microbatch_id % (pipeline_model_parallel_size * num_model_chunks)
        model_chunk_id = microbatch_id_in_group // pipeline_model_parallel_size
        if not forward:
            model_chunk_id = num_model_chunks - model_chunk_id - 1
        return model_chunk_id

    if pipeline_model_parallel_size > 1:
        if parallel_state.get_virtual_pipeline_model_parallel_world_size() is not None:
            #forward_backward_pipelining_with_interleaving
            total_num_microbatches = num_microbatches * num_model_chunks
            if num_microbatches == pipeline_model_parallel_size:
                num_warmup_microbatches = total_num_microbatches
            else:
                num_warmup_microbatches = (pipeline_model_parallel_size - pipeline_parallel_rank - 1) * 2
                num_warmup_microbatches += (num_model_chunks - 1) * pipeline_model_parallel_size
                num_warmup_microbatches = min(num_warmup_microbatches, total_num_microbatches)
            num_microbatches_remaining = total_num_microbatches - num_warmup_microbatches
            for k in range(num_warmup_microbatches):
                cur_model_chunk_id = get_model_chunk_id(k, forward=True)
                schedule.append(cur_model_chunk_id+1)
            for k in range(num_microbatches_remaining):
                forward_k = k + num_warmup_microbatches
                cur_model_chunk_id = get_model_chunk_id(forward_k, forward=True)
                schedule.append(cur_model_chunk_id+1)
                backward_model_chunk_id = get_model_chunk_id(k, forward=False)
                schedule.append(-backward_model_chunk_id-1)
            for k in range(num_microbatches_remaining, total_num_microbatches):
                backward_model_chunk_id = get_model_chunk_id(k, forward=False)
                schedule.append(-backward_model_chunk_id-1)
        else:
            #forward_backward_pipelining_without_interleaving
            num_warmup_microbatches = (
                pipeline_model_parallel_size
                - pipeline_parallel_rank
                - 1
            )
            num_warmup_microbatches = min(num_warmup_microbatches, num_microbatches)
            num_microbatches_remaining = num_microbatches - num_warmup_microbatches
            schedule = [1]*num_warmup_microbatches + [1,-1]*num_microbatches_remaining + [-1]*num_warmup_microbatches
    else:
        #forward_backward_no_pipelining
        schedule = [1, -1]
    return schedule

def capture_cuda_graph(trainer, cfg):
    torch.cuda.synchronize()
    torch.distributed.barrier()

    schedule = get_microbatch_schedule(get_num_microbatches(), len(trainer.model) if isinstance(trainer.model, list) else None)
    assert parallel_state.get_pipeline_model_parallel_world_size() == 1, "CUDA graph requires PP=1"
    megatron_callback = next((cb for cb in trainer.callbacks if isinstance(cb, MegatronCommOverlapCallback)), None)
    if cfg.ub_tp_comm_overlap and megatron_callback and megatron_callback.need_tp_overlap_ub_init:
        megatron_callback._init_te_userbuffers(trainer.model.config)
        megatron_callback.need_tp_overlap_ub_init = False
    sequence_parallel = cfg.sequence_parallel
    tensor_model_parallel_size = cfg.tensor_model_parallel_size
    micro_batch_size = cfg.micro_batch_size
    slen = cfg.encoder_seq_length // tensor_model_parallel_size if sequence_parallel else cfg.encoder_seq_length
    slen = slen // cfg.context_parallel_size

    fp8_recipe = None
    if cfg.fp8:
        if cfg.fp8_e4m3:
            fp8_format = recipe.Format.E4M3
        elif cfg.fp8_hybrid:
            fp8_format = recipe.Format.HYBRID
        else:
            raise ValueError("E4M3 and HYBRID are the only supported FP8 formats.")
        fp8_recipe = TEDelayedScaling(
            config=extract_module(trainer.model).decoder.config,
            fp8_format=fp8_format,
            override_linear_precision=(False, False, False),
        )

    callables, sample_args, optional_args = [], [], []
    for l_no, layer in enumerate(extract_module(trainer.model).decoder.layers):
        callables.append(layer)
        graph_input = (
            torch.ones(
                (slen, micro_batch_size, 8192),
                dtype=torch.bfloat16,
                requires_grad=True,
                device='cuda'
            ),
        )
        sample_args.append(graph_input)
        optional_input = {
            'attention_mask': torch.ones(
                (1, 1, cfg.encoder_seq_length // cfg.context_parallel_size, cfg.encoder_seq_length),
                dtype=torch.bool,
                requires_grad=False,
                device='cuda'
            ),
            'rotary_pos_emb': torch.ones(
                (8192 // cfg.context_parallel_size, 1, 1, 128),
                dtype=torch.float32,
                requires_grad=False,
                device='cuda'
            )
        }
        optional_args.append(optional_input)

    graphs = make_graphed_callables(
        tuple(callables),
        tuple(sample_args),
        _order=schedule,
        allow_unused_input=True,
        fp8_enabled=cfg.fp8,
        fp8_recipe=fp8_recipe,
        fp8_weight_caching=cfg.enable_cg_fp8_weight_caching,
        sample_kwargs=optional_args,
    )
    for l_no, layer in enumerate(extract_module(trainer.model).decoder.layers):
        extract_module(trainer.model).decoder.cuda_graphs[l_no] = [graphs[l_no]]

    torch.distributed.barrier()
    if isinstance(trainer.model.module.module, McoreDDP):
        if trainer.model.module.module.use_forward_hook:
            trainer.model.module.module.enable_forward_pre_hook()
