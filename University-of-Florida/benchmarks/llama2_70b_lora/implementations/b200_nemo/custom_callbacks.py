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

import os
import time
from typing import Any, List
from pprint import pprint
from math import ceil

import lightning as pl
import numpy as np
import torch
from cudagraphs import capture_cuda_graph, extract_module
from custom_llama import reset_fp8_state, run_training_warmup
import transformer_engine.pytorch as te
import transformer_engine_torch as tex
import transformer_engine.common.recipe
from megatron.core import parallel_state
from megatron.core.extensions.transformer_engine import TEDotProductAttention
from megatron.core.num_microbatches_calculator import (
    reconfigure_num_microbatches_calculator,
)
from megatron.core.full_cuda_graph import FullCudaGraphWrapper
from mlperf_common.callbacks import LoggingCallback, MLPerfLogger
from omegaconf import DictConfig, OmegaConf


def get_context_parallel_world_size_override():
    return 1

original_get_context_parallel_world_size = parallel_state.get_context_parallel_world_size


class CustomCallback(LoggingCallback):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.gbs = cfg.model.global_batch_size
        self.iter_time = time.time()
        self.enable_perf_debug = int(os.environ.get("ENABLE_PERF_DEBUG", "0"))
        self.enable_mem_profile = int(os.environ.get("ENABLE_MEMORY_PROFILE", "0"))
        if self.enable_mem_profile:
            torch.cuda.memory._record_memory_history(max_entries=100000)
        self.cp_group = None
        self.cp_global_ranks = None
        self.cp_stream = None
        self.logging_interval = int(os.environ.get("LOGGING_INTERVAL", "10"))

        # Healing states
        self.healing_precision = cfg.model.healing_precision
        self.healing_lambda = None

        # Pre-quantization
        self.pre_quantized_model = cfg.model.pre_quantized_model
        self.nvfp4_quantizer = None
        self.fp4_cpu_params = []
        self.fp8_quantizer = None
        self.fp8_cpu_params = []
        self.store_quantized_params_on_gpu = cfg.model.store_gpu

        self.first_last_layers_bf16 = cfg.model.first_last_layers_bf16
        self.num_layers_at_start_in_bf16 = cfg.model.num_layers_at_start_in_bf16
        self.num_layers_at_end_in_bf16 = cfg.model.num_layers_at_end_in_bf16

        if cfg.model.custom.warmup_healing:
            _original_full_cudagraph_wrapper_init = FullCudaGraphWrapper.__init__
            cuda_graph_warmup_steps = cfg.model.custom.warmup_train_steps + 1

            def _patched_full_cudagraph_wrapper_init(self, forward_backward_func, cuda_graph_warmup_steps=cuda_graph_warmup_steps):
                return _original_full_cudagraph_wrapper_init(
                    self,
                    forward_backward_func,
                    cuda_graph_warmup_steps=cuda_graph_warmup_steps,
                )

            FullCudaGraphWrapper.__init__ = _patched_full_cudagraph_wrapper_init

    def __deepcopy__(self, memo):
        return CustomCallback(self.cfg)

    def _healing_setup(self):
        """Setup healing recipe"""
        if self.healing_precision == "FP8_DS":
            self.healing_lambda = lambda config: transformer_engine.common.recipe.DelayedScaling(
                amax_history_len=self.cfg.model.fp8_amax_history_len,
                amax_compute_algo=self.cfg.model.fp8_amax_compute_algo,
                reduce_amax=self.cfg.model.reduce_amax,
                fp8_dpa=self.cfg.model.fp8_dot_product_attention,
            )
        elif self.healing_precision == "MXFP8":
            self.healing_lambda = lambda config: transformer_engine.common.recipe.MXFP8BlockScaling()
        else:
            raise ValueError(f"Unsupported healing precision: {self.healing_precision}")

    def _pre_quantize_model(self, trainer, pl_module):
        """Pre-quantize model to FP8 and NVFP4"""
        # Set up NVFP4 quantizer
        from transformer_engine.pytorch.tensor.nvfp4_tensor import NVFP4Quantizer
        self.nvfp4_quantizer = NVFP4Quantizer()

        # Set up FP8 quantizer for healing, which can be in either FP8 DS or MXFP8
        if self.healing_precision == "FP8_DS":
            from transformer_engine.pytorch.tensor.float8_tensor import Float8Quantizer
            self.fp8_quantizer = Float8Quantizer(
                    scale=torch.ones(1, dtype=torch.float32, device=torch.cuda.current_device()),
                    amax=torch.zeros(1, dtype=torch.float32, device=torch.cuda.current_device()),
                    fp8_dtype=tex.DType.kFloat8E4M3)
        elif self.healing_precision == "MXFP8":
            from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
            self.fp8_quantizer = MXFP8Quantizer(tex.DType.kFloat8E4M3)
        else:
            raise ValueError(f"Unsupported healing precision: {self.healing_precision}")

        if not self.cfg.model.custom.warmup_healing:
            # Without warmup for healing
            self.fp8_cpu_params = self._get_quantized_params_cpu(trainer.model, self.fp8_quantizer, self.healing_precision)
            _ = self._get_quantized_params_cpu(trainer.model, self.nvfp4_quantizer, "NVFP4", replace=True)
        else:
            # With warmup for healing
            self.fp4_cpu_params = self._get_quantized_params_cpu(trainer.model, self.nvfp4_quantizer, "NVFP4")
            self.fp8_cpu_params = self._get_quantized_params_cpu(trainer.model, self.fp8_quantizer, self.healing_precision)
            _ = self._get_quantized_params_cpu(trainer.model, self.fp8_quantizer, self.healing_precision, replace=True)

        # We now have NVFP4 parameters in GPU memory (for training)
        # and FP8 parameters in host memory (for healing)
        self.pre_quantized_model = True

        if trainer.global_rank == 0:
            print(f"Pre-quantized model to {self.healing_precision} on CPU and NVFP4 in-place")

    def _get_quantized_params_cpu(self, model, quantizer, qtype: str, replace: bool = False) -> List[torch.Tensor]:
        # Unwrap layers from model
        extracted_module = extract_module(model)
        layers = extracted_module.decoder.layers
        layer_count = len(layers)

        quantized_params = []
        for layer_idx, layer in enumerate(layers):
            # Skip first and last BF16 layers
            if self.first_last_layers_bf16:
                if layer_idx < self.num_layers_at_start_in_bf16 or layer_idx >= layer_count - self.num_layers_at_end_in_bf16:
                    quantized_params.append([]) # Append empty list for consistency
                    continue

            # Quantize weights of TE modules
            quantized_layer_params = []
            for name, module in layer.named_modules():
                if not isinstance(module, (te.Linear, te.LayerNormLinear)):
                    continue
                if not hasattr(module, 'weight'):
                    continue
                param = module.weight
                with torch.no_grad():
                    qparam = quantizer(param.detach())
                    # Validate quantized tensor based on type
                    if qtype in ["MXFP8", "NVFP4"]:
                        assert qparam._rowwise_data is not None, "No rowwise data."
                        assert qparam._columnwise_data is not None, "No columnwise data."
                    elif qtype == "FP8_DS":
                        assert qparam._data is not None, "No data."
                        # Float8Tensor may or may not have transpose data
                    else:
                        raise ValueError(f"Unsupported quantization type: {qtype}")
                    
                    if replace:
                        setattr(module, 'weight', torch.nn.Parameter(qparam, requires_grad=False))
                    else:
                        qparam = qparam.clone()
                        if not self.store_quantized_params_on_gpu:
                            # Move data to CPU based on quantization type
                            if qtype in ["MXFP8", "NVFP4"]:
                                qparam._rowwise_data = qparam._rowwise_data.cpu()
                                qparam._columnwise_data = qparam._columnwise_data.cpu()
                            elif qtype == "FP8_DS":
                                qparam._data = qparam._data.cpu()
                                if hasattr(qparam, '_transpose') and qparam._transpose is not None:
                                    qparam._transpose = qparam._transpose.cpu()
                        quantized_layer_params.append(qparam)
            quantized_params.append(quantized_layer_params)

        return quantized_params

    def _set_quantized_params_cpu(self, model, cpu_params, qtype: str):
        # Unwrap layers from model
        extracted_module = extract_module(model)
        layers = extracted_module.decoder.layers
        layer_count = len(layers)

        for layer_idx, layer in enumerate(layers):
            # Skip first and last BF16 layers
            if self.first_last_layers_bf16:
                if layer_idx < self.num_layers_at_start_in_bf16 or layer_idx >= layer_count - self.num_layers_at_end_in_bf16:
                    continue

            for name, module in layer.named_modules():
                if not isinstance(module, (te.Linear, te.LayerNormLinear)):
                    continue
                if not hasattr(module, 'weight'):
                    continue
                with torch.no_grad():
                    weight = cpu_params[layer_idx].pop(0)

                    if not self.store_quantized_params_on_gpu:
                        # Move data to GPU based on quantization type
                        if qtype in ["MXFP8", "NVFP4"]:
                            weight._rowwise_data = weight._rowwise_data.to(torch.cuda.current_device())
                            weight._columnwise_data = weight._columnwise_data.to(torch.cuda.current_device())
                        elif qtype == "FP8_DS":
                            weight._data = weight._data.to(torch.cuda.current_device())
                            if hasattr(weight, '_transpose') and weight._transpose is not None:
                                weight._transpose = weight._transpose.to(torch.cuda.current_device())
                        else:
                            raise ValueError(f"Unsupported quantization type: {qtype}")

                    # Replace weight with quantized weight on GPU
                    setattr(module, 'weight', torch.nn.Parameter(weight, requires_grad=False))

            if self.cfg.model.use_transformer_engine_op_fuser:
                layer.mlp._fused_impl = (layer.mlp._make_fused_impl(),)
                layer.self_attention.linear_proj._fused_branches = layer.self_attention.linear_proj._make_fused_branches()
                layer.self_attention.linear_qkv._fused_branches = layer.self_attention.linear_qkv._make_fused_branches()


    def on_train_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            print(f"Optimized config:\n{OmegaConf.to_yaml(self.cfg)}", flush=True)
            print("\nMCore config:", flush=True)
            pprint(trainer.model.config)

        # Setup healing recipe and pre-quantize model
        self._healing_setup()
        if self.pre_quantized_model:
            self._pre_quantize_model(trainer, pl_module)

        super().on_train_start(trainer, pl_module)

    def on_train_end(self, trainer, pl_module):
        layers = trainer.model.module.module.module.module.decoder.layers
        for layer in layers:
            if hasattr(layer, "cudagraph_manager"):
                layer.cudagraph_manager = None

        FullCudaGraphWrapper.cuda_graph = None
        super().on_train_end(trainer, pl_module)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_rank == 0:
            if self.enable_perf_debug:
                torch.cuda.synchronize()
                print(f"iter {trainer.global_step} : {time.time() - self.iter_time}")
                torch.cuda.synchronize()
                self.iter_time = time.time()

            if trainer.global_step == 1 and self.enable_mem_profile:
                torch.cuda.memory._record_memory_history(max_entries=100000)

            if trainer.global_step == 3 and self.enable_mem_profile:
                torch.cuda.memory._dump_snapshot("/results/lora.pickle")
                print("memory profile written to /results/lora.pickle")

        if trainer.global_step % self.logging_interval == 0:
            self.mllogger.event(
                key="tracked_stats",
                value={"train_loss": outputs["loss"].item()},
                metadata={
                    self.mllogger.constants.SAMPLES_COUNT: trainer.global_step
                    * self.gbs,
                    "lr": trainer.optimizers[0].param_groups[0]["lr"],
                    "step": trainer.global_step,
                },
            )
        
        # start healing from healing_iter step 
        if trainer.global_step + 1 == self.cfg.model.healing_iter:
            if trainer.global_rank == 0:
                print("FP8 Healing starting...")

            # Reset cuda graphs
            torch.distributed.barrier()
            FullCudaGraphWrapper.cuda_graph["training"] = None
            FullCudaGraphWrapper.cuda_graph["validation"] = None
            FullCudaGraphWrapper.result['training'] = None
            FullCudaGraphWrapper.result['validation'] = None
            if self.cfg.model.reset_cg_after_healing:
                FullCudaGraphWrapper.curr_iteration["training"] = 0
                FullCudaGraphWrapper.curr_iteration["validation"] = 0

            # Switch to pre-quantized FP8 parameters
            if self.pre_quantized_model:
                assert self.fp8_cpu_params, "FP8 parameters not found"
                self._set_quantized_params_cpu(trainer.model, self.fp8_cpu_params, self.healing_precision)

            # Switch to healing recipe
            import megatron.core.fp4_utils
            megatron.core.fp4_utils.get_fp4_recipe = self.healing_lambda

        return super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)

    def on_validation_start(self, trainer, pl_module):
        super().on_validation_start(trainer, pl_module)

        if not trainer.warmup:
            # subsequent evaluations are every 384 sequences
            trainer.val_check_interval = ceil(int(os.environ.get("VAL_CHECK_INTERVAL", 384)) / self.gbs)
            trainer.val_check_batch = ceil(int(os.environ.get("VAL_CHECK_INTERVAL", 384)) / self.gbs)

        if self.cfg.model.eval_cp:
            trainer.model.config.context_parallel_size = self.cfg.model.eval_cp
            if self.cp_group is None:
                self.cp_group = parallel_state.get_context_parallel_group()
                self.cp_global_ranks = parallel_state.get_context_parallel_global_ranks()
                self.cp_stream = trainer.model.module.module.module.module.decoder.layers[0].self_attention.core_attention.cp_stream

            parallel_state.get_context_parallel_world_size = get_context_parallel_world_size_override
            for module in trainer.model.modules():
                for index, child in enumerate(module.modules()):
                    if isinstance(child, TEDotProductAttention) and hasattr(
                        child, "set_context_parallel_group"
                    ):
                        child.set_context_parallel_group(
                            cp_group=None,
                            cp_global_ranks=None,
                            cp_stream=None,
                            cp_comm_type=None,
                        )
            trainer.model.module.module.module.module.rotary_pos_emb.cp_group = None

        if self.cfg.model.val_micro_batch_size:
            reconfigure_num_microbatches_calculator(
                rank=torch.distributed.get_rank(),
                rampup_batch_size=None,
                global_batch_size=self.cfg.model.val_global_batch_size,
                micro_batch_size=self.cfg.model.val_micro_batch_size,
                data_parallel_size=trainer.val_dataloaders.batch_sampler.data_parallel_size,
            )

    def on_validation_end(self, trainer, pl_module):
        if self.cfg.model.eval_cp:
            trainer.model.config.context_parallel_size = self.cfg.model.context_parallel_size
            parallel_state.get_context_parallel_world_size = original_get_context_parallel_world_size
            for module in trainer.model.modules():
                for index, child in enumerate(module.modules()):
                    if isinstance(child, TEDotProductAttention) and hasattr(
                        child, "set_context_parallel_group"
                    ):
                        child.set_context_parallel_group(
                            cp_group=self.cp_group,
                            cp_global_ranks=self.cp_global_ranks,
                            cp_stream=self.cp_stream,
                            cp_comm_type=self.cfg.model.cp_comm_type,
                        )
            trainer.model.module.module.module.module.rotary_pos_emb.cp_group = self.cp_group

        if self.cfg.model.val_micro_batch_size:
            reconfigure_num_microbatches_calculator(
                rank=torch.distributed.get_rank(),
                rampup_batch_size=None,
                global_batch_size=self.cfg.model.global_batch_size,
                micro_batch_size=self.cfg.model.micro_batch_size,
                data_parallel_size=trainer.train_dataloader.batch_sampler.data_parallel_size,
            )

        super().on_validation_end(trainer, pl_module)

    def warmup(self, trainer, pl_module):
        if self.pre_quantized_model and self.cfg.model.custom.warmup_healing:
            # Healing warmup
            import megatron.core.fp4_utils
            original_fp4_recipe = megatron.core.fp4_utils.get_fp4_recipe
            healing_fp8_recipe = self.healing_lambda
            megatron.core.fp4_utils.get_fp4_recipe = healing_fp8_recipe

            run_training_warmup(
                trainer,
                self.cfg.model.custom.warmup_train_steps,
                self.cfg.model.custom.warmup_validation_steps,
            )
            if (
                self.cfg.model.fp8
                and self.cfg.model.custom.reset_fp8_stats_after_warmup
            ):
                reset_fp8_state(pl_module)
            megatron.core.fp4_utils.get_fp4_recipe = original_fp4_recipe

            # Replace model with NVFP4 params
            self._set_quantized_params_cpu(trainer.model, self.fp4_cpu_params, "NVFP4")

        if self.cfg.model.external_cuda_graph:
            self.log_custom_timedelta("before_cuda_graph", trainer.global_step)
            capture_cuda_graph(trainer, self.cfg.model)
            self.log_custom_timedelta("cuda_graph", trainer.global_step)
        if self.cfg.model.custom.warmup:
            # Main training warmup
            run_training_warmup(
                trainer,
                self.cfg.model.custom.warmup_train_steps,
                self.cfg.model.custom.warmup_validation_steps,
            )
            if (
                self.cfg.model.fp8
                and self.cfg.model.custom.reset_fp8_stats_after_warmup
            ):
                reset_fp8_state(pl_module)


class MetricsLogger(MLPerfLogger):
    def __init__(
        self,
        cfg: DictConfig,
        model: pl.LightningModule,
    ):
        super().__init__(CustomCallback, model, "val_loss_sum", 0.925, "min", cfg)
        self.gbs = cfg.model.global_batch_size
        self.mbs = cfg.model.micro_batch_size
        self.cfg = cfg

    def __deepcopy__(self, memo):
        output = MetricsLogger(self.cfg, self.model)
        if self.trainer is not None:
            output.trainer = self.trainer
        return output

    def compute_hyperparams(
        self, params: dict[str, Any], *args, **kwargs
    ) -> dict[str, Any]:
        return {
            self.mllogger.constants.GLOBAL_BATCH_SIZE: self.gbs,
            self.mllogger.constants.TRAIN_SAMPLES: np.load(
                f"{self.cfg.data_root}/train.npy", allow_pickle=True
            ).shape[0],
            self.mllogger.constants.EVAL_SAMPLES: np.load(
                f"{self.cfg.data_root}/validation.npy", allow_pickle=True
            ).shape[0],
            self.mllogger.constants.GRADIENT_ACCUMULATION_STEPS: int(
                os.getenv("MINIBS", "1")
            )
            // self.mbs,
        }

    def compute_validation_metric(self, metrics: dict[str, float]) -> float:
        return metrics["val_loss_sum"] / metrics["val_loss_count"]
