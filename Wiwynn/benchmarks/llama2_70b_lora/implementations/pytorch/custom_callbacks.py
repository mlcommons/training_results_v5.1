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
from typing import Any

import lightning as pl
import numpy as np
import torch
from cudagraphs import capture_cuda_graph
from custom_llama import reset_fp8_state, run_training_warmup
from megatron.core import parallel_state
from megatron.core.extensions.transformer_engine import TEDotProductAttention
from megatron.core.num_microbatches_calculator import (
    reconfigure_num_microbatches_calculator,
)
from megatron.core.parallel_state import get_context_parallel_group
from mlperf_common.callbacks import LoggingCallback, MLPerfLogger
from omegaconf import DictConfig


class CustomCallback(LoggingCallback):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.gbs = cfg.model.global_batch_size
        self.memory_profile = 0
        self.iter = 0
        self.iter_time = time.time()
        self.enable_mem_profile = False
        self.enable_perf_debug = False
        if "ENABLE_PERF_DEBUG" in os.environ:
            self.enable_perf_debug = True
        if "ENABLE_MEMORY_PROFILE" in os.environ:
            self.enable_mem_profile = True
            torch.cuda.memory._record_memory_history(max_entries=100000)
        self.cp_group = None
        self.cp_global_ranks = None

    def __deepcopy__(self, memo):
        return CustomCallback(self.cfg)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_rank == 0:
            self.iter += 1
            if self.enable_perf_debug:
                torch.cuda.synchronize()
                print(f"iter {self.iter} : {time.time() - self.iter_time}")
                torch.cuda.synchronize()
                self.iter_time = time.time()
            if self.iter == 1 and self.enable_mem_profile:
                torch.cuda.memory._record_memory_history(max_entries=100000)

            if self.iter == 3 and self.enable_mem_profile:
                torch.cuda.memory._dump_snapshot(f"/results/lora.pickle")
                print("memory profile written")

        if trainer.global_step % 10 == 0:
            self.mllogger.event(
                "train_loss",
                value=outputs["loss"].item(),
                metadata={
                    self.mllogger.constants.SAMPLES_COUNT: trainer.global_step
                    * self.gbs,
                    "lr": trainer.optimizers[0].param_groups[0]["lr"],
                },
            )

        return super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)

    def on_validation_start(self, trainer, pl_module):
        super().on_validation_start(trainer, pl_module)

        if not trainer.warmup:
            # subsequent evaluations are every 384 sequences
            trainer.val_check_interval = (
                int(os.environ.get("VAL_CHECK_INTERVAL", 384)) // self.gbs
            )
            trainer.val_check_batch = (
                int(os.environ.get("VAL_CHECK_INTERVAL", 384)) // self.gbs
            )

        if self.cfg.model.eval_cp:
            trainer.model.config.context_parallel_size = self.cfg.model.eval_cp

            def get_context_parallel_world_size_override():
                return int(self.cfg.model.eval_cp)

            self.cp_group = parallel_state.get_context_parallel_group()
            self.cp_global_ranks = parallel_state.get_context_parallel_global_ranks()
            parallel_state.get_context_parallel_world_size = (
                get_context_parallel_world_size_override
            )
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
            trainer.model.module.module.module.module.rotary_pos_emb.forward.cache_clear()

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
            trainer.model.config.context_parallel_size = (
                self.cfg.model.context_parallel_size
            )

            def get_context_parallel_world_size_override():
                if (
                    torch.distributed.is_available()
                    and torch.distributed.is_initialized()
                ):
                    return torch.distributed.get_world_size(
                        group=get_context_parallel_group()
                    )
                else:
                    return 0

            parallel_state.get_context_parallel_world_size = (
                get_context_parallel_world_size_override
            )
            for module in trainer.model.modules():
                for index, child in enumerate(module.modules()):
                    if isinstance(child, TEDotProductAttention) and hasattr(
                        child, "set_context_parallel_group"
                    ):
                        child.set_context_parallel_group(
                            cp_group=self.cp_group,
                            cp_global_ranks=self.cp_global_ranks,
                            cp_stream=torch.cuda.Stream(),
                            cp_comm_type=self.cfg.model.cp_comm_type,
                        )
            trainer.model.module.module.module.module.rotary_pos_emb.forward.cache_clear()

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
        if self.cfg.model.external_cuda_graph:
            self.log_custom_timedelta("before_cuda_graph", trainer.global_step)
            capture_cuda_graph(trainer, self.cfg.model)
            self.log_custom_timedelta("cuda_graph", trainer.global_step)
        if self.cfg.model.custom.warmup:
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
                "/data/train.npy", allow_pickle=True
            ).shape[0],
            self.mllogger.constants.EVAL_SAMPLES: np.load(
                "/data/validation.npy", allow_pickle=True
            ).shape[0],
            self.mllogger.constants.GRADIENT_ACCUMULATION_STEPS: int(
                os.getenv("MINIBS", "1")
            )
            // self.mbs,
        }

    def compute_validation_metric(self, metrics: dict[str, float]) -> float:
        return metrics["val_loss_sum"] / metrics["val_loss_count"]
