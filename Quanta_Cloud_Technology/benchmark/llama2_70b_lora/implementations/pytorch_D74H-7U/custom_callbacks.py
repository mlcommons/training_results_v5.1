# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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
from itertools import repeat

from typing import Dict, Optional
import numpy as np
import torch
from megatron.core.num_microbatches_calculator import get_num_microbatches
from megatron.core import parallel_state
from megatron.core.transformer.custom_layers.transformer_engine import TEDelayedScaling
from mlperf_logging.mllog import constants
from nemo.collections.nlp.data.language_modeling.megatron.blendable_dataset import BlendableDataset
from nemo.collections.nlp.data.language_modeling.megatron.megatron_batch_samplers import MegatronPretrainingBatchSampler
from nemo.collections.nlp.models.language_modeling.megatron_gpt_sft_model import MegatronGPTSFTModel
from nemo.utils import logging
from pytorch_lightning import Callback
from pytorch_lightning.loggers import Logger
from pytorch_lightning.utilities import rank_zero_only
from torch.utils.data import default_collate
from transformer_engine.common import recipe
from transformer_engine.pytorch import make_graphed_callables
from megatron.core.distributed import DistributedDataParallel as McoreDDP
from megatron.core.transformer.module import Float16Module as MCoreFloat16Module
from nemo.collections.nlp.modules.common.megatron.module import Float16Module


def extract_module(model):
    if isinstance(model, list):
        return extract_module(model[0])
    elif isinstance(model, (McoreDDP, Float16Module, MCoreFloat16Module)):
        return extract_module(model.module)
    else:
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
    logging.info(f'In capture_cuda_graph')
    start = time.time()
    schedule = get_microbatch_schedule(get_num_microbatches(), len(trainer.model.model) if isinstance(trainer.model.model, list) else None)
    assert parallel_state.get_pipeline_model_parallel_world_size() == 1, "CUDA graph requires PP=1"
    if cfg.ub_tp_comm_overlap and trainer.model.initialize_ub:
        trainer.model.initialize_ub_func()
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
            config=extract_module(trainer.model.model).decoder.config,
            fp8_format=fp8_format,
            override_linear_precision=(False, False, False),
        )

    callables = []
    sample_args = []
    optional_args = []
    for l_no, layer in enumerate(extract_module(trainer.model.model).decoder.layers):
        callables.append(layer)
        graph_input = (torch.ones((slen, micro_batch_size, 8192), dtype=torch.bfloat16, requires_grad=True, device='cuda'),)
        sample_args.append(graph_input)
        optional_input = {'rotary_pos_emb': torch.ones((8192//cfg.context_parallel_size, 1, 1, 128), dtype=torch.float32, requires_grad=False, device='cuda')}
        optional_args.append(optional_input)
    graphs = make_graphed_callables(tuple(callables), tuple(sample_args), _order=schedule, allow_unused_input=True, fp8_enabled=cfg.fp8, fp8_recipe=fp8_recipe, fp8_weight_caching=cfg.enable_cg_fp8_weight_caching, sample_kwargs=optional_args,)
    for l_no, layer in enumerate(extract_module(trainer.model.model).decoder.layers):
        extract_module(trainer.model.model).decoder.cuda_graphs[l_no] = [graphs[l_no]]
    trainer.model.initialize_ub = False
    torch.distributed.barrier()
    logging.info(f'CUDAgraph capture done')


def run_training_warmup(trainer, warmup_train_steps, warmup_validation_steps):
    torch.distributed.barrier()
    start = time.time()
    # Run forward and backward (no optimizer step)
    for _ in range(warmup_train_steps):
        trainer.model.training_step(trainer.model.get_synthetic_input())
    torch.distributed.barrier()
    logging.info(f"Finished training warmup: {time.time() - start}s. Starting validation warmup")

    if warmup_validation_steps > 0:
        trainer.training = False
        for _ in range(warmup_validation_steps):
            trainer.model.validation_step(trainer.model.get_synthetic_input(validation=True))
        trainer.training = True
        trainer.model.validation_step_outputs.clear()

    # For GPT `zero_grad` is a noop, but included here for completeness
    trainer.model.zero_grad()
    trainer._logger_connector.reset_results()
    trainer._logger_connector.reset_metrics()
    torch.distributed.barrier()
    logging.info(f"Time spent in run_training_warmup: {time.time() - start}s")


def reset_fp8_state(model):
    """Sets `fp8_initialized` flag to False in every TE layer which will force reinitialization."""
    logging.info("Forcing FP8 stats reinitialization")

    def reset_fp8(m):
        if hasattr(m, "fp8_initialized"):
            m.fp8_initialized = False

    models = model.model
    for model in models if isinstance(models, list) else [models]:
        model.apply(reset_fp8)


class Timer:
    def __init__(self, gbs):
        self.start_time = None
        self.stop_time = None
        self.elapsed_time = 0
        self.samples = 0
        self.gbs = gbs

    def start(self):
        self.start_time = time.time()

    def stop(self):
        self.stop_time = time.time()
        self.samples += self.gbs
        self.elapsed_time += self.stop_time - self.start_time

    def get_throughput(self):
        throughput = self.samples / self.elapsed_time
        self.samples = 0
        self.elapsed_time = 0
        return throughput


class CustomCallback(Callback):
    def __init__(self, cfg, mllogger):
        super().__init__()
        self.cfg = cfg
        self.gbs = cfg.model.global_batch_size
        self.mllogger = mllogger
        self.timer = Timer(self.gbs)
        self.force_success = os.getenv("FORCE_SUCCESS_STATUS", "0") == "1"

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self.timer.start()
        return super().on_train_batch_start(trainer, pl_module, batch, batch_idx)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.timer.stop()
        if trainer.global_step % 10 == 0:
            self.mllogger.event(
                "train_loss",
                value=outputs["loss"].item(),
                metadata={
                    "samples_count": trainer.global_step * self.gbs,
                    "lr": trainer.optimizers[0].param_groups[0]["lr"],
                },
            )

        return super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)

    def on_validation_start(self, trainer, pl_module):
        self.log_throughput(trainer.global_step)
        self.mllogger.end(
            constants.BLOCK_STOP,
            metadata={"samples_count": trainer.global_step * self.gbs},
            sync=False,
        )
        self.mllogger.start(
            key=constants.EVAL_START,
            metadata={"samples_count": trainer.global_step * self.gbs},
            sync=False,
        )
        # subsequent evaluations are every 384 sequences
        trainer.val_check_interval = int(os.environ.get("VAL_CHECK_INTERVAL", 384)) // self.gbs
        trainer.val_check_batch = int(os.environ.get("VAL_CHECK_INTERVAL", 384)) // self.gbs
        return super().on_validation_start(trainer, pl_module)

    def on_validation_end(self, trainer, pl_module):
        if not trainer.should_stop:
            self.mllogger.start(
                constants.BLOCK_START,
                metadata={"samples_count": trainer.global_step * self.gbs},
                sync=False,
            )
        return super().on_validation_end(trainer, pl_module)

    def on_train_start(self, trainer, pl_module):
        if pl_module.cfg.enable_cuda_graph:
            capture_cuda_graph(trainer, pl_module.cfg)
        # Warmup
        if pl_module.cfg.custom.warmup:
            run_training_warmup(
                trainer, pl_module.cfg.custom.warmup_train_steps, pl_module.cfg.custom.warmup_validation_steps
            )
            if pl_module.cfg.fp8 and pl_module.cfg.custom.reset_fp8_stats_after_warmup:
                reset_fp8_state(pl_module)

        self.mllogger.log_init_stop_run_start()
        self.mllogger.start(constants.BLOCK_START, metadata={"samples_count": 0}, sync=False)

    def on_train_end(self, trainer, pl_module):
        if not trainer.run_stop_logged:
            self.mllogger.end(
                constants.RUN_STOP,
                metadata={
                    "samples_count": trainer.global_step * self.gbs,
                    "status": "aborted" if not self.force_success else "success",
                },
            )
        return super().on_train_end(trainer, pl_module)

    def log_throughput(self, global_step):
        throughput = self.timer.get_throughput()
        self.mllogger.event(
            key="tracked_stats",
            metadata={"step": global_step * self.gbs},
            value={"throughput": throughput},
        )


class MetricsLogger(Logger):
    def __init__(self, cfg, mllogger, trainer):
        super().__init__()
        self.cfg = cfg
        self.gbs = cfg.model.global_batch_size
        self.mllogger = mllogger
        self.experiment = None
        self.trainer = trainer
        trainer.run_stop_logged = False

    def log_metrics(self, metrics, step):
        if "val_loss" in metrics:
            val_loss = metrics["val_loss"]
            self.mllogger.event(
                constants.EVAL_ACCURACY,
                value=val_loss,
                metadata={"samples_count": self.trainer.global_step * self.gbs},
            )
            self.mllogger.end(
                key=constants.EVAL_STOP,
                metadata={"samples_count": self.trainer.global_step * self.gbs},
                sync=False,
            )

            if val_loss < 0.925:
                self.trainer.should_stop = True
                self.trainer.run_stop_logged = True
                self.mllogger.end(
                    constants.RUN_STOP,
                    value=val_loss,
                    metadata={
                        "samples_count": self.trainer.global_step * self.gbs,
                        "status": "success",
                    },
                )

    @rank_zero_only
    def log_hyperparams(self, params, *args, **kwargs):
        self.mllogger.event(key=constants.CACHE_CLEAR, value=True)
        self.mllogger.start(key=constants.INIT_START)
        self.mllogger.mlperf_submission_log(benchmark="llama2_70b_lora", num_nodes=self.cfg.trainer.num_nodes)
        self.mllogger.event(
            key=constants.SEED,
            value=self.cfg.model.seed,
            sync=False,
            unique=True,
        )
        self.mllogger.event(
            key=constants.GLOBAL_BATCH_SIZE,
            value=self.cfg.model.global_batch_size,
            sync=False,
        )
        self.mllogger.event(
            key=constants.TRAIN_SAMPLES,
            value=np.load("/data/train.npy", allow_pickle=True).shape[0],
        )
        self.mllogger.event(
            key=constants.EVAL_SAMPLES,
            value=np.load("/data/validation.npy", allow_pickle=True).shape[0],
        )
        self.mllogger.event(
            key=constants.OPT_LR_WARMUP_FACTOR,
            value=self.cfg.model.optim.sched.warmup_ratio,
        )
        self.mllogger.event(
            key=constants.OPT_ADAMW_WEIGHT_DECAY,
            value=self.cfg.model.optim.weight_decay,
        )
        self.mllogger.event(
            key=constants.OPT_GRADIENT_CLIP_NORM,
            value=self.cfg.trainer.gradient_clip_val,
        )
        ga = int(os.getenv("MINIBS", "1")) // self.cfg.model.micro_batch_size
        self.mllogger.event(key=constants.GRADIENT_ACCUMULATION_STEPS, value=ga)
        self.mllogger.event(key=constants.OPT_LR_TRAINING_STEPS, value=self.cfg.trainer.max_steps)
        self.mllogger.event(key=constants.OPT_BASE_LR, value=self.cfg.model.optim.lr)
        self.mllogger.event(key="lora_rank", value=self.cfg.model.peft.lora_tuning.adapter_dim)
        self.mllogger.event(key="lora_alpha", value=self.cfg.model.peft.lora_tuning.alpha)

    @property
    def name(self):
        return "mlperf-metrics"

    @property
    def version(self):
        return 1


class CustomMegatronGPTSFTModel(MegatronGPTSFTModel):
    def get_synthetic_input(self, validation=False):
        # Needed because init_global_step is not initialized at warmup
        self.init_global_step = self.trainer.global_step

        # Synthetic data generation below may need to change
        # if either dataset or sampler changes
        if not isinstance(self._train_ds, BlendableDataset) or not isinstance(
            self._train_dl.batch_sampler, MegatronPretrainingBatchSampler
        ):
            raise NotImplementedError(
                f"No synthetic data implementation for "
                f'dataset "{self._train_ds}" and '
                f'data sampler "{self._train_dl.batch_sampler}"'
            )

        # Create arbitrary text of sequence length
        seq_length = self.cfg.encoder_seq_length
        text = torch.ones(seq_length + 1, dtype=torch.int64) * 3545

        tokens = text[:-1].contiguous()
        tokens[-1] = 2

        labels = text[1:].contiguous()
        labels[-1] = 2

        attention_mask_shape = [self.cfg.micro_batch_size, seq_length, seq_length]
        attention_mask = torch.ones(attention_mask_shape, dtype=torch.bool)

        loss_mask = torch.ones(seq_length, dtype=torch.int64)
        loss_mask[-1] = 0

        position_ids = torch.tensor([i for i in range(seq_length)], dtype=torch.int64)
        position_ids[-1] = 0

        single_data = {
            "tokens": tokens,
            "labels": labels,
            "loss_mask": loss_mask,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
        }

        batch = default_collate([single_data] * self.cfg.micro_batch_size * get_num_microbatches())
        batch["token_count"] = [seq_length - 1 for _ in range(self.cfg.micro_batch_size * get_num_microbatches())]
        if validation:
            return repeat((batch, 0, 0))
        return repeat(batch)
