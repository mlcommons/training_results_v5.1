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

from itertools import repeat

import torch
from megatron.core.num_microbatches_calculator import get_num_microbatches
from nemo.collections.llm.gpt.model import LlamaModel
from torch.utils.data import default_collate
from nemo.lightning.megatron_parallel import MegatronLossReduction
from typing import Dict, List, Tuple
from megatron.core import parallel_state
from nemo.collections.nlp.modules.common.megatron.utils import average_losses_across_data_parallel_group


def run_training_warmup(trainer, warmup_train_steps, warmup_validation_steps):
    trainer.warmup = True
    torch.distributed.barrier()
    for _ in range(warmup_train_steps):
        trainer.model.training_step(trainer.model.get_synthetic_input())
    torch.distributed.barrier()

    if warmup_validation_steps > 0:
        with torch.no_grad():
            trainer.fit_loop.epoch_loop.val_loop.on_run_start()
            trainer.training = False
            trainer.validating = True
            for _ in range(warmup_validation_steps):
                trainer.model.validation_step(trainer.model.get_synthetic_input(validation=True))
            trainer.fit_loop.epoch_loop.val_loop.on_run_end()
            trainer.training = True
            trainer.validating = False

    # For GPT `zero_grad` is a noop, but included here for completeness
    trainer.model.zero_grad()
    trainer._logger_connector.reset_results()
    trainer._logger_connector.reset_metrics()
    torch.distributed.barrier()
    trainer.warmup = False


def reset_fp8_state(model):
    """Sets `fp8_initialized` flag to False in every TE layer which will force reinitialization."""
    def reset_fp8(m):
        if hasattr(m, "fp8_initialized"):
            m.fp8_initialized = False
            m.reset_fp8_meta_tensors()
    model.apply(reset_fp8)


class CustomLlamaModel(LlamaModel):
    def configure_model(self):
        import transformer_engine.pytorch as te
        self.custom_callback.log_custom_timedelta("before_model_init")
        with torch.no_grad(), te.fp8_model_init():
            super().configure_model()
        self.custom_callback.log_custom_timedelta("after_model_init")
        s = torch.cuda.Stream()
        torch.cuda.set_stream(s)

    def get_synthetic_input(self, validation=False):
        # Needed because init_global_step is not initialized at warmup
        self.init_global_step = self.trainer.global_step

        # Create arbitrary text of sequence length
        seq_length = 8192
        text = torch.ones(seq_length + 1, dtype=torch.int64) * 3545

        tokens = text[:-1].contiguous()
        tokens[-1] = 2

        labels = text[1:].contiguous()
        labels[-1] = 2

        attention_mask_shape = [1, seq_length, seq_length]
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

        batch = default_collate([single_data] * get_num_microbatches())
        batch["token_count"] = [seq_length - 1 for _ in range(get_num_microbatches())]
        return repeat((batch, 0, 0))

    @property
    def training_loss_reduction(self):
        if not self._training_loss_reduction:
            self._training_loss_reduction = MaskedTokenLossReduction()

        return self._training_loss_reduction

    @property
    def validation_loss_reduction(self):
        if not self._validation_loss_reduction:
            self._validation_loss_reduction = MaskedTokenLossReduction(validation_step=True, val_drop_last=False, cp_eval=self.config.cp_eval)

        return self._validation_loss_reduction


class MaskedTokenLossReduction(MegatronLossReduction):
    def __init__(self, validation_step: bool = False, val_drop_last: bool = True, cp_eval: int = None) -> None:
        super().__init__()
        self.validation_step = validation_step
        self.train_step = not validation_step
        self.val_drop_last = val_drop_last
        self.disabled_cp_for_eval = cp_eval is not None

    def forward(
        self, batch: Dict[str, torch.Tensor], per_token_losses: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if isinstance(per_token_losses, tuple):
            per_token_losses, loss_mask = per_token_losses
            batch["loss_mask"] = loss_mask
        masked_losses = per_token_losses * batch["loss_mask"]

        cp_size = parallel_state.get_context_parallel_world_size()
        if cp_size > 1 and (self.train_step or not self.disabled_cp_for_eval):
            torch.distributed.all_reduce(batch["loss_mask"], group=parallel_state.get_context_parallel_group())

        num_valid_tokens = batch["loss_mask"].sum(1)
        loss_for_ub = torch.sum(masked_losses, dim=1) / num_valid_tokens
        loss_for_ub = torch.where(num_valid_tokens == 0, torch.zeros_like(loss_for_ub), loss_for_ub)

        if cp_size > 1 and (self.train_step or not self.disabled_cp_for_eval):
            torch.distributed.all_reduce(loss_for_ub, group=parallel_state.get_context_parallel_group())

        if self.validation_step and not self.val_drop_last:
            num_valid_tokens_in_ub = (num_valid_tokens > 0).long()
            loss_sum_and_ub_size_all_gpu = torch.stack([loss_for_ub.clone().detach(), num_valid_tokens_in_ub.clone().detach()], dim=1)
            if self.disabled_cp_for_eval:
                torch.distributed.all_reduce(loss_sum_and_ub_size_all_gpu)
            else:
                torch.distributed.all_reduce(loss_sum_and_ub_size_all_gpu, group=parallel_state.get_data_parallel_group())
            return loss_for_ub, {"loss_sum_and_ub_size": loss_sum_and_ub_size_all_gpu}

        reduced_loss = average_losses_across_data_parallel_group([loss_for_ub])
        return loss_for_ub, {"avg": reduced_loss}

    def reduce(self, losses_reduced_per_micro_batch) -> torch.Tensor:
        if losses_reduced_per_micro_batch:
            if "avg" in losses_reduced_per_micro_batch[0]:
                loss_tensors_list = [loss_reduced["avg"] for loss_reduced in losses_reduced_per_micro_batch]
                loss_tensor = torch.concat(loss_tensors_list)
                return loss_tensor.mean()

            loss_sum_tensors_list: List[torch.Tensor] = [
                loss_sum["loss_sum_and_ub_size"]
                for loss_sum in losses_reduced_per_micro_batch
            ]
            loss_sum = (
                torch.vstack(loss_sum_tensors_list).sum(dim=0)
                if len(loss_sum_tensors_list) > 0
                else torch.tensor([0.0, 0.0], device=torch.cuda.current_device())
            )
            return loss_sum

        return torch.tensor(0.0, device=torch.cuda.current_device())
