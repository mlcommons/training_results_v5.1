# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import logging
import math
import os
import time
from argparse import Namespace
from datetime import datetime
from itertools import repeat
from pathlib import Path
from typing import Any, Dict, Optional, Union

import nemo

try:
    # TODO: Remove this once we have full transition to Lightning 2.0
    import lightning.pytorch as pl
    from lightning.pytorch import Callback
    from lightning.pytorch.loops import _TrainingEpochLoop as TrainingEpochLoop
    from lightning.pytorch.utilities import rank_zero_only
except ImportError:
    import pytorch_lightning as pl
    from pytorch_lightning import Callback
    from pytorch_lightning.loops import _TrainingEpochLoop as TrainingEpochLoop
    from pytorch_lightning.utilities import rank_zero_only

import torch
from lightning_fabric.utilities.cloud_io import get_filesystem
from megatron.core import parallel_state
from megatron.core.datasets.gpt_dataset import _get_ltor_masks_and_position_ids
from megatron.core.num_microbatches_calculator import get_num_microbatches
from mlperf_common.callbacks import LoggingCallback, MLPerfLogger, mllogger
from nemo.collections.nlp.data.language_modeling.megatron.data_samplers import (
    MegatronPretrainingSampler,
)
from nemo.collections.nlp.data.language_modeling.megatron.gpt_dataset import (
    MockGPTDataset,
)
from nemo.collections.nlp.data.language_modeling.megatron.megatron_batch_samplers import (
    BaseMegatronBatchSampler,
)
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import (
    MegatronGPTModel,
)
from nemo.constants import NEMO_ENV_VARNAME_TESTING
from nemo.utils import AppState
from nemo.utils import logging as nl_logging
from nemo.utils.env_var_parsing import get_envbool
from nemo.utils.exp_manager import SkipResumeTrainingValidationLoop, TimingCallback
from nemo.utils.formatters.base import BaseNeMoFormatter, DebugNeMoFormatter
from nemo.utils.get_rank import is_global_rank_zero
from nemo.utils.timers import NamedTimer
from torch.utils.data import default_collate
from transformer_engine.common import recipe
from transformer_engine.pytorch import make_graphed_callables

_PATH = Union[str, Path]

logger = logging.getLogger(__name__)


# Not used for now since it is included in mlperf-common: https://github.com/NVIDIA/mlperf-common/blob/main/mlperf_common/callbacks/logging.py
def compute_consumed_mllog_tokens(
    trainer, init_global_step, global_batch_size, micro_batch_size, seq_length
):

    # no need to do this now, since we don't resume
    # but keeping this line for future references for now
    # current_step = trainer.global_step - init_global_step
    current_step = trainer.global_step

    gbs = global_batch_size
    model_gbs = (
        AppState().data_parallel_size * micro_batch_size * get_num_microbatches()
    )
    assert gbs == model_gbs, (gbs, model_gbs)
    consumed_samples = current_step * gbs
    return int(consumed_samples)


def run_training_warmup(
    trainer, warmup_train_steps, warmup_validation_steps, enable_cuda_graph
):
    torch.distributed.barrier()

    mock_dataloader = iter(trainer.mock_dataset.train_dataloader())

    # Warmup for training
    if torch.distributed.get_rank() == 0:
        logger.info(f"Starting training warmup")
    start = time.time()
    for _ in range(warmup_train_steps):
        # Runs forward / backward step
        _ = trainer.model.training_step(data=mock_dataloader)

        # clears all gradients
        for chunk in trainer.model:
            chunk.module.zero_grad_buffer()
            chunk.module.zero_grad()

    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        logger.info(f"Finished training warmup: {time.time() - start} s. ")

    if not enable_cuda_graph:
        # Warmup for validation
        start = time.time()
        if torch.distributed.get_rank() == 0:
            logger.info(f"Starting validation warmups")
        trainer.testing = True
        trainer.training = False
        for _ in range(warmup_validation_steps):
            # Runs forward step
            # it makes no difference whether we use mock_dataset.train_dl or val_dl
            # since it's random data anyway
            _ = trainer.model.validation_step(data=mock_dataloader)

            # no need to clear grad

        trainer.testing = False
        trainer.training = True
        torch.distributed.barrier()
        if torch.distributed.get_rank() == 0:
            logger.info(f"Finished validation warmup: {time.time() - start} s. ")

    # For GPT `zero_grad` is a noop, but included here for completeness
    trainer.model.zero_grad()
    trainer._logger_connector.reset_results()
    trainer._logger_connector.reset_metrics()

    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        logger.info(f"Time spent in run_training_warmup: {time.time() - start}s")


def reset_fp8_state(model):
    """Sets `fp8_initialized` flag to False in every TE layer which will force reinitialization."""

    for layer in model.modules():
        if hasattr(layer, "fp8_initialized"):
            layer.fp8_initialized = False


class CustomCallback(LoggingCallback):
    def __init__(self, cfg):
        super().__init__()

        # warmups
        self.run_warmup_on_synth_data = cfg.model.custom.run_warmup_on_synth_data
        self.warmup_train_steps = cfg.model.custom.warmup_train_steps
        self.warmup_validation_steps = cfg.model.custom.warmup_validation_steps
        self.reset_fp8_stats_after_warmup = (
            cfg.model.fp8 and cfg.model.custom.reset_fp8_stats_after_warmup
        )

        # Checkpoint init global step
        self.init_global_step = cfg.model.custom.init_global_step

        # data_module
        self.global_batch_size = cfg.model.global_batch_size
        self.micro_batch_size = cfg.model.micro_batch_size
        self.seq_length = cfg.model.encoder_seq_length

        # Debugging related
        if cfg.model.custom.force_success_status:
            self.status = mllogger.constants.SUCCESS
        else:
            self.status = mllogger.constants.ABORTED
        self.is_target_reached = False
        self.is_run_stop_already_logged = False
        self.enable_cuda_graph = cfg.model.overwritten_attributes.enable_cuda_graph

        self.tokens_per_block = (
            cfg.trainer.val_check_interval
            * cfg.model.global_batch_size
            * cfg.model.encoder_seq_length
        )
        self.iter_after_valid = False

    def __deepcopy__(self, memo):
        return CustomCallback(self.cfg)

    def warmup(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if self.run_warmup_on_synth_data:
            run_training_warmup(
                trainer,
                self.warmup_train_steps,
                self.warmup_validation_steps,
                self.enable_cuda_graph,
            )
            if self.reset_fp8_stats_after_warmup:
                reset_fp8_state(pl_module)

        for callback in trainer.callbacks:
            if isinstance(callback, DeltaTimingCallback):
                callback.trigger_tick()

    @rank_zero_only
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        print(f":::MLLOG Weight initialization: {state_dict.keys()}")
        return super().load_state_dict(state_dict)


class MetricsLogger(MLPerfLogger):
    def __init__(
        self,
        model,
        cfg,
        optim,
        train_loss_key="reduced_train_loss",
        val_loss_key="val_loss",
        timing_keys=(
            "train_step_time",
            "train_epoch_time",
            "validation_step_time",
            "validation_epoch_time",
        ),
        throughput_key="train_epoch_time",
    ):
        self.target_val_log_ppl = cfg.model.custom.target_log_ppl
        self.extend_run_evals = cfg.model.custom.extend_run_evals

        super().__init__(
            CustomCallback,
            model,
            val_loss_key,
            self.target_val_log_ppl,
            "min",
            cfg,
        )

        # TODO: HACK : Fix this
        self.init_global_step = 0

        # Nemo 2.0

        # data_module
        self.global_batch_size = cfg.model.global_batch_size
        self.micro_batch_size = cfg.model.micro_batch_size
        self.seq_length = cfg.model.encoder_seq_length

        self.optim = optim

        # Nemo 1.0
        self.model = model
        self.val_loss_key = val_loss_key
        self.train_loss_key = train_loss_key
        self.timing_keys = timing_keys
        self.throughput_key = throughput_key

    def __deepcopy__(self, memo):
        output = MetricsLogger(
            self.model,
            self.cfg,
            self.optim,
            self.train_loss_key,
            self.val_loss_key,
            self.timing_keys,
            self.throughput_key,
        )
        return output

    def log_metrics(
        self, metrics: Dict[str, float], step: Optional[int] = None
    ) -> None:
        super().log_metrics(metrics, step)
        self._log_throughputs(metrics, step)
        if bool(os.getenv("ENABLE_TRAIN_BARRIER", "")):
            torch.cuda.synchronize()
            torch.distributed.barrier()
            if bool(os.getenv("LOG_TRAIN_BARRIER", "")):
                logger.info(f"Train Step End")
        # Consumed samples is shifted by 1 (in terms of gbs), beacuse `trainer.global_step`
        # is not incremented by the time `consumed_samples` is logged (in model forward)
        # Recomputing in here:
        # TODO: Fix this
        # if 'consumed_samples' in self.trainer.callback_metrics:
        #     correct_consumed_samples = self.model.compute_consumed_samples(self.trainer.global_step - self.init_global_step)
        #     self.trainer.callback_metrics['consumed_samples'].fill_(correct_consumed_samples)

    def _log_throughputs(self, metrics: Dict[str, float], step: Optional[int] = None):
        samples = (step + 1) * self.global_batch_size
        for timing_key in self.timing_keys:
            if timing_key in metrics:
                timing = metrics[timing_key]
                loss_data = {}
                if self.train_loss_key in metrics:
                    loss_data[self.train_loss_key] = metrics[self.train_loss_key]
                if os.environ.get("USE_DATETIME", "0") == "1":
                    self.mllogger.event(
                        key="tracked_stats",
                        metadata={"step": samples},
                        value={
                            timing_key: timing,
                            **loss_data,
                            "time_now": str(datetime.now()),
                        },
                    )
                else:
                    self.mllogger.event(
                        key="tracked_stats",
                        metadata={"step": samples},
                        value={timing_key: timing, **loss_data},
                    )

        if self.throughput_key in metrics:
            timing = metrics[self.throughput_key]
            throughput = samples / timing
            self.mllogger.event(
                key="tracked_stats",
                metadata={"step": samples},
                value={"throughput": throughput},
            )


class EpochTimingCallback(TimingCallback):
    def __init__(self, timer: NamedTimer):
        # NOTE: don't call super().__init__() to reuse timer
        self.timer = timer

    def _on_epoch_start(self, name):
        self._on_batch_start(name)

    def _on_epoch_end(self, name, pl_module):
        self.timer.stop(name)
        pl_module.log(
            name, self.timer[name], on_step=False, on_epoch=True, batch_size=1
        )

    def on_validation_epoch_start(self, trainer, pl_module):
        self._on_batch_start("validation_epoch_timing")

    def on_validation_epoch_end(self, trainer, pl_module):
        self._on_epoch_end("validation_epoch_timing", pl_module)

    def on_train_epoch_start(self, trainer, pl_module):
        self._on_batch_start("train_epoch_timing")

    def on_train_epoch_end(self, trainer, pl_module):
        self._on_epoch_end("train_epoch_timing", pl_module)


def print_once(*args, **kwargs):
    if torch.distributed.get_rank():
        return
    print(*args, **kwargs)


class DeltaTimingCallback(Callback):
    def __init__(self):
        # TODO: Fix this. CustomCallback should call this.
        # self.t0 = None. Why not self.t0 = time.time() on validation
        self.t0 = 0
        self.total_train_step_time = [0, 0]  # total_time, num of iterations
        self.total_valid_step_time = [0, 0]

    def trigger_tick(self):
        self.t0 = time.time()

    def on_train_start(self, trainer, pl_module):
        self.t0 = time.time()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        t1 = time.time()
        d = t1 - self.t0
        self.total_train_step_time[0] += d
        self.total_train_step_time[1] += 1
        self.t0 = t1

        pl_module.log(
            "train_step_time",
            torch.as_tensor(d),
            on_step=True,
            on_epoch=False,
            batch_size=1,
        )

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        t1 = time.time()
        d = t1 - self.t0
        self.total_valid_step_time[0] += d
        self.total_valid_step_time[1] += 1
        self.t0 = t1

        pl_module.log(
            "validation_step_time",
            torch.as_tensor(d),
            on_step=True,
            on_epoch=False,
            batch_size=1,
        )

    def on_train_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        print_once(
            f"Average train_step_time {self.total_train_step_time[0]/self.total_train_step_time[1]}"
        )


class CustomMegatronGPTModel(MegatronGPTModel):
    # setup_training_data, on_train_start and setup_data_mmap are here to move dataloader initialization past RUN_START tag
    # PTL has iter(dl) call in it's fit_loop.setup_data method, which spawns 2 processes that immediately prefetch data
    def setup_training_data(self, cfg):
        if not self.cfg.data.delay_data_init:
            return super().setup_training_data(self.cfg.data)

        if hasattr(self, "_train_ds"):
            consumed_samples = self.compute_consumed_samples(0)
            # Assign a dummy dataloader with MockGPTDataset for self._train_dl to run PTL's setup_data() method so that the actual data is not prefetched
            # during the iter() call in setup_data().
            self._train_dl = self.build_pretraining_data_loader(
                MockGPTDataset(
                    cfg,
                    self.tokenizer,
                    "train",
                    num_samples=self.cfg.global_batch_size * self.trainer.max_steps,
                    seq_length=cfg.seq_length,
                    seed=1,
                ),
                consumed_samples,
            )

    def on_train_start(self) -> None:
        # Call on_train_start of MegatronBaseModel
        super().on_train_start()
        if not self.cfg.data.delay_data_init:
            return

        # Build the actual dataloader with self._train_ds
        consumed_samples = self.compute_consumed_samples(0)
        self._train_dl = self.build_pretraining_data_loader(
            self._train_ds, consumed_samples
        )
        # Setyp MMap before fit_loop initializes dataloader but after the run_start tag
        self.setup_data_mmap()
        # Reset fit_loop._combined_loader to None.
        self.trainer.fit_loop._combined_loader = None
        # Redo setup_data from PTL's fit_loop.py
        self.trainer.fit_loop.setup_data()

    def setup_data_mmap(self):
        if self.cfg.data.get("delay_data_mmap", False) and not isinstance(
            self._train_ds, MockGPTDataset
        ):
            if self._train_ds:
                self._train_ds.create_data_mmap()
            if self._validation_ds:
                self._validation_ds.create_data_mmap()
            if self._test_ds:
                self._test_ds.create_data_mmap()

    def get_synthetic_input_training(self):
        if isinstance(self._train_ds, MockGPTDataset):
            single_data = self._train_ds[0]
        else:
            text = (
                torch.ones(self.cfg.data.seq_length + 1, dtype=torch.int64) * 3545
            )  # some token
            text[-1] = 0

            tokens = text[:-1].contiguous()
            labels = text[1:].contiguous()

            train_ds = self._train_ds.datasets[0]

            if self.cfg.data.legacy_dataset:
                arg_list = [
                    tokens,
                    train_ds.eos_id,
                    train_ds.reset_position_ids,
                    train_ds.reset_attention_mask,
                    train_ds.eod_mask_loss,
                    False,
                ]
            else:
                arg_list = [
                    tokens,
                    train_ds.config.tokenizer.eod,
                    train_ds.config.reset_position_ids,
                    train_ds.config.reset_attention_mask,
                    train_ds.config.eod_mask_loss,
                    False,
                ]
            attention_mask, loss_mask, position_ids = _get_ltor_masks_and_position_ids(
                *arg_list
            )

            single_data = {
                "tokens": tokens,
                "labels": labels,
                "loss_mask": loss_mask,
                "position_ids": position_ids,
            }
        if isinstance(self._train_dl.batch_sampler, BaseMegatronBatchSampler):
            batch = default_collate(
                [single_data] * self.cfg.micro_batch_size * get_num_microbatches()
            )
        elif isinstance(self._train_dl.batch_sampler, MegatronPretrainingSampler):
            batch = default_collate([single_data] * self.cfg.micro_batch_size)
            batch = repeat(batch)
        else:
            raise NotImplementedError(
                f'No synthetic data implementation for data sampler "{self._train_dl.batch_sampler}"'
            )
        return batch

    def get_synthetic_input_validation(self):
        seq_length = self.cfg.data.seq_length
        text = torch.ones(seq_length + 1, dtype=torch.int64) * 3545  # some token
        text[-1] = 0

        tokens = text[:-1].contiguous()
        labels = text[1:].contiguous()

        loss_mask = torch.ones(seq_length, dtype=torch.float32)
        loss_mask[-1] = 0.0

        position_ids = torch.tensor([i for i in range(seq_length)], dtype=torch.int64)

        single_data = {
            "tokens": tokens,
            "labels": labels,
            "loss_mask": loss_mask,
            "position_ids": position_ids,
        }
        if isinstance(self._validation_dl.batch_sampler, BaseMegatronBatchSampler):
            batch = default_collate(
                [single_data] * self.cfg.micro_batch_size * get_num_microbatches()
            )
        elif isinstance(self._validation_dl.batch_sampler, MegatronPretrainingSampler):
            batch = default_collate([single_data] * self.cfg.micro_batch_size)
            batch = repeat(batch)
        else:
            raise NotImplementedError(
                f'No synthetic data implementation for data sampler "{self._validation_dl.batch_sampler}"'
            )
        return batch

    def _register_sharded_tensor_state_dict_hooks_if_available(self) -> None:
        logger.info(
            "Overriding _register_sharded_tensor_state_dict_hooks_if_available"
            " to mitigate incompatibility of PTL and PyTorch"
        )
        return

    def _extract_consumed_samples_from_ckpt(self, ckpt_path):
        consumed_samples = super()._extract_consumed_samples_from_ckpt(ckpt_path)
        if consumed_samples == 0 and self.cfg.custom.override_zero_consumed_samples:
            consumed_samples = (
                self.cfg.custom.init_global_step * self.cfg.global_batch_size
            )
            logger.info(f"Overriding consumed_samples from 0 to {consumed_samples}")
        return consumed_samples

    def set_training(self, value):
        self.training = value


def configure_pre_validation_training_loop(trainer: pl.Trainer) -> None:
    if type(trainer.fit_loop.epoch_loop) != TrainingEpochLoop and not isinstance(
        trainer.fit_loop.epoch_loop, SkipResumeTrainingValidationLoop
    ):
        return
    loop = PreValidationTrainingValidationLoop(trainer.min_steps, trainer.max_steps)
    loop.trainer = trainer
    trainer.fit_loop.epoch_loop = loop


class PreValidationTrainingValidationLoop(TrainingEpochLoop):
    """
    Extend the PTL Epoch loop to run validating on start.
    """

    def __init__(self, min_steps: Optional[int] = None, max_steps: int = -1) -> None:
        super().__init__(min_steps, max_steps)
        self.restarting = True

    def _should_check_val_fx(self) -> bool:
        if self.restarting and self.global_step == 0:
            return True
        return super()._should_check_val_fx()


def setup_auxiliary_loggers(log_marker="AUX"):
    """Sets up non-NeMo loggers. Must be called after NeMo logging is set up.

    - Adds formatting to all logs
    - Removes INFO handlers on non-zero-ranks
    """

    class CustomFormatter(BaseNeMoFormatter):
        DEFAULT_FORMAT = BaseNeMoFormatter.DEFAULT_FORMAT.replace("NeMo", log_marker)

    class CustomDebugFormatter(DebugNeMoFormatter):
        DEFAULT_FORMAT = DebugNeMoFormatter.DEFAULT_FORMAT.replace("NeMo", log_marker)

    root = logging.getLogger()
    if not root.handlers:
        logger.warning(f"Failed to setup auxiliary loggers. Empty root logger handlers")
        return

    root_handler = root.handlers[0]
    if not isinstance(root_handler, logging.StreamHandler):
        logger.warning(
            f"Failed to setup auxiliary loggers. Unexpected root logger handler: {root.handlers[0]}"
        )
        return

    if get_envbool(NEMO_ENV_VARNAME_TESTING, False):
        root_handler.setFormatter(CustomDebugFormatter())
        root.setLevel(logging.DEBUG)
    elif is_global_rank_zero():
        root_handler.setFormatter(CustomFormatter())
    else:
        # removing INFO handlers for non-zero ranks
        root.handlers.clear()


class PrintArtifacts(pl.Callback):
    """
    Callback to print training artificats.

    Printing the config outside a config might not be accurate because there are config transformations
    being done during .setup() or .connect() and on_fit_start() as well sometimes.
    For example, Strategy.connect() is called automatically by lightning API during trainer.fit()

    Moreover overriding on_train_batch_start() is the best possible place as the model is going to be training
    in the very next step.
    """

    def __init__(self, config_dir_prefix):
        self.run_once = True
        self.config_dir = f"{config_dir_prefix}_config"
        logging.info(f"Logging runtime model config to {self.config_dir}")

    @rank_zero_only
    def on_train_batch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if self.run_once:
            logging.info("\n\n### Model Runtime Config ### \n")
            logging.info(f"{pprint(pl_module.config,width=80)}")

            trainer_context = TrainerContext.from_trainer(trainer)
            trainer_context.io_dump(self.config_dir, yaml_attrs=["model", "trainer"])
            logging.info(
                f"Config logged at {self.config_dir}/model.yaml and {self.config_dir}/trainer.yaml"
            )

            self.run_once = False

class MemoryProfileCallback(pl.Callback):
    def __init__(
        self,
        file_prefix="memdump", 
        max_entries=1000000,
        rank_0_only=True,

        start_location="init",
        end_location="train_start",
        force_oom_before_stop=False,
    ):
        self.file_prefix = file_prefix
        self.max_entries = max_entries
        self.force_oom_before_stop = force_oom_before_stop

        # process group not initialized at this part, using this method to get the global rank and world size
        global_rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        # customize this list if other ranks need profiling
        profile_ranks = [0] if rank_0_only else list(range(world_size))
        self.do_profile = global_rank in profile_ranks

        self.start_location = start_location
        self.end_location = end_location

        self.maybe_start_here("init")

    def force_oom(self):
        # adds a bunch of large tensors here to trigger CUDA OOM
        print("[Memory Profiler] Forcing OOM")
        mems = []
        size = int(1e9)
        for idx in range(10):
            mem = torch.arange(start=size*idx, end=size*(idx+1), device=torch.device("cuda:0"), dtype=torch.int64) 
            mems.append(mem)
            total_mem_use = sum([m.nelement() * m.element_size() / 1000**3 for m in mems])
            print(f"[Memory Profiler] Total mem use: {total_mem_use} GB")
        print("[Memory Profiler]", sum([x[-5:].to(torch.device("cpu")) for x in mems]).tolist())

    def maybe_start_here(self, current_location):
        if self.do_profile and self.start_location == current_location:
            torch.cuda.memory._record_memory_history(max_entries = self.max_entries)

    def maybe_stop_here(self, current_location):
        if self.do_profile and self.end_location == current_location:
            # first trigger OOM before saving
            if self.force_oom_before_stop:
                self.force_oom()

            # saves the profiles
            filename = f"/mem_dump/{self.file_prefix}.pickle"
            torch.cuda.memory._dump_snapshot(filename)
            torch.cuda.memory._record_memory_history(enabled=None)

    def cleanup_after_oom(self):
        import traceback
        memdump_filename = f"/mem_dump/{self.file_prefix}_failed.pickle"
        traceback_filename = f"/mem_dump/{self.file_prefix}_failed.traceback"
        with open(traceback_filename, "w") as f:
            traceback.print_exc(file=f)
        torch.cuda.memory._dump_snapshot(memdump_filename)
        torch.cuda.memory._record_memory_history(enabled=None)
            
    def on_train_start(self, trainer, pl_module):
        self.maybe_start_here("train_start")
        self.maybe_stop_here("train_start")
