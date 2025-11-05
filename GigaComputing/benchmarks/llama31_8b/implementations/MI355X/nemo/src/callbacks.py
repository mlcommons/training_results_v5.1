# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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


### MLLogger
from mlperf_logging import mllog
from mlperf_logging.mllog import constants
import torch.distributed as dist
import os
from prof_handler import get_profiler, TOTAL_WARMUP_STEPS, TOTAL_ACTIVE_STEPS
import time

def is_dist_avail_and_initialized():
    return (dist.is_available() and dist.is_initialized())

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def barrier():
    if not is_dist_avail_and_initialized():
        return
    
    dist.barrier()

class MLLogger:
    def __init__(self, filepath="/results/mlperf_llama31_8b.out", default_stack_offset=2):
        self.logger = mllog.get_mllogger()
        mllog.config(default_stack_offset=default_stack_offset, filename=filepath)

    def start(self, **kwargs):
        if get_rank() == 0:
            self.logger.start(**kwargs)

    def end(self, **kwargs):
        if get_rank() == 0:
            self.logger.end(**kwargs)

    def event(self, **kwargs):
        if get_rank() == 0:
            self.logger.event(**kwargs)

mllogger = MLLogger()

### Preemptive checkpoint callbacks
import lightning.pytorch as pl
from nemo.utils import logging

class PreemptiveStop(pl.Callback):
    """Preemptively stop training at a given global step. Allows stopping training before reaching
    the max steps. Useful for testing checkpoint save and resume.

    Args:
        stop_on_step (int): Stop training when trainer.global_step reaches this value.
            Checked at the start of every step.
    """

    def __init__(self, stop_on_step: int):
        self.stop_on_step = stop_on_step

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx
    ) -> None:
        if trainer.global_step >= self.stop_on_step:
            logging.info(f"Global step {trainer.global_step} >= {self.stop_on_step}, signaling Trainer to stop.")
            trainer.should_stop = True
            # skip EarlyStopping validation unless val_check_interval met
            if trainer.global_step % trainer.val_check_interval != 0:
                trainer.limit_val_batches = 0


### Metrics Logger
from pytorch_lightning.loggers import Logger
from pytorch_lightning.utilities import rank_zero_only

class MetricsLogger(Logger):
    def __init__(
        self, 
        init_global_step, global_batch_size, seq_length,
        target_log_ppl, 
        train_loss_key = "reduced_train_loss",
        val_loss_key = "val_loss", 
        train_step_time_in_s = "train_step_timing in s",
        train_step_time_atol=7200,
    ):
        super().__init__()

        self.init_global_step = init_global_step
        self.gbs = global_batch_size
        self.seq_len = seq_length

        self.target = target_log_ppl
        self.train_loss_key = train_loss_key
        self.val_loss_key = val_loss_key
        self.is_target_reached = False

        self.train_step_time_in_s = train_step_time_in_s
        self.train_step_time_atol = train_step_time_atol

    def log_metrics(self, metrics, step):
        if self.val_loss_key in metrics:
            self.log_validation_loss(metrics, step)

        if self.train_step_time_in_s in metrics:
            step_time = metrics[self.train_step_time_in_s]
            assert step_time <= self.train_step_time_atol, f"Logged train step time ({step_time}) is slower than tolerable ({self.train_step_time_atol}). "

    def log_validation_loss(self, metrics, step):
        consumed_samples = step * self.gbs

        loss = metrics[self.val_loss_key]

        # Output both EPOCH_NUM and SAMPLES_COUNT until this PR is merged:
        # https://github.com/mlcommons/logging/pull/434
        # After merge we can remove EPOCH_NUM
        mllogger.event(
            key=constants.EVAL_ACCURACY,
            value=loss,
            metadata={
                constants.EPOCH_NUM: consumed_samples,
                constants.SAMPLES_COUNT: consumed_samples
            }
        )

        if not self.is_target_reached and loss <= self.target:
            self.is_target_reached = True

    @rank_zero_only
    def log_hyperparams(self, params, *args, **kwargs):
        pass

    @property
    def name(self):
        return 'mlperf-metrics'

    @property
    def version(self):
        return 1

### MLPerf callbacks
def compute_consumed_mllog_samples(trainer, init_global_step, global_batch_size, seq_length):
    consumed_samples = (
        trainer.global_step * global_batch_size
    )
    return int(consumed_samples) # we log the epoch numbers in sequences, not tokens

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

class MLPerfCallback(pl.Callback):
    def __init__(
        self, 
        global_batch_size, 
        micro_batch_size,
        sequence_length,
        init_global_step,
        eval_every,
        configs={}
    ):
        mllogger.event(key=constants.CACHE_CLEAR, value=True)
        mllogger.start(key=constants.INIT_START)
        super().__init__()

        self.init_global_step = init_global_step
        self.gbs = global_batch_size
        self.mbs = micro_batch_size
        self.seq_len = sequence_length
        self.eval_every = eval_every

        self.is_target_reached = False
        self.status = constants.ABORTED
        self.configs = configs
        
        self.torchprof = None
        self.rpd = None
        self.rpd_status = None
        self.profiler = os.getenv("PROFILER", '')
        
        if self.profiler == 'pytorch':
            self.torchprof = get_profiler('torchprof')
            self.torchprof.start()
        if self.profiler == 'rpd':
            logging.info("Using RPD profiler")
            from rpdTracerControl import rpdTracerControl
            rpdTracerControl.setFilename(name=f"trace.rpd", append=True)
            self.rpd = rpdTracerControl()
        self.timer = Timer(self.gbs)
        self.log_freq = int(os.environ.get("TRAINING_LOSS_LOG_FREQ", 1000000))

    def consumed_samples(self, trainer):
        return compute_consumed_mllog_samples(trainer, self.init_global_step, self.gbs, self.seq_len)

    def set_success_status(self):
        self.status = constants.SUCCESS
        self.is_target_reached = True
        
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self.timer.start()
        
        if self.profiler:
            logging.info(f"RPD settings: global_step: {trainer.global_step}, TOTAL_WARMUP_STEPS: {TOTAL_WARMUP_STEPS}, TOTAL_ACTIVE_STEPS: {TOTAL_ACTIVE_STEPS}")
        
        if self.rpd and trainer.global_step >= TOTAL_WARMUP_STEPS and not self.rpd_status:
            logging.info("Starting RPD profiler")
            self.rpd.start()
            self.rpd_status = "running"
        
        if self.torchprof: 
            self.torchprof.step()            

        if trainer.global_step > TOTAL_WARMUP_STEPS + TOTAL_ACTIVE_STEPS:
            if self.torchprof:
                self.torchprof.stop()        
            if self.rpd and self.rpd_status == "running":
                logging.info("Stopping RPD profiler")
                self.rpd.stop()
                self.rpd_status = "finished"
        
        return super().on_train_batch_start(trainer, pl_module, batch, batch_idx)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.timer.stop()

        if trainer.global_step % self.log_freq == 0:
            mllogger.event(
                key="train_loss",
                value=outputs["loss"].item(),
                metadata={
                    constants.SAMPLES_COUNT: trainer.global_step
                    * self.gbs,
                    "lr": trainer.optimizers[0].param_groups[0]["lr"],
                },
        )

        return super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)

    @rank_zero_only
    def on_train_epoch_start(self, trainer, pl_module):
        mllogger.start(key=constants.EPOCH_START, metadata={constants.SAMPLES_COUNT: self.consumed_samples(trainer)})
        mllogger.start(key=constants.BLOCK_START, metadata={constants.SAMPLES_COUNT: self.consumed_samples(trainer)})
                
        return super().on_train_epoch_start(trainer, pl_module)
    
    @rank_zero_only
    def on_train_epoch_end(self, trainer, pl_module):
        mllogger.end(key=constants.EPOCH_STOP, metadata={constants.SAMPLES_COUNT: self.consumed_samples(trainer)})
        return super().on_train_epoch_end(trainer, pl_module)

    def on_train_end(self, trainer, pl_module):
        # for every occurrences, run on all ranks to allow sync
        barrier()
        if self.rpd and self.rpd_status == "running":
            self.rpd.stop()
            self.rpd_status = "finished"
        run_duration_ms = time.time() - self.train_start
        mllogger.end(key=constants.RUN_STOP,
            metadata={
                "samples_count": trainer.global_step * self.gbs,
                "status": "success",
                "duration": f"{run_duration_ms} sec -> {run_duration_ms / 60.0} minutes"
            },)
        mllogger.event(key="train_samples", value=self.consumed_samples(trainer))
        return super().on_train_end(trainer, pl_module)
    
    @rank_zero_only
    def log_eval_start(self, trainer, pl_module):
        mllogger.end(key=constants.BLOCK_STOP, metadata={constants.SAMPLES_COUNT: self.consumed_samples(trainer)})
        mllogger.start(key=constants.EVAL_START, metadata={constants.SAMPLES_COUNT: self.consumed_samples(trainer)})

    def on_validation_start(self, trainer, pl_module):
        self.log_throughput(trainer.global_step)
        trainer.val_check_interval = self.eval_every
        trainer.val_check_batch = self.eval_every
        self.log_eval_start(trainer, pl_module)
        return super().on_validation_start(trainer, pl_module)

    def on_validation_end(self, trainer, pl_module):
        mllogger.end(key=constants.EVAL_STOP, metadata={constants.SAMPLES_COUNT: self.consumed_samples(trainer)})

        for logger in trainer.loggers:
            if isinstance(logger, MetricsLogger):
                if logger.is_target_reached:
                    trainer.should_stop = True
                    self.set_success_status()

        if not trainer.should_stop:
            mllogger.start(key=constants.BLOCK_START, metadata={constants.SAMPLES_COUNT: self.consumed_samples(trainer)})

        return super().on_validation_end(trainer, pl_module)

    @rank_zero_only
    def load_state_dict(self, state_dict):
        print(f":::MLLOG Weight initialization: {state_dict.keys()}")
        return super().load_state_dict(state_dict)
    
    def on_train_start(self, trainer, pl_module):
        # run on all ranks to allow synchronization
        barrier()

        for key, value in self.configs.items():
            mllogger.event(key=key, value=value)

        mllogger.end(key=constants.INIT_STOP)
        mllogger.start(key=constants.RUN_START)
        self.train_start = time.time()

        
    def log_throughput(self, global_step):
        throughput = self.timer.get_throughput()
        mllogger.event(
            key="tracked_stats",
            metadata={"step": global_step * self.gbs},
            value={"throughput": throughput},
        )
