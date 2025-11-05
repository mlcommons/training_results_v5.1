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

import os
import math

import lightning_optimized  # noqa: F401
from custom_callbacks import MetricsLogger
import torch
from omegaconf import DictConfig
from omegaconf.omegaconf import OmegaConf
import gc
import hydra
from torch import nn
from hydra.utils import instantiate
from lightning.pytorch import seed_everything
from nemo.collections.llm.api import _setup
from mlperf_common.callbacks import mllogger
from nemo.lightning import MegatronStrategy, MegatronMixedPrecision
from megatron.core.optimizer import OptimizerConfig

from flux_logging import setup_logging, disable_print
from custom_flux import CustomMegatronFluxModel

# To log only on rank-0
rank = int(os.getenv("SLURM_PROCID", 0))
# ensure that each rank gets a different seed!
# This is crucial for training so that different noise is added across ranks.

setup_logging()
# set the device to the local rank
torch.cuda.set_device(int(os.getenv("SLURM_LOCALID", "0")))


def get_rank():
    return int(os.getenv("SLURM_PROCID", 0))


OmegaConf.register_new_resolver("add", lambda x, y: x + y)
OmegaConf.register_new_resolver("mul", lambda x, y: x * y)
OmegaConf.register_new_resolver("floor_div", lambda x, y: x // y)
OmegaConf.register_new_resolver("ceil_div", lambda x, y: math.ceil(x / y))
OmegaConf.register_new_resolver("if", lambda x, y, z: y if x else z)
OmegaConf.register_new_resolver("floor", lambda x: int(x // 1))


def get_torch_dtype(name: str):
    mapping = {
        "float32": torch.float32,
        "float": torch.float,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "int32": torch.int32,
        "int64": torch.int64,
    }
    return mapping[name]


def instantiate_strategy(cfg: DictConfig):
    cfg_copy = OmegaConf.to_container(cfg)
    if "pipeline_dtype" in cfg_copy:
        cfg_copy["pipeline_dtype"] = get_torch_dtype(cfg_copy["pipeline_dtype"])
    if "ddp" in cfg_copy:
        cfg_copy["ddp"] = instantiate(cfg_copy["ddp"])
    strategy = MegatronStrategy(**cfg_copy)

    def validation_step_patch(self, dataloader_iter, *args, **kwargs):
        """
        After self.model.validation_step, out has been accumulated across dp ranks and micro_batches.
        it consists of 2 8-element tensors, one for the loss sum and one for the loss count at that timestep.

        This is how we accumulate these values across batches, adapted from LoRA.
        Unfortunately lightning log does not support tensors, so we have to log them as individual values.
        These values will then be accessed by the logging callback to compute the weighted average.
        """
        with self.precision_plugin.val_step_context():
            val_loss_sum, val_loss_count = self.model.validation_step(
                dataloader_iter, *args, **kwargs
            )
            self.lightning_module.log("val_loss_sum", val_loss_sum, reduce_fx="sum")
            self.lightning_module.log("val_loss_count", val_loss_count, reduce_fx="sum")
            return val_loss_sum, val_loss_count

    MegatronStrategy.validation_step = validation_step_patch
    return strategy


def instantiate_plugins(cfg: DictConfig):
    cfg_copy = OmegaConf.to_container(cfg)
    if "params_dtype" in cfg_copy:
        cfg_copy["params_dtype"] = get_torch_dtype(cfg_copy["params_dtype"])
    if "pipeline_dtype" in cfg_copy:
        cfg_copy["pipeline_dtype"] = get_torch_dtype(cfg_copy["pipeline_dtype"])
    plugins = MegatronMixedPrecision(**cfg_copy)
    return plugins

def instantiate_model(cfg: DictConfig, seed: int):
    cfg_copy = OmegaConf.to_container(cfg)
    cfg_copy["flux_params"]["flux_config"]["init_method"] = nn.init.xavier_uniform_
    cfg_copy["flux_params"]["flux_config"]["output_layer_init_method"] = nn.init.xavier_uniform_
    cfg_copy["flux_params"] = instantiate(cfg_copy["flux_params"])
    model = CustomMegatronFluxModel(**cfg_copy, seed=seed)
    return model


def instantiate_optimizer_config(cfg: DictConfig):
    cfg_copy = OmegaConf.to_container(cfg)
    if "params_dtype" in cfg_copy:
        cfg_copy["params_dtype"] = get_torch_dtype(cfg_copy["params_dtype"])
    optimizer_config = OptimizerConfig(**cfg_copy)
    return optimizer_config


@hydra.main(version_base=None, config_path="conf", config_name="flux1_schnell")
def main(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)
    if cfg.disable_print:
        disable_print()

    seed_everything(cfg.seed, workers=True, verbose=False)
    torch._dynamo.config.cache_size_limit = 128
    torch._dynamo.config.recompile_limit = 64
    if get_rank() == 0:
        mllogger.start(key=mllogger.constants.INIT_START)
        mllogger.mlperf_submission_log(
            benchmark="flux1", num_nodes=cfg.trainer.num_nodes
        )
        mllogger.event(key="target_accuracy", value=cfg.target_accuracy)

    model = instantiate_model(cfg.model, seed=cfg.seed)
    data = instantiate(cfg.data)
    trainer_partial = instantiate(
        cfg.trainer,
        val_check_interval=math.ceil(
            cfg.trainer.val_check_interval / cfg.data.global_batch_size
        ),
    )
    # necessary to convert dtype string to torch.dtype
    strategy = instantiate_strategy(cfg.strategy)
    plugins = instantiate_plugins(cfg.plugins)
    optim_partial = instantiate(cfg.optim)
    optimizer_config = instantiate_optimizer_config(cfg.optim.config)
    optim = optim_partial(config=optimizer_config)
    logger = MetricsLogger(cfg, model)
    custom_callback = logger.callback
    trainer = trainer_partial(strategy=strategy, plugins=plugins, logger=logger)
    logger.set_trainer(trainer)
    logger.log_hyperparams()

    if get_rank() == 0:
        mllogger.event(
            key=mllogger.constants.OPT_LR_TRAINING_STEPS,
            value=cfg.trainer.max_steps,
        )

    app_state = _setup(
        model=model,
        data=data,
        trainer=trainer,
        log=None,
        resume=None,
        optim=optim,
        tokenizer=None,
        model_transform=None,
    )
    trainer.callbacks.append(custom_callback)

    if get_rank() == 0:
        mllogger.event(key=mllogger.constants.SEED, value=cfg.seed, sync=False)

    gc.disable()

    if fname := os.environ.get("STAT_CALLBACK_FNAME"):
        from mlperf_common.callbacks import StatsLogCallback

        trainer.callbacks.append(StatsLogCallback(save_path=fname))

    trainer.fit(model, data)


if __name__ == "__main__":
    main()
