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

# set this to WARNING or DEBUG to see debug info or warnings
TRAIN_LOG_LEVEL = str(os.getenv("TRAIN_LOG_LEVEL", "ERROR"))

import logging as py_logging
LOG_LEVEL=py_logging._nameToLevel[TRAIN_LOG_LEVEL]

if LOG_LEVEL > py_logging.WARNING:
    import warnings
    warnings.filterwarnings('ignore')

# disable warning and info messages from dependency modules
py_logging.getLogger('megatron.core.dist_checkpointing.strategies.zarr').setLevel(LOG_LEVEL)
import torch
torch._logging.set_logs(all=LOG_LEVEL)
from nemo.utils import logging
logging.set_verbosity(verbosity_level=LOG_LEVEL)

import torch.multiprocessing as mp
from callbacks.custom_callbacks import CustomCallback, CustomMegatronGPTSFTModel, MetricsLogger
from mlperf_common.frameworks.pyt import PyTCommunicationHandler
from mlperf_common.logging import MLLoggerWrapper
from nemo.collections.nlp.parts.megatron_trainer_builder import MegatronLMPPTrainerBuilder
from nemo.collections.nlp.parts.peft_config import PEFT_CONFIG_MAP
from nemo.core.config import hydra_runner
from omegaconf.omegaconf import OmegaConf
from pytorch_lightning.utilities import rank_zero_only
import gc


def string_to_bool(text):
    if text is None or text.lower() == "false":
        return False
    elif text.lower() == "true":
        return True
    raise ValueError("The string must be 'true' or 'false', case insensitive.")


mp.set_start_method("spawn", force=True)

OmegaConf.register_new_resolver("add", lambda x, y: x + y)
OmegaConf.register_new_resolver("floor_div", lambda x, y: x // y)

load_llama_ckpt = string_to_bool(os.getenv("LOAD_CKPT", "True"))


def create_model(cfg, trainer):
    peft_cfg_cls = PEFT_CONFIG_MAP[cfg.model.peft.peft_scheme]
    path = cfg.model.restore_from_path if load_llama_ckpt else "/workspace/code/conf/base"
    model_cfg = CustomMegatronGPTSFTModel.merge_cfg_with(path, cfg)
    if load_llama_ckpt:
        model = CustomMegatronGPTSFTModel.restore_from(cfg.model.restore_from_path, model_cfg, trainer=trainer)
    else:
        model = CustomMegatronGPTSFTModel(model_cfg, trainer)
    model.add_adapter(peft_cfg_cls(model_cfg))
    return model


@rank_zero_only
def print_model_data_types(model):
    """
    Traverse through all components of a PyTorch model and print their data types.
    
    Args:
        model (nn.Module): The PyTorch model to traverse.
    """
    print("Model Data Types:")
    print("-" * 30)
    for name, param in model.named_parameters():
        print(f"Parameter: {name} - Data type: {param.dtype}")
    print("-" * 30)
    for name, buffer in model.named_buffers():
        print(f"Buffer: {name} - Data type: {buffer.dtype}")
    print("-" * 30)
    print(f"DEBUG train.py model created \n ********** {model} \n ********")


@hydra_runner(config_path="conf", config_name="megatron_gpt_peft_tuning_config")
def main(cfg) -> None:
    OmegaConf.resolve(cfg)
    logging.debug("\n\n************** Experiment configuration ***********")
    logging.debug(f"\n{OmegaConf.to_yaml(cfg)}")
    mllogger = MLLoggerWrapper(PyTCommunicationHandler())

    callbacks_list = []
    if int(os.environ.get("ENABLE_MEMORY_PROFILING", 0)) == 1:
        if os.getenv("PROFILER") == 'rpd':
            raise ValueError("Memory profiling can't be used with RPD Profiling")
        from callbacks.memory_callbacks import MemoryProfilerCallback
        mem_prof_callbacks = MemoryProfilerCallback()
        callbacks_list.append(mem_prof_callbacks)
    
    custom_callback = CustomCallback(cfg, mllogger)
    callbacks_list.append(custom_callback)

    precision = cfg.trainer.precision
    trainer = MegatronLMPPTrainerBuilder(cfg).create_trainer(callbacks=callbacks_list)
    cfg.trainer.precision = precision
    
    if cfg['exp_manager']['create_wandb_logger']:
        from nemo.utils.exp_manager import exp_manager
        exp_manager(trainer, cfg.get("exp_manager", None))
        model = create_model(cfg, trainer)
        trainer.loggers[0].watch(model, log_graph=False)
    else:
        trainer.loggers.append(MetricsLogger(cfg, mllogger, trainer))
        model = create_model(cfg, trainer)
    
    gc.disable()
    trainer.fit(model)


if __name__ == "__main__":
    main()
