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

import logging as base_logging
base_logging.getLogger("pytorch_lightning.utilities.rank_zero").setLevel(base_logging.ERROR)
base_logging.getLogger("nemo.lightning.pytorch.strategies.megatron_strategy").setLevel(base_logging.ERROR)
base_logging.getLogger("lightning.pytorch.accelerators.cuda").setLevel(base_logging.ERROR)
base_logging.getLogger("lightning.pytorch.trainer.connectors.signal_connector").setLevel(base_logging.ERROR)
base_logging.getLogger("lightning.fabric.utilities.distributed").setLevel(base_logging.ERROR)
base_logging.getLogger("torch_tensorrt.dynamo.conversion.converter_utils").setLevel(base_logging.ERROR)

from nemo.utils import logging
logging.setLevel(logging.ERROR)

import warnings
warnings.filterwarnings("ignore")

import atexit
import os
from dataclasses import make_dataclass

import torch
from omegaconf import DictConfig
from omegaconf.omegaconf import OmegaConf

torch.cuda.set_device(int(os.getenv("SLURM_LOCALID", "0")))

import gc

import hydra
import torch
from custom_callbacks import MetricsLogger
from custom_llama import CustomLlamaModel
from nemo.lightning.pytorch.strategies.utils import _destroy_dist_connection
from lightning.pytorch import seed_everything
from megatron.core import parallel_state
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig
from mlperf_common.callbacks import mllogger
from nemo import lightning as nl
from nemo.collections import llm
from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer
from nemo.collections.llm.api import _setup
from nemo.collections.llm.gpt.data.packed_sequence import PackedSequenceSpecs
from nemo.collections.llm.peft.lora import LoRA
from nemo.lightning import AutoResume
from nemo.lightning.pytorch.callbacks.megatron_comm_overlap import (
    MegatronCommOverlapCallback,
)
from nemo.lightning.pytorch.optim import CosineAnnealingScheduler
from nemo.lightning.pytorch.strategies.utils import RestoreConfig
from omegaconf import DictConfig
from omegaconf.omegaconf import OmegaConf


def init_mp_state(cfg, cp):
    parallel_state.destroy_model_parallel()
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=cfg.model.tensor_model_parallel_size,
        context_parallel_size=cp,
        pipeline_model_parallel_size=cfg.model.pipeline_model_parallel_size,
    )


def get_rank():
    return int(os.getenv("SLURM_PROCID", 0))


def prepare_dataset(
    cfg: DictConfig = None,
    tokenizer: AutoTokenizer = None,
) -> llm.FineTuningDataModule:
    data_root = cfg.data_root
    if not os.path.exists(data_root):
        raise FileNotFoundError(f"Data root directory not found: {data_root}")
    if tokenizer is None:
        raise ValueError("Tokenizer must be provided")

    train_path = f"{data_root}/train.npy"
    val_path = f"{data_root}/validation.npy"

    if not os.path.exists(train_path) or not os.path.exists(val_path):
        raise FileNotFoundError(f"Training or validation data not found in {data_root}")

    packed_sequence_specs = PackedSequenceSpecs(
        packed_sequence_size=cfg.model.encoder_seq_length,
        packed_train_data_path=f"{data_root}/train.npy",
        packed_val_data_path=f"{data_root}/validation.npy",
    )

    def transform_dataloader(self, dataloader, consumed_samples=0):
        from nemo.lightning.data import add_megatron_sampler

        mode = getattr(dataloader, "mode", "train")
        if cfg.model.eval_cp and mode == "validation":
            init_mp_state(cfg, cfg.model.eval_cp)
        data_parallel_rank = parallel_state.get_data_parallel_rank()
        data_parallel_size = parallel_state.get_data_parallel_world_size()
        if cfg.model.eval_cp and mode == "validation":
            init_mp_state(cfg, cfg.model.context_parallel_size)
        if mode == "validation" and cfg.model.val_micro_batch_size:
            micro_batch_size = cfg.model.val_micro_batch_size
            global_batch_size = cfg.model.val_global_batch_size
        else:
            micro_batch_size = cfg.model.micro_batch_size
            global_batch_size = cfg.model.global_batch_size

        return add_megatron_sampler(
            dataloader,
            micro_batch_size=micro_batch_size,
            global_batch_size=global_batch_size,
            rampup_batch_size=self.rampup_batch_size,
            consumed_samples=self.init_consumed_samples if mode == "train" else 0,
            dataloader_type=self.dataloader_type,
            drop_last=mode not in ["validation", "test", "predict"],
            dataloader_mode=mode,
            rank=data_parallel_rank,
            world_size=data_parallel_size,
        )

    nl.MegatronDataSampler.transform_dataloader = transform_dataloader

    def none_pack_metadata(self):
        return None

    llm.FineTuningDataModule.pack_metadata = property(none_pack_metadata)

    return llm.FineTuningDataModule(
        dataset_root=data_root,
        tokenizer=tokenizer,
        seq_length=cfg.model.encoder_seq_length,
        micro_batch_size=cfg.model.micro_batch_size,
        global_batch_size=cfg.model.global_batch_size,
        persistent_workers=True,
        seed=cfg.model.seed,
        packed_sequence_specs=packed_sequence_specs,
        dataset_kwargs={
            "return_cu_seqlen": False,
        },
    )


def prepare_optimizer(
    max_steps: int = 1024,
    lr: float = 4e-4,
    warmup_steps: int = 0,
    use_distributed_optimizer: bool = False,
    overlap_param_gather_with_optimizer_step: bool = False,
) -> nl.MegatronOptimizerModule:
    if get_rank() == 0:
        mllogger.event(key=mllogger.constants.OPT_BASE_LR, value=lr)
        mllogger.event(key=mllogger.constants.OPT_ADAMW_WEIGHT_DECAY, value=0.0001)
        mllogger.event(key=mllogger.constants.OPT_GRADIENT_CLIP_NORM, value=0.3)

    optimizer_config = OptimizerConfig(
        # General
        optimizer="adam",
        lr=lr,
        clip_grad=0.3,
        weight_decay=0.0001,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_eps=1e-08,
        # Precision
        bf16=True,
        params_dtype=torch.bfloat16,
        # Distributed
        use_distributed_optimizer=use_distributed_optimizer,
        overlap_param_gather_with_optimizer_step=overlap_param_gather_with_optimizer_step,
    )
    scheduler = CosineAnnealingScheduler(
        max_steps=max_steps,
        warmup_steps=warmup_steps,
        constant_steps=0,
        min_lr=0,
    )
    if get_rank() == 0:
        mllogger.event(
            key=mllogger.constants.OPT_LR_WARMUP_FACTOR,
            value=warmup_steps / max_steps,
        )
    return nl.MegatronOptimizerModule(config=optimizer_config, lr_scheduler=scheduler)


def prepare_model(tokenizer: AutoTokenizer, cfg: DictConfig):
    if get_rank() == 0:
        mllogger.event(key="lora_rank", value=16)
        mllogger.event(key="lora_alpha", value=32)

    peft = LoRA(
        dim=16,
        alpha=32,
        dropout=0.1,
        a2a_experimental=True,
        dropout_position="pre",
        lora_A_init_method="kaiming",
        target_modules=["linear_proj", "linear_qkv"],
    )

    llama2_config = llm.Llama2Config70B(
        num_layers=cfg.model.num_layers,
        seq_length=cfg.model.encoder_seq_length,
        tp_comm_overlap_disable_qkv=True,
        cp_comm_type=cfg.model.cp_comm_type,
        fp8_dot_product_attention=cfg.model.fp8_dot_product_attention,
        cross_entropy_loss_fusion=False,
        activation_func_fp8_input_store=cfg.model.activation_func_fp8_input_store,
        # Temporarily disable gradient_accumulation_fusion as MCore optimizer doesn't support this
        gradient_accumulation_fusion=False if cfg.model.external_cuda_graph else True,
        bias_dropout_fusion=True,
        disable_parameter_transpose_cache=False,
        external_cuda_graph=cfg.model.external_cuda_graph,
        enable_cuda_graph=cfg.model.enable_cuda_graph,
        cpu_offloading=cfg.model.cpu_offloading,
        cpu_offloading_num_layers=cfg.model.cpu_offloading_num_layers,
        cpu_offloading_activations=cfg.model.cpu_offloading_activations,
        cpu_offloading_weights=cfg.model.cpu_offloading_weights,
    )
    llama2_config.cp_eval = cfg.model.eval_cp
    model = CustomLlamaModel(llama2_config, tokenizer=tokenizer)
    resume = None
    if cfg.load_ckpt:
        restore_config = RestoreConfig(
            path=cfg.ckpt_root,
            adapter_path=None,
            load_model_state=True,
            load_optim_state=False,
            load_artifacts=False,
        )
        resume = AutoResume(restore_config=restore_config)

    return peft, model, resume


def prepare_training_strategy(
    cfg: DictConfig
) -> tuple[nl.MegatronStrategy, nl.MegatronMixedPrecision, MegatronCommOverlapCallback]:
    def validation_step_patch(self, dataloader_iter, *args, **kwargs):
        with self.precision_plugin.val_step_context():
            out = self.model.validation_step(dataloader_iter, *args, **kwargs)
            self.lightning_module.log("val_loss_sum", out[0], reduce_fx="sum")
            self.lightning_module.log("val_loss_count", out[1], reduce_fx="sum")
            return out

    nl.MegatronStrategy.validation_step = validation_step_patch

    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=cfg.model.tensor_model_parallel_size,
        pipeline_model_parallel_size=cfg.model.pipeline_model_parallel_size,
        context_parallel_size=cfg.model.context_parallel_size,
        sequence_parallel=cfg.model.sequence_parallel,
        pipeline_dtype=torch.bfloat16,
        ckpt_load_directly_on_device=False,
        ckpt_parallel_load=False,
        ckpt_load_strictness="log_all",
        gradient_as_bucket_view=True,
        use_te_rng_tracker=cfg.model.use_te_rng_tracker,
        ddp=DistributedDataParallelConfig(
            overlap_grad_reduce=cfg.ddp.overlap_grad_reduce,
            overlap_param_gather=cfg.ddp.overlap_param_gather,
            fp8_param_gather=cfg.ddp.fp8_param_gather,
            average_in_collective=cfg.ddp.average_in_collective,
            use_distributed_optimizer=cfg.optim.use_distributed_optimizer,
        ),
    )

    precision = nl.MegatronMixedPrecision(
        precision="bf16-mixed",
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
        autocast_enabled=True,
        grad_reduce_in_fp32=False,
        fp8="hybrid",
        fp8_amax_history_len=cfg.model.fp8_amax_history_len,
        fp8_amax_compute_algo=cfg.model.fp8_amax_compute_algo,
        fp8_params=cfg.model.fp8_params,
        fp8_dot_product_attention=cfg.model.fp8_dot_product_attention,
    )

    tp_comm_overlap_cfg = None
    if cfg.model.ub_tp_comm_overlap:
        tp_comm_overlap_cfg = OmegaConf.to_container(cfg.model.ub_tp_comm_overlap_cfg)
        TPCommOverlapConfig = make_dataclass(
            "TPCommOverlapConfig",
            [(k, type(v)) for k, v in tp_comm_overlap_cfg.items()],
        )
        tp_comm_overlap_cfg = TPCommOverlapConfig(**tp_comm_overlap_cfg)

    overlap_callback = MegatronCommOverlapCallback(
        tp_comm_overlap=cfg.model.ub_tp_comm_overlap,
        tp_comm_overlap_cfg=tp_comm_overlap_cfg,
        overlap_grad_reduce=cfg.ddp.overlap_grad_reduce,
        overlap_param_gather=cfg.ddp.overlap_param_gather,
        overlap_param_gather_with_optimizer_step=cfg.optim.overlap_param_gather_with_optimizer_step,
    )

    return strategy, precision, overlap_callback


OmegaConf.register_new_resolver("add", lambda x, y: x + y)
OmegaConf.register_new_resolver("floor_div", lambda x, y: x // y)
OmegaConf.register_new_resolver("if", lambda x, y, z: y if x else z)
OmegaConf.register_new_resolver("floor", lambda x: int(x // 1))


@hydra.main(
    version_base=None, config_path="conf", config_name="megatron_gpt_peft_tuning_config"
)
def main(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)
    assert (
        cfg.model.eval_cp == 1 or cfg.model.eval_cp is None
    ), "model.eval_cp must be set to 1 or left unset"

    if get_rank() == 0:
        mllogger.start(key=mllogger.constants.INIT_START)
        mllogger.mlperf_submission_log(
            benchmark="llama2_70b_lora", num_nodes=cfg.trainer.num_nodes
        )

    tokenizer = AutoTokenizer("/ckpt/context/nemo_tokenizer")
    data = prepare_dataset(cfg, tokenizer)

    optimizer = prepare_optimizer(
        max_steps=cfg.trainer.max_steps,
        warmup_steps=cfg.optim.sched.warmup_steps,
        lr=cfg.optim.lr,
        use_distributed_optimizer=cfg.optim.use_distributed_optimizer,
        overlap_param_gather_with_optimizer_step=cfg.optim.overlap_param_gather_with_optimizer_step,
    )

    peft, model, resume = prepare_model(tokenizer=tokenizer, cfg=cfg)
    strategy, precision, overlap_callback = prepare_training_strategy(cfg)

    logger = MetricsLogger(cfg, model)
    custom_callback = logger.callback

    if get_rank() == 0:
        mllogger.event(
            key=mllogger.constants.OPT_LR_TRAINING_STEPS,
            value=cfg.trainer.max_steps,
        )

    trainer = nl.Trainer(
        max_steps=cfg.trainer.max_steps,
        limit_val_batches=cfg.trainer.limit_val_batches,
        val_check_interval=cfg.trainer.val_check_interval,
        devices=cfg.trainer.devices,
        num_nodes=cfg.trainer.num_nodes,
        accelerator="gpu",
        strategy=strategy,
        plugins=precision,
        num_sanity_val_steps=0,
        accumulate_grad_batches=1,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        use_distributed_sampler=False,
        log_every_n_steps=0,
        callbacks=[overlap_callback],
        logger=logger,
    )
    logger.set_trainer(trainer)
    logger.log_hyperparams()
    model.custom_callback = custom_callback

    app_state = _setup(
        model=model,
        data=data,
        trainer=trainer,
        log=None,
        resume=resume,
        optim=optimizer,
        tokenizer=tokenizer,
        model_transform=peft,
    )
    trainer.callbacks.append(custom_callback)
    if os.environ.get("DEBUGGING_CALLBACK", "False").lower() in ("true", "1", "t"):
        from mlperf_common.callbacks import StatsLogCallback
        trainer.callbacks.append(StatsLogCallback())
    if get_rank() == 0:
        mllogger.event(key=mllogger.constants.SEED, value=cfg.model.seed, sync=False)
    seed_everything(cfg.model.seed, workers=True, verbose=False)
    gc.disable()
    trainer.fit(model, data)
    atexit.unregister(_destroy_dist_connection)


if __name__ == "__main__":
    main()
