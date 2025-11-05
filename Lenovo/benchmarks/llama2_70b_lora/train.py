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
import logging
from math import floor, ceil

logging.getLogger("nemo.utils.import_utils").disabled = True
logging.getLogger("megatron.core.utils").disabled = True

# To log only on rank-0
rank = int(os.getenv("SLURM_PROCID", 0))


class RankZeroFilter(logging.Filter):
    def filter(self, record):
        return rank == 0


root = logging.getLogger()
root.addFilter(RankZeroFilter())

import warnings

warnings.filterwarnings("ignore")

from dataclasses import make_dataclass

import torch
from omegaconf import DictConfig
from omegaconf.omegaconf import OmegaConf

torch.cuda.set_device(int(os.getenv("SLURM_LOCALID", "0")))


import gc

# PATCH MCORE LOAD TO AVOID OOM. This must happen before nemo imports.
import megatron.core.dist_checkpointing
from megatron.core.dist_checkpointing.dict_utils import extract_matching_values, merge
from megatron.core.dist_checkpointing.mapping import ShardedObject, apply_factory_merges
from megatron.core.dist_checkpointing.serialization import determine_global_metadata, extract_sharded_base, load_preprocess, load_sharded_metadata, parse_strict_flag, validate_integrity_and_strict_load, validate_sharded_objects_handling, verify_checkpoint_and_load_strategy
from megatron.core.dist_checkpointing.validation import StrictHandling
original_load = megatron.core.dist_checkpointing.serialization.load

def new_load(
    sharded_state_dict,
    checkpoint_dir,
    sharded_strategy=None,
    common_strategy=None,
    validate_access_integrity=True,
    strict=StrictHandling.ASSUME_OK_UNEXPECTED,
):
   
    sharded_strategy, common_strategy = verify_checkpoint_and_load_strategy(
        checkpoint_dir, sharded_strategy, common_strategy
    )

    # Dequantize all FP8 tensors in the state dict into their corresponding high-precision tensors.
    # Retaining FP8 tensors in the state dict can cause issues in the following two cases:
    #   1. Sometimes, when the precision of the checkpoint is higher than that of the model params,
    #      we want to directly use the state dict to initialize the main params. If the FP8 tensors
    #      in this sharded state dict are not converted to high-precision tensors, the loaded
    #      tensors will already be quantized, which defeats the purpose of initializing the main
    #      params with a high-precision state dict;
    #   2. When using delayed scaling, this loading process writes an extra value into the global
    #      amax_history buffer of Transformer Engine, which is undesirable.
    
    # This causes OOM. commenting it out.
    #force_all_tensors_to_non_fp8(sharded_state_dict)

    common_state_dict = common_strategy.load_common(checkpoint_dir)

    sharded_state_dict, nonpersistent_state_dict, sh_ten_factories = load_preprocess(
        sharded_state_dict
    )
    merge(common_state_dict, nonpersistent_state_dict)

    # At this point we are only dealing with ShardedBase objects
    sharded_state_dict, _ = extract_sharded_base(sharded_state_dict)

    # Validation
    ckpt_sharded_metadata = None
    local_metadata, global_metadata = None, None
    strict = parse_strict_flag(strict)
    if StrictHandling.requires_explicit_ckpt_mismatch_check(strict):
        ckpt_sharded_metadata = load_sharded_metadata(
            checkpoint_dir, sharded_strategy, common_strategy  # type: ignore[arg-type]
        )
    if validate_access_integrity or StrictHandling.requires_global_app_metadata(strict):
        local_metadata, global_metadata = determine_global_metadata(sharded_state_dict)

    sharded_state_dict, missing_keys, unexpected_keys = validate_integrity_and_strict_load(
        sharded_state_dict,
        strict,
        validate_access_integrity,
        local_metadata,
        global_metadata,
        ckpt_sharded_metadata,
    )

    # ShardedBase loading
    if not sharded_strategy.can_handle_sharded_objects:
        validate_sharded_objects_handling(sharded_strategy, common_strategy)
        sharded_objects_state_dict, sharded_state_dict = extract_matching_values(
            sharded_state_dict, lambda v: isinstance(v, ShardedObject)
        )
        sharded_objects = common_strategy.load_sharded_objects(
            sharded_objects_state_dict, checkpoint_dir
        )
        merge(common_state_dict, sharded_objects)

    loaded_state_dict = sharded_strategy.load(sharded_state_dict, checkpoint_dir)

    merge(common_state_dict, loaded_state_dict)

    loaded_state_dict = apply_factory_merges(common_state_dict, sh_ten_factories)

    if StrictHandling.requires_returning_mismatch_keys(strict):
        return common_state_dict, missing_keys, unexpected_keys
    else:
        return common_state_dict

megatron.core.dist_checkpointing.load = new_load

import hydra
import torch
from custom_callbacks import MetricsLogger
from custom_llama import CustomLlamaModel
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


from nemo.collections.llm.fn.base import _map_module


def _map(
    module,
    func,
    leaf_only: bool = False,
    **kwargs,
):
    return _map_module(module, func, leaf_only=leaf_only, **kwargs)


# Replace the original map function
import nemo.collections.llm.fn.base

nemo.collections.llm.fn.base.map = _map


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
        num_workers=cfg.dataloader.num_workers,
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
        dropout_recompute=cfg.model.dropout_recompute,
    )

    llama2_config = llm.Llama2Config70B(
        num_layers=cfg.model.num_layers,
        seq_length=cfg.model.encoder_seq_length,
        tp_comm_overlap_disable_qkv=True,
        cp_comm_type=cfg.model.cp_comm_type,
        fp8_dot_product_attention=cfg.model.fp8_dot_product_attention,
        cross_entropy_loss_fusion=False,
        activation_func_fp8_input_store=cfg.model.activation_func_fp8_input_store,
        gradient_accumulation_fusion=False,
        bias_dropout_fusion=True,
        disable_parameter_transpose_cache=False,
        external_cuda_graph=cfg.model.external_cuda_graph,
        enable_cuda_graph=cfg.model.enable_cuda_graph,
        cuda_graph_scope=cfg.model.cuda_graph_scope,
        cpu_offloading=cfg.model.cpu_offloading,
        cpu_offloading_num_layers=cfg.model.cpu_offloading_num_layers,
        cpu_offloading_activations=cfg.model.cpu_offloading_activations,
        cpu_offloading_weights=cfg.model.cpu_offloading_weights,
        cpu_offloading_double_buffering=cfg.model.cpu_offloading_double_buffering,
        recompute_granularity=cfg.model.recompute_granularity,
        recompute_method=cfg.model.recompute_method,
        recompute_num_layers=cfg.model.recompute_num_layers,
        distribute_saved_activations=cfg.model.distribute_saved_activations,
        recompute_modules=cfg.model.recompute_modules,
        use_transformer_engine_op_fuser=cfg.model.use_transformer_engine_op_fuser,
        fused_single_qkv_rope=cfg.model.fused_single_qkv_rope,
        fp4_param=cfg.model.fp4_param,
        fp4=cfg.model.fp4,
    )

    llama2_config.cp_eval = cfg.model.eval_cp
    model = CustomLlamaModel(llama2_config, tokenizer=tokenizer)
    resume = None
    if cfg.load_ckpt:
        restore_config = RestoreConfig(
            path=cfg.ckpt_root,
            load_model_state=True,
            load_optim_state=False,
            load_artifacts=False,
        )
        resume = AutoResume(restore_config=restore_config)

    return peft, model, resume


def prepare_training_strategy(
    cfg: DictConfig,
) -> tuple[nl.MegatronStrategy, nl.MegatronMixedPrecision, MegatronCommOverlapCallback]:
    def validation_step_patch(self, dataloader_iter, *args, **kwargs):
        with self.precision_plugin.val_step_context():
            out = self.model.validation_step(dataloader_iter, *args, **kwargs)
            self.lightning_module.log("val_loss_sum", out[0], reduce_fx="sum")
            self.lightning_module.log("val_loss_count", out[1], reduce_fx="sum")
            return out

    nl.MegatronStrategy.validation_step = validation_step_patch

    def teardown_patch(self):
        return

    nl.MegatronStrategy.teardown = teardown_patch
    data_parallel_sharding_strategy = (
        "optim_grads_params" if cfg.model.fsdp == "megatron" else "no_shard"
    )

    # Create cluster environment for the strategy
    # This fixed issues with interactively testing multi-node configs
    from lightning.pytorch.plugins.environments import SLURMEnvironment
    cluster_env = SLURMEnvironment() if "SLURM_PROCID" in os.environ else None

    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=cfg.model.tensor_model_parallel_size,
        pipeline_model_parallel_size=cfg.model.pipeline_model_parallel_size,
        context_parallel_size=cfg.model.context_parallel_size,
        sequence_parallel=cfg.model.sequence_parallel,
        pipeline_dtype=torch.bfloat16,
        cluster_environment=cluster_env,
        ckpt_load_directly_on_device=False,
        ckpt_parallel_load=False,
        ckpt_load_optimizer=False,
        ckpt_load_main_params=False,
        ckpt_load_strictness="log_all",
        gradient_as_bucket_view=True,
        use_te_rng_tracker=cfg.model.use_te_rng_tracker,
        fsdp=cfg.model.fsdp,
        use_sharp=cfg.model.use_sharp,
        ddp=DistributedDataParallelConfig(
            overlap_grad_reduce=cfg.ddp.overlap_grad_reduce,
            overlap_param_gather=cfg.ddp.overlap_param_gather,
            fp8_param_gather=cfg.ddp.fp8_param_gather,
            average_in_collective=cfg.ddp.average_in_collective,
            use_distributed_optimizer=cfg.optim.use_distributed_optimizer,
            use_custom_fsdp=cfg.model.fsdp == "megatron",
            data_parallel_sharding_strategy=data_parallel_sharding_strategy,
            nccl_ub=cfg.ddp.nccl_ub,
            fsdp_double_buffer=cfg.ddp.nccl_ub,
        ),
    )

    assert not (cfg.model.fp8 and cfg.model.fp4), "fp8 and fp4 cannot be enabled at the same time"

    fp8_type = "hybrid" if cfg.model.fp8 else None
    fp4_type = "e2m1" if cfg.model.fp4 else None

    precision = nl.MegatronMixedPrecision(
        precision="bf16-mixed",
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
        autocast_enabled=True,
        grad_reduce_in_fp32=False,
        first_last_layers_bf16=cfg.model.first_last_layers_bf16,
        num_layers_at_start_in_bf16=cfg.model.num_layers_at_start_in_bf16,
        num_layers_at_end_in_bf16=cfg.model.num_layers_at_end_in_bf16,
        # fp4
        fp4=fp4_type,
        fp4_recipe=cfg.model.fp4_recipe,
        # fp8
        fp8=fp8_type,
        fp8_recipe=cfg.model.fp8_recipe,
        fp8_amax_history_len=cfg.model.fp8_amax_history_len,
        fp8_amax_compute_algo=cfg.model.fp8_amax_compute_algo,
        fp8_param_gather=cfg.model.fp8_param_gather,
        fp8_dot_product_attention=cfg.model.fp8_dot_product_attention,
    )

    tp_comm_overlap_cfg = None
    overlap_callback = None
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
OmegaConf.register_new_resolver("ceil_div", lambda x, y: ceil(x / y))
OmegaConf.register_new_resolver("if", lambda x, y, z: y if x else z)
OmegaConf.register_new_resolver("floor", lambda x: floor(x))
OmegaConf.register_new_resolver("ceil", lambda x: ceil(x))


@hydra.main(
    version_base=None, config_path="conf", config_name="megatron_gpt_peft_tuning_config"
)
def main(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)
    assert cfg.model.eval_cp == 1 or cfg.model.eval_cp is None, (
        "model.eval_cp must be set to 1 or left unset"
    )

    if get_rank() == 0:
        mllogger.start(key=mllogger.constants.INIT_START)
        mllogger.mlperf_submission_log(
            benchmark="llama2_70b_lora", num_nodes=cfg.trainer.num_nodes
        )
        mllogger.event(key="target_accuracy", value=0.925)

    tokenizer = AutoTokenizer(f"{cfg.ckpt_root}/context/nemo_tokenizer")
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

    callbacks = []
    if overlap_callback:
        callbacks.append(overlap_callback)

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
        callbacks=callbacks,
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

    if fname := os.environ.get('STAT_CALLBACK_FNAME'):
        from mlperf_common.callbacks import StatsLogCallback
        trainer.callbacks.append(StatsLogCallback(save_path=fname))

    if get_rank() == 0:
        mllogger.event(key=mllogger.constants.SEED, value=cfg.model.seed, sync=False)
    seed_everything(cfg.model.seed, workers=True, verbose=False)
    gc.disable()
    trainer.fit(model, data)


if __name__ == "__main__":
    main()
