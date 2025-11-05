import os 
import math
import argparse
import logging
from typing import Optional
import warnings
warnings.filterwarnings('ignore')

import torch
from nemo.collections.common.tokenizers import AutoTokenizer
from nemo import lightning as nl
from nemo.collections.llm.gpt.data import PreTrainingDataModule
from nemo.collections.llm.gpt.model import GPTModel, Llama31Config8B
from nemo.lightning.pytorch.optim import CosineAnnealingScheduler, MegatronOptimizerModule
from nemo.utils import logging as nemo_logging
from megatron.core.optimizer import OptimizerConfig
from megatron.core.distributed import DistributedDataParallelConfig
from callbacks import PreemptiveStop, MLPerfCallback, MetricsLogger

loggers = [
    'multistorageclient.config',
    'megatron.core.rerun_state_machine',
    'lightning.pytorch.callbacks.model_summary',
    'lightning.pytorch.accelerators.cuda',
]

for logger in loggers:
    logging.getLogger(logger).setLevel(logging.ERROR)

nemo_logging.set_verbosity(logging.ERROR)

SEQ_LENGTH = 8192


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Llama3.1 Pretraining (No NeMo-Run)")
    parser.add_argument("--tag", type=str, help="Optional experiment tag", required=False, default="")

    model_group = parser.add_argument_group("Model arguments")
    model_group.add_argument("--size", type=str, default="8b", help="Choose the model to be trained", choices=["8b"])
    model_group.add_argument("--nodes", type=int, required=True, help="Number of nodes to be used")
    model_group.add_argument("--gpus_per_node", type=int, required=True, help="Number of GPUs per node")
    model_group.add_argument("--initial_ckpt_path", type=str, default=None)
    model_group.add_argument("--use_ckpt", action="store_true", help="If set, then resume from the initial checkpoint path")
    model_group.add_argument("--resume_from_hf", action="store_true", help="Setting this knob indicates that we are resuming from a weight-only checkpoint")
    model_group.add_argument("--ckpt_start_step", type=int, default=0, help="Sets this value to how many steps the resumed checkpoint is already trained on")
    model_group.add_argument("--continual_ckpt_path", type=str, default=None, help="Sets this to the path that saves the checkpoint")
    model_group.add_argument("--save_ckpt", action="store_true", help="If set, then we save the checkpoint at the end of the experiment")
    model_group.add_argument("--tensor_parallel_size", type=int, default=1, help="Set tensor parallelism to the model")
    model_group.add_argument("--pipeline_parallel_size", type=int, default=1, help="Set pipeline parallelism to the model")
    model_group.add_argument("--context_parallel_size", type=int, default=1, help="Set context parallelism to the model")
    model_group.add_argument("--fp8_params", action="store_true", help="Load model parameters in FP8")

    data_group = parser.add_argument_group("Dataset arguments")
    data_group.add_argument("--gbs", type=int, default=1152, help="Global batch size, should be divisible by PP")
    data_group.add_argument("--mbs", type=int, default=1, help="Micro batch size")
    data_group.add_argument("--max_lr", type=float, default=1e-4, help="Peak learning rate. Min LR will be 0.1 of max_lr")
    data_group.add_argument("--eval_every", type=int, default=12288, help="Evaluate at least every N training sequences")
    data_group.add_argument("--start_eval_at", type=int, default=0, help="Start evaluation at N training sequences")
    data_group.add_argument("--eval_tokens", type=int, default=1024, help="Evaluate using at least N evaluation sequences")
    data_group.add_argument('--max_steps', type=int, default=1200000, help="Maximum number of steps that each experiment partition will train on. None means no restriction on max steps. ")
    data_group.add_argument('--warmup_steps', type=int, default=None, help="Number of steps for LR warmup")
    data_group.add_argument("--use_full_dataset", action="store_true", help="If set, then we use the full dataset, instead of the last 256/1024 shards")
    data_group.add_argument("--tokenizer_path", type=str, default="/model", help="Tokenizer path that's used to tokenize the dataset")

    experiment_group = parser.add_argument_group("Experiment management arguments")
    experiment_group.add_argument("--seed", type=int, default=1234, help="random seed")
    experiment_group.add_argument("--target_log_ppl", type=float, default=3.3)
    experiment_group.add_argument("--step_time_atol", type=int, default=1600, help="train step time atol")

    return parser


def get_optimizer(warmup_steps: int, max_lr: float = 1e-4):
    opt_cfg = OptimizerConfig(
        optimizer="adam",
        lr=max_lr,
        weight_decay=0.1,
        bf16=True,
        fp16=False,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_eps=1e-5,
        use_distributed_optimizer=True,
        clip_grad=1.0
    )

    min_lr =  (0.1 * max_lr)
    sched = CosineAnnealingScheduler(
        warmup_steps=warmup_steps,
        constant_steps=0,
        min_lr=min_lr,
    )

    optimizer = MegatronOptimizerModule(
        config=opt_cfg,
        lr_scheduler=sched,
    )

    return optimizer


def get_strategy(
        tensor_parallel_size: Optional[int] = None,
        pipeline_parallel_size: Optional[int] = None,
        context_parallel_size: Optional[int] = None,
):
    from megatron.core.dist_checkpointing.validation import StrictHandling
    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=tensor_parallel_size,
        pipeline_model_parallel_size=pipeline_parallel_size,
        virtual_pipeline_model_parallel_size=None,
        context_parallel_size=context_parallel_size,
        gradient_as_bucket_view=True,
        ckpt_async_save=False,
        ckpt_load_strictness=StrictHandling.LOG_ALL,
        ckpt_parallel_load=True,
        ddp=DistributedDataParallelConfig(
            check_for_nan_in_grad=True,
            grad_reduce_in_fp32=False,
            average_in_collective=True,
            use_distributed_optimizer=True,
            fp8_param_gather=True,
        ),
    )

    return strategy


def get_pretrain_config(
    nnodes: int,
    ngpus_per_node: int,
    max_steps: int,
    warmup_steps: int,
    max_lr: float = 1e-4,
    eval_every: Optional[int] = None,
    eval_batches: Optional[int] = None,
    fp8_params: bool = True,
    tensor_parallel_size: Optional[int] = None,
    pipeline_parallel_size: Optional[int] = None,
    context_parallel_size: Optional[int] = None,
) -> tuple:

    # Model Config
    model_config = Llama31Config8B()
    model_config.seq_length = SEQ_LENGTH
    model_config.fp8 = "hybrid"
    model_config.fp8_margin = 0
    model_config.fp8_amax_history_len = 4
    model_config.fp8_amax_compute_algo = 'most_recent'
    model_config.tp_only_amax_red = True

    precision = nl.MegatronMixedPrecision(
        precision="bf16-mixed",
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
        autocast_enabled=True,
        grad_reduce_in_fp32=False,
        fp8="hybrid",
        fp8_margin=0,
        fp8_amax_history_len=4,
        fp8_amax_compute_algo='most_recent',
        fp8_params=fp8_params,
        fp8_dot_product_attention=False,
    )

    # Optimizer & Strategy
    optimizer = get_optimizer(warmup_steps, max_lr)
    strategy = get_strategy(tensor_parallel_size,
                            pipeline_parallel_size,
                            context_parallel_size)

    # Trainer Config
    trainer_config = {
        'accelerator': 'gpu',
        'devices': ngpus_per_node,
        'num_nodes': nnodes,
        'max_steps': max_steps,
        'val_check_interval': eval_every if eval_every else None,
        'limit_val_batches': eval_batches,
        'limit_test_batches': eval_batches,
        'num_sanity_val_steps': 0,  # Override for MLPerf
        'strategy': strategy,
        'plugins': precision,
        'use_distributed_sampler': False,
        'enable_checkpointing': False,  # Will be overridden later if needed
        'logger': False,  # Disable default logger, we'll add our own
        'enable_model_summary': True,
        'enable_progress_bar': False,
    }

    return model_config, optimizer, trainer_config


def get_data_module(
    gbs: int = 288,
    mbs: int = 4,
    tokenizer_path: Optional[str] = "",
    seed: Optional[int] = 1234,
    use_full_dataset: Optional[bool] = False,
    max_steps: Optional[int] = None,
) -> PreTrainingDataModule:

    tokenizer = AutoTokenizer(pretrained_model_name=tokenizer_path)

    dataset_path = "/data/"

    if use_full_dataset:
        train_datasets = sum([["12.5", f"{dataset_path}/c4-train.en_{idx}_text_document"] for idx in range(8)], [])
    else:
        train_datasets = sum([["10", f"{dataset_path}/c4-train.en_{idx}_text_document"] for idx in [6]], [])

    data_paths = {
        "train": train_datasets,
        "validation": [
            f"{dataset_path}/c4-validation-91205-samples.en_text_document"
        ],
        "test": [
            f"{dataset_path}/c4-validation-91205-samples.en_text_document"
        ],
    }

    data_module = PreTrainingDataModule(
        tokenizer=tokenizer,
        paths=data_paths,
        num_workers=0,
        seq_length=SEQ_LENGTH,
        global_batch_size=gbs,
        micro_batch_size=mbs,
        index_mapping_dir="/npy_indices",
        seed=seed,
        # Option to reset the position IDs in the dataset at an interval.
        reset_position_ids=False,
        # Option to reset the attention mask from the dataset.
        reset_attention_mask=False,
        # Option to enable the EOD mask loss.
        eod_mask_loss=False,
        # Rampup batch size, should be in format of [start_global_batch_size, batch_size_increment, ramup_samples].
        rampup_batch_size=None,
        mmap_bin_files=False,
    )

    # Set num_train_samples if max_steps is provided
    if max_steps:
        data_module.num_train_samples = max_steps * gbs

    return data_module


def run_training(
    model_config,
    data_module,
    optim_config,
    trainer_config,
    callbacks,
    resume_config=None,
    checkpoint_config=None,
    extra_loggers=None
):

    trainer = nl.Trainer(**trainer_config)

    if callbacks:
        trainer.callbacks.extend(callbacks)

    # TODO: look into fixing loggers
    if extra_loggers:
        if len(extra_loggers) == 1:
            trainer.logger = extra_loggers[0]
        else:
            from lightning.pytorch.loggers import LoggerCollection
            trainer.logger = LoggerCollection(extra_loggers)

    if checkpoint_config:
        from lightning.pytorch.callbacks import ModelCheckpoint
        checkpoint_callback = ModelCheckpoint(**checkpoint_config)
        trainer.callbacks.append(checkpoint_callback)

    # Setup resume
    if resume_config:
        # Handle resume logic here
        pass  # TODO: Implement resume logic

    model = GPTModel(config=model_config, optim=optim_config, tokenizer=data_module.tokenizer)
    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    args = get_parser().parse_args()
    if args.tag and not args.tag.startswith("-"):
        args.tag = "-" + args.tag

    eval_every_n_batches = math.ceil(args.eval_every / args.gbs)
    eval_batches = math.ceil(args.eval_tokens / args.gbs)
    if args.start_eval_at == 0:
        start_eval_at = math.ceil(args.start_eval_at / args.gbs)
    else:
        start_eval_at = eval_every_n_batches

    # Collect all HP configs for MLPerf
    from mlperf_logging.mllog import constants
    tp = args.tensor_parallel_size
    pp = args.pipeline_parallel_size
    cp = args.context_parallel_size
    dp = (args.nodes * args.gpus_per_node) // (tp * pp * cp)
    mini_batch_size = (args.gbs // dp)
    grad_accumulation_steps = mini_batch_size // args.mbs

    # Create fresh data module for each experiment
    data_module = get_data_module(
        gbs=args.gbs,
        mbs=args.mbs,
        tokenizer_path=args.tokenizer_path,
        seed=args.seed,
        use_full_dataset=args.use_full_dataset,
        max_steps=args.max_steps,
    )

    # Get training configuration
    model_config, optim_config, trainer_config = get_pretrain_config(
        max_lr=args.max_lr,
        nnodes=args.nodes,
        ngpus_per_node=args.gpus_per_node,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        eval_every=eval_every_n_batches,
        eval_batches=eval_batches,
        fp8_params=args.fp8_params,
        tensor_parallel_size=args.tensor_parallel_size,
        pipeline_parallel_size=args.pipeline_parallel_size,
        context_parallel_size=args.context_parallel_size,
    )

    configs = {
        # HPs
        constants.GLOBAL_BATCH_SIZE: args.gbs,
        constants.GRADIENT_ACCUMULATION_STEPS: grad_accumulation_steps,
        constants.MAX_SEQUENCE_LENGTH: SEQ_LENGTH,
        constants.EVAL_SAMPLES: args.eval_tokens,
        constants.SEED: args.seed,
        constants.INIT_CHECKPOINT_STEP: args.ckpt_start_step,
        constants.OPT_NAME: "adamw",
        constants.OPT_BASE_LR: optim_config.config.lr,
        constants.OPT_ADAMW_BETA_1: optim_config.config.adam_beta1,
        constants.OPT_ADAMW_BETA_2: optim_config.config.adam_beta2,
        constants.OPT_ADAMW_EPSILON: optim_config.config.adam_eps,
        constants.OPT_ADAMW_WEIGHT_DECAY: optim_config.config.weight_decay,
        constants.OPT_GRADIENT_CLIP_NORM: optim_config.config.clip_grad,
        constants.OPT_END_LR: optim_config.lr_scheduler.min_lr,
        constants.OPT_LR_WARMUP_STEPS: optim_config.lr_scheduler.warmup_steps,
        constants.OPT_LR_DECAY_STEPS: args.max_steps - optim_config.lr_scheduler.warmup_steps,
        constants.OPT_LR_DECAY_SCHEDULE: "cosine with linear warmup",
        constants.SUBMISSION_BENCHMARK: "llama31_8b",
        constants.SUBMISSION_DIVISION: constants.CLOSED,
        constants.SUBMISSION_STATUS: constants.ONPREM,
        constants.SUBMISSION_ORG: os.getenv('MLPERF_SUBMISSION_ORG', 'AMD'),
        constants.SUBMISSION_PLATFORM: os.getenv('MLPERF_SUBMISSION_PLATFORM', 'MI355X'),
        # TODO: update to constants.MAX_STEPS after mlperf_logging 5.1.0 was released
        'max_steps': args.max_steps,
    }

    # Setup callbacks
    callbacks = []
    experiment_max_steps = args.ckpt_start_step + args.max_steps

    callbacks.extend([
        PreemptiveStop(stop_on_step=experiment_max_steps),
        MLPerfCallback(
            global_batch_size=args.gbs,
            micro_batch_size=args.mbs,
            sequence_length=SEQ_LENGTH,
            eval_every=eval_every_n_batches,
            init_global_step=args.ckpt_start_step,
            configs=configs,
        ),
    ])

    # TODO: revisit loggers
    extra_loggers = [
        MetricsLogger(
            init_global_step=args.ckpt_start_step,
            global_batch_size=args.gbs,
            seq_length=SEQ_LENGTH,
            target_log_ppl=args.target_log_ppl,
            train_step_time_atol=args.step_time_atol,
        ),
    ]

    # Setup checkpointing
    checkpoint_config = None
    if args.save_ckpt:
        checkpoint_name = f"checkpoint-seed-{args.seed}-par-{experiment_max_steps}-steps"
        checkpoint_dir = os.path.join(args.continual_ckpt_path, checkpoint_name)
        checkpoint_config = {
            'dirpath': checkpoint_dir,
            'filename': 'checkpoint',
            'every_n_train_steps': experiment_max_steps,
            'save_top_k': 0,
            'save_last': False,
            'save_weights_only': False,
            'monitor': "consumed_samples",
            'mode': "max",
        }
        trainer_config['enable_checkpointing'] = True
    else:
        trainer_config['enable_checkpointing'] = False

    # Setup resume
    resume_config = None
    if args.use_ckpt and args.initial_ckpt_path:
        resume_config = {
            'resume_from_path': args.initial_ckpt_path,
            'resume_from_hf': args.resume_from_hf,
        }

    # Run training
    run_training(
        model_config=model_config,
        data_module=data_module,
        optim_config=optim_config,
        trainer_config=trainer_config,
        callbacks=callbacks,
        resume_config=resume_config,
        checkpoint_config=checkpoint_config,
        extra_loggers=extra_loggers
    )
