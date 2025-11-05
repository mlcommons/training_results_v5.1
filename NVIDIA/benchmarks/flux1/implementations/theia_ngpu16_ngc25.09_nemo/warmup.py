import os

import torch

from flux_logging import get_rank_zero_logger
from megatron.core.num_microbatches_calculator import get_micro_batch_size

rank = int(os.getenv("SLURM_PROCID", 0))
logger = get_rank_zero_logger(__name__, rank)


def clear_gradients(trainer):
    for chunk in trainer.model:
        chunk.module.zero_grad_buffer()
        chunk.module.zero_grad()


def run_training_warmup(trainer, warmup_train_steps, warmup_validation_steps):
    trainer.warmup = True
    torch.distributed.barrier()
    logger.info("Starting training warmup for %d steps", warmup_train_steps)
    optimizer = trainer.strategy.optimizers[0]
    for group in optimizer.param_groups:
        group["betas_"] = group["betas"]
        group["bias_correction_"] = group["bias_correction"]
        group["betas"] = [1.0, 1.0]
        group["weight_decay_"] = group["weight_decay"]
        group["weight_decay"] = 0.0
        group["pre_mult_wd_"] = group["pre_mult_wd"]
        group["pre_mult_wd"] = 0.0
        group["bias_correction"] = False

    for _ in range(warmup_train_steps):
        torch.cuda.synchronize()
        torch.distributed.barrier()
        logger.info("Training warmup step %d", _)
        trainer.model.training_step(
            trainer.model.get_synthetic_batch(get_micro_batch_size())
        )
        optimizer.zero_grad()
        optimizer.step()
        torch.cuda.synchronize()
        clear_gradients(trainer)

    torch.cuda.synchronize()
    torch.distributed.barrier()
    logger.info("Finished training warmup")

    # Recover optimizer configs changed by warmup
    for group in optimizer.param_groups:
        group["betas"] = group["betas_"]
        group["bias_correction"] = group["bias_correction_"]
        group["weight_decay"] = group["weight_decay_"]
        group["pre_mult_wd"] = group["pre_mult_wd_"]
        del group["betas_"]
        del group["bias_correction_"]
        del group["weight_decay_"]
        del group["pre_mult_wd_"]
        if "step" in group:
           del group["step"]

    if warmup_validation_steps > 0:
        with torch.no_grad():
            trainer.fit_loop.epoch_loop.val_loop.on_run_start()
            trainer.training = False
            trainer.validating = True
            logger.info(
                "Starting validation warmup for %d steps", warmup_validation_steps
            )
            for _ in range(warmup_validation_steps):
                logger.info("Validation warmup step %d", _)
                torch.cuda.synchronize()
                torch.distributed.barrier()
                trainer.model.validation_step(
                    trainer.model.get_synthetic_batch(get_micro_batch_size())
                )
                torch.cuda.synchronize()
        trainer.fit_loop.epoch_loop.val_loop.on_run_end()
        trainer.training = True
        trainer.validating = False
        torch.distributed.barrier()
        logger.info("Finished validation warmup")

    clear_gradients(trainer)
    trainer.model.zero_grad()
    trainer._logger_connector.reset_results()
    trainer._logger_connector.reset_metrics()
    torch.cuda.synchronize()
    torch.distributed.barrier()
    trainer.warmup = False
    logger.info("Finished warmup")

def reset_fp8_state(model, reset_fp8_meta_tensors=False):
    """Sets `fp8_initialized` flag to False in every TE layer which will force reinitialization."""
    def reset_fp8(m):
        if hasattr(m, "fp8_initialized"):
            m.fp8_initialized = False
            if reset_fp8_meta_tensors:
                m.reset_fp8_meta_tensors()

    model.apply(reset_fp8)
