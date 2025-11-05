import argparse
import math
import logging
import warnings
warnings.filterwarnings('ignore')

import torch.distributed as dist
from nemo.collections.llm.gpt.data import build_pretraining_datamodule
from nemo.utils import logging as nemo_logging
from train import get_data_module

logging.getLogger('multistorageclient.config').setLevel(logging.ERROR)
nemo_logging.set_verbosity(logging.ERROR)

def build_data_index_standalone(data_module, max_steps, val_check_interval, limit_val_batches, limit_test_batches):
    """Build data index without nemo-run orchestration"""
    data_module.num_dataset_builder_threads = 64

    build_pretraining_datamodule(
        datamodule=data_module,
        trainer_max_steps=max_steps,
        trainer_val_check_interval=val_check_interval,
        trainer_limit_val_batches=limit_val_batches,
        trainer_limit_test_batches=limit_test_batches,
    )


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Llama3.1 Pretraining data preprocess")
    data_group = parser.add_argument_group("Dataset arguments")
    data_group.add_argument("--gbs", type=int, default=1152, help="Global batch size, should be divisible by PP")
    data_group.add_argument("--mbs", type=int, default=1, help="Micro batch size")
    data_group.add_argument("--eval_every", type=int, default=12288, help="Evaluate at least every N training sequences")
    data_group.add_argument("--start_eval_at", type=int, default=0, help="Start evaluation at N training sequences")
    data_group.add_argument("--eval_tokens", type=int, default=1024, help="Evaluate using at least N evaluation sequences")
    data_group.add_argument('--max_steps', type=int, default=1200000, help="Maximum number of steps that each experiment partition will train on. None means no restriction on max steps. ")
    data_group.add_argument("--use_full_dataset", action="store_true", help="If set, then we use the full dataset, instead of the last 256/1024 shards")
    data_group.add_argument("--tokenizer_path", type=str, default="/model", help="Tokenizer path that's used to tokenize the dataset")

    experiment_group = parser.add_argument_group("Experiment management arguments")
    experiment_group.add_argument("--seed", type=int, default=1234, help="random seed")

    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()

    temp_data = get_data_module(
        gbs=args.gbs,
        mbs=args.mbs,
        tokenizer_path=args.tokenizer_path,
        seed=args.seed,
        use_full_dataset=args.use_full_dataset,
        max_steps=args.max_steps,
    )

    eval_every_n_batches = math.ceil(args.eval_every / args.gbs)
    eval_batches = math.ceil(args.eval_tokens / args.gbs)
    if args.start_eval_at == 0:
        start_eval_at = math.ceil(args.start_eval_at / args.gbs)
    else:
        start_eval_at = eval_every_n_batches

    build_data_index_standalone(
        temp_data,
        args.max_steps,
        eval_every_n_batches,
        eval_batches,
        eval_batches
    )

    dist.destroy_process_group()
