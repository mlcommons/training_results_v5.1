import logging
from nemo.utils import logging as nemo_logging
import builtins
import warnings


class RankZeroFilter(logging.Filter):
    def __init__(self, rank):
        self.rank = rank

    def filter(self, record):
        return self.rank == 0


def get_rank_zero_logger(name, rank):
    logger = logging.getLogger(name)
    logger.addFilter(RankZeroFilter(rank))
    return logger


def disable_print():
    def do_nothing(*args, **kwargs):
        pass

    builtins.print = do_nothing


def setup_logging():
    nemo_logging.setLevel(logging.ERROR)
    logging.getLogger("lightning").setLevel(logging.ERROR)
    logging.getLogger().setLevel(logging.ERROR)

    warnings.filterwarnings("ignore")
    warnings.filterwarnings(
        "ignore", message=".*Could not find the bitsandbytes CUDA binary.*"
    )
