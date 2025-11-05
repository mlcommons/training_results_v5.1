"""
1. capture memory stats at specific lighnting callback internvals
2. Capture snapshot from setup to teardown
"""

import os
import time
import torch
import pandas as pd
from lightning.pytorch.callbacks import Callback


LOGDIR = os.getenv("LOGDIR", "/results")
ARTIFACTS_DIR = os.path.join(LOGDIR, "/artifacts")
MEMORY_LOGS_DIR = os.path.join(ARTIFACTS_DIR, "/memory_logs")

MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT = 100000


class MemoryProfilerCallback(Callback):
    def __init__(
        self,
        output_dir="/results/artifacts/memory_logs/",
        profile_interval=1,
        collect_detailed_stats=True,
    ):
        super().__init__()

        self.enabled = int(os.environ.get("ENABLE_MEMORY_PROFILING", 0)) == 1
        if not self.enabled:
            return

        print(f"Memory Logs will be saved to: {output_dir}")

        self.output_dir = output_dir
        self.profile_interval = profile_interval
        self.collect_detailed_stats = collect_detailed_stats
        self.snapshot_count = 0
        os.makedirs(output_dir, exist_ok=True)

        self.memory_data = []
        self.stats_data = []

    def bytes_to_mb(self, b):
        return b / (1024 * 1024)

    def bytes_to_gb(self, b):
        return b / (1024 * 1024 * 1024)

    def get_memory_usage(self):
        metrics = {
            "allocated": torch.cuda.memory_allocated(),
            "reserved": torch.cuda.memory_reserved(),
            "max_allocated": torch.cuda.max_memory_allocated(),
            "max_reserved": torch.cuda.max_memory_reserved(),
        }

        # basic stats
        metrics["allocated_MB"] = self.bytes_to_mb(metrics["allocated"])
        metrics["reserved_MB"] = self.bytes_to_mb(metrics["reserved"])
        metrics["max_allocated_MB"] = self.bytes_to_mb(metrics["max_allocated"])
        metrics["max_reserved_MB"] = self.bytes_to_mb(metrics["max_reserved"])

        metrics["allocated_GB"] = self.bytes_to_gb(metrics["allocated"])
        metrics["reserved_GB"] = self.bytes_to_gb(metrics["reserved"])
        metrics["max_allocated_GB"] = self.bytes_to_gb(metrics["max_allocated"])
        metrics["max_reserved_GB"] = self.bytes_to_gb(metrics["max_reserved"])

        # details stats
        if self.collect_detailed_stats:
            try:
                stats = torch.cuda.memory_stats()
                # requested, active, inactive @ current, there's lots here, can add more.
                memory_stats_bytes = {
                    "requested_bytes_current": stats.get(
                        "requested_bytes.all.current", 0
                    ),
                    "requested_bytes_allocated": stats.get(
                        "requested_bytes.all.allocated", 0
                    ),
                    "active_bytes_current": stats.get("active_bytes.all.current", 0),
                    "active_bytes_allocated": stats.get(
                        "active_bytes.all.allocated", 0
                    ),
                    "inactive_split_bytes_current": stats.get(
                        "inactive_split_bytes.all.current", 0
                    ),
                    "inactive_split_bytes_allocated": stats.get(
                        "inactive_split_bytes.all.allocated", 0
                    ),
                }
                metrics.update(memory_stats_bytes)

                for key, value in list(memory_stats_bytes.items()):
                    if isinstance(value, (int, float)) and "bytes" in key:
                        metrics[f"{key}_MB"] = self.bytes_to_mb(value)
            except Exception as e:
                print(f"Error getting detailed memory stats: {e}")

        return metrics

    def _save_memory_data(self, pl_module, stage):
        if not self.enabled:
            return

        rank = getattr(pl_module.trainer, "global_rank", 0)
        global_step = getattr(pl_module.trainer, "global_step", 0)
        timestamp = time.time()

        memory_metrics = self.get_memory_usage()

        memory_metrics.update(
            {
                "global_step": global_step,
                "stage": stage,
                "timestamp": timestamp,
                "rank": rank,
            }
        )

        self.memory_data.append(memory_metrics)

    def _save_dataframes(self):
        if not self.memory_data:
            return

        time_str = time.strftime("%Y%m%d-%H%M%S")
        rank = self.memory_data[0]["rank"]

        df = pd.DataFrame(self.memory_data)
        df.to_csv(
            f"{self.output_dir}/memory_metrics_rank{rank}_{time_str}.csv", index=False
        )

    def setup(self, trainer, pl_module, stage=None):
        self._save_memory_data(pl_module, "setup")

    def on_fit_start(self, trainer, pl_module):
        self._save_memory_data(pl_module, "fit_start")

    def on_fit_end(self, trainer, pl_module):
        self._save_memory_data(pl_module, "fit_end")

    def on_train_start(self, trainer, pl_module):
        self._save_memory_data(pl_module, "train_start")

    def on_train_end(self, trainer, pl_module):
        self._save_memory_data(pl_module, "train_end")

    def on_train_epoch_start(self, trainer, pl_module):
        self._save_memory_data(pl_module, "train_epoch_start")

    def on_train_epoch_end(self, trainer, pl_module):
        self._save_memory_data(pl_module, "train_epoch_end")
        # reset peak mem at the end of every epoch
        torch.cuda.reset_peak_memory_stats(device=None)

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self._save_memory_data(pl_module, "train_batch_start")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._save_memory_data(pl_module, "train_batch_end")

    def on_before_backward(self, trainer, pl_module, loss):
        self._save_memory_data(pl_module, "before_backward")

    def on_after_backward(self, trainer, pl_module):
        self._save_memory_data(pl_module, "after_backward")

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        self._save_memory_data(pl_module, "before_optimizer_step")

    def on_before_zero_grad(self, trainer, pl_module, optimizer):
        self._save_memory_data(pl_module, "before_zero_grad")

    def teardown(self, trainer, pl_module, stage=None):
        self._save_memory_data(pl_module, "teardown")
        torch.cuda.memory._dump_snapshot(
            f"{self.output_dir}/memory_snapshot_events-{MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT}.pickle"
        )
        torch.cuda.memory._record_memory_history(enabled=None)
        self._save_dataframes()

    def on_exception(self, trainer, pl_module, exception):
        self._save_dataframes()
