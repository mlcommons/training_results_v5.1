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

from typing import Literal

from nemo.collections.diffusion.data.diffusion_energon_datamodule import (
    DiffusionDataModule,
)
from megatron.energon import get_train_dataset


class CustomDiffusionDataModule(DiffusionDataModule):
    def datasets_provider(self, worker_config, split: Literal["train", "val"] = "val"):
        """
        Provide the dataset for training or validation.

        This method retrieves the dataset for the specified split (either 'train' or 'val') and configures
        it according to the worker configuration.

        Parameters:
        worker_config: Configuration for the data loader workers.
        split (Literal['train', 'val'], optional): The data split to retrieve ('train' or 'val'). Defaults to 'val'.

        Returns:
        Dataset: The dataset configured for the specified split.
        """
        if split not in {"train", "val"}:
            raise ValueError(
                "Invalid value for split. Allowed values are 'train' or 'val'."
            )
        if self.use_train_split_for_val:
            split = "train"
        if split == "val":
            virtual_epoch_length = 0
        else:
            virtual_epoch_length = self.virtual_epoch_length
        _dataset = get_train_dataset(
            self.path,
            batch_size=self.micro_batch_size,
            task_encoder=self.task_encoder,
            worker_config=worker_config,
            max_samples_per_sequence=self.max_samples_per_sequence,
            shuffle_buffer_size=None,
            split_part=split,
            virtual_epoch_length=virtual_epoch_length,
            packing_buffer_size=self.packing_buffer_size,
        )
        return _dataset
