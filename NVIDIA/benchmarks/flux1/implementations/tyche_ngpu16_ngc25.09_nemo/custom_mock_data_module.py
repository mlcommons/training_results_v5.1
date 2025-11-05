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

from typing import List, Optional

import lightning.pytorch as pl
import torch
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader, Dataset

from nemo.collections.diffusion.data.diffusion_mock_datamodule import (
    MockDataModule,
    _MockT2IDataset,
)


class CustomMockDataModule(MockDataModule):

    def setup(self, stage: str = "") -> None:
        """
        Sets up datasets for training, validation, and testing.

        Args:
            stage (str): The stage of the process (e.g., 'fit', 'test'). Default is an empty string.
        """
        self._train_ds = CustomMockT2IDataset(
            image_H=256,
            image_W=256,
            length=self.num_train_samples,
            image_precached=self.image_precached,
            text_precached=self.text_precached,
        )
        self._validation_ds = CustomMockT2IDataset(
            image_H=256,
            image_W=256,
            length=self.num_val_samples,
            image_precached=self.image_precached,
            text_precached=self.text_precached,
            include_timestep=True,
        )
        self._test_ds = CustomMockT2IDataset(
            image_H=256,
            image_W=256,
            length=self.num_test_samples,
            image_precached=self.image_precached,
            text_precached=self.text_precached,
            include_timestep=True,
        )


class CustomMockT2IDataset(_MockT2IDataset):
    def __init__(self, *args, include_timestep=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.include_timestep = include_timestep

    def __getitem__(self, index):
        item = super().__getitem__(index)
        if self.image_precached:
            item["mean"] = torch.randn(self.latent_shape)
            item["logvar"] = torch.randn(self.latent_shape)
            item["prompt_embeds"] = torch.ones(self.prompt_embeds_shape)
            item["pooled_prompt_embeds"] = torch.ones(self.pooped_prompt_embeds_shape)
            item["text_ids"] = torch.zeros(self.text_ids_shape)
        if self.include_timestep:
            item["timestep"] = torch.tensor(index % 8)
        return item
