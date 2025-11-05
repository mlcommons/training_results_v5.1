# Copyright The Lightning AI team.
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

from typing import Union
from torch import Tensor

import numbers
import torch
import lightning.pytorch

from lightning.pytorch.trainer.connectors.logger_connector.result import (
    _METRICS,
    _ResultCollection,
    _get_default_dtype,
)
from lightning.pytorch import LightningModule


def metrics_without_prog_bar(self, on_step: bool) -> _METRICS:
    metrics = _METRICS(callback={}, log={}, pbar={})

    for _, result_metric in self.valid_items():
        # extract forward_cache or computed from the _ResultMetric
        value = self._get_cache(result_metric, on_step)
        if not isinstance(value, Tensor):
            continue

        name, forked_name = self._forked_name(result_metric, on_step)

        # populate logging metrics
        if result_metric.meta.logger:
            metrics["log"][forked_name] = value

        # populate callback metrics. callback metrics don't take `_step` forked metrics
        if self.training or result_metric.meta.on_epoch and not on_step:
            metrics["callback"][name] = value
            metrics["callback"][forked_name] = value

    return metrics


def __to_tensor(self, value: Union[Tensor, numbers.Number], name: str) -> Tensor:
    value = (
        value.clone().detach()
        if isinstance(value, Tensor)
        else torch.tensor(value, dtype=_get_default_dtype())
        # else torch.tensor(value, device=self.device, dtype=_get_default_dtype())
    )
    if not torch.numel(value) == 1:
        raise ValueError(
            f"`self.log({name}, {value})` was called, but the tensor must have a single element."
            f" You can try doing `self.log({name}, {value}.mean())`"
        )
    value = value.squeeze()
    return value


_ResultCollection.__orig_metrics__ = _ResultCollection.metrics
_ResultCollection.metrics = metrics_without_prog_bar

LightningModule.__orig_to_tensor__ = LightningModule._LightningModule__to_tensor
LightningModule._LightningModule__to_tensor = __to_tensor
