# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.utils.data

from diffusion.training.utils.checkpointer import MultiRankCheckpointer
from utils import misc
from utils.fsdp_checkpointer import FSDPCheckpointer
from utils.model import Model
from utils.trainer import Trainer as BaseTrainer


class Trainer(BaseTrainer):
    def __init__(self, config):
        super(Trainer, self).__init__(config)
        if config.trainer.distributed_parallelism == "ddp":
            self.checkpointer = MultiRankCheckpointer(config.checkpoint, config.job, callbacks=self.callbacks)
        elif config.trainer.distributed_parallelism == "fsdp":
            self.checkpointer = FSDPCheckpointer(config.checkpoint, config.job, callbacks=self.callbacks)
        else:
            raise ValueError(f"Unsupported distributed parallelism: {config.trainer.distributed_parallelism}")

    @torch.no_grad()
    def validate(
        self,
        model: Model,
        dataloader_val: torch.utils.data.DataLoader,
        iteration: int = 0,
    ) -> None:
        """Validation without ema_scope — FSDP models manage EMA internally."""
        self.callbacks.on_validation_start(model, dataloader_val, iteration=iteration)
        model.eval()
        for val_iter, data_batch in enumerate(dataloader_val):
            if self.config.trainer.max_val_iter is not None and val_iter >= self.config.trainer.max_val_iter:
                break
            data_batch = misc.to(data_batch, device="cuda")
            self.callbacks.on_validation_step_start(model, data_batch, iteration=iteration)
            output_batch, loss = model.validation_step(data_batch, iteration)
            self.callbacks.on_validation_step_end(
                model, data_batch, output_batch, loss, iteration=iteration
            )
        self.callbacks.on_validation_end(model, iteration=iteration)
