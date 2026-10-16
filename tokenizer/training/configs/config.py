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

"""Default config for cosmos/tokenizer project."""

from typing import Any, List

import attrs

from tokenizer.training.configs.base.model import DefaultModelConfig
from tokenizer.training.configs.registry import register_configs
from tokenizer.training.trainer import TokenizerTrainer
from utils import config


@attrs.define(slots=False)
class Config(config.Config):
    defaults: List[Any] = attrs.field(
        factory=lambda: [
            "_self_",
            {"data_train": "lidarverse_cross_dataset_range_video"},
            {"data_val": "lidarverse_cross_dataset_range_video"},
            {"optimizer": "fused_adam"},
            {"scheduler": "warmup"},
            {"network": "lidarverse_tokenizer"},
            {"loss": "video"},
            {"metric": "reconstruction"},
            {"checkpoint": "local"},
            {"callbacks": "float32"},
            {"experiment": None},
        ]
    )


def make_config():
    c = Config(
        model=DefaultModelConfig,
        optimizer=None,
        scheduler=None,
        dataloader_train=None,
        dataloader_val=None,
        checkpoint=None,
    )
    c.job.project = "posttraining"
    c.job.group = "debug"
    c.job.name = "default_${now:%Y-%m-%d}_${now:%H-%M-%S}"

    c.trainer.type = TokenizerTrainer
    c.trainer.run_validation = True

    c.trainer.seed = 1234
    c.trainer.max_iter = 10_000_000
    c.trainer.validation_iter = 5000
    c.trainer.max_val_iter = 1
    c.trainer.logging_iter = 100

    c.trainer.callbacks = None
    c.trainer.ddp.static_graph = True
    c.trainer.ddp.find_unused_parameters = False
    register_configs()

    # Import only the public experiment. Dynamic package discovery previously
    # registered hundreds of internal and ablation configs at startup.
    from tokenizer.training.configs.experiments import (  # noqa: F401
        transformer_vae_8x8x1_z64_cross_dataset3_pandar_pandaset_nuscenes32_gridrope_mask_pts_scratch_200k,
        pai_release,
    )

    return c
