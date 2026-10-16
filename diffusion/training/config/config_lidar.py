# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Minimal configuration entry point for public LidarVerse experiments."""

import importlib
from typing import Any, List

import attrs

from diffusion.training.config.base.model import LidarModelConfig
from diffusion.training.config.text2world.registry import (
    register_configs as register_configs_text2world,
)
from diffusion.training.models.model import DiffusionModel
from diffusion.training.trainer import Trainer
from utils import config
from utils.lazy_config import PLACEHOLDER
from utils.lazy_config import LazyCall as L
from utils.lazy_config import LazyDict


@attrs.define(slots=False)
class Config(config.Config):
    defaults: List[Any] = attrs.field(
        factory=lambda: [
            "_self_",
            {"data_train": None},
            {"data_val": None},
            {"optimizer": "fusedadamw"},
            {"scheduler": "lambdalinear"},
            {"callbacks": None},
            {"net": "mininet_2b"},
            {"conditioner": "add_fps_image_size_padding_mask"},
            {"fsdp": None},
            {"ema": "power"},
            {"vae": "lidarverse_tokenizer"},
            {"checkpoint": "local"},
            {"ckpt_klass": "fsdp"},
            {"experiment": None},
        ]
    )
    model_obj: LazyDict = L(DiffusionModel)(config=PLACEHOLDER)
    optimizer_fake: LazyDict | None = None
    scheduler_fake: LazyDict | None = None


_PUBLIC_EXPERIMENT_MODULE = (
    "diffusion.training.config.text2world_hdmaptolidar."
    "experiment_lidarverse"
)
_PUBLIC_EXPERIMENTS = {
    "lidarverse_stage1_bidir_21f",
    "lidarverse_stage2_causal_9f",
    "lidarverse_stage3_sfdmd_9f",
    "lidarverse_pai_stage1_finetune",
    "lidarverse_pai_stage2_finetune",
    "lidarverse_pai_stage3_finetune",
}


def _register_lidarverse_experiments() -> None:
    from hydra.core.config_store import ConfigStore

    importlib.import_module(_PUBLIC_EXPERIMENT_MODULE).register_experiments(
        ConfigStore.instance()
    )
    importlib.import_module(
        "diffusion.training.config.text2world_hdmaptolidar.experiment_pai_release"
    ).register_experiments(ConfigStore.instance())


def make_config(experiment_name: str | None = None):
    if experiment_name is not None and experiment_name not in _PUBLIC_EXPERIMENTS:
        raise ValueError(
            f"unknown public LidarVerse experiment {experiment_name!r}; "
            f"choose one of {sorted(_PUBLIC_EXPERIMENTS)}"
        )

    c = Config(
        model=LidarModelConfig(),
        optimizer=None,
        scheduler=None,
        dataloader_train=None,
        dataloader_val=None,
    )
    c.job.project = "lidarverse"
    c.job.group = "training"
    c.job.name = "delete_${now:%Y-%m-%d}_${now:%H-%M-%S}"
    c.trainer.type = Trainer
    c.trainer.max_iter = 400_000
    c.trainer.logging_iter = 10
    c.trainer.validation_iter = 100
    c.trainer.run_validation = False
    c.trainer.callbacks = None
    c.checkpoint = None

    register_configs_text2world()
    _register_lidarverse_experiments()
    return c
