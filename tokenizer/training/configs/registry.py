# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Minimal Hydra registry for the public LidarVerse tokenizer."""

from hydra.core.config_store import ConfigStore

from tokenizer.training.configs.base.callback import (
    FLOAT32_CALLBACKS,
    WANDBLIDAR_CALLBACK,
)
from tokenizer.training.configs.base.checkpoint import CHECKPOINT_LOCAL
from tokenizer.training.configs.base.loss import VideoLossConfig
from tokenizer.training.configs.base.metric import MetricConfig
from tokenizer.training.configs.base.nets.lidarverse import (
    LIDARVERSE_TOKENIZER_CONFIG,
)
from tokenizer.training.configs.base.optim import (
    FusedAdamConfig,
    WarmupCosineLRConfig,
    WarmupLRConfig,
)
from tokenizer.training.datasets.lidar_datasets.tokenizer_dataloader import (
    get_cross_dataset_lidar_range_map_dataloader,
)


TRAIN_DATASETS = (
    "nvidia_pai_range_video_train",
    "pandaset_range_video_train",
    "nuscenes_range_video_train",
)
VAL_DATASETS = (
    "nvidia_pai_range_video_val",
    "pandaset_range_video_val",
    "nuscenes_range_video_val",
)


def register_configs():
    cs = ConfigStore.instance()
    cs.store(
        group="data_train",
        package="dataloader_train",
        name="lidarverse_cross_dataset_range_video",
        node=get_cross_dataset_lidar_range_map_dataloader(
            dataset_names=TRAIN_DATASETS,
            validation=False,
            mix_datasets_across_ranks=True,
            dataset_sampling_weights=(1.0 / 3.0,) * 3,
            seed=0,
            num_workers=4,
            prefetch_factor=2,
        ),
    )
    cs.store(
        group="data_val",
        package="dataloader_val",
        name="lidarverse_cross_dataset_range_video",
        node=get_cross_dataset_lidar_range_map_dataloader(
            dataset_names=VAL_DATASETS,
            validation=True,
            seed=0,
            num_workers=4,
            prefetch_factor=2,
        ),
    )
    cs.store(
        group="network",
        package="model.config.network",
        name="lidarverse_tokenizer",
        node=LIDARVERSE_TOKENIZER_CONFIG,
    )
    cs.store(
        group="optimizer",
        package="optimizer",
        name="fused_adam",
        node=FusedAdamConfig,
    )
    cs.store(
        group="scheduler",
        package="scheduler",
        name="warmup",
        node=WarmupLRConfig,
    )
    cs.store(
        group="scheduler",
        package="scheduler",
        name="warmup_cosine",
        node=WarmupCosineLRConfig,
    )
    cs.store(
        group="loss",
        package="model.config.loss",
        name="video",
        node=VideoLossConfig,
    )
    cs.store(
        group="metric",
        package="model.config.metric",
        name="reconstruction",
        node=MetricConfig,
    )
    cs.store(
        group="checkpoint",
        package="checkpoint",
        name="local",
        node=CHECKPOINT_LOCAL,
    )
    cs.store(
        group="callbacks",
        package="trainer.callbacks",
        name="float32",
        node=FLOAT32_CALLBACKS,
    )
    cs.store(
        group="callbacks",
        package="trainer.callbacks",
        name="wandbLidar",
        node=WANDBLIDAR_CALLBACK,
    )
