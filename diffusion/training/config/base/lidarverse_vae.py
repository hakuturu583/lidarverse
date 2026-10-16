# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Released LidarVerse tokenizer wrapper for diffusion training/inference."""

import os

import omegaconf

from diffusion.training.module.pretrained_vae_base import ConfigBasedVideoVAE
from utils.lazy_config import LazyCall as L


def get_lidarverse_tokenizer(
    resolution: str = "native",
    chunk_duration: int = 21,
) -> omegaconf.dictconfig.DictConfig:
    del resolution, chunk_duration
    return L(ConfigBasedVideoVAE)(
        experiment_name=(
            "transformer_vae_8x8x1_z64_cross_dataset3_pandar_pandaset_"
            "nuscenes32_gridrope_mask_pts_scratch_200k"
        ),
        checkpoint_path=os.environ.get(
            "LIDARVERSE_TOKENIZER_CHECKPOINT",
            "checkpoints/lidarverse-tokenizer/ckpt.pt",
        ),
        mean_std_fp=os.environ.get(
            "LIDARVERSE_TOKENIZER_LATENT_STATS",
            "checkpoints/lidarverse-tokenizer/latent_stats.pt",
        ),
        latent_ch=64,
        is_bf16=False,
        spatial_compression=8,
        apply_validity_mask=True,
        # The public checkpoint has EMA weights materialized as network.*.
        use_ema=False,
    )
