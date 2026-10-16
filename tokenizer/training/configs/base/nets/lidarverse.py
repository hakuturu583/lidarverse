# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Network config for the public range-only, native-grid LidarVerse tokenizer."""

from tokenizer.networks.common import ContinuousFormulation
from tokenizer.networks.transformers.transformer_vae_cross_dataset import (
    CrossDatasetTransformerVAE,
)
from utils.lazy_config import LazyCall as L
from utils.lazy_config import LazyDict

PAI_TOKENIZER_NAME = (
    "transformer_lidar_video_tokenizer_8x8x1_z64_pandar_2p5to100_1024_ema40k"
)


LIDARVERSE_TOKENIZER_CONFIG: LazyDict = L(CrossDatasetTransformerVAE)(
    resolution=(64, 1024),
    in_channels=2,
    out_channels=1,
    z_dim=64,
    base_channels=128,
    depths=[4, 2, 4],
    num_heads=[4, 16, 16],
    patch_size=(2, 2),
    window_size=(5, 35),
    dilation=[1, 1, 1],
    temporal_downsample=[False, False, False],
    temporal_upsample=[False, False, False],
    decoder_depths=None,
    decoder_num_heads=None,
    decoder_dilation=None,
    decoder_temporal_upsample=None,
    out_patch_size=None,
    activation_checkpointing=True,
    encoder_autocast_dtype=None,
    mlp_ratio=3.0,
    dropout=0.0,
    mapping_depth=2,
    positional_embedding="rope_only",
    formulation=ContinuousFormulation.VAE.name,
    bottleneck_3d=False,
    bottleneck_3d_max_t=32,
    bottleneck_3d_causal_time=True,
    bottleneck_3d_rope=False,
    mask_as_input=True,
    predict_validity=True,
)
