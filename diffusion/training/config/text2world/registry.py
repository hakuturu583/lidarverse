# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Minimal shared Hydra registry for LidarVerse diffusion models."""

from copy import deepcopy
from typing import Dict

from hydra.core.config_store import ConfigStore

from diffusion.checkpointers.ema_fsdp_checkpointer import (
    CheckpointConfig,
    FSDPCheckpointer,
)
from diffusion.config.base.conditioner import (
    FPSConfig,
    ImageSizeConfig,
    NumFramesConfig,
    PaddingMaskConfig,
    TextConfig,
)
from diffusion.training.conditioner import VideoConditioner
from diffusion.training.config.base.lidarverse_vae import (
    get_lidarverse_tokenizer,
)
from diffusion.training.config.base.optim import (
    FusedAdamWConfig,
    LambdaLinearSchedulerConfig,
)
from diffusion.training.networks.lvg_dit import LVGDiTForLidar
from diffusion.training.networks.minimal_v4_dit import SACConfig
from utils.ema import PowerEMATracker
from utils.lazy_config import PLACEHOLDER
from utils.lazy_config import LazyCall as L
from utils.lazy_config import LazyDict


FSDP_CHECKPOINTER: Dict[str, str] = L(FSDPCheckpointer)()

VideoConditionerFpsSizePaddingConfig: LazyDict = L(VideoConditioner)(
    text=TextConfig(),
    fps=FPSConfig(),
    num_frames=NumFramesConfig(),
    image_size=ImageSizeConfig(),
    padding_mask=PaddingMaskConfig(),
)

MININET_2B_Config: LazyDict = L(LVGDiTForLidar)(
    max_img_h=240,
    max_img_w=240,
    max_frames=128,
    in_channels=16,
    out_channels=16,
    patch_spatial=2,
    patch_temporal=1,
    model_channels=2048,
    num_blocks=28,
    num_heads=16,
    concat_padding_mask=True,
    pos_emb_cls="rope3d",
    pos_emb_learnable=True,
    pos_emb_interpolation="crop",
    use_adaln_lora=True,
    adaln_lora_dim=256,
    atten_backend="minimal_a2a",
    extra_per_block_abs_pos_emb=False,
    rope_h_extrapolation_ratio=1.0,
    rope_w_extrapolation_ratio=1.0,
    rope_t_extrapolation_ratio=1.0,
    sac_config=L(SACConfig)(),
)

PowerEMAConfig: LazyDict = L(PowerEMATracker.initialize_multi_rank_ema)(
    model=PLACEHOLDER,
    enabled=True,
    rate=0.10,
    num=3,
)


def register_configs() -> None:
    cs = ConfigStore.instance()
    cs.store(
        group="optimizer",
        package="optimizer",
        name="fusedadamw",
        node=FusedAdamWConfig,
    )
    cs.store(
        group="scheduler",
        package="scheduler",
        name="lambdalinear",
        node=LambdaLinearSchedulerConfig,
    )
    cs.store(
        group="net",
        package="model.net",
        name="mininet_2b",
        node=deepcopy(MININET_2B_Config),
    )
    cs.store(
        group="conditioner",
        package="model.conditioner",
        name="add_fps_image_size_padding_mask",
        node=VideoConditionerFpsSizePaddingConfig,
    )
    cs.store(
        group="vae",
        package="model.vae",
        name="lidarverse_tokenizer",
        node=get_lidarverse_tokenizer(),
    )
    cs.store(
        group="ema",
        package="model.ema",
        name="power",
        node=PowerEMAConfig,
    )
    cs.store(
        group="checkpoint",
        package="checkpoint",
        name="local",
        node=CheckpointConfig(
            save_iter=1000,
            load_path="",
            load_training_state=False,
            strict_resume=True,
        ),
    )
    cs.store(
        group="ckpt_klass",
        package="checkpoint.type",
        name="fsdp",
        node=FSDP_CHECKPOINTER,
    )

