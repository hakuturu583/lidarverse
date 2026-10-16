# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Train the Waymo-free LidarVerse range tokenizer from scratch."""

from hydra.core.config_store import ConfigStore

from utils import log
from utils.lazy_config import LazyDict


EXPERIMENT_NAME = (
    "transformer_vae_8x8x1_z64_cross_dataset3_pandar_pandaset_"
    "nuscenes32_gridrope_mask_pts_scratch_200k"
)
DATA_GROUP = "lidarverse_cross_dataset_range_video"

LIDARVERSE_TOKENIZER_EXPERIMENT = LazyDict(
    dict(
        defaults=[
            {"override /network": "lidarverse_tokenizer"},
            {"override /loss": "video"},
            {"override /data_train": DATA_GROUP},
            {"override /data_val": DATA_GROUP},
            {"override /optimizer": "fused_adam"},
            {"override /callbacks": ["float32", "wandbLidar"]},
            "_self_",
        ],
        job=dict(
            project="lidarverse",
            group="tokenizer",
            name="LidarVerse-Tokenizer-8x8x1-Z64-CrossDataset-Scratch-200K",
        ),
        checkpoint=dict(
            load_path=None,
            load_training_state=False,
            strict_resume=True,
            save_iter=2000,
            jit=dict(enabled=False),
        ),
        trainer=dict(
            max_iter=200000,
            validation_iter=1000,
            max_val_iter=15,
            logging_iter=100,
        ),
        model=dict(
            config=dict(
                disc_optimizer=dict(lr=0.00016),
                ema=dict(enabled=True, beta=0.999),
                precision="float32",
                loss=dict(
                    config=dict(
                        perceptual=None,
                        mask_from_batch=True,
                        color=dict(
                            config=dict(normalize_by_valid_mask=True)
                        ),
                        point_loss=dict(
                            config=dict(
                                enabled=True,
                                values=[1.0],
                                normalize_by_valid_mask=True,
                            )
                        ),
                        local_chamfer=dict(
                            config=dict(enabled=False, values=[0.0])
                        ),
                        gradient=dict(
                            config=dict(enabled=False, values=[0.0])
                        ),
                        validity_bce=dict(
                            config=dict(
                                enabled=True,
                                boundaries=[0],
                                values=[1.0],
                                min_value=-1.0,
                                raydrop_eps=0.02,
                                mask_from_batch=True,
                            )
                        ),
                        kl=dict(
                            config=dict(boundaries=[0], values=[1e-6])
                        ),
                    )
                ),
            )
        ),
        optimizer=dict(lr=2e-5),
        dataloader_train=dict(
            batch_size=1,
            mix_datasets_across_ranks=True,
            dataset_sampling_weights=(1.0 / 3.0,) * 3,
            dataset=dict(
                shared_overrides=dict(
                    process_input="mono",
                    process_output="mono",
                    sample_n_frames=8,
                    load_intensity=False,
                    emit_valid_mask=True,
                    repeat_col=1,
                    repeat_temporal=1,
                ),
                per_dataset_overrides=dict(
                    nvidia_pai_range_video_train=dict(
                        min_range=2.5,
                        max_range=100.0,
                        downsample_factor_col=3,
                        repeat_row=1,
                        lidar_size=[128, 1200],
                        lidar_resize_size=[-1, -1],
                    ),
                    pandaset_range_video_train=dict(
                        min_range=2.5,
                        max_range=100.0,
                        repeat_row=1,
                        lidar_size=[64, 1024],
                        lidar_resize_size=[-1, -1],
                    ),
                    nuscenes_range_video_train=dict(
                        min_range=2.5,
                        max_range=100.0,
                        LIDAR_FPS=20,
                        frame_sampling_every_n_frames=1,
                        center_vehicle_forward=True,
                        repeat_row=1,
                        resample_width=1024,
                        lidar_size=[32, 1024],
                        lidar_resize_size=[-1, -1],
                    ),
                ),
            ),
        ),
        dataloader_val=dict(
            batch_size=1,
            dataset=dict(
                shared_overrides=dict(
                    process_input="mono",
                    process_output="mono",
                    sample_n_frames=8,
                    load_intensity=False,
                    emit_valid_mask=True,
                    repeat_col=1,
                    repeat_temporal=1,
                ),
                per_dataset_overrides=dict(
                    nvidia_pai_range_video_val=dict(
                        min_range=2.5,
                        max_range=100.0,
                        downsample_factor_col=3,
                        repeat_row=1,
                        lidar_size=[128, 1200],
                        lidar_resize_size=[-1, -1],
                    ),
                    pandaset_range_video_val=dict(
                        min_range=2.5,
                        max_range=100.0,
                        repeat_row=1,
                        lidar_size=[64, 1024],
                        lidar_resize_size=[-1, -1],
                    ),
                    nuscenes_range_video_val=dict(
                        min_range=2.5,
                        max_range=100.0,
                        LIDAR_FPS=20,
                        frame_sampling_every_n_frames=1,
                        center_vehicle_forward=True,
                        repeat_row=1,
                        resample_width=1024,
                        lidar_size=[32, 1024],
                        lidar_resize_size=[-1, -1],
                    ),
                ),
            ),
        ),
    )
)

if LIDARVERSE_TOKENIZER_EXPERIMENT["checkpoint"]["load_path"] is not None:
    raise ValueError("The public tokenizer must train from scratch")

log.info(f"Registering experiment: {EXPERIMENT_NAME}")
ConfigStore.instance().store(
    group="experiment",
    package="_global_",
    name=EXPERIMENT_NAME,
    node=LIDARVERSE_TOKENIZER_EXPERIMENT,
)
