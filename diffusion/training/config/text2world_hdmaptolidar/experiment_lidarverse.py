# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public LidarVerse training recipes.

Stage 1 starts directly from an adapted Cosmos-Predict2 2B checkpoint. Stage 2
trains the causal four-frame student with teacher forcing, and Stage 3 applies
self-forcing distribution matching on nine-frame self-rollouts.
"""

from copy import deepcopy
import os

from hydra.core.config_store import ConfigStore

from diffusion.training.config.base.optim import (
    FusedAdamWConfig,
    LambdaLinearSchedulerConfig,
)
from diffusion.training.config.text2world.registry import MININET_2B_Config
from diffusion.training.datasets.lidar_dataset.hdmaptolidar_dataloader import (
    CrossDatasetHdmapLidarDataset,
    HomogeneousMultiHdmapDataLoader,
)
from diffusion.training.models.hdmaptolidar_model_rf import (
    FSDPHdmapToLidarFactorizedBidirectionalRFModel,
    FSDPHdmapToLidarFactorizedTeacherForcingRFModel,
)
from diffusion.training.models.hdmaptolidar_model_sf_dmd import (
    FSDPHdmapToLidarSFDMDModel,
)
from diffusion.training.networks.minimal_v4_dit import SACConfig
from diffusion.training.trainer_sf_dmd import SFDMDTrainer
from utils.lazy_config import PLACEHOLDER
from utils.lazy_config import LazyCall as L
from utils.lazy_config import LazyDict


STAGE1_NAME = "lidarverse_stage1_bidir_21f"
STAGE2_NAME = "lidarverse_stage2_causal_9f"
STAGE3_NAME = "lidarverse_stage3_sfdmd_9f"

_LATENT_CHANNELS = 64
_STAGE1_FRAMES = 21
_CAUSAL_FRAMES = 9
_DATASET_NAMES = ("nvidia_pai", "nuscenes")
_PREFIX_FRAMES_21 = [0, 1, 2, 4, 8, 10, 12, 16]
_PREFIX_PROBS_21 = [0.5] + [0.5 / 7.0] * 7
_GEOMETRY_FRAMES_21 = [1, 2, 4, 8, 10, 12, 16, 20]
_PREFIX_FRAMES_9 = [0, 1, 2, 4, 8]
_PREFIX_PROBS_9 = [0.5] + [0.125] * 4
_GEOMETRY_FRAMES_9 = [1, 2, 4, 8]


def _checkpoint(env_name: str, default: str) -> str:
    return os.environ.get(env_name, default)


def _build_cross_dataset_dataloader(
    *,
    validation: bool,
    sample_n_frames: int,
    num_workers: int = 16,
):
    suffix = "val" if validation else "train"
    dataset_names = [
        f"lidarverse_nvidia_pai_{suffix}",
        f"lidarverse_nuscenes_{suffix}",
    ]
    shared = {
        "sample_n_frames": sample_n_frames,
        "frame_sampling_method": "sequential",
        "process_input": "mono",
        "is_video_tokenizer": False,
        "blockwise_warped_history_condition": False,
        "hdmap_multichannel": False,
        "return_hdmap_semantic": False,
        "return_moving_boxes": True,
        "resolve_moving_box_track_ids": False,
        "return_full_lidar_metadata": True,
        "load_intensity": False,
    }
    per_dataset = {
        f"lidarverse_nvidia_pai_{suffix}": {
            "min_range": 2.5,
            "max_range": 100.0,
            "sensor_fps": 10.0,
            "frame_sampling_every_n_frames": 1,
            "moving_box_root": os.environ.get(
                "LIDARVERSE_NVIDIA_PAI_ROOT", "data/nvidia_pai"
            ),
        },
        f"lidarverse_nuscenes_{suffix}": {
            "min_range": 2.5,
            "max_range": 100.0,
            "hdmap_map_max_dist": 100.0,
            "sensor_fps": 2.0,
            "frame_sampling_every_n_frames": 10,
            "repeat_row": 1,
            "lidar_size": [32, 1024],
        },
    }
    return HomogeneousMultiHdmapDataLoader(
        dataset=CrossDatasetHdmapLidarDataset(
            dataset_names=dataset_names,
            shared_overrides=shared,
            per_dataset_overrides=per_dataset,
        ),
        batch_size=1,
        validation=validation,
        dataset_sampling_weights=[0.5, 0.5],
        drop_last=True,
        pin_memory=True,
        num_workers=num_workers,
    )


def _dataloader(*, validation: bool, sample_n_frames: int):
    return L(_build_cross_dataset_dataloader)(
        validation=validation,
        sample_n_frames=sample_n_frames,
        num_workers=int(os.environ.get("LIDARVERSE_DATALOADER_WORKERS", "16")),
    )


def _factorized_model_config(num_frames: int) -> dict:
    prefix_frames = (
        _PREFIX_FRAMES_21 if num_frames == _STAGE1_FRAMES else _PREFIX_FRAMES_9
    )
    prefix_probs = (
        _PREFIX_PROBS_21 if num_frames == _STAGE1_FRAMES else _PREFIX_PROBS_9
    )
    geometry_frames = (
        _GEOMETRY_FRAMES_21
        if num_frames == _STAGE1_FRAMES
        else _GEOMETRY_FRAMES_9
    )
    return {
        # Spatial dimensions are a shape hint only. Native-grid batches carry
        # their own H/W; T is checked by the bidirectional/SF-DMD contracts.
        "latent_shape": [_LATENT_CHANNELS, num_frames, 16, 150],
        "hdmap_channels": 1,
        "additional_concat_ch": _LATENT_CHANNELS,
        "skip_hdmap_proj": False,
        "encode_hdmap_with_vae": True,
        "hdmap_dropout_prob": 0.0,
        "loss_reduce": "mean",
        "sigma_data": 1.0,
        "shift": 5,
        "train_time_distribution": "logitnormal",
        "train_time_weight": "uniform",
        "input_video_has_prepend": False,
        "blockwise_warped_history_condition": True,
        "blockwise_warped_history_concat": True,
        "blockwise_warped_history_backend": "dataloader",
        "blockwise_spatial_map_self_prediction_prob": 0.0,
        "stage2_known_mask_concat": True,
        "imaginaire_conditioned_prefix_frames": prefix_frames,
        "imaginaire_conditioned_prefix_probs": prefix_probs,
        "stage2_factorized_prefix_frames": prefix_frames,
        "stage2_factorized_prefix_probs": prefix_probs,
        "stage2_factorized_hdmap_modes": ["null", "full"],
        "stage2_factorized_hdmap_probs": [0.5, 0.5],
        "stage2_factorized_geometry_modes_k0": [
            "null",
            "rolling_static",
            "frozen_object",
        ],
        "stage2_factorized_geometry_probs_k0": [0.1, 0.45, 0.45],
        "stage2_factorized_geometry_modes_known": [
            "null",
            "rolling_static",
            "frozen_object",
        ],
        "stage2_factorized_geometry_probs_known": [0.1, 0.45, 0.45],
        "stage2_factorized_hdmap_bbox_channel_start": 4,
        "stage2_factorized_force_prefix_k": -1,
        "stage2_factorized_force_hdmap_mode": "",
        "stage2_factorized_force_geometry_mode": "",
        "factorized_decouple_geometry_observations": True,
        "factorized_geometry_observation_frames": geometry_frames,
        "factorized_geometry_observation_probs": [
            1.0 / len(geometry_frames)
        ] * len(geometry_frames),
        "factorized_force_geometry_observations": -1,
        "factorized_geometry_block_size": 4,
        "factorized_geometry_prefix_frames": 1,
        "factorized_geometry_rolling_shutter": True,
        "factorized_geometry_actor_box_time_mode": "frame",
        "ema": {"enabled": True},
        "fsdp_enabled": True,
        "fsdp": {
            "policy": "block",
            "checkpoint": False,
            "min_num_params": 1024,
            "sharding_group_size": 8,
            "sharding_strategy": "hybrid",
        },
        "net": {
            "in_channels": _LATENT_CHANNELS * 3 + 1,
            "out_channels": _LATENT_CHANNELS,
            "side_condition_channels": 0,
            "num_frame_type_embeddings": 0,
            "num_dataset_embeddings": len(_DATASET_NAMES),
            "timestep_scale": 0.001,
            "rope_h_extrapolation_ratio": 3.0,
            "rope_w_extrapolation_ratio": 3.0,
            "rope_t_extrapolation_ratio": 1.0,
            "rope_enable_fps_modulation": False,
            "use_wan_fp32_strategy": True,
        },
    }


stage1_config = LazyDict(
    {
        "defaults": [
            {"override /net": "mininet_2b"},
            {"override /ckpt_klass": "fsdp"},
            {"override /checkpoint": "local"},
            {"override /vae": "lidarverse_tokenizer"},
            {"override /conditioner": "add_fps_image_size_padding_mask"},
            "_self_",
        ],
        "job": {
            "project": "posttraining",
            "group": "lidarverse",
            "name": STAGE1_NAME,
        },
        "optimizer": {
            "lr": 2 ** (-14.3),
            "weight_decay": 0.1,
            "betas": [0.9, 0.99],
            "eps": 1.0e-10,
        },
        "checkpoint": {
            "save_iter": 200,
            "save_at_iters": [2000, 4000, 6000, 8000, 10000, 15000, 20000],
            "keep_last_checkpoints": 2,
            "broadcast_via_filesystem": False,
            "load_path": _checkpoint(
                "LIDARVERSE_COSMOS_WARMSTART",
                "checkpoints/cosmos-predict2/cosmos2_2b_lidar_warmstart.pt",
            ),
            "load_training_state": False,
            "strict_resume": False,
            "keys_not_to_resume": [],
        },
        "trainer": {
            "max_iter": 20000,
            "distributed_parallelism": "fsdp",
            "logging_iter": 10,
            "grad_accum_iter": 1,
            "run_validation": True,
            "validation_iter": 1000,
            "max_val_iter": 20,
        },
        "model_parallel": {
            "sequence_parallel": False,
            "tensor_model_parallel_size": 1,
            "context_parallel_size": 1,
        },
        "model": _factorized_model_config(_STAGE1_FRAMES),
        "model_obj": L(FSDPHdmapToLidarFactorizedBidirectionalRFModel)(
            config=PLACEHOLDER,
            fsdp_checkpointer=PLACEHOLDER,
        ),
        "scheduler": {
            "warm_up_steps": [2500],
            "cycle_lengths": [20000],
            "f_start": [1.0e-6],
            "f_max": [1.0],
            "f_min": [1.0],
        },
        "dataloader_train": _dataloader(
            validation=False, sample_n_frames=_STAGE1_FRAMES
        ),
        "dataloader_val": _dataloader(
            validation=True, sample_n_frames=_STAGE1_FRAMES
        ),
    },
    # Derived stages store the trainer class directly in trainer.type.
    flags={"allow_objects": True},
)
stage1_config["model"].update(
    diffusion_forcing_block_size=_STAGE1_FRAMES,
    teacher_forcing_prefix_frames=0,
    factorized_bidirectional_stage1=True,
)
stage1_config["model"]["net"].update(
    atten_backend="minimal_a2a",
    temporal_block_size=0,
    temporal_prefix_frames=0,
)


stage2_config = deepcopy(stage1_config)
stage2_config["job"]["name"] = STAGE2_NAME
stage2_config["checkpoint"].update(
    save_iter=200,
    save_at_iters=[2000, 4000, 6000, 8000, 10000],
    load_path=_checkpoint(
        "LIDARVERSE_STAGE1_CHECKPOINT",
        "checkpoints/lidarverse-stage1/iter_000020000_ema_model.pt",
    ),
    load_ema_to_reg=False,
    load_training_state=False,
    strict_resume=False,
)
stage2_config["trainer"].update(
    max_iter=10000,
    validation_iter=1000,
)
stage2_config["scheduler"].update(
    warm_up_steps=[1000],
    cycle_lengths=[10000],
)
stage2_config["model"] = _factorized_model_config(_CAUSAL_FRAMES)
stage2_config["model"].update(
    diffusion_forcing_block_size=4,
    teacher_forcing_prefix_frames=1,
    factorized_bidirectional_stage1=False,
)
stage2_config["model"]["net"].update(
    atten_backend="torch-flex",
    temporal_block_size=4,
    temporal_prefix_frames=1,
    # FlexAttention supports full-block recomputation, not selective caching.
    sac_config=L(SACConfig)(mode="block_wise"),
)
stage2_config["model_obj"] = L(
    FSDPHdmapToLidarFactorizedTeacherForcingRFModel
)(
    config=PLACEHOLDER,
    fsdp_checkpointer=PLACEHOLDER,
)
stage2_config["dataloader_train"] = _dataloader(
    validation=False, sample_n_frames=_CAUSAL_FRAMES
)
stage2_config["dataloader_val"] = _dataloader(
    validation=True, sample_n_frames=_CAUSAL_FRAMES
)


def _score_net_config():
    net = deepcopy(MININET_2B_Config)
    net.in_channels = _LATENT_CHANNELS * 3 + 1
    net.out_channels = _LATENT_CHANNELS
    net.patch_spatial = stage2_config["model"]["net"].get(
        "patch_spatial", MININET_2B_Config.patch_spatial
    )
    net.side_condition_channels = 0
    net.num_frame_type_embeddings = 0
    net.num_dataset_embeddings = len(_DATASET_NAMES)
    net.timestep_scale = 0.001
    net.rope_h_extrapolation_ratio = 3.0
    net.rope_w_extrapolation_ratio = 3.0
    net.rope_t_extrapolation_ratio = 1.0
    net.rope_enable_fps_modulation = False
    net.use_wan_fp32_strategy = True
    net.atten_backend = "minimal_a2a"
    net.temporal_block_size = 0
    net.temporal_prefix_frames = 0
    net.sac_config = L(SACConfig)(mode="none")
    return net


stage3_config = deepcopy(stage2_config)
stage3_config["job"]["name"] = STAGE3_NAME
stage3_config["checkpoint"].update(
    save_iter=250,
    save_at_iters=[500, 1000, 1500, 2000],
    load_path=_checkpoint(
        "LIDARVERSE_STAGE2_CHECKPOINT",
        "checkpoints/lidarverse-stage2/iter_000010000.pt",
    ),
    load_ema_to_reg=True,
    load_training_state=False,
    strict_resume=True,
)
stage3_config["trainer"].update(
    type=SFDMDTrainer,
    max_iter=2000,
    logging_iter=1,
    run_validation=False,
    grad_accum_iter=1,
)
stage3_config["optimizer"].update(
    lr=2.0e-6,
    weight_decay=0.01,
    betas=[0.9, 0.999],
    eps=1.0e-8,
)
fake_optimizer = deepcopy(FusedAdamWConfig)
fake_optimizer.lr = 4.0e-7
fake_optimizer.weight_decay = 0.01
fake_optimizer.betas = [0.9, 0.999]
fake_optimizer.eps = 1.0e-8
stage3_config["optimizer_fake"] = fake_optimizer
fake_scheduler = deepcopy(LambdaLinearSchedulerConfig)
fake_scheduler.warm_up_steps = [0]
fake_scheduler.cycle_lengths = [10000000000000]
fake_scheduler.f_start = [1.0]
fake_scheduler.f_max = [1.0]
fake_scheduler.f_min = [1.0]
stage3_config["scheduler_fake"] = fake_scheduler
stage3_config["scheduler"].update(
    warm_up_steps=[0],
    cycle_lengths=[10000000000000],
    f_start=[1.0],
    f_max=[1.0],
    f_min=[1.0],
)
stage3_config["model"].update(
    sfdmd_num_real_frames=_CAUSAL_FRAMES,
    sfdmd_window_frames=_CAUSAL_FRAMES,
    sfdmd_fake_updates_per_cycle=1,
    sfdmd_student_update_freq=5,
    sfdmd_num_steps=4,
    sfdmd_sde_eta=1.0,
    sfdmd_kv_cache_frames=9,
    sfdmd_kv_sink_frames=1,
    sfdmd_tbptt_blocks=2,
    sfdmd_allow_truncated_bptt=True,
    sfdmd_run_kv_equivalence_gate=False,
    sfdmd_skip_zero_crossattn=True,
    sfdmd_stream_condition_vae_encode=True,
    sfdmd_condition_vae_cache_frames=9,
    sfdmd_accumulated_geometry_warp=False,
    sfdmd_geometry_source=False,
    sfdmd_spatial_map_radius_m=100.0,
    sfdmd_spatial_map_voxel_m=0.15,
    sfdmd_spatial_map_voxel_policy="persistent_static",
    sfdmd_spatial_map_observation_unit="block",
    sfdmd_spatial_map_static_min_observations=3,
    sfdmd_spatial_map_candidate_ttl_updates=1,
    sfdmd_spatial_map_include_latest=False,
    sfdmd_spatial_map_free_space_clear=True,
    sfdmd_spatial_map_free_space_margin_m=0.75,
    sfdmd_spatial_map_free_space_misses_to_remove=2,
    sfdmd_artifixer_matching=True,
    sfdmd_imaginaire_matching=False,
    sfdmd_clean_x0_kv_commit=True,
    sfdmd_factorized_conditioning=True,
    sfdmd_temporal_history_source="generated",
    sfdmd_geometry_history_source="generated",
    sfdmd_temporal_prefix_frames=1,
    sfdmd_ema_start_cycle=200,
    sfdmd_teacher_guidance=3.0,
    sfdmd_critic_min_sigma=0.02,
    sfdmd_critic_max_sigma=0.98,
    sfdmd_max_grad_norm=10.0,
    sfdmd_teacher_checkpoint=_checkpoint(
        "LIDARVERSE_STAGE2_EMA_CHECKPOINT",
        "checkpoints/lidarverse-stage2/iter_000010000_ema_model.pt",
    ),
    sfdmd_teacher_net=_score_net_config(),
    sfdmd_min_range=1.0,
    sfdmd_max_range=100.0,
    sfdmd_min_value=-1.0,
    sfdmd_inverse_depth=False,
    train_time_distribution="uniform",
    shift_clamp_percentiles=None,
)
stage3_config["model"]["net"].update(
    temporal_block_size=4,
    temporal_prefix_frames=1,
    side_condition_channels=0,
    sac_config=L(SACConfig)(mode="none"),
)
stage3_config["model"]["fsdp"]["use_orig_params"] = True
stage3_config["model"]["ema"].update(beta=0.99, num=1)
stage3_config["model"]["vae"] = {
    "use_ema": False,
    "encoder_is_bf16": True,
    "decoder_is_bf16": False,
}
stage3_config["model_obj"] = L(FSDPHdmapToLidarSFDMDModel)(
    config=PLACEHOLDER,
    fsdp_checkpointer=PLACEHOLDER,
)


def register_experiments(cs: ConfigStore) -> None:
    for name, node in (
        (STAGE1_NAME, stage1_config),
        (STAGE2_NAME, stage2_config),
        (STAGE3_NAME, stage3_config),
    ):
        cs.store(
            group="experiment",
            package="_global_",
            name=name,
            node=node,
        )
