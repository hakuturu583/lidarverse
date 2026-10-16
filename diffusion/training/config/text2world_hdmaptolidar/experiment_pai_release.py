# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PAI post-training from processed release weights; no internal dataset roots."""

from copy import deepcopy
import os

from diffusion.training.config.text2world_hdmaptolidar.experiment_lidarverse import (
    stage1_config, stage2_config, stage3_config, _factorized_model_config,
)
from utils.processed_pai import loader_config


def register_experiments(cs):
    for stage, base, package in ((1, stage1_config, "bidir"), (2, stage2_config, "causal"), (3, stage3_config, "sfdmd")):
        node = deepcopy(base)
        name = f"lidarverse_pai_stage{stage}_finetune"
        node.job.name = name
        node.checkpoint.update(load_path=os.environ.get(f"LIDARVERSE_STAGE{stage}_CHECKPOINT",
            f"checkpoints/lidarverse-{package}-pai/ema_model.pt"), load_training_state=False,
            load_ema_to_reg=False, strict_resume=True, save_iter=200, save_at_iters=[])
        node.trainer.update(max_iter=2000, run_validation=False)
        node.scheduler.update(warm_up_steps=[0], cycle_lengths=[2000], f_start=[1.], f_max=[1.], f_min=[1.])
        node.dataloader_train = loader_config()
        node.dataloader_val = loader_config(validation=True)
        model = node.model
        factors = _factorized_model_config(21)
        for key, value in factors.items():
            if key.startswith(("stage2_factorized", "imaginaire_conditioned", "factorized_geometry_observation")):
                model[key] = value
        model.update(latent_shape=[64, 13 if stage == 3 else 21, 16, 128],
                     precision="bfloat16", enable_shift_stretch=False, shift_clamp_percentiles=None,
                     blockwise_warped_history_backend="torch_spatial_map",
                     blockwise_spatial_map_geometry_root_path="", blockwise_spatial_map_geometry_filter_list_path="",
                     blockwise_spatial_map_min_range=2.5, blockwise_spatial_map_max_range=100.,
                     blockwise_spatial_map_voxel_m=.10 if stage == 3 else .15, blockwise_spatial_map_radius_m=100.,
                     blockwise_spatial_map_voxel_policy="persistent_static",
                     blockwise_spatial_map_observation_unit="frame", blockwise_spatial_map_static_min_observations=2,
                     blockwise_spatial_map_include_latest=False, blockwise_spatial_map_candidate_ttl_updates=1,
                     blockwise_spatial_map_free_space_clear=True, blockwise_spatial_map_free_space_margin_m=.75,
                     blockwise_spatial_map_free_space_misses_to_remove=2)
        model.net.update(patch_spatial=[1, 2], num_dataset_embeddings=3, rope_scale_positions_by_patch=True)
        model.vae = dict(experiment_name="transformer_lidar_video_tokenizer_8x8x1_z64_pandar_2p5to100_1024_ema40k",
                         checkpoint_path=os.environ.get("LIDARVERSE_TOKENIZER_CHECKPOINT", "checkpoints/lidarverse-tokenizer-pai/ckpt.pt"),
                         mean_std_fp=os.environ.get("LIDARVERSE_TOKENIZER_LATENT_STATS", "checkpoints/lidarverse-tokenizer-pai/latent_stats.pt"),
                         use_ema=False, strict_checkpoint=True, is_bf16=False,
                         encoder_is_bf16=False, decoder_is_bf16=False)
        if stage == 3:
            model.update(sfdmd_num_real_frames=21, sfdmd_window_frames=13, sfdmd_min_range=2.5,
                         sfdmd_spatial_map_voxel_m=.10,
                         sfdmd_teacher_checkpoint=os.environ.get("LIDARVERSE_SCORE_CHECKPOINT",
                             "checkpoints/lidarverse-causal-pai/ema_model.pt"))
            model.sfdmd_teacher_net.update(patch_spatial=[1, 2], num_dataset_embeddings=3,
                                          rope_scale_positions_by_patch=True)
        cs.store(group="experiment", package="_global_", name=name, node=node)
