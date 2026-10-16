# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Self-forcing DMD for block-causal HDMap-to-LiDAR generation.

A frozen bidirectional teacher and a trainable copy provide the real and fake
score functions used by DMD. The student generates non-overlapping causal
chunks with bounded per-layer K/V history. ArtiFixer mode normally stores the
selected noisy-query K/V; the TF handoff can instead commit finalized clean
history while retaining the first chunk as an attention sink and supporting
graph-connected replay through the full generated rollout. Long rollouts always
use generated latent/range history; target LiDAR is retained only to define
tensor shapes.
"""

from __future__ import annotations

import copy
import functools
import os
from dataclasses import dataclass
from typing import Any

import torch
from megatron.core import parallel_state
from torch import Tensor, nn
from torch.nn import functional as F
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from diffusion.training.datasets.lidar_dataset.forecast_warp import (
    TorchSpatialPointMap,
    build_stage2_factorized_geometry,
    lidar_poses_for_global_tokens,
    local_hit_density_confidence,
    spatial_warp_window,
    warp_blockwise_history_to_video,
)
from diffusion.training.models.hdmaptolidar_model import (
    diffusion_fsdp_class_decorator,
)
from diffusion.training.models.hdmaptolidar_model_rf import (
    HdmapToLidarDiffusionForcingRFModel,
    _sensor_tables_from_ray_directions,
    _sample_world_synchronized_categorical,
)
from diffusion.training.utils.diffusion_forcing import temporal_chunk_slices
from diffusion.training.utils.sf_dmd import (
    active_slice_loss,
    complete_score_state_dict,
    confidence_composite_geometry,
    confidence_weighted_geometry_source,
    dmd_pseudo_target,
    fixed_sde_sigmas,
    rf_velocity_to_x0,
    sde_renoise,
    sfdmd_nonoverlap_layout,
)
from utils import distributed, log
from utils.lazy_config import instantiate as lazy_instantiate


_DIRECT_FORWARD_KEY = "__sfdmd_direct_student_forward__"


@dataclass
class SFDMDReplayRecord:
    """Everything needed to replay one selected three-latent query."""

    block_index: int
    window_start: int
    global_start: int
    score_active_start: int
    score_active_end: int
    replay_xt: Tensor
    replay_sigma: float
    student_condition: Any
    student_hdmap: Tensor
    student_side_condition: Tensor
    student_known_mask: Tensor | None = None
    student_active_start: int = 0
    student_active_end: int = -1
    generated_block: Tensor | None = None
    history_commit_block: Tensor | None = None
    generated_window: Tensor | None = None
    score_condition: Any = None
    score_hdmap: Tensor | None = None
    score_side_condition: Tensor | None = None
    score_known_mask: Tensor | None = None
    conditioned_prefix_frames: int = 0


@dataclass
class SFDMDLongRollout:
    """One cached self-forced trajectory shared by a complete DMD cycle."""

    records: list[SFDMDReplayRecord]
    generated_latents: Tensor
    generated_real_latents: Tensor
    num_real_frames: int
    exit_step: int
    conditioned_prefix_frames: int = 0
    kv_seed_frames: int = 0
    conditioned_prefix_latents: Tensor | None = None
    conditioned_prefix_condition: Any = None
    conditioned_prefix_hdmap: Tensor | None = None
    conditioned_prefix_side_condition: Tensor | None = None
    geometry_source_latents: Tensor | None = None
    geometry_confidence_latents: Tensor | None = None
    factorized_hdmap_mode: str = ""
    factorized_geometry_mode: str = ""
    factorized_geometry_observations: int = 0


@dataclass
class SFDMDScoreBatch:
    """One ArtiFixer-style critic source draw shared by the full rollout."""

    sigma: Tensor
    timestep: Tensor
    source: Tensor


class _RFScoreModule(nn.Module):
    """Bidirectional RF velocity network with Stage-2 conditioning."""

    def __init__(
        self,
        net: nn.Module,
        *,
        concat_geometry: bool = False,
        concat_known_mask: bool = False,
    ):
        super().__init__()
        self.net = net
        self.concat_geometry = concat_geometry
        self.concat_known_mask = concat_known_mask

    def forward(
        self,
        x_t: Tensor,
        timesteps: Tensor,
        hdmap: Tensor,
        side_condition: Tensor | None,
        known_mask: Tensor | None,
        condition_dict: dict[str, Any],
    ) -> Tensor:
        if x_t.shape[:1] != hdmap.shape[:1] or x_t.shape[2:] != hdmap.shape[2:]:
            raise ValueError(
                "score hdmap must align with x_t except channels: "
                f"x_t={tuple(x_t.shape)}, hdmap={tuple(hdmap.shape)}"
        )
        net_input = torch.cat([x_t, hdmap.to(x_t.dtype)], dim=1)
        if self.concat_geometry:
            if side_condition is None:
                raise ValueError(
                    "score network requires concatenated geometry"
                )
            if side_condition.shape[:1] != x_t.shape[:1] or (
                side_condition.shape[2:] != x_t.shape[2:]
            ):
                raise ValueError(
                    "score geometry must align with x_t except channels: "
                    f"x_t={tuple(x_t.shape)}, "
                    f"geometry={tuple(side_condition.shape)}"
                )
            net_input = torch.cat(
                [net_input, side_condition.to(x_t.dtype)], dim=1
            )
        if self.concat_known_mask:
            if known_mask is None:
                known_mask = x_t.new_zeros((x_t.shape[0], x_t.shape[2]))
            else:
                known_mask = known_mask.to(
                    device=x_t.device, dtype=x_t.dtype
                )
                if known_mask.ndim == 1:
                    known_mask = known_mask[None]
                if known_mask.shape[0] == 1 and x_t.shape[0] != 1:
                    known_mask = known_mask.expand(x_t.shape[0], -1)
            if known_mask.shape != (x_t.shape[0], x_t.shape[2]):
                raise ValueError(
                    "score known mask must be (B,T), got "
                    f"{tuple(known_mask.shape)} for {tuple(x_t.shape)}"
                )
            known_channel = known_mask[:, None, :, None, None].expand(
                -1, 1, -1, x_t.shape[3], x_t.shape[4]
            )
            net_input = torch.cat([net_input, known_channel], dim=1)
        t_for_net = (
            timesteps.squeeze(-1)
            if timesteps.ndim == 2 and timesteps.shape[1] == 1
            else timesteps
        )
        network_condition = dict(condition_dict)
        if (
            not self.concat_geometry
            and int(getattr(self.net, "side_condition_channels", 0)) > 0
        ):
            if side_condition is None:
                raise ValueError(
                    "score network requires accumulated-geometry side condition"
                )
            network_condition["side_condition"] = side_condition.to(
                net_input.dtype
            )
        return self.net(
            x=net_input,
            timesteps=t_for_net.to(net_input.dtype),
            **network_condition,
        ).float()


class HdmapToLidarSFDMDModel(HdmapToLidarDiffusionForcingRFModel):
    """Fixed-step, long-rollout Self-Forcing DMD model."""

    def __init__(self, config):
        super().__init__(config)
        self.sfdmd_num_real_frames = int(
            getattr(config, "sfdmd_num_real_frames", 41)
        )
        self.sfdmd_window_frames = int(
            getattr(config, "sfdmd_window_frames", 8)
        )
        self.sfdmd_fake_updates = int(
            getattr(config, "sfdmd_fake_updates_per_cycle", 5)
        )
        self.sfdmd_kv_cache_frames = int(
            getattr(
                config,
                "sfdmd_kv_cache_frames",
                2 * self.df_block_size,
            )
        )
        self.sfdmd_kv_sink_frames = int(
            getattr(config, "sfdmd_kv_sink_frames", 0)
        )
        self.sfdmd_tbptt_blocks = int(
            getattr(config, "sfdmd_tbptt_blocks", 1)
        )
        self.sfdmd_allow_truncated_bptt = bool(
            getattr(config, "sfdmd_allow_truncated_bptt", False)
        )
        self.sfdmd_run_kv_equivalence_gate = bool(
            getattr(config, "sfdmd_run_kv_equivalence_gate", True)
        )
        self.sfdmd_skip_zero_crossattn = bool(
            getattr(config, "sfdmd_skip_zero_crossattn", False)
        )
        self.sfdmd_stream_condition_vae_encode = bool(
            getattr(config, "sfdmd_stream_condition_vae_encode", False)
        )
        self.sfdmd_condition_vae_cache_frames = int(
            getattr(config, "sfdmd_condition_vae_cache_frames", 6)
        )
        self.sfdmd_accumulated_geometry_warp = bool(
            getattr(config, "sfdmd_accumulated_geometry_warp", False)
        )
        self.sfdmd_geometry_source = bool(
            getattr(config, "sfdmd_geometry_source", False)
        )
        self.sfdmd_geometry_source_noise_floor = float(
            getattr(config, "sfdmd_geometry_source_noise_floor", 0.1)
        )
        self.sfdmd_geometry_source_preencode_confidence = bool(
            getattr(
                config,
                "sfdmd_geometry_source_preencode_confidence",
                False,
            )
        )
        self.sfdmd_geometry_source_density_kernel = int(
            getattr(config, "sfdmd_geometry_source_density_kernel", 5)
        )
        self.sfdmd_geometry_source_density_saturation_hits = float(
            getattr(
                config,
                "sfdmd_geometry_source_density_saturation_hits",
                5.0,
            )
        )
        self.sfdmd_spatial_map_radius_m = float(
            getattr(config, "sfdmd_spatial_map_radius_m", 75.0)
        )
        self.sfdmd_spatial_map_voxel_m = float(
            getattr(config, "sfdmd_spatial_map_voxel_m", 0.2)
        )
        self.sfdmd_spatial_map_voxel_policy = str(
            getattr(config, "sfdmd_spatial_map_voxel_policy", "newest")
        )
        self.sfdmd_spatial_map_observation_unit = str(
            getattr(config, "sfdmd_spatial_map_observation_unit", "block")
        )
        self.sfdmd_spatial_map_static_min_observations = int(
            getattr(config, "sfdmd_spatial_map_static_min_observations", 2)
        )
        self.sfdmd_spatial_map_candidate_ttl_updates = int(
            getattr(config, "sfdmd_spatial_map_candidate_ttl_updates", 1)
        )
        self.sfdmd_spatial_map_include_latest = bool(
            getattr(config, "sfdmd_spatial_map_include_latest", True)
        )
        self.sfdmd_spatial_map_free_space_clear = bool(
            getattr(config, "sfdmd_spatial_map_free_space_clear", False)
        )
        self.sfdmd_spatial_map_free_space_margin_m = float(
            getattr(config, "sfdmd_spatial_map_free_space_margin_m", 0.75)
        )
        self.sfdmd_spatial_map_free_space_misses_to_remove = int(
            getattr(
                config,
                "sfdmd_spatial_map_free_space_misses_to_remove",
                2,
            )
        )
        self.sfdmd_sde_eta = float(
            getattr(config, "sfdmd_sde_eta", 1.0)
        )
        self.sfdmd_num_steps = int(
            getattr(config, "sfdmd_num_steps", 4)
        )
        self.sfdmd_artifixer_matching = bool(
            getattr(config, "sfdmd_artifixer_matching", False)
        )
        self.sfdmd_imaginaire_matching = bool(
            getattr(config, "sfdmd_imaginaire_matching", False)
        )
        self.sfdmd_clean_x0_kv_commit = bool(
            getattr(config, "sfdmd_clean_x0_kv_commit", False)
        )
        self.sfdmd_factorized_conditioning = bool(
            getattr(config, "sfdmd_factorized_conditioning", False)
        )
        self.sfdmd_temporal_history_source = str(
            getattr(config, "sfdmd_temporal_history_source", "generated")
        )
        self.sfdmd_geometry_history_source = str(
            getattr(config, "sfdmd_geometry_history_source", "generated")
        )
        self.sfdmd_factorized_prefix_frames = tuple(
            int(value) for value in config.stage2_factorized_prefix_frames
        )
        self.sfdmd_factorized_prefix_probs = torch.tensor(
            tuple(
                float(value)
                for value in config.stage2_factorized_prefix_probs
            ),
            dtype=torch.float32,
        )
        self.sfdmd_factorized_hdmap_modes = tuple(
            str(value) for value in config.stage2_factorized_hdmap_modes
        )
        self.sfdmd_factorized_hdmap_probs = torch.tensor(
            tuple(
                float(value)
                for value in config.stage2_factorized_hdmap_probs
            ),
            dtype=torch.float32,
        )
        self.sfdmd_factorized_geometry_modes = tuple(
            str(value)
            for value in config.stage2_factorized_geometry_modes_known
        )
        self.sfdmd_factorized_geometry_probs = torch.tensor(
            tuple(
                float(value)
                for value in config.stage2_factorized_geometry_probs_known
            ),
            dtype=torch.float32,
        )
        self.sfdmd_factorized_geometry_observation_frames = tuple(
            int(value)
            for value in config.factorized_geometry_observation_frames
        )
        self.sfdmd_factorized_geometry_observation_probs = torch.tensor(
            tuple(
                float(value)
                for value in config.factorized_geometry_observation_probs
            ),
            dtype=torch.float32,
        )
        self.sfdmd_factorized_actor_box_time_mode = str(
            getattr(
                config,
                "factorized_geometry_actor_box_time_mode",
                "frame",
            )
        )
        if self.sfdmd_factorized_actor_box_time_mode not in (
            "frame",
            "current_to_next",
            "previous_to_current",
        ):
            raise ValueError(
                "unsupported factorized_geometry_actor_box_time_mode "
                f"{self.sfdmd_factorized_actor_box_time_mode!r}"
            )
        self.imaginaire_conditioned_prefix_frames = tuple(
            int(value)
            for value in getattr(
                config,
                "imaginaire_conditioned_prefix_frames",
                [0, 1],
            )
        )
        prefix_probs = tuple(
            float(value)
            for value in getattr(
                config,
                "imaginaire_conditioned_prefix_probs",
                [0.7, 0.3],
            )
        )
        if not self.imaginaire_conditioned_prefix_frames:
            self.imaginaire_conditioned_prefix_frames = (0, 1)
        if not prefix_probs:
            prefix_probs = (0.7, 0.3)
        if len(self.imaginaire_conditioned_prefix_frames) != len(
            prefix_probs
        ):
            raise ValueError(
                "Imaginaire prefix frames/probabilities must have the same "
                "length"
            )
        if (
            any(value < 0 for value in self.imaginaire_conditioned_prefix_frames)
            or any(value < 0.0 for value in prefix_probs)
            or sum(prefix_probs) <= 0.0
        ):
            raise ValueError(
                "Imaginaire prefix frames must be non-negative and prefix "
                "probabilities must have positive mass"
            )
        self.imaginaire_conditioned_prefix_probs = torch.tensor(
            prefix_probs, dtype=torch.float32
        ) / sum(prefix_probs)
        self.sfdmd_temporal_prefix_frames = int(
            getattr(config, "sfdmd_temporal_prefix_frames", 0)
        )
        self.sfdmd_student_update_freq = int(
            getattr(config, "sfdmd_student_update_freq", 5)
        )
        self.sfdmd_ema_start_cycle = int(
            getattr(config, "sfdmd_ema_start_cycle", 0)
        )
        self.sfdmd_teacher_guidance = float(
            getattr(config, "sfdmd_teacher_guidance", 1.0)
        )
        self.sfdmd_critic_min_sigma = float(
            getattr(config, "sfdmd_critic_min_sigma", 0.02)
        )
        self.sfdmd_critic_max_sigma = float(
            getattr(config, "sfdmd_critic_max_sigma", 0.98)
        )
        self.sfdmd_max_grad_norm = float(
            getattr(config, "sfdmd_max_grad_norm", 1.0)
        )
        self.sfdmd_teacher_checkpoint = str(
            getattr(config, "sfdmd_teacher_checkpoint")
        )
        if self._sfdmd_window_tokens() != int(config.latent_shape[1]):
            raise ValueError(
                "SF-DMD window length must match the trained latent shape: "
                f"window_tokens={self._sfdmd_window_tokens()}, "
                f"latent T={config.latent_shape[1]}"
            )
        if (
            self.sfdmd_num_real_frames - self.sfdmd_window_frames
        ) % self.df_block_size != 0:
            raise ValueError(
                "num_real_frames-window_frames must be divisible by the DF "
                f"block size: ({self.sfdmd_num_real_frames}-"
                f"{self.sfdmd_window_frames}) % {self.df_block_size} != 0"
            )
        if self.sfdmd_fake_updates <= 0:
            raise ValueError("sfdmd_fake_updates_per_cycle must be positive")
        if self.sfdmd_condition_vae_cache_frames <= 0:
            raise ValueError(
                "sfdmd_condition_vae_cache_frames must be positive"
            )
        if self.sfdmd_stream_condition_vae_encode:
            vae_net = getattr(getattr(self, "vae", None), "vae_net", None)
            if not getattr(
                vae_net, "supports_encoder_temporal_kv_cache", False
            ):
                raise ValueError(
                    "streamed SF-DMD condition encoding requires a "
                    "1x-temporal causal-attention TransformerVAE encoder"
                )
        if not 0.0 <= self.sfdmd_sde_eta <= 1.0:
            raise ValueError(
                f"sfdmd_sde_eta must be in [0,1], got {self.sfdmd_sde_eta}"
            )
        if self.sfdmd_artifixer_matching and self.sfdmd_sde_eta != 1.0:
            raise ValueError(
                "ArtiFixer matching uses a fresh source endpoint at every "
                "transition; set sfdmd_sde_eta=1.0"
            )
        if self.sfdmd_artifixer_matching and self.sfdmd_geometry_source:
            student_side_channels = int(
                getattr(config.net, "side_condition_channels", 0)
            )
            score_side_channels = int(
                getattr(config.sfdmd_teacher_net, "side_condition_channels", 0)
            )
            if (
                not self.sfdmd_accumulated_geometry_warp
                or student_side_channels <= 0
                or score_side_channels != student_side_channels
            ):
                raise ValueError(
                    "geometry-source ArtiFixer matching requires accumulated "
                    "geometry and identical positive student/score side "
                    "projection widths"
                )
        if self.sfdmd_num_steps <= 0:
            raise ValueError(
                f"sfdmd_num_steps must be positive, got {self.sfdmd_num_steps}"
            )
        if self.sfdmd_student_update_freq <= 0:
            raise ValueError("sfdmd_student_update_freq must be positive")
        if (
            self.sfdmd_artifixer_matching
            and self.sfdmd_imaginaire_matching
        ):
            raise ValueError(
                "ArtiFixer and Imaginaire SF-DMD matching are exclusive"
            )
        if (
            self.sfdmd_clean_x0_kv_commit
            and not self.sfdmd_artifixer_matching
        ):
            raise ValueError(
                "explicit clean-x0 K/V commitment is only supported by the "
                "ArtiFixer-matched TF handoff"
            )
        if self.sfdmd_factorized_conditioning:
            distributions = (
                (
                    self.sfdmd_factorized_prefix_frames,
                    self.sfdmd_factorized_prefix_probs,
                    "prefix",
                ),
                (
                    self.sfdmd_factorized_hdmap_modes,
                    self.sfdmd_factorized_hdmap_probs,
                    "HDMap",
                ),
                (
                    self.sfdmd_factorized_geometry_modes,
                    self.sfdmd_factorized_geometry_probs,
                    "geometry",
                ),
                (
                    self.sfdmd_factorized_geometry_observation_frames,
                    self.sfdmd_factorized_geometry_observation_probs,
                    "geometry observations",
                ),
            )
            for values, probabilities, label in distributions:
                if (
                    not values
                    or len(values) != probabilities.numel()
                    or bool((probabilities < 0).any())
                    or float(probabilities.sum()) <= 0.0
                ):
                    raise ValueError(
                        f"invalid factorized SF-DMD {label} distribution"
                    )
                probabilities /= probabilities.sum()
            if (
                not self.sfdmd_artifixer_matching
                or self.sfdmd_imaginaire_matching
                or not self.sfdmd_clean_x0_kv_commit
            ):
                raise ValueError(
                    "factorized SF-DMD requires the ArtiFixer clean-commit "
                    "handoff"
                )
            if self.input_video_has_prepend:
                raise ValueError(
                    "factorized SF-DMD requires unique input frames"
                )
            if not bool(config.stage2_known_mask_concat) or not bool(
                config.blockwise_warped_history_concat
            ):
                raise ValueError(
                    "factorized SF-DMD requires concatenated geometry and "
                    "the known-frame mask channel"
                )
            if not bool(config.factorized_decouple_geometry_observations):
                raise ValueError(
                    "factorized SF-DMD requires K-independent geometry "
                    "observations"
                )
            if tuple(config.stage2_factorized_geometry_modes_k0) != (
                self.sfdmd_factorized_geometry_modes
            ) or not torch.allclose(
                torch.tensor(
                    config.stage2_factorized_geometry_probs_k0,
                    dtype=torch.float32,
                )
                / sum(config.stage2_factorized_geometry_probs_k0),
                self.sfdmd_factorized_geometry_probs,
            ):
                raise ValueError(
                    "factorized SF-DMD requires one K-independent geometry "
                    "mode distribution"
                )
            if set(self.sfdmd_factorized_hdmap_modes) - {"null", "full"}:
                raise ValueError(
                    "factorized SF-DMD supports null/full HDMap modes"
                )
            if set(self.sfdmd_factorized_geometry_modes) - {
                "null",
                "rolling_static",
                "frozen_object",
            }:
                raise ValueError(
                    "factorized SF-DMD supports null, rolling_static, and "
                    "frozen_object geometry"
                )
            if any(
                value < 0 or value >= self._sfdmd_total_tokens()
                for value in self.sfdmd_factorized_prefix_frames
            ):
                raise ValueError(
                    "factorized SF-DMD prefixes must leave a generated frame"
                )
            if any(
                value <= 0 or value > self._sfdmd_total_tokens()
                for value in (
                    self.sfdmd_factorized_geometry_observation_frames
                )
            ):
                raise ValueError(
                    "factorized SF-DMD geometry observation counts must fit "
                    "the rollout"
                )
            if (
                int(getattr(config.net, "side_condition_channels", 0)) != 0
                or int(
                    getattr(
                        config.sfdmd_teacher_net,
                        "side_condition_channels",
                        0,
                    )
                )
                != 0
            ):
                raise ValueError(
                    "factorized SF-DMD concatenates geometry and must not use "
                    "a side projection"
                )
        valid_history_sources = {"generated", "ground_truth"}
        if self.sfdmd_temporal_history_source not in valid_history_sources:
            raise ValueError(
                "sfdmd_temporal_history_source must be 'generated' or "
                f"'ground_truth', got {self.sfdmd_temporal_history_source!r}"
            )
        if self.sfdmd_geometry_history_source not in valid_history_sources:
            raise ValueError(
                "sfdmd_geometry_history_source must be 'generated' or "
                f"'ground_truth', got {self.sfdmd_geometry_history_source!r}"
            )
        if (
            self.sfdmd_temporal_history_source == "ground_truth"
            and not self.sfdmd_clean_x0_kv_commit
        ):
            raise ValueError(
                "ground-truth temporal history requires clean-x0 K/V commits"
            )
        if (
            self.sfdmd_geometry_history_source == "ground_truth"
            and not self.sfdmd_factorized_conditioning
        ):
            raise ValueError(
                "ground-truth geometry history requires factorized conditioning"
            )
        if self.sfdmd_ema_start_cycle < 0:
            raise ValueError("sfdmd_ema_start_cycle must be non-negative")
        if self.sfdmd_teacher_guidance < 1.0:
            raise ValueError("sfdmd_teacher_guidance must be at least 1")
        if not (
            0.0
            <= self.sfdmd_critic_min_sigma
            < self.sfdmd_critic_max_sigma
            <= 1.0
        ):
            raise ValueError(
                "SF-DMD critic sigma bounds must satisfy "
                "0 <= min < max <= 1, got "
                f"{self.sfdmd_critic_min_sigma}/"
                f"{self.sfdmd_critic_max_sigma}"
            )
        if self.sfdmd_max_grad_norm <= 0.0:
            raise ValueError("sfdmd_max_grad_norm must be positive")
        if self.sfdmd_spatial_map_radius_m <= 0.0:
            raise ValueError("sfdmd_spatial_map_radius_m must be positive")
        if not 0.0 <= self.sfdmd_geometry_source_noise_floor <= 1.0:
            raise ValueError(
                "sfdmd_geometry_source_noise_floor must be in [0,1]"
            )
        if (
            self.sfdmd_geometry_source_density_kernel <= 0
            or self.sfdmd_geometry_source_density_kernel % 2 == 0
        ):
            raise ValueError(
                "sfdmd_geometry_source_density_kernel must be positive and odd"
            )
        if self.sfdmd_geometry_source_density_saturation_hits <= 0.0:
            raise ValueError(
                "sfdmd_geometry_source_density_saturation_hits must be positive"
            )
        if self.sfdmd_spatial_map_voxel_m <= 0.0:
            raise ValueError("sfdmd_spatial_map_voxel_m must be positive")
        if self.sfdmd_spatial_map_voxel_policy not in (
            "newest",
            "persistent_static",
        ):
            raise ValueError(
                "sfdmd_spatial_map_voxel_policy must be 'newest' or "
                "'persistent_static'"
            )
        if self.sfdmd_spatial_map_observation_unit not in (
            "block",
            "frame",
        ):
            raise ValueError(
                "sfdmd_spatial_map_observation_unit must be 'block' or 'frame'"
            )
        if self.sfdmd_spatial_map_static_min_observations < 2:
            raise ValueError(
                "sfdmd_spatial_map_static_min_observations must be at least 2"
            )
        if self.sfdmd_spatial_map_candidate_ttl_updates < 1:
            raise ValueError(
                "sfdmd_spatial_map_candidate_ttl_updates must be positive"
            )
        if (
            self.sfdmd_spatial_map_free_space_clear
            and self.sfdmd_spatial_map_voxel_policy != "persistent_static"
        ):
            raise ValueError(
                "sfdmd_spatial_map_free_space_clear requires "
                "sfdmd_spatial_map_voxel_policy='persistent_static'"
            )
        if self.sfdmd_spatial_map_free_space_margin_m <= 0.0:
            raise ValueError(
                "sfdmd_spatial_map_free_space_margin_m must be positive"
            )
        if self.sfdmd_spatial_map_free_space_misses_to_remove < 1:
            raise ValueError(
                "sfdmd_spatial_map_free_space_misses_to_remove must be positive"
            )
        temporal_chunk_slices(
            self._sfdmd_total_tokens(),
            self.df_block_size,
            self.sfdmd_temporal_prefix_frames,
        )
        temporal_chunk_slices(
            self._sfdmd_window_tokens(),
            self.df_block_size,
            self.sfdmd_temporal_prefix_frames,
        )
        if self.sfdmd_imaginaire_matching:
            temporal_chunk_slices(
                self.sfdmd_kv_cache_frames,
                self.df_block_size,
                self.sfdmd_temporal_prefix_frames,
            )
            if self.sfdmd_kv_sink_frames != 0:
                raise ValueError(
                    "Imaginaire-matched SF-DMD does not use a KV sink"
                )
        else:
            if self.sfdmd_temporal_prefix_frames:
                if (
                    self.sfdmd_kv_cache_frames
                    < self.sfdmd_temporal_prefix_frames + self.df_block_size
                    or (
                        self.sfdmd_kv_cache_frames
                        - self.sfdmd_temporal_prefix_frames
                    )
                    % self.df_block_size
                    != 0
                ):
                    raise ValueError(
                        "prefix-aware sfdmd_kv_cache_frames must retain the "
                        "prefix and a positive whole number of causal blocks"
                    )
                if (
                    self.sfdmd_kv_sink_frames
                    != self.sfdmd_temporal_prefix_frames
                ):
                    raise ValueError(
                        "prefix-aware sfdmd_kv_sink_frames must equal the "
                        "temporal prefix width"
                    )
            else:
                if (
                    self.sfdmd_kv_cache_frames < self.df_block_size
                    or self.sfdmd_kv_cache_frames % self.df_block_size != 0
                ):
                    raise ValueError(
                        "sfdmd_kv_cache_frames must retain a positive whole "
                        "number of causal blocks"
                    )
                if (
                    self.sfdmd_kv_sink_frames < 0
                    or self.sfdmd_kv_sink_frames
                    >= self.sfdmd_kv_cache_frames
                    or self.sfdmd_kv_sink_frames % self.df_block_size != 0
                ):
                    raise ValueError(
                        "sfdmd_kv_sink_frames must retain a non-negative whole "
                        "number of causal blocks and be smaller than the "
                        "persistent cache"
                    )
        if self.sfdmd_tbptt_blocks <= 0:
            raise ValueError("sfdmd_tbptt_blocks must be positive")
        full_bptt_blocks = len(
            self._sfdmd_chunk_slices(self._sfdmd_total_tokens())
        )
        artifixer_sink_frames = (
            self.sfdmd_temporal_prefix_frames or self.df_block_size
        )
        artifixer_cache_frames = (
            self.sfdmd_temporal_prefix_frames + 2 * self.df_block_size
            if self.sfdmd_temporal_prefix_frames
            else 2 * self.df_block_size
        )
        if (
            self.sfdmd_allow_truncated_bptt
            and not self.sfdmd_artifixer_matching
        ):
            raise ValueError(
                "truncated SF-DMD BPTT is only supported by the explicit "
                "ArtiFixer handoff"
            )
        if self.sfdmd_artifixer_matching and (
            self.sfdmd_kv_sink_frames != artifixer_sink_frames
            or self.sfdmd_kv_cache_frames != artifixer_cache_frames
        ):
            raise ValueError(
                "ArtiFixer-aligned LiDAR SF-DMD requires the first chunk as "
                "an attention sink and two chunks of retained history"
            )
        if (
            self.sfdmd_artifixer_matching
            and self.sfdmd_tbptt_blocks != full_bptt_blocks
            and not self.sfdmd_allow_truncated_bptt
        ):
            raise ValueError(
                "canonical ArtiFixer-aligned LiDAR SF-DMD requires "
                f"full-rollout BPTT ({full_bptt_blocks} blocks); set "
                "sfdmd_allow_truncated_bptt=True only for an explicit "
                "memory-bounded handoff"
            )
        if self.sfdmd_imaginaire_matching:
            if self.input_video_has_prepend:
                raise ValueError(
                    "Imaginaire-matched SF-DMD requires unique input frames; "
                    "set input_video_has_prepend=False"
                )
            if self.sfdmd_temporal_prefix_frames <= 0:
                raise ValueError(
                    "Imaginaire-matched SF-DMD requires a positive temporal "
                    "prefix"
                )
            allowed_conditioned_prefixes = {
                0,
                self.sfdmd_temporal_prefix_frames,
            }
            if not set(self.imaginaire_conditioned_prefix_frames).issubset(
                allowed_conditioned_prefixes
            ):
                raise ValueError(
                    "Imaginaire SF-DMD conditioned prefixes must contain "
                    "whole initial chunks; expected only "
                    f"{sorted(allowed_conditioned_prefixes)}, got "
                    f"{self.imaginaire_conditioned_prefix_frames}"
                )
            if (
                int(getattr(config.net, "temporal_block_size", 0))
                != self.df_block_size
                or int(
                    getattr(config.net, "temporal_prefix_frames", 0)
                )
                != self.sfdmd_temporal_prefix_frames
            ):
                raise ValueError(
                    "Imaginaire SF-DMD model/network temporal layouts must "
                    "match"
                )
            if self.sfdmd_num_real_frames != self.sfdmd_window_frames:
                raise ValueError(
                    "Imaginaire-matched SF-DMD scores one complete rollout; "
                    "sfdmd_num_real_frames must equal sfdmd_window_frames"
                )
            if (
                self.sfdmd_kv_cache_frames
                != self._sfdmd_window_tokens()
                or self.sfdmd_kv_sink_frames != 0
                or self.sfdmd_tbptt_blocks != 1
            ):
                raise ValueError(
                    "Imaginaire-matched SF-DMD requires a full-sequence "
                    "clean K/V cache, no sink, and one-block replay"
                )
            insufficient_map_observations = (
                self.sfdmd_spatial_map_static_min_observations < 3
                if self.sfdmd_spatial_map_observation_unit == "block"
                else self.sfdmd_spatial_map_static_min_observations < 2
            )
            if (
                not self.sfdmd_accumulated_geometry_warp
                or self.sfdmd_spatial_map_include_latest
                or insufficient_map_observations
                or self.sfdmd_geometry_source
            ):
                raise ValueError(
                    "Imaginaire-matched LiDAR SF-DMD requires accumulated "
                    "strict geometry side conditioning, no latest sweep, "
                    "and pure Gaussian initialization"
                )

        self._warp_min_range = float(getattr(config, "sfdmd_min_range"))
        self._warp_max_range = float(getattr(config, "sfdmd_max_range"))
        self._warp_min_value = float(getattr(config, "sfdmd_min_value"))
        self._warp_inverse_depth = bool(
            getattr(config, "sfdmd_inverse_depth", False)
        )
        geometry_root = str(
            getattr(config, "sfdmd_geometry_root_path", "") or ""
        )
        geometry_filter = str(
            getattr(config, "sfdmd_geometry_filter_list_path", "") or ""
        )
        self._warp_geometry = None
        if geometry_root and geometry_filter:
            from tokenizer.training.datasets.lidar_datasets.waymo_ncore import (
                load_waymo_lidar_geometry,
            )

            self._warp_geometry = load_waymo_lidar_geometry(
                root_path=geometry_root,
                filter_list_path=geometry_filter,
                resample_width=int(
                    getattr(config, "sfdmd_geometry_resample_width", -1)
                ),
            )
        if self.sfdmd_skip_zero_crossattn:
            if not hasattr(self.net, "enable_zero_crossattn_skip"):
                raise NotImplementedError(
                    f"{type(self.net).__name__} does not support zero text "
                    "cross-attention bypass"
                )
            student_blocks = self.net.enable_zero_crossattn_skip()
            log.info(
                "Enabled exact zero-text cross-attention bypass on "
                f"{student_blocks} student blocks"
            )
        self._set_up_score_models()
        log.info(
            "HdmapToLidarSFDMDModel: "
            f"real_frames={self.sfdmd_num_real_frames}, "
            f"input_video_has_prepend={self.input_video_has_prepend}, "
            f"blocks={len(self._sfdmd_chunk_slices(self._sfdmd_total_tokens()))}, "
            f"kv_cache_frames={self.sfdmd_kv_cache_frames}, "
            f"kv_sink_frames={self.sfdmd_kv_sink_frames}, "
            f"tbptt_blocks={self.sfdmd_tbptt_blocks}, "
            "allow_truncated_bptt="
            f"{self.sfdmd_allow_truncated_bptt}, "
            f"skip_zero_crossattn={self.sfdmd_skip_zero_crossattn}, "
            f"stream_condition_encode="
            f"{self.sfdmd_stream_condition_vae_encode}, "
            f"condition_cache_frames="
            f"{self.sfdmd_condition_vae_cache_frames}, "
            f"accumulated_geometry={self.sfdmd_accumulated_geometry_warp}, "
            f"geometry_source={self.sfdmd_geometry_source}, "
            "geometry_source_preencode_confidence="
            f"{self.sfdmd_geometry_source_preencode_confidence}, "
            "geometry_noise_floor="
            f"{self.sfdmd_geometry_source_noise_floor}, "
            f"spatial_radius_m={self.sfdmd_spatial_map_radius_m}, "
            f"spatial_voxel_m={self.sfdmd_spatial_map_voxel_m}, "
            f"spatial_voxel_policy={self.sfdmd_spatial_map_voxel_policy}, "
            "spatial_observation_unit="
            f"{self.sfdmd_spatial_map_observation_unit}, "
            "spatial_static_min_obs="
            f"{self.sfdmd_spatial_map_static_min_observations}, "
            "spatial_candidate_ttl="
            f"{self.sfdmd_spatial_map_candidate_ttl_updates}, "
            f"spatial_include_latest={self.sfdmd_spatial_map_include_latest}, "
            "spatial_free_clear="
            f"{self.sfdmd_spatial_map_free_space_clear}, "
            "spatial_free_margin_m="
            f"{self.sfdmd_spatial_map_free_space_margin_m}, "
            "spatial_free_misses="
            f"{self.sfdmd_spatial_map_free_space_misses_to_remove}, "
            f"num_steps={self.sfdmd_num_steps}, "
            f"sde_eta={self.sfdmd_sde_eta}, "
            f"artifixer_matching={self.sfdmd_artifixer_matching}, "
            f"imaginaire_matching={self.sfdmd_imaginaire_matching}, "
            f"clean_x0_kv_commit={self.sfdmd_clean_x0_kv_commit}, "
            "factorized_conditioning="
            f"{self.sfdmd_factorized_conditioning}, "
            "temporal_history_source="
            f"{self.sfdmd_temporal_history_source}, "
            "geometry_history_source="
            f"{self.sfdmd_geometry_history_source}, "
            f"student_update_freq={self.sfdmd_student_update_freq}, "
            f"ema_start_cycle={self.sfdmd_ema_start_cycle}, "
            f"teacher_guidance={self.sfdmd_teacher_guidance}, "
            f"max_grad_norm={self.sfdmd_max_grad_norm}, "
            f"fake_updates={self.sfdmd_fake_updates}, "
            f"teacher={self.sfdmd_teacher_checkpoint}"
        )

    def _configure_warp_geometry(
        self,
        data_batch: dict[str, Any],
        reference: Tensor,
    ) -> None:
        """Select calibrated spinning-LiDAR geometry for this sensor batch."""
        sensor_rays = data_batch.get("sensor_ray_directions")
        if sensor_rays is None:
            if self._warp_geometry is None:
                raise KeyError(
                    "SF-DMD requires sensor_ray_directions when no legacy "
                    "geometry fallback is configured"
                )
            return
        rays = torch.as_tensor(
            sensor_rays,
            device=reference.device,
            dtype=torch.float32,
        )
        if rays.ndim != 4 or rays.shape[-1] != 3:
            raise ValueError(
                "sensor_ray_directions must be (B,H,W,3), got "
                f"{tuple(rays.shape)}"
            )
        if tuple(rays.shape[1:3]) != tuple(reference.shape[-2:]):
            raise ValueError(
                "sensor rays and range images must use the same native grid: "
                f"rays={tuple(rays.shape[1:3])}, "
                f"video={tuple(reference.shape[-2:])}"
            )
        dataset_ids = data_batch.get("sensor_dataset_id")
        if dataset_ids is not None:
            ids = torch.as_tensor(dataset_ids).reshape(-1)
            if ids.shape[0] != rays.shape[0] or not bool(
                (ids == ids[0]).all()
            ):
                raise ValueError(
                    "SF-DMD batches must be homogeneous in sensor_dataset_id"
                )
        depth_bounds = data_batch.get("sensor_depth_bounds")
        if depth_bounds is None:
            raise KeyError(
                "cross-sensor SF-DMD requires sensor_depth_bounds"
            )
        depth_bounds = torch.as_tensor(
            depth_bounds, device=reference.device, dtype=torch.float32
        )
        if depth_bounds.ndim == 1:
            depth_bounds = depth_bounds[None]
        if depth_bounds.shape != (rays.shape[0], 2):
            raise ValueError(
                "sensor_depth_bounds must be (B,2), got "
                f"{tuple(depth_bounds.shape)}"
            )
        if not bool(
            torch.isclose(
                depth_bounds, depth_bounds[:1].expand_as(depth_bounds)
            ).all()
        ):
            raise ValueError(
                "SF-DMD batches must share one sensor depth contract"
            )
        self._warp_min_range = float(depth_bounds[0, 0].item())
        self._warp_max_range = float(depth_bounds[0, 1].item())
        ray_grid = rays[0].contiguous()
        elevations, azimuths, row_offsets = (
            _sensor_tables_from_ray_directions(
                ray_grid, allow_approximate=True
            )
        )
        self._warp_geometry = (
            ray_grid,
            elevations,
            azimuths,
            row_offsets,
        )

    def _sfdmd_chunk_slices(
        self, total_tokens: int
    ) -> list[tuple[int, int]]:
        return temporal_chunk_slices(
            total_tokens,
            self.df_block_size,
            self.sfdmd_temporal_prefix_frames,
        )

    def _sfdmd_total_tokens(self) -> int:
        return self.sfdmd_num_real_frames + int(
            self.input_video_has_prepend
        )

    def _sfdmd_window_tokens(self) -> int:
        return self.sfdmd_window_frames + int(
            self.input_video_has_prepend
        )

    def _sfdmd_chunk_frames(self, global_start: int) -> int:
        if self.sfdmd_temporal_prefix_frames and global_start == 0:
            return self.sfdmd_temporal_prefix_frames
        return self.df_block_size

    def _sample_imaginaire_conditioned_prefix(
        self, *, num_frames: int, device: torch.device
    ) -> int:
        if not self.sfdmd_imaginaire_matching:
            return 0
        index = _sample_world_synchronized_categorical(
            self.imaginaire_conditioned_prefix_probs.to(device=device)
        )
        prefix_frames = self.imaginaire_conditioned_prefix_frames[index]
        if prefix_frames >= num_frames:
            raise ValueError(
                "Imaginaire conditioned prefix must leave at least one "
                f"generated frame, got K={prefix_frames} for T={num_frames}"
            )
        return prefix_frames

    def _sample_factorized_rollout_modes(
        self,
        *,
        num_frames: int,
        device: torch.device,
    ) -> tuple[int, str, str, int, Tensor]:
        prefix_index = _sample_world_synchronized_categorical(
            self.sfdmd_factorized_prefix_probs.to(device=device)
        )
        prefix_frames = self.sfdmd_factorized_prefix_frames[prefix_index]
        if prefix_frames >= num_frames:
            raise ValueError(
                "factorized SF-DMD prefix must leave a generated frame"
            )
        hdmap_index = _sample_world_synchronized_categorical(
            self.sfdmd_factorized_hdmap_probs.to(device=device)
        )
        geometry_index = _sample_world_synchronized_categorical(
            self.sfdmd_factorized_geometry_probs.to(device=device)
        )
        hdmap_mode = self.sfdmd_factorized_hdmap_modes[hdmap_index]
        geometry_mode = self.sfdmd_factorized_geometry_modes[geometry_index]
        geometry_observations = 0
        geometry_source_mask = torch.zeros(
            num_frames, device=device, dtype=torch.float32
        )
        if geometry_mode == "frozen_object":
            observation_index = _sample_world_synchronized_categorical(
                self.sfdmd_factorized_geometry_observation_probs.to(
                    device=device
                )
            )
            geometry_observations = (
                self.sfdmd_factorized_geometry_observation_frames[
                    observation_index
                ]
            )
            geometry_source_mask[:geometry_observations] = 1.0
        return (
            prefix_frames,
            hdmap_mode,
            geometry_mode,
            geometry_observations,
            geometry_source_mask,
        )

    def _factorized_known_mask_window(
        self,
        known_mask: Tensor,
        window_start: int,
    ) -> Tensor:
        mask_video = known_mask[None, None, :, None, None]
        return self._window_clip(
            mask_video,
            window_start,
            self.sfdmd_window_frames,
        )[0, 0, :, 0, 0].contiguous()

    @torch.no_grad()
    def _build_factorized_rollout_geometry(
        self,
        source_video: Tensor,
        data_batch: dict[str, Any],
        poses: Tensor,
        geometry_mode: str,
        geometry_source_mask: Tensor,
    ) -> Tensor:
        ray_dirs, elev, azim, row_off = self._warp_geometry
        motion_batch = (
            data_batch.get("pose_sensor_start_end")
            if bool(self.config.factorized_geometry_rolling_shutter)
            else None
        )
        if (
            bool(self.config.factorized_geometry_rolling_shutter)
            and motion_batch is None
        ):
            raise KeyError(
                "factorized SF-DMD rolling-shutter geometry requires "
                "pose_sensor_start_end"
            )
        required_boxes = (
            "moving_box_object_to_world",
            "moving_box_lwh",
            "moving_box_valid",
            "moving_box_track_id",
        )
        missing = [
            name for name in required_boxes if data_batch.get(name) is None
        ]
        if missing:
            raise KeyError(
                "factorized SF-DMD geometry is missing batch fields "
                f"{missing}"
            )
        source_pose_batch = data_batch.get(
            "factorized_geometry_source_lidar_to_world"
        )
        source_motion_batch = data_batch.get(
            "factorized_geometry_source_pose_sensor_start_end"
        )
        warped = []
        for batch_idx in range(source_video.shape[0]):
            geometry_i, _, _ = build_stage2_factorized_geometry(
                source_video[batch_idx, :1],
                poses[batch_idx],
                geometry_source_mask,
                geometry_mode,
                pose_sensor_start_end=(
                    motion_batch[batch_idx]
                    if motion_batch is not None
                    else None
                ),
                geometry_source_lidar_to_world=(
                    source_pose_batch[batch_idx]
                    if source_pose_batch is not None
                    else None
                ),
                geometry_source_pose_sensor_start_end=(
                    source_motion_batch[batch_idx]
                    if source_motion_batch is not None
                    else None
                ),
                actor_box_time_mode=(
                    self.sfdmd_factorized_actor_box_time_mode
                ),
                box_object_to_world=data_batch[
                    "moving_box_object_to_world"
                ][batch_idx],
                box_lwh=data_batch["moving_box_lwh"][batch_idx],
                box_valid=data_batch["moving_box_valid"][batch_idx],
                box_track_id=data_batch["moving_box_track_id"][batch_idx],
                block_size=int(self.config.factorized_geometry_block_size),
                prefix_frames=int(
                    self.config.factorized_geometry_prefix_frames
                ),
                input_has_prepend=self.input_video_has_prepend,
                ray_dirs=ray_dirs,
                row_elevations_rad=elev,
                column_azimuths_rad=azim,
                row_azimuth_offsets_rad=row_off,
                min_range=float(
                    self.config.blockwise_spatial_map_min_range
                ),
                max_range=float(
                    self.config.blockwise_spatial_map_max_range
                ),
                min_value=float(
                    self.config.blockwise_spatial_map_min_value
                ),
                inverse_depth=bool(
                    self.config.blockwise_spatial_map_inverse_depth
                ),
                radius_m=float(
                    self.config.blockwise_spatial_map_radius_m
                ),
                voxel_size_m=float(
                    self.config.blockwise_spatial_map_voxel_m
                ),
                voxel_policy=str(
                    self.config.blockwise_spatial_map_voxel_policy
                ),
                observation_unit=str(
                    self.config.blockwise_spatial_map_observation_unit
                ),
                static_min_observations=int(
                    self.config.blockwise_spatial_map_static_min_observations
                ),
                candidate_ttl_updates=int(
                    self.config.blockwise_spatial_map_candidate_ttl_updates
                ),
                include_latest=bool(
                    self.config.blockwise_spatial_map_include_latest
                ),
                free_space_clear=bool(
                    self.config.blockwise_spatial_map_free_space_clear
                ),
                free_space_margin_m=float(
                    self.config.blockwise_spatial_map_free_space_margin_m
                ),
                free_space_misses_to_remove=int(
                    self.config.blockwise_spatial_map_free_space_misses_to_remove
                ),
            )
            warped.append(geometry_i)
        return torch.stack(warped).to(
            device=source_video.device, dtype=source_video.dtype
        )

    # ------------------------------------------------------------------
    # Teacher / fake-score setup
    # ------------------------------------------------------------------

    def _sfdmd_sigmas(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> Tensor:
        return fixed_sde_sigmas(
            num_steps=self.sfdmd_num_steps,
            shift=self._rf_shift,
            device=device,
            dtype=dtype,
        )

    def _sfdmd_sigmas_for_block(
        self,
        block_index: int,
        *,
        device: torch.device | str,
    ) -> Tensor:
        if not self.sfdmd_imaginaire_matching:
            return self._sfdmd_sigmas(device=device)
        values = (
            [1.0, 15.0 / 16.0, 5.0 / 6.0, 5.0 / 8.0, 0.0]
            if block_index == 0
            else [1.0, 5.0 / 6.0, 0.0]
        )
        return torch.tensor(values, device=device, dtype=torch.float32)

    def _sample_rollout_exit_step(self, device: torch.device) -> int:
        """Sample one shared ArtiFixer exit step for every block and rank."""
        if not self.sfdmd_artifixer_matching:
            return self.sfdmd_num_steps - 1
        if distributed.get_rank() == 0:
            exit_step = torch.randint(
                self.sfdmd_num_steps,
                (1,),
                device=device,
                dtype=torch.long,
            )
        else:
            exit_step = torch.zeros(1, device=device, dtype=torch.long)
        torch.distributed.broadcast(exit_step, src=0)
        return int(exit_step.item())

    def _build_score_module(self) -> _RFScoreModule:
        return _RFScoreModule(
            lazy_instantiate(copy.deepcopy(self.config.sfdmd_teacher_net)),
            concat_geometry=bool(
                self.config.blockwise_warped_history_concat
            ),
            concat_known_mask=bool(self.config.stage2_known_mask_concat),
        )

    @staticmethod
    def _net_state_from_checkpoint(path: str) -> dict[str, Tensor]:
        model_path = path.replace(".pt", "_reg_model.pt")
        if os.path.exists(model_path):
            path = model_path
        state = torch.load(
            path,
            map_location=lambda storage, loc: storage,
            weights_only=False,
        )
        if isinstance(state, dict) and "model" in state and isinstance(
            state["model"], dict
        ):
            state = state["model"]
        if not isinstance(state, dict):
            raise TypeError(f"checkpoint {path} is not a state dict")
        prefixes = ("module.net.", "model.net.", "net.")
        for prefix in prefixes:
            selected = {
                f"net.{key[len(prefix):]}": value
                for key, value in state.items()
                if key.startswith(prefix)
            }
            if selected:
                return selected
        # A dedicated fake-score checkpoint is already an _RFScoreModule state.
        if any(key.startswith("net.") for key in state):
            return state
        raise KeyError(f"checkpoint {path} contains no net.* tensors")

    def _load_score_module(
        self,
        module: _RFScoreModule,
        path: str,
        fallback_state: dict[str, Tensor] | None = None,
    ) -> None:
        state, dropped = complete_score_state_dict(
            self._net_state_from_checkpoint(path),
            module.state_dict(),
            fallback_state,
        )
        result = module.load_state_dict(state, strict=True)
        if dropped:
            log.info(
                f"Ignored {len(dropped)} score-checkpoint tensors absent from "
                "the bidirectional critic"
            )
        log.info(f"Loaded score network {path}: {result}")

    @staticmethod
    def _score_wrap_policy(module: _RFScoreModule):
        classes = module.net.fsdp_wrap_block_cls
        if not isinstance(classes, (list, tuple, set)):
            classes = [classes]
        return functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=set(classes),
        )

    def _sync_full_module(self, module: nn.Module) -> None:
        mesh = parallel_state.fsdp_device_mesh
        distributed.sync_model_states(module, mesh.get_group(mesh_dim="shard"))
        distributed.sync_model_states(
            module, mesh.get_group(mesh_dim="replicate")
        )

    @staticmethod
    def _rank0_action_or_raise(action, description: str) -> None:
        error = None
        if distributed.get_rank() == 0:
            try:
                action()
            except Exception as exc:  # noqa: BLE001
                error = f"{description}: {exc!r}"
        objects = [error]
        torch.distributed.broadcast_object_list(objects, src=0)
        if objects[0] is not None:
            raise RuntimeError(objects[0])

    def _set_up_score_models(self) -> None:
        teacher = self._build_score_module().cuda().to(**self.tensor_kwargs)
        self._rank0_action_or_raise(
            lambda: self._load_score_module(
                teacher, self.sfdmd_teacher_checkpoint
            ),
            "teacher checkpoint load failed",
        )
        teacher.eval().requires_grad_(False)

        fake_score = self._build_score_module().cuda().to(**self.tensor_kwargs)
        fake_load_path = None
        checkpointer = getattr(self, "fsdp_checkpointer", None)
        if hasattr(checkpointer, "get_fake_score_load_path"):
            fake_load_path = checkpointer.get_fake_score_load_path()

        def _initialize_fake_score():
            if fake_load_path is not None:
                self._load_score_module(
                    fake_score,
                    fake_load_path,
                    fallback_state=teacher.state_dict(),
                )
            else:
                fake_score.load_state_dict(teacher.state_dict(), strict=True)
        self._rank0_action_or_raise(
            _initialize_fake_score,
            "fake-score initialization failed",
        )
        if self.sfdmd_skip_zero_crossattn:
            teacher_blocks = teacher.net.enable_zero_crossattn_skip()
            fake_blocks = fake_score.net.enable_zero_crossattn_skip()
            log.info(
                "Enabled exact zero-text cross-attention bypass on "
                f"{teacher_blocks} teacher and {fake_blocks} fake-score blocks"
            )
        if isinstance(self.model, FSDP):
            self._sync_full_module(teacher)
            self._sync_full_module(fake_score)
            config = self.config
            strategy = {
                "full": ShardingStrategy.FULL_SHARD,
                "hybrid": ShardingStrategy.HYBRID_SHARD,
            }[config.fsdp.sharding_strategy]
            mesh = parallel_state.fsdp_device_mesh
            if config.fsdp.sharding_strategy == "hybrid":
                process_group = (
                    mesh.get_group(mesh_dim="shard"),
                    mesh.get_group(mesh_dim="replicate"),
                )
            else:
                process_group = mesh.get_group(mesh_dim="shard")
            teacher = FSDP(
                teacher,
                sync_module_states=True,
                sharding_strategy=strategy,
                auto_wrap_policy=self._score_wrap_policy(teacher),
                process_group=process_group,
                device_id=torch.cuda.current_device(),
                limit_all_gathers=True,
            )
            fake_score = FSDP(
                fake_score,
                sync_module_states=True,
                sharding_strategy=strategy,
                auto_wrap_policy=self._score_wrap_policy(fake_score),
                process_group=process_group,
                device_id=torch.cuda.current_device(),
                limit_all_gathers=True,
            )
            log.info("Sharded frozen teacher score model with FSDP")
        self.teacher_score = teacher
        self.fake_score = fake_score

    def train(self, mode: bool = True):
        super().train(mode)
        self.teacher_score.eval()
        return self

    # ------------------------------------------------------------------
    # FSDP-safe student and score forwards
    # ------------------------------------------------------------------

    def _direct_student_forward(self, payload: dict[str, Any]) -> Tensor:
        previous_known_mask = getattr(
            self, "_current_stage2_known_frame_mask", None
        )
        self._current_hdmap = payload["hdmap"]
        self._current_blockwise_warped_history = payload["side_condition"]
        self._current_stage2_known_frame_mask = payload.get("known_mask")
        self._current_sfdmd_temporal_positions = payload.get(
            "temporal_positions"
        )
        self._current_sfdmd_self_attn_kv_cache = payload.get("kv_cache")
        self._current_sfdmd_store_self_attn_kv = payload.get(
            "store_kv", False
        )
        self._current_sfdmd_self_attn_kv_max_frames = (
            self.sfdmd_kv_cache_frames
        )
        self._current_sfdmd_self_attn_kv_sink_frames = (
            self.sfdmd_kv_sink_frames
        )
        self._current_sfdmd_detach_self_attn_kv = payload.get(
            "detach_kv", True
        )
        try:
            return self._rf_network_forward(
                payload["x_t"],
                payload["timesteps"],
                payload["condition"],
            )
        finally:
            self._current_stage2_known_frame_mask = previous_known_mask
            self._current_sfdmd_temporal_positions = None
            self._current_sfdmd_self_attn_kv_cache = None
            self._current_sfdmd_store_self_attn_kv = False
            self._current_sfdmd_self_attn_kv_sink_frames = 0
            self._current_sfdmd_detach_self_attn_kv = True

    def training_step(
        self, data_batch: dict[str, Any], iteration: int
    ) -> tuple[dict[str, Any], Tensor]:
        if data_batch.get(_DIRECT_FORWARD_KEY, False):
            velocity = self._direct_student_forward(data_batch)
            output = {"velocity": velocity}
            if data_batch.get("store_kv", False):
                # FSDP rebuilds nested input containers while moving them to the
                # compute device, so mutations to the input cache list do not
                # propagate to the caller. Return the updated list explicitly.
                output["kv_cache"] = data_batch["kv_cache"]
            return output, velocity.new_zeros(())
        raise RuntimeError(
            "HdmapToLidarSFDMDModel is driven by SFDMDTrainer cycles, not the "
            "single-loss Trainer.training_step path"
        )

    def _student_velocity(
        self,
        x_t: Tensor,
        timesteps: Tensor,
        condition: Any,
        hdmap: Tensor,
        side_condition: Tensor,
        kv_cache: list[dict[str, Tensor] | None],
        global_start: int,
        *,
        store_kv: bool = False,
        detach_kv: bool = True,
        framewise_clean_commit: bool = False,
        known_mask: Tensor | None = None,
    ) -> Tensor:
        block_size = x_t.shape[2]
        expected_block_size = self._sfdmd_chunk_frames(global_start)
        if framewise_clean_commit:
            if (
                not (
                    self.sfdmd_imaginaire_matching
                    or self.sfdmd_clean_x0_kv_commit
                )
                or not store_kv
                or block_size != 1
            ):
                raise ValueError(
                    "framewise clean commits require a clean-commit mode, "
                    "store_kv=True, and exactly one frame"
                )
        elif block_size != expected_block_size:
            raise ValueError(
                f"student KV call must contain {expected_block_size} frames, "
                f"got {block_size}"
            )
        payload = {
            _DIRECT_FORWARD_KEY: True,
            "x_t": x_t.to(**self.tensor_kwargs),
            "timesteps": timesteps,
            "condition": condition,
            "hdmap": hdmap,
            "side_condition": side_condition,
            "known_mask": known_mask,
            "kv_cache": kv_cache,
            "store_kv": store_kv,
            "detach_kv": detach_kv,
            "temporal_positions": torch.arange(
                global_start,
                global_start + block_size,
                device=x_t.device,
            ),
        }
        if isinstance(self.model, FSDP):
            output, _ = self.model(payload, 0)
            if store_kv:
                updated_cache = output.get("kv_cache")
                if updated_cache is None:
                    raise RuntimeError(
                        "FSDP student forward did not return its updated KV cache"
                    )
                kv_cache[:] = updated_cache
            return output["velocity"].float()
        return self._direct_student_forward(payload).float()

    def new_student_kv_cache(self) -> list[dict[str, Tensor] | None]:
        net = self.net
        while hasattr(net, "module"):
            net = net.module
        return [None] * len(net.blocks)

    @torch.no_grad()
    def seed_student_conditioned_prefix(
        self,
        rollout: SFDMDLongRollout,
        kv_cache: list[dict[str, Tensor] | None],
    ) -> None:
        """Restore observed clean-prefix K/V before generated replay."""
        seed_frames = (
            rollout.kv_seed_frames
            if rollout.kv_seed_frames > 0
            else rollout.conditioned_prefix_frames
        )
        if seed_frames == 0:
            return
        if (
            rollout.conditioned_prefix_latents is None
            or rollout.conditioned_prefix_condition is None
            or rollout.conditioned_prefix_hdmap is None
            or rollout.conditioned_prefix_side_condition is None
        ):
            raise RuntimeError(
                "conditioned SF-DMD rollout is missing prefix replay state"
            )
        self._seed_student_clean_history(
            rollout.conditioned_prefix_latents,
            rollout.conditioned_prefix_condition,
            rollout.conditioned_prefix_hdmap,
            rollout.conditioned_prefix_side_condition,
            kv_cache,
            0,
            known_mask=rollout.conditioned_prefix_latents.new_ones(
                seed_frames
            ),
        )

    def _seed_student_clean_history(
        self,
        clean_block: Tensor,
        condition: Any,
        hdmap: Tensor,
        side_condition: Tensor,
        kv_cache: list[dict[str, Tensor] | None],
        global_start: int,
        *,
        detach_kv: bool = True,
        known_mask: Tensor | None = None,
    ) -> None:
        """Commit finalized clean history one frame at a time."""
        block_frames = clean_block.shape[2]
        if (
            hdmap.shape[2] != block_frames
            or side_condition.shape[2] != block_frames
        ):
            raise ValueError(
                "clean-history conditions must match the finalized block: "
                f"clean={tuple(clean_block.shape)}, hdmap={tuple(hdmap.shape)}, "
                f"side={tuple(side_condition.shape)}"
            )
        total_frames = self._sfdmd_total_tokens()
        if global_start < 0 or global_start + block_frames > total_frames:
            raise ValueError(
                f"clean-history block [{global_start}:"
                f"{global_start + block_frames}] exceeds T={total_frames}"
            )
        for offset in range(block_frames):
            frame_index = global_start + offset
            if frame_index == total_frames - 1:
                break
            clean_timesteps = torch.zeros(
                (clean_block.shape[0], 1),
                device=clean_block.device,
                dtype=torch.float32,
            )
            self._student_velocity(
                (
                    clean_block[:, :, offset : offset + 1].detach()
                    if detach_kv
                    else clean_block[:, :, offset : offset + 1]
                ),
                clean_timesteps,
                condition,
                hdmap[:, :, offset : offset + 1].contiguous(),
                side_condition[:, :, offset : offset + 1].contiguous(),
                kv_cache,
                frame_index,
                store_kv=True,
                detach_kv=detach_kv,
                framewise_clean_commit=True,
                known_mask=(
                    None
                    if known_mask is None
                    else known_mask[offset : offset + 1]
                ),
            )

    @staticmethod
    def detach_student_kv_cache(
        kv_cache: list[dict[str, Tensor] | None],
    ) -> None:
        """Cut the replay graph while preserving the bounded cache contents."""
        for layer_idx, layer_cache in enumerate(kv_cache):
            if layer_cache is None:
                continue
            kv_cache[layer_idx] = {
                "k": layer_cache["k"].detach(),
                "v": layer_cache["v"].detach(),
            }

    @torch.no_grad()
    def verify_student_kv_equivalence(
        self, data_batch: dict[str, Any]
    ) -> dict[str, Tensor]:
        """Gate the real checkpoint on full versus cached chunkwise output."""
        self._set_sfdmd_dataset_ids(data_batch, data_batch["video"])
        self._configure_warp_geometry(data_batch, data_batch["video"])
        window = self.sfdmd_window_frames
        window_tokens = self._sfdmd_window_tokens()
        chunks = self._sfdmd_chunk_slices(window_tokens)
        poses = torch.as_tensor(
            data_batch["lidar_to_world"],
            device=data_batch["video"].device,
            dtype=torch.float32,
        )
        wb = dict(data_batch)
        wb["video"] = self._window_clip(data_batch["video"], 0, window)
        wb["lidar_hdmap"] = self._window_clip(
            data_batch["lidar_hdmap"], 0, window
        )
        spatial_maps = (
            self._new_spatial_maps(
                data_batch["video"].shape[0],
                data_batch["video"].device,
            )
            if self.sfdmd_accumulated_geometry_warp
            else None
        )
        global_warp = (
            data_batch["video"][:, :1, :window_tokens].new_full(
                (
                    data_batch["video"].shape[0],
                    1,
                    window_tokens,
                    data_batch["video"].shape[-2],
                    data_batch["video"].shape[-1],
                ),
                self._warp_min_value,
            )
            if self.sfdmd_accumulated_geometry_warp
            else None
        )
        condition, hdmap, side, _, _ = self._prepare_stage_context(
            wb,
            poses,
            0,
            0,
            window_tokens,
            None,
            spatial_maps,
            global_warp,
            None,
            None,
        )
        batch_size = hdmap.shape[0]
        generator = torch.Generator(device=hdmap.device)
        generator.manual_seed(20260713)
        x = torch.randn(
            (
                batch_size,
                int(self.state_shape[0]),
                window_tokens,
                int(self.state_shape[2]),
                int(self.state_shape[3]),
            ),
            generator=generator,
            device=hdmap.device,
            dtype=torch.float32,
        )
        timestep_values = torch.linspace(
            125.0,
            875.0,
            len(chunks),
            device=hdmap.device,
            dtype=torch.float32,
        )
        timesteps = torch.cat(
            [
                timestep_values[index].expand(end - start)
                for index, (start, end) in enumerate(chunks)
            ]
        )[None].expand(batch_size, -1)
        full_payload = {
            _DIRECT_FORWARD_KEY: True,
            "x_t": x.to(**self.tensor_kwargs),
            "timesteps": timesteps,
            "condition": condition,
            "hdmap": hdmap,
            "side_condition": side,
        }
        if isinstance(self.model, FSDP):
            full_output, _ = self.model(full_payload, 0)
            full = full_output["velocity"].float()
        else:
            full = self._direct_student_forward(full_payload).float()

        cache = self.new_student_kv_cache()
        cached_parts = []
        cache_token_counts = []
        for start, end in chunks:
            cached_parts.append(
                self._student_velocity(
                    x[:, :, start:end],
                    timesteps[:, start:end],
                    condition,
                    hdmap[:, :, start:end],
                    side[:, :, start:end],
                    cache,
                    start,
                    store_kv=True,
                )
            )
            missing_layers = [
                index for index, entry in enumerate(cache)
                if entry is None
            ]
            if missing_layers:
                raise RuntimeError(
                    "DiT KV-cache store did not persist through the student "
                    f"forward; missing layers={missing_layers}"
                )
            cache_token_counts.append(int(cache[0]["k"].shape[1]))
        cached = torch.cat(cached_parts, dim=2)
        abs_error = (cached - full).abs()
        max_abs = abs_error.max()
        mean_abs = abs_error.mean()
        per_block = []
        for start, end in chunks:
            error = abs_error[:, :, start:end]
            per_block.append(
                {
                    "start": start,
                    "max": float(error.max()),
                    "mean": float(error.mean()),
                }
            )
        atol = float(
            getattr(self.config, "sfdmd_kv_equivalence_atol", 2.0e-2)
        )
        # Block 0 has no cached history, so its discrepancy measures only the
        # BF16 FlexAttention drift caused by changing the compiled query shape
        # from a masked full sequence to one unmasked causal chunk. Cached
        # blocks must remain within that baseline instead of satisfying
        # elementwise allclose near zero, where relative tolerance cannot help.
        baseline_max = per_block[0]["max"]
        baseline_mean = per_block[0]["mean"]
        history_max = max(row["max"] for row in per_block[1:])
        history_mean = max(row["mean"] for row in per_block[1:])
        history_max_budget = max(baseline_max, 1.25 * atol)
        history_mean_budget = max(baseline_mean, 0.25 * atol)
        local_gate_passed = (
            bool(torch.isfinite(max_abs))
            and bool(torch.isfinite(mean_abs))
            and history_max <= history_max_budget
            and history_mean <= history_mean_budget
        )
        local_failure = None
        if not local_gate_passed:
            rank = (
                torch.distributed.get_rank()
                if torch.distributed.is_initialized()
                else 0
            )
            local_failure = (
                f"rank={rank}: max_abs={float(max_abs):.6g}, "
                f"mean_abs={float(mean_abs):.6g}, base_atol={atol}, "
                f"history_budgets=(max={history_max_budget:.6g}, "
                f"mean={history_mean_budget:.6g}), "
                f"cache_tokens={cache_token_counts}, per_block={per_block}"
            )
        failures = [local_failure]
        if torch.distributed.is_initialized():
            failures = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(failures, local_failure)
        failures = [failure for failure in failures if failure is not None]
        if failures:
            raise RuntimeError(
                "actual student failed the distributed chunkwise DiT KV-cache "
                "gate: " + "; ".join(failures)
            )
        gate_metrics = torch.stack(
            [
                max_abs.detach(),
                mean_abs.detach(),
                max_abs.new_tensor(baseline_max),
                max_abs.new_tensor(history_max),
            ]
        )
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(
                gate_metrics, op=torch.distributed.ReduceOp.MAX
            )
        log.info(
            "Student DiT KV-cache gate passed: "
            f"global_max_abs={float(gate_metrics[0]):.6g}, "
            f"global_mean_abs_max={float(gate_metrics[1]):.6g}, "
            f"cache_tokens={cache_token_counts}, per_block={per_block}"
        )
        return {
            "sfdmd/kv_gate_max_abs": gate_metrics[0],
            "sfdmd/kv_gate_mean_abs": gate_metrics[1],
            "sfdmd/kv_gate_baseline_max": gate_metrics[2],
            "sfdmd/kv_gate_history_max": gate_metrics[3],
        }

    def _set_sfdmd_dataset_ids(
        self,
        data_batch: dict[str, Any],
        reference: Tensor,
    ) -> None:
        self._current_sensor_dataset_ids = None
        if self._num_dataset_embeddings == 0:
            return
        dataset_ids = data_batch.get("sensor_dataset_id")
        if dataset_ids is None:
            raise KeyError(
                "dataset-conditioned SF-DMD requires sensor_dataset_id"
            )
        dataset_ids = torch.as_tensor(
            dataset_ids, device=reference.device, dtype=torch.long
        ).reshape(-1)
        if dataset_ids.shape != (reference.shape[0],):
            raise ValueError(
                "sensor_dataset_id must have one value per SF-DMD sample, "
                f"got {tuple(dataset_ids.shape)} for B={reference.shape[0]}"
            )
        if bool(
            (
                (dataset_ids < 0)
                | (dataset_ids >= self._num_dataset_embeddings)
            ).any()
        ):
            raise ValueError(
                "sensor_dataset_id contains a value outside [0, "
                f"{self._num_dataset_embeddings})"
            )
        self._current_sensor_dataset_ids = dataset_ids

    def _condition_dict(
        self,
        condition: Any,
        *,
        batch_size: int,
        device: torch.device,
    ) -> dict[str, Any]:
        condition_dict = condition.to_dict()
        dataset_ids = self._current_sensor_dataset_ids
        if dataset_ids is not None:
            dataset_ids = dataset_ids.to(device=device, dtype=torch.long)
            if dataset_ids.shape[0] != batch_size:
                if batch_size % dataset_ids.shape[0] != 0:
                    raise ValueError(
                        "dataset ids cannot be repeated to match the score "
                        f"batch: ids={tuple(dataset_ids.shape)}, B={batch_size}"
                    )
                dataset_ids = dataset_ids.repeat(
                    batch_size // dataset_ids.shape[0]
                )
            condition_dict["dataset_ids_B"] = dataset_ids
        return condition_dict

    def _score_velocity(
        self,
        score_model: nn.Module,
        x_t: Tensor,
        timesteps: Tensor,
        hdmap: Tensor,
        side_condition: Tensor | None,
        condition: Any,
        conditioned_prefix_frames: int = 0,
        known_mask: Tensor | None = None,
    ) -> Tensor:
        score_timesteps = timesteps
        if known_mask is not None or conditioned_prefix_frames > 0:
            if timesteps.ndim == 1:
                score_timesteps = timesteps[:, None].expand(
                    -1, x_t.shape[2]
                ).clone()
            elif timesteps.shape == (x_t.shape[0], 1):
                score_timesteps = timesteps.expand(-1, x_t.shape[2]).clone()
            elif timesteps.shape != (x_t.shape[0], x_t.shape[2]):
                raise ValueError(
                    "score timesteps must be (B,), (B,1), or (B,T), got "
                    f"{tuple(timesteps.shape)}"
                )
            if known_mask is None:
                score_timesteps[:, :conditioned_prefix_frames] = 0.0
            else:
                known = known_mask.to(
                    device=x_t.device, dtype=torch.bool
                )
                if known.ndim == 1:
                    known = known[None]
                if known.shape[0] == 1 and x_t.shape[0] != 1:
                    known = known.expand(x_t.shape[0], -1)
                if known.shape != (x_t.shape[0], x_t.shape[2]):
                    raise ValueError(
                        "score known mask must align with x_t, got "
                        f"{tuple(known.shape)} for {tuple(x_t.shape)}"
                    )
                score_timesteps.masked_fill_(known, 0.0)
        return score_model(
            x_t.to(**self.tensor_kwargs),
            score_timesteps,
            hdmap.to(**self.tensor_kwargs),
            None
            if side_condition is None
            else side_condition.to(**self.tensor_kwargs),
            known_mask,
            self._condition_dict(
                condition,
                batch_size=x_t.shape[0],
                device=x_t.device,
            ),
        ).float()

    # ------------------------------------------------------------------
    # Long generated-history rollout
    # ------------------------------------------------------------------

    def _window_clip(
        self, x_long: Tensor, start: int, window_frames: int
    ) -> Tensor:
        """Return one model-aligned temporal window."""
        num_real = x_long.shape[2] - int(self.input_video_has_prepend)
        if start < 0 or start + window_frames > num_real:
            raise ValueError(
                f"window [{start}:{start + window_frames}] outside "
                f"{num_real} real frames"
            )
        if not self.input_video_has_prepend:
            return x_long[
                :, :, start : start + window_frames
            ].contiguous()
        real = x_long[:, :, 1 + start : 1 + start + window_frames]
        return torch.cat([real[:, :, :1], real], dim=2).contiguous()

    def _generated_warp(
        self,
        target_video: Tensor,
        poses: Tensor,
        window_start: int,
        global_start: int,
        decoded_prefix: Tensor | None,
        spatial_maps: list[TorchSpatialPointMap] | None,
        global_warp: Tensor | None,
        global_geometry_source: Tensor | None,
        global_geometry_confidence: Tensor | None,
        block_frames: int | None = None,
    ) -> tuple[Tensor, Tensor | None, Tensor | None]:
        batch_size, _, t_cur, height, width = target_video.shape
        if block_frames is None:
            block_frames = self._sfdmd_chunk_frames(global_start)
        ray_dirs, elev, azim, row_off = self._warp_geometry
        if self.sfdmd_accumulated_geometry_warp:
            if spatial_maps is None or len(spatial_maps) != batch_size:
                raise ValueError(
                    "accumulated geometry requires one spatial map per batch"
                )
            if global_warp is None:
                raise ValueError(
                    "accumulated geometry requires a global raw-warp cache"
                )
            active_warp = []
            active_source = []
            active_confidence = []
            for batch_idx in range(batch_size):
                target_poses = lidar_poses_for_global_tokens(
                    poses[batch_idx],
                    global_start,
                    block_frames,
                    input_has_prepend=self.input_video_has_prepend,
                )
                if self.sfdmd_geometry_source:
                    source, hit, temporal_confidence = spatial_maps[
                        batch_idx
                    ].render_soft_proposal(target_poses)
                    density_confidence = local_hit_density_confidence(
                        hit,
                        kernel_size=(
                            self.sfdmd_geometry_source_density_kernel
                        ),
                        saturation_hits=(
                            self.sfdmd_geometry_source_density_saturation_hits
                        ),
                    )
                    confidence = (
                        temporal_confidence * density_confidence
                    ).clamp_(0.0, 1.0)
                    empty = source.new_full((), self._warp_min_value)
                    active_warp.append(
                        confidence.unsqueeze(0) * source
                        + (1.0 - confidence.unsqueeze(0)) * empty
                    )
                    active_source.append(source)
                    active_confidence.append(confidence.unsqueeze(0))
                else:
                    warped, _ = spatial_maps[batch_idx].render(target_poses)
                    active_warp.append(warped)
            active = torch.stack(active_warp, dim=0).to(
                device=target_video.device, dtype=target_video.dtype
            )
            global_end = global_start + block_frames
            global_warp[:, :, global_start:global_end] = active
            source_window = None
            confidence_window = None
            if self.sfdmd_geometry_source:
                if (
                    global_geometry_source is None
                    or global_geometry_confidence is None
                ):
                    raise ValueError(
                        "soft geometry requires global source/confidence caches"
                    )
                source = torch.stack(active_source, dim=0).to(
                    device=target_video.device, dtype=target_video.dtype
                )
                confidence = torch.stack(active_confidence, dim=0).to(
                    device=target_video.device, dtype=torch.float32
                )
                global_geometry_source[
                    :, :, global_start:global_end
                ] = source
                global_geometry_confidence[
                    :, :, global_start:global_end
                ] = confidence
                source_window = spatial_warp_window(
                    global_geometry_source,
                    window_start=window_start,
                    window_frames=(
                        t_cur - int(self.input_video_has_prepend)
                    ),
                    input_has_prepend=self.input_video_has_prepend,
                )
                confidence_window = spatial_warp_window(
                    global_geometry_confidence,
                    window_start=window_start,
                    window_frames=(
                        t_cur - int(self.input_video_has_prepend)
                    ),
                    input_has_prepend=self.input_video_has_prepend,
                )
            return spatial_warp_window(
                global_warp,
                window_start=window_start,
                window_frames=t_cur - int(self.input_video_has_prepend),
                input_has_prepend=self.input_video_has_prepend,
            ), source_window, confidence_window

        source = target_video[:, :1].new_full(
            (batch_size, 1, t_cur, height, width),
            self._warp_min_value,
        )
        if decoded_prefix is not None:
            source[:, :, : decoded_prefix.shape[2]] = decoded_prefix[:, :1]
        warped_batch = []
        for batch_idx in range(batch_size):
            warped, _ = warp_blockwise_history_to_video(
                video=source[batch_idx],
                lidar_to_world=poses[
                    batch_idx,
                    window_start : window_start + t_cur - 1,
                ],
                block_size=self.df_block_size,
                ray_dirs=ray_dirs,
                row_elevations_rad=elev,
                column_azimuths_rad=azim,
                row_azimuth_offsets_rad=row_off,
                min_range=self._warp_min_range,
                max_range=self._warp_max_range,
                min_value=self._warp_min_value,
                inverse_depth=self._warp_inverse_depth,
                history_mode="all",
            )
            warped_batch.append(torch.from_numpy(warped))
        return (
            torch.stack(warped_batch).to(
                device=target_video.device, dtype=target_video.dtype
            ),
            None,
            None,
        )

    def _new_spatial_maps(
        self, batch_size: int, device: torch.device
    ) -> list[TorchSpatialPointMap]:
        ray_dirs, elev, azim, row_off = self._warp_geometry
        return [
            TorchSpatialPointMap(
                ray_dirs=ray_dirs,
                row_elevations_rad=elev,
                column_azimuths_rad=azim,
                row_azimuth_offsets_rad=row_off,
                min_range=self._warp_min_range,
                max_range=self._warp_max_range,
                min_value=self._warp_min_value,
                inverse_depth=self._warp_inverse_depth,
                radius_m=self.sfdmd_spatial_map_radius_m,
                voxel_size_m=self.sfdmd_spatial_map_voxel_m,
                voxel_policy=self.sfdmd_spatial_map_voxel_policy,
                observation_unit=self.sfdmd_spatial_map_observation_unit,
                static_min_observations=(
                    self.sfdmd_spatial_map_static_min_observations
                ),
                candidate_ttl_updates=(
                    self.sfdmd_spatial_map_candidate_ttl_updates
                ),
                include_latest=self.sfdmd_spatial_map_include_latest,
                free_space_clear=self.sfdmd_spatial_map_free_space_clear,
                free_space_margin_m=(
                    self.sfdmd_spatial_map_free_space_margin_m
                ),
                free_space_misses_to_remove=(
                    self.sfdmd_spatial_map_free_space_misses_to_remove
                ),
                device=device,
            )
            for _ in range(batch_size)
        ]

    def _append_generated_geometry(
        self,
        spatial_maps: list[TorchSpatialPointMap],
        decoded_new: Tensor,
        poses: Tensor,
        global_start: int,
    ) -> None:
        """Append newly committed real sweeps using the model token layout."""
        for batch_idx, spatial_map in enumerate(spatial_maps):
            spatial_map.append_block(
                normalized_block=decoded_new[batch_idx, :1],
                lidar_to_world=poses[batch_idx],
                global_start=global_start,
                input_has_prepend=self.input_video_has_prepend,
            )

    def _prepare_stage_context(
        self,
        window_batch: dict[str, Any],
        poses: Tensor,
        window_start: int,
        global_start: int,
        t_cur: int,
        decoded_prefix: Tensor | None,
        spatial_maps: list[TorchSpatialPointMap] | None,
        global_warp: Tensor | None,
        global_geometry_source: Tensor | None,
        global_geometry_confidence: Tensor | None,
        *,
        fixed_condition: Any = None,
        fixed_hdmap: Tensor | None = None,
        fixed_warp: Tensor | None = None,
        block_frames: int | None = None,
    ) -> tuple[Any, Tensor, Tensor, Tensor | None, Tensor | None]:
        stage_batch = dict(window_batch)
        target = window_batch["video"][:, :, :t_cur].contiguous()
        stage_batch["video"] = torch.full_like(target, self._warp_min_value)
        stage_batch["lidar_hdmap"] = window_batch["lidar_hdmap"][
            :, :, :t_cur
        ].contiguous()
        if fixed_warp is None:
            warped, geometry_source, geometry_confidence = (
                self._generated_warp(
                    target,
                    poses,
                    window_start,
                    global_start,
                    decoded_prefix,
                    spatial_maps,
                    global_warp,
                    global_geometry_source,
                    global_geometry_confidence,
                    block_frames,
                )
            )
        else:
            if fixed_warp.shape != target[:, :1].shape:
                raise ValueError(
                    "fixed factorized geometry must match the condition "
                    f"window: geometry={tuple(fixed_warp.shape)}, "
                    f"target={tuple(target[:, :1].shape)}"
                )
            warped = fixed_warp
            geometry_source = None
            geometry_confidence = None
        geometry_z = None
        confidence_z = None
        if self.sfdmd_geometry_source:
            if geometry_source is None or geometry_confidence is None:
                raise RuntimeError(
                    "SF-DMD geometry source/confidence was not rendered"
                )
            geometry_input = (
                confidence_composite_geometry(
                    geometry_source,
                    geometry_confidence,
                    self._warp_min_value,
                )
                if self.sfdmd_geometry_source_preencode_confidence
                else geometry_source
            )
            geometry_z = self.encode(geometry_input.float()).contiguous()
            confidence_z = F.adaptive_avg_pool3d(
                geometry_confidence.float(), geometry_z.shape[2:]
            ).clamp_(0.0, 1.0)
        if fixed_condition is not None:
            if fixed_hdmap is None:
                raise ValueError("fixed_hdmap is required with fixed_condition")
            side_z = self.encode(warped.float()).contiguous().to(
                self.diffusion_dtype
            )
            return (
                fixed_condition,
                fixed_hdmap[:, :, :t_cur].detach().clone(),
                side_z,
                None if geometry_z is None else geometry_z.detach(),
                None if confidence_z is None else confidence_z.detach(),
            )
        stage_batch["lidar_blockwise_warped_history"] = warped
        _, _, condition = self.get_data_and_condition(stage_batch)
        if self._current_hdmap is None:
            raise RuntimeError("SF-DMD requires VAE-encoded HDMap conditioning")
        if self._current_blockwise_warped_history is None:
            raise RuntimeError("SF-DMD requires generated warp-history conditioning")
        return (
            condition,
            self._current_hdmap.detach().clone(),
            self._current_blockwise_warped_history.detach().clone(),
            None if geometry_z is None else geometry_z.detach(),
            None if confidence_z is None else confidence_z.detach(),
        )

    def _prepare_stream_stage_context(
        self,
        window_batch: dict[str, Any],
        poses: Tensor,
        window_start: int,
        global_start: int,
        t_cur: int,
        decoded_prefix: Tensor | None,
        spatial_maps: list[TorchSpatialPointMap] | None,
        global_warp: Tensor | None,
        global_geometry_source: Tensor | None,
        global_geometry_confidence: Tensor | None,
        active: slice,
        temporal_kv_cache,
        geometry_temporal_kv_cache,
        *,
        fixed_warp: Tensor | None = None,
    ) -> tuple[
        Any,
        Tensor,
        Tensor,
        Tensor | None,
        Tensor | None,
        Any,
        Any,
    ]:
        """Encode only the active student condition block with VAE KV cache."""
        stage_batch = dict(window_batch)
        target = window_batch["video"][:, :, :t_cur].contiguous()
        if fixed_warp is None:
            warped, geometry_source, geometry_confidence = (
                self._generated_warp(
                    target,
                    poses,
                    window_start,
                    global_start,
                    decoded_prefix,
                    spatial_maps,
                    global_warp,
                    global_geometry_source,
                    global_geometry_confidence,
                    active.stop - active.start,
                )
            )
        else:
            if fixed_warp.shape != target[:, :1].shape:
                raise ValueError(
                    "fixed factorized geometry must match the streamed "
                    f"condition window: geometry={tuple(fixed_warp.shape)}, "
                    f"target={tuple(target[:, :1].shape)}"
                )
            warped = fixed_warp
            geometry_source = None
            geometry_confidence = None
        active_batch = dict(stage_batch)
        active_batch["video"] = torch.full_like(
            target[:, :, active], self._warp_min_value
        )
        active_batch["lidar_hdmap"] = window_batch["lidar_hdmap"][
            :, :, active
        ].contiguous()
        active_batch["lidar_blockwise_warped_history"] = warped[
            :, :, active
        ].contiguous()
        condition, new_cache = self.get_online_condition_stream(
            active_batch,
            temporal_kv_cache=temporal_kv_cache,
            max_cache_frames=self.sfdmd_condition_vae_cache_frames,
            allow_training=True,
        )
        geometry_z = None
        confidence_z = None
        new_geometry_cache = geometry_temporal_kv_cache
        if self.sfdmd_geometry_source:
            if geometry_source is None or geometry_confidence is None:
                raise RuntimeError(
                    "streamed SF-DMD geometry source/confidence was not rendered"
                )
            geometry_input = (
                confidence_composite_geometry(
                    geometry_source[:, :, active],
                    geometry_confidence[:, :, active],
                    self._warp_min_value,
                )
                if self.sfdmd_geometry_source_preencode_confidence
                else geometry_source[:, :, active]
            )
            geometry_z, new_geometry_cache = self.encode_stream(
                geometry_input.contiguous().float(),
                temporal_kv_cache=geometry_temporal_kv_cache,
                max_cache_frames=self.sfdmd_condition_vae_cache_frames,
            )
            geometry_z = geometry_z.contiguous()
            confidence_z = F.adaptive_avg_pool3d(
                geometry_confidence[:, :, active].float(),
                geometry_z.shape[2:],
            ).clamp_(0.0, 1.0)
        if self._current_hdmap is None:
            raise RuntimeError(
                "streamed SF-DMD condition encode did not store HDMap"
            )
        if self._current_blockwise_warped_history is None:
            raise RuntimeError(
                "streamed SF-DMD condition encode did not store warp history"
            )
        active_frames = active.stop - active.start
        if self._current_hdmap.shape[2] != active_frames:
            raise RuntimeError(
                "streamed SF-DMD HDMap returned the wrong block length: "
                f"{tuple(self._current_hdmap.shape)}"
            )
        return (
            condition,
            self._current_hdmap.detach().clone(),
            self._current_blockwise_warped_history.detach().clone(),
            None if geometry_z is None else geometry_z.detach(),
            None if confidence_z is None else confidence_z.detach(),
            new_cache,
            new_geometry_cache,
        )

    @torch.no_grad()
    def _encode_score_hdmap(self, hdmap_window: Tensor) -> Tensor:
        """Keep teacher/fake-score HDMap conditioning on the trained layout."""
        hdmap_z = self.encode(hdmap_window.float()).contiguous().to(
            self.diffusion_dtype
        )
        if hdmap_z.shape[2] != self._sfdmd_window_tokens():
            raise RuntimeError(
                "score HDMap encode returned the wrong window length: "
                f"{tuple(hdmap_z.shape)}"
            )
        return hdmap_z.detach()

    def _sample_stage(
        self,
        *,
        block_index: int,
        window_start: int,
        global_start: int,
        score_active_start: int,
        condition: Any,
        hdmap: Tensor,
        side_condition: Tensor,
        geometry_source: Tensor | None,
        geometry_confidence: Tensor | None,
        kv_cache: list[dict[str, Tensor] | None],
        exit_step: int,
        known_prefix_latents: Tensor | None = None,
        known_mask: Tensor | None = None,
        history_commit_latents: Tensor | None = None,
    ) -> tuple[Tensor, SFDMDReplayRecord]:
        batch_size = hdmap.shape[0]
        block_frames = hdmap.shape[2]
        if (
            side_condition.shape[2] != block_frames
            or block_frames != self._sfdmd_chunk_frames(global_start)
        ):
            raise ValueError(
                "SF-DMD student conditions do not match the rollout chunk: "
                f"hdmap={tuple(hdmap.shape)}, "
                f"side={tuple(side_condition.shape)}, "
                f"global_start={global_start}"
            )
        channels, height, width = (
            int(self.state_shape[0]),
            int(self.state_shape[2]),
            int(self.state_shape[3]),
        )
        noise = torch.randn(
            batch_size,
            channels,
            block_frames,
            height,
            width,
            device=hdmap.device,
            dtype=torch.float32,
        )
        latents = noise
        known_frames = 0
        if known_prefix_latents is not None:
            known_frames = int(known_prefix_latents.shape[2])
            if (
                known_prefix_latents.shape[:2] != latents.shape[:2]
                or known_prefix_latents.shape[3:] != latents.shape[3:]
                or known_frames <= 0
                or known_frames >= block_frames
            ):
                raise ValueError(
                    "mixed SF-DMD blocks require a non-empty proper latent "
                    "prefix aligned with the active block"
                )
            latents[:, :, :known_frames] = known_prefix_latents.float()
        if known_mask is None:
            known_mask = latents.new_zeros(block_frames)
            known_mask[:known_frames] = 1.0
        else:
            known_mask = known_mask.to(
                device=latents.device, dtype=torch.float32
            )
        if known_mask.shape != (block_frames,) or int(
            known_mask.sum().item()
        ) != known_frames:
            raise ValueError(
                "student known mask must identify the contiguous clean "
                f"prefix, got mask={known_mask.tolist()}, "
                f"known_frames={known_frames}"
            )
        if history_commit_latents is not None and (
            history_commit_latents.shape != latents.shape
        ):
            raise ValueError(
                "history commit latents must match the sampled block: "
                f"commit={tuple(history_commit_latents.shape)}, "
                f"sample={tuple(latents.shape)}"
            )
        if self.sfdmd_geometry_source:
            if geometry_source is None or geometry_confidence is None:
                raise RuntimeError(
                    "SF-DMD geometry-source sampling requires source/confidence"
                )
            if geometry_source.shape != noise.shape or (
                geometry_confidence.shape
                != (batch_size, 1, *noise.shape[2:])
            ):
                raise ValueError(
                    "SF-DMD geometry source/confidence shape mismatch: "
                    f"source={tuple(geometry_source.shape)}, "
                    f"confidence={tuple(geometry_confidence.shape)}, "
                    f"noise={tuple(noise.shape)}"
                )
            latents = confidence_weighted_geometry_source(
                noise,
                geometry_source,
                geometry_confidence,
                self.sfdmd_geometry_source_noise_floor,
            )
        sigmas = self._sfdmd_sigmas_for_block(
            block_index, device=latents.device
        )
        replay_step = (
            len(sigmas) - 2
            if self.sfdmd_imaginaire_matching
            else exit_step
        )
        replay_xt = None
        replay_sigma = None
        for step_idx in range(len(sigmas) - 1):
            sigma = sigmas[step_idx]
            sigma_next = sigmas[step_idx + 1]
            if step_idx == replay_step:
                replay_xt = latents.detach().clone()
                replay_sigma = float(sigma)
            active_t = torch.full(
                (batch_size,),
                float(sigma * self.rectified_flow.num_train_timesteps),
                device=latents.device,
                dtype=torch.float32,
            )
            timesteps = active_t[:, None].expand(
                -1, block_frames
            ).clone()
            if known_frames:
                timesteps[:, :known_frames] = 0.0
            velocity = self._student_velocity(
                latents,
                timesteps,
                condition,
                hdmap,
                side_condition,
                kv_cache,
                global_start,
                store_kv=(
                    not self.sfdmd_imaginaire_matching
                    and not self.sfdmd_clean_x0_kv_commit
                    and history_commit_latents is None
                    and step_idx == exit_step
                ),
                known_mask=known_mask,
            )
            x0 = rf_velocity_to_x0(latents, velocity, sigma)
            if known_frames:
                x0[:, :, :known_frames] = known_prefix_latents.float()
            if self.sfdmd_artifixer_matching and step_idx == exit_step:
                latents = x0
                break
            if float(sigma_next) > 0.0:
                eps_fresh = torch.randn_like(latents)
                transition_source = eps_fresh
                if self.sfdmd_artifixer_matching and self.sfdmd_geometry_source:
                    assert geometry_source is not None
                    assert geometry_confidence is not None
                    transition_source = confidence_weighted_geometry_source(
                        eps_fresh,
                        geometry_source,
                        geometry_confidence,
                        self.sfdmd_geometry_source_noise_floor,
                    )
                if (
                    self.sfdmd_artifixer_matching
                    or self.sfdmd_imaginaire_matching
                ):
                    # Matched recipes redraw their source endpoint every step.
                    next_latents = (
                        (1.0 - sigma_next) * x0
                        + sigma_next * transition_source
                    )
                else:
                    next_latents = sde_renoise(
                        x0=x0,
                        eps_implied=velocity + x0,
                        eps_fresh=eps_fresh,
                        sigma_next=sigma_next,
                        eta=self.sfdmd_sde_eta,
                    )
            else:
                next_latents = x0
            if known_frames:
                next_latents[:, :, :known_frames] = (
                    known_prefix_latents.float()
                )
            latents = next_latents

        assert replay_xt is not None and replay_sigma is not None
        if (
            self.sfdmd_imaginaire_matching
            or self.sfdmd_clean_x0_kv_commit
            or history_commit_latents is not None
        ):
            self._seed_student_clean_history(
                (
                    latents
                    if history_commit_latents is None
                    else history_commit_latents.float()
                ),
                condition,
                hdmap,
                side_condition,
                kv_cache,
                global_start,
                known_mask=known_mask,
            )
        record = SFDMDReplayRecord(
            block_index=block_index,
            window_start=window_start,
            global_start=global_start,
            score_active_start=score_active_start,
            score_active_end=(
                score_active_start + block_frames - known_frames
            ),
            replay_xt=replay_xt,
            replay_sigma=replay_sigma,
            student_condition=condition,
            student_hdmap=hdmap,
            student_side_condition=side_condition,
            student_known_mask=known_mask.detach().clone(),
            student_active_start=known_frames,
            student_active_end=block_frames,
            generated_block=latents.detach().clone(),
            history_commit_block=(
                None
                if history_commit_latents is None
                else history_commit_latents.detach().clone()
            ),
        )
        return latents.detach(), record

    @torch.no_grad()
    def rollout_sfdmd(
        self, data_batch: dict[str, Any], cycle: int
    ) -> SFDMDLongRollout:
        del cycle
        video = data_batch["video"]
        hdmap = data_batch["lidar_hdmap"]
        self._set_sfdmd_dataset_ids(data_batch, video)
        self._configure_warp_geometry(data_batch, video)
        poses = torch.as_tensor(
            data_batch["lidar_to_world"],
            device=video.device,
            dtype=torch.float32,
        )
        if video.shape[0] != 1:
            raise ValueError(
                f"SF-DMD currently requires batch_size=1, got {video.shape[0]}"
            )
        total_tokens = self._sfdmd_total_tokens()
        if video.shape[2] != total_tokens:
            raise ValueError(
                f"expected model-aligned video T={total_tokens}, "
                f"got {video.shape[2]}"
            )
        if tuple(poses.shape[1:]) != (
            self.sfdmd_num_real_frames,
            4,
            4,
        ):
            raise ValueError(
                "lidar_to_world must contain one pose per real frame, got "
                f"{tuple(poses.shape)}"
            )

        window = self.sfdmd_window_frames
        window_tokens = self._sfdmd_window_tokens()
        chunks = self._sfdmd_chunk_slices(total_tokens)
        num_blocks = len(chunks)
        factorized_hdmap_mode = "full"
        factorized_geometry_mode = "rolling_static"
        factorized_geometry_observations = 0
        factorized_geometry_source_mask = torch.zeros(
            total_tokens, device=video.device, dtype=torch.float32
        )
        if self.sfdmd_factorized_conditioning:
            (
                conditioned_prefix_frames,
                factorized_hdmap_mode,
                factorized_geometry_mode,
                factorized_geometry_observations,
                factorized_geometry_source_mask,
            ) = self._sample_factorized_rollout_modes(
                num_frames=total_tokens,
                device=video.device,
            )
        else:
            conditioned_prefix_frames = (
                self._sample_imaginaire_conditioned_prefix(
                    num_frames=total_tokens,
                    device=video.device,
                )
            )
        if any(
            start < conditioned_prefix_frames < end
            for start, end in chunks
        ) and not self.sfdmd_factorized_conditioning:
            raise NotImplementedError(
                "Imaginaire SF-DMD does not support a chunk that mixes "
                "conditioned and generated frames"
            )
        kv_seed_frames = max(
            (
                end
                for _, end in chunks
                if end <= conditioned_prefix_frames
            ),
            default=0,
        )
        exit_step = self._sample_rollout_exit_step(video.device)
        channels, _, height, width = map(int, self.state_shape)
        global_z = torch.zeros(
            1,
            channels,
            total_tokens,
            height,
            width,
            device=video.device,
            dtype=torch.float32,
        )
        global_side_z = None
        global_geometry_z = None
        global_geometry_confidence_z = None
        global_decoded = torch.full(
            (
                1,
                video.shape[1],
                total_tokens,
                video.shape[-2],
                video.shape[-1],
            ),
            (
                self._warp_min_value
                if self.sfdmd_factorized_conditioning
                else 0.0
            ),
            device=video.device,
            dtype=torch.float32,
        )
        ground_truth_z = None
        if self.sfdmd_temporal_history_source == "ground_truth":
            ground_truth_z = self.encode(video.float()).float()
            if ground_truth_z.shape != global_z.shape:
                raise RuntimeError(
                    "ground-truth history VAE encode returned the wrong shape: "
                    f"{tuple(ground_truth_z.shape)} != {tuple(global_z.shape)}"
                )
        conditioned_prefix_z = None
        if conditioned_prefix_frames > 0:
            conditioned_prefix_z = (
                self.encode(
                    video[:, :, :conditioned_prefix_frames].float()
                ).float()
                if ground_truth_z is None
                else ground_truth_z[:, :, :conditioned_prefix_frames]
            )
            expected_shape = (
                global_z.shape[0],
                global_z.shape[1],
                conditioned_prefix_frames,
                global_z.shape[3],
                global_z.shape[4],
            )
            if conditioned_prefix_z.shape != expected_shape:
                raise RuntimeError(
                    "conditioned prefix VAE encode returned the wrong shape: "
                    f"{tuple(conditioned_prefix_z.shape)} != "
                    f"{expected_shape}"
                )
            if self.sfdmd_factorized_conditioning:
                global_decoded[:, :, :conditioned_prefix_frames] = video[
                    :, :, :conditioned_prefix_frames
                ].float()
        records: list[SFDMDReplayRecord] = []
        conditioned_prefix_condition = None
        conditioned_prefix_hdmap = None
        conditioned_prefix_side = None
        vae_cache = None
        condition_vae_kv_cache = None
        geometry_vae_kv_cache = None
        student_kv_cache = self.new_student_kv_cache()
        global_hdmap_z = None
        spatial_maps = (
            self._new_spatial_maps(video.shape[0], video.device)
            if (
                self.sfdmd_accumulated_geometry_warp
                and not self.sfdmd_factorized_conditioning
            )
            else None
        )
        global_warp = (
            video[:, :1].new_full(
                (
                    video.shape[0],
                    1,
                    total_tokens,
                    video.shape[-2],
                    video.shape[-1],
                ),
                self._warp_min_value,
            )
            if (
                self.sfdmd_accumulated_geometry_warp
                and not self.sfdmd_factorized_conditioning
            )
            else None
        )
        global_geometry_source = (
            video[:, :1].new_full(
                (
                    video.shape[0],
                    1,
                    total_tokens,
                    video.shape[-2],
                    video.shape[-1],
                ),
                self._warp_min_value,
            )
            if self.sfdmd_geometry_source
            else None
        )
        global_geometry_confidence = (
            video[:, :1].new_zeros(
                (
                    video.shape[0],
                    1,
                    total_tokens,
                    video.shape[-2],
                    video.shape[-1],
                ),
                dtype=torch.float32,
            )
            if self.sfdmd_geometry_source
            else None
        )
        seed_score_condition = None
        seed_score_hdmap = None
        fixed_factorized_geometry = None
        if self.sfdmd_factorized_conditioning and (
            factorized_geometry_mode != "rolling_static"
            or self.sfdmd_geometry_history_source == "ground_truth"
        ):
            fixed_factorized_geometry = (
                self._build_factorized_rollout_geometry(
                    video,
                    data_batch,
                    poses,
                    factorized_geometry_mode,
                    factorized_geometry_source_mask,
                )
            )
        seed_blocks = (
            num_blocks
            if self.sfdmd_imaginaire_matching
            else len(self._sfdmd_chunk_slices(window_tokens))
        )

        for block_index, (chunk_start, chunk_end) in enumerate(chunks):
            if self.sfdmd_imaginaire_matching:
                global_start = chunk_start
                global_end = chunk_end
                window_start = 0
                score_active_start = chunk_start
                score_active_end = chunk_end
            else:
                (
                    global_start,
                    window_start,
                    score_active_start,
                    score_active_end,
                ) = sfdmd_nonoverlap_layout(
                    block_index,
                    block_size=self.df_block_size,
                    score_window_tokens=window_tokens,
                    prefix_frames=self.sfdmd_temporal_prefix_frames,
                )
                if global_start != chunk_start:
                    raise RuntimeError(
                        "SF-DMD chunk/layout disagreement: "
                        f"chunk={chunk_start}:{chunk_end}, "
                        f"layout_start={global_start}"
                    )
                global_end = chunk_end
            wb = dict(data_batch)
            wb["video"] = self._window_clip(
                video, window_start, window
            )
            wb["lidar_hdmap"] = self._window_clip(
                hdmap, window_start, window
            )
            factorized_window_geometry = None
            if self.sfdmd_factorized_conditioning:
                factorized_geometry = fixed_factorized_geometry
                if (
                    factorized_geometry_mode == "rolling_static"
                    and self.sfdmd_geometry_history_source == "generated"
                ):
                    factorized_geometry = (
                        self._build_factorized_rollout_geometry(
                            global_decoded,
                            data_batch,
                            poses,
                            factorized_geometry_mode,
                            factorized_geometry_source_mask,
                        )
                    )
                assert factorized_geometry is not None
                factorized_window_geometry = self._window_clip(
                    factorized_geometry,
                    window_start,
                    window,
                )

            if block_index == 0:
                decoded_prefix = None
            elif block_index < seed_blocks:
                decoded_prefix = global_decoded[
                    :, :, :global_start
                ].contiguous()
            else:
                first_real_token = window_start + 1
                prefix_real_frames = score_active_start - 1
                decoded_prefix = torch.cat(
                    [
                        global_decoded[
                            :,
                            :,
                            first_real_token : first_real_token + 1,
                        ],
                        global_decoded[
                            :,
                            :,
                            first_real_token : (
                                first_real_token + prefix_real_frames
                            ),
                        ],
                    ],
                    dim=2,
                )

            active = slice(
                score_active_start,
                score_active_end,
            )
            # Score networks consume their bidirectional training layout. The
            # student sees one active block and carries clean history in K/V.
            if self.sfdmd_stream_condition_vae_encode:
                (
                    condition,
                    student_hdmap,
                    student_side,
                    student_geometry,
                    student_geometry_confidence,
                    condition_vae_kv_cache,
                    geometry_vae_kv_cache,
                ) = self._prepare_stream_stage_context(
                    wb,
                    poses,
                    window_start,
                    global_start,
                    window_tokens,
                    decoded_prefix,
                    spatial_maps,
                    global_warp,
                    global_geometry_source,
                    global_geometry_confidence,
                    active,
                    condition_vae_kv_cache,
                    geometry_vae_kv_cache,
                    fixed_warp=factorized_window_geometry,
                )
                if self.sfdmd_factorized_conditioning:
                    if factorized_hdmap_mode == "null":
                        student_hdmap.zero_()
                    if factorized_geometry_mode == "null":
                        student_side.zero_()
                if block_index == 0 or block_index >= seed_blocks:
                    score_condition = condition
                    score_hdmap = self._encode_score_hdmap(
                        wb["lidar_hdmap"]
                    )
                    if (
                        self.sfdmd_factorized_conditioning
                        and factorized_hdmap_mode == "null"
                    ):
                        score_hdmap.zero_()
                else:
                    assert seed_score_condition is not None
                    assert seed_score_hdmap is not None
                    score_condition = seed_score_condition
                    score_hdmap = seed_score_hdmap
                if block_index == 0:
                    seed_score_condition = score_condition
                    seed_score_hdmap = score_hdmap
            else:
                if block_index == 0 or block_index >= seed_blocks:
                    (
                        score_condition,
                        score_hdmap,
                        score_side,
                        score_geometry,
                        score_geometry_confidence,
                    ) = self._prepare_stage_context(
                        wb,
                        poses,
                        window_start,
                        global_start,
                        window_tokens,
                        decoded_prefix,
                        spatial_maps,
                        global_warp,
                        global_geometry_source,
                        global_geometry_confidence,
                        fixed_warp=factorized_window_geometry,
                        block_frames=global_end - global_start,
                    )
                    if block_index == 0:
                        seed_score_condition = score_condition
                        seed_score_hdmap = score_hdmap
                    condition, hdmap_z, side_z, geometry_z, confidence_z = (
                        score_condition,
                        score_hdmap,
                        score_side,
                        score_geometry,
                        score_geometry_confidence,
                    )
                else:
                    assert seed_score_condition is not None
                    assert seed_score_hdmap is not None
                    (
                        condition,
                        hdmap_z,
                        side_z,
                        geometry_z,
                        confidence_z,
                    ) = (
                        self._prepare_stage_context(
                            wb,
                            poses,
                            window_start,
                            global_start,
                            window_tokens,
                            decoded_prefix,
                            spatial_maps,
                            global_warp,
                            global_geometry_source,
                            global_geometry_confidence,
                            block_frames=global_end - global_start,
                            fixed_condition=seed_score_condition,
                            fixed_hdmap=seed_score_hdmap,
                            fixed_warp=factorized_window_geometry,
                        )
                    )
                if self.sfdmd_factorized_conditioning:
                    if factorized_hdmap_mode == "null":
                        hdmap_z.zero_()
                    if factorized_geometry_mode == "null":
                        side_z.zero_()
                student_hdmap = hdmap_z[:, :, active].contiguous()
                student_side = side_z[:, :, active].contiguous()
                student_geometry = (
                    None
                    if geometry_z is None
                    else geometry_z[:, :, active].contiguous()
                )
                student_geometry_confidence = (
                    None
                    if confidence_z is None
                    else confidence_z[:, :, active].contiguous()
                )

            if global_side_z is None:
                global_side_z = student_side.new_zeros(
                    (
                        student_side.shape[0],
                        student_side.shape[1],
                        total_tokens,
                        student_side.shape[3],
                        student_side.shape[4],
                    ),
                    dtype=torch.float32,
                )
                global_hdmap_z = student_hdmap.new_zeros(
                    (
                        student_hdmap.shape[0],
                        student_hdmap.shape[1],
                        total_tokens,
                        student_hdmap.shape[3],
                        student_hdmap.shape[4],
                    ),
                    dtype=torch.float32,
                )
            global_side_z[:, :, global_start:global_end] = (
                student_side.float()
            )
            assert global_hdmap_z is not None
            global_hdmap_z[:, :, global_start:global_end] = (
                student_hdmap.float()
            )
            if self.sfdmd_geometry_source:
                if (
                    student_geometry is None
                    or student_geometry_confidence is None
                ):
                    raise RuntimeError(
                        "geometry-source rollout did not produce latent source "
                        "and confidence"
                    )
                if global_geometry_z is None:
                    global_geometry_z = student_geometry.new_zeros(
                        (
                            student_geometry.shape[0],
                            student_geometry.shape[1],
                            total_tokens,
                            student_geometry.shape[3],
                            student_geometry.shape[4],
                        ),
                        dtype=torch.float32,
                    )
                    global_geometry_confidence_z = (
                        student_geometry_confidence.new_zeros(
                            (
                                student_geometry_confidence.shape[0],
                                1,
                                total_tokens,
                                student_geometry_confidence.shape[3],
                                student_geometry_confidence.shape[4],
                            ),
                            dtype=torch.float32,
                        )
                    )
                global_geometry_z[:, :, global_start:global_end] = (
                    student_geometry.float()
                )
                assert global_geometry_confidence_z is not None
                global_geometry_confidence_z[
                    :, :, global_start:global_end
                ] = student_geometry_confidence.float()

            record = None
            if global_end <= conditioned_prefix_frames:
                if conditioned_prefix_z is None:
                    raise RuntimeError(
                        "conditioned SF-DMD chunk has no clean latent"
                    )
                sampled = conditioned_prefix_z[
                    :, :, global_start:global_end
                ].contiguous()
                self._seed_student_clean_history(
                    sampled,
                    condition,
                    student_hdmap,
                    student_side,
                    student_kv_cache,
                    global_start,
                    known_mask=sampled.new_ones(sampled.shape[2]),
                )
                conditioned_prefix_condition = condition
                conditioned_prefix_hdmap = student_hdmap.detach().clone()
                conditioned_prefix_side = student_side.detach().clone()
            else:
                known_in_block = max(
                    0,
                    min(
                        global_end,
                        conditioned_prefix_frames,
                    )
                    - global_start,
                )
                block_known_mask = video.new_zeros(
                    global_end - global_start,
                    dtype=torch.float32,
                )
                block_known_mask[:known_in_block] = 1.0
                sampled, record = self._sample_stage(
                    block_index=block_index,
                    window_start=window_start,
                    global_start=global_start,
                    score_active_start=(
                        score_active_start + known_in_block
                    ),
                    condition=condition,
                    hdmap=student_hdmap,
                    side_condition=student_side,
                    geometry_source=student_geometry,
                    geometry_confidence=student_geometry_confidence,
                    kv_cache=student_kv_cache,
                    exit_step=exit_step,
                    known_prefix_latents=(
                        None
                        if known_in_block == 0
                        else conditioned_prefix_z[
                            :,
                            :,
                            global_start : global_start + known_in_block,
                        ].contiguous()
                    ),
                    known_mask=block_known_mask,
                    history_commit_latents=(
                        None
                        if ground_truth_z is None
                        else ground_truth_z[
                            :, :, global_start:global_end
                        ].contiguous()
                    ),
                )
            global_z[:, :, global_start:global_end] = sampled
            decoded_new, vae_cache = self.decode_stream(
                sampled, vae_cache
            )
            if decoded_new.ndim == 4:
                decoded_new = decoded_new.unsqueeze(2)
            if decoded_new.shape[2] != global_end - global_start:
                raise RuntimeError(
                    "streaming VAE returned the wrong block length: "
                    f"{tuple(decoded_new.shape)}"
                )
            global_decoded[:, :, global_start:global_end] = (
                decoded_new.float()
            )
            if (
                self.sfdmd_factorized_conditioning
                and global_start < conditioned_prefix_frames
            ):
                known_end = min(global_end, conditioned_prefix_frames)
                global_decoded[:, :, global_start:known_end] = video[
                    :, :, global_start:known_end
                ].float()
            if spatial_maps is not None:
                self._append_generated_geometry(
                    spatial_maps,
                    decoded_new,
                    poses,
                    global_start,
                )
            if record is not None:
                record.conditioned_prefix_frames = (
                    conditioned_prefix_frames
                )
                if self.sfdmd_factorized_conditioning:
                    global_known_mask = video.new_zeros(
                        total_tokens, dtype=torch.float32
                    )
                    global_known_mask[:conditioned_prefix_frames] = 1.0
                    record.score_known_mask = (
                        self._factorized_known_mask_window(
                            global_known_mask,
                            window_start,
                        )
                    )
                record.score_condition = (
                    seed_score_condition
                    if block_index < seed_blocks
                    else score_condition
                )
                record.score_hdmap = (
                    seed_score_hdmap
                    if block_index < seed_blocks
                    else score_hdmap
                )
                records.append(record)

            if block_index == seed_blocks - 1:
                first_window = global_z[
                    :, :, :window_tokens
                ].detach().clone()
                assert global_side_z is not None
                first_side_window = global_side_z[
                    :, :, :window_tokens
                ].detach().clone()
                for seed_record in records:
                    seed_record.generated_window = first_window
                    seed_record.score_side_condition = first_side_window
            elif block_index >= seed_blocks:
                assert record is not None
                record.generated_window = self._window_clip(
                    global_z, window_start, window
                ).detach().clone()
                assert global_side_z is not None
                record.score_side_condition = self._window_clip(
                    global_side_z, window_start, window
                ).detach().clone()

        expected_records = sum(
            end > conditioned_prefix_frames for _, end in chunks
        )
        if len(records) != expected_records:
            raise RuntimeError(
                f"expected {expected_records} replay blocks, "
                f"got {len(records)}"
            )
        if any(record.generated_window is None for record in records):
            raise RuntimeError("SF-DMD rollout left score windows incomplete")
        if any(record.score_side_condition is None for record in records):
            raise RuntimeError(
                "SF-DMD rollout left score side conditions incomplete"
            )
        return SFDMDLongRollout(
            records=records,
            generated_latents=global_z.detach(),
            generated_real_latents=global_z[
                :, :, int(self.input_video_has_prepend) :
            ].detach(),
            num_real_frames=self.sfdmd_num_real_frames,
            exit_step=exit_step,
            conditioned_prefix_frames=conditioned_prefix_frames,
            kv_seed_frames=kv_seed_frames,
            conditioned_prefix_latents=(
                None
                if conditioned_prefix_z is None or kv_seed_frames == 0
                else conditioned_prefix_z[
                    :, :, :kv_seed_frames
                ].detach()
            ),
            conditioned_prefix_condition=conditioned_prefix_condition,
            conditioned_prefix_hdmap=(
                None
                if global_hdmap_z is None or kv_seed_frames == 0
                else global_hdmap_z[:, :, :kv_seed_frames].detach()
            ),
            conditioned_prefix_side_condition=(
                None
                if global_side_z is None or kv_seed_frames == 0
                else global_side_z[:, :, :kv_seed_frames].detach()
            ),
            geometry_source_latents=(
                None
                if global_geometry_z is None
                else global_geometry_z.detach()
            ),
            geometry_confidence_latents=(
                None
                if global_geometry_confidence_z is None
                else global_geometry_confidence_z.detach()
            ),
            factorized_hdmap_mode=(
                factorized_hdmap_mode
                if self.sfdmd_factorized_conditioning
                else ""
            ),
            factorized_geometry_mode=(
                factorized_geometry_mode
                if self.sfdmd_factorized_conditioning
                else ""
            ),
            factorized_geometry_observations=(
                factorized_geometry_observations
                if self.sfdmd_factorized_conditioning
                else 0
            ),
        )

    # ------------------------------------------------------------------
    # DMD losses
    # ------------------------------------------------------------------

    @staticmethod
    def _rollout_noisy_mask(
        rollout: SFDMDLongRollout,
        reference: Tensor,
    ) -> Tensor:
        mask = reference.new_ones(
            (1, 1, reference.shape[2], 1, 1),
            dtype=torch.float32,
        )
        mask[:, :, : rollout.conditioned_prefix_frames] = 0.0
        return mask

    def _sample_critic_sigma(self, batch_size: int) -> tuple[Tensor, Tensor]:
        if self.sfdmd_artifixer_matching:
            # ArtiFixer draws an integer base timestep, applies only the
            # rational RF shift, then clamps the resulting float timestep.
            # Do not use apply_shift_reparam(): the inherited student config
            # also enables LTX percentile stretching, which ArtiFixer omits.
            num_timesteps = int(self.rectified_flow.num_train_timesteps)
            base_timestep = torch.randint(
                0,
                num_timesteps,
                (batch_size,),
                device=self.tensor_kwargs_fp32["device"],
            ).to(dtype=torch.float32)
            raw_u = base_timestep / num_timesteps
            sigma = self._rf_shift * raw_u / (
                1.0 + (self._rf_shift - 1.0) * raw_u
            )
            timestep = (sigma * num_timesteps).clamp(
                int(self.sfdmd_critic_min_sigma * num_timesteps),
                int(self.sfdmd_critic_max_sigma * num_timesteps),
            )
            sigma = timestep / num_timesteps
            return sigma, timestep

        raw_u = self.rectified_flow.sample_train_time(batch_size).to(
            **self.tensor_kwargs_fp32
        )
        sigma = self.rectified_flow.apply_shift_reparam(raw_u)
        timestep = sigma * self.rectified_flow.num_train_timesteps
        return sigma, timestep

    def new_score_batch(self, rollout: SFDMDLongRollout) -> SFDMDScoreBatch:
        """Draw one critic timestep/source field for the complete trajectory."""
        sigma, timestep = self._sample_critic_sigma(
            rollout.generated_latents.shape[0]
        )
        noise = torch.randn_like(rollout.generated_latents)
        source = noise
        if self.sfdmd_artifixer_matching and self.sfdmd_geometry_source:
            if (
                rollout.geometry_source_latents is None
                or rollout.geometry_confidence_latents is None
            ):
                raise RuntimeError(
                    "geometry-source critic batch requires rollout geometry"
                )
            source = confidence_weighted_geometry_source(
                noise,
                rollout.geometry_source_latents,
                rollout.geometry_confidence_latents,
                self.sfdmd_geometry_source_noise_floor,
            )
        return SFDMDScoreBatch(
            sigma=sigma,
            timestep=timestep,
            source=source,
        )

    def _score_window_from_global(
        self, tensor: Tensor, record: SFDMDReplayRecord
    ) -> Tensor:
        if record.window_start == 0:
            return tensor[:, :, : self._sfdmd_window_tokens()]
        return self._window_clip(
            tensor,
            record.window_start,
            self.sfdmd_window_frames,
        )

    @staticmethod
    def _score_noisy_state(
        generated: Tensor,
        source: Tensor,
        sigma_5d: Tensor,
        record: SFDMDReplayRecord,
    ) -> tuple[Tensor, Tensor]:
        effective_sigma = sigma_5d
        if record.score_known_mask is not None:
            known = record.score_known_mask.to(
                device=generated.device, dtype=generated.dtype
            )
            if known.ndim == 1:
                known = known[None]
            if known.shape[0] == 1 and generated.shape[0] != 1:
                known = known.expand(generated.shape[0], -1)
            if known.shape != (generated.shape[0], generated.shape[2]):
                raise ValueError(
                    "score known mask must align with generated window"
                )
            effective_sigma = sigma_5d * (
                1.0 - known[:, None, :, None, None]
            )
        return (
            effective_sigma * source
            + (1.0 - effective_sigma) * generated,
            effective_sigma,
        )

    def fake_score_loss(
        self,
        rollout: SFDMDLongRollout,
        update_index: int,
        score_batch: SFDMDScoreBatch | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        if self.sfdmd_imaginaire_matching:
            record = rollout.records[0]
            assert record.score_hdmap is not None
            generated = rollout.generated_latents.float()
            if score_batch is None:
                score_batch = self.new_score_batch(rollout)
            sigma_5d = score_batch.sigma[:, None, None, None, None]
            noisy_mask = self._rollout_noisy_mask(rollout, generated)
            sigma_effective = sigma_5d * noisy_mask
            x_t = (
                sigma_effective * score_batch.source
                + (1.0 - sigma_effective) * generated
            )
            target_v = score_batch.source - generated
            fake_v = self._score_velocity(
                self.fake_score,
                x_t,
                score_batch.timestep,
                record.score_hdmap,
                record.score_side_condition,
                record.score_condition,
                record.conditioned_prefix_frames,
                record.score_known_mask,
            )
            squared_error = (fake_v - target_v).square()
            active_count = noisy_mask.expand_as(squared_error).sum().clamp_min(
                1.0
            )
            loss = (squared_error * noisy_mask).sum() / active_count
            return loss, {
                "fake_score_loss": loss.detach(),
                "critic_sigma": score_batch.sigma.mean().detach(),
                "critic_block": loss.new_tensor(-1.0),
                "conditioned_prefix_frames": loss.new_tensor(
                    float(rollout.conditioned_prefix_frames)
                ),
            }

        record = rollout.records[update_index % len(rollout.records)]
        assert record.generated_window is not None
        assert record.score_hdmap is not None
        generated = record.generated_window.float()
        if score_batch is None:
            score_batch = self.new_score_batch(rollout)
        sigma, timestep = score_batch.sigma, score_batch.timestep
        source = self._score_window_from_global(score_batch.source, record)
        sigma_5d = sigma[:, None, None, None, None]
        x_t, _ = self._score_noisy_state(
            generated, source, sigma_5d, record
        )
        target_v = source - generated
        fake_v = self._score_velocity(
            self.fake_score,
            x_t,
            timestep,
            record.score_hdmap,
            record.score_side_condition,
            record.score_condition,
            record.conditioned_prefix_frames,
            record.score_known_mask,
        )
        loss = active_slice_loss(
            fake_v,
            target_v,
            record.score_active_start,
            record.score_active_end,
        )
        return loss, {
            "fake_score_loss": loss.detach(),
            "critic_sigma": sigma.mean().detach(),
            "critic_block": loss.new_tensor(float(record.block_index)),
        }

    @torch.no_grad()
    def _guided_score_x0(
        self,
        score_model: nn.Module,
        x_t: Tensor,
        timestep: Tensor,
        sigma_5d: Tensor,
        record: SFDMDReplayRecord,
        guidance_override: float | None = None,
    ) -> Tensor:
        assert record.score_hdmap is not None
        cond_v = self._score_velocity(
            score_model,
            x_t,
            timestep,
            record.score_hdmap,
            record.score_side_condition,
            record.score_condition,
            record.conditioned_prefix_frames,
            record.score_known_mask,
        )
        guidance = (
            self.sfdmd_teacher_guidance
            if guidance_override is None
            else guidance_override
        )
        if guidance > 1.0:
            uncond_v = self._score_velocity(
                score_model,
                x_t,
                timestep,
                torch.zeros_like(record.score_hdmap),
                record.score_side_condition,
                record.score_condition,
                record.conditioned_prefix_frames,
                record.score_known_mask,
            )
            cond_v = uncond_v + guidance * (cond_v - uncond_v)
        return rf_velocity_to_x0(x_t, cond_v, sigma_5d)

    @torch.no_grad()
    def artifixer_student_targets(
        self, rollout: SFDMDLongRollout
    ) -> dict[int, tuple[Tensor, dict[str, Tensor]]]:
        """Build globally normalized DMD targets over the full generated rollout."""
        if self.sfdmd_imaginaire_matching:
            record = rollout.records[0]
            assert record.score_hdmap is not None
            generated = rollout.generated_latents.float()
            score_batch = self.new_score_batch(rollout)
            sigma_5d = score_batch.sigma[:, None, None, None, None]
            noisy_mask = self._rollout_noisy_mask(rollout, generated)
            sigma_effective = sigma_5d * noisy_mask
            x_t = (
                sigma_effective * score_batch.source
                + (1.0 - sigma_effective) * generated
            )
            fake_x0 = self._guided_score_x0(
                self.fake_score,
                x_t,
                score_batch.timestep,
                sigma_effective,
                record,
                guidance_override=1.0,
            )
            teacher_x0 = self._guided_score_x0(
                self.teacher_score,
                x_t,
                score_batch.timestep,
                sigma_effective,
                record,
            )
            normalizer_count = noisy_mask.expand_as(generated).sum(
                dim=(1, 2, 3, 4), keepdim=True
            ).clamp_min(1.0)
            normalizer = (
                (generated - teacher_x0).abs() * noisy_mask
            ).sum(
                dim=(1, 2, 3, 4), keepdim=True
            ) / normalizer_count + 1.0e-6
            vsd_grad = torch.nan_to_num(
                (fake_x0 - teacher_x0) / normalizer
            ) * noisy_mask
            pseudo_target = generated - vsd_grad
            targets = {}
            for replay_record in rollout.records:
                start = replay_record.global_start
                end = start + replay_record.replay_xt.shape[2]
                targets[replay_record.block_index] = (
                    pseudo_target[:, :, start:end].detach(),
                    {
                        "vsd_weight": normalizer.reciprocal().mean(),
                        "teacher_fake_x0_l1": (
                            teacher_x0[:, :, start:end]
                            - fake_x0[:, :, start:end]
                        ).abs().mean(),
                        "teacher_gen_x0_l1": (
                            teacher_x0[:, :, start:end]
                            - generated[:, :, start:end]
                        ).abs().mean(),
                        "global_vsd_normalizer": normalizer.mean(),
                        "critic_sigma": score_batch.sigma.mean(),
                    },
                )
            return targets

        score_batch = self.new_score_batch(rollout)
        sigma_5d = score_batch.sigma[:, None, None, None, None]
        generated_parts = []
        teacher_parts = []
        fake_parts = []
        for record in rollout.records:
            assert record.generated_window is not None
            generated = record.generated_window.float()
            source = self._score_window_from_global(
                score_batch.source, record
            )
            x_t, sigma_effective = self._score_noisy_state(
                generated,
                source,
                sigma_5d,
                record,
            )
            fake_x0 = self._guided_score_x0(
                self.fake_score,
                x_t,
                score_batch.timestep,
                sigma_effective,
                record,
            )
            teacher_x0 = self._guided_score_x0(
                self.teacher_score,
                x_t,
                score_batch.timestep,
                sigma_effective,
                record,
            )
            active = slice(
                record.score_active_start, record.score_active_end
            )
            generated_parts.append(generated[:, :, active])
            teacher_parts.append(teacher_x0[:, :, active])
            fake_parts.append(fake_x0[:, :, active])

        generated_all = torch.cat(generated_parts, dim=2)
        teacher_all = torch.cat(teacher_parts, dim=2)
        fake_all = torch.cat(fake_parts, dim=2)
        normalizer = (
            generated_all - teacher_all
        ).abs().mean(dim=(1, 2, 3, 4), keepdim=True) + 1.0e-6
        vsd_grad = torch.nan_to_num(
            (fake_all - teacher_all) / normalizer
        )
        pseudo_target = generated_all - vsd_grad

        targets = {}
        offset = 0
        for record in rollout.records:
            active_frames = (
                record.score_active_end - record.score_active_start
            )
            end = offset + active_frames
            target = pseudo_target[:, :, offset:end].detach()
            targets[record.block_index] = (
                target,
                {
                    "vsd_weight": (
                        normalizer.clamp_min(1.0e-12).reciprocal().mean()
                    ),
                    "teacher_fake_x0_l1": (
                        teacher_all[:, :, offset:end]
                        - fake_all[:, :, offset:end]
                    ).abs().mean(),
                    "teacher_gen_x0_l1": (
                        teacher_all[:, :, offset:end]
                        - generated_all[:, :, offset:end]
                    ).abs().mean(),
                    "global_vsd_normalizer": normalizer.mean(),
                    "critic_sigma": score_batch.sigma.mean(),
                },
            )
            offset = end
        return targets

    @torch.no_grad()
    def _student_pseudo_target(
        self, record: SFDMDReplayRecord
    ) -> tuple[Tensor, dict[str, Tensor]]:
        assert record.generated_window is not None
        assert record.score_hdmap is not None
        generated = record.generated_window.float()
        sigma, timestep = self._sample_critic_sigma(generated.shape[0])
        sigma_5d = sigma[:, None, None, None, None]
        epsilon = torch.randn_like(generated)
        x_t, sigma_effective = self._score_noisy_state(
            generated, epsilon, sigma_5d, record
        )
        fake_v = self._score_velocity(
            self.fake_score,
            x_t,
            timestep,
            record.score_hdmap,
            record.score_side_condition,
            record.score_condition,
            record.conditioned_prefix_frames,
            record.score_known_mask,
        )
        teacher_v = self._score_velocity(
            self.teacher_score,
            x_t,
            timestep,
            record.score_hdmap,
            record.score_side_condition,
            record.score_condition,
            record.conditioned_prefix_frames,
            record.score_known_mask,
        )
        fake_x0 = rf_velocity_to_x0(x_t, fake_v, sigma_effective)
        teacher_x0 = rf_velocity_to_x0(
            x_t, teacher_v, sigma_effective
        )
        active = slice(
            record.score_active_start, record.score_active_end
        )
        target_active, weight = dmd_pseudo_target(
            generated[:, :, active],
            teacher_x0[:, :, active],
            fake_x0[:, :, active],
        )
        return target_active, {
            "vsd_weight": weight.mean(),
            "teacher_fake_x0_l1": (
                teacher_x0[:, :, active] - fake_x0[:, :, active]
            ).abs().mean(),
            "teacher_gen_x0_l1": (
                teacher_x0[:, :, active] - generated[:, :, active]
            ).abs().mean(),
        }

    def student_replay_loss(
        self,
        record: SFDMDReplayRecord,
        kv_cache: list[dict[str, Tensor] | None],
        target_payload: tuple[Tensor, dict[str, Tensor]] | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        if target_payload is None:
            if (
                self.sfdmd_artifixer_matching
                or self.sfdmd_imaginaire_matching
            ):
                raise RuntimeError(
                    "matched SF-DMD replay requires the trajectory-wide "
                    "precomputed target payload"
                )
            target_active, metrics = self._student_pseudo_target(record)
        else:
            target_active, metrics = target_payload
        replay_sigma = record.replay_xt.new_tensor(record.replay_sigma)
        active_t = replay_sigma * self.rectified_flow.num_train_timesteps
        timesteps = torch.full(
            (
                record.replay_xt.shape[0],
                record.replay_xt.shape[2],
            ),
            float(active_t),
            device=record.replay_xt.device,
            dtype=torch.float32,
        )
        if record.student_known_mask is not None:
            known = record.student_known_mask.to(
                device=timesteps.device, dtype=torch.bool
            )
            timesteps[:, known] = 0.0
        velocity = self._student_velocity(
            record.replay_xt,
            timesteps,
            record.student_condition,
            record.student_hdmap,
            record.student_side_condition,
            kv_cache,
            record.global_start,
            store_kv=(
                not self.sfdmd_imaginaire_matching
                and not self.sfdmd_clean_x0_kv_commit
                and record.history_commit_block is None
            ),
            detach_kv=self.sfdmd_imaginaire_matching,
            known_mask=record.student_known_mask,
        )
        x0_pred = rf_velocity_to_x0(
            record.replay_xt,
            velocity,
            replay_sigma,
        )
        if record.history_commit_block is not None:
            self._seed_student_clean_history(
                record.history_commit_block,
                record.student_condition,
                record.student_hdmap,
                record.student_side_condition,
                kv_cache,
                record.global_start,
                detach_kv=False,
                known_mask=record.student_known_mask,
            )
        elif self.sfdmd_clean_x0_kv_commit:
            clean_commit = x0_pred
            if (
                record.student_active_start > 0
                and record.generated_block is not None
            ):
                clean_commit = x0_pred.clone()
                clean_commit[
                    :, :, : record.student_active_start
                ] = record.generated_block[
                    :, :, : record.student_active_start
                ]
            self._seed_student_clean_history(
                clean_commit,
                record.student_condition,
                record.student_hdmap,
                record.student_side_condition,
                kv_cache,
                record.global_start,
                detach_kv=False,
                known_mask=record.student_known_mask,
            )
        elif self.sfdmd_imaginaire_matching:
            if record.generated_block is None:
                raise RuntimeError(
                    "Imaginaire replay record has no finalized clean block"
                )
            with torch.no_grad():
                self._seed_student_clean_history(
                    record.generated_block,
                    record.student_condition,
                    record.student_hdmap,
                    record.student_side_condition,
                    kv_cache,
                    record.global_start,
                )
        if self.sfdmd_imaginaire_matching:
            per_frame_loss = (
                0.5
                * (x0_pred.float() - target_active.float()).square()
            ).mean(dim=(0, 1, 3, 4))
            loss = per_frame_loss.sum()
        elif self.sfdmd_artifixer_matching:
            # Match ArtiFixer's FP64 pseudo-target MSE. Casting the scalar back
            # is unnecessary; autograd propagates through the double cast.
            active_start = record.student_active_start
            active_end = (
                x0_pred.shape[2]
                if record.student_active_end < 0
                else record.student_active_end
            )
            active_x0 = x0_pred[:, :, active_start:active_end]
            loss = 0.5 * F.mse_loss(
                active_x0.double(),
                target_active.double(),
                reduction="mean",
            )
        else:
            loss = active_slice_loss(
                x0_pred,
                target_active,
                0,
                x0_pred.shape[2],
                half=True,
            )
        metrics.update(
            {
                "student_vsd_loss": loss.detach(),
                "student_block": loss.new_tensor(float(record.block_index)),
                "student_x0_abs": x0_pred.abs().mean().detach(),
            }
        )
        return loss, metrics


@diffusion_fsdp_class_decorator
class _FSDPHdmapToLidarSFDMDModel(HdmapToLidarSFDMDModel):
    pass


class FSDPHdmapToLidarSFDMDModel(_FSDPHdmapToLidarSFDMDModel):
    """Named public FSDP class used by Hydra experiment configs."""

    pass


class HdmapToLidarImaginaireSFDMDModel(HdmapToLidarSFDMDModel):
    """Explicit entry point for the Imaginaire-matched SF-DMD recipe."""

    def __init__(self, config):
        super().__init__(config)
        if not self.sfdmd_imaginaire_matching:
            raise ValueError(
                "Imaginaire SF-DMD model requires "
                "sfdmd_imaginaire_matching=True"
            )


@diffusion_fsdp_class_decorator
class _FSDPHdmapToLidarImaginaireSFDMDModel(
    HdmapToLidarImaginaireSFDMDModel
):
    pass


class FSDPHdmapToLidarImaginaireSFDMDModel(
    _FSDPHdmapToLidarImaginaireSFDMDModel
):
    pass
