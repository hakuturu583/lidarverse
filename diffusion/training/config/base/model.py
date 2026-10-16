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

from typing import List, Optional

import attrs

from diffusion.training.config.base.ema import PowerEMAConfig
from diffusion.training.modules.edm_sde import EDMSDE
from utils.lazy_config import LazyCall as L
from utils.lazy_config import LazyDict


@attrs.define(slots=False)
class FSDPConfig:
    policy: str = "block"
    checkpoint: bool = False
    min_num_params: int = 1024
    sharding_group_size: int = 8
    sharding_strategy: str = "full"
    use_orig_params: bool = False


@attrs.define(slots=False)
class DefaultModelConfig:
    vae: LazyDict = None
    conditioner: LazyDict = None
    net: LazyDict = None
    ema: LazyDict = PowerEMAConfig
    sde: LazyDict = L(EDMSDE)(
        p_mean=0.0,
        p_std=1.0,
        sigma_max=80,
        sigma_min=0.0002,
    )
    sigma_data: float = 0.5
    camera_sample_weight: LazyDict = LazyDict(
        dict(
            enabled=False,
            weight=5.0,
        )
    )
    aesthetic_finetuning: LazyDict = LazyDict(
        dict(
            enabled=False,
        )
    )
    loss_mask_enabled: bool = False
    loss_masking: LazyDict = None
    loss_add_logvar: bool = True
    precision: str = "bfloat16"
    input_data_key: str = "video"  # key to fetch input data from data_batch
    input_image_key: str = "images_1024"  # key to fetch input image from data_batch
    loss_reduce: str = "sum"
    loss_scale: float = 1.0
    latent_shape: List[int] = [16, 24, 44, 80]  # 24 corresponig to 136 frames
    fsdp_enabled: bool = False
    use_torch_compile: bool = False
    fsdp: FSDPConfig = attrs.field(factory=FSDPConfig)
    use_dummy_temporal_dim: bool = False  # Whether to use dummy temporal dimension in data
    adjust_video_noise: bool = False  # whether or not adjust video noise accroding to the video length
    peft_control: LazyDict | None = None
    num_latents_to_drop: int = 0  # number of P-latents to discard after encoding


@attrs.define(slots=False)
class MultiviewModelConfig(DefaultModelConfig):
    n_views: int = 6


@attrs.define(slots=False)
class LidarModelConfig(DefaultModelConfig):
    # Optional second tokenizer for an aligned per-return LiDAR attribute.
    # The primary VAE continues to encode range and HDMap inputs; when present,
    # this VAE encodes only the intensity channel.
    intensity_vae: LazyDict = None
    n_views: int = 4
    rgb_latent_shape: List[int] = [88, 160]
    lidar_latent_shape: List[int] = [64, 224]
    rgb_shape: List[int] = [704, 1280]
    lidar_shape: List[int] = [512, 1792]
    # Image-conditioned mininet RF (ImageToLidarMininetRectifiedFlowModel):
    # number of RGB conditioning views temporal-concatenated onto the lidar
    # latent, their dataset view keys, and the RGB image-VAE latent channel
    # count (zero-padded up to the lidar latent ch before the temporal stack).
    n_rgb_views: int = 0
    rgb_view_keys: List[int] = attrs.field(factory=list)
    image_latent_ch: int = 16
    # Classifier-free guidance on the RGB condition: per-sample probability of
    # zeroing ALL RGB condition frames during training (learns the RGB-
    # unconditional branch). 0.0 = always conditional (no image CFG).
    rgb_dropout_prob: float = 0.0
    # VIDEO image-conditioned mininet RF (VideoImageToLidarMininetRectifiedFlowModel):
    # per-frame counts on the joint temporal axis, RGB pixel-time stride (RGB
    # temporal compression / lidar temporal compression), RGB input resize
    # (HxW) before the Wan encode, and the per-sample probability of dropping a
    # random suffix of RGB future frames (lidar extrapolation training).
    lidar_latent_t: int = 1
    rgb_latent_t: int = 1
    rgb_temporal_stride: int = 4
    rgb_resize_hw: List[int] = attrs.field(factory=lambda: [256, 448])
    # Encode each RGB frame as an independent T=1 clip (no VAE temporal
    # compression) -> Tr == #RGB input frames -> 1:1 with lidar frames.
    rgb_per_frame_encode: bool = False
    # Prepend a duplicate of RGB latent frame 0 per view (mirrors the lidar VAE
    # frame-0 prepend) -> Tr+1 RGB frames matching lidar Tl, exact 1:1 RoPE
    # pixel-time alignment (no off-by-one).
    rgb_prepend_first_frame: bool = False
    # PV-VAE-style RGB future-frame dropping (mirrors the lidar VAE's
    # frame_drop_{prob,min_latent,max_latent}): with prob ``future_frame_drop_prob``
    # drop the trailing ``k ~ U[min, max]`` RGB latent frames (same cut across
    # views). ``max < 0`` => Tr-1 (keep only the first frame); ``min=0`` =>
    # sometimes keep all. 0 prob = no future-frame dropping.
    future_frame_drop_prob: float = 0.0
    future_frame_drop_min: int = 0
    future_frame_drop_max: int = -1
    # Cross-attn RGB conditioning: spatial avg-pool factor applied to the RGB
    # latent HxW before tokenizing into image-context K/V (keeps h/w structure,
    # trims token count). 1 = no pooling.
    rgb_context_pool: int = 1
    hdmap_channels: int = 0
    additional_concat_ch: int = 0
    skip_hdmap_proj: bool = False
    encode_hdmap_with_vae: bool = False
    # BEV-native hdmap conditioning: pool the multi-channel BEV hdmap raster
    # (one channel per semantic class, solid box footprints) down to the latent
    # grid with a learnable conv encoder and channel-concat it. Mutually
    # exclusive with encode_hdmap_with_vae / skip_hdmap_proj.
    bev_hdmap_pool: bool = False
    bev_hdmap_channels: int = 7
    bev_hdmap_size: int = 256
    bev_hdmap_out_size: int = 32
    bev_hdmap_base_channels: int = 64
    # BEV-native hdmap conditioning via a FROZEN pretrained Wan2.2 RGB VAE:
    # render the BEV raster to a 3-ch image and encode it (16x spatial, z=48),
    # then channel-concat that latent (no learnable projection). Mutually
    # exclusive with bev_hdmap_pool / encode_hdmap_with_vae / skip_hdmap_proj.
    # ``bev_hdmap_render_mode``: "colored" (per-class palette) or "occupancy".
    bev_hdmap_wan_encode: bool = False
    bev_hdmap_wan_ckpt: str = "checkpoints/wan_vae_pretrained/Wan2.2_VAE.pth"
    bev_hdmap_render_mode: str = "colored"
    hdmap_dropout_prob: float = 0.0
    # Mixture of training-time hdmap dropout regimes. When dropout fires for
    # a sample (prob = hdmap_dropout_prob), one mode from this list is drawn
    # uniformly:
    #   "full"            -- zero the entire hdmap latent (CFG-style)
    #   "frame_bernoulli" -- independently zero each frame at
    #                        hdmap_frame_dropout_rate
    #   "keep_first_k"    -- zero all frames except the first
    #                        hdmap_keep_first_k frames
    # Default = ("full",) preserves the legacy CFG-only behavior, so all
    # pre-existing configs train exactly as before. Only video-mode hdmap
    # latents (5-D, with a temporal axis) honor frame-wise modes; image-mode
    # latents (4-D, T=1) fall back to "full" silently.
    hdmap_dropout_modes: List[str] = attrs.field(factory=lambda: ["full"])
    hdmap_frame_dropout_rate: float = 0.5
    hdmap_keep_first_k: int = 1
    # ArtiFixer stage-2 conditioning dropout. One decision is shared across the
    # batch: with probability ``hdmap_dropout_prob`` drop every latent frame;
    # otherwise retain a uniformly sampled non-empty prefix. This is applied in
    # latent space because LiDAR HDMap conditioning has no separate opacity mask.
    hdmap_artifixer_dropout: bool = False
    shift: float = 1.0
    # LTX training stretch (Fig. 10): map shifted-t percentiles (0.5, 99.9) to [0,1].
    # None = shift only (legacy). Not the same as inference ``enable_shift_stretch``.
    shift_clamp_percentiles: Optional[tuple] = None
    # LTX stretch (inference only): remap schedule clean end to this t (e.g. 0.1).
    enable_shift_stretch: bool = False
    shift_stretch_terminal: float = 0.1
    train_time_distribution: str = "uniform"
    train_time_weight: str = "uniform"
    # RF prediction parameterization. "velocity" (default, legacy): the net
    # regresses the flow velocity v = noise - clean. "clean": the net regresses
    # the clean latent x0 directly (VGGT-World z-prediction); the sampler
    # converts x0_hat -> velocity via v = (x_t - x0_hat)/sigma. Loss is then a
    # clean-target MSE ||x0_hat - x0||^2 (uniform-weighted, VGGT eq. 12 style).
    rf_prediction_type: str = "velocity"
    # Paired single-frame autoregressive baselines. Sensor2Sensor concatenates
    # one previous-frame latent; LaGen-SDE concatenates previous, foreground,
    # and background latents and FiLM-modulates the previous latent by the
    # relative ego pose.
    autoregressive_condition_count: int = 0
    autoregressive_pose_film: bool = False
    autoregressive_pose_film_hidden_channels: int = 256
    autoregressive_noise_modulation_max: float = 0.0
    autoregressive_ego_action_only: bool = False
    autoregressive_geometry_root_path: str = ""
    autoregressive_geometry_filter_list: str = ""
    autoregressive_min_range: float = 2.5
    autoregressive_max_range: float = 75.0
    autoregressive_min_value: float = -1.0
    autoregressive_inverse_depth: bool = False
    # Blockwise Diffusion Forcing (HDMap->LiDAR or forecast RF). A positive
    # block size samples one independent RF time per temporal block and pairs it
    # with a block-causal DiT mask of the same size. Zero preserves the legacy single
    # time per full clip. The current 8-frame Waymo recipe has latent T=9
    # (prepend + 8 real frames), so block_size=3 gives three causal blocks.
    diffusion_forcing_block_size: int = 0
    # Match ArtiFixer's stage-2 Diffusion Forcing time sampler and weighting:
    # 1000 discrete uniform base-sigma levels, shift-only reparameterization,
    # and the reference bell-shaped per-timestep loss weights.
    diffusion_forcing_artifixer_matching: bool = False
    # Causal teacher-forcing variants optimize only the denoiser, replay clean
    # prefix K/V, and optionally rescale the RF objective.
    optimizer_net_only: bool = False
    teacher_forcing_prefix_frames: int = 0
    teacher_forcing_detach_clean_kv: bool = False
    teacher_forcing_loss_scale: float = 1.0
    # Imaginaire causal-training mixture over the number of leading clean
    # latent frames. Dedicated Imaginaire models supply their reference
    # defaults when these lists are empty; other model classes ignore them.
    imaginaire_conditioned_prefix_frames: List[int] = attrs.field(
        factory=list
    )
    imaginaire_conditioned_prefix_probs: List[float] = attrs.field(
        factory=list
    )
    # Existing LiDAR video datasets prepend a duplicate of frame 0 before VAE
    # encoding. Imaginaire-style training instead consumes unique frames and
    # treats the actual frame 0 as its singleton temporal prefix.
    input_video_has_prepend: bool = True
    # Optional second geometry condition for block-causal HDMap->LiDAR RF.
    # The dataloader accumulates only lidar sweeps from blocks preceding each
    # target block, renders them at each target pose, and the model encodes the
    # resulting range video with the frozen lidar VAE. The latent enters through
    # the network's zero-initialized side patch projection; False is a no-op.
    blockwise_warped_history_condition: bool = False
    # Concatenate the VAE-encoded geometry after the LiDAR and HDMap slots
    # instead of using the side patch projection. This is opt-in so faithful
    # ArtiFixer and Imaginaire recipes keep their upstream architecture.
    blockwise_warped_history_concat: bool = False
    # Source of the blockwise warped-history range video. "dataloader" keeps
    # the legacy worker-side NumPy all-history union. "torch_spatial_map"
    # constructs a causal per-sample TorchSpatialPointMap on the training device.
    blockwise_warped_history_backend: str = "dataloader"
    blockwise_spatial_map_radius_m: float = 75.0
    blockwise_spatial_map_voxel_m: float = 0.2
    blockwise_spatial_map_voxel_policy: str = "newest"
    # "frame" counts distinct frame hits inside each committed block; map
    # visibility and candidate expiry remain block-boundary operations.
    blockwise_spatial_map_observation_unit: str = "block"
    blockwise_spatial_map_static_min_observations: int = 2
    blockwise_spatial_map_candidate_ttl_updates: int = 1
    blockwise_spatial_map_include_latest: bool = True
    # Per-global-microstep chance of rebuilding the training map from one
    # detached terminal-sigma Euler prediction instead of ground-truth sweeps.
    # The decision is synchronized because this path adds an FSDP forward.
    blockwise_spatial_map_self_prediction_prob: float = 0.0
    blockwise_spatial_map_free_space_clear: bool = False
    blockwise_spatial_map_free_space_margin_m: float = 0.75
    blockwise_spatial_map_free_space_misses_to_remove: int = 2
    blockwise_spatial_map_geometry_root_path: str = ""
    blockwise_spatial_map_geometry_filter_list_path: str = ""
    blockwise_spatial_map_geometry_resample_width: int = -1
    blockwise_spatial_map_min_range: float = 2.5
    blockwise_spatial_map_max_range: float = 75.0
    blockwise_spatial_map_min_value: float = -1.0
    blockwise_spatial_map_inverse_depth: bool = False
    # Replace the Gaussian RF endpoint with a confidence-weighted soft
    # geometry proposal. Its confidence-composited range render also enters
    # through the DiT side projection.
    blockwise_geometry_source: bool = False
    blockwise_geometry_source_noise_floor: float = 0.2
    # Composite uncertain projected pixels against empty range before the
    # nonlinear VAE encode. Latent confidence/noise blending still applies.
    blockwise_geometry_source_preencode_confidence: bool = False
    blockwise_geometry_source_dropout_prob: float = 0.0
    blockwise_geometry_source_density_kernel: int = 5
    blockwise_geometry_source_density_saturation_hits: float = 5.0
    # Isolated Stage-2 multi-task conditioning experiment. One mode is sampled
    # for the full distributed step so every rank follows the same model path.
    stage2_multitask_modes: List[str] = attrs.field(factory=list)
    stage2_multitask_probs: List[float] = attrs.field(factory=list)
    stage2_multitask_sampling: str = "categorical"
    stage2_multitask_known_mask_concat: bool = False
    stage2_multitask_forecast_observed_frames: int = 6
    stage2_multitask_resim_known_min_per_block: int = 1
    stage2_multitask_resim_known_max_per_block: int = 2
    stage2_multitask_resim_drop_block_prob: float = 0.1
    stage2_multitask_world_hdmap_dropout_prob: float = 0.1
    stage2_multitask_hdmap_bbox_channel_start: int = 4
    # Factorized Stage-2 conditioning. Prefix length, HDMap form, and geometry
    # provenance are sampled without task labels or task embeddings.
    stage2_known_mask_concat: bool = False
    stage2_factorized_prefix_frames: List[int] = attrs.field(factory=list)
    stage2_factorized_prefix_probs: List[float] = attrs.field(factory=list)
    stage2_factorized_hdmap_modes: List[str] = attrs.field(factory=list)
    stage2_factorized_hdmap_probs: List[float] = attrs.field(factory=list)
    stage2_factorized_geometry_modes_k0: List[str] = attrs.field(factory=list)
    stage2_factorized_geometry_probs_k0: List[float] = attrs.field(
        factory=list
    )
    stage2_factorized_geometry_modes_known: List[str] = attrs.field(
        factory=list
    )
    stage2_factorized_geometry_probs_known: List[float] = attrs.field(
        factory=list
    )
    factorized_geometry_block_size: int = 4
    factorized_geometry_prefix_frames: int = 1
    # Deskew source sweeps before factorized geometry accumulation and
    # re-skew each rendered condition into the target sweep convention.
    factorized_geometry_rolling_shutter: bool = False
    # ``frame`` preserves the legacy one-box-pose-per-sweep behavior.
    # The other modes interpolate adjacent actor boxes by approximate ray time.
    factorized_geometry_actor_box_time_mode: str = "frame"
    # When enabled, frozen geometry samples its own observed-prefix length G
    # instead of reusing the clean-latent prefix length K.
    factorized_decouple_geometry_observations: bool = False
    factorized_geometry_observation_frames: List[int] = attrs.field(
        factory=list
    )
    factorized_geometry_observation_probs: List[float] = attrs.field(
        factory=list
    )
    factorized_force_geometry_observations: int = -1
    factorized_bidirectional_stage1: bool = False
    stage2_factorized_hdmap_bbox_channel_start: int = 4
    stage2_factorized_force_prefix_k: int = -1
    stage2_factorized_force_hdmap_mode: str = ""
    stage2_factorized_force_geometry_mode: str = ""
    # Teacher-forcing discrete consistency distillation. The frozen teacher is
    # the causal Stage-2 network; earlier blocks are replayed as clean context
    # while one selected block follows an adjacent shifted RF ladder interval.
    tfdcm_total_steps: int = 48
    tfdcm_skipping_interval_steps: int = 1
    tfdcm_timestep_shift: float = 5.0
    tfdcm_teacher_guidance: float = 1.0
    tfdcm_teacher_checkpoint: str = ""
    tfdcm_loss_scale: float = 100.0
    # Self-Forcing DMD (HDMap->LiDAR DF only). Defaults keep every existing
    # experiment unchanged; the dedicated SF-DMD model validates/populates all
    # fields before use.
    sfdmd_num_real_frames: int = 41
    sfdmd_window_frames: int = 8
    sfdmd_fake_updates_per_cycle: int = 5
    sfdmd_kv_cache_frames: int = 6
    # Optional permanent prefix retained when the bounded DiT K/V cache rolls.
    # ArtiFixer's three-frame adaptation uses sink3 + recent3 as persistent
    # history, then appends the current three-frame query for nine total frames.
    sfdmd_kv_sink_frames: int = 0
    # Number of generated three-frame blocks linked by autograd during student
    # replay before cached K/V is detached (truncated BPTT).
    sfdmd_tbptt_blocks: int = 1
    # Keep canonical ArtiFixer on full-rollout BPTT unless an isolated,
    # memory-bounded handoff explicitly opts into truncation.
    sfdmd_allow_truncated_bptt: bool = False
    sfdmd_kv_equivalence_atol: float = 2.0e-2
    sfdmd_run_kv_equivalence_gate: bool = True
    # Skip bias-free text cross-attention only after a runtime assertion that
    # the supplied text embedding is exactly zero.
    sfdmd_skip_zero_crossattn: bool = False
    sfdmd_stream_condition_vae_encode: bool = False
    sfdmd_condition_vae_cache_frames: int = 6
    sfdmd_accumulated_geometry_warp: bool = False
    sfdmd_geometry_source: bool = False
    sfdmd_geometry_source_noise_floor: float = 0.1
    sfdmd_geometry_source_preencode_confidence: bool = False
    sfdmd_geometry_source_density_kernel: int = 5
    sfdmd_geometry_source_density_saturation_hits: float = 5.0
    sfdmd_spatial_map_radius_m: float = 75.0
    sfdmd_spatial_map_voxel_m: float = 0.2
    sfdmd_spatial_map_voxel_policy: str = "newest"
    # Kept separate so Stage 4 must explicitly match its input checkpoint.
    sfdmd_spatial_map_observation_unit: str = "block"
    sfdmd_spatial_map_static_min_observations: int = 2
    sfdmd_spatial_map_candidate_ttl_updates: int = 1
    sfdmd_spatial_map_include_latest: bool = True
    sfdmd_spatial_map_free_space_clear: bool = False
    sfdmd_spatial_map_free_space_margin_m: float = 0.75
    sfdmd_spatial_map_free_space_misses_to_remove: int = 2
    sfdmd_num_steps: int = 4
    sfdmd_sde_eta: float = 1.0
    # ArtiFixer-compatible DMD mode: randomly select one denoising exit step,
    # use fresh-noise SDE transitions, train one full-rollout critic update per
    # iteration, and update the student every ``student_update_freq`` iterations.
    sfdmd_artifixer_matching: bool = False
    sfdmd_imaginaire_matching: bool = False
    sfdmd_clean_x0_kv_commit: bool = False
    # Preserve the task-free Stage-2 prefix/HDMap/geometry distribution during
    # ArtiFixer-style self-forcing rollouts.
    sfdmd_factorized_conditioning: bool = False
    # Training-only controls for isolating temporal and spatial self-forcing.
    # Existing experiments retain generated history by default.
    sfdmd_temporal_history_source: str = "generated"
    sfdmd_geometry_history_source: str = "generated"
    sfdmd_temporal_prefix_frames: int = 0
    sfdmd_student_update_freq: int = 5
    sfdmd_ema_start_cycle: int = 0
    sfdmd_teacher_guidance: float = 1.0
    sfdmd_critic_min_sigma: float = 0.02
    sfdmd_critic_max_sigma: float = 0.98
    sfdmd_max_grad_norm: float = 1.0
    sfdmd_teacher_checkpoint: str = ""
    sfdmd_teacher_net: LazyDict | None = None
    sfdmd_geometry_root_path: str = ""
    sfdmd_geometry_filter_list_path: str = ""
    sfdmd_geometry_resample_width: int = -1
    sfdmd_min_range: float = 2.5
    sfdmd_max_range: float = 75.0
    sfdmd_min_value: float = -1.0
    sfdmd_inverse_depth: bool = False
    # ---- lidar point-cloud forecasting (LidarForecastRectifiedFlowModel) ----
    # Number of clean PAST context frames in a forecasting clip (the remaining
    # sample_n_frames - n_past frames are the FUTURE frames to predict). Only
    # consumed by the forecasting model; harmless default for other configs.
    n_past: int = 0
    # Per-sample prob of dropping (zeroing) the ego-pose conditioning during
    # training so the model learns a pose-unconditional branch -> pose CFG.
    pose_dropout_prob: float = 0.0
    # Per-sample prob of "history dropout": hide the k OLDEST past frames
    # (structured prefix drop, k~U{1..n_past-1}) by zeroing their latent content
    # + pose (mask stays 1 = "present but empty context"). Teaches the generative
    # forecaster to work with less history than a deterministic model needs.
    history_dropout_prob: float = 0.0
    # Forecasting only: Vista-style anchor concat. When True, broadcast the clean
    # most-recent past (t=0) latent across all T frames and channel-concat it to
    # the noised input (requires net.in_channels = 2*latent_ch). Gives future
    # frames a direct input-level view of the anchor instead of relying purely on
    # temporal attention. Default False -> existing configs unchanged.
    anchor_concat: bool = False
    # Forecasting only: replace the broadcast anchor with a per-frame latent
    # encoded from observed lidar sweeps warped into each target ego pose.
    # Shares the same extra latent_ch input slot as anchor_concat.
    warped_history_concat: bool = False
    # Dataloader rasterization mode for warped_history_concat.
    # "latest" uses only the newest observed sweep; "all" z-buffers all past.
    warped_history_mode: str = "all"
    # Whether to also use the legacy flattened-SE(3) ego_pose_head. Geometry-only
    # conditioning sets this False and net.ego_pose_dim=0.
    use_pose_condition: bool = True
    # Forecasting DF only: route the VAE-encoded observed-history warp through
    # the block-causal checkpoint's independently projected side condition.
    warped_history_side_condition: bool = False
    # Forecasting DF only: append this many always-zero channels in the legacy
    # HDMap concat slot, preserving the warm-start x_embedder shape while the
    # lidar-only warp enters through the side projection.
    forecast_null_concat_channels: int = 0
    # Selects the prefix-aware block-causal forecast model in inference tools.
    forecast_diffusion_forcing: bool = False
    # Forecasting only: Vista-style anchor CFG. Per-sample prob of dropping
    # (zeroing) the anchor concat during training so it becomes a CFG target
    # (uncond pass zeros the concat). 0.0 -> anchor is always-on context (no
    # anchor CFG). Vista uses 0.15.
    anchor_dropout_prob: float = 0.0
