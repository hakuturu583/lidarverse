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

"""Loss config options

Loss weights are scheduled using a piecewise linear LR schedule. The schedule is defined by a list of boundaries and values.

`boundaries` is a list of integers representing the iteration at which the weight value changes.
`values` is a list of floats representing the weight value at each boundary. It should have one more value than `boundaries`.

Example:
 A loss's weight will be:
    values[0] when step <= boundaries[0],
    values[1] when step > boundaries[0] and step <= boundaries[1],
    ..., and
    values[-1] when step > boundaries[-1].
"""
import attrs

from tokenizer.training.losses import ReduceMode
from tokenizer.training.losses.continuous import (
    ColorLoss,
    ExtraLidarDepthLoss,
    FlowLoss,
    GradientLoss,
    EikonalLoss,
    IntensityTailL1Loss,
    InvalidDepthSentinelLoss,
    KLLoss,
    NormalLoss,
    LocalChamferLoss,
    PerceptualLoss,
    TokenizerLoss,
    PointLoss,
    RepaAlignLoss,
    SurfaceConcentrationLoss,
    TemporalDiffLoss,
    ValidityBCELoss,
    RaydropMSELoss,
    ValidityLPIPSLoss,
    VideoConsistencyLoss,
    VoxelOccupancyBCELoss,
)
from utils.lazy_config import LazyCall as L
from utils.lazy_config import LazyDict


@attrs.define(slots=False)
class KLConfig:
    # each step is greater than boundaries[-1], so weight=values[-1]
    boundaries: list[int] = [0]
    values: list[float] = [1e-6]


@attrs.define(slots=False)
class PerceptualConfig:
    lpips_boundaries: list[int] = [500000]
    lpips_values: list[float] = [0.1, 0.073]
    # Layer weights for linearly combining the multi-layer vgg-based losses.
    layer_weights: list[float] = [1.0 / 2.6, 1.0 / 4.8, 1.0 / 3.7, 1.0 / 5.6, 10.0 / 1.5]
    # Gram loss, whether to turn on, and what weights to use.
    gram_enabled: bool = True
    gram_boundaries: list[int] = [500000]
    gram_values: list[float] = [0.0, 0.062]
    # Corr loss, whether to turn on, and what weights to use.
    corr_enabled: bool = False
    corr_boundaries: list[int] = [0]
    corr_values: list[float] = [0.0]
    # In the example training memory usage dropped from 64.03 GiB to 60.54 GiB
    # with checkpointing enabled for this loss for about 3.2% slowdown.
    # With checkpointing this and PerceptualLoss memory usage dropped
    # from 64.03 GiB to 52.94 GiB for about 18% slowdown
    # more details in MR:949
    checkpoint_activations: bool = False


@attrs.define(slots=False)
class ColorConfig:
    # Color (RGB) basic loss and its weight schedule.
    norm: str = "L1"
    boundaries: list[int] = [0]
    values: list[float] = [1.0]
    # Opt-in valid-return mean. Default False preserves historical masked
    # losses, whose magnitude is proportional to the valid-pixel fraction.
    normalize_by_valid_mask: bool = False


@attrs.define(slots=False)
class PointConfig:
    # Coord loss and its weight schedule.
    enabled: bool = False
    norm: str = "L1"
    boundaries: list[int] = [0]
    values: list[float] = [1.0]
    normalize_by_valid_mask: bool = False


@attrs.define(slots=False)
class FlowConfig:
    # Flow loss and its weight schedule.
    boundaries: list[int] = [250000]
    values: list[float] = [0.0, 0.01]
    scale: int = 2
    # Flow loss depends on RAFT, as such it requires a specific dtype.
    dtype: str = "bfloat16"
    # In the example training memory usage dropped from 28GB to 23GB
    # with checkpointing enabled for this loss
    # With checkpointing this and PerceptualLoss memory usage dropped
    # from 64.03 GiB to 52.94 GiB for about 18% slowdown
    # more details in MR:949
    checkpoint_activations: bool = False
    enabled: bool = False


@attrs.define(slots=False)
class GradientConfig:
    enabled: bool = False
    boundaries: list[int] = [0]
    values: list[float] = [1.0]


@attrs.define(slots=False)
class LocalChamferConfig:
    # One-sided windowed Chamfer (recon -> nearest GT in a (2r+1)^2 range-image
    # window, meters). Direct floater killer; differentiable (Charbonnier
    # sqrt(d^2+eps)); drop-in for point_loss (radius=0 == Euclidean point error).
    # Needs the de-repeated point maps (computed when this OR point_loss/normal
    # is enabled). Candidates restricted to valid GT returns via MASK_KEY.
    enabled: bool = False
    boundaries: list[int] = [0]
    values: list[float] = [0.0]
    radius: int = 2  # azimuth (W) window half-size
    # Elevation (H) window half-size: <0 mirrors ``radius`` (isotropic, default);
    # 0 = AZIMUTH-ONLY (no cross-ring matching -> preserves ring texture).
    radius_h: int = -1
    eps: float = 1e-4


@attrs.define(slots=False)
class TemporalDiffConfig:
    # PV-VAE matched temporal-difference (motion-aware) loss.
    enabled: bool = False
    boundaries: list[int] = [0]
    values: list[float] = [0.0]
    # Normalized no-return clamp value; pairs with either endpoint at/below
    # ``min_value + raydrop_eps`` (in either gt or recon) are dropped to
    # avoid raydrop / tree-foliage flicker driving the loss.
    min_value: float = -1.0
    raydrop_eps: float = 0.02
    # Ramp the per-sample weight by the interpolative noise level
    # (``noise_tau`` exposed by the network): ~0 on clean, full on noisy.
    tau_ramp: bool = True
    tau_ramp_power: float = 1.0


@attrs.define(slots=False)
class ValidityBCEConfig:
    # BCE on the decoder's predicted validity (raydrop) mask. Requires the
    # network to be built with ``predict_validity=True`` (adds one extra
    # decoder output channel of validity logits). arxiv:2605.22809 Eq. 4.
    enabled: bool = False
    boundaries: list[int] = [0]
    values: list[float] = [0.0]
    # Normalized no-return clamp value; GT mask = (range > min_value + eps).
    min_value: float = -1.0
    raydrop_eps: float = 0.02
    # When True, the GT validity is read from the dataloader-provided MASK_KEY
    # (per-row norm runs) instead of thresholding recon_target.
    mask_from_batch: bool = False
    # Optional class-balanced BCE. When set, compute the mean BCE separately
    # over GT no-return and valid pixels, then assign this fraction of the loss
    # to no-return pixels and the remainder to valid pixels. None preserves the
    # historical per-pixel mean.
    invalid_class_weight: float | None = None


@attrs.define(slots=False)
class IntensityTailL1Config:
    # Additional valid-return L1 averaged over each sensor's rare bright tail.
    # Thresholds are on the raw linear [0,1] scale and indexed by
    # sensor_dataset_id.
    enabled: bool = False
    boundaries: list[int] = [0]
    values: list[float] = [0.0]
    intensity_channel: int = 1
    linear_thresholds: list[float] = [0.99]
    intensity_transform: str = "log1p"


@attrs.define(slots=False)
class InvalidDepthSentinelConfig:
    # Low-weight L1 anchor for the otherwise-unsupervised depth output at
    # explicit no-return pixels. Requires MASK_KEY to carry GT validity.
    enabled: bool = False
    boundaries: list[int] = [0]
    values: list[float] = [0.0]
    sentinel: float = -1.0


@attrs.define(slots=False)
class RaydropMSEConfig:
    # OmniGen L_r: MSE on the rendered raydrop PROBABILITY (sigmoid of the
    # validity logits) vs the binary return mask. Same target derivation as
    # ValidityBCEConfig; differs only in MSE vs BCE. arXiv:2512.14225.
    enabled: bool = False
    boundaries: list[int] = [0]
    values: list[float] = [0.0]
    min_value: float = -1.0
    raydrop_eps: float = 0.02
    mask_from_batch: bool = False


@attrs.define(slots=False)
class NormalConfig:
    # Surface-normal consistency (Sensor2Sensor L_normals, arXiv:2605.22809).
    # Finite-differences per-pixel normals from the projected 3D points and
    # matches recon vs GT on valid returns -- constrains the surface derivative
    # (flattens ground) rather than absolute range. Reuses the de-repeated point
    # maps the model stashes for point_loss; no-op if neither is enabled.
    enabled: bool = False
    boundaries: list[int] = [0]
    values: list[float] = [0.0]
    mode: str = "cosine"  # "cosine" (default) or "l1"
    # Match the dataset depth range; no-returns sit at min_range and are dropped.
    min_range: float = 5.0
    max_range: float = 100.0
    valid_eps: float = 0.5


@attrs.define(slots=False)
class EikonalConfig:
    # Eikonal regularizer (IGR/SIREN; LODE grad_constraint): ||grad phi|| -> 1 on
    # the decoded SDF volume so phi is a metric distance field (clean unit-normal
    # field, smooth zero level-set -> good normals without GT-normal supervision).
    # Only the decomp-BEV SDF tokenizer (use_sdf=True) stashes sdf_density; no-op
    # elsewhere. num_samples>0 = random interior voxels/item (memory-frugal); 0 =
    # full volume.
    enabled: bool = False
    boundaries: list[int] = [0]
    values: list[float] = [0.0]
    num_samples: int = 65536


@attrs.define(slots=False)
class SurfaceConcentrationConfig:
    # Concentrate the renderer's per-ray weight distribution on the GT surface
    # (kills floaters so E[d] is clean). Only for the BEV occupancy tokenizers,
    # whose forward stashes render_weights / render_sample_depths /
    # render_gt_depth / render_valid in the output dict. No-op for other nets.
    enabled: bool = False
    boundaries: list[int] = [0]
    values: list[float] = [0.0]
    # Squared (d_s - d_gt) is divided by depth_scale^2 to keep the loss O(1)
    # (set near the sensor max range). Used only in mode="variance".
    depth_scale: float = 100.0
    # "variance" (smooth proxy, default) or "surface_indicator" (exact Copilot4D
    # Eq.5: Σ_i 1(|h_i-D_gt|>epsilon) w_i). epsilon is in meters.
    mode: str = "variance"
    epsilon: float = 0.1


@attrs.define(slots=False)
class VoxelOccupancyBCEConfig:
    # Direct occupancy supervision (Copilot4D): BCE on the decoder's 3D binary
    # voxel logits vs the voxelized input occupancy. Only for the Copilot4D
    # tokenizer, whose forward stashes voxel_logits / voxel_occupancy_target in
    # the output dict. No-op for other nets.
    enabled: bool = False
    boundaries: list[int] = [0]
    values: list[float] = [0.0]
    # Upweight the rare occupied voxels (occupancy is sparse). 0 disables.
    pos_weight: float = 10.0


@attrs.define(slots=False)
class ExtraLidarConfig:
    # Multi-lidar depth reconstruction: L1 on the extra (blind-spot) lidars'
    # rendered depth vs measured range (normalized space), masked to returns.
    # Only for the BEV occupancy tokenizer built with ``multi_lidar=True``, whose
    # forward stashes extra_depth / extra_gt_depth / extra_valid. No-op otherwise.
    enabled: bool = False
    boundaries: list[int] = [0]
    values: list[float] = [0.0]


@attrs.define(slots=False)
class ValidityLPIPSConfig:
    # LPIPS on the predicted validity mask (arxiv:2605.22809). Requires the
    # network to be built with ``predict_validity=True``. Loads its own VGG.
    enabled: bool = False
    boundaries: list[int] = [0]
    values: list[float] = [0.0]
    min_value: float = -1.0
    raydrop_eps: float = 0.02
    checkpoint_activations: bool = False
    mask_from_batch: bool = False


@attrs.define(slots=False)
class VideoConsistencyConfig:
    # Add consistency loss between overlapped video frames
    boundaries: list[int] = [250000]
    values: list[float] = [0.0, 0.01]
    enabled: bool = False
    num_frames: int = 9
    step: int = 1


@attrs.define(slots=False)
class RepaAlignConfig:
    # REPA / Volt conv-teacher alignment: pull a trainable projection of the
    # latent toward frozen Utonia (PTv3) BEV features (masked cosine). Requires
    # the network built with ``utonia_align=True`` (UniPAD tokenizer, bev_patchify).
    enabled: bool = False
    boundaries: list[int] = [0]
    values: list[float] = [0.0]


@attrs.define(slots=False)
class VideoLoss:
    # The combined loss function, and its reduction mode.
    color: LazyDict = L(ColorLoss)(config=ColorConfig())
    kl: LazyDict = L(KLLoss)(config=KLConfig())
    perceptual: LazyDict = L(PerceptualLoss)(config=PerceptualConfig())
    flow: LazyDict = L(FlowLoss)(config=FlowConfig())
    video_consistency: LazyDict = L(VideoConsistencyLoss)(config=VideoConsistencyConfig())
    reduce: str = ReduceMode.MEAN.value  # model.config.loss.config.reduce={'MEAN', 'SUM', 'SUM_PER_FRAME'}
    point_loss: LazyDict = L(PointLoss)(config=PointConfig())
    local_chamfer: LazyDict = L(LocalChamferLoss)(config=LocalChamferConfig())
    normal: LazyDict = L(NormalLoss)(config=NormalConfig())
    eikonal: LazyDict = L(EikonalLoss)(config=EikonalConfig())
    gradient: LazyDict = L(GradientLoss)(config=GradientConfig())
    temporal_diff: LazyDict = L(TemporalDiffLoss)(config=TemporalDiffConfig())
    intensity_tail_l1: LazyDict = L(IntensityTailL1Loss)(
        config=IntensityTailL1Config()
    )
    validity_bce: LazyDict = L(ValidityBCELoss)(config=ValidityBCEConfig())
    invalid_depth_sentinel: LazyDict = L(InvalidDepthSentinelLoss)(
        config=InvalidDepthSentinelConfig()
    )
    raydrop_mse: LazyDict = L(RaydropMSELoss)(config=RaydropMSEConfig())
    validity_lpips: LazyDict = L(ValidityLPIPSLoss)(config=ValidityLPIPSConfig())
    surface_concentration: LazyDict = L(SurfaceConcentrationLoss)(config=SurfaceConcentrationConfig())
    voxel_occupancy_bce: LazyDict = L(VoxelOccupancyBCELoss)(config=VoxelOccupancyBCEConfig())
    extra_lidar: LazyDict = L(ExtraLidarDepthLoss)(config=ExtraLidarConfig())
    repa_align: LazyDict = L(RepaAlignLoss)(config=RepaAlignConfig())
    # Restrict the depth losses (color / gradient / point) to VALID returns.
    # When True, the GT validity mask (range channel > min_value + raydrop_eps)
    # replaces the all-ones MASK_KEY: color & gradient weight by it, and
    # point_loss drops no-return points. The decoder is then not penalized for
    # failing to reproduce the no-return clamp value -- that pixel is supervised
    # by the validity head instead. Default False = existing all-pixels behavior.
    mask_invalid_depth: bool = False
    mask_invalid_min_value: float = -1.0
    mask_invalid_raydrop_eps: float = 0.02
    # When True, the per-pixel valid mask used by the depth losses comes from
    # the dataloader (``loss_mask`` -> MASK_KEY) rather than being derived from
    # recon_target. Pair with ``emit_valid_mask=True`` on the dataset and
    # ``mask_from_batch=True`` on validity_bce/validity_lpips. Required when
    # per-row normalization is on (the depth sentinel is unrecoverable).
    mask_from_batch: bool = False

@attrs.define(slots=False)
class ImageLoss:
    # The combined loss function, and its reduction mode.
    color: LazyDict = L(ColorLoss)(config=ColorConfig())
    kl: LazyDict = L(KLLoss)(config=KLConfig())
    perceptual: LazyDict = L(PerceptualLoss)(config=PerceptualConfig())
    reduce: str = ReduceMode.MEAN.value  # model.config.loss.config.reduce={'MEAN', 'SUM', 'SUM_PER_FRAME'}

VideoLossConfig: LazyDict = L(TokenizerLoss)(config=VideoLoss())
ImageLossConfig: LazyDict = L(TokenizerLoss)(config=ImageLoss())
