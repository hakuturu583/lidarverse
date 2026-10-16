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

"""The combined loss functions for continuous-space tokenizers training."""
import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import torchvision.models.optical_flow as optical_flow

from tokenizer.networks.utils import batch2time, time2batch
from tokenizer.training.datasets.utils import INPUT_KEY, LATENT_KEY, MASK_KEY, RECON_KEY, RECON_TARGET_KEY
from tokenizer.training.losses import ReduceMode
from tokenizer.training.losses.lpips import LPIPS
from utils.lazy_config import instantiate

# NOTE: TokenizerLoss.forward double-adds the LAST module's loss (lines
# ~58-61). ``gradient`` is intentionally kept last to preserve that
# pre-existing behavior for ``_grad`` configs; new losses (temporal_diff,
# validity_bce, validity_lpips) must go BEFORE it so they are counted exactly once.
_VALID_LOSS_NAMES = ["color", "perceptual", "flow", "kl", "video_consistency", "point_loss", "local_chamfer", "normal", "eikonal", "temporal_diff", "intensity_tail_l1", "validity_bce", "invalid_depth_sentinel", "raydrop_mse", "validity_lpips", "surface_concentration", "voxel_occupancy_bce", "extra_lidar", "repa_align", "gradient"]
VIDEO_CONSISTENCY_LOSS = "video_consistency"
RECON_CONSISTENCY_KEY = f"{RECON_KEY}_consistency"
POINT_LOSS = "point_loss"


def _loss_enabled(node) -> bool:
    """Whether a loss node should actually be instantiated.

    A loss whose inner ``config`` sets ``enabled=False`` is skipped ENTIRELY
    (replaced by ``NullLoss``), not merely made a no-op in ``forward``. This
    keeps disabled losses from constructing heavy backbones at __init__ time --
    ``ValidityLPIPSLoss``/``PerceptualLoss`` build a VGG and ``FlowLoss`` builds
    RAFT, each of which downloads a checkpoint. Those downloads race/corrupt
    across ranks on multi-node jobs, so a loss that's off by default must not
    pull them in. Losses without an ``enabled`` flag (color/kl/perceptual)
    default to enabled and are unaffected.
    """
    inner = getattr(node, "config", None)
    if inner is None:
        return True
    try:
        return bool(getattr(inner, "enabled", True))
    except Exception:
        return True


class TokenizerLoss(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        _reduce = ReduceMode(config.reduce.upper()) if hasattr(config, "reduce") else None
        self.reduce = _reduce.function
        self.loss_modules = nn.ModuleDict()
        for key in _VALID_LOSS_NAMES:
            node = getattr(config, key) if hasattr(config, key) else None
            # Gate on the inner config's ``enabled`` flag so a disabled loss is
            # never instantiated (-> NullLoss), keeping its backbone/checkpoint
            # from loading. NullLoss still occupies the dict key, so model.py /
            # callbacks that index loss_modules[...] and guard on
            # hasattr(., "enabled") stay correct, and the forward-loop ordering
            # (gradient last, double-added) is unchanged.
            is_valid = node is not None and _loss_enabled(node)
            self.loss_modules[key] = instantiate(node) if is_valid else NullLoss()
        # When enabled, the per-pixel MASK_KEY becomes the GT validity (return)
        # mask instead of all-ones, so the depth losses skip no-return pixels.
        self.mask_invalid_depth = getattr(config, "mask_invalid_depth", False)
        self.mask_invalid_min_value = getattr(config, "mask_invalid_min_value", -1.0)
        self.mask_invalid_raydrop_eps = getattr(config, "mask_invalid_raydrop_eps", 0.02)
        # When True, the per-pixel validity mask comes from the dataloader
        # (``loss_mask`` -> MASK_KEY, set in model.{training,validation}_step)
        # rather than being derived from ``recon_target > min_value + eps``.
        # Required under per-row normalization, where the depth sentinel is
        # unrecoverable. The incoming [B,1,...] mask is broadcast to all
        # channels for the depth losses; validity_bce/lpips read it as target.
        self.mask_from_batch = getattr(config, "mask_from_batch", False)

    def forward(self, inputs, output_batch, iteration) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        loss = dict()
        total_loss = 0.0

        if self.mask_from_batch:
            # Explicit per-pixel valid mask from the dataloader (already in
            # MASK_KEY via model.{training,validation}_step). Broadcast the
            # [B,1,...] mask to all channels; do NOT recompute from recon_target
            # (per-row norm makes the depth sentinel unrecoverable). Left as the
            # default all-ones if no loss_mask was provided.
            _target = inputs[RECON_TARGET_KEY]
            inputs[MASK_KEY] = inputs[MASK_KEY][:, :1].expand_as(_target)
        elif self.mask_invalid_depth:
            # GT validity (return) mask from the depth channel (channel 0),
            # broadcast across all channels. 1 = valid return, 0 = no-return.
            # Same threshold ValidityBCELoss uses, so depth supervision and the
            # validity head are consistent. color/gradient read MASK_KEY
            # directly; point_loss reshapes it to drop no-return points.
            _target = inputs[RECON_TARGET_KEY]
            _thr = self.mask_invalid_min_value + self.mask_invalid_raydrop_eps
            _valid = (_target[:, :1] > _thr).to(_target.dtype)
            inputs[MASK_KEY] = _valid.expand_as(_target)
        else:
            inputs[MASK_KEY] = torch.ones_like(inputs[RECON_TARGET_KEY])
        # Calculates reconstruction losses (`total_loss`).
        for key, module in self.loss_modules.items():
            curr_loss = module(inputs, output_batch, iteration)
            loss.update({k: torch.mean(v) for k, v in curr_loss.items()})
            total_loss += sum([self.reduce(v) if (v.dim() > 0) else v for v in curr_loss.values()])

        loss.update({k: torch.mean(v) for k, v in curr_loss.items()})

        # Computes the overall loss as sum of the reconstruction losses and the generator loss.
        total_loss += sum([self.reduce(v) if (v.dim() > 0) else v for v in curr_loss.values()])
        return dict(loss=loss), total_loss


class WeightScheduler(torch.nn.Module):
    def __init__(self, boundaries, values):
        super().__init__()
        self.boundaries = list(boundaries)
        self.values = list(values)

    def forward(self, iteration):
        for boundary, value in zip(self.boundaries, self.values):
            if iteration < boundary:
                return value
        return self.values[-1]


class NullLoss(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, inputs, output_batch, iteration) -> dict[dict, torch.Tensor]:
        return dict()


class ColorLoss(torch.nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.schedule = WeightScheduler(boundaries=config.boundaries, values=config.values)
        self.normalize_by_valid_mask = getattr(
            config, "normalize_by_valid_mask", False
        )

    def forward(self, inputs, output_batch, iteration) -> dict[str, torch.Tensor]:
        reconstructions = output_batch[RECON_KEY]
        weights = inputs[MASK_KEY]
        if self.normalize_by_valid_mask:
            weights = weights * (
                weights.numel() / weights.sum().clamp_min(1.0)
            )
        recon_target = inputs[RECON_TARGET_KEY]
        recon = weights * torch.abs(recon_target.contiguous() - reconstructions.contiguous())
        color_weighted = self.schedule(iteration) * recon
        if torch.isnan(color_weighted).any():
            raise ValueError("[COLOR] NaN detected in loss")
        return dict(color=color_weighted)

class PointLoss(torch.nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.schedule = WeightScheduler(boundaries=config.boundaries, values=config.values)
        self.enabled = config.enabled
        self.normalize_by_valid_mask = getattr(
            config, "normalize_by_valid_mask", False
        )
        
    def forward(self, inputs, output_batch, iteration) -> dict[str, torch.Tensor]:
        if not self.enabled:
            return dict()

        reconstructions = output_batch[f"{RECON_KEY}_points"].contiguous()
        targets = output_batch[f"{RECON_TARGET_KEY}_points"].contiguous()
        err = torch.abs(targets - reconstructions)  # (B, F, H, W, 3)
        # Drop no-return points using the same per-pixel validity mask the
        # other depth losses use (MASK_KEY; all-ones when masking is disabled,
        # so this is a no-op then). Reshape image-layout (B, C, [T,] H, W) to
        # point layout (B, F, H, W, 1); skip if spatial dims don't line up
        # (e.g. repeat_row/col > 1 -> points are de-repeated).
        mask = inputs.get(MASK_KEY, None)
        if mask is not None:
            m = mask[:, 0].unsqueeze(-1) if mask.ndim == 5 else mask[:, :1].unsqueeze(-1)
            if m.shape[-3:-1] == err.shape[-3:-1] and m.shape[1] in (1, err.shape[1]):
                if self.normalize_by_valid_mask:
                    m = m * (m.numel() / m.sum().clamp_min(1.0))
                err = err * m
        loss = err.mean(dim=1)
        point_weighted = self.schedule(iteration) * loss
        if torch.isnan(point_weighted).any():
            raise ValueError("[POINT] NaN detected in loss")
        return dict(point=point_weighted)


class LocalChamferLoss(torch.nn.Module):
    """One-sided, windowed Chamfer (recon -> nearest GT) -- the floater killer.

    For every reconstructed point we take the distance to the CLOSEST ground-truth
    point inside a ``(2r+1) x (2r+1)`` range-image window centered on the same
    pixel, and penalize that distance. A floater (a recon point hallucinated into
    empty space, typically from edge-smearing) has every GT neighbor far away, so
    it gets a large penalty and the decoder is pulled to snap it onto the nearest
    real surface. A correctly-placed point (even on a sharp depth edge) sits on top
    of a GT point, so its penalty is ~0 -- unlike ``gradient`` which lights up at
    every edge regardless of whether the point is actually wrong.

    Cheap: instead of an all-pairs Chamfer it does a shift-loop with a running min
    over the ``(2*radius_h+1) x (2*radius_w+1)`` window (azimuth W wraps circularly,
    elevation H clamps), so cost is ``O(window * HW)`` with no N^2 distance matrix.

    Window anisotropy (``radius_h`` / ``radius_w``): each range-image ROW is one
    laser ring, so an elevation (H) window lets a recon point match a GT point on
    an ADJACENT ring -- the decoder is then not penalized for getting this ring's
    depth wrong, which smears / merges rings (destroys the ring texture) at high
    weight. Set ``radius_h=0`` (AZIMUTH-ONLY) to forbid cross-ring matching: it
    keeps the edge-floater cleanup (a floater is off in range and has no nearby
    same-row GT) while preserving per-ring depth. ``radius_h<0`` (default) mirrors
    ``radius`` (isotropic) for backward compatibility. Pair an azimuth-only window
    with ``point_loss`` (exact pixel correspondence = the ring anchor); Chamfer is
    then the floater-cleanup term, not the sole geometry loss.

    Differentiable: ``best`` is the per-point min squared distance to a (detached)
    GT point; ``torch.minimum`` routes the gradient through the selected neighbor,
    so the decoder learns to move the point toward that surface. We use the
    Charbonnier / pseudo-Huber form ``sqrt(best + eps)`` -- smooth at 0 (no
    sqrt singularity) and METER-scaled, so it is a drop-in replacement for
    ``point_loss`` (``radius=0`` reduces to the per-point Euclidean point error).

    Reads the de-repeated metric point maps the model stashes (same as PointLoss:
    ``{recon,recon_target}_points``, shape ``(B, F, H, W, 3)``); no-op if absent.
    Candidates are restricted to VALID GT returns (via MASK_KEY) so floaters can
    only snap to real surfaces, and only valid recon positions are scored.
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.schedule = WeightScheduler(boundaries=config.boundaries, values=config.values)
        self.enabled = config.enabled
        self.radius_w = int(config.radius)  # azimuth window half-size
        # Elevation window half-size: <0 mirrors radius (isotropic, backward
        # compatible); 0 = azimuth-only (no cross-ring matching).
        _rh = int(getattr(config, "radius_h", -1))
        self.radius_h = self.radius_w if _rh < 0 else _rh
        self.eps = float(config.eps)

    def forward(self, inputs, output_batch, iteration) -> dict[str, torch.Tensor]:
        if not self.enabled:
            return dict()
        recon = output_batch.get(f"{RECON_KEY}_points", None)
        target = output_batch.get(f"{RECON_TARGET_KEY}_points", None)
        if recon is None or target is None:
            return dict()
        recon = recon.contiguous()
        target = target.detach().contiguous()  # GT is fixed; gradient flows to recon
        B, F, H, W, _ = recon.shape

        # Per-point valid mask (B, F, H, W) from MASK_KEY, matched to point layout
        # like PointLoss; None (skip masking) if spatial dims don't line up.
        mask = inputs.get(MASK_KEY, None)
        valid = None
        if mask is not None:
            m = mask[:, 0] if mask.ndim == 5 else mask[:, 0:1]  # (B,T,H,W) | (B,1,H,W)
            if m.shape[-2:] == (H, W) and m.shape[1] in (1, F):
                if m.shape[1] == 1 and F != 1:
                    m = m.expand(B, F, H, W)
                valid = m > 0.5

        INF = recon.new_tensor(1.0e9)
        best = recon.new_full((B, F, H, W), 1.0e9)
        hb = torch.arange(H, device=recon.device)
        wb = torch.arange(W, device=recon.device)
        for dh in range(-self.radius_h, self.radius_h + 1):
            h_idx = (hb + dh).clamp_(0, H - 1)          # elevation: clamp at edges
            for dw in range(-self.radius_w, self.radius_w + 1):
                w_idx = (wb + dw) % W                   # azimuth: circular wrap
                gt = target.index_select(2, h_idx).index_select(3, w_idx)  # (B,F,H,W,3)
                d = ((recon - gt) ** 2).sum(-1)         # squared dist (B,F,H,W)
                if valid is not None:
                    vshift = valid.index_select(2, h_idx).index_select(3, w_idx)
                    d = torch.where(vshift, d, INF)     # only real surfaces are candidates
                best = torch.minimum(best, d)

        dist = torch.sqrt(best + self.eps)              # Charbonnier ~ meters
        sel = best < (1.0e9 * 0.5)                      # had >=1 valid GT candidate
        if valid is not None:
            sel = sel & valid                           # score only valid recon points
        if sel.any():
            loss = dist[sel].mean()
        else:
            loss = recon.sum() * 0.0
        chamfer_weighted = self.schedule(iteration) * loss
        if torch.isnan(chamfer_weighted).any():
            raise ValueError("[LOCAL_CHAMFER] NaN detected in loss")
        return dict(local_chamfer=chamfer_weighted)


class NormalLoss(torch.nn.Module):
    """Surface-normal consistency (Sensor2Sensor L_normals, arXiv:2605.22809).

    Per-pixel normals are finite-differenced from the projected 3D points in
    range-view (``ComputeNormals`` in the paper):
    ``n = normalize(cross(p_right - p_left, p_down - p_up))``. The recon normals
    are supervised against the GT normals on valid returns. Unlike depth L1 /
    point_loss (which only constrain absolute range), this constrains the spatial
    derivative of the surface, so it directly flattens locally-planar regions --
    the ground in particular.

    Reuses the de-repeated metric point maps the model already stashes for
    point_loss (``{recon,recon_target}_points``, shape ``(N, T, H, W, 3)``);
    no-op if absent (point_loss / normal not enabled -> points not computed).
    De-repeated points are required: repeated rows would give zero diffs.

    Sensor2Sensor wraps this in LPIPS; we use a cheaper pixel-space consistency
    (cosine distance by default, ``mode="l1"`` for L1 on the normal vectors),
    which is the more direct smoothness signal. An LPIPS-on-normals variant can
    be added later like ``ValidityLPIPSLoss`` if perceptual matching is wanted.
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.schedule = WeightScheduler(boundaries=config.boundaries, values=config.values)
        self.enabled = config.enabled
        self.mode = getattr(config, "mode", "cosine")  # "cosine" | "l1"
        # No-returns are clamped to the near plane by geo_converter, so GT point
        # range == min_range there; range in (min_range+eps, max_range-eps)
        # selects valid surface returns. Match the dataset's depth_min/max.
        self.min_range = config.min_range
        self.max_range = config.max_range
        self.valid_eps = config.valid_eps

    @staticmethod
    def _normals(points: torch.Tensor) -> torch.Tensor:
        # points: (N, T, H, W, 3) -> unit normals (N, T, H-2, W-2, 3).
        pu = points[..., :-2, 1:-1, :]
        pd = points[..., 2:, 1:-1, :]
        pl = points[..., 1:-1, :-2, :]
        pr = points[..., 1:-1, 2:, :]
        n = torch.cross(pr - pl, pd - pu, dim=-1)
        return n / (n.norm(dim=-1, keepdim=True) + 1e-6)

    def forward(self, inputs, output_batch, iteration) -> dict[str, torch.Tensor]:
        if not self.enabled or self.schedule(iteration) == 0.0:
            return dict()
        rp = output_batch.get(f"{RECON_KEY}_points", None)
        tp = output_batch.get(f"{RECON_TARGET_KEY}_points", None)
        if rp is None or tp is None:
            return dict()
        n_re = self._normals(rp.contiguous())
        n_gt = self._normals(tp.contiguous())
        # Validity from GT point range (self-contained; aligned to the
        # de-repeated point resolution, unlike the full-res MASK_KEY).
        r = tp.norm(dim=-1)  # (N, T, H, W)
        v = ((r > self.min_range + self.valid_eps) & (r < self.max_range - self.valid_eps)).to(n_gt.dtype)
        # Require all 4 neighbors used by the cross product to be valid.
        vn = v[..., :-2, 1:-1] * v[..., 2:, 1:-1] * v[..., 1:-1, :-2] * v[..., 1:-1, 2:]
        if self.mode == "l1":
            err = (n_re - n_gt).abs().sum(dim=-1)
        else:
            err = 1.0 - (n_re * n_gt).sum(dim=-1)  # cosine distance in [0, 2]
        loss = (err * vn).sum() / vn.sum().clamp_min(1.0)
        normal_weighted = self.schedule(iteration) * loss
        if torch.isnan(normal_weighted).any():
            raise ValueError("[NORMAL] NaN detected in loss")
        return dict(normal=normal_weighted)


class EikonalLoss(torch.nn.Module):
    """Eikonal regularizer (IGR/SIREN; LODE ``grad_constraint``, arXiv:2302.14052).

    Pushes the decoded SDF volume toward ``||grad phi|| = 1`` so phi becomes a
    true *metric* signed-distance field. A unit-gradient field has a clean,
    consistent normal field (``grad phi / ||grad phi||``) and a smooth zero
    level-set, which removes spurious zero-crossings (floaters) and yields good
    normals *without* supervising on (noisy) GT LiDAR normals -- the surface
    location is anchored by the existing render/point losses, exactly the IGR
    recipe. Targets smooth ground as a by-product of the field being distance-like.

    Reads the ``(B*T, n_z, Y, X)`` volume + metric voxel sizes the decomp-BEV
    SDF network stashes (``sdf_density`` / ``sdf_voxel_size``); no-op if absent
    (only that tokenizer provides them; meaningless unless ``use_sdf=True``).
    Gradients are central finite differences scaled by metric voxel size.
    ``num_samples>0`` evaluates the constraint at that many random interior
    voxels per item (memory-frugal at full BEV resolution); ``0`` = whole volume.
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.schedule = WeightScheduler(boundaries=config.boundaries, values=config.values)
        self.enabled = config.enabled
        self.num_samples = config.num_samples

    def forward(self, inputs, output_batch, iteration) -> dict[str, torch.Tensor]:
        if not self.enabled or self.schedule(iteration) == 0.0:
            return dict()
        phi = output_batch.get("sdf_density", None)
        vsize = output_batch.get("sdf_voxel_size", None)
        if phi is None or vsize is None:
            return dict()
        vz, vy, vx = (float(v) for v in vsize)
        B, Z, Y, X = phi.shape
        interior = max(Z - 2, 0) * max(Y - 2, 0) * max(X - 2, 0)
        if interior == 0:
            return dict()
        if self.num_samples and 0 < self.num_samples < interior:
            n = self.num_samples
            zc = torch.randint(1, Z - 1, (B, n), device=phi.device)
            yc = torch.randint(1, Y - 1, (B, n), device=phi.device)
            xc = torch.randint(1, X - 1, (B, n), device=phi.device)
            b = torch.arange(B, device=phi.device)[:, None].expand(B, n)
            gz = (phi[b, zc + 1, yc, xc] - phi[b, zc - 1, yc, xc]) / (2 * vz)
            gy = (phi[b, zc, yc + 1, xc] - phi[b, zc, yc - 1, xc]) / (2 * vy)
            gx = (phi[b, zc, yc, xc + 1] - phi[b, zc, yc, xc - 1]) / (2 * vx)
        else:
            gz = (phi[:, 2:, 1:-1, 1:-1] - phi[:, :-2, 1:-1, 1:-1]) / (2 * vz)
            gy = (phi[:, 1:-1, 2:, 1:-1] - phi[:, 1:-1, :-2, 1:-1]) / (2 * vy)
            gx = (phi[:, 1:-1, 1:-1, 2:] - phi[:, 1:-1, 1:-1, :-2]) / (2 * vx)
        grad_norm = torch.sqrt(gx * gx + gy * gy + gz * gz + 1e-12)
        loss = (grad_norm - 1.0).abs().mean()
        eik_weighted = self.schedule(iteration) * loss
        if torch.isnan(eik_weighted).any():
            raise ValueError("[EIKONAL] NaN detected in loss")
        return dict(eikonal=eik_weighted)


class GradientLoss(torch.nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.schedule = WeightScheduler(boundaries=config.boundaries, values=config.values)
        self.enabled = config.enabled

    def forward(self, inputs, output_batch, iteration) -> dict[str, torch.Tensor]:
        if not self.enabled:
            return dict()

        reconstructions = output_batch[RECON_KEY]
        targets = inputs[RECON_TARGET_KEY]
        loss = self.multi_scale_grad_loss(reconstructions, targets, inputs[MASK_KEY])
        loss_weighted = self.schedule(iteration) * loss
        if torch.isnan(loss_weighted).any():
            raise ValueError("[GRADIENT] NaN detected in loss")
        return dict(gradient=loss_weighted)

    def multi_scale_grad_loss(self, prediction, target, mask):
        total = 0
        for scale in range(4):
            step = pow(2, scale)
            total += self.grad_loss(
                prediction[..., ::step, ::step],
                target[..., ::step, ::step],
                mask[..., ::step, ::step],
            )
        return total

    def grad_loss(self, prediction, target, mask, beta=1.0):
        """Huber gradient loss along the last two dims (H, W).

        Works for (B, C, H, W) and (B, C, T, H, W) inputs.
        """
        M = torch.sum(mask, dim=(-2, -1))
        diff = prediction - target
        diff = mask * diff
        # W-direction (last dim)
        grad_w = F.smooth_l1_loss(diff[..., :, 1:], diff[..., :, :-1], reduction="none", beta=beta)
        mask_w = mask[..., :, 1:] * mask[..., :, :-1]
        grad_w = mask_w * grad_w
        # H-direction (second-to-last dim)
        grad_h = F.smooth_l1_loss(diff[..., 1:, :], diff[..., :-1, :], reduction="none", beta=beta)
        mask_h = mask[..., 1:, :] * mask[..., :-1, :]
        grad_h = mask_h * grad_h

        image_loss = torch.sum(grad_w, dim=(-2, -1)) + torch.sum(grad_h, dim=(-2, -1))
        return torch.sum(image_loss) / torch.sum(M).clamp(min=1)


class TemporalDiffLoss(torch.nn.Module):
    """PV-VAE-style matched temporal-difference loss (motion-aware).

    Penalizes ``(x_hat[t] - x_hat[t-1]) - (x[t] - x[t-1])`` so the recon's
    frame-to-frame change matches GT's. Because it is a *matched*
    difference, constant ego motion appears in both terms and cancels --
    no ego warp / pose plumbing needed -- while temporally-incoherent
    jitter (recon oscillating where GT is smooth) is penalized directly.

    Two lidar-specific knobs:
      * ``raydrop_eps``: a depth pixel at the no-return clamp value
        (``min_value`` in normalized space) flickers frame-to-frame; both
        endpoints of a pair must sit above ``min_value + raydrop_eps`` for
        the pair to count. This drops the raydrop / tree-foliage flicker
        that would otherwise dominate the loss.
      * ``tau_ramp``: when the network exposes the per-sample interpolative
        noise level ``noise_tau``, scale this loss per-sample by
        ``tau ** tau_ramp_power``. Clean samples (tau=0) get ~zero weight
        so crisp motion is preserved at low noise, and the smoothness
        constraint ramps in exactly where the noisy decode would otherwise
        oscillate.
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.schedule = WeightScheduler(boundaries=config.boundaries, values=config.values)
        self.enabled = config.enabled
        self.min_value = config.min_value
        self.raydrop_eps = config.raydrop_eps
        self.tau_ramp = config.tau_ramp
        self.tau_ramp_power = config.tau_ramp_power

    def forward(self, inputs, output_batch, iteration) -> dict[str, torch.Tensor]:
        if not self.enabled or self.schedule(iteration) == 0.0:
            return dict()

        recon = output_batch[RECON_KEY]
        target = inputs[RECON_TARGET_KEY]
        if recon.ndim != 5 or recon.shape[2] < 2:
            # Needs (B, C, T, H, W) with T >= 2.
            return dict()

        d_recon = recon[:, :, 1:] - recon[:, :, :-1]
        d_target = target[:, :, 1:] - target[:, :, :-1]
        err = torch.abs(d_recon - d_target)  # (B, C, T-1, H, W)

        # Raydrop / no-return gate on the depth channel (channel 0): a pair
        # counts only if BOTH endpoints are valid returns in BOTH gt and
        # recon. Broadcasts over the channel dim.
        thr = self.min_value + self.raydrop_eps
        valid = (target[:, :1] > thr) & (recon[:, :1] > thr)  # (B, 1, T, H, W)
        valid_pair = (valid[:, :, 1:] & valid[:, :, :-1]).to(err.dtype)

        # Per-sample tau ramp: ~0 weight on clean samples, full on noisy.
        if self.tau_ramp and "noise_tau" in output_batch:
            tau = output_batch["noise_tau"].detach().clamp(min=0.0)
            w_tau = tau.pow(self.tau_ramp_power).reshape(-1, 1, 1, 1, 1)
            err = err * w_tau

        err = err * valid_pair
        denom = valid_pair.sum().clamp(min=1.0) * err.shape[1]  # × C (mean over channels)
        loss = err.sum() / denom
        loss_weighted = self.schedule(iteration) * loss
        if torch.isnan(loss_weighted).any():
            raise ValueError("[TEMPORAL_DIFF] NaN detected in loss")
        return dict(temporal_diff=loss_weighted)


class ValidityBCELoss(torch.nn.Module):
    """Binary cross-entropy on a predicted per-pixel validity (raydrop) mask.

    Matches ``L^BCE_validity`` in Sensor2Sensor (arXiv:2605.22809, Eq. 4):
    the decoder emits one extra channel of validity logits, and the target is
    the binary return mask (1 = valid return, 0 = no return). For our lidar
    range maps a no-return / raydrop pixel is clamped to ``min_value`` in the
    normalized range channel (channel 0), so the GT mask is simply
    ``range > min_value + raydrop_eps``. (Intensity / elongation terms from
    the paper are intentionally omitted.)
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.schedule = WeightScheduler(boundaries=config.boundaries, values=config.values)
        self.enabled = config.enabled
        self.min_value = config.min_value
        self.raydrop_eps = config.raydrop_eps
        # Read the GT validity from the dataloader-provided MASK_KEY instead of
        # thresholding recon_target (per-row norm breaks the depth sentinel).
        self.mask_from_batch = getattr(config, "mask_from_batch", False)
        self.invalid_class_weight = getattr(
            config, "invalid_class_weight", None
        )
        if self.invalid_class_weight is not None and not (
            0.0 <= self.invalid_class_weight <= 1.0
        ):
            raise ValueError(
                "validity_bce.invalid_class_weight must be in [0, 1], got "
                f"{self.invalid_class_weight}"
            )

    def forward(self, inputs, output_batch, iteration) -> dict[str, torch.Tensor]:
        if not self.enabled or self.schedule(iteration) == 0.0:
            return dict()
        logits = output_batch.get("validity_logits", None)
        if logits is None:
            return dict()

        if self.mask_from_batch:
            valid = inputs[MASK_KEY][:, :1].to(logits.dtype)  # 1=return
        else:
            target = inputs[RECON_TARGET_KEY]
            thr = self.min_value + self.raydrop_eps
            valid = (target[:, :1] > thr).to(logits.dtype)  # (B, 1, [T,] H, W) 1=return
        loss_map = F.binary_cross_entropy_with_logits(logits, valid, reduction="none")
        # Deterministic near-plane no-returns are culled geometrically (depth <
        # depth_min), so don't supervise the head there -- skip those pixels.
        near = output_batch.get("validity_near_mask", None)
        keep = (
            ~near.bool()
            if near is not None
            else torch.ones_like(valid, dtype=torch.bool)
        )
        if self.invalid_class_weight is not None:
            invalid_mask = keep & (valid < 0.5)
            valid_mask = keep & (valid >= 0.5)
            invalid_count = invalid_mask.sum()
            valid_count = valid_mask.sum()
            invalid_loss = (
                loss_map * invalid_mask.to(loss_map.dtype)
            ).sum() / invalid_count.clamp_min(1)
            valid_loss = (
                loss_map * valid_mask.to(loss_map.dtype)
            ).sum() / valid_count.clamp_min(1)
            invalid_weight = self.invalid_class_weight
            valid_weight = 1.0 - invalid_weight
            present_weight = (
                invalid_weight * (invalid_count > 0).to(loss_map.dtype)
                + valid_weight * (valid_count > 0).to(loss_map.dtype)
            )
            loss = (
                invalid_weight * invalid_loss + valid_weight * valid_loss
            ) / present_weight.clamp_min(torch.finfo(loss_map.dtype).eps)
        else:
            keep_float = keep.to(loss_map.dtype)
            loss = (loss_map * keep_float).sum() / keep_float.sum().clamp_min(
                1.0
            )
        loss_weighted = self.schedule(iteration) * loss
        if torch.isnan(loss_weighted).any():
            raise ValueError("[VALIDITY_BCE] NaN detected in loss")
        return dict(validity_bce=loss_weighted)


class InvalidDepthSentinelLoss(torch.nn.Module):
    """Weakly anchor no-return depth pixels to the normalized sentinel."""

    def __init__(self, config) -> None:
        super().__init__()
        self.schedule = WeightScheduler(
            boundaries=config.boundaries, values=config.values
        )
        self.enabled = config.enabled
        self.sentinel = float(config.sentinel)

    def forward(self, inputs, output_batch, iteration) -> dict[str, torch.Tensor]:
        if not self.enabled or self.schedule(iteration) == 0.0:
            return dict()
        reconstruction = output_batch[RECON_KEY][:, :1]
        valid = inputs[MASK_KEY][:, :1].to(reconstruction.dtype)
        invalid = 1.0 - valid
        error = torch.abs(reconstruction - self.sentinel)
        loss = (error * invalid).sum() / invalid.sum().clamp_min(1.0)
        loss_weighted = self.schedule(iteration) * loss
        if torch.isnan(loss_weighted).any():
            raise ValueError("[INVALID_DEPTH_SENTINEL] NaN detected in loss")
        return dict(invalid_depth_sentinel=loss_weighted)


class RaydropMSELoss(torch.nn.Module):
    """MSE on the predicted raydrop PROBABILITY (OmniGen L_r).

    OmniGen (arXiv:2512.14225) renders a raydrop probability and supervises it
    with MSE against the binary return mask (1=return, 0=no-return), unlike the
    BCE head. We keep the network emitting a validity LOGIT (so inference /
    callbacks / near-plane culling are unchanged) and apply ``sigmoid`` here,
    then MSE -- identical target derivation as ``ValidityBCELoss``.
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.schedule = WeightScheduler(boundaries=config.boundaries, values=config.values)
        self.enabled = config.enabled
        self.min_value = config.min_value
        self.raydrop_eps = config.raydrop_eps
        self.mask_from_batch = getattr(config, "mask_from_batch", False)

    def forward(self, inputs, output_batch, iteration) -> dict[str, torch.Tensor]:
        if not self.enabled or self.schedule(iteration) == 0.0:
            return dict()
        logits = output_batch.get("validity_logits", None)
        if logits is None:
            return dict()

        if self.mask_from_batch:
            valid = inputs[MASK_KEY][:, :1].to(logits.dtype)  # 1=return
        else:
            target = inputs[RECON_TARGET_KEY]
            thr = self.min_value + self.raydrop_eps
            valid = (target[:, :1] > thr).to(logits.dtype)
        prob = torch.sigmoid(logits)
        loss_map = (prob - valid) ** 2
        near = output_batch.get("validity_near_mask", None)
        if near is not None:
            keep = (~near.bool()).to(loss_map.dtype)
            loss = (loss_map * keep).sum() / keep.sum().clamp_min(1.0)
        else:
            loss = loss_map.mean()
        loss_weighted = self.schedule(iteration) * loss
        if torch.isnan(loss_weighted).any():
            raise ValueError("[RAYDROP_MSE] NaN detected in loss")
        return dict(raydrop_mse=loss_weighted)


class SurfaceConcentrationLoss(torch.nn.Module):
    """Force the renderer's per-ray weight distribution onto the GT surface.

    The differentiable depth raycast outputs, per ray, a set of sample weights
    ``w_s`` (alpha-compositing) over discrete depths ``d_s``. The rendered depth
    is the mean ``E[d]=Σ w_s d_s``; when the weights are diffuse/multimodal the
    mean lands in empty space between lumps -> "floaters". This loss penalizes
    the opacity-normalized variance of that depth distribution about the GT
    range, i.e. the (normalized) spread ``Σ w_s (d_s - d_gt)^2 / Σ w_s``. Driving
    it to 0 collapses the weights onto the true surface, so ``E[d]`` is clean
    (no eval-time median-depth hack needed) and rays terminate at the surface.

    Reads the renderer outputs the BEV models stash in ``output_batch``
    (``render_weights`` (N,R,S), ``render_sample_depths`` (S,), ``render_gt_depth``
    (N,R), ``render_valid`` (N,R)). Applied only to valid (return) rays --
    no-return rays have no surface and are left to the validity head. Squared
    distance is normalized by ``depth_scale``^2 to keep the loss O(1).
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.schedule = WeightScheduler(boundaries=config.boundaries, values=config.values)
        self.enabled = config.enabled
        self.depth_scale = config.depth_scale
        # "variance": opacity-normalized E[(d-d_gt)^2] (smooth proxy).
        # "surface_indicator": the EXACT Copilot4D Eq.5 middle term
        #   Σ_i 1(|h_i - D_gt| > eps) * w_i  (suppress weight farther than eps
        #   meters from the GT surface -> free-space + beyond-surface in one).
        self.mode = getattr(config, "mode", "variance")
        self.epsilon = getattr(config, "epsilon", 0.1)

    def forward(self, inputs, output_batch, iteration) -> dict[str, torch.Tensor]:
        if not self.enabled or self.schedule(iteration) == 0.0:
            return dict()
        w = output_batch.get("render_weights", None)          # (N, R, S)
        s = output_batch.get("render_sample_depths", None)    # (S,)
        d = output_batch.get("render_gt_depth", None)         # (N, R)
        m = output_batch.get("render_valid", None)            # (N, R)
        if w is None or s is None or d is None or m is None:
            return dict()
        m = m.to(w.dtype)
        if self.mode == "surface_indicator":
            far = (torch.abs(s.view(1, 1, -1) - d.unsqueeze(-1)) > self.epsilon).to(w.dtype)
            per_ray = (w * far).sum(-1)                        # (N, R) weight off-surface
        else:
            sq = ((s.view(1, 1, -1) - d.unsqueeze(-1)) / self.depth_scale) ** 2  # (N,R,S)
            opacity = w.sum(-1).clamp_min(1e-6)
            per_ray = (w * sq).sum(-1) / opacity              # (N, R) normalized spread
        loss = (per_ray * m).sum() / m.sum().clamp_min(1.0)
        loss_weighted = self.schedule(iteration) * loss
        if torch.isnan(loss_weighted).any():
            raise ValueError("[SURFACE_CONCENTRATION] NaN detected in loss")
        return dict(surface_concentration=loss_weighted)


class VoxelOccupancyBCELoss(torch.nn.Module):
    """Direct occupancy supervision on the decoder's 3D binary voxel logits.

    This is the genuinely-faithful Copilot4D signal: the input point cloud is
    voxelized into a binary occupancy grid, and the decoder's coarse voxel-logit
    head (used for empty-space skipping at inference) is supervised against it
    with BCE. Unlike the implicit render-only path, this gives the latent a
    direct geometric target, so occupancy isn't learned only through the
    depth/opacity of a few hundred ray samples.

    Reads ``voxel_logits`` and ``voxel_occupancy_target`` (both (N, Z, Hv, Wv))
    from the network's training output dict. ``pos_weight`` upweights the rare
    occupied voxels (occupancy is sparse -> BCE is otherwise dominated by empty
    space). No-op for nets that don't emit these keys.
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.schedule = WeightScheduler(boundaries=config.boundaries, values=config.values)
        self.enabled = config.enabled
        self.pos_weight = config.pos_weight

    def forward(self, inputs, output_batch, iteration) -> dict[str, torch.Tensor]:
        if not self.enabled or self.schedule(iteration) == 0.0:
            return dict()
        logits = output_batch.get("voxel_logits", None)
        target = output_batch.get("voxel_occupancy_target", None)
        if logits is None or target is None:
            return dict()
        pw = logits.new_tensor(self.pos_weight) if self.pos_weight > 0 else None
        loss = F.binary_cross_entropy_with_logits(logits, target.to(logits.dtype), pos_weight=pw)
        loss_weighted = self.schedule(iteration) * loss
        if torch.isnan(loss_weighted).any():
            raise ValueError("[VOXEL_OCCUPANCY_BCE] NaN detected in loss")
        return dict(voxel_occupancy_bce=loss_weighted)


class ExtraLidarDepthLoss(torch.nn.Module):
    """Multi-lidar reconstruction: depth L1 on the extra (blind-spot) lidars.

    The BEV occupancy tokenizer renders the dumped blind-spot lasers' rays (each
    with its own origin/direction in the top-sensor frame) through the SAME
    decoded occupancy volume as the top lidar. This loss supervises those rays'
    rendered depth against the measured range (both in the network's normalized
    depth space), masked to real returns (padding excluded). No-op for nets that
    don't emit these keys (e.g. at validation / non-multi-lidar runs).

    Reads ``extra_depth`` (N, K), ``extra_gt_depth`` (N, K), ``extra_valid``
    (N, K) from the network's training output dict.
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.schedule = WeightScheduler(boundaries=config.boundaries, values=config.values)
        self.enabled = config.enabled

    def forward(self, inputs, output_batch, iteration) -> dict[str, torch.Tensor]:
        if not self.enabled or self.schedule(iteration) == 0.0:
            return dict()
        pred = output_batch.get("extra_depth", None)          # (N, K)
        gt = output_batch.get("extra_gt_depth", None)         # (N, K)
        m = output_batch.get("extra_valid", None)             # (N, K)
        if pred is None or gt is None or m is None:
            return dict()
        m = m.to(pred.dtype)
        err = torch.abs(pred - gt) * m
        loss = err.sum() / m.sum().clamp_min(1.0)
        loss_weighted = self.schedule(iteration) * loss
        if torch.isnan(loss_weighted).any():
            raise ValueError("[EXTRA_LIDAR] NaN detected in loss")
        return dict(extra_lidar=loss_weighted)


class ValidityLPIPSLoss(LPIPS):
    """LPIPS perceptual loss on the predicted validity (raydrop) mask
    (Sensor2Sensor, arXiv:2605.22809 ``L^LPIPS_validity``).

    The single-channel mask is replicated to 3 channels and mapped from
    [0, 1] to [-1, 1] before the (frozen VGG) LPIPS backbone. GT target is
    the binary return mask ``range > min_value + raydrop_eps``; prediction is
    ``sigmoid(validity_logits)``. No-op when disabled / weight 0 / logits
    absent. Loads its own VGG (the standard ``perceptual`` LPIPS is a
    separate instance; for this experiment it is disabled, so this is the
    only LPIPS backbone resident).
    """

    def __init__(self, config) -> None:
        super().__init__(config.checkpoint_activations)
        self.net = self.net.eval()
        self.schedule = WeightScheduler(boundaries=config.boundaries, values=config.values)
        self.enabled = config.enabled
        self.min_value = config.min_value
        self.raydrop_eps = config.raydrop_eps
        self.mask_from_batch = getattr(config, "mask_from_batch", False)

    def forward(self, inputs, output_batch, iteration) -> dict[str, torch.Tensor]:
        if not self.enabled or self.schedule(iteration) == 0.0:
            return dict()
        logits = output_batch.get("validity_logits", None)
        if logits is None:
            return dict()

        if self.mask_from_batch:
            gt = inputs[MASK_KEY][:, :1].to(logits.dtype)  # (B, 1, [T,] H, W)
        else:
            target = inputs[RECON_TARGET_KEY]
            thr = self.min_value + self.raydrop_eps
            gt = (target[:, :1] > thr).to(logits.dtype)   # (B, 1, [T,] H, W)
        pred = torch.sigmoid(logits)                   # (B, 1, [T,] H, W)
        # Fold time into batch so the 2D VGG sees per-frame maps.
        if pred.ndim == 5:
            pred, _ = time2batch(pred)
            gt, _ = time2batch(gt)
        # [0,1] -> [-1,1], replicate mono -> RGB for the VGG backbone.
        pred3 = (pred * 2.0 - 1.0).repeat(1, 3, 1, 1)
        gt3 = (gt * 2.0 - 1.0).repeat(1, 3, 1, 1)
        val = super().forward(pred3, gt3)  # (N, 1, 1, 1) summed over layers
        loss_weighted = self.schedule(iteration) * val.mean()
        if torch.isnan(loss_weighted).any():
            raise ValueError("[VALIDITY_LPIPS] NaN detected in loss")
        return dict(validity_lpips=loss_weighted)


class IntensityTailL1Loss(torch.nn.Module):
    """Extra valid-return L1 supervision for each sensor's bright tail.

    Thresholds are specified on the raw linear [0, 1] intensity scale and
    selected by ``sensor_dataset_id``. The reconstruction error remains in the
    model's normalized intensity space. Normalizing by the number of selected
    pixels prevents the rare tail's gradient from vanishing with its frequency.
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.schedule = WeightScheduler(boundaries=config.boundaries, values=config.values)
        self.enabled = config.enabled
        self.intensity_channel = config.intensity_channel
        self.linear_thresholds = list(config.linear_thresholds)
        self.intensity_transform = config.intensity_transform

    def forward(self, inputs, output_batch, iteration) -> dict[str, torch.Tensor]:
        weight = self.schedule(iteration)
        if not self.enabled or weight == 0.0:
            return dict()

        target = inputs[RECON_TARGET_KEY]
        pred = output_batch[RECON_KEY]
        if target.shape[1] <= self.intensity_channel:
            raise ValueError(
                f"[INTENSITY_TAIL_L1] Missing intensity channel {self.intensity_channel} "
                f"in target shape {tuple(target.shape)}"
            )

        thresholds = target.new_tensor(self.linear_thresholds)
        dataset_ids = inputs.get("sensor_dataset_id")
        if dataset_ids is None and thresholds.numel() == 1:
            dataset_ids = torch.zeros(
                target.shape[0], device=target.device, dtype=torch.long
            )
        elif dataset_ids is None:
            raise ValueError(
                "[INTENSITY_TAIL_L1] sensor_dataset_id is required when "
                "multiple thresholds are configured"
            )
        else:
            dataset_ids = torch.as_tensor(
                dataset_ids, device=target.device, dtype=torch.long
            ).reshape(-1)
        if dataset_ids.numel() == 1 and target.shape[0] > 1:
            dataset_ids = dataset_ids.expand(target.shape[0])
        if dataset_ids.numel() != target.shape[0]:
            raise ValueError(
                f"[INTENSITY_TAIL_L1] Expected {target.shape[0]} dataset IDs, "
                f"got {dataset_ids.numel()}"
            )

        if (
            dataset_ids.min().item() < 0
            or dataset_ids.max().item() >= thresholds.numel()
        ):
            raise ValueError(
                f"[INTENSITY_TAIL_L1] Dataset IDs {dataset_ids.tolist()} exceed "
                f"{thresholds.numel()} configured thresholds"
            )
        threshold = thresholds[dataset_ids]
        if self.intensity_transform == "log1p":
            threshold = 2.0 * torch.log1p(threshold) / torch.log(
                threshold.new_tensor(2.0)
            ) - 1.0
        elif self.intensity_transform == "linear":
            threshold = 2.0 * threshold - 1.0
        else:
            raise ValueError(
                f"[INTENSITY_TAIL_L1] Unsupported transform: {self.intensity_transform}"
            )

        target_intensity = target[:, self.intensity_channel]
        pred_intensity = pred[:, self.intensity_channel]
        threshold = threshold.reshape(
            target.shape[0], *([1] * (target_intensity.ndim - 1))
        )
        valid = inputs[MASK_KEY][:, 0].to(torch.bool)
        tail = valid & (target_intensity >= threshold)
        tail_f = tail.to(pred_intensity.dtype)
        loss = (
            torch.abs(pred_intensity - target_intensity) * tail_f
        ).sum() / tail_f.sum().clamp_min(1.0)
        loss_weighted = weight * loss
        if torch.isnan(loss_weighted).any():
            raise ValueError("[INTENSITY_TAIL_L1] NaN detected in loss")
        return dict(intensity_tail_l1=loss_weighted)


class RepaAlignLoss(torch.nn.Module):
    """REPA / Volt-style conv-teacher alignment. The network already computes
    the masked cosine distance between the projected latent and the frozen
    Utonia (PTv3) BEV teacher features and stashes it as a scalar in the output
    dict (``repa_align``); here we just apply the weight schedule. No-op when
    the network didn't produce it (disabled / eval / non-UniPAD net)."""

    def __init__(self, config) -> None:
        super().__init__()
        self.enabled = config.enabled
        self.schedule = WeightScheduler(boundaries=config.boundaries, values=config.values)

    def forward(self, inputs, output_batch, iteration) -> dict[str, torch.Tensor]:
        val = output_batch.get("repa_align", None) if isinstance(output_batch, dict) else None
        if not self.enabled or val is None:
            return dict()
        return dict(repa_align=self.schedule(iteration) * val)


class KLLoss(torch.nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.schedule = WeightScheduler(boundaries=config.boundaries, values=config.values)

    def kl(self, mean, logvar):
        _dims = [idx for idx in range(1, mean.ndim)]
        var = torch.exp(logvar)
        return 0.5 * (torch.pow(mean, 2) + var - 1.0 - logvar)

    def forward(self, inputs, output_batch, iteration) -> dict[str, torch.Tensor]:
        if "posteriors" not in output_batch:  # No KL loss for discrete tokens.
            return dict()
        mean, logvar = output_batch["posteriors"]
        if mean.ndim == 1:  # No KL if the mean is a scalar.
            return dict()
        kl = self.kl(mean, logvar)
        kl_weighted = self.schedule(iteration) * kl
        if torch.isnan(kl_weighted).any():
            raise ValueError("[KL] NaN detected in loss")
        return dict(kl=kl_weighted)


class PerceptualLoss(LPIPS):
    """Relevant changes that're internal to us:

    - Remove linear projection layers, simply use the raw pre-normalized features.
    - Use pyramid-layer weights: [1.0 / 2.6, 1.0 / 4.8, 1.0 / 3.7, 1.0 / 5.6, 10.0 / 1.5].
    - Accepts pixel-wise masks and modulates the features before norm calculation.
    - Implements gram-matrix and correlation losses.
    """

    def __init__(self, config):
        super(PerceptualLoss, self).__init__(config.checkpoint_activations)
        self.net = self.net.eval()
        self.gram_enabled = config.gram_enabled
        self.corr_enabled = config.corr_enabled
        self.layer_weights = list(config.layer_weights)
        self.lpips_schedule = WeightScheduler(config.lpips_boundaries, config.lpips_values)
        self.gram_schedule = WeightScheduler(config.gram_boundaries, config.gram_values)
        self.corr_schedule = WeightScheduler(config.corr_boundaries, config.corr_values)
        self.checkpoint_activations = config.checkpoint_activations

    def _temporal_gram_matrix(self, x, batch_size=None):
        x = batch2time(x, batch_size)
        c, t, h, w = x.shape[-4], x.shape[-3], x.shape[-2], x.shape[-1]
        reshaped_x = torch.reshape(x, [-1, c, t * h * w])
        return torch.matmul(reshaped_x, reshaped_x.transpose(1, 2)) / float(t * h * w)

    def _gram_matrix(self, x, batch_size=None):
        if batch_size is not None and x.shape[0] != batch_size:
            return self._temporal_gram_matrix(x, batch_size)
        c, h, w = x.shape[-3], x.shape[-2], x.shape[-1]
        reshaped_x = torch.reshape(x, [-1, c, h * w])
        return torch.matmul(reshaped_x, reshaped_x.transpose(1, 2)) / float(h * w)

    def forward(self, inputs, output_batch, iteration):
        output_dict = dict()
        reconstructions = output_batch[RECON_KEY]
        weights = inputs[MASK_KEY]
        input_images = inputs[INPUT_KEY]

        if input_images.ndim == 5:
            input_images, batch_size = time2batch(input_images)
            reconstructions, _ = time2batch(reconstructions)
            weights, _ = time2batch(weights)
        else:
            batch_size = input_images.shape[0]

        in0_input, in1_input = (self.scaling_layer(input_images), self.scaling_layer(reconstructions))
        outs0, outs1 = self.net(in0_input), self.net(in1_input)

        _layer_weights = self.layer_weights
        weights_map, res, diffs = {}, {}, {}
        for kk in range(len(self.chns)):
            weights_map[kk] = torch.nn.functional.interpolate(weights[:, :1, ...], outs0[kk].shape[-2:])
            diffs[kk] = weights_map[kk] * torch.abs(outs0[kk] - outs1[kk])
            res[kk] = _layer_weights[kk] * diffs[kk].mean([1, 2, 3], keepdim=True)

        val = res[0]
        for ll in range(1, len(self.chns)):
            val += res[ll]
        # Scale by number of pixels to match pixel-wise losses.
        val = val.expand(-1, input_images.shape[-3], input_images.shape[-2], input_images.shape[-1])
        if batch_size != input_images.shape[0]:
            val = batch2time(val, batch_size)
        if torch.isnan(val).any():
            raise ValueError("[LPIPS] NaN detected in loss")
        output_dict["lpips"] = self.lpips_schedule(iteration) * val

        if self.gram_enabled and self.gram_schedule(iteration) > 0.0:
            num_chans = len(self.chns)
            grams0 = [self._gram_matrix(weights_map[kk] * outs0[kk], batch_size) for kk in range(num_chans)]
            grams1 = [self._gram_matrix(weights_map[kk] * outs1[kk], batch_size) for kk in range(num_chans)]
            gram_diffs = [(grams0[kk] - grams1[kk]) ** 2 for kk in range(num_chans)]
            grams_res = [_layer_weights[kk] * gram_diffs[kk].mean([1, 2], keepdim=True) for kk in range(num_chans)]
            gram_val = grams_res[0]
            for ll in range(1, len(self.chns)):
                gram_val += grams_res[ll]

            # Scale by number of total pixels to match pixel-wise losses.
            gram_val = gram_val.unsqueeze(1).expand(
                -1, input_images.shape[-3], input_images.shape[-2], input_images.shape[-1]
            )
            if batch_size != input_images.shape[0]:
                gram_val = batch2time(gram_val, batch_size)
            if torch.isnan(gram_val).any():
                raise ValueError("[GRAM] NaN detected in loss")
            output_dict["gram"] = self.gram_schedule(iteration) * gram_val
        return output_dict

    def torch_compile(self):
        """
        This method invokes torch.compile() on this loss
        """
        # cuda-graphs crash after 1k iterations
        self.net = torch.compile(self.net, dynamic=False)


class FlowLoss(torch.nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.schedule = WeightScheduler(config.boundaries, config.values)
        self.scale = config.scale
        self.dtype = getattr(torch, config.dtype)
        self.checkpoint_activations = config.checkpoint_activations
        self.enabled = config.enabled

        current_device = torch.device(torch.cuda.current_device())

        # In order to be able to run model in bf16 we need to change make_coords_grid()
        # to allow it to return arbitrary type provided by us in argument
        # the line from orginal implementation that caused results to be only fp32 is commented
        # Additionally I've changed that function to run on GPU instead of CPU, which results in
        # less graph breaks when torch.compile() is used
        # This function is copied from
        # https://github.com/pytorch/vision/blob/main/torchvision/models/optical_flow/_utils.py#L22
        # commit: b06ea39d5f0adbe949d08257837bda912339e415
        def make_coords_grid(
            batch_size: int, h: int, w: int, device: torch.device = current_device, dtype: torch.dtype = self.dtype
        ):
            # Original: def make_coords_grid(batch_size: int, h: int, w: int, device: str = "cpu"):
            device = torch.device(device)
            coords = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
            coords = torch.stack(coords[::-1], dim=0).to(dtype)
            # Original: coords = torch.stack(coords[::-1], dim=0).float()
            return coords[None].repeat(batch_size, 1, 1, 1)

        # We also need to specify output dtype of torch.linspace() in index_pyramid()
        # method of CorrBlock, otherwise it uses default fp32 dtype as output.
        # Additionally I've changed that function to run on GPU instead of CPU, which results in
        # less graph breaks when torch.compile() is used
        # This function is copied from
        # https://github.com/pytorch/vision/blob/main/torchvision/models/optical_flow/raft.py#L394
        # commit: b06ea39d5f0adbe949d08257837bda912339e415
        def index_pyramid(
            self, centroids_coords, dtype: torch.dtype = self.dtype, device: torch.device = current_device
        ):
            # Original: def index_pyramid(self, centroids_coords):
            """Return correlation features by indexing from the pyramid."""
            neighborhood_side_len = 2 * self.radius + 1  # see note in __init__ about out_channels
            di = torch.linspace(-self.radius, self.radius, neighborhood_side_len, dtype=dtype, device=device)
            dj = torch.linspace(-self.radius, self.radius, neighborhood_side_len, dtype=dtype, device=device)
            # Original: di = torch.linspace(-self.radius, self.radius, neighborhood_side_len)
            # Original: dj = torch.linspace(-self.radius, self.radius, neighborhood_side_len)
            delta = torch.stack(torch.meshgrid(di, dj, indexing="ij"), dim=-1).to(centroids_coords.device)
            delta = delta.view(1, neighborhood_side_len, neighborhood_side_len, 2)

            batch_size, _, h, w = centroids_coords.shape  # _ = 2
            centroids_coords = centroids_coords.permute(0, 2, 3, 1).reshape(batch_size * h * w, 1, 1, 2)

            indexed_pyramid = []
            for corr_volume in self.corr_pyramid:
                sampling_coords = centroids_coords + delta  # end shape is (batch_size * h * w, side_len, side_len, 2)
                indexed_corr_volume = optical_flow.raft.grid_sample(
                    corr_volume, sampling_coords, align_corners=True, mode="bilinear"
                ).view(batch_size, h, w, -1)
                indexed_pyramid.append(indexed_corr_volume)
                centroids_coords = centroids_coords / 2

            corr_features = torch.cat(indexed_pyramid, dim=-1).permute(0, 3, 1, 2).contiguous()

            expected_output_shape = (batch_size, self.out_channels, h, w)
            if corr_features.shape != expected_output_shape:
                raise ValueError(
                    f"Output shape of index pyramid is incorrect. Should be {expected_output_shape}, got {corr_features.shape}"
                )

            return corr_features

        optical_flow.raft.make_coords_grid = make_coords_grid
        optical_flow.raft.CorrBlock.index_pyramid = index_pyramid

        flow_model = optical_flow.raft_large(pretrained=True, progress=False)
        flow_model.requires_grad_(False)
        flow_model.eval()
        flow_model = flow_model.to(self.dtype)

        self.flow_model = flow_model

    def _run_model(self, input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
        """Runs flow_model in the forward mode on explicit dtype=float32.

        Args:
            input1: First video frames batch, layout (T, C, H, W), bfloat16.
            input2: Next video frames batch, layout (T, C, H, W), bfloat16.

        Returns:
            Forward optical flow, (T, 2, H, W), bfloat16.
        """
        input_dtype = input1.dtype
        flow_output = self.flow_model.to(self.dtype)(input1.to(self.dtype), input2.to(self.dtype))[-1]
        return flow_output.to(input_dtype)

    def _run_model_fwd(self, input_video: torch.Tensor) -> torch.Tensor:
        """Runs foward flow on a batch of videos, one batch at a time.
        Args:
            input_video: The input batch of videos, layout (B, T, C, H, W).

        Returns:
            Forward optical flow, layout (B, 2, T-1, H, W).
        """
        output_list = list()
        for fwd_input_frames in input_video:
            fwd_input_frames = fwd_input_frames.transpose(1, 0)
            fwd_flow_output = self._run_model(fwd_input_frames[:-1], fwd_input_frames[1:])
            output_list.append(fwd_flow_output.transpose(1, 0))
        return torch.stack(output_list, dim=0)

    def _bidirectional_flow(self, input_video: torch.Tensor) -> torch.Tensor:
        """The bidirectional optical flow on a batch of videos.

        The forward and backward flows are averaged to get the bidirectional flow.
        To reduce memory pressure, the input video is scaled down by a factor of `self.scale`,
        and rescaled back to match other pixel-wise losses.

        Args:
            input_video: The input batch of videos, layout (B, T, C, H, W).

        Returns:
            Biderectinoal flow, layout (B, 2, T-1, H, W).
        """
        # scale down the input video to reduce memory pressure.
        t, h, w = input_video.shape[-3:]
        input_video_scaled = F.interpolate(input_video, (t, h // self.scale, w // self.scale), mode="trilinear")

        # forward flow.
        if self.checkpoint_activations:
            fwd_flow_output = checkpoint.checkpoint(self._run_model_fwd, input_video_scaled, use_reentrant=False)
        else:
            fwd_flow_output = self._run_model_fwd(input_video_scaled)

        # backward flow.
        input_video_scaled = input_video_scaled.flip([2])
        if self.checkpoint_activations:
            bwd_flow_output = checkpoint.checkpoint(self._run_model_fwd, input_video_scaled, use_reentrant=False)
        else:
            bwd_flow_output = self._run_model_fwd(input_video_scaled)
        bwd_flow_output = bwd_flow_output.flip([2])

        # bidirectional flow, concat fwd and bwd along temporal axis.
        flow_input = torch.cat([fwd_flow_output, bwd_flow_output], dim=2)
        return self.scale * F.interpolate(flow_input, (2 * (t - 1), h, w), mode="trilinear")

    def forward(
        self, inputs: dict[str, torch.Tensor], output_batch: dict[str, torch.Tensor], iteration: int
    ) -> dict[str, torch.Tensor]:
        input_images = inputs[INPUT_KEY]
        if input_images.ndim == 4 or input_images.shape[2] == 1:
            return dict()
        if not self.enabled or self.schedule(iteration) == 0.0:
            return dict()

        # Biderectional flow (B, 2, 2*(T-1), H, W)
        flow_input = self._bidirectional_flow(input_images)
        flow_recon = self._bidirectional_flow(output_batch[RECON_KEY])

        # L1 loss on the flow. (B, 1, 2*(T-1), H, W)
        flow_loss = torch.abs(flow_input - flow_recon).mean(dim=1, keepdim=True)

        flow_loss_weighted = self.schedule(iteration) * flow_loss
        if torch.isnan(flow_loss_weighted).any():
            raise ValueError("[FLOW] NaN detected in loss")
        return dict(flow=flow_loss_weighted)

    def torch_compile(self):
        """
        This method invokes torch.compile() on this loss
        """
        self.flow_model = torch.compile(self.flow_model, dynamic=False)


class VideoConsistencyLoss(torch.nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.schedule = WeightScheduler(boundaries=config.boundaries, values=config.values)
        self.enabled = config.enabled
        self.num_frames = config.num_frames
        self.step = config.step
        self.num_windows = None

    def shuffle(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        For input video of [B, 3, T, H, W], this function will reshape the video to
        the shape of [B*(T-num_frames+1)//step, 3, num_frames, H, W] using a sliding window
        This function is used to compute the temporal consistency between overlapped frames
        to enable temporal consistency
        """
        assert len(inputs.shape) == 5, f"inputs shape should be [B, 3, T, H, W]. currently {inputs.shape}"
        B, C, T, H, W = inputs.shape
        assert T >= self.num_frames, f"inputs {T} should be greater than {self.num_frames}"

        # [B, C, num_windows, H, W, num_frames]
        outputs = inputs.unfold(dimension=2, size=self.num_frames, step=self.step)
        self.num_windows = outputs.shape[2]
        outputs = einops.rearrange(outputs, "b c m h w n -> (b m) c n h w")

        return outputs

    def forward(self, inputs, output_batch, iteration) -> dict[str, torch.Tensor]:
        if not self.enabled or self.num_windows is None:
            return dict()
        if self.schedule(iteration) == 0.0:
            return dict()
        # reshape output_batch to compute loss between overlapped frames
        reconstructions = output_batch[RECON_CONSISTENCY_KEY]
        B, C, T, H, W = reconstructions.shape

        assert T == self.num_frames, f"reconstruction shape invalid (shape[2] should be {self.num_frames})"
        assert (
            B % self.num_windows == 0
        ), f"reconstruction shape invalid (shape[0]={B} not dividable by {self.num_windows})"

        B = B // self.num_windows
        videos = reconstructions.view(B, self.num_windows, C, self.num_frames, H, W)

        # Compute the L1 distance between overlapped frames for all windows at once
        diff = torch.mean(torch.abs(videos[:, :-1, :, self.step :, :, :] - videos[:, 1:, :, : -self.step, :, :]))
        diff_weighted = self.schedule(iteration) * diff

        if LATENT_KEY not in output_batch:
            return dict(frame_consistency=diff_weighted)

        B_latent, C_latent, T_latent, H_latent, W_latent = output_batch["latent"].shape
        assert B_latent % self.num_windows == 0, f"latent batches should be divisible by {self.num_windows}"

        latents = output_batch[LATENT_KEY].view(
            B_latent // self.num_windows, self.num_windows, C_latent, T_latent, H_latent, W_latent
        )
        temporal_rate = self.num_frames // T_latent
        spatial_rate = (H // H_latent) * (W // W_latent)
        step_latent = self.step // temporal_rate
        latent_diff = torch.mean(
            torch.abs(latents[:, :-1, :, step_latent:, :, :] - latents[:, 1:, :, :-step_latent, :, :])
        )
        latent_diff_weighted = self.schedule(iteration) * latent_diff * (C * temporal_rate * spatial_rate) / (C_latent)
        return dict(frame_consistency=diff_weighted, latent_consistency=latent_diff_weighted)

    def unshuffle(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        For input video of [B*num_windows, 3, num_frames, H, W], this function will
        undo the shuffle to a tensor of shape [B, 3, T, H, W]
        """
        assert len(inputs.shape) == 5, f"inputs shape should be [B, 3, T, H, W]. currently {inputs.shape}"
        B, C, T, H, W = inputs.shape
        assert T == self.num_frames, f"inputs shape invalid (shape[2] should be {self.num_frames})"
        assert B % self.num_windows == 0, f"inputs shape invalid (shape[0]={B} not dividable by {self.num_windows})"

        B = B // self.num_windows
        videos = inputs.view(B, self.num_windows, C, self.num_frames, H, W)

        T = self.num_frames + (self.num_windows - 1) * self.step
        current_device = torch.device(torch.cuda.current_device())
        outputs = torch.zeros(B, C, T, H, W).to(inputs.dtype).to(current_device)
        counter = torch.zeros_like(outputs)
        for i in range(self.num_windows):
            outputs[:, :, i * self.step : i * self.step + self.num_frames, :, :] += videos[:, i, :, :, :, :]
            counter[:, :, i * self.step : i * self.step + self.num_frames, :, :] += 1
        outputs = outputs / counter

        return outputs
