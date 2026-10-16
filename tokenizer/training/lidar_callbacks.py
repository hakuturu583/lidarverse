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

"""Tokenizer callbacks extended from base callbacks."""

import io
import math
import time
from typing import Any

from omegaconf import OmegaConf
import imageio
import numpy as np
import torch.distributed as dist
import torch
import torch.nn.functional as F
import torchvision
import wandb
from PIL import Image
from torch._dynamo.eval_frame import OptimizedModule as torch_OptimizedModule

from utils.config import Config
from utils.model import Model
from utils.trainer import Trainer
from utils import callback, distributed, log, wandb_utils
from tokenizer.training.callbacks import make_video_grid
from tokenizer.training.datasets.lidar_datasets.configs import COMMONDATA_CONFIG
from utils.per_row_norm import get_per_row_normalizer, invert_per_row_norm
from tokenizer.training.model import PREDICTION

_UINT8_MAX_F = float(np.iinfo(np.uint8).max)

from utils.lidar_rangemap import (
    undo_row_col_temporal_repeat,
    unnormalize_and_reduce_channels,
    unnormalize_intensity,
    colorcode_depth_maps,
)
from utils.visualize.point_cloud import (
    render_range_map_to_point_cloud,
    render_range_map_to_point_cloud_with_intensity,
    vis_point_cloud_to_numpy,
)

# Per-lidar viz palette: top lidar gray, blind-spot lidars distinct bold colors
# (order = local laser index in the dumped NPZ, typically FRONT / SIDE_LEFT /
# SIDE_RIGHT / REAR).
_TOP_COLOR = np.array([0.55, 0.55, 0.55], dtype=np.float32)
_LASER_PALETTE = np.array(
    [
        [1.00, 0.20, 0.20],  # red
        [0.20, 0.90, 0.30],  # green
        [0.25, 0.55, 1.00],  # blue
        [1.00, 0.70, 0.10],  # orange
        [0.85, 0.35, 1.00],  # purple
        [0.10, 0.85, 0.85],  # cyan
    ],
    dtype=np.float32,
)


def wandb_range_image(range_map_input, range_map_recon, ray_directions=None):
    # range_map_input: [N,  H, W]
    # range_map_recon: [N,  H, W]

    device = range_map_input.device
    vis = render_range_map_to_point_cloud(
        range_map_input.cpu().numpy(), range_map_recon.cpu().numpy(), filter_outlier=False,
        ray_directions=ray_directions,
    )  # shape: (bs, H, W, 3), uint8

    vis_images = torch.from_numpy(vis).float() / _UINT8_MAX_F
    vis_images = vis_images.permute(0, 3, 1, 2).to(device)  # shape: (bs, 3, H, W)

    image_grid = torchvision.utils.make_grid(vis_images.clamp_(0, 1), nrow=1, pad_value=0, normalize=False)
    image_grid = image_grid.permute(1, 2, 0).cpu().detach().numpy()
    pil_image = Image.fromarray((image_grid * _UINT8_MAX_F).astype("uint8"))
    return wandb.Image(pil_image)


def wandb_range_video(range_map_input, range_map_recon, fps=10, ray_directions=None):
    # range_map_input: [N, T, H, W]
    # range_map_recon: [N, T, H, W]

    device = range_map_input.device
    vis = render_range_map_to_point_cloud(
        range_map_input.squeeze(0).cpu().numpy(), range_map_recon.squeeze(0).cpu().numpy(), filter_outlier=False,
        ray_directions=ray_directions,
    )  # shape: (t, H, W, 3), uint8
    vis_videos = torch.from_numpy(vis).float() / _UINT8_MAX_F
    vis_videos = vis_videos.unsqueeze(0).permute(0, -1, 1, 2, 3).to(device)  # shape: (1, 3, t, H, W)

    video_grid = make_video_grid(vis_videos.clamp_(0, 1), nrow=vis_videos.shape[0], padding=0)
    mem_file = io.BytesIO()
    imageio.mimsave(mem_file, video_grid, "mp4", fps=fps, codec="libx264")
    mem_file.seek(0)
    return wandb.Video(mem_file, fps=fps, format="mp4")


@torch.no_grad()
def _fit_pca_basis(fv: torch.Tensor):
    """fv: (M, D) valid features -> (mean (D,), V (D, 3)) top-3 PCA basis."""
    if fv.shape[0] > 200_000:
        idx = torch.randperm(fv.shape[0], device=fv.device)[:200_000]
        fv = fv[idx]
    mean = fv.mean(0)
    _, _, Vt = torch.pca_lowrank(fv - mean, q=3, niter=4)
    return mean, Vt[:, :3]


@torch.no_grad()
def _apply_pca_rgb(feat: torch.Tensor, valid: torch.Tensor, mean: torch.Tensor,
                   V: torch.Tensor, n_sigma: float = 2.0, std3=None):
    """feat (D,T,H,W), valid (T,H,W) bool -> (T,H,W,3) uint8 (invalid=black).

    Projects ``feat`` onto the supplied basis ``(mean, V)`` and normalizes each
    PC channel by ``n_sigma * std`` (std fit on valid entries, or reuse ``std3``
    so two maps share the same color scale)."""
    D = feat.shape[0]
    u = torch.einsum("dthw,dk->kthw", feat - mean.view(D, 1, 1, 1), V)  # (3,T,H,W)
    u_np = u.float().cpu().numpy()
    vm = valid.cpu().numpy()
    if std3 is None:
        std3 = np.stack([u_np[k][vm].std() if vm.any() else 1.0 for k in range(3)])
    rgb = np.zeros(u_np.shape[1:] + (3,), dtype=np.uint8)
    for k in range(3):
        n = np.clip(u_np[k] / (n_sigma * (std3[k] + 1e-6)), -1.0, 1.0)
        rgb[..., k] = (0.5 * (n + 1.0) * 255.0).astype(np.uint8)
    rgb[~vm] = 0
    return rgb, std3


def _upsample_nn_rgb(rgb: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return rgb
    return rgb.repeat(factor, axis=1).repeat(factor, axis=2)


def wandb_rgb_media(rgb: np.ndarray, fps: int = 10):
    """rgb: (T, H, W, 3) uint8 -> wandb.Image (T==1) or wandb.Video."""
    if rgb.shape[0] == 1:
        return wandb.Image(Image.fromarray(rgb[0]))
    mem_file = io.BytesIO()
    imageio.mimsave(mem_file, rgb, "mp4", fps=fps, codec="libx264")
    mem_file.seek(0)
    return wandb.Video(mem_file, fps=fps, format="mp4")


def wandb_image(images, iteration):
    vis_images = (images + 1) / 2  # [-1,1] to [0,1]
    image_grid = torchvision.utils.make_grid(vis_images.clamp_(0, 1), nrow=1, pad_value=0, normalize=False)
    image_grid = image_grid.permute(1, 2, 0).cpu().detach().numpy()
    pil_image = Image.fromarray((image_grid * _UINT8_MAX_F).astype("uint8"))
    return wandb.Image(pil_image)


def wandb_video(videos, iteration, fps=10):
    # videos: [B, C, T, H, W]
    vis_videos = (videos + 1.0) / 2.0
    video_grid = make_video_grid(vis_videos.clamp_(0, 1), nrow=vis_videos.shape[0], padding=0)
    mem_file = io.BytesIO()
    imageio.mimsave(mem_file, video_grid, "mp4", fps=fps, codec="libx264")
    mem_file.seek(0)
    return wandb.Video(mem_file, fps=fps, format="mp4")


def wandb_intensity_image(intensity_gt, intensity_recon):
    """Visualize intensity maps as colormapped images (viridis). Input: [N, H, W] in [0,1]."""
    from matplotlib import cm
    cmap = cm.get_cmap("viridis")

    gt_rgb = torch.from_numpy(cmap(intensity_gt.cpu().clamp(0, 1).numpy())[..., :3]).float()
    re_rgb = torch.from_numpy(cmap(intensity_recon.cpu().clamp(0, 1).numpy())[..., :3]).float()

    paired = torch.cat([gt_rgb, re_rgb], dim=-2)  # [N, H, 2W, 3]
    paired = paired.permute(0, 3, 1, 2)  # [N, 3, H, 2W]

    grid = torchvision.utils.make_grid(paired.clamp_(0, 1), nrow=1, pad_value=0, normalize=False)
    grid = grid.permute(1, 2, 0).cpu().numpy()
    return wandb.Image(Image.fromarray((grid * _UINT8_MAX_F).astype("uint8")))


def wandb_intensity_video(intensity_gt, intensity_recon, fps=10):
    """Visualize intensity maps as colormapped video. Input: [N, T, H, W] in [0,1]."""
    from matplotlib import cm
    cmap = cm.get_cmap("viridis")

    gt_rgb = torch.from_numpy(cmap(intensity_gt.squeeze(0).cpu().clamp(0, 1).numpy())[..., :3]).float()
    re_rgb = torch.from_numpy(cmap(intensity_recon.squeeze(0).cpu().clamp(0, 1).numpy())[..., :3]).float()

    paired = torch.cat([gt_rgb, re_rgb], dim=-2)  # [T, H, 2W, 3]
    paired = paired.unsqueeze(0).permute(0, 4, 1, 2, 3)  # [1, 3, T, H, 2W]

    video_grid = make_video_grid(paired.clamp_(0, 1), nrow=1, padding=0)
    mem_file = io.BytesIO()
    imageio.mimsave(mem_file, video_grid, "mp4", fps=fps, codec="libx264")
    mem_file.seek(0)
    return wandb.Video(mem_file, fps=fps, format="mp4")


def wandb_range_image_intensity(range_map_input, range_map_recon, intensity_input, intensity_recon, ray_directions=None):
    """Point cloud visualization colored by intensity. Input: all [N, H, W] numpy."""
    vis = render_range_map_to_point_cloud_with_intensity(
        range_map_input, range_map_recon, intensity_input, intensity_recon, filter_outlier=False,
        ray_directions=ray_directions,
    )  # uint8
    vis_images = torch.from_numpy(vis).float() / _UINT8_MAX_F
    vis_images = vis_images.permute(0, 3, 1, 2)
    grid = torchvision.utils.make_grid(vis_images.clamp_(0, 1), nrow=1, pad_value=0, normalize=False)
    grid = grid.permute(1, 2, 0).cpu().numpy()
    return wandb.Image(Image.fromarray((grid * _UINT8_MAX_F).astype("uint8")))


def wandb_range_video_intensity(range_map_input, range_map_recon, intensity_input, intensity_recon, fps=10, ray_directions=None):
    """Point cloud video colored by intensity. Input: all [N, T, H, W] numpy."""
    vis = render_range_map_to_point_cloud_with_intensity(
        range_map_input.squeeze(0), range_map_recon.squeeze(0),
        intensity_input.squeeze(0), intensity_recon.squeeze(0), filter_outlier=False,
        ray_directions=ray_directions,
    )  # uint8
    vis_videos = torch.from_numpy(vis).float() / _UINT8_MAX_F
    vis_videos = vis_videos.unsqueeze(0).permute(0, -1, 1, 2, 3)
    video_grid = make_video_grid(vis_videos.clamp_(0, 1), nrow=1, padding=0)
    mem_file = io.BytesIO()
    imageio.mimsave(mem_file, video_grid, "mp4", fps=fps, codec="libx264")
    mem_file.seek(0)
    return wandb.Video(mem_file, fps=fps, format="mp4")


def wandb_mask_image(mask_gt, mask_pred):
    """Side-by-side validity mask (grayscale). Input: [N, H, W] in [0,1]
    (white = valid return, black = no-return). ``mask_gt`` is the binary GT
    return mask; ``mask_pred`` is the decoder's predicted validity prob."""
    gt = mask_gt.cpu().clamp(0, 1).unsqueeze(-1).repeat(1, 1, 1, 3)   # [N,H,W,3]
    pr = mask_pred.cpu().clamp(0, 1).unsqueeze(-1).repeat(1, 1, 1, 3)
    paired = torch.cat([gt, pr], dim=-2).permute(0, 3, 1, 2)          # [N,3,H,2W]
    grid = torchvision.utils.make_grid(paired, nrow=1, pad_value=0, normalize=False)
    grid = grid.permute(1, 2, 0).cpu().numpy()
    return wandb.Image(Image.fromarray((grid * _UINT8_MAX_F).astype("uint8")))


def wandb_mask_video(mask_gt, mask_pred, fps=10):
    """Side-by-side validity mask video (grayscale). Input: [N, T, H, W]."""
    gt = mask_gt.squeeze(0).cpu().clamp(0, 1).unsqueeze(-1).repeat(1, 1, 1, 3)  # [T,H,W,3]
    pr = mask_pred.squeeze(0).cpu().clamp(0, 1).unsqueeze(-1).repeat(1, 1, 1, 3)
    paired = torch.cat([gt, pr], dim=-2).unsqueeze(0).permute(0, 4, 1, 2, 3)    # [1,3,T,H,2W]
    video_grid = make_video_grid(paired.clamp_(0, 1), nrow=1, padding=0)
    mem_file = io.BytesIO()
    imageio.mimsave(mem_file, video_grid, "mp4", fps=fps, codec="libx264")
    mem_file.seek(0)
    return wandb.Video(mem_file, fps=fps, format="mp4")


def _multilidar_colored_points(
    top_pts, ex_pts, ex_valid, ex_laser, ex_keep=None
):
    """Stack top + per-laser-colored blind-spot points into (P,3) xyz + (P,3) rgb.

    top_pts: (Nt,3) top-lidar points (already validity-filtered) or None.
    ex_pts: (K,3) extra points; ex_valid: (K,) real-return mask; ex_laser: (K,)
    local laser index (-1 = padding); ex_keep: extra per-ray keep mask (e.g.
    opacity>=thr) AND-ed with ex_valid, or None."""
    pts_list, col_list = [], []
    if top_pts is not None and top_pts.shape[0] > 0:
        pts_list.append(top_pts)
        col_list.append(np.tile(_TOP_COLOR, (top_pts.shape[0], 1)))
    if ex_pts is not None:
        m = ex_valid.astype(bool)
        if ex_keep is not None:
            m = m & ex_keep.astype(bool)
        if m.any():
            ep = ex_pts[m]
            el = ex_laser[m].astype(np.int64)
            ec = _LASER_PALETTE[np.clip(el, 0, len(_LASER_PALETTE) - 1)]
            pts_list.append(ep)
            col_list.append(ec)
    if not pts_list:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32)
    return (
        np.concatenate(pts_list, 0).astype(np.float32),
        np.concatenate(col_list, 0).astype(np.float32),
    )


def _multilidar_pcd_frame(
    top_gt, top_recon, ex_gt, ex_recon, ex_valid, ex_laser, ex_keep=None,
):
    """One (H, 2W, 3) uint8 frame: perspective point-cloud render (GT | recon)
    in the same front view as ``val/range_map``, top lidar gray + each
    blind-spot lidar a distinct color, overlaid in ONE scene."""
    g_pts, g_col = _multilidar_colored_points(top_gt, ex_gt, ex_valid, ex_laser)
    r_pts, r_col = _multilidar_colored_points(
        top_recon, ex_recon, ex_valid, ex_laser, ex_keep=ex_keep
    )
    img_g = vis_point_cloud_to_numpy(g_pts, g_col)
    img_r = vis_point_cloud_to_numpy(r_pts, r_col)
    return np.concatenate([img_g, img_r], axis=1).astype(np.uint8)


def _fig_to_rgb(fig) -> np.ndarray:
    """Rasterize a Matplotlib fig to a FIXED-size (figsize*dpi) (H,W,3) uint8
    array via the Agg canvas buffer -- fixed size (no bbox_inches='tight') so
    per-frame arrays stack into a valid mp4."""
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
    return buf[..., :3].copy()


def _frames_to_media(frames, fps=4):
    """Pack a list of equal-size (H,W,3) uint8 frames into a wandb.Video (mp4)
    when >1 frame, else a single wandb.Image."""
    if len(frames) == 1:
        return wandb.Image(Image.fromarray(frames[0]))
    mem_file = io.BytesIO()
    imageio.mimsave(mem_file, np.stack(frames, 0), "mp4", fps=fps, codec="libx264")
    mem_file.seek(0)
    return wandb.Video(mem_file, fps=fps, format="mp4")


def _multilidar_bev_frame(
    top_gt, top_recon, ex_gt, ex_recon, ex_valid, ex_laser, ex_keep=None,
    laser_names=None, R=None,
):
    """One BEV (x-y, top) + side (x-z, bottom) scatter frame, GT (left) |
    recon (right), as an (H,W,3) uint8 array.

    Top lidar gray for context; each blind-spot lidar a distinct color. Auto-
    zoomed to the extra-lidar extent (pass a fixed ``R`` for a stable video).
    ``ex_keep`` (opacity>=thr) additionally masks the recon extras."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    vm = ex_valid.astype(bool)
    eg = ex_gt[vm]
    el = ex_laser[vm].astype(np.int64)
    rm = vm & (ex_keep.astype(bool) if ex_keep is not None else np.ones_like(vm))
    er = ex_recon[rm]
    erl = ex_laser[rm].astype(np.int64)

    # Zoom to the blind-spot extent (they're near-range), padded; clamp [15, 80].
    if R is None:
        if eg.shape[0] > 0:
            R = float(np.percentile(np.abs(eg[:, :2]), 99)) * 1.3
        else:
            R = 40.0
        R = float(np.clip(R, 15.0, 80.0))

    fig, axes = plt.subplots(2, 2, figsize=(12, 12), dpi=100)

    def _scatter(ax, top, ex, exl, a, b):
        if top is not None and top.shape[0] > 0:
            ax.scatter(top[:, a], top[:, b], c=[_TOP_COLOR], s=0.4, alpha=0.35)
        if ex.shape[0] > 0:
            cols = _LASER_PALETTE[np.clip(exl, 0, len(_LASER_PALETTE) - 1)]
            ax.scatter(ex[:, a], ex[:, b], c=cols, s=1.5)

    _scatter(axes[0, 0], top_gt, eg, el, 0, 1)
    _scatter(axes[0, 1], top_recon, er, erl, 0, 1)
    _scatter(axes[1, 0], top_gt, eg, el, 0, 2)
    _scatter(axes[1, 1], top_recon, er, erl, 0, 2)

    axes[0, 0].set_title(f"BEV GT (x-y)  [{eg.shape[0]} extra pts]", fontsize=11)
    axes[0, 1].set_title("BEV recon (x-y)", fontsize=11)
    axes[1, 0].set_title("side GT (x-z)", fontsize=11)
    axes[1, 1].set_title("side recon (x-z)", fontsize=11)
    for ax in axes[0]:
        ax.set_xlim(-R, R); ax.set_ylim(-R, R); ax.set_aspect("equal")
    for ax in axes[1]:
        ax.set_xlim(-R, R); ax.set_ylim(-5, 8); ax.set_aspect("equal")
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([]); ax.set_facecolor("black")

    # Legend (top + per-laser).
    n_lasers = int(ex_laser[vm].max()) + 1 if vm.any() else 0
    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=_TOP_COLOR,
                      markersize=6, label="top")]
    for li in range(n_lasers):
        name = laser_names[li] if (laser_names is not None and li < len(laser_names)) else f"laser {li}"
        handles.append(Line2D([0], [0], marker="o", color="w",
                              markerfacecolor=_LASER_PALETTE[li % len(_LASER_PALETTE)],
                              markersize=6, label=str(name)))
    axes[0, 0].legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.6)
    fig.tight_layout()
    arr = _fig_to_rgb(fig)
    plt.close(fig)
    return arr


# Default blind-spot laser names (dump order in scripts/dump_waymo_lidar_extra.py).
_EXTRA_LASER_NAMES = ("FRONT", "SIDE_LEFT", "SIDE_RIGHT", "REAR")


def _multilidar_range_frame(
    gt_depth, recon_depth, pred_valid, laser, row, col, laser_hw, max_range=75.0,
):
    """One (H,W,3) uint8 frame: per-blind-spot-lidar range image + mask. Rows =
    lasers; cols = [GT range | recon range | GT mask | predicted-validity mask].
    The extra lidars are sparse returns, so each laser's (H,W) grid is scatter-
    filled at its measured (row,col) pixels (GT + recon depth at SAME pixels)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    L = int(laser_hw.shape[0])
    fig, axes = plt.subplots(L, 4, figsize=(18, 2.6 * L + 0.5), dpi=100, squeeze=False)
    col_titles = ["GT range", "recon range", "GT mask", "pred validity"]
    for li in range(L):
        H, W = int(laser_hw[li, 0]), int(laser_hw[li, 1])
        gt_img = np.full((H, W), np.nan, np.float32)
        rc_img = np.full((H, W), np.nan, np.float32)
        gm = np.zeros((H, W), np.float32)
        pm = np.zeros((H, W), np.float32)
        idx = laser == li
        r, c = row[idx], col[idx]
        keep = (r >= 0) & (c >= 0) & (r < H) & (c < W)
        r, c = r[keep], c[keep]
        gt_img[r, c] = gt_depth[idx][keep]
        rc_img[r, c] = recon_depth[idx][keep]
        gm[r, c] = 1.0
        pm[r, c] = pred_valid[idx][keep].astype(np.float32)
        name = _EXTRA_LASER_NAMES[li] if li < len(_EXTRA_LASER_NAMES) else f"laser {li}"
        for j, (img, cmap, vmax) in enumerate(
            [(gt_img, "turbo", max_range), (rc_img, "turbo", max_range),
             (gm, "gray", 1.0), (pm, "gray", 1.0)]
        ):
            ax = axes[li, j]
            ax.imshow(img, cmap=cmap, vmin=0, vmax=vmax, aspect="auto", interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            if li == 0:
                ax.set_title(col_titles[j], fontsize=11)
            if j == 0:
                ax.set_ylabel(f"{name}\n({H}x{W})", fontsize=10)
    fig.tight_layout()
    arr = _fig_to_rgb(fig)
    plt.close(fig)
    return arr


# ---------------------------------------------------------------------------
# SSVAE-style latent spectral diagnostics (arXiv:2512.05394). See
# scripts/analyze_latent_spectral_ssvae.py for the standalone / documented
# version; these are the lightweight single-number forms logged at val time.
# ---------------------------------------------------------------------------


def _dct_matrix(n: int, device, dtype) -> torch.Tensor:
    """Orthonormal DCT-II basis (n x n), DC at row 0."""
    idx = torch.arange(n, device=device, dtype=dtype)
    k = idx.unsqueeze(1)
    M = torch.cos((2.0 * idx + 1.0) * k * math.pi / (2.0 * n)) * math.sqrt(2.0 / n)
    M[0, :] = M[0, :] / math.sqrt(2.0)
    return M


def _dct3d_energy(z: torch.Tensor, Mt, Mh, Mw) -> torch.Tensor:
    """z: (B, C, T, H, W). Global separable 3D-DCT; returns per-coef energy
    |coef|^2 summed over (B, C): shape (T, H, W)."""
    d = torch.einsum("ti,bcihw->bcthw", Mt, z)
    d = torch.einsum("hj,bctjw->bcthw", Mh, d)
    d = torch.einsum("wk,bcthk->bcthw", Mw, d)
    return d.pow(2).sum(dim=(0, 1))


def _spectral_centroid_lf(energy_grid: torch.Tensor):
    """Energy-weighted centroid on the normalized L1-frequency axis + the
    low-frequency (first-quartile) energy fraction. Returns
    (centroid, bias_index, lf_frac) as python floats. White spectrum ->
    centroid ~ 0.5, bias_index ~ 0; low-freq biased -> centroid < 0.5,
    bias_index > 0 (more diffusable)."""
    T, H, W = energy_grid.shape
    dev = energy_grid.device
    ii, jj, kk = torch.meshgrid(
        torch.arange(T, device=dev), torch.arange(H, device=dev),
        torch.arange(W, device=dev), indexing="ij",
    )
    freq = (ii + jj + kk).double()
    fmax = freq.max().clamp_min(1.0)
    e = energy_grid.double()
    tot = e.sum().clamp_min(1e-12)
    centroid = float((freq / fmax * e).sum() / tot)
    lf_frac = float(e[freq <= 0.25 * fmax].sum() / tot)
    return centroid, 1.0 - 2.0 * centroid, lf_frac


# --- LCR local correlation (ported verbatim from the official SSVAE
# ssvae/modules/autoencoding/regularizers/calc_local_corr.py) -----------------


def _pearson_autocorr(x: torch.Tensor, renorm: bool = True) -> torch.Tensor:
    D = x.size(-1)
    if renorm:
        x = x - x.mean(dim=-1, keepdim=True)
        x = x / (torch.norm(x, dim=-1, keepdim=True) + 1e-8)
    sim = torch.matmul(x, x.transpose(1, 2))
    N = x.size(1)
    mask = torch.triu(torch.ones(N, N, device=x.device), diagonal=1)
    mean_corr = sim[:, mask == 1].mean()
    return mean_corr if renorm else mean_corr / D


def _windowed_localcorr(latents: torch.Tensor, ws=(2, 2, 2)):
    """latents: (B, D, F, H, W). Returns (LC_first, LC_video, LC_avg)."""
    B, D, F, H, W = latents.shape
    tw, hw, ww = ws
    t = latents.reshape(B, D, F, H // hw, hw, W // ww, ww).permute(
        0, 2, 3, 5, 4, 6, 1,
    ).reshape(B, F, H // hw * W // ww, hw * ww, D)
    first = t[:, :1].reshape(-1, hw * ww, D)
    lc_first = _pearson_autocorr(first)
    if F > 1 and (F - 1) % tw == 0:
        vid = t[:, 1:]
        _, Fv, SWN, SN, _ = vid.size()
        vid = vid.reshape(B, Fv // tw, tw, SWN, SN, D).permute(
            0, 1, 3, 2, 4, 5,
        ).reshape(B * (Fv // tw) * SWN, tw * SN, D)
        lc_video = _pearson_autocorr(vid)
        lc_avg = (lc_first + lc_video) / 2
    else:
        lc_video, lc_avg = None, lc_first
    return lc_first, lc_video, lc_avg


class WandBLidarCallback(callback.Callback):
    """Extends the base WandBCallback for Tokenizers."""

    def __init__(self, config: Config, trainer: Trainer):
        super().__init__(config, trainer)

    def on_train_start(self, model: Model, iteration: int = 0) -> None:
        wandb_utils.init_wandb(self.config, model=model)
        if distributed.is_rank0():
            self.start_iteration_time = time.time()

    def on_before_optimizer_step(
        self,
        model_ddp: distributed.DistributedDataParallel,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        grad_scaler: torch.amp.GradScaler,
        iteration: int = 0,
    ) -> None:  # Log the curent learning rate.
        if iteration % self.config.trainer.logging_iter == 0 and distributed.is_rank0():
            if model_ddp.module.network.training:
                wandb.log({"optim/lr": scheduler.get_last_lr()[0]}, step=iteration)
                wandb.log({"optim/grad_scale": grad_scaler.get_scale()}, step=iteration)
            if model_ddp.module.disc is not None and model_ddp.module.disc.training:
                wandb.log({"optim_disc/lr": scheduler.get_last_lr()[0]}, step=iteration)
                wandb.log({"optim_disc/grad_scale": grad_scaler.get_scale()}, step=iteration)

    def on_training_step_end(
        self,
        model: Model,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:  # Log the timing results (over a number of iterations) and the training loss.
        if (iteration % self.config.trainer.logging_iter == 0) and distributed.is_rank0():
            timer_results = self.trainer.training_timer.compute_average_results()
            network = model.network if not isinstance(model.network, torch_OptimizedModule) else model.network._orig_mod
            avg_time = (time.time() - self.start_iteration_time) / self.config.trainer.logging_iter
            wandb.log({"timer/iter": avg_time}, step=iteration)
            wandb.log({f"timer/{key}": value for key, value in timer_results.items()}, step=iteration)
            wandb.log({"train/loss": loss}, step=iteration)
            wandb.log({"iteration": iteration}, step=iteration)
            for loss_key in output_batch.get("loss", dict()):
                loss_val = output_batch["loss"][loss_key]
                wandb.log({f"train/{loss_key}": loss_val}, step=iteration)

            self.trainer.training_timer.reset()

    def on_validation_start(
        self, model: Model, dataloader_val: torch.utils.data.DataLoader, iteration: int = 0
    ) -> None:
        # Cache for collecting data/output batches.
        self._val_cache: dict[str, Any] = dict(
            data_batches=[],
            output_batches=[],
            loss=torch.tensor(0.0, device="cuda"),
            sample_size=torch.tensor(0, device="cuda"),
            # Render one media batch per sensor dataset. Single-dataset
            # validation still renders only its first batch.
            media_logged_datasets=set(),
            # Cross-dataset scalar metrics are reduced once at validation end;
            # logging rank-0 batch values repeatedly at the same W&B step is
            # both noisy and order-dependent.
            dataset_metrics={},
            # SSVAE-style latent spectral accumulators (lazily shaped on the
            # first lidar batch; reduced + logged in on_validation_end).
            spec_energy=None,
            spec_sum_xx=None,
            spec_n=torch.tensor(0.0, device="cuda"),
            lc_sum=torch.zeros(3, device="cuda"),
            lc_n=torch.tensor(0.0, device="cuda"),
        )
        
    def _accumulate_dataset_metrics(
        self,
        dataset_tag: str,
        metrics: dict[str, Any],
        batch_size: int,
    ) -> None:
        dataset_cache = self._val_cache["dataset_metrics"].setdefault(
            dataset_tag, {}
        )
        device = self._val_cache["loss"].device
        for name, value in metrics.items():
            if value is None:
                continue
            scalar = torch.as_tensor(
                value, device=device, dtype=torch.float32
            ).detach().mean()
            entry = dataset_cache.setdefault(
                name,
                {
                    "sum": torch.zeros((), device=device),
                    "count": torch.zeros((), device=device),
                },
            )
            entry["sum"] += scalar * batch_size
            entry["count"] += batch_size

    # Fixed t values used for noise-robustness eval (interpolative latent corruption).
    # Skips t=0.0 since clean reconstruction is already covered by val/depth_mae.
    _NOISE_LEVELS: tuple[float, ...] = (0.05, 0.10, 0.15, 0.20)

    @staticmethod
    def _encoder_input(network, data_batch, input_key) -> torch.Tensor:
        """Build the encoder input exactly as ``TransformerVAE.forward`` does.

        For ``mask_as_input`` nets the encoder expects a 2-channel
        ``[depth, mask]`` tensor, but that concat lives in ``forward`` -- NOT in
        ``encode`` -- so any callback that re-encodes (``encode(x)`` directly)
        must replicate it, else the patch-embed conv (in_channels=2) gets a
        1-channel input and raises. Prefers the dataloader's explicit
        ``loss_mask`` (correct under per-row norm), falls back to the same
        ``depth > min+eps`` derivation ``forward`` uses.
        """
        x = data_batch[input_key].float()
        if getattr(network, "mask_as_input", False):
            vm = data_batch.get("loss_mask", None)
            if vm is not None:
                valid = vm.to(x.dtype)
            else:
                valid = (
                    x[:, :1]
                    > (network.mask_input_min_value + network.mask_input_eps)
                ).to(x.dtype)
            x = torch.cat([x, valid], dim=1)
        return x

    @torch.no_grad()
    def _render_utonia_pca(self, network, data_batch, input_key, output_batch) -> dict:
        """PCA-RGB viz of the frozen Utonia teacher (the VF-loss alignment target).

        Logs (first val sample only, rank0):
          * ``val/utonia_pca``  -- per-pixel teacher features PCA'd to RGB at the
            FULL range resolution (reads like a semantic map).
          * ``val/utonia_pca_recon`` -- [teacher(GT) | teacher(recon)] at FULL
            resolution, SAME GT PCA basis/std (color-comparable), so divergence
            shows where the recon loses foundation-model-perceived structure.
          * ``val/utonia_align`` -- [pooled teacher | projected latent z'] at the
            LATENT grid, SAME PCA basis (z' lives in the teacher's feature space
            via the linear ``utonia_proj``), so aligned features share colors.

        No-op for non-utonia nets; best-effort (any failure -> warning).
        """
        media: dict = {}
        if not getattr(network, "utonia_align", False):
            return media
        try:
            x = data_batch[input_key].float()[:1]            # first sample only
            if x.ndim == 4:
                x = x.unsqueeze(2)
            depth = x[:, :1]                                 # (1, 1, T, H, W)
            teacher = network.utonia(depth)                  # (1, D, T, H, W)
            if teacher.ndim == 4:
                teacher = teacher.unsqueeze(2)
            feat = teacher[0]                                # (D, T, H, W)
            valid_full = feat.abs().sum(0) > 0               # (T, H, W)
            fv = feat.permute(1, 2, 3, 0)[valid_full]        # (M, D)
            if fv.shape[0] < 3:
                return media
            mean, V = _fit_pca_basis(fv)
            rgb_full, full_std = _apply_pca_rgb(feat, valid_full, mean, V)
            media["val/utonia_pca"] = wandb_rgb_media(rgb_full)

            # Teacher run on the RECONSTRUCTED output, PCA'd with the SAME GT
            # basis (mean/V/std) so colors are directly comparable. Shows whether
            # the recon preserves the foundation-model-perceived structure.
            recon = output_batch.get(PREDICTION, None)
            if recon is not None:
                recon = recon.float()[:1]
                if recon.ndim == 4:
                    recon = recon.unsqueeze(2)
                recon_depth = recon[:, :1].clamp(-1, 1)       # (1, 1, T, H, W)
                teacher_r = network.utonia(recon_depth)
                if teacher_r.ndim == 4:
                    teacher_r = teacher_r.unsqueeze(2)
                feat_r = teacher_r[0]                         # (D, T, H, W)
                valid_r = feat_r.abs().sum(0) > 0
                rgb_recon, _ = _apply_pca_rgb(feat_r, valid_r, mean, V, std3=full_std)
                sbs_full = np.concatenate([rgb_full, rgb_recon], axis=2)  # along W
                media["val/utonia_pca_recon"] = wandb_rgb_media(sbs_full)

            # Alignment view at the latent grid: pooled teacher vs projected z'.
            sample = output_batch.get("latent", None)
            if sample is not None and hasattr(network, "utonia_proj"):
                sample = sample[:1].float()                  # (1, z, Tz, Hz, Wz)
                Tz, Hz, Wz = sample.shape[2], sample.shape[3], sample.shape[4]
                tvalid = (feat.abs().sum(0, keepdim=True) > 0).to(feat.dtype)[None]
                num = F.adaptive_avg_pool3d(teacher * tvalid, (Tz, Hz, Wz))
                den = F.adaptive_avg_pool3d(tvalid, (Tz, Hz, Wz))
                f_grid = (num / den.clamp_min(1e-6))[0]      # (D, Tz, Hz, Wz)
                vgrid = den[0, 0] > 0                         # (Tz, Hz, Wz)
                # z' = utonia_proj(sample), per-frame (Conv2d). B=1 -> fold T.
                z2 = sample[0].permute(1, 0, 2, 3)           # (Tz, z, Hz, Wz)
                z_proj = network.utonia_proj(z2).permute(1, 0, 2, 3)  # (D, Tz, Hz, Wz)
                gv = f_grid.permute(1, 2, 3, 0)[vgrid]
                if gv.shape[0] >= 3:
                    gmean, gV = _fit_pca_basis(gv)
                    t_rgb, gstd = _apply_pca_rgb(f_grid, vgrid, gmean, gV)
                    z_rgb, _ = _apply_pca_rgb(z_proj, vgrid, gmean, gV, std3=gstd)
                    sbs = np.concatenate([t_rgb, z_rgb], axis=2)  # along W
                    media["val/utonia_align"] = wandb_rgb_media(_upsample_nn_rgb(sbs, 4))
        except Exception as e:  # noqa: BLE001
            log.warning(f"utonia PCA viz failed: {e}")
        return media

    def _compute_noise_robust_metrics(
        self,
        network,
        data_batch: dict[str, torch.Tensor],
        input_key: str,
        target_images_unnorm: torch.Tensor,
        valid_mask_target: torch.Tensor,
        unnorm_args: dict,
        dataset_config: dict,
        render_media: bool = True,
    ) -> tuple[dict[str, float], dict[str, object]]:
        """Per-noise-level reconstruction + auto-consistency for decoder finetune runs.

        For each t in _NOISE_LEVELS, in normalized latent space:
            z_t = (1 - t) * z_0 + t * gamma * eps
        then:
            depth_mae_t      = MAE(unnormalize(decode(z_t, [t])), GT)
            consistency_l2_t = RMS(encode(decode(z_t, [t]))_norm - z_0_norm)
        and renders a side-by-side GT-vs-recon point cloud for visual inspection.

        Returns ({}, {}) when the network has no loaded latent_stats
        (i.e. non-finetune networks), so this is a no-op there.
        """
        if not getattr(network, "has_latent_stats", False):
            return {}, {}

        # LTX networks route t into AdaLN-modulated decoder; plain VAE does not.
        has_ltx = float(getattr(network, "denoising_t_max", 0.0)) > 0.0
        # gamma matches training-time noise scale (sigma_data for LTX, latent_norm_noise_sigma else).
        gamma = (
            float(getattr(network, "sigma_data", 1.0))
            if has_ltx
            else float(getattr(network, "latent_norm_noise_sigma", 1.0))
        )

        metrics: dict[str, float] = {}
        media: dict[str, object] = {}
        x = self._encoder_input(network, data_batch, input_key)
        T_input = x.shape[2]

        with torch.no_grad():
            z0, _ = network.encode(x)  # sets network._input_T = T_input
            latent_mean = network.latent_mean.to(z0)
            latent_std = network.latent_std.to(z0)
            z0_norm = (z0 - latent_mean) / latent_std

            try:
                for t_val in self._NOISE_LEVELS:
                    B = z0_norm.shape[0]
                    t_view = torch.full(
                        (B,) + (1,) * (z0_norm.dim() - 1),
                        t_val,
                        device=z0.device,
                        dtype=z0.dtype,
                    )
                    eps = torch.randn_like(z0_norm) * gamma
                    zt_norm = (1.0 - t_view) * z0_norm + t_view * eps
                    zt = zt_norm * latent_std + latent_mean

                    # Restore _input_T (the consistency re-encode below mutates it).
                    network._input_T = T_input
                    if has_ltx:
                        t_scalar = torch.full(
                            (B,), t_val, device=z0.device, dtype=z0.dtype
                        )
                        xt_hat_raw = network.decode(zt, t=t_scalar)
                    else:
                        xt_hat_raw = network.decode(zt)

                    # Auto-consistency: re-encode the raw decode output (encoder
                    # expects pre-clamp / pre-undo_repeat (B,C,T,H,W) format).
                    zt_recon, _ = network.encode(xt_hat_raw)
                    zt_recon_norm = (zt_recon - latent_mean) / latent_std
                    consistency = (
                        (zt_recon_norm - z0_norm).pow(2).mean().sqrt().item()
                    )
                    metrics[f"val/consistency_l2_t{t_val:.2f}"] = consistency

                    # Depth metric: mirror the existing unnormalize pipeline.
                    xt_hat = torch.clamp(xt_hat_raw.float(), -1, 1)
                    xt_hat = undo_row_col_temporal_repeat(
                        xt_hat,
                        dataset_config["repeat_row"],
                        dataset_config["repeat_col"],
                        dataset_config["repeat_temporal"],
                    )
                    xt_unnorm_result = unnormalize_and_reduce_channels(
                        xt_hat, **unnorm_args
                    )
                    if isinstance(xt_unnorm_result, tuple):
                        xt_unnorm = xt_unnorm_result[0]
                        valid_mask_t = xt_unnorm_result[1]
                    else:
                        xt_unnorm = xt_unnorm_result
                        valid_mask_t = valid_mask_target
                    valid = valid_mask_target & valid_mask_t
                    metrics[f"val/depth_mae_t{t_val:.2f}"] = (
                        torch.abs(target_images_unnorm[valid] - xt_unnorm[valid])
                        .mean()
                        .item()
                    )

                    # Side-by-side point cloud render for visual inspection
                    # (mirrors the clean recon path in on_validation_step_end).
                    if render_media and target_images_unnorm.ndim == 3:
                        media[f"val/range_map_t{t_val:.2f}"] = wandb_range_image(
                            target_images_unnorm, xt_unnorm
                        )
                    elif render_media and target_images_unnorm.ndim == 4:
                        media[f"val/range_map_t{t_val:.2f}"] = wandb_range_video(
                            target_images_unnorm, xt_unnorm
                        )
            finally:
                network._input_T = T_input

        return metrics, media

    @torch.no_grad()
    def _accumulate_spectral_stats(self, network, data_batch, input_key) -> None:
        """Encode the batch and accumulate SSVAE-style latent spectral
        statistics into ``self._val_cache``. Best-effort: any failure is
        swallowed so it can never break validation."""
        # A single spectral tensor cannot aggregate native sensor grids with
        # different latent H/W. Avoid silently reporting only the first sensor;
        # cross-dataset runs use the per-sensor reconstruction diagnostics.
        if getattr(network, "uses_sensor_metadata", False):
            return
        try:
            x = self._encoder_input(network, data_batch, input_key)
            prev_T = getattr(network, "_input_T", None)
            z, _ = network.encode(x)  # (B, C, T, H, W); mutates _input_T
            if prev_T is not None:
                network._input_T = prev_T

            # Per-frame tokenizers (e.g. copilot4d) fold time into batch and
            # return a 4D latent (B, C, H, W). Treat that as a single latent
            # frame (T=1) so the analysis below logs instead of silently
            # skipping on the 5D unpack.
            if z.ndim == 4:
                z = z.unsqueeze(2)

            # Per-channel standardize (SSVAE preprocessing): use loaded stats
            # when available, else this batch's per-channel moments.
            if getattr(network, "has_latent_stats", False):
                zc = (z - network.latent_mean.to(z)) / network.latent_std.to(z)
            else:
                m = z.mean(dim=(0, 2, 3, 4), keepdim=True)
                s = z.std(dim=(0, 2, 3, 4), keepdim=True).clamp_min(1e-6)
                zc = (z - m) / s
            zc = zc.float()
            B, C, T, H, W = zc.shape
            dev = zc.device

            # 3D-DCT energy (cache the per-axis basis matrices by latent shape).
            if getattr(self, "_dct_cache_key", None) != (T, H, W):
                self._dct_mt = _dct_matrix(T, dev, torch.float32)
                self._dct_mh = _dct_matrix(H, dev, torch.float32)
                self._dct_mw = _dct_matrix(W, dev, torch.float32)
                self._dct_cache_key = (T, H, W)
            e = _dct3d_energy(zc, self._dct_mt, self._dct_mh, self._dct_mw)

            c = self._val_cache
            if c["spec_energy"] is None:
                c["spec_energy"] = torch.zeros_like(e, dtype=torch.float64)
                c["spec_sum_xx"] = torch.zeros(C, C, dtype=torch.float64, device=dev)
            c["spec_energy"] += e.double()
            v = zc.permute(0, 2, 3, 4, 1).reshape(-1, C).double()
            c["spec_sum_xx"] += v.t() @ v
            c["spec_n"] += v.shape[0]

            # Local correlation (size-2 windows; crop to divisible dims).
            Tc = 1 + 2 * ((T - 1) // 2) if T > 1 else 1
            Hc, Wc = 2 * (H // 2), 2 * (W // 2)
            if Hc >= 2 and Wc >= 2:
                lf, lv, la = _windowed_localcorr(zc[:, :, :Tc, :Hc, :Wc])
                c["lc_sum"][0] += lf.double()
                c["lc_sum"][1] += lv.double() if lv is not None else 0.0
                c["lc_sum"][2] += la.double()
                c["lc_n"] += 1.0
        except Exception as exc:  # noqa: BLE001
            log.warning(f"spectral-stats accumulation skipped: {exc}")

    def _get_viz_ray_dirs(self, dataset_config):
        """Cached (H, W, 3) ray-direction override for the point-cloud viz.

        ``None`` for the default Pandar128 geometry; the sensor's measured
        elevation/azimuth tables for the NCore backends (the viz renderer
        otherwise asserts on the 128-row Pandar128 table vs Waymo's 64 / nuScenes'
        32 rows)."""
        if getattr(self, "_viz_ray_dirs_cached", False):
            return self._viz_ray_dirs
        self._viz_ray_dirs = None
        _backend = dataset_config.get("dataset_backend", "radym")
        if _backend == "waymo_ncore":
            from tokenizer.training.datasets.lidar_datasets.waymo_ncore import (
                load_waymo_ray_directions,
            )
            self._viz_ray_dirs = load_waymo_ray_directions(
                root_path=dataset_config["root_path"],
                filter_list_path=dataset_config.get("filter_list_path", None),
                lidar_id=dataset_config.get("lidar_id", "lidar_top"),
                resample_width=dataset_config.get("resample_width", -1),
            )
        elif _backend == "nuscenes_ncore":
            from tokenizer.training.datasets.lidar_datasets.nuscenes_ncore import (
                load_nuscenes_ray_directions,
            )
            self._viz_ray_dirs = load_nuscenes_ray_directions(
                root_path=dataset_config["root_path"],
                filter_list_path=dataset_config.get("filter_list_path", None),
                lidar_id=dataset_config.get("lidar_id", "lidar_top"),
                resample_width=dataset_config.get("resample_width", -1),
                center_vehicle_forward=dataset_config.get(
                    "center_vehicle_forward", False
                ),
            )
        self._viz_ray_dirs_cached = True
        return self._viz_ray_dirs

    def on_validation_step_end(
        self,
        model: Model,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:  # Collect the validation batch and aggregate the overall loss.
        # Get data config
        sensor_names = data_batch.get("sensor_dataset_name")
        if sensor_names is not None:
            if len(set(sensor_names)) != 1:
                raise ValueError(
                    "Cross-dataset validation batches must be sensor-homogeneous"
                )
            dataset_name = sensor_names[0]
            dataset_config = OmegaConf.create(COMMONDATA_CONFIG[dataset_name])
            extra_config = OmegaConf.create(
                self.config.dataloader_val.dataset.shared_overrides
            )
            per_dataset_overrides = self.config.dataloader_val.dataset.get(
                "per_dataset_overrides", {}
            )
            extra_config = OmegaConf.merge(
                extra_config, per_dataset_overrides.get(dataset_name, {})
            )
        else:
            dataset_name = self.config.dataloader_val.dataset.dataset_name
            dataset_config = OmegaConf.create(COMMONDATA_CONFIG[dataset_name])
            extra_config = OmegaConf.create(self.config.dataloader_val.dataset)
        dataset_config = OmegaConf.merge(dataset_config, extra_config)
        dataset_config = OmegaConf.to_container(dataset_config, resolve=True)
        if sensor_names is None:
            dataset_tag = "default"
            val_prefix = "val"
        else:
            dataset_tag = (
                "pandaset"
                if "pandaset" in dataset_name
                else "nuscenes"
                if "nuscenes" in dataset_name
                else "waymo"
                if "waymo" in dataset_name
                else "pandar"
            )
            val_prefix = f"val/{dataset_tag}"
        # Geometry override for the point-cloud viz (None = Pandar128 default).
        sensor_rays = data_batch.get("sensor_ray_directions")
        viz_ray_dirs = (
            sensor_rays[0].detach().cpu()
            if sensor_rays is not None
            else self._get_viz_ray_dirs(dataset_config)
        )
        
        # Collect the validation batch and aggregate the overall loss.
        _input_key = model.get_input_key(data_batch)
        batch_size = data_batch[_input_key].shape[0]
        self._val_cache["loss"] += loss * batch_size
        self._val_cache["sample_size"] += batch_size

        # Cross-dataset validation renders the first batch from each sensor.
        # Metrics / spectral stats below still run every batch.
        render_media = (
            dataset_tag not in self._val_cache["media_logged_datasets"]
        )

        # Use recon_target as reference for consistency with loss (side-by-side and lidar metrics)
        target_images = data_batch["recon_target"].float()  # shape: [N, C, H, W] or [N, C, T, H, W]
        output_images = output_batch[PREDICTION].float()

        if dataset_config.get("intensity_only", False):
            # An intensity-only VAE has no depth channel to unproject. Evaluate
            # only on the explicit depth-derived return mask and avoid reporting
            # misleading depth/point-cloud metrics from intensity channel 0.
            valid = data_batch.get("loss_mask")
            if valid is None:
                raise ValueError(
                    "intensity-only validation requires an explicit loss_mask"
                )
            valid = valid[:, :1].to(torch.bool)
            target_intensity = target_images[:, :1].clamp(-1, 1)
            output_intensity = output_images[:, :1].clamp(-1, 1)
            selected = valid.expand_as(target_intensity)
            selected_f = selected.to(target_intensity.dtype)
            count = selected_f.sum().clamp_min(1.0)
            normalized_error = output_intensity - target_intensity
            normalized_mae = (
                normalized_error.abs() * selected_f
            ).sum() / count
            normalized_rmse = torch.sqrt(
                (normalized_error.square() * selected_f).sum() / count
            )

            if dataset_config.get("intensity_transform", "legacy") == "log1p":
                scale = target_intensity.new_tensor(np.log(2.0))
                target_linear = torch.expm1(
                    (target_intensity + 1.0) * 0.5 * scale
                )
                output_linear = torch.expm1(
                    (output_intensity + 1.0) * 0.5 * scale
                )
            else:
                target_linear = (target_intensity + 1.0) * 0.5
                output_linear = (output_intensity + 1.0) * 0.5
            linear_error = output_linear - target_linear
            linear_mae = (linear_error.abs() * selected_f).sum() / count
            linear_rmse = torch.sqrt(
                (linear_error.square() * selected_f).sum() / count
            )

            scalar_metrics: dict[str, Any] = {
                "loss": loss,
                "intensity_normalized_mae": normalized_mae,
                "intensity_normalized_rmse": normalized_rmse,
                "intensity_mae": linear_mae,
                "intensity_rmse": linear_rmse,
            }
            scalar_metrics.update(output_batch.get("loss", {}))
            if sensor_names is not None:
                self._accumulate_dataset_metrics(
                    dataset_tag, scalar_metrics, batch_size
                )
            elif wandb.run is not None:
                wandb.log(
                    {
                        f"{val_prefix}/{key}": value
                        for key, value in scalar_metrics.items()
                    },
                    step=iteration,
                )

            if render_media and wandb.run is not None:
                target_viz = torch.where(
                    valid, target_linear, torch.zeros_like(target_linear)
                )
                output_viz = torch.where(
                    valid, output_linear, torch.zeros_like(output_linear)
                )
                if target_viz.ndim == 5:
                    media = wandb_intensity_video(
                        target_viz[:1, 0], output_viz[:1, 0]
                    )
                else:
                    media = wandb_intensity_image(
                        target_viz[:1, 0], output_viz[:1, 0]
                    )
                wandb.log(
                    {f"{val_prefix}/intensity_reconstruction": [media]},
                    step=iteration,
                )
                self._val_cache["media_logged_datasets"].add(dataset_tag)
            return

        # Invert per-row z-score back to [-1, 1] space so all downstream viz /
        # metrics / clamps below operate in the dataset's native normalized
        # space (no-op when per-row norm is disabled).
        _per_row_normalizer = get_per_row_normalizer(
            dataset_config.get("per_row_normalize", False),
            dataset_config.get("per_row_stats_path", ""),
            min_range=dataset_config["min_range"],
            max_range=dataset_config["max_range"],
            min_value=dataset_config["min_value"],
            smooth_sigma=dataset_config.get("per_row_smooth_sigma", 5.0),
            log_depth=False,
        )
        target_images = invert_per_row_norm(target_images, _per_row_normalizer)
        output_images = invert_per_row_norm(output_images, _per_row_normalizer)

        result_image = torch.cat((target_images, output_images), dim=-1)
        n_channels = result_image.shape[1]
        
        n_channels = result_image.shape[1]
        wandb_media = None
        
        # Raw channel visualization only works for 1 or 3 channels (mono/repeat).
        # Dual-channel (range+intensity) is visualized properly below via colormapped images.
        if render_media and dataset_config["process_output"] != "band" and n_channels in (1, 3):

            if result_image.ndim == 5 and result_image.shape[2] > 1:
                wandb_media = wandb_video(result_image, iteration)
            elif result_image.ndim == 5 and result_image.shape[2] == 1:
                wandb_media = wandb_image(result_image[:, :, 0, ...], iteration)
            else:
                wandb_media = wandb_image(result_image, iteration)

        # Masked reconstruction range-image: same [GT | recon] layout as
        # val/lidar_reconstruction, but the recon panel has its predicted
        # no-return pixels (sigmoid(validity_logit) < 0.5) set to the no-return
        # clamp (min_value) -- i.e. what the model would actually emit at
        # inference. Only for validity-head nets (predict_validity=True).
        # Built from the pre-de-repeat tensors so it aligns with wandb_media.
        wandb_mask_recon_media = None
        _validity_logits = output_batch.get("validity_logits", None)
        if (
            render_media
            and _validity_logits is not None
            and dataset_config["process_output"] != "band"
            and n_channels in (1, 3)
        ):
            _pred_keep = torch.sigmoid(_validity_logits.float())[:, :1] >= 0.5
            _masked_out = torch.where(
                _pred_keep, output_images, torch.full_like(output_images, dataset_config["min_value"])
            )
            _result_masked = torch.cat((target_images, _masked_out), dim=-1)
            if _result_masked.ndim == 5 and _result_masked.shape[2] > 1:
                wandb_mask_recon_media = wandb_video(_result_masked, iteration)
            elif _result_masked.ndim == 5 and _result_masked.shape[2] == 1:
                wandb_mask_recon_media = wandb_image(_result_masked[:, :, 0, ...], iteration)
            else:
                wandb_mask_recon_media = wandb_image(_result_masked, iteration)

        # get the range_map
        is_lidar = data_batch["data_type"][0] == "lidar"
        has_intensity = dataset_config.get("load_intensity", False)
        # Predicted-validity-mask media (only populated when the network has a
        # validity head, i.e. predict_validity=True). Init here so the
        # logging block never hits an undefined name.
        wandb_mask_media = None
        wandb_pcd_masked_media = None
        wandb_range_map_media = None
        wandb_multilidar_media = None
        wandb_multilidar_pcd_media = None
        wandb_multilidar_range_media = None
        utonia_media: dict = {}
        assert dataset_config["min_value"] == -1
        if is_lidar:
            # target_images is recon_target (set above); unnormalize for range-map metrics
            intensity_transform = dataset_config.get(
                "intensity_transform", "legacy"
            )
            target_images = torch.clamp(target_images, -1, 1)
            output_images = torch.clamp(output_images, -1, 1)

            # reduce row and col repeat
            target_images = undo_row_col_temporal_repeat(
                target_images,
                dataset_config["repeat_row"],
                dataset_config["repeat_col"],
                dataset_config["repeat_temporal"],
            )
            output_images = undo_row_col_temporal_repeat(
                output_images,
                dataset_config["repeat_row"],
                dataset_config["repeat_col"],
                dataset_config["repeat_temporal"],
            )
            # Cross-dataset batches carry true rays on the repeated model
            # grid. Validation media is rendered after undoing those repeats,
            # so reduce the ray grid in lockstep with the range map.
            if viz_ray_dirs is not None:
                map_hw = tuple(target_images.shape[-2:])
                ray_hw = tuple(viz_ray_dirs.shape[:2])
                repeat_row = int(dataset_config["repeat_row"])
                repeat_col = int(dataset_config["repeat_col"])
                repeated_hw = (
                    map_hw[0] * repeat_row,
                    map_hw[1] * repeat_col,
                )
                if ray_hw == repeated_hw:
                    viz_ray_dirs = viz_ray_dirs[
                        ::repeat_row, ::repeat_col
                    ]
            log_intensity_target = (
                target_images[:, -1]
                if has_intensity and intensity_transform == "log1p"
                else None
            )
            log_intensity_recon = (
                output_images[:, -1]
                if has_intensity and intensity_transform == "log1p"
                else None
            )

            process_output = dataset_config["process_output"]
            unnorm_args = dict(
                max_range=dataset_config["max_range"],
                min_range=dataset_config["min_range"],
                min_value=dataset_config["min_value"],
                inverse_depth=dataset_config["inverse_depth"],
                process_output=process_output,
                near_buffer=0.5,
                far_buffer=0.5,
                inv_depth_threshold=dataset_config["inv_depth_threshold"],
                num_bands=dataset_config.get("num_bands", 3),
                has_intensity=has_intensity,
                intensity_transform=intensity_transform,
            )

            target_unnorm_result = unnormalize_and_reduce_channels(target_images, **unnorm_args)
            output_unnorm_result = unnormalize_and_reduce_channels(output_images, **unnorm_args)

            if has_intensity:
                target_images_unnorm, valid_mask_target, intensity_target = target_unnorm_result
                output_images_unnorm, valid_mask_recon, intensity_recon = output_unnorm_result
            else:
                target_images_unnorm, valid_mask_target = target_unnorm_result
                output_images_unnorm, valid_mask_recon = output_unnorm_result

            valid_mask = valid_mask_target & valid_mask_recon

            # --- depth point cloud visualization (original: fixed colors) ---
            if render_media:
                if target_images_unnorm.ndim == 3:
                    wandb_range_map_media = wandb_range_image(target_images_unnorm, output_images_unnorm, ray_directions=viz_ray_dirs)
                elif target_images_unnorm.ndim == 4:
                    wandb_range_map_media = wandb_range_video(target_images_unnorm, output_images_unnorm, ray_directions=viz_ray_dirs)

            # --- multi-lidar (blind-spot) reconstruction viz ---
            # Overlay the extra lidars (each a distinct color) with the top lidar
            # (gray) in ONE scene, GT | recon. Perspective PCD + BEV/side panels
            # + per-laser range/mask. VIDEO: one frame per temporal step (sample
            # 0), packed into an mp4; single-frame -> one image.
            _extra_recon = output_batch.get("extra_recon_points", None)
            if render_media and _extra_recon is not None and viz_ray_dirs is not None:
                _rd = np.asarray(viz_ray_dirs, dtype=np.float32)  # (H, W, 3)
                _is_video_ml = target_images_unnorm.ndim == 4
                T_ml = target_images_unnorm.shape[1] if _is_video_ml else 1

                # Top-lidar range (sample 0, all frames) -> np.
                _tgt = target_images_unnorm[0]  # (T,H,W) video / (H,W) image
                _rec = output_images_unnorm[0]
                if torch.is_tensor(_tgt):
                    _tgt = _tgt.detach().cpu().numpy()
                if torch.is_tensor(_rec):
                    _rec = _rec.detach().cpu().numpy()
                # Predicted validity (sample 0, all frames) to mask the top recon
                # (matches val/lidar_reconstruction_mask). Falls back to range>0.
                _pp = None
                _vlog = output_batch.get("validity_logits", None)
                if _vlog is not None:
                    _v = undo_row_col_temporal_repeat(
                        _vlog.float(),
                        dataset_config["repeat_row"],
                        dataset_config["repeat_col"],
                        dataset_config["repeat_temporal"],
                    )
                    _v = torch.sigmoid(_v)[:, 0]  # (N, [T,] H, W)
                    _pp = _v[0].detach().cpu().numpy()  # (T,H,W) / (H,W)

                _laser_t = data_batch.get("extra_laser", None)
                _op_t = output_batch.get("extra_recon_opacity", None)
                _row_t = data_batch.get("extra_row", None)
                _col_t = data_batch.get("extra_col", None)
                _hw_t = data_batch.get("extra_laser_hw", None)
                _origin_t = data_batch.get("extra_ray_origin", None)
                _have_range = _row_t is not None and _hw_t is not None and _origin_t is not None
                _laser_hw = _hw_t[0].cpu().numpy() if _have_range else None  # (L,2) clip-const

                def _top_pts(t):
                    tgt = _tgt[t] if _is_video_ml else _tgt
                    rec = _rec[t] if _is_video_ml else _rec
                    tm = tgt > 0
                    rmask = rec > 0
                    if _pp is not None:
                        pp = _pp[t] if _is_video_ml else _pp
                        rmask = rmask & (pp >= 0.5)
                    return (
                        (_rd[tm] * tgt[tm][:, None]).astype(np.float32),
                        (_rd[rmask] * rec[rmask][:, None]).astype(np.float32),
                    )

                def _extra_at(t):
                    # output_batch extras are (B*T, K, ...); sample 0 occupies
                    # rows 0..T-1, so frame t == row t. data_batch extras are
                    # (B, T, K, ...) -> [0, t].
                    ex_gt = output_batch["extra_gt_points"][t].float().cpu().numpy()
                    ex_rec = _extra_recon[t].float().cpu().numpy()
                    ex_val = output_batch["extra_valid_points"][t].float().cpu().numpy()
                    ex_laser = (
                        _laser_t[0, t].reshape(-1).cpu().numpy()
                        if _laser_t is not None
                        else np.zeros(ex_val.shape[0], dtype=np.int64)
                    )
                    ex_keep = (_op_t[t].float().cpu().numpy() >= 0.5) if _op_t is not None else None
                    return ex_gt, ex_rec, ex_val, ex_laser, ex_keep

                per_frame = [(_top_pts(t), _extra_at(t)) for t in range(T_ml)]

                # Fixed BEV zoom across frames (stable video): union of GT extras.
                _all_xy = [
                    ex_gt[ex_val.astype(bool)][:, :2]
                    for (_, (ex_gt, _, ex_val, _, _)) in per_frame
                    if ex_val.astype(bool).any()
                ]
                if _all_xy:
                    _R_ml = float(np.clip(np.percentile(np.abs(np.concatenate(_all_xy, 0)), 99) * 1.3, 15.0, 80.0))
                else:
                    _R_ml = 40.0

                pcd_frames, bev_frames, range_frames = [], [], []
                for t, ((top_gt_pts, top_rec_pts), (ex_gt, ex_rec, ex_val, ex_laser, ex_keep)) in enumerate(per_frame):
                    pcd_frames.append(_multilidar_pcd_frame(
                        top_gt_pts, top_rec_pts, ex_gt, ex_rec, ex_val, ex_laser, ex_keep=ex_keep,
                    ))
                    bev_frames.append(_multilidar_bev_frame(
                        top_gt_pts, top_rec_pts, ex_gt, ex_rec, ex_val, ex_laser, ex_keep=ex_keep, R=_R_ml,
                    ))
                    if _have_range:
                        ex_row = _row_t[0, t].reshape(-1).cpu().numpy()
                        ex_col = _col_t[0, t].reshape(-1).cpu().numpy()
                        origin_t = _origin_t[0, t].reshape(-1, 3).float().cpu().numpy()  # (K,3)
                        gt_depth = data_batch["extra_gt_range"][0, t].reshape(-1).float().cpu().numpy()
                        recon_depth = np.linalg.norm(ex_rec - origin_t, axis=1)
                        pred_valid = ex_keep if ex_keep is not None else ex_val.astype(bool)
                        range_frames.append(_multilidar_range_frame(
                            gt_depth, recon_depth, pred_valid,
                            ex_laser, ex_row, ex_col, _laser_hw,
                            max_range=float(dataset_config["max_range"]),
                        ))

                wandb_multilidar_pcd_media = _frames_to_media(pcd_frames)
                wandb_multilidar_media = _frames_to_media(bev_frames)
                if range_frames:
                    wandb_multilidar_range_media = _frames_to_media(range_frames)

            # --- depth metrics ---
            # For validity-head nets, evaluate depth ONLY over GT valid returns
            # (valid_mask_target) -- the region the depth loss is trained on
            # (mask_invalid_depth). The recon's depth at no-return pixels is
            # unsupervised garbage; intersecting with recon-in-range
            # (valid_mask) would optimistically drop the recon's worst
            # out-of-range pixels. Non-validity nets keep the prior region.
            has_validity_head = output_batch.get("validity_logits", None) is not None
            depth_metric_mask = valid_mask_target if has_validity_head else valid_mask
            range_map_target = target_images_unnorm[depth_metric_mask]
            range_map_recon = output_images_unnorm[depth_metric_mask]

            mae = torch.abs(range_map_target - range_map_recon).mean().item()
            rmse = torch.sqrt(torch.mean((range_map_target - range_map_recon) ** 2)).item()
            relative_error = (torch.abs(range_map_target - range_map_recon) / (range_map_target + 1e-6)).mean().item()

            # --- noise-robustness metrics (only for decoder finetune runs) ---
            network = (
                model.network
                if not isinstance(model.network, torch_OptimizedModule)
                else model.network._orig_mod
            )
            noise_metrics, noise_media = self._compute_noise_robust_metrics(
                network=network,
                data_batch=data_batch,
                input_key=_input_key,
                target_images_unnorm=target_images_unnorm,
                valid_mask_target=valid_mask_target,
                unnorm_args=unnorm_args,
                dataset_config=dataset_config,
                render_media=render_media,
            )

            # SSVAE-style latent spectral diagnostics (accumulated across
            # val batches, reduced + logged in on_validation_end).
            self._accumulate_spectral_stats(network, data_batch, _input_key)

            # Utonia (PTv3) teacher PCA-RGB viz (VF-loss alignment target).
            # rank0 + first batch only (one extra frozen PTv3 forward).
            if render_media and distributed.is_rank0():
                utonia_media = self._render_utonia_pca(
                    network, data_batch, _input_key, output_batch
                )

            # --- predicted validity (raydrop) mask + mask-filtered points ---
            # Present only when the network has a validity head
            # (predict_validity=True). `validity_logits` are at full output
            # resolution, so apply the SAME row/col/temporal de-repeat used
            # for the range maps before comparing / filtering.
            validity_logits = output_batch.get("validity_logits", None)
            pred_prob = None  # set below when a validity head is present
            if render_media and validity_logits is not None:
                vlog = undo_row_col_temporal_repeat(
                    validity_logits.float(),
                    dataset_config["repeat_row"],
                    dataset_config["repeat_col"],
                    dataset_config["repeat_temporal"],
                )
                pred_prob = torch.sigmoid(vlog)[:, 0]  # (N, [T,] H, W)
                thr = dataset_config["min_value"] + 0.02
                # GT validity for the viz. Prefer the dataloader's explicit
                # pre-per-row mask (per-row z-scoring breaks the depth>thr
                # derivation); fall back to thresholding the (min-max) depth.
                _explicit_valid = data_batch.get("loss_mask", None)
                if _explicit_valid is not None:
                    _ev = undo_row_col_temporal_repeat(
                        _explicit_valid.float(),
                        dataset_config["repeat_row"],
                        dataset_config["repeat_col"],
                        dataset_config["repeat_temporal"],
                    )
                    gt_valid = _ev[:, 0]  # (N, [T,] H, W)
                else:
                    gt_valid = (target_images[:, 0] > thr).float()  # (N, [T,] H, W)
                # Apply the predicted mask: drop points the decoder marks as
                # no-return (prob < 0.5) before rendering the recon points.
                output_masked = torch.where(
                    pred_prob >= 0.5,
                    output_images_unnorm,
                    torch.zeros_like(output_images_unnorm),
                )
                if output_images_unnorm.ndim == 3:
                    wandb_mask_media = wandb_mask_image(gt_valid, pred_prob)
                    wandb_pcd_masked_media = wandb_range_image(
                        target_images_unnorm, output_masked, ray_directions=viz_ray_dirs
                    )
                elif output_images_unnorm.ndim == 4:
                    wandb_mask_media = wandb_mask_video(gt_valid, pred_prob)
                    wandb_pcd_masked_media = wandb_range_video(
                        target_images_unnorm, output_masked, ray_directions=viz_ray_dirs
                    )

            # --- intensity visualizations and metrics ---
            wandb_intensity_media = None
            wandb_intensity_mask_media = None
            wandb_pcd_intensity_media = None
            wandb_pcd_intensity_mask_media = None
            intensity_mae = None
            intensity_rmse = None
            intensity_log_mae = None
            intensity_high_log_mae = None
            if has_intensity:
                int_gt = intensity_target
                int_re = intensity_recon
                int_gt_viz = int_gt
                int_re_viz = int_re

                # intensity range image (colormapped side-by-side)
                if render_media:
                    if int_gt.ndim == 3:
                        wandb_intensity_media = wandb_intensity_image(
                            int_gt_viz, int_re_viz
                        )
                    elif int_gt.ndim == 4:
                        wandb_intensity_media = wandb_intensity_video(
                            int_gt_viz, int_re_viz
                        )

                    # Masked intensity recon: same [GT | recon] colormapped layout,
                    # but the recon panel's predicted no-return pixels
                    # (sigmoid(validity_logit) < 0.5) are zeroed -- the intensity
                    # counterpart of `val/lidar_reconstruction_mask` (depth). Shows
                    # whether the predicted raydrop mask lines up with intensity.
                    if pred_prob is not None:
                        int_re_masked = torch.where(
                            pred_prob >= 0.5,
                            int_re_viz,
                            torch.zeros_like(int_re_viz),
                        )
                        if int_gt.ndim == 3:
                            wandb_intensity_mask_media = wandb_intensity_image(
                                int_gt_viz, int_re_masked
                            )
                        elif int_gt.ndim == 4:
                            wandb_intensity_mask_media = wandb_intensity_video(
                                int_gt_viz, int_re_masked
                            )

                    # point cloud colored by intensity
                    tgt_np = target_images_unnorm.cpu().numpy()
                    out_np = output_images_unnorm.cpu().numpy()
                    int_gt_np = int_gt_viz.cpu().numpy()
                    int_re_np = int_re_viz.cpu().numpy()
                    if tgt_np.ndim == 3:
                        wandb_pcd_intensity_media = wandb_range_image_intensity(
                            tgt_np, out_np, int_gt_np, int_re_np, ray_directions=viz_ray_dirs
                        )
                    elif tgt_np.ndim == 4:
                        wandb_pcd_intensity_media = wandb_range_video_intensity(
                            tgt_np, out_np, int_gt_np, int_re_np, ray_directions=viz_ray_dirs
                        )

                    # masked point cloud colored by intensity: recon points the
                    # decoder marks as no-return (prob < 0.5) are dropped via the
                    # zeroed `output_masked` depth (range 0 < min_range filters
                    # them out) and the matching `int_re_masked` coloring. The
                    # intensity counterpart of `val/range_map_masked`.
                    if pred_prob is not None:
                        out_masked_np = output_masked.cpu().numpy()
                        int_re_masked_np = int_re_masked.cpu().numpy()
                        if tgt_np.ndim == 3:
                            wandb_pcd_intensity_mask_media = wandb_range_image_intensity(
                                tgt_np, out_masked_np, int_gt_np, int_re_masked_np, ray_directions=viz_ray_dirs
                            )
                        elif tgt_np.ndim == 4:
                            wandb_pcd_intensity_mask_media = wandb_range_video_intensity(
                                tgt_np, out_masked_np, int_gt_np, int_re_masked_np, ray_directions=viz_ray_dirs
                            )

                # intensity metrics (on valid depth pixels; same region as the
                # depth metrics -- GT valid returns for validity-head nets).
                int_gt_valid = int_gt[depth_metric_mask]
                int_re_valid = int_re[depth_metric_mask]
                intensity_mae = torch.abs(int_gt_valid - int_re_valid).mean().item()
                intensity_rmse = torch.sqrt(torch.mean((int_gt_valid - int_re_valid) ** 2)).item()
                if intensity_transform == "log1p":
                    log_gt = log_intensity_target[depth_metric_mask]
                    log_re = log_intensity_recon[depth_metric_mask]
                    intensity_log_mae = torch.abs(log_gt - log_re).mean().item()
                    high_thr = torch.quantile(log_gt.float(), 0.95)
                    high = log_gt >= high_thr
                    intensity_high_log_mae = torch.abs(
                        log_gt[high] - log_re[high]
                    ).mean().item()

        else:
            mae = None
            wandb_range_map_media = None
            has_intensity = False
            noise_metrics = {}
            noise_media: dict[str, object] = {}

        scalar_metrics: dict[str, Any] = {"loss": loss}
        if mae is not None:
            scalar_metrics.update(
                depth_mae=mae,
                depth_rmse=rmse,
                depth_relative_error=relative_error,
            )
            scalar_metrics.update(
                {
                    key.removeprefix("val/"): value
                    for key, value in noise_metrics.items()
                }
            )
        if has_intensity and intensity_mae is not None:
            scalar_metrics.update(
                intensity_mae=intensity_mae,
                intensity_rmse=intensity_rmse,
            )
            if intensity_log_mae is not None:
                scalar_metrics.update(
                    intensity_log_mae=intensity_log_mae,
                    intensity_high95_log_mae=intensity_high_log_mae,
                )
        scalar_metrics.update(output_batch.get("loss", {}))
        scalar_metrics.update(
            {
                metric_key.upper(): metric_val
                for metric_key, metric_val in output_batch.get(
                    "metric", {}
                ).items()
            }
        )
        if sensor_names is not None:
            self._accumulate_dataset_metrics(
                dataset_tag, scalar_metrics, batch_size
            )

        if wandb.run is not None:
            if sensor_names is None:
                wandb.log(
                    {
                        f"{val_prefix}/{key}": value
                        for key, value in scalar_metrics.items()
                    },
                    step=iteration,
                )
            for k, m in noise_media.items():
                scoped_key = f"{val_prefix}/{k.removeprefix('val/')}"
                wandb.log({scoped_key: [m]}, step=iteration)

            if wandb_media is not None:
                if is_lidar:
                    wandb.log(
                        {f"{val_prefix}/lidar_reconstruction": [wandb_media]},
                        step=iteration,
                    )
                else:
                    wandb.log(
                        {f"{val_prefix}/rgb_reconstruction": [wandb_media]},
                        step=iteration,
                    )

            if wandb_range_map_media is not None:
                wandb.log(
                    {f"{val_prefix}/range_map": [wandb_range_map_media]},
                    step=iteration,
                )

            if wandb_multilidar_media is not None:
                wandb.log(
                    {f"{val_prefix}/multilidar_bev": [wandb_multilidar_media]},
                    step=iteration,
                )
            if wandb_multilidar_pcd_media is not None:
                wandb.log(
                    {f"{val_prefix}/multilidar_pcd": [wandb_multilidar_pcd_media]},
                    step=iteration,
                )
            if wandb_multilidar_range_media is not None:
                wandb.log(
                    {
                        f"{val_prefix}/multilidar_range": [
                            wandb_multilidar_range_media
                        ]
                    },
                    step=iteration,
                )

            if wandb_intensity_media is not None:
                wandb.log(
                    {f"{val_prefix}/intensity_map": [wandb_intensity_media]},
                    step=iteration,
                )
            if wandb_intensity_mask_media is not None:
                wandb.log(
                    {
                        f"{val_prefix}/range_map_intensity_mask": [
                            wandb_intensity_mask_media
                        ]
                    },
                    step=iteration,
                )
            if wandb_pcd_intensity_media is not None:
                wandb.log(
                    {f"{val_prefix}/pcd_intensity": [wandb_pcd_intensity_media]},
                    step=iteration,
                )
            if wandb_pcd_intensity_mask_media is not None:
                wandb.log(
                    {
                        f"{val_prefix}/pcd_intensity_masked": [
                            wandb_pcd_intensity_mask_media
                        ]
                    },
                    step=iteration,
                )

            if wandb_mask_media is not None:
                wandb.log(
                    {f"{val_prefix}/validity_mask": [wandb_mask_media]},
                    step=iteration,
                )
            if wandb_pcd_masked_media is not None:
                wandb.log(
                    {f"{val_prefix}/range_map_masked": [wandb_pcd_masked_media]},
                    step=iteration,
                )
            if wandb_mask_recon_media is not None:
                wandb.log(
                    {
                        f"{val_prefix}/lidar_reconstruction_mask": [
                            wandb_mask_recon_media
                        ]
                    },
                    step=iteration,
                )

            for _k, _m in utonia_media.items():
                scoped_key = f"{val_prefix}/{_k.removeprefix('val/')}"
                wandb.log({scoped_key: [_m]}, step=iteration)

        # Set on all ranks so every rank skips later media for this sensor.
        if render_media:
            self._val_cache["media_logged_datasets"].add(dataset_tag)

    def on_before_dataloading(self, iteration: int = 0) -> None:
        if iteration % self.config.trainer.logging_iter == 0 and distributed.is_rank0():
            self.start_iteration_time = time.time()
            
    def on_validation_end(self, model: Model, iteration: int = 0) -> None:
        # Compute the average validation loss across all devices.
        dist.all_reduce(self._val_cache["loss"], op=dist.ReduceOp.SUM)
        dist.all_reduce(self._val_cache["sample_size"], op=dist.ReduceOp.SUM)
        loss = self._val_cache["loss"].item() / self._val_cache["sample_size"]

        # Reduce SSVAE-style latent spectral accumulators across ranks. The
        # ``spec_energy is not None`` guard is symmetric across ranks (the run
        # is lidar-or-not for all of them), so the collectives stay matched.
        c = self._val_cache
        spectral = None
        if c.get("spec_energy") is not None:
            dist.all_reduce(c["spec_energy"], op=dist.ReduceOp.SUM)
            dist.all_reduce(c["spec_sum_xx"], op=dist.ReduceOp.SUM)
            dist.all_reduce(c["spec_n"], op=dist.ReduceOp.SUM)
            dist.all_reduce(c["lc_sum"], op=dist.ReduceOp.SUM)
            dist.all_reduce(c["lc_n"], op=dist.ReduceOp.SUM)
            centroid, bias_index, lf_frac = _spectral_centroid_lf(c["spec_energy"])
            autocorr = c["spec_sum_xx"] / c["spec_n"].clamp_min(1.0)
            autocorr = 0.5 * (autocorr + autocorr.t())
            ev = torch.linalg.eigvalsh(autocorr).clamp_min(0.0)
            p = ev / ev.sum().clamp_min(1e-12)
            eff_rank = float(torch.exp(-(p * (p + 1e-12).log()).sum()))
            lc = (c["lc_sum"] / c["lc_n"].clamp_min(1.0)).tolist()
            spectral = {
                "val/spectral_bias_index": bias_index,
                "val/spectral_centroid": centroid,
                "val/spectral_lf_energy_frac": lf_frac,
                "val/latent_eff_rank": eff_rank,
                "val/latent_lc_first": lc[0],
                "val/latent_lc_video": lc[1],
                "val/latent_lc_avg": lc[2],
            }

        dataset_metrics = {}
        for dataset_tag in sorted(c["dataset_metrics"]):
            for metric_name in sorted(c["dataset_metrics"][dataset_tag]):
                entry = c["dataset_metrics"][dataset_tag][metric_name]
                dist.all_reduce(entry["sum"], op=dist.ReduceOp.SUM)
                dist.all_reduce(entry["count"], op=dist.ReduceOp.SUM)
                dataset_metrics[
                    f"val/{dataset_tag}/{metric_name}"
                ] = (
                    entry["sum"] / entry["count"].clamp_min(1.0)
                ).item()

        # Log data/stats of validation set to W&B.
        if distributed.is_rank0():
            log.info(f"Validation loss (iteration {iteration}): {loss:4f}")
            wandb.log({"val/loss": loss}, step=iteration)
            if dataset_metrics:
                wandb.log(dataset_metrics, step=iteration)
            if spectral is not None:
                wandb.log(spectral, step=iteration)
                log.info(
                    f"Latent spectral (iter {iteration}): "
                    f"bias_index={spectral['val/spectral_bias_index']:.3f} "
                    f"eff_rank={spectral['val/latent_eff_rank']:.2f} "
                    f"lc_avg={spectral['val/latent_lc_avg']:.3f}"
                )

    def on_train_end(self, model: Model, iteration: int = 0) -> None:
        wandb.finish()
