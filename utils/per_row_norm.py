# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-row depth normalization helpers (vendored, no external deps).

The lidar range map is heavily heteroscedastic across elevation rows: ground
rows have std ~1 m, mid rows ~23 m. Plain min-max normalization to ``[-1, 1]``
preserves this disparity, so the model's first layers waste capacity learning
per-row scale. This module standardizes per-row residuals around precomputed
training-set statistics.

The pipeline is:

    metric depth -> ``normalize_range_map`` -> ``[-1, 1]`` space (existing)
                 -> :func:`apply_per_row_norm` -> unit-variance per row (new)

Inversion:

    unit-variance per row -> :func:`invert_per_row_norm` -> ``[-1, 1]`` space
                          -> ``unnormalize_range_map`` -> metric depth (existing)

Stats are loaded from a ``.npz`` with per-row ``means`` / ``stds`` in metric
units (e.g. ``scripts/analyze_per_row_depth.py`` or r2flow's depth-stats dump)
and converted to the dataset's normalized space via the same
``min_range``/``max_range``/``min_value`` used by ``normalize_range_map``.

Smoothing the raw per-row stats is recommended (default
``smooth_sigma = 5.0`` rows): adjacent-row stats become nearly identical, so
objects spanning multiple rows are not distorted by the normalization.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


def _gaussian_kernel(sigma: float, radius: int | None = None) -> np.ndarray:
    if radius is None:
        radius = max(1, int(round(3.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-(x ** 2) / (2.0 * sigma ** 2))
    return (k / k.sum()).astype(np.float32)


def _gaussian_smooth_1d(arr: np.ndarray, sigma: float) -> np.ndarray:
    """Reflect-pad and convolve a 1D array with a Gaussian kernel."""
    if sigma <= 0:
        return arr.astype(np.float32)
    kernel = _gaussian_kernel(sigma)
    pad = (len(kernel) - 1) // 2
    padded = np.pad(arr.astype(np.float32), pad, mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def metric_stats_to_normalized(
    means_metric: np.ndarray,
    stds_metric: np.ndarray,
    *,
    min_range: float,
    max_range: float,
    min_value: float = -1.0,
    log_depth: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert per-row (mean, std) to the dataset's ``[-1, 1]`` space.

    The min-max normalization ``(d - min_range) / (max_range - min_range) * 2 - 1``
    is affine, so:

        mean_norm = (mean - min_range) / span * scale - shift_offset
        std_norm  = std / span * scale

    where ``scale = 2`` and ``shift_offset = 1`` for ``min_value = -1`` (the
    project default), and ``scale = 1, shift_offset = 0`` for ``min_value = 0``.

    When ``log_depth=True`` the input ``means_metric`` / ``stds_metric`` are
    expected to be the mean / std of ``log(d)`` (since ``mean(log(d)) !=
    log(mean(d))``), and the affine is taken in log space:

        span      = log(max_range) - log(min_range)
        mean_norm = (log_mean - log(min_range)) / span * scale - shift
        std_norm  = log_std / span * scale
    """
    if log_depth:
        c_min = float(np.log(min_range))
        c_max = float(np.log(max_range))
    else:
        c_min = min_range
        c_max = max_range
    span = c_max - c_min
    scale = 2.0 if min_value == -1 else 1.0
    shift = 1.0 if min_value == -1 else 0.0
    means_norm = (means_metric - c_min) / span * scale - shift
    stds_norm = stds_metric / span * scale
    return means_norm.astype(np.float32), stds_norm.astype(np.float32)


@dataclass
class PerRowNormalizer:
    """Holds smoothed per-row stats and applies/inverts the normalization.

    The stats are stored in normalized space (i.e. corresponding to ``x`` after
    ``normalize_range_map``). Apply happens after the existing min-max
    normalization, invert happens before ``unnormalize_range_map``.
    """
    means_norm: np.ndarray   # shape (H_native,)
    stds_norm: np.ndarray    # shape (H_native,)
    eps: float = 1e-3        # avoid div-by-zero on rows with no valid returns

    @classmethod
    def from_npz(
        cls,
        path: str,
        *,
        min_range: float,
        max_range: float,
        min_value: float = -1.0,
        smooth_sigma: float = 5.0,
        expected_log_depth: bool = False,
    ) -> "PerRowNormalizer":
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"per-row stats file not found: {path}. Run "
                f"`python scripts/analyze_per_row_depth.py` to generate it."
            )
        data = np.load(path)
        if "means" not in data or "stds" not in data:
            raise KeyError(
                f"per-row stats file {path} is missing 'means' / 'stds' keys "
                f"(got {list(data.keys())})."
            )
        # Validate that the stats file's depth-mode matches the dataset config.
        # Older stats files (no ``log_depth`` key) are assumed to be linear-metric
        # for backward compatibility.
        stats_log_depth = bool(data["log_depth"]) if "log_depth" in data else False
        if stats_log_depth != expected_log_depth:
            raise ValueError(
                f"per-row stats file {path} has log_depth={stats_log_depth} but "
                f"the dataset config expects log_depth={expected_log_depth}. "
                f"Regenerate the stats with the matching depth mode."
            )
        means = data["means"].astype(np.float32)
        stds = data["stds"].astype(np.float32)
        # Replace NaNs (rows with no valid returns) with neutral values so the
        # transform is a no-op there. Such rows usually have no model loss
        # anyway because the validity mask zeros them out.
        nan_mask = np.isnan(means) | np.isnan(stds) | (stds <= 0)
        if nan_mask.any():
            # Use nearest non-NaN neighbour for a reasonable extrapolation,
            # falling back to global mean/std if the entire array is NaN.
            valid = ~nan_mask
            if not valid.any():
                global_mean = float(np.nanmean(means)) if not np.isnan(means).all() else 0.0
                global_std = float(np.nanmean(stds)) if not np.isnan(stds).all() else 1.0
                means = np.full_like(means, global_mean)
                stds = np.full_like(stds, global_std)
            else:
                idx_valid = np.where(valid)[0]
                for i in np.where(nan_mask)[0]:
                    j = idx_valid[np.argmin(np.abs(idx_valid - i))]
                    means[i] = means[j]
                    stds[i] = stds[j]

        # Smooth before converting to normalized space (smoothing is in metric
        # units; equivalent up to the affine transform anyway).
        means_smooth = _gaussian_smooth_1d(means, smooth_sigma)
        stds_smooth = _gaussian_smooth_1d(stds, smooth_sigma)

        means_norm, stds_norm = metric_stats_to_normalized(
            means_smooth, stds_smooth,
            min_range=min_range, max_range=max_range, min_value=min_value,
            log_depth=stats_log_depth,
        )
        return cls(means_norm=means_norm, stds_norm=stds_norm)

    # ------------------------------------------------------------------
    # Public ops
    # ------------------------------------------------------------------

    def _resample(self, vec: np.ndarray, h: int) -> np.ndarray:
        """Resample a (H_native,) stats vector to (h,) via nearest-exact, so
        per-row identity is preserved when the dataset / model resolution
        differs from the analysis resolution."""
        if vec.shape[0] == h:
            return vec
        x = torch.from_numpy(vec).view(1, 1, 1, -1).float()
        y = F.interpolate(x, size=(1, h), mode="nearest-exact")
        return y.view(-1).numpy()

    def _resample_torch(self, vec: np.ndarray, h: int, device) -> torch.Tensor:
        if vec.shape[0] == h:
            return torch.from_numpy(vec).to(device)
        x = torch.from_numpy(vec).view(1, 1, 1, -1).float()
        y = F.interpolate(x, size=(1, h), mode="nearest-exact").view(-1)
        return y.to(device)

    def apply_numpy(self, x: np.ndarray) -> np.ndarray:
        """Standardize a numpy array of shape ``(..., H, W)``."""
        H = x.shape[-2]
        m = self._resample(self.means_norm, H)
        s = self._resample(self.stds_norm, H)
        m = m.reshape((1,) * (x.ndim - 2) + (H, 1))
        s = s.reshape((1,) * (x.ndim - 2) + (H, 1))
        return (x - m) / (s + self.eps)

    def invert_numpy(self, x: np.ndarray) -> np.ndarray:
        H = x.shape[-2]
        m = self._resample(self.means_norm, H)
        s = self._resample(self.stds_norm, H)
        m = m.reshape((1,) * (x.ndim - 2) + (H, 1))
        s = s.reshape((1,) * (x.ndim - 2) + (H, 1))
        return x * (s + self.eps) + m

    def apply_torch(self, x: torch.Tensor) -> torch.Tensor:
        """Standardize a torch tensor of shape ``(..., H, W)``."""
        H = x.shape[-2]
        m = self._resample_torch(self.means_norm, H, x.device).to(x.dtype)
        s = self._resample_torch(self.stds_norm, H, x.device).to(x.dtype)
        view = (1,) * (x.ndim - 2) + (H, 1)
        return (x - m.view(*view)) / (s.view(*view) + self.eps)

    def invert_torch(self, x: torch.Tensor) -> torch.Tensor:
        H = x.shape[-2]
        m = self._resample_torch(self.means_norm, H, x.device).to(x.dtype)
        s = self._resample_torch(self.stds_norm, H, x.device).to(x.dtype)
        view = (1,) * (x.ndim - 2) + (H, 1)
        return x * (s.view(*view) + self.eps) + m.view(*view)


_normalizer_cache: dict = {}


def get_per_row_normalizer(
    per_row_normalize: bool,
    per_row_stats_path: str,
    *,
    min_range: float,
    max_range: float,
    min_value: float = -1.0,
    smooth_sigma: float = 5.0,
    log_depth: bool = False,
) -> Optional[PerRowNormalizer]:
    """Build (or reuse a cached) :class:`PerRowNormalizer`.

    Returns ``None`` when ``per_row_normalize`` is False, so callers can use the
    result as a drop-in argument to :func:`apply_per_row_norm` /
    :func:`invert_per_row_norm` even when per-row norm is disabled.
    """
    if not per_row_normalize:
        return None
    if not per_row_stats_path:
        raise ValueError(
            "per_row_normalize=True requires per_row_stats_path to point to a "
            ".npz with per-row 'means'/'stds'."
        )
    key = (
        per_row_stats_path,
        float(min_range), float(max_range), float(min_value),
        float(smooth_sigma), bool(log_depth),
    )
    if key not in _normalizer_cache:
        _normalizer_cache[key] = PerRowNormalizer.from_npz(
            per_row_stats_path,
            min_range=float(min_range),
            max_range=float(max_range),
            min_value=float(min_value),
            smooth_sigma=float(smooth_sigma),
            expected_log_depth=bool(log_depth),
        )
    return _normalizer_cache[key]


def apply_per_row_norm(
    x: np.ndarray | torch.Tensor,
    normalizer: Optional[PerRowNormalizer],
) -> np.ndarray | torch.Tensor:
    """Apply per-row normalization if ``normalizer`` is provided; else identity."""
    if normalizer is None:
        return x
    if isinstance(x, torch.Tensor):
        return normalizer.apply_torch(x)
    return normalizer.apply_numpy(x)


def invert_per_row_norm(
    x: np.ndarray | torch.Tensor,
    normalizer: Optional[PerRowNormalizer],
) -> np.ndarray | torch.Tensor:
    """Invert per-row normalization if ``normalizer`` is provided; else identity."""
    if normalizer is None:
        return x
    if isinstance(x, torch.Tensor):
        return normalizer.invert_torch(x)
    return normalizer.invert_numpy(x)
