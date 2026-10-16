# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rectified-flow shift reparam, LTX training stretch, inference terminal stretch.

LTX-Video arXiv:2501.00103 §2.5.2 / Fig. 10 (training):
  After shift, support is limited to the shifted-t ``(p_lo, p_hi)`` percentiles and
  affine-mapped to ``[0, 1]``. Use **truncated logitnormal** draws (not clip-then-map):
  clip piles ~0.5% of mass at exactly ``t=0``; truncation keeps a smooth left tail like
  the paper's blue curve.

Inference: ``stretch_shifts_to_terminal`` remaps the discrete sigma ladder (separate).
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import numpy as np
import torch

DEFAULT_SHIFT_CLAMP_PERCENTILES: Tuple[float, float] = (0.5, 99.9)

_SHIFT_T_QUANTILE_CACHE: dict[tuple, tuple[float, float]] = {}
_STRETCH_U_BOUNDS_CACHE: dict[tuple, tuple[float, float, float, float, float, float]] = {}


def shifted_t(t, shift: float):
    """``t' = shift*t / (1 + (shift-1)*t)``; monotone on ``[0, 1]``."""
    return shift * t / (1.0 + (shift - 1.0) * t)


def inv_shifted_t(t, shift: float):
    """Inverse of :func:`shifted_t` on ``[0, 1]``."""
    t = np.asarray(t, dtype=np.float64) if not isinstance(t, torch.Tensor) else t
    return t / (shift - (shift - 1.0) * t)


def stretch_shifts_to_terminal(
    shifts: np.ndarray,
    terminal: float = 0.1,
) -> np.ndarray:
    """Stretch a monotone inference schedule so ``shifts[-1] == terminal`` (LTX rf.py)."""
    shifts = np.asarray(shifts, dtype=np.float64)
    if shifts.size == 0:
        raise ValueError("shifts must not be empty")
    if not (0.0 < terminal < 1.0):
        raise ValueError(f"terminal must be in (0, 1), got {terminal}")
    one_minus_z = 1.0 - shifts
    scale_factor = one_minus_z[-1] / (1.0 - terminal)
    return 1.0 - (one_minus_z / scale_factor)


def shift_t_quantile_bounds(
    shift: float,
    p_lo: float = DEFAULT_SHIFT_CLAMP_PERCENTILES[0],
    p_hi: float = DEFAULT_SHIFT_CLAMP_PERCENTILES[1],
    n_mc: int = 200_000,
    seed: int = 0,
) -> tuple[float, float]:
    """``(p_lo, p_hi)`` percentiles of ``logitnormal(0,1) ∘ shifted_t`` (no stretch)."""
    key = (float(shift), float(p_lo), float(p_hi), int(n_mc), int(seed))
    if key not in _SHIFT_T_QUANTILE_CACHE:
        rng = np.random.default_rng(seed)
        t = sample_shifted_t_density(shift, n=n_mc, rng=rng, stretch_percentiles=None)
        _SHIFT_T_QUANTILE_CACHE[key] = (
            float(np.percentile(t, p_lo)),
            float(np.percentile(t, p_hi)),
        )
    return _SHIFT_T_QUANTILE_CACHE[key]


def shift_stretch_bounds(
    shift: float,
    percentiles: Tuple[float, float] = DEFAULT_SHIFT_CLAMP_PERCENTILES,
) -> tuple[float, float, float, float, float, float]:
    """``(lo, hi, u_lo, u_hi, z_lo, z_hi)`` for truncated logitnormal stretch."""
    key = (float(shift), float(percentiles[0]), float(percentiles[1]))
    if key not in _STRETCH_U_BOUNDS_CACHE:
        lo, hi = shift_t_quantile_bounds(shift, percentiles[0], percentiles[1])
        eps = 1e-6
        u_lo = float(np.clip(inv_shifted_t(lo, shift), eps, 1.0 - eps))
        u_hi = float(np.clip(inv_shifted_t(hi, shift), u_lo, 1.0 - eps))
        z_lo = math.log(u_lo / (1.0 - u_lo))
        z_hi = math.log(u_hi / (1.0 - u_hi))
        _STRETCH_U_BOUNDS_CACHE[key] = (lo, hi, u_lo, u_hi, z_lo, z_hi)
    return _STRETCH_U_BOUNDS_CACHE[key]


def _affine_stretch_to_unit(t, lo: float, hi: float):
    span = max(float(hi) - float(lo), 1e-12)
    if isinstance(t, torch.Tensor):
        return (t - lo) / span
    return (np.asarray(t, dtype=np.float64) - lo) / span


def _sample_truncated_standard_normal_np(
    n: int,
    low: float,
    high: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """``z ~ N(0,1)`` conditioned on ``z in [low, high]`` (rejection)."""
    out: list[np.ndarray] = []
    need = n
    for _ in range(32):
        if need <= 0:
            break
        z = rng.normal(0.0, 1.0, size=need * 2)
        ok = (z >= low) & (z <= high)
        if ok.any():
            out.append(z[ok])
            need -= int(ok.sum())
    if need > 0:
        out.append(rng.uniform(low, high, size=need))
    return np.concatenate(out)[:n]


def _sample_truncated_standard_normal_torch(
    size: int,
    low: float,
    high: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    low_t = torch.tensor(low, device=device, dtype=dtype)
    high_t = torch.tensor(high, device=device, dtype=dtype)
    buf: list[torch.Tensor] = []
    need = size
    for _ in range(32):
        if need <= 0:
            break
        z = torch.randn(need * 2, device=device, dtype=dtype)
        m = (z >= low_t) & (z <= high_t)
        if m.any():
            buf.append(z[m])
            need -= int(m.sum().item())
    if need > 0:
        buf.append(low_t + (high_t - low_t) * torch.rand(need, device=device, dtype=dtype))
    return torch.cat(buf)[:size]


def sample_stretched_shifted_u(
    n: int,
    shift: float,
    percentiles: Tuple[float, float] = DEFAULT_SHIFT_CLAMP_PERCENTILES,
    *,
    rng: Optional[np.random.Generator] = None,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> Union[np.ndarray, torch.Tensor]:
    """``u = sigmoid(z)`` with ``z ~ N(0,1)`` truncated so ``shifted_t(u) in [lo, hi]``."""
    lo, hi, _u_lo, _u_hi, z_lo, z_hi = shift_stretch_bounds(shift, percentiles)
    if device is not None:
        dtype = dtype or torch.float32
        z = _sample_truncated_standard_normal_torch(n, z_lo, z_hi, device, dtype)
        return torch.sigmoid(z)
    rng = rng if rng is not None else np.random.default_rng(0)
    z = _sample_truncated_standard_normal_np(n, z_lo, z_hi, rng)
    return (1.0 / (1.0 + np.exp(-z))).astype(np.float64)


def stretch_shifted_t_to_unit(
    t,
    shift: float,
    percentiles: Tuple[float, float] = DEFAULT_SHIFT_CLAMP_PERCENTILES,
):
    """Affine map assuming ``t`` already lies in ``[lo, hi]`` (no clip)."""
    lo, hi, _, _, _, _ = shift_stretch_bounds(shift, percentiles)
    return _affine_stretch_to_unit(t, lo, hi)


def clamp_shifted_t_pdf(t, shift: float, clamp_percentiles: Tuple[float, float] = DEFAULT_SHIFT_CLAMP_PERCENTILES):
    """Deprecated alias — prefer truncated sampling via :func:`sample_shifted_t_density`."""
    return stretch_shifted_t_to_unit(t, shift, clamp_percentiles)


def sample_shifted_t_density(
    shift: float,
    n: int = 10_000,
    rng: Optional[np.random.Generator] = None,
    stretch_percentiles: Optional[Tuple[float, float]] = None,
    *,
    clamp_percentiles: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    """MC samples: logitnormal ∘ shift, optionally LTX stretch (truncated, smooth tails)."""
    if clamp_percentiles is not None and stretch_percentiles is None:
        stretch_percentiles = clamp_percentiles
    if stretch_percentiles is not None:
        u = sample_stretched_shifted_u(n, shift, stretch_percentiles, rng=rng)
        t = np.asarray(shifted_t(u, shift), dtype=np.float64)
        lo, hi, _, _, _, _ = shift_stretch_bounds(shift, stretch_percentiles)
        return _affine_stretch_to_unit(t, lo, hi)
    rng = rng if rng is not None else np.random.default_rng(0)
    u_logit = rng.normal(0.0, 1.0, size=n)
    u = 1.0 / (1.0 + np.exp(-u_logit))
    return np.asarray(shifted_t(u, shift), dtype=np.float64)
