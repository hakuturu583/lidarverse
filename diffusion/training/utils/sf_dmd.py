# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lightweight mathematical helpers for Self-Forcing RF-DMD."""

from __future__ import annotations

import torch
from torch import Tensor


def fixed_sde_sigmas(
    *,
    num_steps: int = 4,
    shift: float = 5.0,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Return a fixed uniform-base RF-SDE ladder after timestep shifting."""
    if num_steps <= 0:
        raise ValueError(f"num_steps must be positive, got {num_steps}")
    if shift <= 0.0:
        raise ValueError(f"shift must be positive, got {shift}")
    # Preserve the original four-step schedule exactly for existing runs.
    if num_steps == 4 and shift == 5.0:
        return torch.tensor(
            [1.0, 15.0 / 16.0, 5.0 / 6.0, 5.0 / 8.0, 0.0],
            device=device,
            dtype=dtype,
        )
    base = torch.linspace(
        1.0,
        0.0,
        num_steps + 1,
        device=device,
        dtype=dtype,
    )
    return shift * base / (1.0 + (shift - 1.0) * base)


def confidence_weighted_geometry_source(
    noise: Tensor,
    geometry: Tensor,
    confidence: Tensor,
    noise_floor: float,
) -> Tensor:
    """Match the Stage-2 geometry-source RF endpoint distribution."""
    if geometry.shape != noise.shape:
        raise ValueError(
            "geometry source must match noise: "
            f"geometry={tuple(geometry.shape)}, noise={tuple(noise.shape)}"
        )
    expected_confidence = (noise.shape[0], 1, *noise.shape[2:])
    if confidence.shape != expected_confidence:
        raise ValueError(
            "geometry confidence must be (B,1,T,H,W): "
            f"confidence={tuple(confidence.shape)}, "
            f"expected={expected_confidence}"
        )
    if not 0.0 <= noise_floor <= 1.0:
        raise ValueError(f"noise_floor must be in [0,1], got {noise_floor}")
    source_weight = (1.0 - noise_floor) * confidence.float()
    return (
        source_weight * geometry.float()
        + (1.0 - source_weight) * noise.float()
    )


def confidence_composite_geometry(
    geometry: Tensor,
    confidence: Tensor,
    empty_value: float,
) -> Tensor:
    """Composite projected range geometry against the empty-range value."""
    if confidence.shape != geometry.shape:
        raise ValueError(
            "geometry confidence must match projected geometry: "
            f"confidence={tuple(confidence.shape)}, "
            f"geometry={tuple(geometry.shape)}"
        )
    confidence = confidence.to(
        device=geometry.device,
        dtype=geometry.dtype,
    ).clamp(0.0, 1.0)
    empty = geometry.new_full((), float(empty_value))
    return confidence * geometry + (1.0 - confidence) * empty


def complete_score_state_dict(
    loaded: dict[str, Tensor],
    target: dict[str, Tensor],
    fallback: dict[str, Tensor] | None = None,
) -> tuple[dict[str, Tensor], list[str]]:
    """Select compatible critic weights and fill newly added parameters."""
    completed = {}
    dropped = []
    mismatched = []
    for key, value in loaded.items():
        if key not in target:
            dropped.append(key)
        elif value.shape != target[key].shape:
            mismatched.append(
                f"{key}: checkpoint={tuple(value.shape)} "
                f"target={tuple(target[key].shape)}"
            )
        else:
            completed[key] = value
    if mismatched:
        raise ValueError(
            "score checkpoint shape mismatch: " + "; ".join(mismatched)
        )
    missing = [key for key in target if key not in completed]
    if fallback is not None:
        for key in list(missing):
            value = fallback.get(key)
            if value is not None and value.shape == target[key].shape:
                completed[key] = value
                missing.remove(key)
    if missing:
        raise KeyError(
            "score checkpoint is missing tensors: " + ", ".join(missing)
        )
    return completed, dropped


def rf_velocity_to_x0(
    x_t: Tensor, velocity: Tensor, sigma: Tensor | float
) -> Tensor:
    """Convert the repository's RF velocity ``noise-clean`` to clean x0."""
    sigma_tensor = torch.as_tensor(
        sigma, device=x_t.device, dtype=x_t.dtype
    )
    return x_t - sigma_tensor * velocity


def sde_renoise(
    x0: Tensor,
    eps_implied: Tensor,
    eps_fresh: Tensor,
    sigma_next: Tensor | float,
    eta: float,
) -> Tensor:
    """Advance one RF-SDE step with the repository's eta noise blend."""
    if not 0.0 <= eta <= 1.0:
        raise ValueError(f"eta must be in [0,1], got {eta}")
    if not (x0.shape == eps_implied.shape == eps_fresh.shape):
        raise ValueError(
            "x0/eps_implied/eps_fresh must have equal shapes, got "
            f"{x0.shape}/{eps_implied.shape}/{eps_fresh.shape}"
        )
    sigma = torch.as_tensor(
        sigma_next, device=x0.device, dtype=x0.dtype
    )
    eps_used = (1.0 - eta) * eps_implied + eta * eps_fresh
    return (1.0 - sigma) * x0 + sigma * eps_used


def dmd_pseudo_target(
    generated: Tensor,
    teacher_x0: Tensor,
    fake_x0: Tensor,
    *,
    eps: float = 1.0e-6,
) -> tuple[Tensor, Tensor]:
    """Build the detached adaptive x0-space VSD pseudo target."""
    if not (generated.shape == teacher_x0.shape == fake_x0.shape):
        raise ValueError(
            "generated/teacher_x0/fake_x0 must have equal shapes, got "
            f"{generated.shape}/{teacher_x0.shape}/{fake_x0.shape}"
        )
    reduce_dims = tuple(range(1, generated.ndim))
    weight = 1.0 / (
        (generated.detach() - teacher_x0.detach()).abs().mean(
            dim=reduce_dims, keepdim=True
        )
        + eps
    )
    vsd_grad = fake_x0.detach() - teacher_x0.detach()
    target = generated.detach() - weight * vsd_grad
    return target.detach(), weight.detach()


def active_slice_loss(
    prediction: Tensor,
    target: Tensor,
    active_start: int,
    active_end: int,
    *,
    half: bool = False,
) -> Tensor:
    """Mean squared error over one active temporal block only."""
    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction/target shape mismatch: {prediction.shape}/{target.shape}"
        )
    if not 0 <= active_start < active_end <= prediction.shape[2]:
        raise ValueError(
            f"invalid active slice [{active_start}:{active_end}] for "
            f"T={prediction.shape[2]}"
        )
    loss = (
        prediction[:, :, active_start:active_end]
        - target[:, :, active_start:active_end]
    ).square().mean()
    return 0.5 * loss if half else loss


def normalized_frame_weights(frame_counts: list[int]) -> list[float]:
    """Weight per-record mean losses like one mean over all frames."""
    if not frame_counts or any(count <= 0 for count in frame_counts):
        raise ValueError(
            f"frame_counts must be non-empty and positive, got {frame_counts}"
        )
    total_frames = sum(frame_counts)
    return [count / total_frames for count in frame_counts]


def sfdmd_update_counts(
    num_cycles: int, fake_updates_per_cycle: int
) -> dict[str, int]:
    """Return optimizer-step counts for a cycle-based SF-DMD run."""
    if num_cycles < 0 or fake_updates_per_cycle <= 0:
        raise ValueError(
            "num_cycles must be non-negative and fake_updates_per_cycle positive"
        )
    return {
        "cycles": num_cycles,
        "student": num_cycles,
        "fake_score": num_cycles * fake_updates_per_cycle,
        "total": num_cycles * (fake_updates_per_cycle + 1),
    }


def sfdmd_nonoverlap_layout(
    block_index: int,
    *,
    block_size: int = 3,
    score_window_tokens: int = 9,
    prefix_frames: int = 0,
) -> tuple[int, int, int, int]:
    """Map one global AR block to its local bidirectional score window.

    Returns ``(global_start, real_window_start, active_start, active_end)``.
    The first score window covers every seed chunk. Later windows score the
    newly generated final block. With a temporal prefix, chunk zero is the
    singleton prefix and subsequent chunks have ``block_size`` frames.
    """
    if (
        block_index < 0
        or block_size <= 0
        or score_window_tokens <= 0
        or prefix_frames < 0
        or prefix_frames >= score_window_tokens
        or (score_window_tokens - prefix_frames) % block_size != 0
    ):
        raise ValueError(
            "invalid block/prefix/window layout: "
            f"index={block_index}, prefix={prefix_frames}, "
            f"block={block_size}, window={score_window_tokens}"
        )
    seed_blocks = (
        (1 if prefix_frames else 0)
        + (score_window_tokens - prefix_frames) // block_size
    )
    if prefix_frames and block_index == 0:
        global_start = 0
        active_frames = prefix_frames
    else:
        body_index = block_index - int(bool(prefix_frames))
        global_start = prefix_frames + body_index * block_size
        active_frames = block_size
    if block_index < seed_blocks:
        return global_start, 0, global_start, global_start + active_frames
    real_window_start = global_start - (score_window_tokens - block_size)
    active_start = score_window_tokens - block_size
    return (
        global_start,
        real_window_start,
        active_start,
        score_window_tokens,
    )


def build_sfdmd_block_timesteps(
    batch_size: int,
    num_frames: int,
    active_start: int,
    block_size: int,
    active_timestep: Tensor | float,
    *,
    max_timestep: float = 1000.0,
    device: torch.device | str | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Build clean-prefix / active / pure-noise-future block schedules."""
    active_end = active_start + block_size
    if (
        batch_size <= 0
        or block_size <= 0
        or not 0 <= active_start < active_end <= num_frames
    ):
        raise ValueError(
            f"invalid B/T/active/block: {batch_size}/{num_frames}/"
            f"{active_start}/{block_size}"
        )
    timesteps = torch.full(
        (batch_size, num_frames),
        max_timestep,
        device=device,
        dtype=torch.float32,
    )
    active = torch.zeros_like(timesteps, dtype=torch.bool)
    active[:, active_start:active_end] = True
    active_value = torch.as_tensor(
        active_timestep, device=timesteps.device, dtype=timesteps.dtype
    )
    if active_value.ndim == 0:
        timesteps[active] = active_value
    elif tuple(active_value.shape) == (batch_size,):
        timesteps[:, active_start:active_end] = active_value[:, None]
    else:
        raise ValueError(
            "active_timestep must be scalar or (B,), got "
            f"{tuple(active_value.shape)}"
        )
    known = torch.zeros_like(active)
    known[:, :active_start] = True
    timesteps[known] = 0.0
    return timesteps, known, active
