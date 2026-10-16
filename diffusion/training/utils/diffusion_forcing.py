# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small dependency-free helpers for blockwise Diffusion Forcing."""

import torch
from torch import Tensor


def temporal_chunk_slices(
    num_frames: int,
    block_size: int,
    prefix_frames: int = 0,
) -> list[tuple[int, int]]:
    """Partition time into an optional prefix and complete causal blocks."""
    if num_frames <= 0:
        raise ValueError(f"num_frames must be positive, got {num_frames}")
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    if prefix_frames < 0 or prefix_frames > num_frames:
        raise ValueError(
            "prefix_frames must be in [0, num_frames], got "
            f"{prefix_frames}/{num_frames}"
        )
    body_frames = num_frames - prefix_frames
    if body_frames % block_size != 0:
        raise ValueError(
            "temporal body must contain complete causal blocks: "
            f"({num_frames}-{prefix_frames}) % {block_size} != 0"
        )
    chunks = []
    if prefix_frames:
        chunks.append((0, prefix_frames))
    chunks.extend(
        (start, start + block_size)
        for start in range(prefix_frames, num_frames, block_size)
    )
    return chunks


def temporal_stream_chunk_bounds(
    chunk_index: int,
    block_size: int,
    prefix_frames: int = 0,
) -> tuple[int, int]:
    """Return one cached-call interval for a prefix-plus-block stream."""
    if chunk_index < 0:
        raise ValueError("chunk_index must be non-negative")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if prefix_frames < 0:
        raise ValueError("prefix_frames must be non-negative")
    if prefix_frames and chunk_index == 0:
        return 0, prefix_frames
    body_index = chunk_index - int(prefix_frames > 0)
    start = prefix_frames + body_index * block_size
    return start, start + block_size


def pad_temporal_stream_tokens(
    num_tokens: int,
    block_size: int,
    prefix_frames: int = 0,
) -> int:
    """Pad a stream to a complete prefix-plus-block boundary."""
    if num_tokens < prefix_frames:
        raise ValueError(
            f"num_tokens={num_tokens} is smaller than prefix={prefix_frames}"
        )
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    body_tokens = num_tokens - prefix_frames
    padded_body = (
        (body_tokens + block_size - 1) // block_size
    ) * block_size
    return prefix_frames + padded_body


def is_temporal_stream_boundary(
    num_tokens: int,
    block_size: int,
    prefix_frames: int = 0,
) -> bool:
    """Whether ``num_tokens`` ends after the prefix or a complete block."""
    if block_size <= 0 or prefix_frames < 0 or num_tokens < prefix_frames:
        return False
    return (num_tokens - prefix_frames) % block_size == 0


def temporal_block_allows(
    query_frame: Tensor | int,
    key_frame: Tensor | int,
    block_size: int,
):
    """Whether a query block may attend to a key block."""
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    return (key_frame // block_size) <= (query_frame // block_size)


def temporal_prefix_block_allows(
    query_frame: Tensor | int,
    key_frame: Tensor | int,
    block_size: int,
    prefix_frames: int,
):
    """Prefix-bidirectional, then block-causal temporal attention.

    Prefix queries see only the complete prefix. Future queries see the prefix,
    their own future block, and earlier future blocks. ``prefix_frames=0`` is
    exactly ``temporal_block_allows``.
    """
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    if prefix_frames < 0:
        raise ValueError(
            f"prefix_frames must be non-negative, got {prefix_frames}"
        )
    query_is_prefix = query_frame < prefix_frames
    key_is_prefix = key_frame < prefix_frames
    query_is_future = query_frame >= prefix_frames
    key_is_future = key_frame >= prefix_frames
    query_block = (query_frame - prefix_frames) // block_size
    key_block = (key_frame - prefix_frames) // block_size
    future_allows = key_is_prefix | (
        key_is_future & (key_block <= query_block)
    )
    return (query_is_prefix & key_is_prefix) | (
        query_is_future & future_allows
    )


def teacher_forcing_clean_noisy_allows(
    query_frame: Tensor | int,
    key_frame: Tensor | int,
    clean_frames: int,
    block_size: int,
    prefix_frames: int,
):
    """Visibility for a packed ``[clean | noisy]`` teacher-forcing pass.

    Clean queries use frame-causal inclusive visibility. A noisy query sees
    clean chunks strictly before its own chunk and noisy keys from its own
    chunk. This matches Imaginaire's frame-causal clean-history pass while
    preserving its chunkwise noisy-query layout; text remains cross-attention
    in the LiDAR DiT and is therefore not part of this packed mask.
    """
    if clean_frames <= 0:
        raise ValueError(
            f"clean_frames must be positive, got {clean_frames}"
        )
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    if prefix_frames < 0 or prefix_frames > clean_frames:
        raise ValueError(
            "prefix_frames must be in [0, clean_frames], got "
            f"{prefix_frames}/{clean_frames}"
        )

    query_is_clean = query_frame < clean_frames
    key_is_clean = key_frame < clean_frames
    query_is_noisy = query_frame >= clean_frames
    key_is_noisy = key_frame >= clean_frames
    query_local = query_frame - query_is_noisy * clean_frames
    key_local = key_frame - key_is_noisy * clean_frames

    clean_query_allows = (
        query_is_clean & key_is_clean & (key_local <= query_local)
    )

    query_is_prefix = query_local < prefix_frames
    key_is_prefix = key_local < prefix_frames
    query_is_future = query_local >= prefix_frames
    key_is_future = key_local >= prefix_frames
    query_block = (query_local - prefix_frames) // block_size
    key_block = (key_local - prefix_frames) // block_size
    clean_strictly_precedes = query_is_future & (
        key_is_prefix
        | (key_is_future & (key_block < query_block))
    )
    same_chunk = (query_is_prefix & key_is_prefix) | (
        query_is_future
        & key_is_future
        & (query_block == key_block)
    )
    noisy_query_allows = query_is_noisy & (
        (key_is_clean & clean_strictly_precedes)
        | (key_is_noisy & same_chunk)
    )
    return clean_query_allows | noisy_query_allows


def expand_blockwise(
    block_values_B_K: Tensor,
    num_frames: int,
    block_size: int,
) -> Tensor:
    """Broadcast ``(B,K)`` values onto ``(B,T)`` consecutive frame blocks."""
    if block_values_B_K.ndim != 2:
        raise ValueError(
            f"block_values must be (B,K), got {tuple(block_values_B_K.shape)}"
        )
    if num_frames <= 0 or block_size <= 0:
        raise ValueError(
            f"num_frames and block_size must be positive, got "
            f"{num_frames}, {block_size}"
        )
    num_blocks = (num_frames + block_size - 1) // block_size
    if block_values_B_K.shape[1] != num_blocks:
        raise ValueError(
            f"need K={num_blocks} blocks for T={num_frames}, "
            f"block_size={block_size}; got K={block_values_B_K.shape[1]}"
        )
    block_ids = (
        torch.arange(num_frames, device=block_values_B_K.device)
        // block_size
    )
    return block_values_B_K[:, block_ids]


def expand_chunkwise(
    chunk_values_B_K: Tensor,
    num_frames: int,
    block_size: int,
    prefix_frames: int = 0,
) -> Tensor:
    """Broadcast values over an optional prefix and complete causal blocks."""
    if chunk_values_B_K.ndim != 2:
        raise ValueError(
            f"chunk values must be (B,K), got {tuple(chunk_values_B_K.shape)}"
        )
    chunks = temporal_chunk_slices(
        num_frames,
        block_size,
        prefix_frames,
    )
    if chunk_values_B_K.shape[1] != len(chunks):
        raise ValueError(
            f"need K={len(chunks)} chunks for T={num_frames}, "
            f"prefix={prefix_frames}, block_size={block_size}; got "
            f"K={chunk_values_B_K.shape[1]}"
        )
    return torch.cat(
        [
            chunk_values_B_K[:, index : index + 1].expand(-1, end - start)
            for index, (start, end) in enumerate(chunks)
        ],
        dim=1,
    )


def artifixer_flow_schedule(
    timestep_ids: Tensor,
    *,
    num_train_timesteps: int = 1000,
    shift: float = 5.0,
) -> tuple[Tensor, Tensor, Tensor]:
    """Map discrete ArtiFixer IDs to timestep, sigma, and training weight."""
    if num_train_timesteps <= 0:
        raise ValueError(
            "num_train_timesteps must be positive, got "
            f"{num_train_timesteps}"
        )
    if shift <= 0.0:
        raise ValueError(f"shift must be positive, got {shift}")
    if timestep_ids.dtype not in (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    ):
        raise TypeError(
            f"timestep_ids must be integer, got {timestep_ids.dtype}"
        )
    # IDs are sampled in-range on the training hot path. Keep validation for
    # CPU callers without introducing a device synchronization every update.
    if timestep_ids.device.type == "cpu" and bool(
        ((timestep_ids < 0) | (timestep_ids >= num_train_timesteps)).any()
    ):
        raise ValueError(
            f"timestep_ids must be in [0,{num_train_timesteps})"
        )

    base_ladder = torch.linspace(
        1.0,
        0.0,
        num_train_timesteps + 1,
        device=timestep_ids.device,
        dtype=torch.float32,
    )[:-1]
    sigma_ladder = (
        shift
        * base_ladder
        / (1.0 + (shift - 1.0) * base_ladder)
    )
    timestep_ladder = sigma_ladder * num_train_timesteps
    bell = torch.exp(
        -2.0
        * (
            (
                timestep_ladder - num_train_timesteps / 2.0
            )
            / num_train_timesteps
        )
        ** 2
    )
    bell = bell - bell.min()
    weight_ladder = bell * num_train_timesteps / bell.sum()
    return (
        timestep_ladder[timestep_ids],
        sigma_ladder[timestep_ids],
        weight_ladder[timestep_ids],
    )


def expand_after_prefix(
    block_values_B_K: Tensor,
    num_frames: int,
    block_size: int,
    prefix_frames: int,
    prefix_value: float = 0.0,
) -> Tensor:
    """Broadcast block values after a fixed prefix.

    Returns ``(B,T)`` with ``prefix_value`` in ``[:prefix_frames]`` and
    consecutive ``block_size`` groups thereafter.
    """
    if not 0 <= prefix_frames < num_frames:
        raise ValueError(
            f"prefix_frames={prefix_frames} must be in [0,{num_frames})"
        )
    future_frames = num_frames - prefix_frames
    future = expand_blockwise(
        block_values_B_K,
        num_frames=future_frames,
        block_size=block_size,
    )
    prefix = future.new_full(
        (future.shape[0], prefix_frames),
        float(prefix_value),
    )
    return torch.cat([prefix, future], dim=1)


def mean_after_prefix(values_B_T: Tensor, prefix_frames: int) -> Tensor:
    """Per-sample mean over future frames only."""
    if values_B_T.ndim != 2:
        raise ValueError(
            f"values must be (B,T), got {tuple(values_B_T.shape)}"
        )
    if not 0 <= prefix_frames < values_B_T.shape[1]:
        raise ValueError(
            f"prefix_frames={prefix_frames} must be in "
            f"[0,{values_B_T.shape[1]})"
        )
    return values_B_T[:, prefix_frames:].mean(dim=1)


def build_clean_context_timesteps(
    active_timesteps_B: Tensor,
    known_mask_B_T: Tensor,
) -> Tensor:
    """Set committed prefix frames to t=0 and active frames to the sampler t."""
    if active_timesteps_B.ndim != 1:
        raise ValueError(
            f"active timesteps must be (B,), got {tuple(active_timesteps_B.shape)}"
        )
    if known_mask_B_T.ndim != 2:
        raise ValueError(
            f"known mask must be (B,T), got {tuple(known_mask_B_T.shape)}"
        )
    if active_timesteps_B.shape[0] != known_mask_B_T.shape[0]:
        raise ValueError(
            f"timestep batch {active_timesteps_B.shape[0]} does not match "
            f"mask batch {known_mask_B_T.shape[0]}"
        )
    active_B_T = active_timesteps_B[:, None].expand_as(known_mask_B_T)
    return torch.where(
        known_mask_B_T.to(torch.bool),
        torch.zeros_like(active_B_T),
        active_B_T,
    )


def validate_prefix_mask(
    known_mask_B_T: Tensor,
    block_size: int,
) -> tuple[Tensor, Tensor]:
    """Validate block-aligned known-prefix masks and return known/active counts."""
    if known_mask_B_T.ndim != 2:
        raise ValueError(
            f"known mask must be (B,T), got {tuple(known_mask_B_T.shape)}"
        )
    mask = known_mask_B_T.to(torch.bool)
    _, num_frames = mask.shape
    n_known = mask.sum(dim=1)
    expected = (
        torch.arange(num_frames, device=mask.device)[None, :]
        < n_known[:, None]
    )
    if not torch.equal(mask, expected):
        raise ValueError(
            "known mask must be a contiguous prefix followed by active frames"
        )
    if bool((n_known % block_size != 0).any()):
        raise ValueError(
            f"known prefix lengths {n_known.tolist()} must align to "
            f"block_size={block_size}"
        )
    n_active = num_frames - n_known
    if bool((n_active <= 0).any()) or bool((n_active > block_size).any()):
        raise ValueError(
            f"active lengths must be in [1,{block_size}], got "
            f"{n_active.tolist()}"
        )
    return n_known, n_active
