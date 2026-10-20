# =============================================================================
# Transformer VAE with hierarchical hourglass architecture,
# neighborhood attention (natten) for spatial processing, and
# causal temporal attention for video.
#
# Architecture:
#   Encoder: Tokenizer → [spatial blocks + PatchMerging] (level 0, no temporal)
#            → [spatial→temporal blocks + PatchMerging (+ TemporalMerging)] × (N-1)
#            → global spatial+temporal → z
#   Decoder: z → global temporal+spatial
#            → [TemporalExpanding + PatchExpanding + temporal→spatial blocks] × (N-1)
#            → [PatchExpanding + spatial blocks] (level 0, no temporal) → Detokenizer
#
# No skip connections between encoder and decoder.
#
# Based on: https://github.com/crowsonkb/k-diffusion/blob/master/
#           k_diffusion/models/image_transformer_v2.py
# Call `natten.use_fused_na(True)` for acceleration on GPUs; without NATTEN,
# utils.compat.na2d runs the same neighborhood attention in PyTorch.
# =============================================================================

import contextlib
import functools
import math
from collections import namedtuple
from typing import List, Literal

import einops
import torch
import torch.distributed as dist
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from loguru import logger
from torch import nn
from torch.nn.modules.utils import _pair
from torch.utils.checkpoint import checkpoint

from tokenizer.networks.common import ContinuousFormulation
from utils.compat import na2d
from tokenizer.networks.transformers.rope3d import (
    VideoRopePosition3DEmb,
    apply_rotary_emb,
)
from . import encoding, ops

__all__ = ["TransformerVAE"]

NetworkEval = namedtuple(
    "NetworkEval", ["reconstructions", "posteriors", "latent", "validity_logits"]
)
# Default ``validity_logits`` to None so non-validity networks/call sites that
# construct NetworkEval without it keep working unchanged.
NetworkEval.__new__.__defaults__ = (None,)


def _run_block(module: nn.Module, *args, activation_checkpointing: bool):
    if activation_checkpointing and torch.is_grad_enabled():
        return checkpoint(module, *args, use_reentrant=False)
    return module(*args)


# =============================================================================
# Normalization
# =============================================================================


class RMSNorm(torch.nn.Module):
    def __init__(self, in_dim: int, scale: bool = True, eps: float = 1e-6):
        super().__init__()
        self.in_dim = in_dim
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(in_dim)) if scale else 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fp32 = x.float()
        x_normed = x_fp32 * torch.rsqrt(x_fp32.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x_normed * self.scale).to(x)

    def extra_repr(self) -> str:
        return f"in_dim={self.in_dim}"


# =============================================================================
# Rotary positional encoding (spatial)
# =============================================================================


class AxialRoPE(nn.Module):
    def __init__(self, dim: int, num_heads: int, max_harmonics: List[int]):
        super().__init__()
        freqs_h = self.setup_freqs(num_heads * dim // 4, max_harmonics[0])
        freqs_w = self.setup_freqs(num_heads * dim // 4, max_harmonics[1])
        self.register_buffer("freqs_h", freqs_h.view(dim // 4, num_heads).T)
        self.register_buffer("freqs_w", freqs_w.view(dim // 4, num_heads).T)

    def setup_freqs(self, dim: int, max_harmonics: int):
        return torch.linspace(math.log(1), math.log(max_harmonics), dim).exp().round()

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        coords = einops.rearrange(coords, "b c h w -> b h w c")
        radian_h = coords[..., None, [0]] * self.freqs_h
        radian_w = coords[..., None, [1]] * self.freqs_w
        return torch.cat((radian_h, radian_w), dim=-1)

    @staticmethod
    def rotate(x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        y1 = x1 * theta.cos() - x2 * theta.sin()
        y2 = x1 * theta.sin() + x2 * theta.cos()
        return torch.cat((y1, y2), dim=-1)

    def extra_repr(self) -> str:
        return (
            f"freqs_h={tuple(self.freqs_h.shape)}, freqs_w={tuple(self.freqs_w.shape)}"
        )


# =============================================================================
# Spatial self-attention blocks
# =============================================================================


class GlobalSelfAttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        dropout: float = 0.0,
        rope_max_harmonics: List[int] = (1, 1),
        bias=False,
        eps=1e-6,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // self.num_heads
        self.eps = eps

        self.norm = RMSNorm(dim)
        self.scale = nn.Parameter(torch.full([self.num_heads, 1], math.log(10.0)))
        self.qkv_proj = nn.Linear(dim, dim * 3, bias=bias)
        self.rope = AxialRoPE(self.head_dim, num_heads, rope_max_harmonics)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(dim, dim, bias=bias).apply(ops.zero_out)

    def scale_qk(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        scale = self.scale.clamp(max=math.log(100)).exp().sqrt()
        q = (F.normalize(q, p=2, dim=-1, eps=self.eps) * scale).to(q.dtype)
        k = (F.normalize(k, p=2, dim=-1, eps=self.eps) * scale).to(k.dtype)
        return q, k

    def apply_rope_qk(self, q: torch.Tensor, k: torch.Tensor, coords: torch.Tensor):
        theta = self.rope(coords)
        d = theta.shape[-1] * 2
        assert (q.shape[-1] >= d) and (k.shape[-1] >= d)
        q[..., :d] = self.rope.rotate(q[..., :d], theta).to(q.dtype)
        k[..., :d] = self.rope.rotate(k[..., :d], theta).to(k.dtype)
        return q, k

    def residual(self, x: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.shape
        h = self.norm(x)
        qkv = self.qkv_proj(h)
        q, k, v = einops.rearrange(
            qkv, "B H W (T N D) -> T B H W N D", T=3, D=self.head_dim
        )
        q, k = self.scale_qk(q, k)
        q, k = self.apply_rope_qk(q, k, coords)
        q = einops.rearrange(q, "B H W N D -> B N (H W) D")
        k = einops.rearrange(k, "B H W N D -> B N (H W) D")
        v = einops.rearrange(v, "B H W N D -> B N (H W) D")
        h = F.scaled_dot_product_attention(q, k, v, scale=1.0)
        h = einops.rearrange(h, "B N (H W) D -> B H W (N D)", H=H, W=W)
        h = self.dropout(h)
        h = self.out_proj(h)
        return h

    def forward(self, x: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        return x + self.residual(x, coords)

    def extra_repr(self) -> str:
        return f"head_dim={self.head_dim}, num_heads={self.num_heads}"


class CircularNeighborhoodSelfAttentionBlock(GlobalSelfAttentionBlock):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        kernel_size: List[int],
        dilation: List[int] = 1,
        dropout: float = 0.0,
        rope_max_harmonics: List[int] = (1, 1),
        # When True (default), the W (last spatial) axis is treated as
        # periodic: it is circularly padded before the neighborhood attention
        # so the seam wraps (range-image azimuth, column 0 ~ column W-1). When
        # False, no wrap -- ``natten`` clamps the neighborhood at the W edges
        # (the same boundary handling it already uses for H). Set False for a
        # Cartesian grid (e.g. BEV x-axis) where the two W edges are unrelated.
        circular: bool = True,
    ):
        super().__init__(
            dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            rope_max_harmonics=rope_max_harmonics,
        )
        self.kernel_size = _pair(kernel_size)
        self.dilation = _pair(dilation)
        self.circular = circular
        self._inference_natten_kwargs = {}

    def set_inference_natten_config(
        self,
        *,
        backend: str,
        q_tile_shape: tuple[int, int],
        kv_tile_shape: tuple[int, int],
    ) -> None:
        self._inference_natten_kwargs = {
            "backend": backend,
            "q_tile_shape": q_tile_shape,
            "kv_tile_shape": kv_tile_shape,
        }

    def before_attn(self, q, k, v):
        if not self.circular:
            return q, k, v
        padding = self.kernel_size[1] // 2
        q = F.pad(q, (0, 0, 0, 0, padding, padding), mode="circular")
        k = F.pad(k, (0, 0, 0, 0, padding, padding), mode="circular")
        v = F.pad(v, (0, 0, 0, 0, padding, padding), mode="circular")
        return q, k, v

    def after_attn(self, x):
        if not self.circular:
            return x
        padding = self.kernel_size[1] // 2
        x = x[:, :, padding:-padding]
        return x

    def residual(self, x: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        qkv = self.qkv_proj(h)
        q, k, v = einops.rearrange(
            qkv, "B H W (T N D) -> T B H W N D", T=3, D=self.head_dim
        )
        q, k = self.scale_qk(q, k)
        q, k = self.apply_rope_qk(q, k, coords)
        q, k, v = self.before_attn(q, k, v)
        h = na2d(
            query=q,
            key=k,
            value=v,
            kernel_size=self.kernel_size,
            dilation=self.dilation,
            scale=1.0,
            **self._inference_natten_kwargs,
        )
        h = einops.rearrange(h, "B H W N D -> B H W (N D)")
        h = self.after_attn(h)
        h = self.dropout(h)
        h = self.out_proj(h)
        return h

    def forward(self, x: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        return x + self.residual(x, coords)

    def extra_repr(self) -> str:
        return (
            f"head_dim={self.head_dim}, num_heads={self.num_heads}, "
            f"kernel_size={self.kernel_size}, dilation={self.dilation}"
        )


# =============================================================================
# Causal temporal self-attention
# =============================================================================


class CausalTemporalAttention(nn.Module):
    """Causal self-attention along the temporal dimension.

    Input/output shape: (N, T, C) where N = B*H*W.
    Each position attends only to itself and earlier positions.
    """

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0, bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.norm = RMSNorm(dim)
        self.qkv_proj = nn.Linear(dim, dim * 3, bias=bias)
        self.out_proj = nn.Linear(dim, dim, bias=bias).apply(ops.zero_out)
        self.dropout = nn.Dropout(dropout)
        self._SDPA_MAX_BATCH = 65535

    def _project_qkv(self, x: torch.Tensor):
        h = self.norm(x)
        qkv = self.qkv_proj(h)
        return einops.rearrange(
            qkv, "N T (three heads D) -> three N heads T D",
            three=3, D=self.head_dim,
        )

    def _sdpa(self, q, k, v, *, attn_mask=None, is_causal=False):
        N = q.shape[0]
        if N <= self._SDPA_MAX_BATCH:
            return F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, is_causal=is_causal
            )
        chunks = []
        for i in range(0, N, self._SDPA_MAX_BATCH):
            j = min(i + self._SDPA_MAX_BATCH, N)
            chunks.append(F.scaled_dot_product_attention(
                q[i:j], k[i:j], v[i:j],
                attn_mask=attn_mask, is_causal=is_causal,
            ))
        return torch.cat(chunks, dim=0)

    def _finish(self, x, h):
        h = einops.rearrange(h, "N heads T D -> N T (heads D)")
        h = self.dropout(h)
        h = self.out_proj(h)
        return x + h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = self._project_qkv(x)
        return self._finish(
            x, self._sdpa(q, k, v, is_causal=True)
        )

    def forward_stream(
        self,
        x: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Attend new temporal tokens to cached history and this causal block."""
        q, k_new, v_new = self._project_qkv(x)
        past = 0
        if kv_cache is not None:
            k_past, v_past = kv_cache
            if (
                k_past.shape[:2] != k_new.shape[:2]
                or k_past.shape[3:] != k_new.shape[3:]
                or v_past.shape != k_past.shape
            ):
                raise ValueError(
                    "temporal KV cache shape mismatch: "
                    f"cached k/v={tuple(k_past.shape)}/{tuple(v_past.shape)}, "
                    f"new k={tuple(k_new.shape)}"
                )
            past = k_past.shape[2]
            k = torch.cat([k_past, k_new], dim=2)
            v = torch.cat([v_past, v_new], dim=2)
        else:
            k, v = k_new, v_new

        T_new = q.shape[2]
        q_index = past + torch.arange(T_new, device=q.device)[:, None]
        k_index = torch.arange(past + T_new, device=q.device)[None, :]
        causal_offset_mask = k_index <= q_index
        out = self._finish(
            x,
            self._sdpa(
                q, k, v,
                attn_mask=causal_offset_mask,
                is_causal=False,
            ),
        )
        return out, (k.detach(), v.detach())

    def extra_repr(self) -> str:
        return f"head_dim={self.head_dim}, num_heads={self.num_heads}"


# =============================================================================
# Feed-forward
# =============================================================================


class GEGLU(nn.Linear):
    def __init__(self, in_features, out_features, bias):
        super().__init__(in_features, out_features * 2, bias=bias)

    def forward(self, x):
        h = super().forward(x)
        h, gate = h.chunk(2, dim=-1)
        return h * F.gelu(gate)


class FeedForwardNetwork(nn.Module):
    def __init__(self, dim, mid_dim, dropout=0.0):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.gegelu = GEGLU(dim, mid_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(mid_dim, dim, bias=False).apply(ops.zero_out)

    def residual(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.gegelu(x)
        x = self.dropout(x)
        x = self.linear(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.residual(x)


# =============================================================================
# Spatial block (spatial attention + FFN)
# =============================================================================


class SpatialBlock(nn.Module):
    """Spatial attention + FFN.  Operates on (BT, H, W, C) per-frame."""

    def __init__(
        self,
        in_dim: int,
        num_heads: int,
        attn_type: Literal["local", "global"] = "global",
        kernel_size: List[int] = None,
        dilation: List[int] = 1,
        rope_max_harmonics: List[int] = (1, 1),
        mlp_ratio: float = 3.0,
        dropout: float = 0.0,
        # Forwarded to the local (neighborhood) attention: whether the W axis
        # wraps (periodic azimuth) or clamps (Cartesian). No effect for global.
        circular: bool = True,
    ):
        super().__init__()

        if attn_type == "global":
            self.residual_attn = GlobalSelfAttentionBlock(
                dim=in_dim,
                num_heads=num_heads,
                dropout=dropout,
                rope_max_harmonics=rope_max_harmonics,
            )
        elif attn_type == "local":
            self.residual_attn = CircularNeighborhoodSelfAttentionBlock(
                dim=in_dim,
                num_heads=num_heads,
                kernel_size=kernel_size,
                dilation=dilation,
                dropout=dropout,
                rope_max_harmonics=rope_max_harmonics,
                circular=circular,
            )

        self.residual_ffn = FeedForwardNetwork(
            dim=in_dim,
            mid_dim=int(in_dim * mlp_ratio),
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        x = self.residual_attn(x, coords)
        x = self.residual_ffn(x)
        return x


class TemporalBlock(nn.Module):
    """Causal temporal attention + FFN.  Operates on (BHW, T, C)."""

    def __init__(
        self,
        in_dim: int,
        num_heads: int,
        mlp_ratio: float = 3.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.temporal_attn = CausalTemporalAttention(in_dim, num_heads, dropout)
        self.ffn = FeedForwardNetwork(in_dim, int(in_dim * mlp_ratio), dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.temporal_attn(x)
        x = self.ffn(x)
        return x

    def forward_stream(
        self,
        x: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        x, kv_cache = self.temporal_attn.forward_stream(x, kv_cache)
        x = self.ffn(x)
        return x, kv_cache


class CausalTemporalConv(nn.Module):
    """Causal temporal convolution along T (WAN-style).

    The conv analog of :class:`CausalTemporalAttention` -- same
    (N, T, C) interface (N = B*H*W), same zero-init residual so the
    block is a residual add at init. Implemented as a depth-preserving
    ``Conv1d`` over the T axis with left-only causal padding (kernel-1),
    so each output only depends on the current and past frames -- the
    ``kernel_size x 1 x 1`` case of WanVAE's ``CausalConv3d`` (spatial
    mixing is left to the surrounding neighborhood-attention blocks, so
    swapping only the temporal mixer stays a clean ablation).
    """

    def __init__(self, dim: int, kernel_size: int = 3, dropout: float = 0.0):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.conv = nn.Conv1d(dim, dim, kernel_size=kernel_size, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False).apply(ops.zero_out)
        self.dropout = nn.Dropout(dropout)
        self.causal_pad = kernel_size - 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, T, C)
        h = self.norm(x)
        h = einops.rearrange(h, "N T C -> N C T")
        h = F.pad(h, (self.causal_pad, 0))
        h = self.conv(h)
        h = einops.rearrange(h, "N C T -> N T C")
        h = self.dropout(h)
        h = self.out_proj(h)
        return x + h


class TemporalConvBlock(nn.Module):
    """Causal temporal conv + FFN.  Operates on (BHW, T, C).

    Drop-in replacement for :class:`TemporalBlock` (identical constructor
    signature and (N, T, C) forward) that swaps the temporal *attention*
    for a WAN-style causal temporal *convolution*. ``num_heads`` is
    accepted only for signature parity (unused).
    """

    def __init__(
        self,
        in_dim: int,
        num_heads: int,
        mlp_ratio: float = 3.0,
        dropout: float = 0.0,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.temporal_conv = CausalTemporalConv(in_dim, kernel_size, dropout)
        self.ffn = FeedForwardNetwork(in_dim, int(in_dim * mlp_ratio), dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.temporal_conv(x)
        x = self.ffn(x)
        return x


# =============================================================================
# Joint 3D self-attention block
# =============================================================================


class Joint3DSelfAttention(nn.Module):
    """Joint self-attention over flattened (T*H*W) tokens.

    Mirrors :class:`GlobalSelfAttentionBlock` but operates jointly over time
    and space instead of per-frame.

    Two positional-encoding modes are supported via ``use_3d_rope``:

    * **Default (``use_3d_rope=False``)** -- spatial position via the
      existing :class:`AxialRoPE` on ``(h, w)`` (broadcast across time, since
      polar coords don't vary with time), and time position via a *learnable*
      1D embedding added to the pre-attention features. This is what
      original training runs used; kept as the default for backward compat.
      Caveat: the learnable temporal slots past the largest ``T`` seen in
      training are uninitialized noise, so this mode does *not* extrapolate
      cleanly to longer-than-trained sequence lengths at inference.
    * **``use_3d_rope=True``** -- replace both the spatial :class:`AxialRoPE`
      and the learnable temporal PE with a single 3D RoPE
      (:class:`VideoRopePosition3DEmb`) over ``(t, h, w)``. RoPE is a
      rotation (length-preserving), and is fully parameter-free, so the
      angles extrapolate to longer sequences with no untrained parameters.
      The spatial axes of the 3D RoPE play the same role :class:`AxialRoPE`
      did, but without coupling to polar coords -- the bottleneck operates
      on a regular (H_z, W_z) grid so a Cartesian RoPE is well-defined and
      consistent with how the rest of the literature does video VAEs.

    Causality is *orthogonal* to the PE choice: ``causal_time=True`` masks
    the joint attention so each ``(t, h, w)`` token only attends to
    ``(t', h', w')`` with ``t' <= t``. The causal mask is materialized as
    a ``(T*H*W, T*H*W)`` bool tensor so memory scales with ``(T*H*W)^2``.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        max_t: int,
        rope_max_harmonics: List[int] = (1, 1),
        dropout: float = 0.0,
        causal_time: bool = False,
        bias: bool = False,
        eps: float = 1e-6,
        # If True, encode both spatial and temporal position with a single
        # 3D RoPE (length-preserving rotation, no learnable PE buffer) and
        # skip the original 2D AxialRoPE + learnable `temporal_pe` path.
        # ``len_h``/``len_w`` set the spatial cap of the 3D RoPE cache.
        use_3d_rope: bool = False,
        len_h: int = 1,
        len_w: int = 1,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.eps = eps
        self.causal_time = causal_time
        self.use_3d_rope = use_3d_rope
        self.max_t = max_t
        self.max_h = len_h if use_3d_rope else 0
        self.max_w = len_w if use_3d_rope else 0

        self.norm = RMSNorm(dim)
        self.scale = nn.Parameter(torch.full([num_heads, 1], math.log(10.0)))
        self.qkv_proj = nn.Linear(dim, dim * 3, bias=bias)
        if use_3d_rope:
            self.rope = None
            self.temporal_pe = None
            self.rope3d = VideoRopePosition3DEmb(
                head_dim=self.head_dim,
                len_t=max_t,
                len_h=len_h,
                len_w=len_w,
            )
        else:
            self.rope = AxialRoPE(self.head_dim, num_heads, rope_max_harmonics)
            self.temporal_pe = nn.Parameter(torch.zeros(max_t, dim))
            nn.init.trunc_normal_(self.temporal_pe, std=0.02)
            self.rope3d = None
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(dim, dim, bias=bias).apply(ops.zero_out)

    def _scale_qk(self, q: torch.Tensor, k: torch.Tensor):
        scale = self.scale.clamp(max=math.log(100)).exp().sqrt()
        q = (F.normalize(q, p=2, dim=-1, eps=self.eps) * scale).to(q.dtype)
        k = (F.normalize(k, p=2, dim=-1, eps=self.eps) * scale).to(k.dtype)
        return q, k

    def _apply_spatial_rope(self, q: torch.Tensor, k: torch.Tensor, coords: torch.Tensor):
        theta = self.rope(coords)  # (B_c, H, W, N, D//2)
        d = theta.shape[-1] * 2
        assert (q.shape[-1] >= d) and (k.shape[-1] >= d)
        theta = theta.unsqueeze(1)  # broadcast over T → (B_c, 1, H, W, N, D//2)
        q[..., :d] = self.rope.rotate(q[..., :d], theta).to(q.dtype)
        k[..., :d] = self.rope.rotate(k[..., :d], theta).to(k.dtype)
        return q, k

    def residual(self, x: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        B, T, H, W, C = x.shape
        if T > self.max_t:
            raise ValueError(
                f"Joint3DSelfAttention received T={T} but was constructed with "
                f"max_t={self.max_t}. Increase bottleneck_3d_max_t."
            )
        if self.use_3d_rope and (H > self.max_h or W > self.max_w):
            raise ValueError(
                f"Joint3DSelfAttention received (H, W)=({H}, {W}) but the 3D "
                f"RoPE was sized for (len_h, len_w)=({self.max_h}, {self.max_w}). "
                f"This usually means the bottleneck spatial shape changed -- "
                f"rebuild the decoder with larger len_h/len_w."
            )
        h = self.norm(x)

        if self.use_3d_rope:
            # 3D RoPE path. We flatten (T, H, W) to a single sequence axis
            # so `apply_rotary_emb` can broadcast its (S, head_dim) rope
            # tensor over batch + num_heads in one shot.
            qkv = self.qkv_proj(h)
            qkv = einops.rearrange(
                qkv, "B T H W (three N D) -> three B (T H W) N D",
                three=3, D=self.head_dim,
            )
            q, k, v = qkv[0], qkv[1], qkv[2]
            q, k = self._scale_qk(q, k)  # rotation preserves L2 norm
            rope_emb = self.rope3d(T, H, W)  # (T*H*W, head_dim)
            q = apply_rotary_emb(q, rope_emb)
            k = apply_rotary_emb(k, rope_emb)
            q = einops.rearrange(q, "B S N D -> B N S D")
            k = einops.rearrange(k, "B S N D -> B N S D")
            v = einops.rearrange(v, "B S N D -> B N S D")
        else:
            # Original: learnable temporal PE + 2D AxialRoPE over (h, w).
            h = h + self.temporal_pe[:T].view(1, T, 1, 1, C)
            qkv = self.qkv_proj(h)
            q, k, v = einops.rearrange(
                qkv, "B T H W (three N D) -> three B T H W N D",
                three=3, D=self.head_dim,
            )
            q, k = self._scale_qk(q, k)
            q, k = self._apply_spatial_rope(q, k, coords)
            q = einops.rearrange(q, "B T H W N D -> B N (T H W) D")
            k = einops.rearrange(k, "B T H W N D -> B N (T H W) D")
            v = einops.rearrange(v, "B T H W N D -> B N (T H W) D")

        if self.causal_time:
            S = H * W
            t_idx = torch.arange(T, device=q.device).repeat_interleave(S)
            # mask[i, j] = True iff query t_i can attend to key t_j (t_j <= t_i)
            mask = t_idx.unsqueeze(0) <= t_idx.unsqueeze(1)  # (T*S, T*S)
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=1.0)
        else:
            out = F.scaled_dot_product_attention(q, k, v, scale=1.0)

        out = einops.rearrange(out, "B N (T H W) D -> B T H W (N D)", T=T, H=H, W=W)
        out = self.dropout(out)
        out = self.out_proj(out)
        return out

    def forward(self, x: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        return x + self.residual(x, coords)

    def extra_repr(self) -> str:
        pe = (
            f"rope3d(len_t={self.max_t}, len_h={self.max_h}, len_w={self.max_w})"
            if self.use_3d_rope
            else f"max_t={self.max_t}, axial_rope_2d"
        )
        return (
            f"head_dim={self.head_dim}, num_heads={self.num_heads}, "
            f"{pe}, causal_time={self.causal_time}"
        )


class Bottleneck3DBlock(nn.Module):
    """Joint 3D self-attention + FFN. Operates on (B, T, H, W, C).

    Set ``use_3d_rope=True`` (with ``len_h`` / ``len_w`` set to the
    bottleneck's spatial shape) to swap the default learnable temporal PE
    + 2D AxialRoPE for a single 3D RoPE over ``(t, h, w)`` -- see
    :class:`Joint3DSelfAttention` for the tradeoffs. Backward-compatible:
    leaving ``use_3d_rope`` at its default keeps the original behavior.
    """

    def __init__(
        self,
        in_dim: int,
        num_heads: int,
        max_t: int,
        rope_max_harmonics: List[int] = (1, 1),
        mlp_ratio: float = 3.0,
        dropout: float = 0.0,
        causal_time: bool = False,
        use_3d_rope: bool = False,
        len_h: int = 1,
        len_w: int = 1,
    ):
        super().__init__()
        self.attn = Joint3DSelfAttention(
            dim=in_dim,
            num_heads=num_heads,
            max_t=max_t,
            rope_max_harmonics=rope_max_harmonics,
            dropout=dropout,
            causal_time=causal_time,
            use_3d_rope=use_3d_rope,
            len_h=len_h,
            len_w=len_w,
        )
        self.ffn = FeedForwardNetwork(in_dim, int(in_dim * mlp_ratio), dropout)

    def forward(self, x: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        x = self.attn(x, coords)
        x = self.ffn(x)
        return x


# =============================================================================
# Patch / temporal resampling
# =============================================================================


class PatchMerging(nn.Sequential):
    """2x2 spatial downsample: (B, 2H, 2W, C) -> (B, H, W, 2C)"""

    def __init__(self, dim: int):
        super().__init__(
            Rearrange("B (H P1) (W P2) C -> B H W (P1 P2 C)", P1=2, P2=2),
            nn.Linear(4 * dim, 2 * dim, bias=False),
        )


class PatchExpanding(nn.Module):
    """2x2 spatial upsample: (B, H, W, C) -> (B, 2H, 2W, C//2)"""

    def __init__(self, dim: int):
        super().__init__()
        self.linear = nn.Linear(dim, dim * 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = einops.rearrange(x, "B H W (P1 P2 C) -> B (H P1) (W P2) C", P1=2, P2=2)
        return x


class TemporalMerging(nn.Module):
    """Causal 2x temporal downsample (mimics WanVAE downsample3d).

    Uses Conv1d(kernel=3, stride=2) with left-only causal padding.
    Each output only depends on the current and past frames.
    Input:  (B, T, H, W, C)
    Output: (B, ceil(T/2), H, W, C)
    """

    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, kernel_size=3, stride=2, padding=0, bias=False)
        self.causal_pad = 2  # kernel_size - 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, H, W, C = x.shape
        x = einops.rearrange(x, "B T H W C -> (B H W) C T")
        x = F.pad(x, (self.causal_pad, 0))
        x = self.conv(x)
        x = einops.rearrange(x, "(B H W) C T -> B T H W C", B=B, H=H, W=W)
        return x


class TemporalMergingFirstFrameSpecial(nn.Module):
    """First-frame-special causal 2x temporal downsample (WAN-style).

    Mirrors :class:`WanVAE2P1`'s singleton-first-frame behavior at the
    transformer-VAE merge stage: the first input slot becomes ``z[0]``
    via a per-frame ``Linear`` (no temporal mixing), and the remaining
    ``T_in - 1`` frames are merged in *disjoint* 2-frame chunks via a
    stride-2 ``Conv1d`` with ``kernel=2`` and **no** causal padding --
    each pair stays inside its own chunk:

        ``in[0:1]``  ->  z[0]                   per-frame Linear
        ``in[2i-1:2i+1]``  ->  z[i>=1]          stride-2 kernel-2 conv

    Pair this with :class:`TemporalExpandingWanStyle` in the decoder so
    ``T_decoded = 2*T_z - 1`` exactly matches ``T_in = 1 + 2*(T_z - 1)``.
    No front-crop needed, no boundary information loss.

    Compared to plain :class:`TemporalMerging` (kernel=3, stride=2,
    causal-pad-2):
    * z[0] genuinely depends on in[0] alone (not "in[0] + 2 pad"
      activations); the Linear is the cleanest analog of WAN's CausalConv
      with a fully zero-cached history.
    * z[i>=1] sees both frames of its 2-frame chunk (vs. only the first
      frame + 2 history frames in plain merging) -- so each latent has
      *full* info on the chunk it represents, eliminating the
      "z[i] never sees in[2i+1]" information loss.

    Requires ``T_in`` to be ODD (i.e. ``T_in - 1`` divisible by 2).

    Input:  (B, T_in, H, W, C),  T_in = 2*T_z - 1
    Output: (B, T_z,  H, W, C)
    """

    def __init__(self, dim: int):
        super().__init__()
        # First-frame projection: like WAN's CausalConv3d on a length-1
        # input with zero-padded kernel history -- a linear function of
        # the single frame's features.
        self.first_frame_proj = nn.Linear(dim, dim, bias=False)
        # Disjoint pair merger: kernel=2, stride=2, no padding. Each
        # output index i corresponds to ``rest[2i, 2i+1]`` exclusively
        # (no kernel overlap across chunks, no causal-pad zero leakage).
        self.conv = nn.Conv1d(dim, dim, kernel_size=2, stride=2, padding=0, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T_in, H, W, C = x.shape
        if T_in == 1:
            return self.first_frame_proj(x)
        first = self.first_frame_proj(x[:, :1])  # (B, 1, H, W, C)
        rest = x[:, 1:]
        if rest.shape[1] % 2 != 0:
            raise ValueError(
                f"TemporalMergingFirstFrameSpecial requires (T_in - 1) "
                f"divisible by 2 at this level, got T_in={T_in}. With "
                f"``temporal_first_frame_special=True`` and N levels of "
                f"temporal merge, the input T must satisfy "
                f"T = 1 + (2 ** N) * (T_z - 1) -- e.g. T=9 for N=1, T_z=5."
            )
        rest = einops.rearrange(rest, "B T H W C -> (B H W) C T")
        rest = self.conv(rest)  # (N, C, (T_in - 1) // 2)
        rest = einops.rearrange(rest, "(B H W) C T -> B T H W C", B=B, H=H, W=W)
        return torch.cat([first, rest], dim=1)


class TemporalExpanding(nn.Module):
    """Causal 2x temporal upsample (mimics WanVAE upsample3d).

    Uses a causal Conv1d that doubles channels, then interleaves the two
    channel halves along time to achieve 2x upsampling.  No future leakage
    because the underlying conv is causal (left-pad only, stride=1).
    Input:  (B, T//2, H, W, C)
    Output: (B, T, H, W, C)
    """

    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim * 2, kernel_size=3, stride=1, padding=0, bias=False)
        self.causal_pad = 2  # kernel_size - 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T_in, H, W, C = x.shape
        x = einops.rearrange(x, "B T H W C -> (B H W) C T")
        x = F.pad(x, (self.causal_pad, 0))
        x = self.conv(x)  # (N, 2C, T_in)
        x = einops.rearrange(
            x, "N (two C) T -> N C (T two)", two=2,
        )  # interleave: [t0_a, t0_b, t1_a, t1_b, ...] → 2*T_in frames
        x = einops.rearrange(x, "(B H W) C T -> B T H W C", B=B, H=H, W=W)
        return x


class TemporalExpandingWanStyle(nn.Module):
    """Causal 2x temporal upsample with WAN-style first-frame handling.

    Mirrors :class:`WanVAE2P1.decode`: the first input frame produces a
    single output frame (just the "a" half of the causal conv's doubled
    output -- i.e. no temporal duplication), while every subsequent input
    frame produces 2 output frames. Composed twice (matching the WAN
    encoder's 4x temporal compression with its special first-frame chunk),
    a latent of length ``T_z`` decodes to ``4*T_z - 3`` output frames -- the
    same length as the original input when ``T_in = 1 + 4*(T_z - 1)``, so
    no leading-frame crop is needed.

    Input:  (B, T_in, H, W, C)
    Output: (B, 2*T_in - 1, H, W, C)  (or 1 frame if T_in == 1)
    """

    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim * 2, kernel_size=3, stride=1, padding=0, bias=False)
        self.causal_pad = 2  # kernel_size - 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T_in, H, W, C = x.shape
        h = einops.rearrange(x, "B T H W C -> (B H W) C T")
        h = F.pad(h, (self.causal_pad, 0))
        h = self.conv(h)  # (N, 2C, T_in)
        h = einops.rearrange(
            h, "N (two C) T -> N C (T two)", two=2,
        )  # interleave: [t0_a, t0_b, t1_a, t1_b, ...] → 2*T_in frames along T
        h = einops.rearrange(h, "(B H W) C T -> B T H W C", B=B, H=H, W=W)
        # Keep t0_a as the single "first frame" output; drop t0_b. Keep both
        # halves for every other input frame.
        if T_in == 1:
            return h[:, :1]
        return torch.cat([h[:, :1], h[:, 2:]], dim=1)


class Tokenizer(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        patch_size: List[int],
    ):
        patch_size = _pair(patch_size)
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=patch_size,
                stride=patch_size,
                padding=0,
                bias=False,
            ),
            Rearrange("B C H W -> B H W C"),
        )


class Detokenizer(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        patch_size: List[int],
    ):
        patch_size = _pair(patch_size)
        super().__init__(
            RMSNorm(in_channels),
            nn.Linear(
                in_channels,
                out_channels * patch_size[0] * patch_size[1],
                bias=False,
            ),
            Rearrange(
                "B H W (P1 P2 C) -> B C (H P1) (W P2)",
                P1=patch_size[0],
                P2=patch_size[1],
            ),
        )


# =============================================================================
# Positional embedding
# =============================================================================


class LearnablePositionalEmbedding(nn.Module):
    def __init__(self, out_dim: int, resolution: List[int]):
        super().__init__()
        self.embedding = nn.Parameter(torch.zeros(1, *resolution, out_dim))
        nn.init.trunc_normal_(self.embedding, std=0.02)

    def forward(self, coords=None):
        return self.embedding

    def extra_repr(self):
        return f"resolution={tuple(self.embedding.shape[1:3])}, out_dim={self.embedding.shape[3]}"


class ZeroPositionalEmbedding(nn.Module):
    """No additive PE; attention blocks still apply coordinate-based RoPE."""

    def __init__(self, out_dim: int):
        super().__init__()
        self.out_dim = out_dim

    def forward(self, coords: torch.Tensor):
        return coords.new_zeros(
            coords.shape[0], coords.shape[2], coords.shape[3], self.out_dim
        )

    def extra_repr(self):
        return f"out_dim={self.out_dim}"


class AbsolutePositionalEmbedding(nn.Module):
    def __init__(self, method: str, out_dim: int, resolution: List[int]):
        super().__init__()
        self.resolution = resolution
        if method == "spherical_harmonics":
            self.embedding = encoding.SphericalHarmonics(levels=5)
            emb_dim = self.embedding.extra_ch
        elif method == "fourier_features":
            self.embedding = encoding.FourierFeatures(self.resolution)
            emb_dim = self.embedding.extra_ch
        elif method == "polar_coordinates":
            self.embedding = nn.Identity()
            emb_dim = 2
        else:
            raise ValueError(f"Unknown positional embedding method: {method}")
        self.linear = nn.Linear(emb_dim, out_dim, bias=False)

    def forward(self, coords: torch.Tensor):
        h = self.embedding(coords)
        h = einops.rearrange(h, "B C H W -> B H W C")
        h = self.linear(h)
        return h


class MappingFeedForwardNetwork(nn.Module):
    def __init__(self, dim, mid_dim, dropout=0.0):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.gegelu = GEGLU(dim, mid_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(mid_dim, dim, bias=False).apply(ops.zero_out)

    def residual(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.gegelu(x)
        x = self.dropout(x)
        x = self.linear(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.residual(x)


class MappingNetwork(nn.Sequential):
    def __init__(self, dim, mid_dim, depth=1, dropout=0.0):
        super().__init__(
            RMSNorm(dim),
            *[MappingFeedForwardNetwork(dim, mid_dim, dropout) for _ in range(depth)],
            RMSNorm(dim),
        )


# =============================================================================
# Encoder
# =============================================================================


class Encoder(nn.Module):
    def __init__(
        self,
        resolution: List[int],
        in_channels: int,
        z_dim: int,
        base_channels: int = 128,
        patch_size: List[int] = (1, 4),
        window_size: List[int] = (3, 9),
        depths: List[int] = (3, 3, 3, 3),
        num_heads: List[int] = (2, 4, 8, 16),
        dilation: List[int] = (1, 1, 1, 1),
        temporal_downsample: List[bool] = (True, False, False),
        mlp_ratio: float = 3.0,
        dropout: float = 0.0,
        mapping_depth: int = 2,
        positional_embedding: str = "learnable_embedding",
        # If True, every level with ``temporal_downsample[i]=True`` uses
        # :class:`TemporalMergingFirstFrameSpecial` instead of the plain
        # :class:`TemporalMerging`. Yields clean WAN-style chunk
        # boundaries (z[0]=enc(in[0]), z[i>=1]=enc(disjoint 2-frame
        # chunk)) at the cost of requiring T_in - 1 to be divisible
        # by ``2 ** sum(temporal_downsample)``.
        temporal_first_frame_special: bool = False,
        # ---- Bottleneck options (mirror Decoder) ----
        # If True, replace the encoder bottleneck's factorized (global spatial
        # + temporal) attention with a single joint 3D self-attention over
        # (T*H*W) tokens. Down-levels are unchanged. Default False keeps the
        # original factorized bottleneck (bit-identical to existing configs).
        bottleneck_3d: bool = False,
        bottleneck_3d_max_t: int = 32,
        bottleneck_3d_causal_time: bool = False,
        bottleneck_3d_rope: bool = False,
        # If False, the local (neighborhood) attention does NOT wrap the W
        # axis (clamps at the edges instead). Default True preserves the
        # range-image azimuth-periodic behavior. Set False for a Cartesian
        # BEV grid where the left/right W edges are unrelated.
        circular_padding: bool = True,
        # Temporal-mixing mechanism for the down/mid temporal blocks:
        # "attention" (:class:`CausalTemporalAttention`, default) or "conv"
        # (WAN-style :class:`CausalTemporalConv`, kernel ``temporal_conv_kernel``).
        temporal_mixer: Literal["attention", "conv"] = "attention",
        temporal_conv_kernel: int = 3,
        activation_checkpointing: bool = False,
    ):
        super().__init__()
        self.resolution = _pair(resolution)
        self.patch_size = _pair(patch_size)
        self.depths = depths
        self.temporal_downsample = temporal_downsample
        self.bottleneck_3d = bottleneck_3d
        self.temporal_mixer = temporal_mixer
        self.activation_checkpointing = bool(activation_checkpointing)
        self._temporal_cls = (
            functools.partial(TemporalConvBlock, kernel_size=temporal_conv_kernel)
            if temporal_mixer == "conv"
            else TemporalBlock
        )

        token_size = torch.tensor(self.resolution) // torch.tensor(self.patch_size)
        max_harmonics = (token_size / 2).int()

        self.tokenizer = Tokenizer(in_channels, base_channels, patch_size)

        if positional_embedding == "rope_only":
            self.spatial_pe = ZeroPositionalEmbedding(out_dim=base_channels)
        elif positional_embedding == "learnable_embedding":
            self.spatial_pe = LearnablePositionalEmbedding(
                out_dim=base_channels,
                resolution=token_size.tolist(),
            )
        else:
            self.spatial_pe = nn.Sequential(
                AbsolutePositionalEmbedding(
                    method=positional_embedding,
                    out_dim=base_channels,
                    resolution=token_size.tolist(),
                ),
                MappingNetwork(
                    base_channels,
                    int(base_channels * mlp_ratio),
                    depth=mapping_depth,
                ),
            )

        # Down levels: spatial blocks + temporal blocks + merging
        # Skip temporal attention at level 0 (highest resolution) for efficiency.
        self.down_levels = nn.ModuleDict()
        for i, num_blocks in enumerate(depths[:-1]):
            dim_i = base_channels << i
            spatial_blocks = nn.ModuleList()
            for j in range(num_blocks):
                spatial_blocks.append(
                    SpatialBlock(
                        in_dim=dim_i,
                        num_heads=num_heads[i],
                        attn_type="local",
                        kernel_size=window_size,
                        dilation=1 if j % 2 == 0 else dilation[i],
                        mlp_ratio=mlp_ratio,
                        dropout=dropout,
                        rope_max_harmonics=(max_harmonics >> i).clamp(min=1),
                        circular=circular_padding,
                    )
                )
            self.down_levels[f"spatial_{i}"] = spatial_blocks
            if i > 0:
                temporal_blocks = nn.ModuleList()
                for j in range(num_blocks):
                    temporal_blocks.append(
                        self._temporal_cls(
                            in_dim=dim_i,
                            num_heads=num_heads[i],
                            mlp_ratio=mlp_ratio,
                            dropout=dropout,
                        )
                    )
                self.down_levels[f"temporal_{i}"] = temporal_blocks
            self.down_levels[f"merge_{i}"] = PatchMerging(dim_i)
            if temporal_downsample[i]:
                merge_cls = (
                    TemporalMergingFirstFrameSpecial
                    if temporal_first_frame_special
                    else TemporalMerging
                )
                self.down_levels[f"temporal_merge_{i}"] = merge_cls(dim_i * 2)

        # Bottleneck: either factorized (global spatial + temporal), or a
        # single joint 3D self-attention over (T*H*W) tokens.
        i = len(depths) - 1
        bottleneck_dim = base_channels << i
        bottleneck_size = (token_size >> i).tolist()
        if bottleneck_3d:
            self.mid_3d = nn.ModuleList()
            for _ in range(depths[-1]):
                self.mid_3d.append(
                    Bottleneck3DBlock(
                        in_dim=bottleneck_dim,
                        num_heads=num_heads[-1],
                        max_t=bottleneck_3d_max_t,
                        rope_max_harmonics=(max_harmonics >> i).clamp(min=1),
                        mlp_ratio=mlp_ratio,
                        dropout=dropout,
                        causal_time=bottleneck_3d_causal_time,
                        use_3d_rope=bottleneck_3d_rope,
                        len_h=bottleneck_size[0],
                        len_w=bottleneck_size[1],
                    )
                )
        else:
            self.mid_spatial = nn.ModuleList()
            self.mid_temporal = nn.ModuleList()
            for j in range(depths[-1]):
                self.mid_spatial.append(
                    SpatialBlock(
                        in_dim=bottleneck_dim,
                        num_heads=num_heads[-1],
                        attn_type="global",
                        mlp_ratio=mlp_ratio,
                        dropout=dropout,
                        rope_max_harmonics=(max_harmonics >> i).clamp(min=1),
                    )
                )
                self.mid_temporal.append(
                    self._temporal_cls(
                        in_dim=bottleneck_dim,
                        num_heads=num_heads[-1],
                        mlp_ratio=mlp_ratio,
                        dropout=dropout,
                    )
                )

        # Project to z_dim * 2 (mean + logvar)
        self.head = nn.Sequential(
            RMSNorm(bottleneck_dim),
            nn.Linear(bottleneck_dim, z_dim * 2, bias=False),
            Rearrange("B H W C -> B C H W"),
        )

    def forward(self, x: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T, H, W) video tensor
            coords: (1, 2, H, W) polar coordinates
        Returns:
            (B, z_dim*2, T', H_z, W_z) latent parameters
        """
        B, C, T, H, W = x.shape
        coords = coords.to(device=x.device, dtype=x.dtype)
        if coords.shape[0] not in (1, B):
            raise ValueError(
                f"Coordinate batch {coords.shape[0]} must be 1 or input batch {B}"
            )
        c = F.avg_pool2d(coords, kernel_size=self.patch_size, stride=self.patch_size)

        # Tokenize per-frame
        x = einops.rearrange(x, "B C T H W -> (B T) C H W")
        c_frames = (
            einops.repeat(c, "B C H W -> (B T) C H W", T=T)
            if c.shape[0] == B
            else c
        )
        h = self.tokenizer(x) + self.spatial_pe(c_frames)  # (BT, H', W', C0)
        h = einops.rearrange(h, "(B T) H W C -> B T H W C", B=B, T=T)

        T_cur = T
        for i in range(len(self.depths) - 1):
            _, _, Hc, Wc, _ = h.shape
            has_temporal = f"temporal_{i}" in self.down_levels
            t_blocks = self.down_levels[f"temporal_{i}"] if has_temporal else [None] * len(self.down_levels[f"spatial_{i}"])
            for s_block, t_block in zip(self.down_levels[f"spatial_{i}"], t_blocks):
                h = einops.rearrange(h, "B T H W C -> (B T) H W C")
                c_frames = (
                    einops.repeat(c, "B C H W -> (B T) C H W", T=T_cur)
                    if c.shape[0] == B
                    else c
                )
                h = _run_block(
                    s_block,
                    h,
                    c_frames,
                    activation_checkpointing=self.activation_checkpointing,
                )
                if t_block is not None:
                    h = einops.rearrange(
                        h, "(B T) H W C -> (B H W) T C", B=B, T=T_cur
                    )
                    h = _run_block(
                        t_block,
                        h,
                        activation_checkpointing=self.activation_checkpointing,
                    )
                    h = einops.rearrange(
                        h, "(B H W) T C -> B T H W C", B=B, H=Hc, W=Wc
                    )
                else:
                    h = einops.rearrange(
                        h, "(B T) H W C -> B T H W C", B=B, T=T_cur
                    )

            # Spatial downsample per-frame
            h = einops.rearrange(h, "B T H W C -> (B T) H W C")
            h = self.down_levels[f"merge_{i}"](h)
            h = einops.rearrange(h, "(B T) H W C -> B T H W C", B=B, T=T_cur)
            c = F.avg_pool2d(c, kernel_size=2, stride=2)

            # Temporal downsample
            if self.temporal_downsample[i]:
                h = self.down_levels[f"temporal_merge_{i}"](h)
                T_cur = h.shape[1]

        # Bottleneck: joint 3D attention, or factorized spatial-then-temporal.
        _, _, Hc, Wc, _ = h.shape
        if self.bottleneck_3d:
            for block in self.mid_3d:
                h = _run_block(
                    block,
                    h,
                    c,
                    activation_checkpointing=self.activation_checkpointing,
                )
        else:
            for s_block, t_block in zip(self.mid_spatial, self.mid_temporal):
                h = einops.rearrange(h, "B T H W C -> (B T) H W C")
                c_frames = (
                    einops.repeat(c, "B C H W -> (B T) C H W", T=T_cur)
                    if c.shape[0] == B
                    else c
                )
                h = _run_block(
                    s_block,
                    h,
                    c_frames,
                    activation_checkpointing=self.activation_checkpointing,
                )
                h = einops.rearrange(h, "(B T) H W C -> (B H W) T C", B=B, T=T_cur)
                h = _run_block(
                    t_block,
                    h,
                    activation_checkpointing=self.activation_checkpointing,
                )
                h = einops.rearrange(h, "(B H W) T C -> B T H W C", B=B, H=Hc, W=Wc)

        # Head per-frame
        h = einops.rearrange(h, "B T H W C -> (B T) H W C")
        h = self.head(h)  # (BT, z_dim*2, H_z, W_z)
        h = einops.rearrange(h, "(B T) C H W -> B C T H W", B=B, T=T_cur)
        return h

    @staticmethod
    def _trim_temporal_kv(
        kv: tuple[torch.Tensor, torch.Tensor],
        max_cache_frames: int | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if max_cache_frames is None:
            return kv
        if max_cache_frames <= 0:
            raise ValueError("max_cache_frames must be positive or None")
        k, v = kv
        return (
            k[:, :, -max_cache_frames:].contiguous(),
            v[:, :, -max_cache_frames:].contiguous(),
        )

    def forward_stream(
        self,
        x_new: torch.Tensor,
        coords: torch.Tensor,
        temporal_kv_cache: dict[
            str, tuple[torch.Tensor, torch.Tensor]
        ] | None = None,
        max_cache_frames: int | None = None,
    ) -> tuple[
        torch.Tensor,
        dict[str, tuple[torch.Tensor, torch.Tensor]],
    ]:
        """Encode only newly appended frames with causal temporal KV caches."""
        if self.bottleneck_3d:
            raise NotImplementedError(
                "streaming encode does not support a joint 3D bottleneck"
            )
        if any(self.temporal_downsample):
            raise NotImplementedError(
                "streaming encode currently requires 1x temporal compression"
            )
        if self.temporal_mixer != "attention":
            raise NotImplementedError(
                "streaming encode currently requires temporal attention"
            )

        B, _, T_new, _, _ = x_new.shape
        coords = coords.to(device=x_new.device, dtype=x_new.dtype)
        c = F.avg_pool2d(
            coords, kernel_size=self.patch_size, stride=self.patch_size
        )
        x_new = einops.rearrange(
            x_new, "B C T H W -> (B T) C H W"
        )
        h = self.tokenizer(x_new) + self.spatial_pe(c)
        h = einops.rearrange(
            h, "(B T) H W C -> B T H W C", B=B, T=T_new
        )
        old_cache = temporal_kv_cache or {}
        new_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

        for i in range(len(self.depths) - 1):
            _, _, Hc, Wc, _ = h.shape
            has_temporal = f"temporal_{i}" in self.down_levels
            t_blocks = (
                self.down_levels[f"temporal_{i}"]
                if has_temporal
                else [None] * len(self.down_levels[f"spatial_{i}"])
            )
            for j, (s_block, t_block) in enumerate(
                zip(self.down_levels[f"spatial_{i}"], t_blocks)
            ):
                h = einops.rearrange(
                    h, "B T H W C -> (B T) H W C"
                )
                h = s_block(h, c)
                if t_block is not None:
                    h = einops.rearrange(
                        h,
                        "(B T) H W C -> (B H W) T C",
                        B=B,
                        T=T_new,
                    )
                    cache_key = f"down_{i}_{j}"
                    h, kv = t_block.forward_stream(
                        h, old_cache.get(cache_key)
                    )
                    new_cache[cache_key] = self._trim_temporal_kv(
                        kv, max_cache_frames
                    )
                    h = einops.rearrange(
                        h,
                        "(B H W) T C -> B T H W C",
                        B=B,
                        H=Hc,
                        W=Wc,
                    )
                else:
                    h = einops.rearrange(
                        h,
                        "(B T) H W C -> B T H W C",
                        B=B,
                        T=T_new,
                    )

            h = einops.rearrange(
                h, "B T H W C -> (B T) H W C"
            )
            h = self.down_levels[f"merge_{i}"](h)
            h = einops.rearrange(
                h,
                "(B T) H W C -> B T H W C",
                B=B,
                T=T_new,
            )
            c = F.avg_pool2d(c, kernel_size=2, stride=2)

        _, _, Hc, Wc, _ = h.shape
        for j, (s_block, t_block) in enumerate(
            zip(self.mid_spatial, self.mid_temporal)
        ):
            h = einops.rearrange(
                h, "B T H W C -> (B T) H W C"
            )
            h = s_block(h, c)
            h = einops.rearrange(
                h,
                "(B T) H W C -> (B H W) T C",
                B=B,
                T=T_new,
            )
            cache_key = f"mid_{j}"
            h, kv = t_block.forward_stream(h, old_cache.get(cache_key))
            new_cache[cache_key] = self._trim_temporal_kv(
                kv, max_cache_frames
            )
            h = einops.rearrange(
                h,
                "(B H W) T C -> B T H W C",
                B=B,
                H=Hc,
                W=Wc,
            )

        h = einops.rearrange(h, "B T H W C -> (B T) H W C")
        h = self.head(h)
        h = einops.rearrange(
            h, "(B T) C H W -> B C T H W", B=B, T=T_new
        )
        return h, new_cache


# =============================================================================
# Decoder
# =============================================================================


class Decoder(nn.Module):
    def __init__(
        self,
        resolution: List[int],
        out_channels: int,
        z_dim: int,
        base_channels: int = 128,
        patch_size: List[int] = (1, 4),
        window_size: List[int] = (3, 9),
        depths: List[int] = (3, 3, 3, 3),
        num_heads: List[int] = (2, 4, 8, 16),
        dilation: List[int] = (1, 1, 1, 1),
        temporal_downsample: List[bool] = (True, False, False),
        temporal_upsample: List[bool] | None = None,
        mlp_ratio: float = 3.0,
        dropout: float = 0.0,
        mapping_depth: int = 2,
        positional_embedding: str = "learnable_embedding",
        stem_patchify: bool = False,
        # ---- Bottleneck options ----
        # If True, replace the bottleneck's factorized (temporal + global
        # spatial) attention with a single joint 3D self-attention over
        # (T*H*W) tokens. The up-levels are unchanged (still skip temporal
        # attn at the highest resolution as before).
        bottleneck_3d: bool = False,
        # Max T accepted by the bottleneck-3D's learnable temporal PE. Only
        # used when bottleneck_3d=True. Set this >= the largest T_z (encoder
        # latent frames) you'll ever feed the decoder.
        bottleneck_3d_max_t: int = 32,
        # If True, the joint 3D attention is causal in time (token (t, h, w)
        # only attends to (t', h', w') with t' <= t). The causal mask is
        # materialized as a (T*H*W, T*H*W) bool tensor, so memory scales as
        # O((T*H*W)^2). Bidirectional is the default for non-streaming use.
        bottleneck_3d_causal_time: bool = False,
        # If True (and bottleneck_3d=True), encode position in the joint 3D
        # attention with a parameter-free 3D RoPE (`VideoRopePosition3DEmb`)
        # over (t, h, w) instead of the default (learnable temporal PE + 2D
        # spatial AxialRoPE). Recommended for length-extrapolation: the
        # learnable temporal PE has uninitialized slots past the largest T
        # seen in training (= noise at inference); 3D RoPE has none. The
        # spatial axes of the 3D RoPE replace AxialRoPE -- the bottleneck
        # operates on a regular (H_z, W_z) grid so Cartesian RoPE is fine
        # and decouples positional encoding from polar `coords`.
        # Orthogonal to `bottleneck_3d_causal_time` -- both can be combined.
        bottleneck_3d_rope: bool = False,
        # If True, replace each TemporalExpanding with TemporalExpandingWanStyle,
        # which mirrors the WAN decoder's first-frame-special behavior:
        # T_in -> 2*T_in - 1 per level. Composed twice this maps T_z latents
        # to 4*T_z - 3 frames, so when paired with the WAN encoder there's no
        # over-production and no leading-frame crop needed downstream.
        temporal_expand_wan_style: bool = False,
        # ---- Detokenizer (output) options ----
        # Patch size for the final unpatchify (pixel-shuffle) layer. Defaults
        # to `patch_size` (symmetric with the stem, the original behavior).
        # Setting this larger than `patch_size` lets the decoder run the top
        # (highest-resolution) up-levels at a coarser feature grid and rely
        # on the final per-pixel Linear + rearrange to upsample to the output
        # resolution -- substantially cheaper since the top level dominates
        # the activation memory. The user is responsible for making sure
        # `bottleneck_size = resolution // out_patch_size >> n_down` matches
        # the encoder's latent spatial shape (i.e. reduce `depths` by
        # `log2(out_patch_size / patch_size)` to compensate).
        out_patch_size: List[int] | None = None,
        # ---- HDmap injection (V1 / V3) ----
        # When True, build a sibling ``stem_hdmap`` projection of the same
        # shape as ``stem`` but zero-init. At forward time, if ``hdmap_z``
        # is supplied, the decoder computes
        #   h = stem(z) + stem_hdmap(hdmap_z) + spatial_pe(...)
        # which is mathematically equivalent to channel-concating
        # [z; hdmap_z] and using a doubled-input Linear -- but keeps the
        # original ``stem`` weights loadable strictly from the warm-start
        # ckpt and makes the iter-0 forward bit-identical (zero-init
        # ``stem_hdmap`` contributes nothing). See
        # ``..._mixclean005_hdmap_v1*`` configs for the design notes.
        hdmap_inject_stem: bool = False,
        # When not None, build per-up-level injection convs that add
        # raw-hdmap features (per-frame ``avg_pool2d`` to the matching
        # spatial grid, then 1x1 conv) at the entry of each enabled
        # up-level (post-``expand_i``, pre-blocks). List length must
        # equal ``n_down = len(depths) - 1``; indexed by up-level i
        # (i=0 is the top up-level). Each enabled entry adds a zero-init
        # ``Conv2d(hdmap_in_channels, dim_i, kernel_size=1, bias=False)``.
        # See ``..._mixclean005_hdmap_v1v3`` for the multi-injection-point
        # design.
        hdmap_inject_raw_at_up_levels: list[bool] | None = None,
        # Channel count of the *raw* hdmap tensor passed to forward
        # (the per-up-level V3 injections need this to size the 1x1
        # conv input). Defaults to 3 (matches ``process_input="repeat"``
        # which produces a 3-channel hdmap). Ignored when
        # ``hdmap_inject_raw_at_up_levels`` is None or all-False.
        hdmap_in_channels: int = 3,
        # If False, the local (neighborhood) attention does NOT wrap the W
        # axis (clamps at the edges). Default True keeps the range-image
        # azimuth-periodic behavior. Set False for a Cartesian BEV grid.
        circular_padding: bool = True,
        # Temporal-mixing mechanism for the mid/up temporal blocks:
        # "attention" (:class:`CausalTemporalAttention`, default) or "conv"
        # (WAN-style :class:`CausalTemporalConv`, kernel ``temporal_conv_kernel``).
        temporal_mixer: Literal["attention", "conv"] = "attention",
        temporal_conv_kernel: int = 3,
        activation_checkpointing: bool = False,
    ):
        super().__init__()
        self.resolution = _pair(resolution)
        self.patch_size = _pair(patch_size)
        self.out_patch_size = (
            _pair(out_patch_size) if out_patch_size is not None else self.patch_size
        )
        self.depths = depths
        self.temporal_upsample = temporal_upsample if temporal_upsample is not None else temporal_downsample
        self.stem_patchify = stem_patchify
        self.bottleneck_3d = bottleneck_3d
        self.activation_checkpointing = bool(activation_checkpointing)
        self._temporal_cls = (
            functools.partial(TemporalConvBlock, kernel_size=temporal_conv_kernel)
            if temporal_mixer == "conv"
            else TemporalBlock
        )
        self.temporal_expand_wan_style = temporal_expand_wan_style
        self.hdmap_inject_stem = bool(hdmap_inject_stem)
        if hdmap_inject_raw_at_up_levels is not None:
            if len(hdmap_inject_raw_at_up_levels) != len(depths) - 1:
                raise ValueError(
                    f"hdmap_inject_raw_at_up_levels must have length "
                    f"n_down = len(depths) - 1 = {len(depths) - 1}, "
                    f"got length {len(hdmap_inject_raw_at_up_levels)}."
                )
            self.hdmap_inject_raw_at_up_levels = list(
                bool(b) for b in hdmap_inject_raw_at_up_levels
            )
        else:
            self.hdmap_inject_raw_at_up_levels = None
        self.hdmap_in_channels = int(hdmap_in_channels)
        temporal_expand_cls = (
            TemporalExpandingWanStyle if temporal_expand_wan_style else TemporalExpanding
        )

        # `token_size` is the feature grid at the top up-level, i.e. *before*
        # the final detokenizer's unpatchify -- this is what RoPE harmonics
        # and the coordinate pyramid index. With `out_patch_size > patch_size`
        # the top up-level sits at a coarser grid than the output resolution.
        token_size = torch.tensor(self.resolution) // torch.tensor(self.out_patch_size)
        max_harmonics = (token_size / 2).int()
        n_down = len(depths) - 1
        bottleneck_dim = base_channels << n_down
        bottleneck_size = (token_size >> n_down).tolist()

        # Project from z_dim to bottleneck channels.
        # By default the stem is a per-pixel channel projection (used in the
        # full TransformerVAE, where the Encoder's Tokenizer has already
        # patchified the input by `patch_size`). When `stem_patchify=True`,
        # we instead apply a Conv2d with stride=patch_size, so the decoder
        # can be paired with an external encoder (e.g. the WAN encoder) whose
        # latent spatial shape is patch_size larger than the bottleneck.
        if stem_patchify and tuple(self.patch_size) != (1, 1):
            self.stem = nn.Sequential(
                nn.Conv2d(
                    z_dim,
                    bottleneck_dim,
                    kernel_size=self.patch_size,
                    stride=self.patch_size,
                    bias=False,
                ),
                Rearrange("B C H W -> B H W C"),
            )
        else:
            self.stem = nn.Sequential(
                Rearrange("B C H W -> B H W C"),
                nn.Linear(z_dim, bottleneck_dim, bias=False),
            )
        if self.hdmap_inject_stem:
            # Sibling stem for hdmap_z, identical shape to ``self.stem`` but
            # zero-init. ``stem(z) + stem_hdmap(hdmap_z)`` is mathematically
            # equivalent to channel-concating [z; hdmap_z] and using a
            # doubled-input projection (see V1 docstring); decomposing it
            # this way keeps the original ``stem`` weights loadable as-is
            # from the warm-start ckpt and makes iter-0 forward bit-
            # identical (zero-init contributes nothing).
            if stem_patchify and tuple(self.patch_size) != (1, 1):
                hdmap_stem_proj = nn.Conv2d(
                    z_dim,
                    bottleneck_dim,
                    kernel_size=self.patch_size,
                    stride=self.patch_size,
                    bias=False,
                )
                nn.init.zeros_(hdmap_stem_proj.weight)
                self.stem_hdmap = nn.Sequential(
                    hdmap_stem_proj,
                    Rearrange("B C H W -> B H W C"),
                )
            else:
                hdmap_stem_lin = nn.Linear(z_dim, bottleneck_dim, bias=False)
                nn.init.zeros_(hdmap_stem_lin.weight)
                self.stem_hdmap = nn.Sequential(
                    Rearrange("B C H W -> B H W C"),
                    hdmap_stem_lin,
                )

        # Positional embedding at bottleneck resolution
        if positional_embedding == "rope_only":
            self.spatial_pe = ZeroPositionalEmbedding(out_dim=bottleneck_dim)
        elif positional_embedding == "learnable_embedding":
            self.spatial_pe = LearnablePositionalEmbedding(
                out_dim=bottleneck_dim,
                resolution=bottleneck_size,
            )
        else:
            self.spatial_pe = nn.Sequential(
                AbsolutePositionalEmbedding(
                    method=positional_embedding,
                    out_dim=bottleneck_dim,
                    resolution=bottleneck_size,
                ),
                MappingNetwork(
                    bottleneck_dim,
                    int(bottleneck_dim * mlp_ratio),
                    depth=mapping_depth,
                ),
            )

        # Bottleneck: either factorized (temporal + global spatial), or a
        # single joint 3D self-attention over (T*H*W) tokens.
        if bottleneck_3d:
            self.mid_3d = nn.ModuleList()
            for _ in range(depths[-1]):
                self.mid_3d.append(
                    Bottleneck3DBlock(
                        in_dim=bottleneck_dim,
                        num_heads=num_heads[-1],
                        max_t=bottleneck_3d_max_t,
                        rope_max_harmonics=(max_harmonics >> n_down).clamp(min=1),
                        mlp_ratio=mlp_ratio,
                        dropout=dropout,
                        causal_time=bottleneck_3d_causal_time,
                        use_3d_rope=bottleneck_3d_rope,
                        # 3D RoPE caches per-axis frequencies up to (len_t,
                        # len_h, len_w). Spatial caps come from the
                        # bottleneck's own grid; the temporal cap is shared
                        # with the learnable-PE path via `bottleneck_3d_max_t`.
                        len_h=bottleneck_size[0],
                        len_w=bottleneck_size[1],
                    )
                )
        else:
            self.mid_temporal = nn.ModuleList()
            self.mid_spatial = nn.ModuleList()
            for _ in range(depths[-1]):
                self.mid_temporal.append(
                    self._temporal_cls(
                        in_dim=bottleneck_dim,
                        num_heads=num_heads[-1],
                        mlp_ratio=mlp_ratio,
                        dropout=dropout,
                    )
                )
                self.mid_spatial.append(
                    SpatialBlock(
                        in_dim=bottleneck_dim,
                        num_heads=num_heads[-1],
                        attn_type="global",
                        mlp_ratio=mlp_ratio,
                        dropout=dropout,
                        rope_max_harmonics=(max_harmonics >> n_down).clamp(min=1),
                    )
                )

        # Up levels: temporal expand + spatial expand + temporal→spatial blocks
        # Skip temporal attention at level 0 (highest resolution) for efficiency.
        self.up_levels = nn.ModuleDict()
        for i in reversed(range(n_down)):
            dim_i = base_channels << i
            dim_above = base_channels << (i + 1)

            if self.temporal_upsample[i]:
                self.up_levels[f"temporal_expand_{i}"] = temporal_expand_cls(dim_above)

            self.up_levels[f"expand_{i}"] = PatchExpanding(dim_above)

            if (
                self.hdmap_inject_raw_at_up_levels is not None
                and self.hdmap_inject_raw_at_up_levels[i]
            ):
                # V3 raw-hdmap injection at the entry of up-level i (post-
                # ``expand_{i}``, pre-blocks). Per-frame ``avg_pool2d`` of
                # the raw hdmap to (Hc, Wc) of this stage, then a zero-init
                # 1x1 conv projects 3 -> dim_i. Zero-init keeps the iter-0
                # forward bit-identical; the projection lives under
                # ``hdmap_inject_{i}`` to share the up_levels namespace
                # (no separate ModuleDict, fewer state_dict prefixes).
                inject_conv = nn.Conv2d(
                    self.hdmap_in_channels, dim_i, kernel_size=1, bias=False
                )
                nn.init.zeros_(inject_conv.weight)
                self.up_levels[f"hdmap_inject_{i}"] = inject_conv

            spatial_blocks = nn.ModuleList()
            for j in range(depths[i]):
                spatial_blocks.append(
                    SpatialBlock(
                        in_dim=dim_i,
                        num_heads=num_heads[i],
                        attn_type="local",
                        kernel_size=window_size,
                        dilation=1 if j % 2 == 0 else dilation[i],
                        mlp_ratio=mlp_ratio,
                        dropout=dropout,
                        rope_max_harmonics=(max_harmonics >> i).clamp(min=1),
                        circular=circular_padding,
                    )
                )
            self.up_levels[f"spatial_{i}"] = spatial_blocks
            # Temporal blocks are needed at every level that does a temporal
            # expand (to mix the newly-interpolated frames into a coherent
            # sequence), plus -- as a cheap-but-useful compute heuristic --
            # at every level deeper than the top one. The previous condition
            # was just `i > 0`, which silently skipped temporal attention at
            # the top level even when temporal_upsample[0] was True, leaving
            # the just-expanded frames un-mixed.
            if i > 0 or self.temporal_upsample[i]:
                temporal_blocks = nn.ModuleList()
                for j in range(depths[i]):
                    temporal_blocks.append(
                        self._temporal_cls(
                            in_dim=dim_i,
                            num_heads=num_heads[i],
                            mlp_ratio=mlp_ratio,
                            dropout=dropout,
                        )
                    )
                self.up_levels[f"temporal_{i}"] = temporal_blocks

        self.detokenizer = Detokenizer(base_channels, out_channels, self.out_patch_size)

    @staticmethod
    def _trim_temporal_kv(
        kv: tuple[torch.Tensor, torch.Tensor],
        max_cache_frames: int | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if max_cache_frames is None:
            return kv
        if max_cache_frames <= 0:
            raise ValueError("max_cache_frames must be positive or None")
        k, v = kv
        return (
            k[:, :, -max_cache_frames:].contiguous(),
            v[:, :, -max_cache_frames:].contiguous(),
        )

    def forward(
        self,
        z: torch.Tensor,
        coords: torch.Tensor,
        hdmap_z: torch.Tensor | None = None,
        hdmap_raw: torch.Tensor | None = None,
        temporal_kv_cache: dict[
            str, tuple[torch.Tensor, torch.Tensor]
        ] | None = None,
        return_temporal_kv_cache: bool = False,
        max_cache_frames: int | None = None,
    ) -> torch.Tensor | tuple[
        torch.Tensor, dict[str, tuple[torch.Tensor, torch.Tensor]]
    ]:
        """
        Args:
            z: (B, z_dim, T', H_z, W_z) latent
            coords: (1, 2, H, W) polar coordinates
            hdmap_z: optional (B, z_dim, T', H_z, W_z) hdmap latent (frozen-
                encoder output). Used by V1 stem-concat (added to ``stem(z)``
                via ``stem_hdmap``). Silently ignored when
                ``hdmap_inject_stem=False`` or this is None.
            hdmap_raw: optional (B, hdmap_in_channels, T_full, H_full, W_full)
                raw hdmap video at the same temporal length as the decoder's
                expected output. Used by V3 per-up-level injections (per-
                frame ``avg_pool2d`` to the matching stage grid + zero-init
                1x1 conv). Silently ignored when
                ``hdmap_inject_raw_at_up_levels=None`` or this is None.
        Returns:
            (B, C, T, H, W) reconstructed video
        """
        B = z.shape[0]
        T_cur = z.shape[2]
        n_down = len(self.depths) - 1
        if coords.shape[0] not in (1, B):
            raise ValueError(
                f"Coordinate batch {coords.shape[0]} must be 1 or latent batch {B}"
            )
        streaming = return_temporal_kv_cache
        if temporal_kv_cache is not None and not streaming:
            raise ValueError(
                "temporal_kv_cache requires return_temporal_kv_cache=True"
            )
        if max_cache_frames is not None and not streaming:
            raise ValueError(
                "max_cache_frames requires return_temporal_kv_cache=True"
            )
        if streaming:
            if self.bottleneck_3d:
                raise NotImplementedError(
                    "streaming decode does not support bottleneck_3d"
                )
            if any(self.temporal_upsample):
                raise NotImplementedError(
                    "streaming decode currently requires temporal_upsample=False "
                    "at every decoder level"
                )
            if not all(
                isinstance(block, TemporalBlock)
                for block in self.mid_temporal
            ):
                raise NotImplementedError(
                    "streaming decode currently supports temporal attention, "
                    "not temporal convolution"
                )
        old_cache = temporal_kv_cache or {}
        new_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

        # Build coordinate pyramid. The base sits at the *top up-level* token
        # grid (= resolution // out_patch_size), which is the highest-res grid
        # the attention blocks actually see -- the final detokenizer's
        # pixel-shuffle never gets coords.
        c_base = F.avg_pool2d(coords, kernel_size=self.out_patch_size, stride=self.out_patch_size)
        c_levels = [c_base]
        for i in range(n_down):
            c_levels.append(F.avg_pool2d(c_levels[-1], kernel_size=2, stride=2))

        # Stem per-frame + spatial PE (+ optional V1 hdmap-stem injection)
        z = einops.rearrange(z, "B C T H W -> (B T) C H W")
        c_frames = (
            einops.repeat(
                c_levels[n_down], "B C H W -> (B T) C H W", T=T_cur
            )
            if c_levels[n_down].shape[0] == B
            else c_levels[n_down]
        )
        h = self.stem(z) + self.spatial_pe(c_frames)  # (BT, H_z, W_z, C)
        if self.hdmap_inject_stem and hdmap_z is not None:
            # ``stem(z) + stem_hdmap(hdmap_z)`` is mathematically equivalent
            # to a doubled-input stem on ``[z; hdmap_z]``; zero-init at
            # iter 0 so this is a no-op until SGD learns to use it.
            hdmap_z_flat = einops.rearrange(hdmap_z, "B C T H W -> (B T) C H W")
            h = h + self.stem_hdmap(hdmap_z_flat)
        h = einops.rearrange(h, "(B T) H W C -> B T H W C", B=B, T=T_cur)

        # Bottleneck
        _, _, Hc, Wc, _ = h.shape
        if self.bottleneck_3d:
            # Joint 3D attention: single pass over (T*H*W) tokens.
            for block in self.mid_3d:
                h = _run_block(
                    block,
                    h,
                    c_levels[n_down],
                    activation_checkpointing=self.activation_checkpointing,
                )
        else:
            # Factorized: temporal then global-spatial (decoder order).
            for block_idx, (t_block, s_block) in enumerate(
                zip(self.mid_temporal, self.mid_spatial)
            ):
                h = einops.rearrange(h, "B T H W C -> (B H W) T C")
                if streaming:
                    cache_key = f"mid_temporal.{block_idx}"
                    h, kv = t_block.forward_stream(
                        h, old_cache.get(cache_key)
                    )
                    new_cache[cache_key] = self._trim_temporal_kv(
                        kv, max_cache_frames
                    )
                else:
                    h = _run_block(
                        t_block,
                        h,
                        activation_checkpointing=self.activation_checkpointing,
                    )
                h = einops.rearrange(
                    h, "(B H W) T C -> (B T) H W C", B=B, H=Hc, W=Wc
                )
                c_frames = (
                    einops.repeat(
                        c_levels[n_down],
                        "B C H W -> (B T) C H W",
                        T=T_cur,
                    )
                    if c_levels[n_down].shape[0] == B
                    else c_levels[n_down]
                )
                h = _run_block(
                    s_block,
                    h,
                    c_frames,
                    activation_checkpointing=self.activation_checkpointing,
                )
                h = einops.rearrange(h, "(B T) H W C -> B T H W C", B=B, T=T_cur)

        # Up levels
        for i in reversed(range(n_down)):
            # Temporal upsample
            if self.temporal_upsample[i]:
                h = self.up_levels[f"temporal_expand_{i}"](h)
                T_cur = h.shape[1]

            # Spatial upsample per-frame
            h = einops.rearrange(h, "B T H W C -> (B T) H W C")
            h = self.up_levels[f"expand_{i}"](h)
            h = einops.rearrange(h, "(B T) H W C -> B T H W C", B=B, T=T_cur)

            _, _, Hc, Wc, _ = h.shape

            # V3 raw-hdmap injection at the entry of this up-level. Per-
            # frame ``avg_pool2d`` is causality-preserving (output frame t
            # depends only on input frame t) and the spatial pool target
            # matches this stage's (Hc, Wc). Zero-init 1x1 conv keeps
            # iter-0 forward bit-identical.
            inject_key = f"hdmap_inject_{i}"
            if (
                self.hdmap_inject_raw_at_up_levels is not None
                and self.hdmap_inject_raw_at_up_levels[i]
                and hdmap_raw is not None
                and inject_key in self.up_levels
            ):
                # hdmap_raw: (B, C_h, T_h, H_h, W_h). Rearrange to per-frame
                # before the spatial avg_pool, then 1x1 conv to dim_i.
                if hdmap_raw.shape[2] != T_cur:
                    raise ValueError(
                        f"hdmap_raw temporal length {hdmap_raw.shape[2]} "
                        f"does not match decoder up-level {i} temporal "
                        f"grid {T_cur}. Pass a raw hdmap with the same "
                        f"number of frames as the original lidar video "
                        f"(per-frame, no temporal downsample)."
                    )
                hdmap_raw_flat = einops.rearrange(
                    hdmap_raw, "B C T H W -> (B T) C H W"
                )
                hdmap_feat = F.adaptive_avg_pool2d(
                    hdmap_raw_flat, output_size=(Hc, Wc)
                )
                hdmap_feat = self.up_levels[inject_key](hdmap_feat)
                hdmap_feat = einops.rearrange(
                    hdmap_feat, "(B T) C H W -> B T H W C", B=B, T=T_cur
                )
                h = h + hdmap_feat

            has_temporal = f"temporal_{i}" in self.up_levels
            t_blocks = self.up_levels[f"temporal_{i}"] if has_temporal else [None] * len(self.up_levels[f"spatial_{i}"])
            for block_idx, (t_block, s_block) in enumerate(
                zip(t_blocks, self.up_levels[f"spatial_{i}"])
            ):
                if t_block is not None:
                    h = einops.rearrange(h, "B T H W C -> (B H W) T C")
                    if streaming:
                        if not isinstance(t_block, TemporalBlock):
                            raise NotImplementedError(
                                "streaming decode currently supports temporal "
                                "attention, not temporal convolution"
                            )
                        cache_key = f"up_levels.temporal_{i}.{block_idx}"
                        h, kv = t_block.forward_stream(
                            h, old_cache.get(cache_key)
                        )
                        new_cache[cache_key] = self._trim_temporal_kv(
                            kv, max_cache_frames
                        )
                    else:
                        h = _run_block(
                            t_block,
                            h,
                            activation_checkpointing=self.activation_checkpointing,
                        )
                    h = einops.rearrange(
                        h, "(B H W) T C -> (B T) H W C", B=B, H=Hc, W=Wc
                    )
                else:
                    h = einops.rearrange(h, "B T H W C -> (B T) H W C")
                c_frames = (
                    einops.repeat(
                        c_levels[i], "B C H W -> (B T) C H W", T=T_cur
                    )
                    if c_levels[i].shape[0] == B
                    else c_levels[i]
                )
                h = _run_block(
                    s_block,
                    h,
                    c_frames,
                    activation_checkpointing=self.activation_checkpointing,
                )
                h = einops.rearrange(
                    h, "(B T) H W C -> B T H W C", B=B, T=T_cur
                )

        # Detokenize per-frame
        h = einops.rearrange(h, "B T H W C -> (B T) H W C")
        h = self.detokenizer(h)  # (BT, C_out, H, W)
        h = einops.rearrange(h, "(B T) C H W -> B C T H W", B=B, T=T_cur)
        if streaming:
            missing = set(old_cache) - set(new_cache)
            if missing:
                raise ValueError(
                    f"unused temporal KV cache keys: {sorted(missing)}"
                )
            return h, new_cache
        return h


# =============================================================================
# TransformerVAE
# =============================================================================


class TransformerVAE(nn.Module):
    def __init__(
        self,
        resolution: List[int],
        in_channels: int,
        out_channels: int | None = None,
        z_dim: int = 4,
        # Opt-in checkpoint surgery when widening an existing latent. The
        # source channels retain their exact encoder/quantizer/decoder path;
        # added channels start disconnected at zero.
        latent_warmstart_source_dim: int | None = None,
        base_channels: int = 128,
        patch_size: List[int] = (1, 4),
        window_size: List[int] = (3, 9),
        depths: List[int] = (3, 3, 3, 3),
        num_heads: List[int] = (2, 4, 8, 16),
        dilation: List[int] = (1, 1, 1, 1),
        temporal_downsample: List[bool] = (True, False, False),
        temporal_upsample: List[bool] | None = None,
        # ---- Asymmetric decoder (opt-in) ----
        # By default the decoder mirrors the encoder (same depths / num_heads /
        # dilation / temporal schedule, out_patch_size = patch_size). To build a
        # *plain-ViT* encoder (large patch + few merges) paired with a heavier
        # *progressive* decoder (small out_patch + more expand levels), set
        # these. ``spatial_compression`` / ``temporal_compression`` are always
        # derived from the ENCODER side. The only hard constraint is that the
        # decoder's bottleneck spatial grid
        #   resolution // out_patch_size >> (len(decoder_depths) - 1)
        # equals the encoder's latent grid
        #   resolution // patch_size    >> (len(depths) - 1).
        # When ``decoder_depths`` is set you MUST also pass
        # ``decoder_temporal_upsample`` (length len(decoder_depths) - 1) since
        # the encoder's temporal schedule no longer has the matching length.
        decoder_depths: List[int] | None = None,
        decoder_num_heads: List[int] | None = None,
        decoder_dilation: List[int] | None = None,
        decoder_temporal_upsample: List[bool] | None = None,
        out_patch_size: List[int] | None = None,
        mlp_ratio: float = 3.0,
        dropout: float = 0.0,
        mapping_depth: int = 2,
        positional_embedding: Literal[
            "spherical_harmonics",
            "polar_coordinates",
            "fourier_features",
            "learnable_embedding",
            "rope_only",
        ] = "learnable_embedding",
        formulation: str = ContinuousFormulation.AE.name,
        noise_aug_sigma: float = 0.0,
        # Decoder-only fine-tuning: freeze ``encoder`` + ``quant_conv`` so
        # only ``post_quant_conv`` + ``decoder`` are updated. Pair with
        # ``latent_stats_path`` + ``latent_norm_noise_sigma`` to fine-tune
        # the decoder on noise-degraded latents.
        freeze_encoder: bool = False,
        # Path to a ``latent_stats.pt`` file (a dict with ``mean``/``std``
        # tensors of shape (z_dim,), or a ``(mean, std)`` tuple) used to
        # normalize the latent before adding noise. Required whenever
        # ``latent_norm_noise_sigma > 0``.
        latent_stats_path: str | None = None,
        # Noise added in the NORMALIZED latent space. Interpretation
        # depends on ``latent_norm_noise_sigma_mode``:
        #   "fixed" / "uniform":  σ  (additive noise stddev)
        #   "interpolative":      γ  (noise stddev in the interpolation
        #                              formula z' = (1−τ)z + τγε)
        # Applied only during training. Orthogonal to ``noise_aug_sigma``
        # (which acts on the raw, unnormalized latent).
        latent_norm_noise_sigma: float = 0.0,
        # How to corrupt the normalized latent:
        #   "fixed":         z' = z + σ·ε,  σ = latent_norm_noise_sigma (constant).
        #   "uniform":       z' = z + σ·ε,  σ ~ U(0, latent_norm_noise_sigma)
        #                    per sample. Continuum including clean (σ=0).
        #   "interpolative": z' = (1−τ)·z + τ·γ·ε,  τ ~ U(0, τ_max) per sample,
        #                    γ = latent_norm_noise_sigma,
        #                    τ_max = latent_norm_noise_tau_max (default 1.0).
        #                    DeTok-style: forces full signal replacement at τ=1,
        #                    prevents the "additive shortcut" where the
        #                    original signal still dominates at high noise
        #                    levels (Yang et al., arXiv 2507.15856, Fig 2a).
        #                    Set τ_max < 1 to mimic LTX-style RF training
        #                    range matching ``--decoder_t1=τ_max`` early-stop
        #                    at inference (but without AdaLN t-conditioning;
        #                    use TransformerVAELTX for that).
        latent_norm_noise_sigma_mode: Literal["fixed", "uniform", "interpolative"] = "fixed",
        # Upper bound of τ in interpolative mode (ignored for other modes).
        # τ ~ U(0, τ_max). Defaults to 1.0 (full DeTok range). Use a smaller
        # value (e.g. 0.20) to restrict training to the low-noise regime that
        # the diffusion sampler actually leaves uncovered when early-stopping
        # at ``--decoder_t1=τ_max``.
        latent_norm_noise_tau_max: float = 1.0,
        # Per-sample probability of drawing τ = 0 (a perfectly clean latent)
        # *in place of* the normal τ ~ U(0, τ_max) draw, in the
        # "interpolative" mode. 0.0 (default) keeps the original behavior;
        # 0.5 gives a 50/50 mixture of (clean) and (uniform-up-to-τ_max).
        # Useful as a "noise-vs-clean mixture" training regime: half the
        # gradient signal preserves the base decoder's clean-signal
        # behavior (so the decoder doesn't drift off the τ=0 manifold the
        # way pure noise-FT does), the other half adds noise robustness.
        # Particularly relevant when τ_max is small (e.g. 0.10): without
        # this knob, every training sample sees *some* noise, which can
        # over-train noise-handling pathways at the expense of clean
        # reconstruction fidelity.
        latent_norm_noise_clean_prob: float = 0.0,
        # Whether to apply the noise in the normalized latent space (default)
        # or directly on the raw encoder output (DeTok-pure).
        #   True  (default): encode → (z - mean) / std → corrupt → · std + mean
        #                    → decode. Requires ``latent_stats_path``. Use this
        #                    when the encoder is frozen so the (mean, std)
        #                    reference is stable and τ has a clean unit-Gaussian
        #                    meaning.
        #   False: encode → corrupt → decode (no normalize/denormalize). Use
        #          this when the encoder is trainable -- frozen pretrained
        #          stats become a stale reference frame and the decoder ends
        #          up trained against a different latent distribution than it
        #          sees at inference (once stats are recomputed). Matches
        #          DeTok's recipe (Yang et al., arXiv 2507.15856). ``γ`` /
        #          ``σ`` now denote noise std in the raw latent space, so set
        #          them to the empirical encoder std (≈ 1 for a KL-VAE).
        latent_norm_noise_use_normalization: bool = True,
        # EMA momentum for online tracking of (latent_mean, latent_std) from
        # the raw encoder output during training. 0.0 = disabled (the loaded
        # ``latent_stats_path`` is used statically, as before). When > 0, each
        # training step:
        #   1. compute per-channel batch mean / std of z_0 (the raw encoder
        #      output, pre-normalization), reducing over (B, T, H, W);
        #   2. all-reduce across DDP ranks (op = AVG) so every rank tracks
        #      the same world-wide stats;
        #   3. EMA-update the persistent ``latent_mean`` / ``latent_std``
        #      buffers in-place:  buf = m * buf + (1 - m) * batch_stat.
        # Use this when the encoder is unfrozen (DeTok-style FT) so that
        # τ keeps its calibrated meaning even as the encoder marginal drifts
        # away from ``latent_stats_path``. Requires
        # ``latent_norm_noise_use_normalization=True`` and a non-None
        # ``latent_stats_path`` (the file is used as the EMA initialization).
        # Buffers become persistent when EMA is on, so the warm-start ckpt
        # must allow loose loading (set ``strict_resume=False``) or carry
        # them already.
        # Typical values: 0.999 (slow, ~1k-iter horizon), 0.99 (fast, ~100
        # iters).
        latent_norm_noise_ema_momentum: float = 0.0,
        # WAN-style first-frame-special temporal merging in the encoder.
        # See :class:`TemporalMergingFirstFrameSpecial` for details.
        # Pair with ``temporal_expand_wan_style=True`` so the decoder's
        # ``TemporalExpandingWanStyle`` mirrors the encoder's chunk
        # boundaries: ``T_in = 1 + (2 ** N) * (T_z - 1)`` exactly,
        # no front-crop, no boundary information loss.
        temporal_first_frame_special: bool = False,
        # WAN-style first-frame-special temporal expanding in the decoder.
        # When True every level with ``temporal_upsample[i]=True`` uses
        # :class:`TemporalExpandingWanStyle` (T_in -> 2*T_in - 1) instead
        # of plain :class:`TemporalExpanding` (T_in -> 2*T_in). Designed
        # to be paired with ``temporal_first_frame_special=True``.
        temporal_expand_wan_style: bool = False,
        # ---- Temporal-mixing mechanism (ablation) ----
        # "attention" (default): temporal mixing via
        # :class:`CausalTemporalAttention` in the down/mid/up temporal blocks.
        # "conv": WAN-style causal temporal convolution
        # (:class:`CausalTemporalConv`, kernel ``temporal_conv_kernel``,
        # spatial 1x1) instead. Spatial mixing (neighborhood attention) is
        # unchanged, so this is a clean temporal-attn-vs-temporal-conv swap.
        temporal_mixer: Literal["attention", "conv"] = "attention",
        temporal_conv_kernel: int = 3,
        # Recompute transformer blocks during backward to reduce activation
        # memory. Disabled by default so existing models are unchanged.
        activation_checkpointing: bool = False,
        # Autocast only the encoder stack and quant projection. Posterior
        # sampling, KL math, latent handoff, and the decoder remain FP32.
        encoder_autocast_dtype: str | None = None,
        # ---- HDmap injection into the decoder (V1 / V3) ----
        # When True (V1): build a sibling zero-init ``stem_hdmap``
        # projection in the decoder, encode the raw hdmap video through
        # the (frozen) ``self.encoder`` to produce ``hdmap_z`` of the
        # same shape as ``z``, and add ``stem_hdmap(hdmap_z)`` at the
        # decoder stem. Iter-0 bit-identical to the warm-start ckpt.
        hdmap_inject_stem: bool = False,
        # When not None (V3): list of length ``n_down = len(depths) - 1``
        # indexed by up-level i. For each ``True`` entry, the decoder
        # builds a zero-init 1x1 conv that injects per-frame avg-pooled
        # raw hdmap features at the entry of up-level i (post-
        # ``expand_i``, pre-blocks). Same iter-0 guarantee.
        hdmap_inject_raw_at_up_levels: list[bool] | None = None,
        # ---- PV-VAE-style predictive frame dropping (training only) ----
        # With probability ``frame_drop_prob`` per training forward, replace
        # the trailing ``k ~ U(frame_drop_min_latent, frame_drop_max_latent)``
        # *latent* frames (per sample) with a learnable pad token before
        # decoding, then reconstruct the FULL video. Because the encoder is
        # causal in time, masking trailing latents is equivalent to encoding
        # only the observed (past) frames + padding -- the decoder must
        # PREDICT the withheld future frames (arxiv:2605.02134). No-op when
        # ``frame_drop_prob == 0`` or in eval mode (validation stays clean
        # so recon metrics remain comparable).
        frame_drop_prob: float = 0.0,
        frame_drop_min_latent: int = 1,
        frame_drop_max_latent: int = 0,
        # ---- Validity (raydrop) mask prediction (arxiv:2605.22809) ----
        # When True, the decoder emits ONE extra output channel of validity
        # logits (1 = valid return, 0 = no return). The data channels are
        # unchanged; ``decode()`` returns only the data channels by default,
        # so all existing call sites are unaffected. The validity logits are
        # surfaced via ``forward()`` output for the BCE loss, and can be
        # retrieved at inference via ``decode(..., return_validity=True)``.
        predict_validity: bool = False,
        # ---- Validity mask as INPUT channel ----
        # When True, ``forward()`` derives a binary validity mask from the
        # (no-return-clamped) depth channel and concatenates it as an extra
        # INPUT channel BEFORE encoding. Pairs with ``predict_validity`` to
        # get a symmetric autoencoder: input ``[depth, mask]`` (2 ch) ->
        # output ``[depth(recon), validity_logit]`` (2 ch). ``in_channels``
        # MUST include this extra channel (e.g. mono depth -> in_channels=2).
        mask_as_input: bool = False,
        mask_input_min_value: float = -1.0,
        mask_input_eps: float = 0.02,
        # ---- Global Skip Connection (GSC, Qwen-Image-VAE-2.0 2605.13565) ----
        # When True, add a residual path that folds the raw ENCODER INPUT
        # (post ``mask_as_input`` concat) into the latent via a space-to-channel
        # (pixel-unshuffle by ``spatial_compression``) + 1x1 projection, added
        # to the sampled latent ``z`` inside ``encode()``. The high-frequency
        # input signal therefore lives IN the latent the DiT diffuses ->
        # diffusion-safe, unlike encoder->decoder DPT skips. Accelerates
        # convergence + sharpens edges at high spatial compression (the paper's
        # f16/f32 regime). Deliberately NOT zero-init: the convergence benefit
        # requires it active from step 0, so only enable on from-scratch (or
        # ``strict_resume=False``) runs. No-op for every existing config
        # (default False -> ``gsc_proj`` is never built, forward path
        # unchanged).
        global_skip: bool = False,
        # ---- Bottleneck 3D attention (encoder AND decoder) ----
        # When True, both the encoder and decoder replace their factorized
        # bottleneck (global spatial + temporal attention) with a single
        # joint 3D self-attention over (T*H*W) tokens (:class:`Bottleneck3DBlock`).
        # Default False -> both keep the factorized bottleneck, bit-identical
        # to every existing config. ``bottleneck_3d_max_t`` caps the temporal
        # PE / RoPE length (set >= largest latent T_z fed to the model);
        # ``bottleneck_3d_causal_time`` masks the joint attention to be causal
        # in time; ``bottleneck_3d_rope`` swaps the (learnable temporal PE +
        # 2D AxialRoPE) for a parameter-free 3D RoPE over (t, h, w).
        bottleneck_3d: bool = False,
        bottleneck_3d_max_t: int = 32,
        bottleneck_3d_causal_time: bool = False,
        bottleneck_3d_rope: bool = False,
        # ---- Utonia (PTv3) RANGE-VIEW feature alignment (REPA / Volt) ----
        # When True, run a frozen pretrained PTv3 (Pointcept/Utonia) on the
        # input range map, masked-pool its per-pixel features down to the
        # latent grid (this is RANGE-VIEW alignment, NOT the UniPAD BEV scatter
        # -- the teacher already lives in the same elevation x azimuth frame as
        # the range-image latent), and minimize a masked cosine distance to a
        # trainable 1x1 projection of the latent. Encoder-side alignment: the
        # gradient flows into the transformer encoder, pulling the latent toward
        # a semantically-meaningful frozen representation (REPA arXiv:2410.06940,
        # Volt arXiv:2604.19609). Teacher is fully frozen + eval. Adds one PTv3
        # forward per step (slow CPU GridSample) and needs the ``utonia`` package
        # (spconv/torch_scatter/flash_attn) + the HF checkpoint -> cluster only.
        # Default False -> no teacher/projector built, every existing config
        # bit-identical.
        utonia_align: bool = False,
        utonia_dataset_name: str = "lidar_range_map_rRow4_smol_norepeat_val",
        utonia_scale: float = 0.2,
        utonia_download_root: str = "",
        # Normalization geometry the teacher uses to unproject the normalized
        # range image -> metric XYZ (must match the dataloader's convention).
        # Defaults match the lidargen lidar default (configs.py COMMONDATA).
        utonia_min_range: float = 5.0,
        utonia_max_range: float = 100.0,
        utonia_min_value: float = -1.0,
        utonia_inverse_depth: bool = False,
        # Run the teacher at a REDUCED azimuth width (0 = full self.resolution[1]).
        # The PTv3 + CPU GridSample cost scales with point count (H*W); the
        # teacher features are pooled to the (Hz, Wz) latent grid anyway, so a
        # coarser azimuth (still >= Wz) is a near-free speedup. Only azimuth (W)
        # is reduced -- elevation rows are physical beams and stay at 128.
        utonia_teacher_width: int = 0,
        # VA-VAE VF loss margins (arXiv:2501.01423). ``vf_margin_cos`` = m1 for
        # the marginal cosine similarity loss (Eq. 2); ``vf_margin_dist`` = m2
        # for the marginal distance-matrix similarity loss (Eq. 3). The overall
        # w_hyper (paper 0.1) is the ``repa_align`` loss weight; the adaptive
        # gradient-ratio weighting (Eq. 4) is intentionally NOT implemented.
        vf_margin_cos: float = 0.5,
        vf_margin_dist: float = 0.25,
        # Memory knobs for the VF loss (default 0 = paper-exact, no subsample):
        #   vf_mdms_window: (wh, ww) -> LOCAL-WINDOW block mdms. Tile the Hz x Wz
        #     latent grid into non-overlapping (wh, ww) patches and compute the
        #     self-similarity matrix WITHIN each patch only -> block-diagonal
        #     (M*nh*nw, P, P), P=wh*ww. Pairs stay spatially local (meaningful
        #     for the very wide range-image azimuth) and it's deterministic.
        #     Requires Hz % wh == 0 and Wz % ww == 0. Takes precedence over
        #     vf_mdms_max_tokens. (0, 0) = disabled.
        #   vf_mdms_max_tokens: else, cap the GLOBAL mdms matrix to S random
        #     VALID tokens per frame -> (M, S, S) instead of (M, N, N) (a
        #     stochastic estimate). mcos is per-pixel/cheap and never subsampled.
        #   vf_supervise_frames: run the teacher + VF loss on only K random
        #     frames per clip (cuts PTv3 forwards AND #matrices). Requires no
        #     temporal compression (latent T == input T); raises otherwise.
        vf_mdms_window: tuple[int, int] = (0, 0),
        vf_mdms_max_tokens: int = 0,
        vf_supervise_frames: int = 0,
    ):
        super().__init__()
        self.resolution = _pair(resolution)
        self.in_channels = in_channels
        self.out_channels = out_channels if out_channels else in_channels
        self.predict_validity = predict_validity
        self.mask_as_input = mask_as_input
        self.mask_input_min_value = float(mask_input_min_value)
        self.mask_input_eps = float(mask_input_eps)
        # When True, ``forward(..., valid_mask=...)`` uses an externally-provided
        # per-pixel validity mask for the ``mask_as_input`` concat instead of
        # deriving it from ``depth > min_value + eps`` (which is unrecoverable
        # under per-row normalization). The model only threads ``valid_mask``
        # when this is True; if the kwarg is None at forward time we fall back
        # to the derived mask, so existing mask configs are unaffected.
        self.uses_valid_mask = mask_as_input
        # Decoder produces data channels (+1 validity logit channel if enabled).
        self._decoder_out_channels = self.out_channels + (1 if predict_validity else 0)
        self.z_dim = z_dim
        self.latent_warmstart_source_dim = latent_warmstart_source_dim
        if latent_warmstart_source_dim is not None:
            assert 0 < latent_warmstart_source_dim < z_dim, (
                "latent_warmstart_source_dim must be positive and smaller than "
                f"z_dim, got {latent_warmstart_source_dim=} and {z_dim=}"
            )
        self.patch_size = _pair(patch_size)
        self.noise_aug_sigma = noise_aug_sigma
        self.freeze_encoder = freeze_encoder
        self.latent_norm_noise_sigma = latent_norm_noise_sigma
        if latent_norm_noise_sigma_mode not in ("fixed", "uniform", "interpolative"):
            raise ValueError(
                f"latent_norm_noise_sigma_mode must be 'fixed', 'uniform', or "
                f"'interpolative', got {latent_norm_noise_sigma_mode!r}"
            )
        self.latent_norm_noise_sigma_mode = latent_norm_noise_sigma_mode
        if not (0.0 < latent_norm_noise_tau_max <= 1.0):
            raise ValueError(
                f"latent_norm_noise_tau_max must be in (0, 1], got {latent_norm_noise_tau_max}"
            )
        self.latent_norm_noise_tau_max = float(latent_norm_noise_tau_max)
        if not (0.0 <= latent_norm_noise_clean_prob <= 1.0):
            raise ValueError(
                f"latent_norm_noise_clean_prob must be in [0, 1], got "
                f"{latent_norm_noise_clean_prob}"
            )
        self.latent_norm_noise_clean_prob = float(latent_norm_noise_clean_prob)
        self.latent_norm_noise_use_normalization = bool(
            latent_norm_noise_use_normalization
        )
        if not (0.0 <= latent_norm_noise_ema_momentum < 1.0):
            raise ValueError(
                f"latent_norm_noise_ema_momentum must be in [0, 1), got "
                f"{latent_norm_noise_ema_momentum}"
            )
        self.latent_norm_noise_ema_momentum = float(latent_norm_noise_ema_momentum)
        if self.latent_norm_noise_ema_momentum > 0.0:
            if not self.latent_norm_noise_use_normalization:
                raise ValueError(
                    "latent_norm_noise_ema_momentum > 0 requires "
                    "latent_norm_noise_use_normalization=True (EMA only "
                    "tracks the normalize-space reference frame)."
                )
            if latent_stats_path is None:
                raise ValueError(
                    "latent_norm_noise_ema_momentum > 0 requires "
                    "latent_stats_path to be set (used as the EMA "
                    "initialization)."
                )

        n_down = len(depths) - 1
        self.spatial_compression = (
            self.patch_size[0] * (2 ** n_down),
            self.patch_size[1] * (2 ** n_down),
        )
        self.temporal_compression = 2 ** sum(temporal_downsample)
        self._encoder_autocast_dtype = (
            getattr(torch, encoder_autocast_dtype)
            if encoder_autocast_dtype
            else None
        )

        coords = encoding.generate_polar_coords(*self.resolution)
        self.register_buffer("coords", coords)

        self.encoder = Encoder(
            resolution=resolution,
            in_channels=in_channels,
            z_dim=z_dim,
            base_channels=base_channels,
            patch_size=patch_size,
            window_size=window_size,
            depths=depths,
            num_heads=num_heads,
            dilation=dilation,
            temporal_downsample=temporal_downsample,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            mapping_depth=mapping_depth,
            positional_embedding=positional_embedding,
            temporal_first_frame_special=temporal_first_frame_special,
            bottleneck_3d=bottleneck_3d,
            bottleneck_3d_max_t=bottleneck_3d_max_t,
            bottleneck_3d_causal_time=bottleneck_3d_causal_time,
            bottleneck_3d_rope=bottleneck_3d_rope,
            temporal_mixer=temporal_mixer,
            temporal_conv_kernel=temporal_conv_kernel,
            activation_checkpointing=activation_checkpointing,
        )

        self.distribution = ContinuousFormulation[formulation].value()
        logger.info(f"TransformerVAE with {formulation} formulation.")

        self.quant_conv = nn.Conv2d(z_dim * 2, z_dim * 2, 1)
        self.post_quant_conv = nn.Conv2d(z_dim, z_dim, 1)

        # Global Skip Connection (GSC): space-to-channel fold of the encoder
        # input -> 1x1 proj -> residual added to the latent in ``encode()``.
        # in-dim = in_channels * (sh * sw) where (sh, sw) = spatial_compression
        # (the pixel-unshuffle factor that brings the input to latent res).
        self.global_skip = bool(global_skip)
        if self.global_skip:
            _sh, _sw = self.spatial_compression
            self.gsc_proj = nn.Conv2d(in_channels * _sh * _sw, z_dim, kernel_size=1)
            logger.info(
                f"TransformerVAE: Global Skip Connection enabled "
                f"(S2C fold {self.spatial_compression}, "
                f"{in_channels * _sh * _sw} -> {z_dim} channels)."
            )

        # PV-VAE predictive frame dropping. ``frame_drop_pad`` is a single
        # learnable latent vector broadcast over the dropped trailing
        # positions (and over H, W); zero-init so warm-start is a gentle
        # start (decoder sees zeros where it must learn to predict).
        self.frame_drop_prob = float(frame_drop_prob)
        self.frame_drop_min_latent = int(frame_drop_min_latent)
        self.frame_drop_max_latent = int(frame_drop_max_latent)
        if not (0.0 <= self.frame_drop_prob <= 1.0):
            raise ValueError(
                f"frame_drop_prob must be in [0, 1], got {self.frame_drop_prob}"
            )
        if self.frame_drop_prob > 0.0:
            if self.frame_drop_max_latent < self.frame_drop_min_latent:
                raise ValueError(
                    "frame_drop_max_latent must be >= frame_drop_min_latent "
                    f"when frame_drop_prob > 0 (got max="
                    f"{self.frame_drop_max_latent}, min={self.frame_drop_min_latent})"
                )
            self.frame_drop_pad = nn.Parameter(torch.zeros(1, z_dim, 1, 1, 1))

        # --- Utonia (PTv3) range-view alignment teacher (frozen) + projector ---
        self.utonia_align = bool(utonia_align)
        if self.utonia_align:
            raise ValueError(
                "Utonia alignment is not included in the public LidarVerse "
                "range-tokenizer recipe"
            )

        self.hdmap_inject_stem = bool(hdmap_inject_stem)
        self.hdmap_inject_raw_at_up_levels = (
            list(bool(b) for b in hdmap_inject_raw_at_up_levels)
            if hdmap_inject_raw_at_up_levels is not None
            else None
        )
        # Whether the network needs an hdmap input at forward time at all.
        self.uses_hdmap = self.hdmap_inject_stem or (
            self.hdmap_inject_raw_at_up_levels is not None
            and any(self.hdmap_inject_raw_at_up_levels)
        )

        # Resolve the (possibly asymmetric) decoder topology. Defaults mirror
        # the encoder so every existing config stays bit-identical.
        _dec_depths = decoder_depths if decoder_depths is not None else depths
        _dec_num_heads = (
            decoder_num_heads if decoder_num_heads is not None else num_heads
        )
        _dec_dilation = decoder_dilation if decoder_dilation is not None else dilation
        if decoder_depths is not None:
            if decoder_temporal_upsample is None:
                raise ValueError(
                    "decoder_depths is set (asymmetric decoder) so the encoder's "
                    "temporal schedule no longer has the matching length; pass "
                    "decoder_temporal_upsample with length "
                    f"len(decoder_depths) - 1 = {len(_dec_depths) - 1}."
                )
            _dec_temporal_upsample = decoder_temporal_upsample
            _dec_temporal_downsample = decoder_temporal_upsample
        else:
            _dec_temporal_upsample = temporal_upsample
            _dec_temporal_downsample = temporal_downsample

        self.decoder = Decoder(
            resolution=resolution,
            out_channels=self._decoder_out_channels,
            z_dim=z_dim,
            base_channels=base_channels,
            patch_size=patch_size,
            out_patch_size=out_patch_size,
            window_size=window_size,
            depths=_dec_depths,
            num_heads=_dec_num_heads,
            dilation=_dec_dilation,
            temporal_downsample=_dec_temporal_downsample,
            temporal_upsample=_dec_temporal_upsample,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            mapping_depth=mapping_depth,
            positional_embedding=positional_embedding,
            temporal_expand_wan_style=temporal_expand_wan_style,
            hdmap_inject_stem=self.hdmap_inject_stem,
            hdmap_inject_raw_at_up_levels=self.hdmap_inject_raw_at_up_levels,
            hdmap_in_channels=in_channels,
            bottleneck_3d=bottleneck_3d,
            bottleneck_3d_max_t=bottleneck_3d_max_t,
            bottleneck_3d_causal_time=bottleneck_3d_causal_time,
            bottleneck_3d_rope=bottleneck_3d_rope,
            temporal_mixer=temporal_mixer,
            temporal_conv_kernel=temporal_conv_kernel,
            activation_checkpointing=activation_checkpointing,
        )
        self.supports_temporal_kv_cache = (
            not bottleneck_3d
            and not any(
                _dec_temporal_upsample
                if _dec_temporal_upsample is not None
                else _dec_temporal_downsample
            )
            and temporal_mixer == "attention"
        )
        self.supports_encoder_temporal_kv_cache = (
            not bottleneck_3d
            and not any(temporal_downsample)
            and temporal_mixer == "attention"
        )

        # Latent normalization stats for normalized-space noise
        # augmentation. Buffers are normally non-persistent so they don't
        # appear in the saved state_dict (the pretrained checkpoint doesn't
        # carry them, and we want ``strict_resume=True`` to keep working).
        # When EMA tracking is on we flip to persistent=True so the
        # accumulated EMA state survives autoresume mid-run; this requires
        # the warm-start config to use ``strict_resume=False`` (the
        # pretrained ckpt won't have these keys).
        self.has_latent_stats = False
        _stats_persistent = self.latent_norm_noise_ema_momentum > 0.0
        if latent_stats_path is not None:
            stats = torch.load(
                latent_stats_path, map_location="cpu", weights_only=False
            )
            if isinstance(stats, (tuple, list)):
                latent_mean_t, latent_std_t = stats
            elif isinstance(stats, dict):
                latent_mean_t = stats["mean"]
                latent_std_t = stats["std"]
            else:
                raise ValueError(
                    f"Unsupported latent_stats format at {latent_stats_path}: "
                    f"{type(stats)}"
                )
            assert latent_mean_t.numel() == z_dim and latent_std_t.numel() == z_dim, (
                f"latent_stats must have z_dim={z_dim} entries, got "
                f"mean={latent_mean_t.numel()}, std={latent_std_t.numel()}"
            )
            target_shape = (1, z_dim, 1, 1, 1)
            self.register_buffer(
                "latent_mean",
                latent_mean_t.float().reshape(*target_shape),
                persistent=_stats_persistent,
            )
            self.register_buffer(
                "latent_std",
                latent_std_t.float().reshape(*target_shape),
                persistent=_stats_persistent,
            )
            self.has_latent_stats = True
            logger.info(
                f"TransformerVAE: loaded latent_stats from {latent_stats_path} "
                f"(latent_norm_noise_sigma={self.latent_norm_noise_sigma}, "
                f"mode={self.latent_norm_noise_sigma_mode}, "
                f"tau_max={self.latent_norm_noise_tau_max}, "
                f"clean_prob={self.latent_norm_noise_clean_prob}, "
                f"ema_momentum={self.latent_norm_noise_ema_momentum}, "
                f"stats_persistent={_stats_persistent})"
            )

        if self.latent_norm_noise_sigma > 0:
            assert self.has_latent_stats or not self.latent_norm_noise_use_normalization, (
                "latent_norm_noise_sigma > 0 requires either latent_stats_path "
                "to be set (use_normalization=True) or "
                "latent_norm_noise_use_normalization=False (DeTok-pure, "
                "noise applied directly on the raw latent)."
            )
            if not self.latent_norm_noise_use_normalization:
                logger.info(
                    f"TransformerVAE: noise injection in RAW latent space "
                    f"(latent_norm_noise_use_normalization=False, "
                    f"sigma/gamma={self.latent_norm_noise_sigma}, "
                    f"mode={self.latent_norm_noise_sigma_mode})."
                )

        if self.freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
            for p in self.quant_conv.parameters():
                p.requires_grad = False
            self.encoder.eval()
            self.quant_conv.eval()
            logger.info(
                "TransformerVAE: encoder + quant_conv frozen (decoder-only fine-tuning)."
            )

    def train(self, mode: bool = True):
        """Keep frozen submodules in eval mode regardless of model.train()."""
        super().train(mode)
        if self.freeze_encoder:
            self.encoder.eval()
            self.quant_conv.eval()
        return self

    def adapt_warmstart_tensor(
        self,
        name: str,
        source: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor | None:
        """Expand the z64 latent interface into a function-preserving z96 one."""
        source_dim = self.latent_warmstart_source_dim
        if source_dim is None:
            return None

        name = name.removeprefix("network.")
        target_dim = self.z_dim
        expanded = source.new_zeros(target.shape)

        if name == "encoder.head.1.weight":
            if source.shape[0] != 2 * source_dim or target.shape[0] != 2 * target_dim:
                raise ValueError(f"Unexpected latent head shapes: {source.shape} -> {target.shape}")
            expanded[:source_dim] = source[:source_dim]
            expanded[target_dim : target_dim + source_dim] = source[source_dim:]
        elif name == "quant_conv.weight":
            expected_source = (2 * source_dim, 2 * source_dim)
            expected_target = (2 * target_dim, 2 * target_dim)
            if source.shape[:2] != expected_source or target.shape[:2] != expected_target:
                raise ValueError(f"Unexpected quant_conv shapes: {source.shape} -> {target.shape}")
            expanded[:source_dim, :source_dim] = source[:source_dim, :source_dim]
            expanded[:source_dim, target_dim : target_dim + source_dim] = source[
                :source_dim, source_dim:
            ]
            expanded[target_dim : target_dim + source_dim, :source_dim] = source[
                source_dim:, :source_dim
            ]
            expanded[
                target_dim : target_dim + source_dim,
                target_dim : target_dim + source_dim,
            ] = source[source_dim:, source_dim:]
        elif name == "quant_conv.bias":
            if source.shape[0] != 2 * source_dim or target.shape[0] != 2 * target_dim:
                raise ValueError(f"Unexpected quant_conv bias shapes: {source.shape} -> {target.shape}")
            expanded[:source_dim] = source[:source_dim]
            expanded[target_dim : target_dim + source_dim] = source[source_dim:]
        elif name == "post_quant_conv.weight":
            if source.shape[:2] != (source_dim, source_dim) or target.shape[:2] != (
                target_dim,
                target_dim,
            ):
                raise ValueError(f"Unexpected post_quant_conv shapes: {source.shape} -> {target.shape}")
            expanded[:source_dim, :source_dim] = source
        elif name == "post_quant_conv.bias":
            if source.shape[0] != source_dim or target.shape[0] != target_dim:
                raise ValueError(
                    f"Unexpected post_quant_conv bias shapes: {source.shape} -> {target.shape}"
                )
            expanded[:source_dim] = source
        elif name == "decoder.stem.1.weight":
            if source.shape[1] != source_dim or target.shape[1] != target_dim:
                raise ValueError(f"Unexpected decoder stem shapes: {source.shape} -> {target.shape}")
            expanded[:, :source_dim] = source
        else:
            return None

        return expanded

    def _resolve_spatial_coords(
        self,
        reference: torch.Tensor,
        spatial_coords: torch.Tensor | None,
        spatial_shape: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        coords = self.coords if spatial_coords is None else spatial_coords
        expected_shape = spatial_shape or tuple(reference.shape[-2:])
        if tuple(coords.shape[-2:]) != tuple(expected_shape):
            raise ValueError(
                f"Coordinate grid {tuple(coords.shape[-2:])} does not match "
                f"input grid {tuple(expected_shape)}"
            )
        return coords.to(device=reference.device)

    def encode(
        self, x: torch.Tensor, spatial_coords: torch.Tensor | None = None
    ):
        """
        Args:
            x: (B, C, H, W) [single-frame] or (B, C, T, H, W)
        Returns:
            sample: (B, z_dim, T', H_z, W_z)
            posteriors: (mu, log_var) each (B, z_dim, T', H_z, W_z)
        """
        # Single-frame (image) input: add a T=1 axis so the video encoder
        # path works unchanged. The causal temporal ops degrade gracefully
        # at T=1 (TemporalMerging pads -> kernel-3 stride-2 -> T stays 1) and
        # ``decode()`` squeezes the axis back. ``_input_T`` records T so the
        # decode crop / squeeze stay consistent.
        if x.ndim == 4:
            x = x.unsqueeze(2)  # (B, C, H, W) -> (B, C, 1, H, W)
        self._input_T = x.shape[2]
        self._input_spatial_shape = tuple(x.shape[-2:])
        coords = self._resolve_spatial_coords(x, spatial_coords)
        encoder_amp = (
            torch.autocast("cuda", dtype=self._encoder_autocast_dtype)
            if self._encoder_autocast_dtype is not None
            and x.device.type == "cuda"
            else contextlib.nullcontext()
        )
        with encoder_amp:
            parameters = self.encoder(
                x, coords
            )  # (B, z_dim*2, T', H_z, W_z)

            # quant_conv is 2D — apply per-frame
            B, C2, Tz, Hz, Wz = parameters.shape
            parameters = einops.rearrange(
                parameters, "B C T H W -> (B T) C H W"
            )
            parameters = self.quant_conv(parameters)
            parameters = einops.rearrange(
                parameters, "(B T) C H W -> B C T H W", B=B, T=Tz
            )
        if self._encoder_autocast_dtype is not None:
            parameters = parameters.float()

        sample, (mu, log_var) = self.distribution(parameters, split=True)

        # Global Skip Connection: fold the raw input pixels into the latent as
        # a residual so high-frequency signal lives IN z (diffusion-safe).
        if self.global_skip:
            sample = sample + self._global_skip_residual(x, sample)

        return sample, (mu, log_var)

    def encode_stream(
        self,
        x_new: torch.Tensor,
        temporal_kv_cache: dict[
            str, tuple[torch.Tensor, torch.Tensor]
        ] | None = None,
        max_cache_frames: int | None = None,
        spatial_coords: torch.Tensor | None = None,
    ):
        """Encode newly appended frames using causal encoder KV caches."""
        if x_new.ndim != 5:
            raise ValueError(
                f"streaming encode requires (B,C,T,H,W), got {x_new.shape}"
            )
        if not self.supports_encoder_temporal_kv_cache:
            raise NotImplementedError(
                "this TransformerVAE topology does not support streaming encode"
            )
        self._input_spatial_shape = tuple(x_new.shape[-2:])
        coords = self._resolve_spatial_coords(x_new, spatial_coords)
        parameters, new_cache = self.encoder.forward_stream(
            x_new,
            coords,
            temporal_kv_cache=temporal_kv_cache,
            max_cache_frames=max_cache_frames,
        )
        B, C2, Tz, Hz, Wz = parameters.shape
        parameters = einops.rearrange(
            parameters, "B C T H W -> (B T) C H W"
        )
        parameters = self.quant_conv(parameters)
        parameters = einops.rearrange(
            parameters,
            "(B T) C H W -> B C T H W",
            B=B,
            T=Tz,
        )
        sample, (mu, log_var) = self.distribution(parameters, split=True)
        if self.global_skip:
            sample = sample + self._global_skip_residual(x_new, sample)
        return (sample, (mu, log_var)), new_cache

    def _global_skip_residual(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Qwen-VAE-2.0 GSC residual (arxiv:2605.13565).

        Space-to-channel (pixel-unshuffle) fold of the encoder input ``x`` down
        to the latent grid, then a 1x1 projection to ``z_dim``. Returned as the
        residual to add to the sampled latent ``z``.

        Args:
            x: (B, C_in, T, H, W) encoder input (already mask-concatenated when
               ``mask_as_input`` is on).
            z: (B, z_dim, Tz, Hz, Wz) sampled latent (for target shape).
        Returns:
            (B, z_dim, Tz, Hz, Wz) residual.

        The spatial fold is lossless (the GSC essence). The temporal axis is
        aligned by adaptive avg-pool when ``T != Tz`` (identity for the
        single-frame stage where T == Tz == 1); a more faithful causal temporal
        fold can be revisited for the video stage.
        """
        B, _, T, H, W = x.shape
        Tz, Hz, Wz = z.shape[2], z.shape[3], z.shape[4]
        ph, pw = H // Hz, W // Wz
        xf = einops.rearrange(
            x, "B C T (h ph) (w pw) -> B (C ph pw) T h w", ph=ph, pw=pw
        )
        if T != Tz:
            xf = F.adaptive_avg_pool3d(xf, output_size=(Tz, Hz, Wz))
        xf = einops.rearrange(xf, "B C T h w -> (B T) C h w")
        g = self.gsc_proj(xf)
        g = einops.rearrange(g, "(B T) C h w -> B C T h w", B=B, T=Tz)
        return g

    def _utonia_vf_loss(self, sample: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """VA-VAE VF loss (arXiv:2501.01423) in RANGE-VIEW, minus adaptive weighting.

        Run the frozen PTv3 teacher on the input depth, masked-pool its per-pixel
        features to the latent grid (teacher already lives in the same elevation x
        azimuth frame as the range-image latent, so no BEV scatter), then sum:

          * mcos (Eq. 2): per-position ReLU(1 - m1 - cos(z'_ij, f_ij)), z' = W z
            (linear ``utonia_proj``). Element-wise / point-to-point alignment.
          * mdms (Eq. 3): pairwise ReLU(|cos(z_i,z_j) - cos(f_i,f_j)| - m2) on the
            RAW latent self-similarity matrix vs the teacher's. Relative-structure
            alignment. Computed per (batch, frame) over the H*W latent positions.

        Both terms are averaged over VALID positions/pairs only (lidar range maps
        are sparse: no-return pixels have no teacher feature). Returns a scalar
        (= mcos + mdms); the overall w_hyper is applied by the ``repa_align`` loss
        weight, and the adaptive gradient-ratio weighting is omitted by design.

        Args:
            sample: (B, z, Tz, Hz, Wz) clean encoder latent.
            x:      (B, C, T, H, W) encoder input (channel 0 = normalized depth).
        """
        depth = x[:, :1]  # (B, 1, [T,] H, W) normalized range
        # Explicit GT validity (raydrop) mask, so the teacher never builds
        # points at no-return pixels. With ``mask_as_input`` the appended 2nd
        # channel IS that mask (1 == return); otherwise derive it from the same
        # ``depth > min_value + eps`` threshold the validity BCE uses.
        if self.mask_as_input and x.shape[1] >= 2:
            vmask = x[:, 1:2]
        else:
            vmask = (
                depth > (self.mask_input_min_value + self.mask_input_eps)
            ).to(depth.dtype)
        # ``sample`` is always 5D (encode() unsqueezes single-frame T=1), but the
        # raw input ``x`` may be 4D (image dataset). Force depth to 5D so the
        # teacher returns 5D and ``adaptive_avg_pool3d`` pools T (not the feature
        # channels) -- a 4D input is silently treated as unbatched (C,D,H,W).
        if depth.ndim == 4:
            depth = depth.unsqueeze(2)  # (B, 1, 1, H, W)
            vmask = vmask.unsqueeze(2)

        # Optional FRAME subsampling: supervise only K random frames per clip.
        # Slices the input (saving K/T of the PTv3 forwards) AND the latent, so
        # only requires the latent/input frame counts to match (no temporal
        # compression). Skipped for single-frame / when K >= T.
        K = getattr(self, "vf_supervise_frames", 0)
        if K and K < depth.shape[2]:
            T_in = depth.shape[2]
            if sample.shape[2] != T_in:
                raise ValueError(
                    f"vf_supervise_frames requires latent T ({sample.shape[2]}) == "
                    f"input T ({T_in}); not supported with temporal compression."
                )
            idx = torch.randperm(T_in, device=depth.device)[:K].sort().values
            depth = depth[:, :, idx].contiguous()
            vmask = vmask[:, :, idx].contiguous()
            sample = sample[:, :, idx].contiguous()

        B, _, Tz, Hz, Wz = sample.shape
        # Run the teacher at a coarser azimuth (point count ~ H*W drives the
        # PTv3 + CPU GridSample cost). Nearest-exact on W only; rows (beams)
        # untouched. depth/vmask are (B, 1, T, H, W) here.
        tw = getattr(self, "utonia_teacher_width", depth.shape[-1])
        if tw and tw != depth.shape[-1]:
            Bd, _, Td, Hd, _ = depth.shape
            depth = F.interpolate(
                depth.reshape(Bd * Td, 1, Hd, depth.shape[-1]),
                size=(Hd, tw), mode="nearest-exact",
            ).reshape(Bd, 1, Td, Hd, tw)
            vmask = F.interpolate(
                vmask.reshape(Bd * Td, 1, Hd, vmask.shape[-1]),
                size=(Hd, tw), mode="nearest-exact",
            ).reshape(Bd, 1, Td, Hd, tw)
        # Teacher features + masked spatial pool to the latent grid, done ONE
        # INPUT FRAME AT A TIME and in bf16. The dense teacher map is
        # (B, D~1224, T, H, W): at full res that's ~3 GB/frame in fp32, and the
        # ``abs().sum`` / ``* tvalid`` temps each spawn another copy -> the
        # multi-frame version OOMs. Per-frame + bf16 bounds the peak to one
        # frame; everything here is a frozen, detached TARGET so no autograd
        # buffers are kept. Each frame is spatially pooled to (Hz, Wz) first
        # (cheap output), then the small stack is temporally pooled T_in -> Tz
        # (identity when there's no temporal compression).
        pool_dtype = torch.bfloat16
        T_in = depth.shape[2]
        with torch.no_grad():
            f_parts: list[torch.Tensor] = []
            v_parts: list[torch.Tensor] = []
            for t in range(T_in):
                teacher_t = self.utonia(
                    depth[:, :, t:t + 1], valid_mask=vmask[:, :, t:t + 1]
                )  # (B, D, 1, H, W), 0 at invalid
                if teacher_t.ndim == 4:
                    teacher_t = teacher_t.unsqueeze(2)
                teacher_t = teacher_t.to(pool_dtype)
                tvalid_t = (
                    teacher_t.abs().sum(dim=1, keepdim=True) > 0
                ).to(pool_dtype)                                # (B, 1, 1, H, W)
                num_t = F.adaptive_avg_pool3d(teacher_t * tvalid_t, (1, Hz, Wz))
                den_t = F.adaptive_avg_pool3d(tvalid_t, (1, Hz, Wz))
                f_parts.append(num_t / den_t.clamp_min(1e-6))   # (B, D, 1, Hz, Wz)
                v_parts.append(den_t)                           # (B, 1, 1, Hz, Wz) valid frac
                del teacher_t, tvalid_t, num_t, den_t
            f_hi = torch.cat(f_parts, dim=2)                    # (B, D, T_in, Hz, Wz)
            v_hi = torch.cat(v_parts, dim=2)                    # (B, 1, T_in, Hz, Wz)
            # Temporal pool T_in -> Tz (identity when T_in == Tz), validity-weighted.
            num = F.adaptive_avg_pool3d(f_hi * v_hi, (Tz, Hz, Wz))
            den = F.adaptive_avg_pool3d(v_hi, (Tz, Hz, Wz))
            f_grid = (num / den.clamp_min(1e-6)).to(sample.dtype)  # (B, D, Tz, Hz, Wz)
            valid = (den > 0).to(sample.dtype)                  # (B, 1, Tz, Hz, Wz)
        f_grid = f_grid.detach()
        valid = valid.detach()

        # Fold (batch, frame) so each "image" is one (C, Hz, Wz) latent frame.
        z = einops.rearrange(sample, "B C T H W -> (B T) C H W")
        f = einops.rearrange(f_grid, "B C T H W -> (B T) C H W")
        m = einops.rearrange(valid, "B C T H W -> (B T) C H W")  # (M, 1, Hz, Wz)
        M = z.shape[0]

        # ---- mcos (Eq. 2): point-to-point cosine, projected latent z' = W z ----
        z_proj = self.utonia_proj(z)                        # (M, D, Hz, Wz)
        z_proj = F.normalize(z_proj, dim=1)
        f_n = F.normalize(f, dim=1)
        cos = (z_proj * f_n).sum(dim=1, keepdim=True)       # (M, 1, Hz, Wz)
        mcos_pix = F.relu(1.0 - self.vf_margin_cos - cos) * m
        loss_mcos = mcos_pix.sum() / m.sum().clamp_min(1.0)

        # ---- mdms (Eq. 3): self-similarity matrices on RAW z vs teacher f ----
        N = Hz * Wz
        zf = einops.rearrange(z, "M C H W -> M (H W) C")
        ff = einops.rearrange(f, "M C H W -> M (H W) C")
        mf = einops.rearrange(m, "M C H W -> M (H W) C").squeeze(-1)  # (M, N)
        zf = F.normalize(zf, dim=-1)
        ff = F.normalize(ff, dim=-1)

        win = getattr(self, "vf_mdms_window", (0, 0))
        wh, ww = int(win[0]), int(win[1])
        S = getattr(self, "vf_mdms_max_tokens", 0)
        if wh > 0 and ww > 0:
            # LOCAL-WINDOW block mdms: tile (Hz, Wz) into (wh, ww) patches and
            # only match token PAIRS inside the same patch -> block-diagonal
            # (M*nh*nw, P, P), P = wh*ww. Pairs stay spatially local.
            if Hz % wh != 0 or Wz % ww != 0:
                raise ValueError(
                    f"vf_mdms_window {(wh, ww)} must tile the latent grid "
                    f"({Hz}x{Wz}); got Hz%wh={Hz % wh}, Wz%ww={Wz % ww}."
                )
            zw = einops.rearrange(
                z, "M C (nh wh) (nw ww) -> (M nh nw) (wh ww) C", wh=wh, ww=ww
            )
            fw = einops.rearrange(
                f, "M C (nh wh) (nw ww) -> (M nh nw) (wh ww) C", wh=wh, ww=ww
            )
            mw = einops.rearrange(
                m, "M C (nh wh) (nw ww) -> (M nh nw) (wh ww) C", wh=wh, ww=ww
            ).squeeze(-1)                                       # (Mw, P)
            zw = F.normalize(zw, dim=-1)
            fw = F.normalize(fw, dim=-1)
            z_sim = torch.bmm(zw, zw.transpose(1, 2))           # (Mw, P, P)
            f_sim = torch.bmm(fw, fw.transpose(1, 2))
            pair_mask = mw.unsqueeze(2) * mw.unsqueeze(1)       # both valid in-window
            mdms_pair = F.relu((z_sim - f_sim).abs() - self.vf_margin_dist) * pair_mask
            loss_mdms = mdms_pair.sum() / pair_mask.sum().clamp_min(1.0)
        elif S and S < N:
            # Subsample S random VALID tokens per frame so the matrix is
            # (M, S, S) instead of (M, N, N). Sampling weight = validity; for an
            # all-invalid frame fall back to uniform and zero its loss. Sampling
            # with replacement is fine: duplicate (i,i) pairs have |Δcos|=0 and
            # are killed by the m2 margin, so they contribute nothing.
            probs = mf.float()
            empty = probs.sum(dim=1, keepdim=True) <= 0          # (M, 1)
            probs = probs + empty.float()                        # uniform if empty
            idx = torch.multinomial(probs, S, replacement=True)  # (M, S)
            gidx = idx.unsqueeze(-1).expand(-1, -1, zf.shape[-1])
            zf = torch.gather(zf, 1, gidx)                       # (M, S, C)
            ff = torch.gather(ff, 1, gidx)
            z_sim = torch.bmm(zf, zf.transpose(1, 2))            # (M, S, S)
            f_sim = torch.bmm(ff, ff.transpose(1, 2))
            mdms_pair = F.relu((z_sim - f_sim).abs() - self.vf_margin_dist)
            frame_w = (~empty).to(mdms_pair.dtype).view(-1, 1, 1)  # drop empty frames
            mdms_pair = mdms_pair * frame_w
            loss_mdms = mdms_pair.sum() / frame_w.expand_as(mdms_pair).sum().clamp_min(1.0)
        else:
            z_sim = torch.bmm(zf, zf.transpose(1, 2))           # (M, N, N) cos(z_i,z_j)
            f_sim = torch.bmm(ff, ff.transpose(1, 2))           # (M, N, N) cos(f_i,f_j)
            pair_mask = mf.unsqueeze(2) * mf.unsqueeze(1)       # (M, N, N) both valid
            mdms_pair = F.relu((z_sim - f_sim).abs() - self.vf_margin_dist) * pair_mask
            loss_mdms = mdms_pair.sum() / pair_mask.sum().clamp_min(1.0)

        return loss_mcos + loss_mdms

    def decode(
        self,
        z: torch.Tensor,
        hdmap_z: torch.Tensor | None = None,
        hdmap_raw: torch.Tensor | None = None,
        return_validity: bool = False,
        spatial_coords: torch.Tensor | None = None,
    ):
        """
        Args:
            z: (B, z_dim, T', H_z, W_z)
            hdmap_z: optional (B, z_dim, T', H_z, W_z) hdmap latent for the V1
                stem-concat injection. Caller is responsible for passing the
                already-encoded hdmap latent (typically obtained by running
                the raw hdmap through the same frozen ``self.encoder`` --
                see ``TransformerVAE.forward`` for the reference plumbing).
            hdmap_raw: optional (B, in_channels, T, H, W) raw hdmap video
                for the V3 per-up-level injections.
        Returns:
            (B, C, T, H, W) — cropped to original T if temporal padding was used
        """
        # post_quant_conv is 2D — apply per-frame
        B, Cz, Tz, Hz, Wz = z.shape
        z = einops.rearrange(z, "B C T H W -> (B T) C H W")
        z = self.post_quant_conv(z)
        z = einops.rearrange(z, "(B T) C H W -> B C T H W", B=B, T=Tz)

        coords = self._resolve_spatial_coords(
            z,
            spatial_coords,
            spatial_shape=getattr(
                self, "_input_spatial_shape", tuple(self.resolution)
            ),
        )
        out = self.decoder(
            z,
            coords,
            hdmap_z=hdmap_z,
            hdmap_raw=hdmap_raw,
        )

        # Crop to original temporal length if padding was applied during encoding
        if hasattr(self, "_input_T") and out.shape[2] > self._input_T:
            out = out[:, :, -self._input_T :]

        # Single-frame input: drop the T=1 axis so the output matches the 4D
        # (B, C, H, W) recon target produced by the image dataloader path.
        if getattr(self, "_input_T", None) == 1:
            out = out.squeeze(2)

        # Split off the validity-logit channel (last) when enabled. By default
        # decode returns only the data channels so all existing call sites
        # (inference, callbacks, DiT) are unaffected.
        if self.predict_validity:
            recon = out[:, : self.out_channels]
            validity_logits = out[:, self.out_channels :]
            if return_validity:
                return recon, validity_logits
            return recon
        if return_validity:
            return out, None
        return out

    def decode_stream(
        self,
        z_new: torch.Tensor,
        temporal_kv_cache: dict[
            str, tuple[torch.Tensor, torch.Tensor]
        ] | None = None,
        hdmap_z: torch.Tensor | None = None,
        hdmap_raw: torch.Tensor | None = None,
        return_validity: bool = False,
        max_cache_frames: int | None = None,
        spatial_coords: torch.Tensor | None = None,
    ):
        """Decode only newly appended latent frames using temporal KV caches.

        This path is exact for the 1x-temporal, attention-based causal decoder:
        spatial blocks and projections run only on ``z_new`` while every
        temporal-attention layer attends to its cached prefix plus the causal
        portion of this new block.
        """
        B, _, T_new, _, _ = z_new.shape
        z_new = einops.rearrange(
            z_new, "B C T H W -> (B T) C H W"
        )
        z_new = self.post_quant_conv(z_new)
        z_new = einops.rearrange(
            z_new, "(B T) C H W -> B C T H W", B=B, T=T_new
        )
        coords = self._resolve_spatial_coords(
            z_new,
            spatial_coords,
            spatial_shape=getattr(
                self, "_input_spatial_shape", tuple(self.resolution)
            ),
        )
        out, new_cache = self.decoder(
            z_new,
            coords,
            hdmap_z=hdmap_z,
            hdmap_raw=hdmap_raw,
            temporal_kv_cache=temporal_kv_cache,
            return_temporal_kv_cache=True,
            max_cache_frames=max_cache_frames,
        )

        if self.predict_validity:
            recon = out[:, : self.out_channels]
            validity_logits = out[:, self.out_channels :]
            if return_validity:
                return (recon, validity_logits), new_cache
            return recon, new_cache
        if return_validity:
            return (out, None), new_cache
        return out, new_cache

    @torch.no_grad()
    def _update_latent_stats_ema(self, z0: torch.Tensor) -> None:
        """In-place EMA update of ``latent_mean`` / ``latent_std`` from z0.

        ``z0`` is the raw encoder output (B, z_dim, T, H, W), BEFORE any
        normalization. We compute per-channel batch mean and std by
        reducing over (B, T, H, W), all-reduce across DDP ranks so every
        rank tracks the same world-wide stats, and then in-place EMA-
        update the persistent buffers with momentum
        ``self.latent_norm_noise_ema_momentum``.

        Notes
        -----
        * Run under ``torch.no_grad()`` -- the stats are normalization
          constants, not parameters; no gradient flows through them.
        * Uses ``var(unbiased=False)`` so the estimator matches an MLE
          (consistent with the empirical std used by
          ``scripts/compute_mean_std.py``).
        * The reduce shape ``(0, 2, 3, 4)`` keeps the channel axis,
          matching the registered buffer shape ``(1, z_dim, 1, 1, 1)``.
        * DDP all-reduce uses ``op=AVG`` so the global stat is the mean
          of the per-rank micro-batch stats; this is the right answer
          when batches per rank are equally sized (the usual case).
        """
        # Per-channel batch stats, shape (1, C, 1, 1, 1).
        reduce_dims = (0, 2, 3, 4)
        batch_mean = z0.mean(dim=reduce_dims, keepdim=True).float()
        batch_var = z0.var(dim=reduce_dims, keepdim=True, unbiased=False).float()

        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(batch_mean, op=dist.ReduceOp.AVG)
            dist.all_reduce(batch_var, op=dist.ReduceOp.AVG)

        batch_std = (batch_var + 1e-6).sqrt()

        m = self.latent_norm_noise_ema_momentum
        self.latent_mean.mul_(m).add_((1.0 - m) * batch_mean)
        self.latent_std.mul_(m).add_((1.0 - m) * batch_std)

    def forward(
        self,
        x: torch.Tensor,
        hdmap: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
        spatial_coords: torch.Tensor | None = None,
    ):
        """
        Args:
            x: (B, C, T, H, W)
            hdmap: optional (B, C, T, H, W) hdmap video at the same temporal
                length as ``x``. When the network was built with any of the
                hdmap injection flags (``hdmap_inject_stem`` /
                ``hdmap_inject_raw_at_up_levels``), this is the conditioning
                signal. When provided AND ``hdmap_inject_stem=True``, the
                hdmap is encoded through the same ``self.encoder`` /
                ``self.quant_conv`` (under ``torch.no_grad``) to produce
                ``hdmap_z`` aligned with ``z``; this is causally aligned by
                construction (same encoder = same temporal kernel + first-
                frame anchor convention). The raw hdmap is forwarded as-is
                to the decoder for V3 per-up-level injections. When ``hdmap``
                is None or all injection flags are off, this argument is
                silently ignored (preserves ``forward(x)`` backward
                compatibility for non-hdmap configs).
        """
        if self.mask_as_input:
            # Append the validity mask as an extra input channel so the
            # 2-channel [depth, mask] input matches the [depth_recon,
            # validity_logit] 2-channel output. Prefer an externally-provided
            # mask (``valid_mask``, computed by the dataloader BEFORE per-row
            # normalization) since deriving it from ``depth > min_value + eps``
            # is wrong once per-row z-scoring shifts the sentinel. Fall back to
            # the derived mask (same threshold the BCE loss uses) when no
            # explicit mask is supplied -> existing mask configs unchanged.
            if valid_mask is not None:
                valid = valid_mask.to(x.dtype)
            else:
                valid = (
                    x[:, :1] > (self.mask_input_min_value + self.mask_input_eps)
                ).to(x.dtype)
            x = torch.cat([x, valid], dim=1)

        if self.freeze_encoder:
            # Detach the encode pathway from autograd: posteriors/sample
            # carry no gradient, KL loss becomes a no-op w.r.t. params.
            with torch.no_grad():
                sample, posteriors = self.encode(x, spatial_coords=spatial_coords)
            sample = sample.detach()
            posteriors = tuple(p.detach() for p in posteriors)
        else:
            sample, posteriors = self.encode(x, spatial_coords=spatial_coords)

        # Encode the hdmap through the same encoder if any of the V1
        # injection paths is wired up. Always under ``torch.no_grad`` --
        # the typical use-case freezes the encoder, and even if it didn't
        # we don't want hdmap-encoding gradients flowing through and
        # changing the encoder's marginal. The ``encode()`` call mutates
        # ``self._input_T``, so save/restore around it to leave the lidar
        # path's value intact (decode() consumes ``_input_T`` for cropping).
        hdmap_z = None
        if hdmap is not None and self.hdmap_inject_stem:
            saved_input_T = getattr(self, "_input_T", None)
            with torch.no_grad():
                hdmap_z, _ = self.encode(hdmap, spatial_coords=spatial_coords)
            hdmap_z = hdmap_z.detach()
            if saved_input_T is not None:
                self._input_T = saved_input_T

        z = sample
        # Per-sample interpolative noise level (shape (B,)), exposed to the
        # loss so a temporal-smoothness term can ramp its weight with the
        # noise level. None when no interpolative noise was applied.
        noise_tau = None
        if self.training:
            # Online EMA update of (latent_mean, latent_std) on the RAW
            # encoder output. Runs before normalization so the buffers
            # track the encoder's true marginal -- not the post-normalize
            # tensor (which would always look ~N(0, 1) by construction
            # and tell us nothing).
            if (
                self.latent_norm_noise_ema_momentum > 0.0
                and self.has_latent_stats
                and self.latent_norm_noise_use_normalization
            ):
                self._update_latent_stats_ema(z)
            if self.latent_norm_noise_sigma > 0 and (
                self.has_latent_stats or not self.latent_norm_noise_use_normalization
            ):
                # Latent-space noise augmentation. Two variants:
                #   use_normalization=True  (default, requires latent_stats):
                #       encode -> normalize -> corrupt -> denormalize -> decode
                #   use_normalization=False (DeTok-pure):
                #       encode -> corrupt -> decode  (no normalize/denormalize)
                # The corruption formula is identical in both spaces; only the
                # reference frame differs.
                if self.latent_norm_noise_use_normalization:
                    latent_mean = self.latent_mean.to(z)
                    latent_std = self.latent_std.to(z)
                    z_work = (z - latent_mean) / latent_std
                else:
                    z_work = z

                if self.latent_norm_noise_sigma_mode == "interpolative":
                    # DeTok: z' = (1−τ)·z + τ·γ·ε, τ ~ U(0, τ_max) per sample.
                    # Full signal replacement at τ=1 prevents the additive
                    # shortcut. γ = self.latent_norm_noise_sigma.
                    # τ_max < 1 restricts to low-noise regime (LTX-style range).
                    tau_shape = (z_work.shape[0],) + (1,) * (z_work.dim() - 1)
                    tau = torch.rand(
                        tau_shape, device=z_work.device, dtype=z_work.dtype
                    ) * self.latent_norm_noise_tau_max
                    if self.latent_norm_noise_clean_prob > 0.0:
                        # Mixed clean / noisy regime: with probability
                        # `clean_prob`, force τ=0 for the sample so the
                        # decoder also sees clean latents during FT. Keeps
                        # the τ=0 manifold sharp while still training the
                        # low-τ denoising pathway.
                        clean_mask = torch.rand(
                            tau_shape, device=z_work.device, dtype=z_work.dtype
                        ) < self.latent_norm_noise_clean_prob
                        tau = torch.where(clean_mask, torch.zeros_like(tau), tau)
                    eps = self.latent_norm_noise_sigma * torch.randn_like(z_work)
                    z_work = (1.0 - tau) * z_work + tau * eps
                    noise_tau = tau.reshape(tau.shape[0])
                elif self.latent_norm_noise_sigma_mode == "uniform":
                    # Per-sample sigma ~ U(0, sigma_max). Shape (B,1,1,1,1)
                    # broadcasts over (C, T, H, W). Each batch element sees
                    # a different noise level; expected sigma is sigma_max/2.
                    sigma_shape = (z_work.shape[0],) + (1,) * (z_work.dim() - 1)
                    sigma = torch.rand(
                        sigma_shape, device=z_work.device, dtype=z_work.dtype
                    ) * self.latent_norm_noise_sigma
                    z_work = z_work + sigma * torch.randn_like(z_work)
                else:  # "fixed"
                    z_work = z_work + self.latent_norm_noise_sigma * torch.randn_like(z_work)

                if self.latent_norm_noise_use_normalization:
                    z = z_work * latent_std + latent_mean
                else:
                    z = z_work
            elif self.noise_aug_sigma > 0:
                z = z + self.noise_aug_sigma * torch.randn_like(z)

            # PV-VAE predictive frame dropping: replace trailing latent
            # frames with the learnable pad token (per sample). Applied
            # AFTER noise so pad tokens are never noised. Decode then
            # reconstructs the full video, predicting the withheld future.
            if self.frame_drop_prob > 0.0 and self.frame_drop_max_latent > 0:
                Bz, _, Tlat = z.shape[0], z.shape[1], z.shape[2]
                # Keep at least 1 latent frame; cap the drop at Tlat - 1.
                max_k = min(self.frame_drop_max_latent, Tlat - 1)
                if max_k >= self.frame_drop_min_latent:
                    do_drop = (
                        torch.rand(Bz, device=z.device) < self.frame_drop_prob
                    )
                    k = torch.randint(
                        self.frame_drop_min_latent, max_k + 1,
                        (Bz,), device=z.device,
                    )
                    keep = torch.where(
                        do_drop, Tlat - k, torch.full_like(k, Tlat)
                    )  # (B,) number of leading latents to keep
                    pos = torch.arange(Tlat, device=z.device)[None, :]
                    replace = (pos >= keep[:, None])  # (B, Tlat) True => pad
                    replace = replace[:, None, :, None, None]
                    z = torch.where(replace, self.frame_drop_pad.to(z.dtype), z)

        # Pass hdmap through to decode for V1 (hdmap_z) and/or V3 (hdmap_raw)
        # injection paths. Both args are silently ignored by the decoder when
        # the corresponding injection flag is off, so this is safe for the
        # default no-hdmap configuration.
        decoded = self.decode(
            z,
            hdmap_z=hdmap_z,
            hdmap_raw=hdmap,
            return_validity=self.predict_validity,
            spatial_coords=spatial_coords,
        )
        if self.predict_validity:
            recon, validity_logits = decoded
        else:
            recon, validity_logits = decoded, None
        if self.training:
            out_dict = dict(
                reconstructions=recon,
                posteriors=posteriors,
                latent=sample,
            )
            if noise_tau is not None:
                out_dict["noise_tau"] = noise_tau
            if validity_logits is not None:
                out_dict["validity_logits"] = validity_logits
            # VA-VAE VF-loss alignment on the CLEAN encoder latent (encoder-side
            # regularization; the noised/frame-dropped ``z`` is a decoder trick).
            if self.utonia_align:
                out_dict["repa_align"] = self._utonia_vf_loss(sample, x)
            return out_dict
        return NetworkEval(
            reconstructions=recon,
            posteriors=posteriors,
            latent=sample,
            validity_logits=validity_logits,
        )


# =========================================================================
# Model factory helpers & registry
# =========================================================================


def _make_tvae(defaults: dict, **kwargs):
    """Merge caller kwargs on top of preset defaults, then build TransformerVAE."""
    for d in (defaults, kwargs):
        if "patch_size" in d:
            ps = d["patch_size"]
            if isinstance(ps, int):
                d["patch_size"] = (ps, ps)
            elif isinstance(ps, list):
                d["patch_size"] = tuple(ps)
    defaults.update(kwargs)
    return TransformerVAE(**defaults)


def TransformerVAE_S(**kwargs):
    return _make_tvae(dict(
        base_channels=64,
        depths=[2, 2, 2, 2],
        num_heads=[2, 4, 4, 8],
        temporal_downsample=[True, False, False],
        mlp_ratio=3.0,
        mapping_depth=2,
    ), **kwargs)


def TransformerVAE_B(**kwargs):
    return _make_tvae(dict(
        base_channels=128,
        depths=[3, 3, 3, 3],
        num_heads=[2, 4, 8, 16],
        temporal_downsample=[True, False, False],
        mlp_ratio=3.0,
        mapping_depth=2,
    ), **kwargs)


def TransformerVAE_L(**kwargs):
    return _make_tvae(dict(
        base_channels=192,
        depths=[4, 4, 4, 4],
        num_heads=[3, 6, 12, 24],
        temporal_downsample=[True, False, False],
        mlp_ratio=3.0,
        mapping_depth=2,
    ), **kwargs)


TransformerVAE_models = {
    "TransformerVAE-S": TransformerVAE_S,
    "TransformerVAE-B": TransformerVAE_B,
    "TransformerVAE-L": TransformerVAE_L,
}
