# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optional CUDA extensions and their PyTorch implementations.

Transformer Engine, NATTEN and Apex are compiled against one PyTorch/CUDA pair
and are not on PyPI as wheels, so `uv sync` does not install them. The networks
take the pieces they use from here: the extension when it imports, otherwise a
PyTorch implementation of the same operation that loads the same checkpoints.
That is also what runs on GPUs an extension build has no kernels for, e.g.
Blackwell workstation cards (RTX PRO 6000, sm_120).

`LIDARVERSE_DISABLE_EXTENSIONS=te,natten,apex` (or `all`) forces the PyTorch
paths.
"""

from __future__ import annotations

import os
import warnings
from functools import lru_cache

import torch
import torch.nn.functional as F
from torch import nn

_DISABLED = {name.strip() for name in os.environ.get("LIDARVERSE_DISABLE_EXTENSIONS", "").split(",") if name.strip()}


def _enabled(name: str) -> bool:
    return not ({name, "all"} & _DISABLED)


# ----------------------------------------------------------------------------
# Transformer Engine: RMSNorm and rotary position embedding
# ----------------------------------------------------------------------------

te = None
_te_apply_rotary_pos_emb = None
if _enabled("te"):
    try:
        import transformer_engine as te  # noqa: F811
        import transformer_engine.pytorch  # noqa: F401

        try:
            from transformer_engine.pytorch.attention.rope import apply_rotary_pos_emb as _te_apply_rotary_pos_emb
        except ImportError:
            from transformer_engine.pytorch.attention import apply_rotary_pos_emb as _te_apply_rotary_pos_emb
    except (ImportError, OSError, RuntimeError):
        te = None
HAS_TE = te is not None


class TorchRMSNorm(nn.Module):
    """``transformer_engine.pytorch.RMSNorm`` (zero_centered_gamma=False):
    statistics in FP32, output in the autocast dtype when autocast is on (as
    Transformer Engine's layers compute), else in the input dtype, a ``weight``
    parameter."""

    def __init__(self, normalized_shape: int, eps: float = 1e-5, device=None, dtype=None, **_):
        super().__init__()
        self.normalized_shape = (normalized_shape,) if isinstance(normalized_shape, int) else tuple(normalized_shape)
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(self.normalized_shape, device=device, dtype=dtype))

    def reset_parameters(self) -> None:
        nn.init.ones_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dims = tuple(range(-len(self.normalized_shape), 0))
        x32 = x.float()
        y = x32 * torch.rsqrt(x32.pow(2).mean(dim=dims, keepdim=True) + self.eps)
        # The networks' Wan FP32 blocks (`amp.autocast(dtype=float32)` around the
        # timestep embedding) assert an FP32 result, which TE gives there.
        device = x.device.type
        if torch.is_autocast_enabled(device):
            dtype = torch.get_autocast_dtype(device)
        else:
            dtype = x.dtype
        return (y * self.weight.float()).to(dtype)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # Checkpoints written with Transformer Engine carry its (FP8) extra state.
        state_dict.pop(prefix + "_extra_state", None)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def extra_repr(self) -> str:
        return f"{self.normalized_shape}, eps={self.eps}"


def RMSNorm(normalized_shape: int, eps: float = 1e-5, **kwargs) -> nn.Module:
    if HAS_TE:
        return te.pytorch.RMSNorm(normalized_shape, eps=eps, **kwargs)
    return TorchRMSNorm(normalized_shape, eps=eps, **kwargs)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def torch_apply_rotary_pos_emb(t: torch.Tensor, freqs: torch.Tensor, tensor_format: str = "sbhd", **_) -> torch.Tensor:
    """Transformer Engine's non-interleaved RoPE: ``freqs`` is ``[s, 1, 1, d]``
    angles; the first ``d`` channels of each head are rotated, in FP32."""
    if tensor_format not in ("sbhd", "bshd"):
        raise ValueError(f"unsupported RoPE tensor format {tensor_format!r}")
    seq_dim = 0 if tensor_format == "sbhd" else 1
    freqs = freqs[: t.shape[seq_dim]]
    if tensor_format == "bshd":
        freqs = freqs.transpose(0, 1)  # [1, s, 1, d]
    rot_dim = freqs.shape[-1]
    rotated, passthrough = t[..., :rot_dim], t[..., rot_dim:]
    freqs = freqs.float()
    out = rotated.float() * freqs.cos() + _rotate_half(rotated.float()) * freqs.sin()
    out = out.to(t.dtype)
    return torch.cat((out, passthrough), dim=-1) if passthrough.numel() else out


def apply_rotary_pos_emb(t: torch.Tensor, freqs: torch.Tensor, tensor_format: str = "sbhd", fused: bool = False, **kwargs):
    if _te_apply_rotary_pos_emb is not None:
        return _te_apply_rotary_pos_emb(t, freqs, tensor_format=tensor_format, fused=fused, **kwargs)
    return torch_apply_rotary_pos_emb(t, freqs, tensor_format=tensor_format)


def DotProductAttention(*args, **kwargs):
    if not HAS_TE:
        raise ImportError(
            "attention backend 'transformer_engine' needs Transformer Engine "
            "(uv sync --extra te); the released checkpoints use 'torch-flex' / "
            "'minimal_a2a', which do not")
    from transformer_engine.pytorch.attention import DotProductAttention as _DotProductAttention

    return _DotProductAttention(*args, **kwargs)


# ----------------------------------------------------------------------------
# NATTEN: 2-D neighborhood attention
# ----------------------------------------------------------------------------

natten = None
if _enabled("natten"):
    try:
        import natten  # noqa: F811
        import natten.functional  # noqa: F401
    except (ImportError, OSError, RuntimeError):
        natten = None
HAS_NATTEN = natten is not None
_natten_failed = False


@lru_cache(maxsize=64)
def _neighborhood_index(length: int, kernel: int, dilation: int, device: torch.device) -> torch.Tensor:
    """(length, kernel) key positions of each query, as NATTEN defines them:
    the window is clamped (not cropped) at the borders, per dilation group."""
    i = torch.arange(length, device=device)
    group, position = i % dilation, i // dilation
    group_length = (length - group + dilation - 1) // dilation
    if bool((group_length < kernel).any()):
        raise ValueError(f"neighborhood kernel {kernel} x dilation {dilation} exceeds length {length}")
    start = torch.minimum((position - kernel // 2).clamp(min=0), group_length - kernel)
    return group[:, None] + (start[:, None] + torch.arange(kernel, device=device)) * dilation


def torch_na2d(query, key, value, kernel_size, dilation=(1, 1), scale=None, max_chunk_elements=1 << 27):
    """``natten.functional.na2d`` (non-causal, no bias) in PyTorch.

    query/key/value: ``(B, H, W, heads, D)``. Keys of each query are gathered
    explicitly and rows are processed in chunks to bound memory."""
    B, H, W, N, D = query.shape
    kh, kw = kernel_size
    dh, dw = dilation
    scale = D ** -0.5 if scale is None else scale
    index_h = _neighborhood_index(H, kh, dh, query.device)
    index_w = _neighborhood_index(W, kw, dw, query.device)
    per_row = B * W * kh * kw * N * D
    rows = max(1, min(H, max_chunk_elements // max(per_row, 1)))
    out = torch.empty_like(query)
    for start in range(0, H, rows):
        rows_h = index_h[start:start + rows]  # (r, kh)
        k = key[:, rows_h][:, :, :, index_w]  # (B, r, kh, W, kw, N, D)
        v = value[:, rows_h][:, :, :, index_w]
        q = query[:, start:start + rows]  # (B, r, W, N, D)
        logits = torch.einsum("brwnd,brawcnd->brwnac", q.float(), k.float()) * scale
        weights = logits.flatten(-2).softmax(-1).view_as(logits)
        out[:, start:start + rows] = torch.einsum("brwnac,brawcnd->brwnd", weights, v.float()).to(query.dtype)
    return out


def na2d(query, key, value, kernel_size, dilation=(1, 1), scale=None, **natten_kwargs):
    """NATTEN when it runs on this GPU; otherwise :func:`torch_na2d`."""
    global _natten_failed
    if HAS_NATTEN and not _natten_failed:
        try:
            return natten.functional.na2d(query=query, key=key, value=value, kernel_size=kernel_size,
                                          dilation=dilation, scale=scale, **natten_kwargs)
        except (RuntimeError, NotImplementedError, ValueError) as error:
            _natten_failed = True
            warnings.warn(f"NATTEN failed ({error}); using the PyTorch neighborhood attention", stacklevel=2)
    return torch_na2d(query, key, value, tuple(kernel_size), tuple(dilation), scale)


# ----------------------------------------------------------------------------
# Apex: multi-tensor L2 norm / scale (gradient clipping)
# ----------------------------------------------------------------------------

multi_tensor_applier = None
amp_C = None
if _enabled("apex"):
    try:
        import amp_C  # noqa: F811
        from apex.multi_tensor_apply import multi_tensor_applier  # noqa: F811
    except (ImportError, OSError):
        amp_C = multi_tensor_applier = None
HAS_APEX = amp_C is not None


def _torch_l2norm(_overflow, tensor_lists, per_tensor=False):
    norms = torch.stack([torch.linalg.vector_norm(t.float()) for t in tensor_lists[0]])
    total = torch.linalg.vector_norm(norms).reshape(1)
    return total, (norms if per_tensor else None)


def _torch_scale(_overflow, tensor_lists, scale):
    sources, targets = tensor_lists[0], tensor_lists[1]
    for source, target in zip(sources, targets):
        target.copy_(source * scale)


if HAS_APEX:
    l2_norm_impl = amp_C.multi_tensor_l2norm
    multi_tensor_scale_impl = amp_C.multi_tensor_scale
else:
    l2_norm_impl = _torch_l2norm
    multi_tensor_scale_impl = _torch_scale

    def multi_tensor_applier(op, noop_flag_buffer, tensor_lists, *args):  # noqa: F811
        return op(noop_flag_buffer, tensor_lists, *args)


# ----------------------------------------------------------------------------
# GPU properties
# ----------------------------------------------------------------------------


@lru_cache(maxsize=None)
def flex_attention_kernel_options(device_index: int | None = None) -> dict | None:
    """FlexAttention tiles that fit the GPU's shared memory.

    PyTorch picks its Hopper tiles for every GPU of compute capability >= 9.0,
    which need more than the ~100 KB per block of Blackwell workstation and Ada
    / Ampere consumer GPUs (sm_120, sm_89, sm_86): Triton then fails with
    "out of resource: shared memory". Smaller tiles run everywhere."""
    if not torch.cuda.is_available():
        return None
    props = torch.cuda.get_device_properties(device_index if device_index is not None else torch.cuda.current_device())
    shared = getattr(props, "shared_memory_per_block_optin", 0)
    if shared and shared >= 160 * 1024:
        return None  # A100 / H100 / B200: PyTorch's defaults
    return {"BLOCK_M": 64, "BLOCK_N": 64, "num_stages": 2, "num_warps": 4}
