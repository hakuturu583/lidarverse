# -----------------------------------------------------------------------------
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# This codebase constitutes NVIDIA proprietary technology and is strictly
# confidential. Any unauthorized reproduction, distribution, or disclosure
# of this code, in whole or in part, outside NVIDIA is strictly prohibited
# without prior written consent.
#
# For inquiries regarding the use of this code in other NVIDIA proprietary
# projects, please contact the Deep Imagination Research Team at
# dir@exchange.nvidia.com.
# -----------------------------------------------------------------------------

import collections
import math
from collections import namedtuple
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple, Union

from diffusion.training.networks.kv_cache import (
    AttentionOpWithKVCache,
    KVCacheConfig,
    select_kv_cache_tokens,
)

try:
    import megatron.core.parallel_state as parallel_state

    USE_MEGATRON = True
except ImportError:
    USE_MEGATRON = False

import numpy as np
import torch
import torch.amp as amp
import transformer_engine as te
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import nn
from torch.distributed import ProcessGroup, get_process_group_ranks
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper as ptd_checkpoint_wrapper

try:
    from torch.utils.checkpoint import CheckpointPolicy, create_selective_checkpoint_contexts
except ImportError:
    CheckpointPolicy = None

from torchvision import transforms

_FLASH_ATTN_3_IMPORT_ERROR = None
try:
    try:
        from flash_attn_interface import flash_attn_func as _flash_attn_3_func
    except (ImportError, OSError):
        from flash_attn_3.flash_attn_interface import (
            flash_attn_func as _flash_attn_3_func,
        )
except (ImportError, OSError) as exc:
    _flash_attn_3_func = None
    _FLASH_ATTN_3_IMPORT_ERROR = exc

try:
    from transformer_engine.pytorch.attention.rope import apply_rotary_pos_emb
except ImportError:
    from transformer_engine.pytorch.attention import apply_rotary_pos_emb
from torch.nn.attention.flex_attention import BlockMask, create_block_mask, flex_attention

from diffusion.training.networks.cosmos_attention import attention as i4_attention_fn
from diffusion.training.context_parallel import split_inputs_cp
from utils import log
from diffusion.conditioner import DataType
from diffusion.training.networks.neighborhood_attn import NeighborhoodAttention
from diffusion.training.networks.a2a_cp import MinimalA2AAttnOp, NattenA2AAttnOp
from diffusion.training.networks.model_weights_stats import WeightTrainingStat
from diffusion.training.networks.selective_activation_checkpoint import SACConfig as _SACConfig
from diffusion.training.utils.diffusion_forcing import (
    teacher_forcing_clean_noisy_allows,
    temporal_prefix_block_allows,
)


# selective activation checkpoint; only apply to the minimal v4 model. if there are change in the networks, some policy will not work as we expect.
def predict2_2B_720_context_fn():
    op_count = collections.defaultdict(int)

    def policy_fn(ctx, func, *args, **kwargs):
        mode = "recompute" if ctx.is_recompute else "forward"
        if func == torch.ops.aten.mm.default:
            op_count_key = f"{mode}_mm_count"
            op_count[op_count_key] = (op_count[op_count_key] + 1) % 16
            if op_count[op_count_key] > 8:  # recompute self attn first 3 linear layers
                return CheckpointPolicy.MUST_SAVE
        if "flash_attn" in str(func):
            op_count_key = f"{mode}_flash_attn_count"
            op_count[op_count_key] = (op_count[op_count_key] + 1) % 2
            if op_count[op_count_key]:
                return CheckpointPolicy.MUST_SAVE
        return CheckpointPolicy.PREFER_RECOMPUTE

    return create_selective_checkpoint_contexts(policy_fn)


def predict2_2B_720_context_fn_aggressive():
    op_count = collections.defaultdict(int)

    def policy_fn(ctx, func, *args, **kwargs):
        # The default policy is to recompute everything. This is the most memory-efficient
        # starting point. We then selectively choose what to save.
        default_policy = CheckpointPolicy.PREFER_RECOMPUTE

        # Save the output of Flash Attention. This is the most computationally
        # expensive part of a transformer block. Saving its output provides a
        # good balance between memory savings and computational overhead.
        if "flash_attn" in str(func):
            return CheckpointPolicy.MUST_SAVE

        # All other operations (e.g., torch.ops.aten.mm.default, layer norms, additions)
        # will fall through to the default policy and be recomputed.
        return default_policy

    return create_selective_checkpoint_contexts(policy_fn)


def predict2_2B_720_context_fn_aggressive_v2():
    """
    The most memory-aggressive checkpointing policy. Recomputes ALL operations.
    """

    def policy_fn(ctx, func, *args, **kwargs):
        # The policy is to always recompute everything.
        # This saves the maximum amount of memory but incurs the highest
        # computational cost during the backward pass.
        return CheckpointPolicy.PREFER_RECOMPUTE

    return create_selective_checkpoint_contexts(policy_fn)


def predict2_14B_720_context_fn():
    op_count = collections.defaultdict(int)

    def policy_fn(ctx, func, *args, **kwargs):
        mode = "recompute" if ctx.is_recompute else "forward"
        if func == torch.ops.aten.mm.default:
            op_count_key = f"{mode}_mm_count"
            op_count[op_count_key] = (op_count[op_count_key] + 1) % 16
            if op_count[op_count_key] > 8:  # recompute self attn first 1 linear layers
                return CheckpointPolicy.MUST_SAVE
        if "flash_attn" in str(func):
            op_count_key = f"{mode}_flash_attn_count"
            op_count[op_count_key] = (op_count[op_count_key] + 1) % 2
            if op_count[op_count_key]:
                return CheckpointPolicy.MUST_SAVE
        return CheckpointPolicy.PREFER_RECOMPUTE

    return create_selective_checkpoint_contexts(policy_fn)


def predict2_14B_720_context_fn_aggressive():
    op_count = collections.defaultdict(int)

    def policy_fn(ctx, func, *args, **kwargs):
        mode = "recompute" if ctx.is_recompute else "forward"
        if func == torch.ops.aten.mm.default:
            op_count_key = f"{mode}_mm_count"
            op_count[op_count_key] = (op_count[op_count_key] + 1) % 16
            if op_count[op_count_key] > 12:  # recompute self attn first 1 linear layers
                return CheckpointPolicy.MUST_SAVE
        return CheckpointPolicy.PREFER_RECOMPUTE

    return create_selective_checkpoint_contexts(policy_fn)


def linear_selfattn_context_fn():
    op_count = collections.defaultdict(int)

    def policy_fn(ctx, func, *args, **kwargs):
        mode = "recompute" if ctx.is_recompute else "forward"
        if func == torch.ops.aten.mm.default:
            return CheckpointPolicy.MUST_SAVE
        if "flash_attn" in str(func):
            op_count_key = f"{mode}_flash_attn_count"
            op_count[op_count_key] = (op_count[op_count_key] + 1) % 2
            if op_count[op_count_key]:
                return CheckpointPolicy.MUST_SAVE
        return CheckpointPolicy.PREFER_RECOMPUTE

    return create_selective_checkpoint_contexts(policy_fn)


class CheckpointMode(str, Enum):
    NONE = "none"
    MM_ONLY = "mm_only"
    BLOCK_WISE = "block_wise"
    LINEAR_SELFATTN = "linear_selfattn"
    PREDICT2_2B_720 = "predict2_2b_720"
    PREDICT2_14B_720 = "predict2_14b_720"
    PREDICT2_2B_720_AGGRESSIVE = "predict2_2b_720_aggressive"
    PREDICT2_2B_720_AGGRESSIVE_V2 = "predict2_2b_720_aggressive_v2"
    PREDICT2_14B_720_AGGRESSIVE = "predict2_14b_720_aggressive"

    def __str__(self) -> str:
        return self.value


@dataclass
class SACConfig(_SACConfig):
    def get_context_fn(self):
        if self.mode == CheckpointMode.LINEAR_SELFATTN:
            return linear_selfattn_context_fn
        elif self.mode == CheckpointMode.PREDICT2_2B_720:
            return predict2_2B_720_context_fn
        elif self.mode == CheckpointMode.PREDICT2_2B_720_AGGRESSIVE:
            return predict2_2B_720_context_fn_aggressive
        elif self.mode == CheckpointMode.PREDICT2_2B_720_AGGRESSIVE_V2:
            return predict2_2B_720_context_fn_aggressive_v2
        elif self.mode == CheckpointMode.PREDICT2_14B_720:
            return predict2_14B_720_context_fn
        elif self.mode == CheckpointMode.PREDICT2_14B_720_AGGRESSIVE:
            return predict2_14B_720_context_fn_aggressive
        else:
            # Reuse parent class implementation for other modes
            return super().get_context_fn()


VideoSize = namedtuple("VideoSize", ["T", "H", "W"])


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


# ---------------------- Feed Forward Network -----------------------
class GPT2FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.activation = nn.GELU()
        self.layer1 = nn.Linear(d_model, d_ff, bias=False)
        self.layer2 = nn.Linear(d_ff, d_model, bias=False)

        self._layer_id = None
        self._dim = d_model
        self._hidden_dim = d_ff
        self.init_weights()

    def init_weights(self) -> None:
        std = 1.0 / math.sqrt(self._dim)
        torch.nn.init.trunc_normal_(self.layer1.weight, std=std, a=-3 * std, b=3 * std)

        # scale init by depth as in https://arxiv.org/abs/1908.11365 -- worked slightly better.
        std = 1.0 / math.sqrt(self._hidden_dim)
        if self._layer_id is not None:
            std = std / math.sqrt(2 * (self._layer_id + 1))
        torch.nn.init.trunc_normal_(self.layer2.weight, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: torch.Tensor):
        x = self.layer1(x)

        x = self.activation(x)
        x = self.layer2(x)
        return x


def torch_attention_op(
    q_B_S_H_D: torch.Tensor,
    k_B_S_H_D: torch.Tensor,
    v_B_S_H_D: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    flatten_heads: bool = True,
) -> torch.Tensor:
    """Scaled dot-product attention with optional mask.

    Inputs are shaped [B, S, H, D]. If flatten_heads=True, flattens heads to return [B, S, H*D].
    Otherwise returns [B, S, H, D].
    """
    q_B_H_S_D = rearrange(q_B_S_H_D, "b s h d -> b h s d")
    k_B_H_S_D = rearrange(k_B_S_H_D, "b s h d -> b h s d")
    v_B_H_S_D = rearrange(v_B_S_H_D, "b s h d -> b h s d")
    result_B_H_S_D = torch.nn.functional.scaled_dot_product_attention(
        q_B_H_S_D, k_B_H_S_D, v_B_H_S_D, attn_mask=attn_mask
    )
    if flatten_heads:
        return rearrange(result_B_H_S_D, "b h s d -> b s (h d)")
    else:
        return rearrange(result_B_H_S_D, "b h s d -> b s h d")


def torch_flash_attention_op(
    q_B_S_H_D: torch.Tensor,
    k_B_S_H_D: torch.Tensor,
    v_B_S_H_D: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    flatten_heads: bool = True,
) -> torch.Tensor:
    """Call PyTorch's fused FlashAttention operator directly.

    This is intended for cached inference, where each call receives only past
    K/V plus one complete bidirectional DF block and therefore needs no mask.
    Calling the operator directly avoids the process-wide ``sdpa_kernel``
    policy context, which would otherwise leak into concurrently running VAE
    attention on the asynchronous clean-commit path.
    """
    if attn_mask is not None:
        raise ValueError(
            "forced SDPA FlashAttention does not support the DF block mask; "
            "use it only for cached inference"
        )
    q_B_H_S_D = rearrange(q_B_S_H_D, "b s h d -> b h s d")
    k_B_H_S_D = rearrange(k_B_S_H_D, "b s h d -> b h s d")
    v_B_H_S_D = rearrange(v_B_S_H_D, "b s h d -> b h s d")
    result_B_H_S_D = torch.ops.aten._scaled_dot_product_flash_attention.default(
        q_B_H_S_D,
        k_B_H_S_D,
        v_B_H_S_D,
        dropout_p=0.0,
        is_causal=False,
        return_debug_mask=False,
    )[0]
    if flatten_heads:
        return rearrange(result_B_H_S_D, "b h s d -> b s (h d)")
    return rearrange(result_B_H_S_D, "b h s d -> b s h d")


def flash_attention_3_op(
    q_B_S_H_D: torch.Tensor,
    k_B_S_H_D: torch.Tensor,
    v_B_S_H_D: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    flatten_heads: bool = True,
) -> torch.Tensor:
    """Call the external FlashAttention-3 Hopper kernel for cached inference."""
    if _flash_attn_3_func is None:
        raise ImportError(
            "cached backend 'fa3' requires flash-attn-3; install the Hopper "
            "package from Dao-AILab/flash-attention"
        ) from _FLASH_ATTN_3_IMPORT_ERROR
    if attn_mask is not None:
        raise ValueError(
            "FlashAttention-3 does not support the DF block mask in this path; "
            "use it only for cached inference"
        )
    if not q_B_S_H_D.is_cuda or torch.cuda.get_device_capability(
        q_B_S_H_D.device
    ) != (9, 0):
        raise RuntimeError("cached backend 'fa3' requires an SM90 Hopper GPU")
    if q_B_S_H_D.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError(
            "cached backend 'fa3' requires FP16 or BF16 Q/K/V, got "
            f"{q_B_S_H_D.dtype}"
        )
    result = _flash_attn_3_func(
        q=q_B_S_H_D,
        k=k_B_S_H_D,
        v=v_B_S_H_D,
        softmax_scale=None,
        causal=False,
        deterministic=False,
    )
    result_B_S_H_D = result[0] if isinstance(result, tuple) else result
    if flatten_heads:
        return rearrange(result_B_S_H_D, "b s h d -> b s (h d)")
    return result_B_S_H_D


def flex_attention_op(
    q_B_S_H_D: torch.Tensor,
    k_B_S_H_D: torch.Tensor,
    v_B_S_H_D: torch.Tensor,
    attn_mask: Optional[BlockMask] = None,
    flatten_heads: bool = True,
) -> torch.Tensor:
    # Rearrange to [B, H, S, D]
    q_B_H_Sq_D = rearrange(q_B_S_H_D, "b s h d -> b h s d")
    k_B_H_Sk_D = rearrange(k_B_S_H_D, "b s h d -> b h s d")
    v_B_H_Sk_D = rearrange(v_B_S_H_D, "b s h d -> b h s d")

    S_q = q_B_H_Sq_D.shape[2]
    S_kv = k_B_H_Sk_D.shape[2]
    # Right-pad to multiples of 128 for optimal FlexAttention kernels
    pad_q = ((S_q + 127) // 128) * 128 - S_q
    pad_kv = ((S_kv + 127) // 128) * 128 - S_kv

    if pad_q > 0:
        q_pad_tensor = torch.zeros(
            (q_B_H_Sq_D.shape[0], q_B_H_Sq_D.shape[1], pad_q, q_B_H_Sq_D.shape[3]),
            device=q_B_H_Sq_D.device,
            dtype=q_B_H_Sq_D.dtype,
        )
        q_cat = torch.cat([q_B_H_Sq_D, q_pad_tensor], dim=2)
    else:
        q_cat = q_B_H_Sq_D

    if pad_kv > 0:
        kv_pad_tensor = torch.zeros(
            (k_B_H_Sk_D.shape[0], k_B_H_Sk_D.shape[1], pad_kv, k_B_H_Sk_D.shape[3]),
            device=k_B_H_Sk_D.device,
            dtype=k_B_H_Sk_D.dtype,
        )
        k_cat = torch.cat([k_B_H_Sk_D, kv_pad_tensor], dim=2)
        v_cat = torch.cat([v_B_H_Sk_D, kv_pad_tensor], dim=2)
    else:
        k_cat, v_cat = k_B_H_Sk_D, v_B_H_Sk_D

    block_mask = None
    if attn_mask is not None and isinstance(attn_mask, BlockMask):
        block_mask = attn_mask
    else:
        # When padding is introduced without an explicit mask, build a validity mask
        if pad_q > 0 or pad_kv > 0:

            def allow_valid(b, h, q_idx, kv_idx):
                return (q_idx < S_q) & (kv_idx < S_kv)

            block_mask = create_block_mask(
                allow_valid,
                B=None,
                H=None,
                Q_LEN=q_cat.shape[2],
                KV_LEN=k_cat.shape[2],
                _compile=True,
                device=q_cat.device,
            )

    if block_mask is not None:
        out_B_H_Sqp_D = torch.compile(flex_attention)(query=q_cat, key=k_cat, value=v_cat, block_mask=block_mask)
    else:
        out_B_H_Sqp_D = torch.compile(flex_attention)(query=q_cat, key=k_cat, value=v_cat)

    out_B_H_Sq_D = out_B_H_Sqp_D[:, :, :S_q] if pad_q > 0 else out_B_H_Sqp_D
    if flatten_heads:
        return rearrange(out_B_H_Sq_D, "b h s d -> b s (h d)")
    else:
        return rearrange(out_B_H_Sq_D, "b h s d -> b s h d")


def i4_attention_op(
    q_B_S_H_D: torch.Tensor,
    k_B_S_H_D: torch.Tensor,
    v_B_S_H_D: torch.Tensor,
    flatten_heads: bool = True,
    **kwargs: dict,
) -> torch.Tensor:
    """
    I4 regular (bidirectional) attention.
    Matches torch_attention_op's signature but omits attn_mask (full attention assumed).
    Ignores any additional kwargs (e.g., video_size).
    """
    out_B_S_H_D = i4_attention_fn(
        query=q_B_S_H_D,
        key=k_B_S_H_D,
        value=v_B_S_H_D,
        is_causal=False,
    )
    if isinstance(out_B_S_H_D, tuple):
        out_B_S_H_D = out_B_S_H_D[0]
    if flatten_heads:
        return rearrange(out_B_S_H_D, "b s h d -> b s (h d)")
    else:
        return out_B_S_H_D


class Attention(nn.Module):
    """
    A flexible attention module supporting both self-attention and cross-attention mechanisms.

    This module implements a multi-head attention layer that can operate in either self-attention
    or cross-attention mode. The mode is determined by whether a context dimension is provided.
    The implementation uses scaled dot-product attention and supports optional bias terms and
    dropout regularization.

    Args:
        query_dim (int): The dimensionality of the query vectors.
        context_dim (int, optional): The dimensionality of the context (key/value) vectors.
            If None, the module operates in self-attention mode using query_dim. Default: None
        n_heads (int, optional): Number of attention heads for multi-head attention. Default: 8
        head_dim (int, optional): The dimension of each attention head. Default: 64
        dropout (float, optional): Dropout probability applied to the output. Default: 0.0
        qkv_format (str, optional): Format specification for QKV tensors. Default: "bshd"
        backend (str, optional): Backend to use for the attention operation. Default: "transformer_engine"

    Examples:
        >>> # Self-attention with 512 dimensions and 8 heads
        >>> self_attn = Attention(query_dim=512)
        >>> x = torch.randn(32, 16, 512)  # (batch_size, seq_len, dim)
        >>> out = self_attn(x)  # (32, 16, 512)

        >>> # Cross-attention
        >>> cross_attn = Attention(query_dim=512, context_dim=256)
        >>> query = torch.randn(32, 16, 512)
        >>> context = torch.randn(32, 8, 256)
        >>> out = cross_attn(query, context)  # (32, 16, 512)
    """

    def __init__(
        self,
        query_dim: int,
        context_dim=None,
        n_heads=8,
        head_dim=64,
        dropout=0.0,
        qkv_format: str = "bshd",
        backend: str = "transformer_engine",
        use_wan_fp32_strategy: bool = False,
    ) -> None:
        super().__init__()
        log.debug(
            f"Setting up {self.__class__.__name__}. Query dim is {query_dim}, context_dim is {context_dim} and using "
            f"{n_heads} heads with a dimension of {head_dim}."
        )
        self.is_selfattn = context_dim is None  # self attention

        assert backend in ["transformer_engine", "torch", "torch-flex", "minimal_a2a", "i4"], (
            f"Invalid backend: {backend}"
        )
        self.backend = backend

        context_dim = query_dim if context_dim is None else context_dim
        inner_dim = head_dim * n_heads

        self.n_heads = n_heads
        self.head_dim = head_dim
        self.qkv_format = qkv_format
        self.query_dim = query_dim
        self.context_dim = context_dim
        self.use_wan_fp32_strategy = use_wan_fp32_strategy

        self.q_proj = nn.Linear(query_dim, inner_dim, bias=False)
        self.q_norm = te.pytorch.RMSNorm(self.head_dim, eps=1e-6)

        self.k_proj = nn.Linear(context_dim, inner_dim, bias=False)
        self.k_norm = te.pytorch.RMSNorm(self.head_dim, eps=1e-6)

        self.v_proj = nn.Linear(context_dim, inner_dim, bias=False)
        self.v_norm = nn.Identity()

        self.output_proj = nn.Linear(inner_dim, query_dim, bias=False)
        self.output_dropout = nn.Dropout(dropout) if dropout > 1e-4 else nn.Identity()

        if self.backend == "transformer_engine":
            from transformer_engine.pytorch.attention import DotProductAttention

            self.attn_op = DotProductAttention(
                self.n_heads,
                self.head_dim,
                num_gqa_groups=self.n_heads,
                attention_dropout=0,
                qkv_format=qkv_format,
                attn_mask_type="no_mask",
            )
        elif self.backend == "minimal_a2a":
            self.attn_op = MinimalA2AAttnOp()
        elif self.backend == "torch":
            self.attn_op = torch_attention_op
        elif self.backend == "torch-flex":
            # FlexAttention backend; returns [B, S, H*D]
            self.attn_op = flex_attention_op
        elif self.backend == "i4":
            # I4 spatio-temporal attention; returns [B, S, H*D]
            self.attn_op = i4_attention_op

        if not hasattr(self.attn_op, "set_context_parallel_group"):

            def set_context_parallel_group(*args, **kwargs) -> None:
                return None

            self.attn_op.set_context_parallel_group = set_context_parallel_group

        self._query_dim = query_dim
        self._context_dim = context_dim
        self._inner_dim = inner_dim

    def init_weights(self) -> None:
        std = 1.0 / math.sqrt(self._query_dim)
        torch.nn.init.trunc_normal_(self.q_proj.weight, std=std, a=-3 * std, b=3 * std)
        std = 1.0 / math.sqrt(self._context_dim)
        torch.nn.init.trunc_normal_(self.k_proj.weight, std=std, a=-3 * std, b=3 * std)
        torch.nn.init.trunc_normal_(self.v_proj.weight, std=std, a=-3 * std, b=3 * std)

        std = 1.0 / math.sqrt(self._inner_dim)
        torch.nn.init.trunc_normal_(self.output_proj.weight, std=std, a=-3 * std, b=3 * std)

        for layer in self.q_norm, self.k_norm, self.v_norm:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()

    def compute_qkv(self, x, context=None, rope_emb=None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = self.q_proj(x)
        context = x if context is None else context
        k = self.k_proj(context)
        v = self.v_proj(context)
        q, k, v = map(
            lambda t: rearrange(t, "b ... (h d) -> b ... h d", h=self.n_heads, d=self.head_dim),
            (q, k, v),
        )

        def apply_norm_and_rotary_pos_emb(q, k, v, rope_emb):
            q = self.q_norm(q)
            k = self.k_norm(k)
            v = self.v_norm(v)
            original_dtype = q.dtype
            if self.is_selfattn and rope_emb is not None:  # only apply to self-attention!
                if self.use_wan_fp32_strategy:  # wan will force q and k to fp32 before rotary pos emb
                    q = q.to(torch.float32)
                    k = k.to(torch.float32)
                q = apply_rotary_pos_emb(q, rope_emb, tensor_format=self.qkv_format, fused=True)
                k = apply_rotary_pos_emb(k, rope_emb, tensor_format=self.qkv_format, fused=True)
                if self.use_wan_fp32_strategy:
                    q = q.to(original_dtype)
                    k = k.to(original_dtype)
            return q, k, v

        q, k, v = apply_norm_and_rotary_pos_emb(q, k, v, rope_emb)

        return q, k, v

    def compute_attention(
        self,
        q,
        k,
        v,
        video_size: Optional[VideoSize] = None,
        kv_cache_cfg: Optional[KVCacheConfig] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ):
        additional_args = {}
        if (
            kv_cache_cfg is not None
            and self.is_selfattn
            and not isinstance(self.attn_op, AttentionOpWithKVCache)
        ):
            if attn_mask is not None:
                raise ValueError(
                    "streaming self-attention receives only cached history "
                    "plus the current causal block, so no full-sequence mask "
                    "may be supplied"
                )
            cache = kv_cache_cfg.cache
            if kv_cache_cfg.run_with_kv and cache is not None:
                k_past, v_past = cache["k"], cache["v"]
                if (
                    k_past.shape[0] != k.shape[0]
                    or k_past.shape[2:] != k.shape[2:]
                    or v_past.shape != k_past.shape
                ):
                    raise ValueError(
                        "DiT KV-cache shape mismatch: "
                        f"cached={tuple(k_past.shape)}/{tuple(v_past.shape)}, "
                        f"current={tuple(k.shape)}/{tuple(v.shape)}"
                    )
                k = torch.cat([k_past, k], dim=1)
                v = torch.cat([v_past, v], dim=1)
            if kv_cache_cfg.store_kv:
                k_store = select_kv_cache_tokens(
                    k,
                    max_cache_tokens=kv_cache_cfg.max_cache_tokens,
                    sink_cache_tokens=kv_cache_cfg.sink_cache_tokens,
                )
                v_store = select_kv_cache_tokens(
                    v,
                    max_cache_tokens=kv_cache_cfg.max_cache_tokens,
                    sink_cache_tokens=kv_cache_cfg.sink_cache_tokens,
                )
                if kv_cache_cfg.detach_cache:
                    k_store = k_store.detach()
                    v_store = v_store.detach()
                kv_cache_cfg.cache = {
                    "k": k_store,
                    "v": v_store,
                }
        if isinstance(self.attn_op, (NattenA2AAttnOp, NeighborhoodAttention)) or self.backend == "i4":
            additional_args["video_size"] = video_size
        if isinstance(self.attn_op, AttentionOpWithKVCache):
            additional_args["kv_cache_cfg"] = kv_cache_cfg
        if attn_mask is not None:
            assert self.backend in ("torch", "torch-flex"), (
                f"attn_mask only supported with the torch / torch-flex backends, got '{self.backend}'"
            )
            additional_args["attn_mask"] = attn_mask

        result = self.attn_op(q, k, v, **additional_args)  # [B, S, H, D]
        return self.output_dropout(self.output_proj(result))

    def forward(
        self,
        x,
        context: Optional[torch.Tensor] = None,
        rope_emb: Optional[torch.Tensor] = None,
        video_size: Optional[VideoSize] = None,
        kv_cache_cfg: Optional[KVCacheConfig] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            x (Tensor): The query tensor of shape [B, Mq, K]
            context (Optional[Tensor]): The key tensor of shape [B, Mk, K] or use x as context [self attention] if None
            rope_emb (Optional[Tensor]): RoPE embedding tensor, or no RoPE embeddings (i.e. in cross attention)
            video_size(VideoSize): Shape [T, H, W]
            attn_mask (Optional[Tensor]): dense torch mask or sparse FlexAttention
                ``BlockMask`` (bool: True = attend). Cross-attention never
                passes one.
        """
        q, k, v = self.compute_qkv(x, context, rope_emb=rope_emb)
        return self.compute_attention(
            q, k, v, video_size=video_size, kv_cache_cfg=kv_cache_cfg, attn_mask=attn_mask
        )

    def set_context_parallel_group(self, process_group, ranks, stream, cp_comm_type: str = "p2p"):
        # self.attn_op.set_context_parallel_group(process_group, ranks, stream, cp_comm_type="a2a")
        self.attn_op.set_context_parallel_group(process_group, ranks, stream, cp_comm_type=cp_comm_type)


class I2VCrossAttention(Attention):
    def __init__(self, *args, img_latent_dim: int = 1024, **kwargs):
        super().__init__(*args, **kwargs)
        inner_dim = self.head_dim * self.n_heads
        self.k_img = nn.Linear(img_latent_dim, inner_dim, bias=False)
        self.v_img = nn.Linear(img_latent_dim, inner_dim, bias=False)
        self.k_img_norm = te.pytorch.RMSNorm(self.head_dim, eps=1e-6)

    def init_weights(self) -> None:
        super().init_weights()
        torch.nn.init.trunc_normal_(self.k_img.weight, std=1.0 / math.sqrt(self._inner_dim))
        torch.nn.init.trunc_normal_(self.v_img.weight, std=1.0 / math.sqrt(self._inner_dim))
        self.k_img_norm.reset_parameters()

    def compute_qkv(
        self, x, context, rope_emb=None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        text_context, img_context = context
        q, k, v = super().compute_qkv(x, text_context, rope_emb)
        k_img = self.k_img(img_context)
        v_img = self.v_img(img_context)
        # Rearrange k_img, v_img
        k_img, v_img = map(
            lambda t: rearrange(t, "b ... (h d) -> b ... h d", h=self.n_heads, d=self.head_dim),
            (k_img, v_img),
        )

        return q, k, v, self.k_img_norm(k_img), v_img

    def compute_attention(self, q, k, v, k_img, v_img):
        result = self.attn_op(q, k, v)  # [B, S, H, D]
        result_img = self.attn_op(q, k_img, v_img)
        return self.output_dropout(self.output_proj(result + result_img))

    def forward(
        self,
        x,
        context=None,
        rope_emb=None,
    ):
        q, k, v, k_img, v_img = self.compute_qkv(x, context, rope_emb)
        return self.compute_attention(q, k, v, k_img, v_img)


class VideoPositionEmb(nn.Module):
    def __init__(self):
        super().__init__()
        self._cp_group = None

    def enable_context_parallel(self, process_group: ProcessGroup):
        self._cp_group = process_group

    def disable_context_parallel(self):
        self._cp_group = None

    @property
    def seq_dim(self):
        return 1

    def forward(
        self,
        x_B_T_H_W_C: torch.Tensor,
        fps=Optional[torch.Tensor],
        temporal_positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        With CP, the function assume that the input tensor is already split.
        It delegates the embedding generation to generate_embeddings function.

        ``temporal_positions`` (optional, shape ``(T,)``): explicit per-frame
        temporal indices for RoPE. When given, the temporal axis is positioned
        at these values instead of ``0..T-1`` -- used to align modalities with
        different temporal compression rates (e.g. pixel-time of lidar vs RGB
        condition frames). Not supported under context parallel.
        """
        B_T_H_W_C = x_B_T_H_W_C.shape
        if self._cp_group is not None:
            assert temporal_positions is None, "temporal_positions not supported with context parallel"
            cp_ranks = get_process_group_ranks(self._cp_group)
            cp_size = len(cp_ranks)
            cp_size_t = cp_size
            if USE_MEGATRON and hasattr(parallel_state, "cp_size_t"):
                # We saved cp_size_t in find_split function for combined temporal and spatial splitting.
                # We need cp_size_t to find out the split values for T and H dimensions for correct embedding calculations.
                cp_size_t = parallel_state.cp_size_t
            cp_size_h = max(1, cp_size // cp_size_t)
            B, T, H, W, C = B_T_H_W_C
            B_T_H_W_C = (B, T * cp_size_t, H * cp_size_h, W, C)
        embeddings = self.generate_embeddings(B_T_H_W_C, fps=fps, temporal_positions=temporal_positions)

        return self._split_for_context_parallel(embeddings)

    def generate_embeddings(self, B_T_H_W_C: torch.Size, fps=Optional[torch.Tensor], temporal_positions=None):
        raise NotImplementedError

    def _split_for_context_parallel(self, embeddings):
        if self._cp_group is not None:
            embeddings = split_inputs_cp(x=embeddings, seq_dim=self.seq_dim, cp_group=self._cp_group)
        return embeddings


class VideoRopePosition3DEmb(VideoPositionEmb):
    def __init__(
        self,
        *,  # enforce keyword arguments
        head_dim: int,
        len_h: int,
        len_w: int,
        len_t: int,
        base_fps: int = 24,
        h_extrapolation_ratio: float = 1.0,
        w_extrapolation_ratio: float = 1.0,
        t_extrapolation_ratio: float = 1.0,
        h_position_stride: float = 1.0,
        w_position_stride: float = 1.0,
        enable_fps_modulation: bool = True,
        **kwargs,  # used for compatibility with other positional embeddings; unused in this class
    ):
        del kwargs
        super().__init__()
        self.register_buffer("seq", torch.arange(max(len_h, len_w, len_t), dtype=torch.float))
        self.base_fps = base_fps
        self.max_h = len_h
        self.max_w = len_w
        self.max_t = len_t
        self.h_position_stride = float(h_position_stride)
        self.w_position_stride = float(w_position_stride)
        if self.h_position_stride <= 0 or self.w_position_stride <= 0:
            raise ValueError(
                "RoPE spatial position strides must be positive, got "
                f"{(self.h_position_stride, self.w_position_stride)}"
            )
        self.enable_fps_modulation = enable_fps_modulation
        dim = head_dim
        dim_h = dim // 6 * 2
        dim_w = dim_h
        dim_t = dim - 2 * dim_h
        assert dim == dim_h + dim_w + dim_t, f"bad dim: {dim} != {dim_h} + {dim_w} + {dim_t}"
        # TODO: (qsh 2025-01-04) this is tricky! https://github.com/pytorch/torchtitan/blob/main/torchtitan/models/llama/model.py#L361
        self.register_buffer(
            "dim_spatial_range",
            torch.arange(0, dim_h, 2)[: (dim_h // 2)].float() / dim_h,
            persistent=True,
        )
        self.register_buffer(
            "dim_temporal_range",
            torch.arange(0, dim_t, 2)[: (dim_t // 2)].float() / dim_t,
            persistent=True,
        )
        self._dim_h = dim_h
        self._dim_t = dim_t

        self.h_ntk_factor = h_extrapolation_ratio ** (dim_h / (dim_h - 2))
        self.w_ntk_factor = w_extrapolation_ratio ** (dim_w / (dim_w - 2))
        self.t_ntk_factor = t_extrapolation_ratio ** (dim_t / (dim_t - 2))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        dim_h = self._dim_h
        dim_t = self._dim_t

        self.seq = torch.arange(max(self.max_h, self.max_w, self.max_t)).float().to(self.dim_spatial_range.device)
        self.dim_spatial_range = (
            torch.arange(0, dim_h, 2)[: (dim_h // 2)].float().to(self.dim_spatial_range.device) / dim_h
        )
        self.dim_temporal_range = (
            torch.arange(0, dim_t, 2)[: (dim_t // 2)].float().to(self.dim_spatial_range.device) / dim_t
        )

    def generate_embeddings(
        self,
        B_T_H_W_C: torch.Size,
        fps: Optional[torch.Tensor] = None,
        h_ntk_factor: Optional[float] = None,
        w_ntk_factor: Optional[float] = None,
        t_ntk_factor: Optional[float] = None,
        temporal_positions: Optional[torch.Tensor] = None,
    ):
        """
        Generate embeddings for the given input size.

        Args:
            B_T_H_W_C (torch.Size): Input tensor size (Batch, Time, Height, Width, Channels).
            fps (Optional[torch.Tensor], optional): Frames per second. Defaults to None.
            h_ntk_factor (Optional[float], optional): Height NTK factor. If None, uses self.h_ntk_factor.
            w_ntk_factor (Optional[float], optional): Width NTK factor. If None, uses self.w_ntk_factor.
            t_ntk_factor (Optional[float], optional): Time NTK factor. If None, uses self.t_ntk_factor.

        Returns:
            Not specified in the original code snippet.
        """
        h_ntk_factor = h_ntk_factor if h_ntk_factor is not None else self.h_ntk_factor
        w_ntk_factor = w_ntk_factor if w_ntk_factor is not None else self.w_ntk_factor
        t_ntk_factor = t_ntk_factor if t_ntk_factor is not None else self.t_ntk_factor

        h_theta = 10000.0 * h_ntk_factor
        w_theta = 10000.0 * w_ntk_factor
        t_theta = 10000.0 * t_ntk_factor

        h_spatial_freqs = 1.0 / (h_theta ** self.dim_spatial_range.float())
        w_spatial_freqs = 1.0 / (w_theta ** self.dim_spatial_range.float())
        temporal_freqs = 1.0 / (t_theta ** self.dim_temporal_range.float())

        B, T, H, W, _ = B_T_H_W_C
        assert H <= self.max_h and W <= self.max_w, (
            f"Input dimensions (H={H}, W={W}) exceed the maximum dimensions (max_h={self.max_h}, max_w={self.max_w})"
        )
        half_emb_h = torch.outer(
            self.seq[:H] * self.h_position_stride,
            h_spatial_freqs,
        )
        half_emb_w = torch.outer(
            self.seq[:W] * self.w_position_stride,
            w_spatial_freqs,
        )

        # Explicit per-frame temporal positions (e.g. pixel-time alignment of
        # heterogeneous-compression modalities) override the default 0..T-1.
        if temporal_positions is not None:
            assert temporal_positions.shape[0] == T, (
                f"temporal_positions {tuple(temporal_positions.shape)} must have length T={T}"
            )
            t_seq = temporal_positions.to(device=temporal_freqs.device, dtype=temporal_freqs.dtype)
        else:
            t_seq = self.seq[:T]

        if self.enable_fps_modulation:
            uniform_fps = (fps is None) or (fps.min() == fps.max())
            assert uniform_fps or B == 1 or T == 1, (
                "For video batch, batch size should be 1 for non-uniform fps. For image batch, T should be 1"
            )

            # apply sequence scaling in temporal dimension
            if fps is None:  # image case
                assert T == 1, "T should be 1 for image batch."
                half_emb_t = torch.outer(t_seq, temporal_freqs)
            else:
                half_emb_t = torch.outer(t_seq / fps[:1] * self.base_fps, temporal_freqs)
        else:
            half_emb_t = torch.outer(t_seq, temporal_freqs)

        em_T_H_W_D = torch.cat(
            [
                repeat(half_emb_t, "t d -> t h w d", h=H, w=W),
                repeat(half_emb_h, "h d -> t h w d", t=T, w=W),
                repeat(half_emb_w, "w d -> t h w d", t=T, h=H),
            ]
            * 2,
            dim=-1,
        )

        return rearrange(em_T_H_W_D, "t h w d -> (t h w) 1 1 d").float()

    @property
    def seq_dim(self):
        return 0


class LearnablePosEmbAxis(VideoPositionEmb):
    def __init__(
        self,
        *,  # enforce keyword arguments
        interpolation: str,
        model_channels: int,
        len_h: int,
        len_w: int,
        len_t: int,
        **kwargs,
    ):
        """
        Args:
            interpolation (str): we curretly only support "crop", ideally when we need extrapolation capacity, we should adjust frequency or other more advanced methods. they are not implemented yet.
        """
        del kwargs  # unused
        super().__init__()
        self.interpolation = interpolation
        assert self.interpolation in ["crop"], f"Unknown interpolation method {self.interpolation}"
        self.model_channels = model_channels

        self.pos_emb_h = nn.Parameter(torch.zeros(len_h, model_channels))
        self.pos_emb_w = nn.Parameter(torch.zeros(len_w, model_channels))
        self.pos_emb_t = nn.Parameter(torch.zeros(len_t, model_channels))

        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.model_channels)
        torch.nn.init.trunc_normal_(self.pos_emb_h, std=std, a=-3 * std, b=3 * std)
        torch.nn.init.trunc_normal_(self.pos_emb_w, std=std, a=-3 * std, b=3 * std)
        torch.nn.init.trunc_normal_(self.pos_emb_t, std=std, a=-3 * std, b=3 * std)

    def generate_embeddings(
        self, B_T_H_W_C: torch.Size, fps=Optional[torch.Tensor], temporal_positions=None
    ) -> torch.Tensor:
        del temporal_positions  # learnable abs pos-emb indexes by crop, no custom positions
        B, T, H, W, _ = B_T_H_W_C
        if self.interpolation == "crop":
            emb_h_H = self.pos_emb_h[:H]
            emb_w_W = self.pos_emb_w[:W]
            emb_t_T = self.pos_emb_t[:T]
            emb = (
                repeat(emb_t_T, "t d-> b t h w d", b=B, h=H, w=W)
                + repeat(emb_h_H, "h d-> b t h w d", b=B, t=T, w=W)
                + repeat(emb_w_W, "w d-> b t h w d", b=B, t=T, h=H)
            )
            assert list(emb.shape)[:4] == [B, T, H, W], f"bad shape: {list(emb.shape)[:4]} != {B, T, H, W}"
        else:
            raise ValueError(f"Unknown interpolation method {self.interpolation}")

        norm = torch.linalg.vector_norm(emb, dim=-1, keepdim=True, dtype=torch.float32)
        norm = torch.add(1e-6, norm, alpha=np.sqrt(norm.numel() / emb.numel()))
        return emb / norm.to(emb.dtype)


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class Timesteps(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.num_channels = num_channels

    def forward(self, timesteps_B_T):
        assert timesteps_B_T.ndim == 2, f"Expected 2D input, got {timesteps_B_T.ndim}"
        # wan need emb to be in fp32
        in_dtype = timesteps_B_T.dtype
        timesteps = timesteps_B_T.flatten().float()
        half_dim = self.num_channels // 2
        exponent = -math.log(10000) * torch.arange(half_dim, dtype=torch.float32, device=timesteps.device)
        exponent = exponent / (half_dim - 0.0)

        emb = torch.exp(exponent)
        emb = timesteps[:, None].float() * emb[None, :]

        sin_emb = torch.sin(emb)
        cos_emb = torch.cos(emb)
        emb = torch.cat([cos_emb, sin_emb], dim=-1)

        return rearrange(emb.to(dtype=in_dtype), "(b t) d -> b t d", b=timesteps_B_T.shape[0], t=timesteps_B_T.shape[1])


class TimestepEmbedding(nn.Module):
    def __init__(self, in_features: int, out_features: int, use_adaln_lora: bool = False):
        super().__init__()
        log.debug(
            f"Using AdaLN LoRA Flag:  {use_adaln_lora}. We enable bias if no AdaLN LoRA for backward compatibility."
        )
        self.in_dim = in_features
        self.out_dim = out_features
        self.linear_1 = nn.Linear(in_features, out_features, bias=not use_adaln_lora)
        self.activation = nn.SiLU()
        self.use_adaln_lora = use_adaln_lora
        if use_adaln_lora:
            self.linear_2 = nn.Linear(out_features, 3 * out_features, bias=False)
        else:
            self.linear_2 = nn.Linear(out_features, out_features, bias=False)

        self.init_weights()

    def init_weights(self) -> None:
        std = 1.0 / math.sqrt(self.in_dim)
        torch.nn.init.trunc_normal_(self.linear_1.weight, std=std, a=-3 * std, b=3 * std)

        std = 1.0 / math.sqrt(self.out_dim)
        torch.nn.init.trunc_normal_(self.linear_2.weight, std=std, a=-3 * std, b=3 * std)

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        emb = self.linear_1(sample)
        emb = self.activation(emb)
        emb = self.linear_2(emb)

        if self.use_adaln_lora:
            adaln_lora_B_T_3D = emb
            emb_B_T_D = sample
        else:
            emb_B_T_D = emb
            adaln_lora_B_T_3D = None

        return emb_B_T_D, adaln_lora_B_T_3D


class FourierFeatures(nn.Module):
    """
    Implements a layer that generates Fourier features from input tensors, based on randomly sampled
    frequencies and phases. This can help in learning high-frequency functions in low-dimensional problems.

    [B] -> [B, D]

    Parameters:
        num_channels (int): The number of Fourier features to generate.
        bandwidth (float, optional): The scaling factor for the frequency of the Fourier features. Defaults to 1.
        normalize (bool, optional): If set to True, the outputs are scaled by sqrt(2), usually to normalize
                                    the variance of the features. Defaults to False.

    Example:
        >>> layer = FourierFeatures(num_channels=256, bandwidth=0.5, normalize=True)
        >>> x = torch.randn(10, 256)  # Example input tensor
        >>> output = layer(x)
        >>> print(output.shape)  # Expected shape: (10, 256)
    """

    def __init__(self, num_channels, bandwidth=1, normalize=False):
        super().__init__()
        self.register_buffer("freqs", 2 * np.pi * bandwidth * torch.randn(num_channels), persistent=True)
        self.register_buffer("phases", 2 * np.pi * torch.rand(num_channels), persistent=True)
        self.gain = np.sqrt(2) if normalize else 1
        self.bandwidth = bandwidth
        self.num_channels = num_channels

        self.reset_parameters()

    def reset_parameters(self) -> None:
        generator = torch.Generator()
        generator.manual_seed(0)
        self.freqs = (
            2 * np.pi * self.bandwidth * torch.randn(self.num_channels, generator=generator).to(self.freqs.device)
        )
        self.phases = 2 * np.pi * torch.rand(self.num_channels, generator=generator).to(self.freqs.device)

    def forward(self, x, gain: float = 1.0):
        """
        Apply the Fourier feature transformation to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.
            gain (float, optional): An additional gain factor applied during the forward pass. Defaults to 1.

        Returns:
            torch.Tensor: The transformed tensor, with Fourier features applied.
        """
        in_dtype = x.dtype
        x = x.to(torch.float32).ger(self.freqs.to(torch.float32)).add(self.phases.to(torch.float32))
        x = x.cos().mul(self.gain * gain).to(in_dtype)
        return x


def _spatial_pair(value, name: str) -> Tuple[int, int]:
    """Normalize a scalar or ``(height, width)`` spatial setting."""
    if isinstance(value, int):
        pair = (value, value)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) != 2:
            raise ValueError(f"{name} must have two entries, got {value!r}")
        pair = (int(value[0]), int(value[1]))
    else:
        raise TypeError(f"{name} must be an int or length-2 sequence, got {value!r}")
    if pair[0] <= 0 or pair[1] <= 0:
        raise ValueError(f"{name} entries must be positive, got {pair}")
    return pair


class PatchEmbed(nn.Module):
    """
    PatchEmbed is a module for embedding patches from an input tensor by applying either 3D or 2D convolutional layers,
    depending on the . This module can process inputs with temporal (video) and spatial (image) dimensions,
    making it suitable for video and image processing tasks. It supports dividing the input into patches
    and embedding each patch into a vector of size `out_channels`.

    Parameters:
    - spatial_patch_size (int | tuple[int, int]): Spatial patch height/width.
    - temporal_patch_size (int): The size of each temporal patch.
    - in_channels (int): Number of input channels. Default: 3.
    - out_channels (int): The dimension of the embedding vector for each patch. Default: 768.
    - bias (bool): If True, adds a learnable bias to the output of the convolutional layers. Default: True.
    """

    def __init__(
        self,
        spatial_patch_size,
        temporal_patch_size,
        in_channels=3,
        out_channels=768,
    ):
        super().__init__()
        self.spatial_patch_size = spatial_patch_size
        self.patch_h, self.patch_w = _spatial_pair(
            spatial_patch_size, "spatial_patch_size"
        )
        self.temporal_patch_size = temporal_patch_size

        self.proj = nn.Sequential(
            Rearrange(
                "b c (t r) (h m) (w n) -> b t h w (c r m n)",
                r=temporal_patch_size,
                m=self.patch_h,
                n=self.patch_w,
            ),
            nn.Linear(
                in_channels
                * self.patch_h
                * self.patch_w
                * temporal_patch_size,
                out_channels,
                bias=False,
            ),
        )
        self.dim = (
            in_channels * self.patch_h * self.patch_w * temporal_patch_size
        )

        self.init_weights()

    def init_weights(self) -> None:
        std = 1.0 / math.sqrt(self.dim)
        torch.nn.init.trunc_normal_(self.proj[1].weight, std=std, a=-3 * std, b=3 * std)

    def forward(self, x):
        """
        Forward pass of the PatchEmbed module.

        Parameters:
        - x (torch.Tensor): The input tensor of shape (B, C, T, H, W) where
            B is the batch size,
            C is the number of channels,
            T is the temporal dimension,
            H is the height, and
            W is the width of the input.

        Returns:
        - torch.Tensor: The embedded patches as a tensor, with shape b t h w c.
        """
        assert x.dim() == 5
        _, _, T, H, W = x.shape
        assert H % self.patch_h == 0 and W % self.patch_w == 0, (
            f"H,W {(H, W)} should be divisible by spatial patch "
            f"{(self.patch_h, self.patch_w)}"
        )
        assert T % self.temporal_patch_size == 0
        x = self.proj(x)
        return x


class FinalLayer(nn.Module):
    """
    The final layer of video DiT.
    """

    def __init__(
        self,
        hidden_size,
        spatial_patch_size,
        temporal_patch_size,
        out_channels,
        use_adaln_lora: bool = False,
        adaln_lora_dim: int = 256,
        use_wan_fp32_strategy: bool = False,
    ):
        super().__init__()
        self.use_wan_fp32_strategy = use_wan_fp32_strategy
        self.patch_h, self.patch_w = _spatial_pair(
            spatial_patch_size, "spatial_patch_size"
        )
        self.layer_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(
            hidden_size,
            self.patch_h
            * self.patch_w
            * temporal_patch_size
            * out_channels,
            bias=False,
        )
        self.hidden_size = hidden_size
        self.n_adaln_chunks = 2
        self.use_adaln_lora = use_adaln_lora
        self.adaln_lora_dim = adaln_lora_dim
        if use_adaln_lora:
            self.adaln_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, adaln_lora_dim, bias=False),
                nn.Linear(adaln_lora_dim, self.n_adaln_chunks * hidden_size, bias=False),
            )
        else:
            self.adaln_modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(hidden_size, self.n_adaln_chunks * hidden_size, bias=False)
            )

        self.init_weights()

    def init_weights(self) -> None:
        std = 1.0 / math.sqrt(self.hidden_size)
        torch.nn.init.trunc_normal_(self.linear.weight, std=std, a=-3 * std, b=3 * std)
        if self.use_adaln_lora:
            torch.nn.init.trunc_normal_(self.adaln_modulation[1].weight, std=std, a=-3 * std, b=3 * std)
            torch.nn.init.zeros_(self.adaln_modulation[2].weight)
        else:
            torch.nn.init.zeros_(self.adaln_modulation[1].weight)

        self.layer_norm.reset_parameters()

    def forward(
        self,
        # x_BT_HW_D,
        x_B_T_H_W_D,
        emb_B_T_D,
        adaln_lora_B_T_3D: Optional[torch.Tensor] = None,
    ):
        if self.use_wan_fp32_strategy:
            assert emb_B_T_D.dtype == torch.float32
        with amp.autocast("cuda", enabled=self.use_wan_fp32_strategy, dtype=torch.float32):
            if self.use_adaln_lora:
                assert adaln_lora_B_T_3D is not None
                shift_B_T_D, scale_B_T_D = (
                    self.adaln_modulation(emb_B_T_D) + adaln_lora_B_T_3D[:, :, : 2 * self.hidden_size]
                ).chunk(2, dim=-1)
            else:
                shift_B_T_D, scale_B_T_D = self.adaln_modulation(emb_B_T_D).chunk(2, dim=-1)

            shift_B_T_1_1_D, scale_B_T_1_1_D = (
                rearrange(shift_B_T_D, "b t d -> b t 1 1 d"),
                rearrange(scale_B_T_D, "b t d -> b t 1 1 d"),
            )

            def _fn(_x_B_T_H_W_D, _norm_layer, _scale_B_T_1_1_D, _shift_B_T_1_1_D):
                return _norm_layer(_x_B_T_H_W_D) * (1 + _scale_B_T_1_1_D) + _shift_B_T_1_1_D

            x_B_T_H_W_D = _fn(x_B_T_H_W_D, self.layer_norm, scale_B_T_1_1_D, shift_B_T_1_1_D)
            x_B_T_H_W_O = self.linear(
                x_B_T_H_W_D
            )  # O = patch_h * patch_w * temporal_patch_size * out_channels
        return x_B_T_H_W_O


class Block(nn.Module):
    """
    A transformer block that combines self-attention, cross-attention and MLP layers with AdaLN modulation.
    Each component (self-attention, cross-attention, MLP) has its own layer normalization and AdaLN modulation.

    Parameters:
        x_dim (int): Dimension of input features
        context_dim (int): Dimension of context features for cross-attention
        num_heads (int): Number of attention heads
        mlp_ratio (float): Multiplier for MLP hidden dimension. Default: 4.0
        use_adaln_lora (bool): Whether to use AdaLN-LoRA modulation. Default: False
        adaln_lora_dim (int): Hidden dimension for AdaLN-LoRA layers. Default: 256
        use_wan_fp32_strategy (bool): Whether to use Wan's FP32 strategy. Default: False
        If True, in Attention layer, if do self-attention, q and k will be forced to fp32 before rotary pos emb
        also, in modulation computation, force entire computation in fp32

    The block applies the following sequence:
    1. Self-attention with AdaLN modulation
    2. Cross-attention with AdaLN modulation
    3. MLP with AdaLN modulation

    Each component uses skip connections and layer normalization.
    """

    def __init__(
        self,
        x_dim: int,
        context_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        use_adaln_lora: bool = False,
        adaln_lora_dim: int = 256,
        backend: str = "transformer_engine",
        image_context_dim: Optional[int] = None,
        use_wan_fp32_strategy: bool = False,
    ):
        super().__init__()
        self.x_dim = x_dim
        self.layer_norm_self_attn = nn.LayerNorm(x_dim, elementwise_affine=False, eps=1e-6)
        self.self_attn = Attention(
            x_dim,
            None,
            num_heads,
            x_dim // num_heads,
            qkv_format="bshd",
            backend=backend,
            use_wan_fp32_strategy=use_wan_fp32_strategy,
        )

        self.layer_norm_cross_attn = nn.LayerNorm(x_dim, elementwise_affine=False, eps=1e-6)

        if image_context_dim is None:
            self.cross_attn = Attention(
                x_dim, context_dim, num_heads, x_dim // num_heads, qkv_format="bshd", backend=backend
            )
        else:
            self.cross_attn = I2VCrossAttention(
                x_dim,
                context_dim,
                num_heads,
                x_dim // num_heads,
                img_latent_dim=image_context_dim,
                qkv_format="bshd",
                backend=backend,
            )

        self.layer_norm_mlp = nn.LayerNorm(x_dim, elementwise_affine=False, eps=1e-6)
        self.mlp = GPT2FeedForward(x_dim, int(x_dim * mlp_ratio))

        self.use_adaln_lora = use_adaln_lora
        if self.use_adaln_lora:
            self.adaln_modulation_self_attn = nn.Sequential(
                nn.SiLU(),
                nn.Linear(x_dim, adaln_lora_dim, bias=False),
                nn.Linear(adaln_lora_dim, 3 * x_dim, bias=False),
            )
            self.adaln_modulation_cross_attn = nn.Sequential(
                nn.SiLU(),
                nn.Linear(x_dim, adaln_lora_dim, bias=False),
                nn.Linear(adaln_lora_dim, 3 * x_dim, bias=False),
            )
            self.adaln_modulation_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(x_dim, adaln_lora_dim, bias=False),
                nn.Linear(adaln_lora_dim, 3 * x_dim, bias=False),
            )
        else:
            self.adaln_modulation_self_attn = nn.Sequential(nn.SiLU(), nn.Linear(x_dim, 3 * x_dim, bias=False))
            self.adaln_modulation_cross_attn = nn.Sequential(nn.SiLU(), nn.Linear(x_dim, 3 * x_dim, bias=False))
            self.adaln_modulation_mlp = nn.Sequential(nn.SiLU(), nn.Linear(x_dim, 3 * x_dim, bias=False))

        self.cp_size = None
        self.use_wan_fp32_strategy = use_wan_fp32_strategy
        # Inference-only optimization enabled by MiniTrainDIT after it verifies
        # that the text context is exactly zero. Keep all modules/parameters so
        # existing checkpoints remain strictly load-compatible.
        self.skip_crossattn = False

    def set_context_parallel_group(self, process_group, ranks, stream, cp_comm_type: str = "p2p"):
        self.cp_size = None if ranks is None else len(ranks)
        self.self_attn.set_context_parallel_group(
            process_group=process_group,
            ranks=ranks,
            stream=stream,
            cp_comm_type=cp_comm_type,
        )

    def reset_parameters(self) -> None:
        self.layer_norm_self_attn.reset_parameters()
        self.layer_norm_cross_attn.reset_parameters()
        self.layer_norm_mlp.reset_parameters()

        if self.use_adaln_lora:
            std = 1.0 / math.sqrt(self.x_dim)
            torch.nn.init.trunc_normal_(self.adaln_modulation_self_attn[1].weight, std=std, a=-3 * std, b=3 * std)
            torch.nn.init.trunc_normal_(self.adaln_modulation_cross_attn[1].weight, std=std, a=-3 * std, b=3 * std)
            torch.nn.init.trunc_normal_(self.adaln_modulation_mlp[1].weight, std=std, a=-3 * std, b=3 * std)
            torch.nn.init.zeros_(self.adaln_modulation_self_attn[2].weight)
            torch.nn.init.zeros_(self.adaln_modulation_cross_attn[2].weight)
            torch.nn.init.zeros_(self.adaln_modulation_mlp[2].weight)
        else:
            torch.nn.init.zeros_(self.adaln_modulation_self_attn[1].weight)
            torch.nn.init.zeros_(self.adaln_modulation_cross_attn[1].weight)
            torch.nn.init.zeros_(self.adaln_modulation_mlp[1].weight)

    def init_weights(self) -> None:
        self.reset_parameters()
        self.self_attn.init_weights()
        self.cross_attn.init_weights()
        self.mlp.init_weights()

    def forward(
        self,
        x_B_T_H_W_D: torch.Tensor,
        emb_B_T_D: torch.Tensor,
        crossattn_emb: torch.Tensor,
        rope_emb_L_1_1_D: Optional[torch.Tensor] = None,
        adaln_lora_B_T_3D: Optional[torch.Tensor] = None,
        extra_per_block_pos_emb: Optional[torch.Tensor] = None,
        kv_cache_cfg: Optional[KVCacheConfig] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if extra_per_block_pos_emb is not None:
            x_B_T_H_W_D = x_B_T_H_W_D + extra_per_block_pos_emb

        with amp.autocast("cuda", enabled=self.use_wan_fp32_strategy, dtype=torch.float32):
            if self.use_adaln_lora:
                shift_self_attn_B_T_D, scale_self_attn_B_T_D, gate_self_attn_B_T_D = (
                    self.adaln_modulation_self_attn(emb_B_T_D) + adaln_lora_B_T_3D
                ).chunk(3, dim=-1)
                shift_cross_attn_B_T_D, scale_cross_attn_B_T_D, gate_cross_attn_B_T_D = (
                    self.adaln_modulation_cross_attn(emb_B_T_D) + adaln_lora_B_T_3D
                ).chunk(3, dim=-1)
                shift_mlp_B_T_D, scale_mlp_B_T_D, gate_mlp_B_T_D = (
                    self.adaln_modulation_mlp(emb_B_T_D) + adaln_lora_B_T_3D
                ).chunk(3, dim=-1)
            else:
                shift_self_attn_B_T_D, scale_self_attn_B_T_D, gate_self_attn_B_T_D = self.adaln_modulation_self_attn(
                    emb_B_T_D
                ).chunk(3, dim=-1)
                shift_cross_attn_B_T_D, scale_cross_attn_B_T_D, gate_cross_attn_B_T_D = (
                    self.adaln_modulation_cross_attn(emb_B_T_D).chunk(3, dim=-1)
                )
                shift_mlp_B_T_D, scale_mlp_B_T_D, gate_mlp_B_T_D = self.adaln_modulation_mlp(emb_B_T_D).chunk(3, dim=-1)

        # Reshape tensors from (B, T, D) to (B, T, 1, 1, D) for broadcasting
        shift_self_attn_B_T_1_1_D = rearrange(shift_self_attn_B_T_D, "b t d -> b t 1 1 d").type_as(x_B_T_H_W_D)
        scale_self_attn_B_T_1_1_D = rearrange(scale_self_attn_B_T_D, "b t d -> b t 1 1 d").type_as(x_B_T_H_W_D)
        gate_self_attn_B_T_1_1_D = rearrange(gate_self_attn_B_T_D, "b t d -> b t 1 1 d").type_as(x_B_T_H_W_D)

        shift_cross_attn_B_T_1_1_D = rearrange(shift_cross_attn_B_T_D, "b t d -> b t 1 1 d").type_as(x_B_T_H_W_D)
        scale_cross_attn_B_T_1_1_D = rearrange(scale_cross_attn_B_T_D, "b t d -> b t 1 1 d").type_as(x_B_T_H_W_D)
        gate_cross_attn_B_T_1_1_D = rearrange(gate_cross_attn_B_T_D, "b t d -> b t 1 1 d").type_as(x_B_T_H_W_D)

        shift_mlp_B_T_1_1_D = rearrange(shift_mlp_B_T_D, "b t d -> b t 1 1 d").type_as(x_B_T_H_W_D)
        scale_mlp_B_T_1_1_D = rearrange(scale_mlp_B_T_D, "b t d -> b t 1 1 d").type_as(x_B_T_H_W_D)
        gate_mlp_B_T_1_1_D = rearrange(gate_mlp_B_T_D, "b t d -> b t 1 1 d").type_as(x_B_T_H_W_D)

        B, T, H, W, D = x_B_T_H_W_D.shape

        def _fn(_x_B_T_H_W_D, _norm_layer, _scale_B_T_1_1_D, _shift_B_T_1_1_D):
            return _norm_layer(_x_B_T_H_W_D) * (1 + _scale_B_T_1_1_D) + _shift_B_T_1_1_D

        normalized_x_B_T_H_W_D = _fn(
            x_B_T_H_W_D,
            self.layer_norm_self_attn,
            scale_self_attn_B_T_1_1_D,
            shift_self_attn_B_T_1_1_D,
        )

        video_size = VideoSize(T=T, H=H, W=W)

        # (ahassani): Hack to correct `video_size` when CP is enabled.
        # I really don't like this, but there doesn't seem to be any central
        # piece of code that's responsible for handling CP/TP that also defines the
        # layout of shardings. Other parts of the code (i.e. RoPE) seem to make this
        # assumption that CP sharding is always done along T.
        if self.cp_size is not None and self.cp_size > 1:
            video_size = VideoSize(T=T * self.cp_size, H=H, W=W)

        result_B_T_H_W_D = rearrange(
            self.self_attn(
                # normalized_x_B_T_HW_D,
                rearrange(normalized_x_B_T_H_W_D, "b t h w d -> b (t h w) d"),
                None,
                rope_emb=rope_emb_L_1_1_D,
                video_size=video_size,
                kv_cache_cfg=kv_cache_cfg,
                attn_mask=self_attn_mask,
            ),
            "b (t h w) d -> b t h w d",
            t=T,
            h=H,
            w=W,
        )
        x_B_T_H_W_D = x_B_T_H_W_D + gate_self_attn_B_T_1_1_D * result_B_T_H_W_D

        def _x_fn(
            _x_B_T_H_W_D,
            layer_norm_cross_attn,
            _scale_cross_attn_B_T_1_1_D,
            _shift_cross_attn_B_T_1_1_D,
            _gate_cross_attn_B_T_1_1_D,
        ):
            _normalized_x_B_T_H_W_D = _fn(
                _x_B_T_H_W_D, layer_norm_cross_attn, _scale_cross_attn_B_T_1_1_D, _shift_cross_attn_B_T_1_1_D
            )
            _result_B_T_H_W_D = rearrange(
                self.cross_attn(
                    rearrange(_normalized_x_B_T_H_W_D, "b t h w d -> b (t h w) d"),
                    crossattn_emb,
                    rope_emb=rope_emb_L_1_1_D,
                ),
                "b (t h w) d -> b t h w d",
                t=T,
                h=H,
                w=W,
            )
            # _x_B_T_H_W_D = _x_B_T_H_W_D + _gate_cross_attn_B_T_1_1_D * _result_B_T_H_W_D
            return _result_B_T_H_W_D

        if not self.skip_crossattn:
            result_B_T_H_W_D = _x_fn(
                x_B_T_H_W_D,
                self.layer_norm_cross_attn,
                scale_cross_attn_B_T_1_1_D,
                shift_cross_attn_B_T_1_1_D,
                gate_cross_attn_B_T_1_1_D,
            )
            x_B_T_H_W_D = (
                result_B_T_H_W_D * gate_cross_attn_B_T_1_1_D
                + x_B_T_H_W_D
            )

        normalized_x_B_T_H_W_D = _fn(
            x_B_T_H_W_D,
            self.layer_norm_mlp,
            scale_mlp_B_T_1_1_D,
            shift_mlp_B_T_1_1_D,
        )
        result_B_T_H_W_D = self.mlp(normalized_x_B_T_H_W_D)
        x_B_T_H_W_D = x_B_T_H_W_D + gate_mlp_B_T_1_1_D * result_B_T_H_W_D
        return x_B_T_H_W_D


class MiniTrainDIT(WeightTrainingStat):
    """
    A clean impl of DIT that can load and  reproduce the training results of the original DIT model in edify_video/v4~(cosmos 1)
    A general implementation of adaln-modulated VIT-like~(DiT) transformer for video processing.

    Args:
        max_img_h (int): Maximum height of the input images.
        max_img_w (int): Maximum width of the input images.
        max_frames (int): Maximum number of frames in the video sequence.
        in_channels (int): Number of input channels (e.g., RGB channels for color images).
        out_channels (int): Number of output channels.
        patch_spatial (int | tuple[int, int]): Spatial patch height/width.
        patch_temporal (int): Temporal resolution of patches for input processing.
        concat_padding_mask (bool): If True, includes a mask channel in the input to handle padding.
        model_channels (int): Base number of channels used throughout the model.
        num_blocks (int): Number of transformer blocks.
        num_heads (int): Number of heads in the multi-head attention layers.
        mlp_ratio (float): Expansion ratio for MLP blocks.
        crossattn_emb_channels (int): Number of embedding channels for cross-attention.
        extra_image_context_dim (int): Number of embedding channels for extra image context.
        pos_emb_cls (str): Type of positional embeddings.
        pos_emb_learnable (bool): Whether positional embeddings are learnable.
        pos_emb_interpolation (str): Method for interpolating positional embeddings.
        min_fps (int): Minimum frames per second.
        max_fps (int): Maximum frames per second.
        use_adaln_lora (bool): Whether to use AdaLN-LoRA.
        adaln_lora_dim (int): Dimension for AdaLN-LoRA.
        rope_h_extrapolation_ratio (float): Height extrapolation ratio for RoPE.
        rope_w_extrapolation_ratio (float): Width extrapolation ratio for RoPE.
        rope_t_extrapolation_ratio (float): Temporal extrapolation ratio for RoPE.
        extra_per_block_abs_pos_emb (bool): Whether to use extra per-block absolute positional embeddings.
        extra_h_extrapolation_ratio (float): Height extrapolation ratio for extra embeddings.
        extra_w_extrapolation_ratio (float): Width extrapolation ratio for extra embeddings.
        extra_t_extrapolation_ratio (float): Temporal extrapolation ratio for extra embeddings.
        n_dense_blocks (`int`, *optional*, defaults to -1):
            Number of blocks that will remain dense (not replaced with sparse attention)
            If -1, no blocks are replaced with sparse attention
            If 0, all blocks use sparse attention
            Otherwise, n_dense_blocks blocks will remain dense, distributed evenly across the network
        natten_parameters (`dict`, *optional*, defaults to None):
            NATTEN (Sparse attention) parameter list.
            The list length must be the same as the number of layers, with each list element
            indicating NATTEN parameters for that layer. If None, NATTEN will not be used in that
            layer and it would remain a full dense self attention. If not None, it must be a
            dictionary/mapping with at least the following key:
              - window_size: `tuple` of size 3 indicating neighborhood attention window size.
                window size of -1 along any dimension means self attention.
            Other optional parameters and their keys:
              - stride: `tuple` of size 3 indicating neighborhood attention stride value.
                stride = 1 is standard neighborhood attention, stride = window size means
                blocked/window self attention (WSA) along that dimension. Any other values are
                strided neighborhood attention. Refer to the GNA paper for more information.

              - dilation: `tuple` of size 3 indicating neighborhood attention dilation value.
                dilation = 1 is standard neighborhood attention. Refer to the DiNAT paper for more
                information.

              - is_causal: `tuple` of 3 booleans indicating whether causal masking is enabled for
                any of the T, H, W dimensions.
    """

    def __init__(
        self,
        max_img_h: int,
        max_img_w: int,
        max_frames: int,
        in_channels: int,
        out_channels: int,
        patch_spatial: Union[int, Tuple[int, int]],
        patch_temporal: int,
        concat_padding_mask: bool = True,
        # attention settings
        model_channels: int = 768,
        num_blocks: int = 10,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        atten_backend: str = "transformer_engine",
        # cross attention settings
        crossattn_emb_channels: int = 1024,
        use_crossattn_projection: bool = False,
        crossattn_proj_in_channels: int = 1024,
        extra_image_context_dim: Optional[int] = None,
        # Learnable positional embeddings ADDED to the projected image-context
        # tokens (only used when ``extra_image_context_dim`` is set AND
        # ``img_context_num_views > 0``). They let RGB K/V tokens carry their
        # view id, pixel-time (soft temporal alignment) and (h, w) spatial
        # location so cross-attention can localize the condition without a
        # shared canvas. Sizes = table extents; ``forward`` takes
        # ``img_context_pos`` (N, 4) = [view, pixel_time, h, w] per token.
        img_context_num_views: int = 0,
        img_context_max_temporal: int = 0,
        img_context_spatial_h: int = 0,
        img_context_spatial_w: int = 0,
        # positional embedding settings
        pos_emb_cls: str = "sincos",
        pos_emb_learnable: bool = False,
        pos_emb_interpolation: str = "crop",
        min_fps: int = 1,
        max_fps: int = 30,
        use_adaln_lora: bool = False,
        adaln_lora_dim: int = 256,
        rope_h_extrapolation_ratio: float = 1.0,
        rope_w_extrapolation_ratio: float = 1.0,
        rope_t_extrapolation_ratio: float = 1.0,
        rope_scale_positions_by_patch: bool = False,
        extra_per_block_abs_pos_emb: bool = False,
        extra_h_extrapolation_ratio: float = 1.0,
        extra_w_extrapolation_ratio: float = 1.0,
        extra_t_extrapolation_ratio: float = 1.0,
        rope_enable_fps_modulation: bool = True,
        sac_config: SACConfig = SACConfig(),
        n_dense_blocks: int = -1,
        natten_parameters: Union[dict, list] = None,
        # if True, will closely match wan's strategy to use fp32 in certain layers/operations
        use_wan_fp32_strategy: bool = False,
        # Optional independently projected latent condition. Its PatchEmbed has
        # the same output shape as x_embedder and is zero-initialized before
        # being added, preserving warm-start behavior exactly at iteration 0.
        side_condition_channels: int = 0,
        # Block-causal temporal self-attention. 0 preserves full bidirectional
        # attention. A positive value groups that many latent frames into one
        # block: queries attend to their own block and all earlier blocks, never
        # later blocks. This is mask-only (no parameters), so bidirectional
        # checkpoints remain strictly load-compatible. Requires torch-flex.
        temporal_block_size: int = 0,
        # Optional clean prefix before temporal blocks. Prefix frames attend
        # bidirectionally within the prefix; later frames are partitioned into
        # ``temporal_block_size`` blocks and attend to the prefix plus current
        # and earlier future blocks. 0 preserves the legacy modulo-T blocks.
        temporal_prefix_frames: int = 0,
        # Number of learnable, ZERO-INIT per-frame "type" embeddings (e.g. lidar
        # vs RGB view-0/1/2). 0 disables the module entirely (bit-identical to
        # before). When >0, ``forward`` accepts ``frame_type_ids`` (T,) and adds
        # the corresponding embedding to every token of that frame. Zero-init =>
        # no-op at iter 0, so a warm-start checkpoint loads unchanged.
        num_frame_type_embeddings: int = 0,
        # Optional global sensor-dataset identity. The selected zero-initialized
        # embedding is added to the AdaLN timestep embedding for every frame,
        # giving all transformer blocks a categorical sensor-domain signal
        # without spending spatial tokens or introducing an ordinal channel.
        num_dataset_embeddings: int = 0,
        # Copilot4D-style per-frame ego-pose conditioning. 0 disables the module
        # entirely (bit-identical to before). When >0 (e.g. 16 = flattened 4x4
        # SE(3)), ``forward`` accepts ``ego_pose_B_T_P`` (B, T, ego_pose_dim) and
        # a zero-init MLP head maps it to a per-frame embedding ADDED to the AdaLN
        # timestep embedding. Living inside the network (not the model wrapper)
        # means it is optimized / EMA'd / checkpointed / FSDP-sharded like every
        # other parameter. Zero-init output => no-op at iter 0.
        ego_pose_dim: int = 0,
    ) -> None:
        super().__init__()
        self.num_frame_type_embeddings = num_frame_type_embeddings
        self.num_dataset_embeddings = int(num_dataset_embeddings)
        if self.num_dataset_embeddings < 0:
            raise ValueError(
                "num_dataset_embeddings must be non-negative, got "
                f"{self.num_dataset_embeddings}"
            )
        self.ego_pose_dim = ego_pose_dim
        self.max_img_h = max_img_h
        self.max_img_w = max_img_w
        self.max_frames = max_frames
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_spatial = patch_spatial
        self.patch_spatial_h, self.patch_spatial_w = _spatial_pair(
            patch_spatial, "patch_spatial"
        )
        self.patch_temporal = patch_temporal
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.model_channels = model_channels
        self.concat_padding_mask = concat_padding_mask
        self.atten_backend = atten_backend
        self.side_condition_channels = int(side_condition_channels)
        assert self.side_condition_channels >= 0, (
            "side_condition_channels must be non-negative, got "
            f"{self.side_condition_channels}"
        )
        self.temporal_block_size = int(temporal_block_size)
        assert self.temporal_block_size >= 0, (
            f"temporal_block_size must be >= 0, got {self.temporal_block_size}"
        )
        self.temporal_prefix_frames = int(temporal_prefix_frames)
        assert self.temporal_prefix_frames >= 0, (
            "temporal_prefix_frames must be >= 0, got "
            f"{self.temporal_prefix_frames}"
        )
        if self.temporal_prefix_frames > 0 and self.temporal_block_size <= 0:
            raise ValueError(
                "temporal_prefix_frames requires temporal_block_size > 0"
            )
        if self.temporal_block_size > 0:
            assert self.atten_backend == "torch-flex", (
                "temporal block-causal attention requires "
                f"atten_backend='torch-flex', got {self.atten_backend!r}"
            )
        if (
            self.atten_backend == "torch-flex"
            and str(sac_config.mode) not in (
                str(CheckpointMode.NONE),
                str(CheckpointMode.BLOCK_WISE),
            )
        ):
            raise ValueError(
                "torch-flex is incompatible with selective activation "
                "checkpoint caching; use sac_config.mode='block_wise' for "
                "ordinary full-block checkpointing, or 'none' to disable it"
            )
        # positional embedding settings
        self.pos_emb_cls = pos_emb_cls
        self.pos_emb_learnable = pos_emb_learnable
        self.pos_emb_interpolation = pos_emb_interpolation
        self.min_fps = min_fps
        self.max_fps = max_fps
        self.rope_h_extrapolation_ratio = rope_h_extrapolation_ratio
        self.rope_w_extrapolation_ratio = rope_w_extrapolation_ratio
        self.rope_t_extrapolation_ratio = rope_t_extrapolation_ratio
        self.rope_scale_positions_by_patch = bool(
            rope_scale_positions_by_patch
        )
        self.extra_per_block_abs_pos_emb = extra_per_block_abs_pos_emb
        self.extra_h_extrapolation_ratio = extra_h_extrapolation_ratio
        self.extra_w_extrapolation_ratio = extra_w_extrapolation_ratio
        self.extra_t_extrapolation_ratio = extra_t_extrapolation_ratio
        self.rope_enable_fps_modulation = rope_enable_fps_modulation
        self.extra_image_context_dim = extra_image_context_dim
        self.build_patch_embed()
        self.build_pos_embed()
        self.use_adaln_lora = use_adaln_lora
        self.adaln_lora_dim = adaln_lora_dim
        self.t_embedder = nn.Sequential(
            Timesteps(model_channels),
            TimestepEmbedding(model_channels, model_channels, use_adaln_lora=use_adaln_lora),
        )
        # Copilot4D ego-pose head: flattened SE(3) (ego_pose_dim) -> Linear ->
        # LayerNorm -> Linear -> model_channels. Final linear ZERO-INIT so the
        # pose embedding is exactly 0 at iter 0 (warm-start-safe additive residual
        # on the AdaLN timestep embedding). Only built when ego_pose_dim > 0.
        if self.ego_pose_dim > 0:
            self.ego_pose_head = nn.Sequential(
                nn.Linear(self.ego_pose_dim, model_channels),
                nn.LayerNorm(model_channels),
                nn.Linear(model_channels, model_channels),
            )
            nn.init.zeros_(self.ego_pose_head[-1].weight)
            nn.init.zeros_(self.ego_pose_head[-1].bias)
        self.use_crossattn_projection = use_crossattn_projection
        self.crossattn_proj_in_channels = crossattn_proj_in_channels
        self.use_wan_fp32_strategy = use_wan_fp32_strategy
        self.skip_crossattn_when_zero = False
        self._zero_crossattn_validated = False
        self._cached_inference_self_attention_backend = "flex"

        self.blocks = nn.ModuleList(
            [
                Block(
                    x_dim=model_channels,
                    context_dim=crossattn_emb_channels,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    use_adaln_lora=use_adaln_lora,
                    adaln_lora_dim=adaln_lora_dim,
                    backend=atten_backend,
                    image_context_dim=None if extra_image_context_dim is None else model_channels,
                    use_wan_fp32_strategy=use_wan_fp32_strategy,
                )
                for _ in range(num_blocks)
            ]
        )

        self.final_layer = FinalLayer(
            hidden_size=self.model_channels,
            spatial_patch_size=self.patch_spatial,
            temporal_patch_size=self.patch_temporal,
            out_channels=self.out_channels,
            use_adaln_lora=self.use_adaln_lora,
            adaln_lora_dim=self.adaln_lora_dim,
            use_wan_fp32_strategy=self.use_wan_fp32_strategy,
        )

        self.t_embedding_norm = te.pytorch.RMSNorm(model_channels, eps=1e-6)
        if extra_image_context_dim is not None:
            self.img_context_proj = nn.Sequential(
                nn.Linear(
                    extra_image_context_dim, model_channels, bias=True
                ),  # help distinguish between image and video context
                nn.GELU(),
            )

        # Learnable positional embeddings for the image-context tokens.
        self.use_img_context_pos_emb = (
            extra_image_context_dim is not None and img_context_num_views > 0
        )
        if self.use_img_context_pos_emb:
            self.img_view_embed = nn.Embedding(img_context_num_views, model_channels)
            self.img_time_embed = nn.Embedding(img_context_max_temporal, model_channels)
            self.img_h_embed = nn.Embedding(img_context_spatial_h, model_channels)
            self.img_w_embed = nn.Embedding(img_context_spatial_w, model_channels)

        if use_crossattn_projection:
            self.crossattn_proj = nn.Sequential(
                nn.Linear(crossattn_proj_in_channels, crossattn_emb_channels, bias=True),
                nn.GELU(),
            )

        if self.num_frame_type_embeddings > 0:
            self.frame_type_embed = nn.Embedding(self.num_frame_type_embeddings, model_channels)
        if self.num_dataset_embeddings > 0:
            self.dataset_embed = nn.Embedding(
                self.num_dataset_embeddings, model_channels
            )

        self.init_weights()
        self.enable_selective_checkpoint(sac_config, self.blocks)

        # Replace self-attention with sparse attention if specified
        if n_dense_blocks != -1:
            self = replace_selfattn_op_with_sparse_attn_op(self, n_dense_blocks, natten_parameters=natten_parameters)

        self._is_context_parallel_enabled = False

    def init_weights(self):
        self.x_embedder.init_weights()
        if self.side_condition_channels > 0:
            torch.nn.init.zeros_(self.side_condition_embedder.proj[1].weight)
        self.pos_embedder.reset_parameters()
        if self.extra_per_block_abs_pos_emb:
            self.extra_pos_embedder.reset_parameters()

        self.t_embedder[1].init_weights()
        for block in self.blocks:
            block.init_weights()

        self.final_layer.init_weights()
        self.t_embedding_norm.reset_parameters()

        if self.extra_image_context_dim is not None:
            self.img_context_proj[0].reset_parameters()

        if getattr(self, "use_img_context_pos_emb", False):
            for emb in (
                self.img_view_embed,
                self.img_time_embed,
                self.img_h_embed,
                self.img_w_embed,
            ):
                torch.nn.init.trunc_normal_(emb.weight, std=0.02)

        # Zero-init the frame-type embeddings so they are a no-op at iter 0
        # (warm-start checkpoints reproduce the pretrained backbone exactly).
        if getattr(self, "num_frame_type_embeddings", 0) > 0:
            torch.nn.init.zeros_(self.frame_type_embed.weight)
        if getattr(self, "num_dataset_embeddings", 0) > 0:
            torch.nn.init.zeros_(self.dataset_embed.weight)

    def build_patch_embed(self):
        (
            concat_padding_mask,
            in_channels,
            patch_spatial,
            patch_temporal,
            model_channels,
        ) = (
            self.concat_padding_mask,
            self.in_channels,
            self.patch_spatial,
            self.patch_temporal,
            self.model_channels,
        )
        in_channels = in_channels + 1 if concat_padding_mask else in_channels
        self.x_embedder = PatchEmbed(
            spatial_patch_size=patch_spatial,
            temporal_patch_size=patch_temporal,
            in_channels=in_channels,
            out_channels=model_channels,
        )
        if self.side_condition_channels > 0:
            self.side_condition_embedder = PatchEmbed(
                spatial_patch_size=patch_spatial,
                temporal_patch_size=patch_temporal,
                in_channels=self.side_condition_channels,
                out_channels=model_channels,
            )

    def build_pos_embed(self):
        if self.pos_emb_cls == "rope3d":
            cls_type = VideoRopePosition3DEmb
        else:
            raise ValueError(f"Unknown pos_emb_cls {self.pos_emb_cls}")

        log.debug(f"Building positional embedding with {self.pos_emb_cls} class, impl {cls_type}")
        kwargs = dict(
            model_channels=self.model_channels,
            len_h=self.max_img_h // self.patch_spatial_h,
            len_w=self.max_img_w // self.patch_spatial_w,
            len_t=self.max_frames // self.patch_temporal,
            max_fps=self.max_fps,
            min_fps=self.min_fps,
            is_learnable=self.pos_emb_learnable,
            interpolation=self.pos_emb_interpolation,
            head_dim=self.model_channels // self.num_heads,
            h_extrapolation_ratio=self.rope_h_extrapolation_ratio,
            w_extrapolation_ratio=self.rope_w_extrapolation_ratio,
            t_extrapolation_ratio=self.rope_t_extrapolation_ratio,
            h_position_stride=(
                self.patch_spatial_h
                if self.rope_scale_positions_by_patch
                else 1
            ),
            w_position_stride=(
                self.patch_spatial_w
                if self.rope_scale_positions_by_patch
                else 1
            ),
            enable_fps_modulation=self.rope_enable_fps_modulation,
        )
        self.pos_embedder = cls_type(
            **kwargs,
        )

        if self.extra_per_block_abs_pos_emb:
            kwargs["h_extrapolation_ratio"] = self.extra_h_extrapolation_ratio
            kwargs["w_extrapolation_ratio"] = self.extra_w_extrapolation_ratio
            kwargs["t_extrapolation_ratio"] = self.extra_t_extrapolation_ratio
            self.extra_pos_embedder = LearnablePosEmbAxis(
                **kwargs,
            )

    def prepare_embedded_sequence(
        self,
        x_B_C_T_H_W: torch.Tensor,
        side_condition_B_C_T_H_W: Optional[torch.Tensor] = None,
        fps: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        temporal_positions: Optional[torch.Tensor] = None,
        frame_type_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Prepares an embedded sequence tensor by applying positional embeddings and handling padding masks.

        Args:
            x_B_C_T_H_W (torch.Tensor): video
            fps (Optional[torch.Tensor]): Frames per second tensor to be used for positional embedding when required.
                                    If None, a default value (`self.base_fps`) will be used.
            padding_mask (Optional[torch.Tensor]): current it is not used

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]:
                - A tensor of shape (B, T, H, W, D) with the embedded sequence.
                - An optional positional embedding tensor, returned only if the positional embedding class
                (`self.pos_emb_cls`) includes 'rope'. Otherwise, None.

        Notes:
            - If `self.concat_padding_mask` is True, a padding mask channel is concatenated to the input tensor.
            - The method of applying positional embeddings depends on the value of `self.pos_emb_cls`.
            - If 'rope' is in `self.pos_emb_cls` (case insensitive), the positional embeddings are generated using
                the `self.pos_embedder` with the shape [T, H, W].
            - If "fps_aware" is in `self.pos_emb_cls`, the positional embeddings are generated using the
            `self.pos_embedder` with the fps tensor.
            - Otherwise, the positional embeddings are generated without considering fps.
        """
        if self.concat_padding_mask:
            padding_mask = transforms.functional.resize(
                padding_mask, list(x_B_C_T_H_W.shape[-2:]), interpolation=transforms.InterpolationMode.NEAREST
            )
            # A FP32 mask must not promote BF16 latents before patch projection.
            padding_mask = padding_mask.to(x_B_C_T_H_W)
            x_B_C_T_H_W = torch.cat(
                [x_B_C_T_H_W, padding_mask.unsqueeze(1).repeat(1, 1, x_B_C_T_H_W.shape[2], 1, 1)], dim=1
            )
        x_B_T_H_W_D = self.x_embedder(x_B_C_T_H_W)
        if self.side_condition_channels > 0:
            if side_condition_B_C_T_H_W is None:
                raise ValueError(
                    "side_condition must be provided when "
                    f"side_condition_channels={self.side_condition_channels}"
                )
            if side_condition_B_C_T_H_W.ndim != 5:
                raise ValueError(
                    "side_condition must be (B,C,T,H,W), got "
                    f"{tuple(side_condition_B_C_T_H_W.shape)}"
                )
            expected = (
                x_B_C_T_H_W.shape[0],
                self.side_condition_channels,
                x_B_C_T_H_W.shape[2],
                x_B_C_T_H_W.shape[3],
                x_B_C_T_H_W.shape[4],
            )
            if tuple(side_condition_B_C_T_H_W.shape) != expected:
                raise ValueError(
                    f"side_condition {tuple(side_condition_B_C_T_H_W.shape)} "
                    f"must be {expected}"
                )
            x_B_T_H_W_D = x_B_T_H_W_D + self.side_condition_embedder(
                side_condition_B_C_T_H_W.type_as(x_B_C_T_H_W)
            )
        elif side_condition_B_C_T_H_W is not None:
            raise ValueError(
                "side_condition was provided but side_condition_channels=0"
            )

        # Zero-init per-frame type embedding (lidar vs RGB view); broadcast over
        # the H, W token grid. No-op at iter 0 (zero weights).
        if frame_type_ids is not None:
            assert getattr(self, "num_frame_type_embeddings", 0) > 0, (
                "frame_type_ids given but the model was built without frame-type embeddings"
            )
            T_tok = x_B_T_H_W_D.shape[1]
            assert frame_type_ids.shape[0] == T_tok, (
                f"frame_type_ids {tuple(frame_type_ids.shape)} must have length T={T_tok}"
            )
            ft = self.frame_type_embed(frame_type_ids.to(x_B_T_H_W_D.device))  # (T, D)
            x_B_T_H_W_D = x_B_T_H_W_D + ft[None, :, None, None, :].type_as(x_B_T_H_W_D)

        if self.extra_per_block_abs_pos_emb:
            extra_pos_emb = self.extra_pos_embedder(x_B_T_H_W_D, fps=fps, temporal_positions=temporal_positions)
        else:
            extra_pos_emb = None

        if "rope" in self.pos_emb_cls.lower():
            return (
                x_B_T_H_W_D,
                self.pos_embedder(x_B_T_H_W_D, fps=fps, temporal_positions=temporal_positions),
                extra_pos_emb,
            )
        x_B_T_H_W_D = x_B_T_H_W_D + self.pos_embedder(x_B_T_H_W_D)  # [B, T, H, W, D]

        return x_B_T_H_W_D, None, extra_pos_emb

    @staticmethod
    def _build_block_attn_mask(
        frame_is_rgb: torch.Tensor,
        T: int,
        H: int,
        W: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Block self-attention mask over the flattened ``(T H W)`` sequence.

        RGB query tokens attend ONLY to RGB key tokens; lidar query tokens
        attend to everything: ``allow[q, k] = is_lidar[t_q] OR is_rgb[t_k]``.
        Returns a bool ``(1, 1, S, S)`` mask (True = attend) broadcastable to
        ``(B, n_heads, S, S)`` for ``scaled_dot_product_attention``.
        """
        assert frame_is_rgb.shape[0] == T, (
            f"frame_is_rgb {tuple(frame_is_rgb.shape)} must have length T={T}"
        )
        tpf = H * W
        is_rgb = frame_is_rgb.to(device=device, dtype=torch.bool)
        token_is_rgb = is_rgb.repeat_interleave(tpf)  # (S,)
        token_is_lidar = ~token_is_rgb
        allow = token_is_lidar[:, None] | token_is_rgb[None, :]  # (S, S)
        return allow[None, None]

    def _build_block_flex_mask(
        self,
        frame_is_rgb: torch.Tensor,
        T: int,
        H: int,
        W: int,
        device: torch.device,
    ) -> BlockMask:
        """Block-sparse FlexAttention version of ``_build_block_attn_mask``.

        Same semantics (``allow[q, k] = is_lidar[t_q] OR is_rgb[t_k]``) but as a
        ``BlockMask`` so ``flex_attention_op`` runs a fused (flash-like) kernel
        and skips fully-masked blocks instead of materializing a dense
        ``(S, S)`` score matrix through ``scaled_dot_product_attention``.

        The mask is a pure function of the (constant) frame layout, so it is
        built once and cached; ``flex_attention_op`` right-pads Q/KV to a
        multiple of 128, so the mask is built at the padded length with a
        validity guard that matches that padding.
        """
        assert frame_is_rgb.shape[0] == T, (
            f"frame_is_rgb {tuple(frame_is_rgb.shape)} must have length T={T}"
        )
        cache = getattr(self, "_flex_block_mask_cache", None)
        if cache is None:
            cache = {}
            self._flex_block_mask_cache = cache
        key = (T, H, W, tuple(frame_is_rgb.to(torch.bool).tolist()))
        cached = cache.get(key)
        if cached is not None:
            return cached

        S = T * H * W
        S_pad = ((S + 127) // 128) * 128
        tpf = H * W
        is_rgb = frame_is_rgb.to(device=device, dtype=torch.bool)
        token_is_rgb = is_rgb.repeat_interleave(tpf)  # (S,)
        if S_pad > S:
            token_is_rgb = torch.cat(
                [token_is_rgb, token_is_rgb.new_zeros(S_pad - S)]
            )
        token_is_lidar = ~token_is_rgb

        def mask_mod(b, h, q_idx, kv_idx):
            valid = (q_idx < S) & (kv_idx < S)
            return valid & (token_is_lidar[q_idx] | token_is_rgb[kv_idx])

        block_mask = create_block_mask(
            mask_mod, B=None, H=None, Q_LEN=S_pad, KV_LEN=S_pad,
            device=device, _compile=True,
        )
        cache[key] = block_mask
        return block_mask

    def _build_temporal_block_flex_mask(
        self,
        T: int,
        H: int,
        W: int,
        device: torch.device,
    ) -> BlockMask:
        """Build a cached block-causal mask over flattened ``(T,H,W)`` tokens.

        An optional prefix attends bidirectionally within itself. All later
        tokens attend to that prefix; within the future, a query in block ``i``
        attends to key blocks ``0..i`` only. Padding tokens added by
        ``flex_attention_op`` are excluded by the validity guard.
        """
        block_size = self.temporal_block_size
        assert block_size > 0
        cache = getattr(self, "_flex_block_mask_cache", None)
        if cache is None:
            cache = {}
            self._flex_block_mask_cache = cache
        prefix_frames = int(getattr(self, "temporal_prefix_frames", 0))
        if prefix_frames > T:
            raise ValueError(
                f"temporal_prefix_frames={prefix_frames} exceeds T={T}"
            )
        key = (
            "temporal_causal",
            T,
            H,
            W,
            block_size,
            prefix_frames,
            str(device),
        )
        cached = cache.get(key)
        if cached is not None:
            return cached

        S = T * H * W
        S_pad = ((S + 127) // 128) * 128
        tokens_per_frame = H * W

        def mask_mod(b, h, q_idx, kv_idx):
            valid = (q_idx < S) & (kv_idx < S)
            q_frame = q_idx // tokens_per_frame
            kv_frame = kv_idx // tokens_per_frame
            return valid & temporal_prefix_block_allows(
                q_frame,
                kv_frame,
                block_size,
                prefix_frames,
            )

        block_mask = create_block_mask(
            mask_mod,
            B=None,
            H=None,
            Q_LEN=S_pad,
            KV_LEN=S_pad,
            device=device,
            _compile=True,
        )
        cache[key] = block_mask
        return block_mask

    def _build_teacher_forcing_flex_mask(
        self,
        clean_frames: int,
        T: int,
        H: int,
        W: int,
        device: torch.device,
    ) -> BlockMask:
        """Build the one-pass ``[clean | noisy]`` teacher-forcing mask."""
        if T != 2 * clean_frames:
            raise ValueError(
                "one-pass teacher forcing requires T=2*clean_frames, got "
                f"T={T}, clean_frames={clean_frames}"
            )
        block_size = self.temporal_block_size
        if block_size <= 0:
            raise ValueError(
                "one-pass teacher forcing requires temporal_block_size > 0"
            )
        prefix_frames = int(self.temporal_prefix_frames)
        cache = getattr(self, "_flex_block_mask_cache", None)
        if cache is None:
            cache = {}
            self._flex_block_mask_cache = cache
        key = (
            "teacher_forcing_clean_noisy",
            clean_frames,
            H,
            W,
            block_size,
            prefix_frames,
            str(device),
        )
        cached = cache.get(key)
        if cached is not None:
            return cached

        S = T * H * W
        S_pad = ((S + 127) // 128) * 128
        tokens_per_frame = H * W

        def mask_mod(b, h, q_idx, kv_idx):
            valid = (q_idx < S) & (kv_idx < S)
            q_frame = q_idx // tokens_per_frame
            kv_frame = kv_idx // tokens_per_frame
            return valid & teacher_forcing_clean_noisy_allows(
                q_frame,
                kv_frame,
                clean_frames,
                block_size,
                prefix_frames,
            )

        block_mask = create_block_mask(
            mask_mod,
            B=None,
            H=None,
            Q_LEN=S_pad,
            KV_LEN=S_pad,
            device=device,
            _compile=True,
        )
        cache[key] = block_mask
        return block_mask

    def enable_zero_crossattn_skip(self) -> int:
        """Bypass a provably zero, bias-free text cross-attention residual."""
        if self.use_crossattn_projection:
            raise ValueError(
                "cannot skip zero cross-attention when a potentially biased "
                "cross-attention projection is enabled"
            )
        if self.extra_image_context_dim is not None:
            raise ValueError(
                "cannot skip cross-attention when image-context tokens are enabled"
            )
        self.skip_crossattn_when_zero = True
        self._zero_crossattn_validated = False
        enabled = 0
        # ``sac_config`` can wrap blocks in PyTorch checkpoint wrappers whose
        # implementation is not exposed as ``.module``. Walk each wrapper's
        # registered submodules so the actual Block receives the runtime flag.
        for wrapped_block in self.blocks:
            implementations = [
                module
                for module in wrapped_block.modules()
                if isinstance(module, Block)
            ]
            if len(implementations) != 1:
                raise RuntimeError(
                    "expected exactly one MiniTrainDIT Block under each "
                    f"registered block, found {len(implementations)} in "
                    f"{type(wrapped_block).__name__}"
                )
            implementations[0].skip_crossattn = True
            enabled += 1
        if enabled != self.num_blocks:
            raise RuntimeError(
                f"enabled zero cross-attention skip on {enabled} blocks, "
                f"expected {self.num_blocks}"
            )
        return enabled

    def validate_zero_crossattn_skip(
        self,
        crossattn_emb: torch.Tensor,
        img_context_emb: torch.Tensor | None = None,
    ) -> None:
        """Validate the zero-context shortcut once, outside compiled forward."""
        if (
            not self.skip_crossattn_when_zero
            or self._zero_crossattn_validated
        ):
            return
        if img_context_emb is not None:
            raise ValueError(
                "zero-cross-attention bypass received image context"
            )
        nonzero = int(torch.count_nonzero(crossattn_emb).item())
        if nonzero:
            max_abs = float(crossattn_emb.abs().max().item())
            raise ValueError(
                "zero-cross-attention bypass requires an exactly zero "
                f"text embedding, got {nonzero} nonzero values "
                f"(max_abs={max_abs})"
            )
        self._zero_crossattn_validated = True

    def disable_selective_checkpoint_for_inference(self) -> int:
        """Remove activation-checkpoint wrappers before eval compilation."""
        if self.training:
            raise RuntimeError(
                "selective checkpoint wrappers may only be removed in eval mode"
            )

        unwrapped = 0
        for block_id, block in self.blocks.named_children():
            implementation = block
            while hasattr(implementation, "_checkpoint_wrapped_module"):
                implementation = implementation._checkpoint_wrapped_module
                unwrapped += 1
            if implementation is not block:
                self.blocks.register_module(block_id, implementation)

        final_layer = self.final_layer
        while hasattr(final_layer, "_checkpoint_wrapped_module"):
            final_layer = final_layer._checkpoint_wrapped_module
            unwrapped += 1
        if final_layer is not self.final_layer:
            self.register_module("final_layer", final_layer)
        return unwrapped

    def set_cached_inference_self_attention_backend(
        self, backend: str
    ) -> int:
        """Select self-attention for a mask-free inference forward.

        Causal video checkpoints use this only with their external KV cache.
        Fully bidirectional checkpoints are also mask-free, so their native
        ``minimal_a2a`` attention can be replaced directly at inference.
        """
        mask_free_native = (
            self.temporal_block_size == 0
            and self.atten_backend == "minimal_a2a"
        )
        if self.atten_backend != "torch-flex" and not mask_free_native:
            raise ValueError(
                "inference backend override expects either a torch-flex "
                "causal checkpoint or a mask-free minimal_a2a bidirectional "
                f"checkpoint, got backend={self.atten_backend!r}, "
                f"temporal_block_size={self.temporal_block_size}"
            )
        implementations = {
            "flex": ("torch-flex", flex_attention_op),
            "sdpa-flash": ("torch", torch_flash_attention_op),
            "fa3": ("torch", flash_attention_3_op),
        }
        if backend not in implementations:
            raise ValueError(
                f"unsupported cached inference backend {backend!r}; "
                f"choose from {sorted(implementations)}"
            )
        if backend == "fa3":
            if _flash_attn_3_func is None:
                raise ImportError(
                    "cached backend 'fa3' requires flash-attn-3; install the "
                    "Hopper package from Dao-AILab/flash-attention"
                ) from _FLASH_ATTN_3_IMPORT_ERROR
            if (
                not torch.cuda.is_available()
                or torch.cuda.get_device_capability() != (9, 0)
            ):
                raise RuntimeError(
                    "cached backend 'fa3' requires an SM90 Hopper GPU"
                )
        backend_name, attention_op = implementations[backend]
        enabled = 0
        for wrapped_block in self.blocks:
            blocks = [
                module
                for module in wrapped_block.modules()
                if isinstance(module, Block)
            ]
            if len(blocks) != 1:
                raise RuntimeError(
                    "expected exactly one MiniTrainDIT Block under each "
                    f"registered block, found {len(blocks)} in "
                    f"{type(wrapped_block).__name__}"
                )
            self_attn = blocks[0].self_attn
            self_attn.backend = backend_name
            if "attn_op" in self_attn._modules:
                del self_attn._modules["attn_op"]
            self_attn.attn_op = attention_op
            enabled += 1
        if enabled != self.num_blocks:
            raise RuntimeError(
                f"set cached self-attention backend on {enabled} blocks, "
                f"expected {self.num_blocks}"
            )
        self._cached_inference_self_attention_backend = backend
        return enabled

    def unpatchify(self, x_B_T_H_W_M):
        x_B_C_Tt_Hp_Wp = rearrange(
            x_B_T_H_W_M,
            "B T H W (p1 p2 t C) -> B C (T t) (H p1) (W p2)",
            p1=self.patch_spatial_h,
            p2=self.patch_spatial_w,
            t=self.patch_temporal,
        )
        return x_B_C_Tt_Hp_Wp

    def forward(
        self,
        x_B_C_T_H_W: torch.Tensor,
        timesteps_B_T: torch.Tensor,
        crossattn_emb: torch.Tensor,
        side_condition_B_C_T_H_W: Optional[torch.Tensor] = None,
        fps: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        data_type: Optional[DataType] = DataType.VIDEO,
        intermediate_feature_ids: Optional[List[int]] = None,
        img_context_emb: Optional[torch.Tensor] = None,
        frame_is_rgb: Optional[torch.Tensor] = None,
        temporal_positions: Optional[torch.Tensor] = None,
        frame_type_ids: Optional[torch.Tensor] = None,
        img_context_pos: Optional[torch.Tensor] = None,
        ego_pose_B_T_P: Optional[torch.Tensor] = None,
        ego_pose_null_mask_B: Optional[torch.Tensor] = None,
        dataset_ids_B: Optional[torch.Tensor] = None,
        self_attn_kv_cache: Optional[
            List[Optional[dict[str, torch.Tensor]]]
        ] = None,
        store_self_attn_kv: bool = False,
        self_attn_kv_max_frames: Optional[int] = None,
        self_attn_kv_sink_frames: int = 0,
        detach_self_attn_kv: bool = True,
        return_self_attn_kv: bool = False,
        teacher_forcing_clean_frames: Optional[int] = None,
    ) -> torch.Tensor | List[torch.Tensor] | Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            x: (B, C, T, H, W) tensor of spatial-temp inputs
            timesteps: (B,) shared or (B,T) per-frame tensor of timesteps
            crossattn_emb: (B, N, D) tensor of cross-attention embeddings
            ego_pose_B_T_P: optional (B, T, ego_pose_dim) raw per-frame ego pose
                (flattened SE(3)). The in-network zero-init ``ego_pose_head``
                (built iff ``ego_pose_dim > 0``) maps it to a per-frame embedding
                ADDED to the timestep embedding that drives every block's AdaLN
                (Copilot4D-style). ``T`` must equal the tokenized temporal length.
            ego_pose_null_mask_B: optional (B,) bool -- True on samples whose ego
                pose should be DROPPED (embedding forced to 0). Used for pose
                dropout during training and the unconditional pass of pose CFG.
            dataset_ids_B: optional (B,) long sensor-dataset identity. Required
                when ``num_dataset_embeddings > 0`` and added globally to the
                AdaLN timestep embedding.
            img_context_pos: optional (N, 4) long -- per image-context-token
                [view, pixel_time, h, w] indices into the learnable image-context
                positional embedding tables (view / time / spatial). Added to the
                projected image-context tokens so RGB K/V carry their location.
            frame_is_rgb: optional (T,) bool -- True on RGB *condition* frames.
                When given, a block self-attention mask is built so RGB query
                tokens attend ONLY to RGB key tokens (their KV is then
                noise-independent => cacheable), while lidar query tokens attend
                to everything. Requires the ``torch`` self-attn backend.
            temporal_positions: optional (T,) -- explicit per-frame RoPE temporal
                positions (pixel-time alignment of heterogeneous modalities).
            frame_type_ids: optional (T,) long -- per-frame type id for the
                zero-init frame-type embedding (lidar vs RGB view).
            return_self_attn_kv: return the updated external cache alongside
                the prediction. This is needed when an FSDP boundary rebuilds
                nested input containers instead of mutating the caller's list.
        """
        assert isinstance(data_type, DataType), (
            f"Expected DataType, got {type(data_type)}. We need discuss this flag later."
        )
        if (
            self.skip_crossattn_when_zero
            and not self._zero_crossattn_validated
        ):
            self.validate_zero_crossattn_skip(
                crossattn_emb,
                img_context_emb,
            )
        x_B_T_H_W_D, rope_emb_L_1_1_D, extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D = self.prepare_embedded_sequence(
            x_B_C_T_H_W,
            side_condition_B_C_T_H_W=side_condition_B_C_T_H_W,
            fps=fps,
            padding_mask=padding_mask,
            temporal_positions=temporal_positions,
            frame_type_ids=frame_type_ids,
        )

        # Build one optional self-attention mask. Diffusion-Forcing temporal
        # causality and heterogeneous RGB/LiDAR modality masking are separate
        # modes for now; combining both is deliberately rejected rather than
        # silently weakening either constraint.
        self_attn_mask = None
        T_tok, H_tok, W_tok = x_B_T_H_W_D.shape[1:4]
        if teacher_forcing_clean_frames is not None:
            if self_attn_kv_cache is not None:
                raise ValueError(
                    "one-pass teacher forcing is incompatible with a KV cache"
                )
            if frame_is_rgb is not None:
                raise ValueError(
                    "one-pass teacher forcing does not support the RGB/LiDAR "
                    "modality mask"
                )
            if self.is_context_parallel_enabled:
                raise ValueError(
                    "one-pass teacher forcing does not support context parallelism"
                )
            if self._cached_inference_self_attention_backend != "flex":
                raise RuntimeError(
                    "one-pass teacher forcing requires the torch-flex backend"
                )
            self_attn_mask = self._build_teacher_forcing_flex_mask(
                int(teacher_forcing_clean_frames),
                T_tok,
                H_tok,
                W_tok,
                device=x_B_T_H_W_D.device,
            )
        elif self_attn_kv_cache is not None:
            if self.temporal_block_size <= 0:
                raise ValueError(
                    "DiT KV caching requires temporal_block_size > 0"
                )
            if frame_is_rgb is not None:
                raise ValueError(
                    "DiT KV caching does not support the RGB/LiDAR modality mask"
                )
            if self.is_context_parallel_enabled:
                raise ValueError(
                    "DiT KV caching does not support context parallelism"
                )
            if len(self_attn_kv_cache) != len(self.blocks):
                raise ValueError(
                    f"expected {len(self.blocks)} per-layer KV caches, got "
                    f"{len(self_attn_kv_cache)}"
                )
            valid_stream_chunk_sizes = {self.temporal_block_size}
            if self.temporal_prefix_frames > 0:
                valid_stream_chunk_sizes.add(self.temporal_prefix_frames)
            if T_tok not in valid_stream_chunk_sizes:
                raise ValueError(
                    "streaming calls must contain the temporal prefix or one "
                    "complete causal block: "
                    f"T={T_tok}, prefix={self.temporal_prefix_frames}, "
                    f"block_size={self.temporal_block_size}"
                )
            if temporal_positions is None:
                raise ValueError(
                    "streaming calls require explicit global temporal_positions"
                )
            if self_attn_kv_sink_frames < 0:
                raise ValueError(
                    "self_attn_kv_sink_frames must be non-negative, got "
                    f"{self_attn_kv_sink_frames}"
                )
            if self_attn_kv_max_frames is None:
                if self_attn_kv_sink_frames:
                    raise ValueError(
                        "self-attention sink frames require a finite cache"
                    )
            elif self_attn_kv_sink_frames >= self_attn_kv_max_frames:
                raise ValueError(
                    "self-attention sink frames must be smaller than the "
                    "persistent cache: "
                    f"{self_attn_kv_sink_frames}/"
                    f"{self_attn_kv_max_frames}"
                )
            # No mask is needed: each query sees only cached earlier blocks and
            # its complete current block (which is bidirectional by definition).
        elif self.temporal_block_size > 0:
            if self._cached_inference_self_attention_backend != "flex":
                raise RuntimeError(
                    "the cached inference self-attention backend override "
                    "cannot run a full block-causal forward; restore backend "
                    "'flex' or supply self_attn_kv_cache"
                )
            assert frame_is_rgb is None, (
                "temporal block-causal and RGB/LiDAR modality masks cannot be "
                "combined in the current implementation"
            )
            self_attn_mask = self._build_temporal_block_flex_mask(
                T_tok, H_tok, W_tok, device=x_B_T_H_W_D.device
            )
        elif frame_is_rgb is not None:
            if getattr(self, "atten_backend", "torch") == "torch-flex":
                # Block-sparse fused kernel -- no dense (S, S) mask.
                self_attn_mask = self._build_block_flex_mask(
                    frame_is_rgb, T_tok, H_tok, W_tok, device=x_B_T_H_W_D.device
                )
            else:
                self_attn_mask = self._build_block_attn_mask(
                    frame_is_rgb, T_tok, H_tok, W_tok, device=x_B_T_H_W_D.device
                )

        # Training batches may keep conditioning tensors in FP32.
        crossattn_emb = crossattn_emb.to(x_B_T_H_W_D)
        if self.use_crossattn_projection:
            crossattn_emb = self.crossattn_proj(crossattn_emb)

        if img_context_emb is not None:
            assert self.extra_image_context_dim is not None, (
                "extra_image_context_dim must be set if img_context_emb is provided"
            )
            img_context_emb = self.img_context_proj(img_context_emb)
            if img_context_pos is not None and getattr(self, "use_img_context_pos_emb", False):
                pos = img_context_pos.to(device=img_context_emb.device, dtype=torch.long)
                pos_emb = (
                    self.img_view_embed(pos[:, 0])
                    + self.img_time_embed(pos[:, 1])
                    + self.img_h_embed(pos[:, 2])
                    + self.img_w_embed(pos[:, 3])
                )  # (N, D)
                img_context_emb = img_context_emb + pos_emb.unsqueeze(0).type_as(img_context_emb)
            context_input = (crossattn_emb, img_context_emb)
        else:
            context_input = crossattn_emb

        with amp.autocast("cuda", enabled=self.use_wan_fp32_strategy, dtype=torch.float32):
            if timesteps_B_T.ndim == 1:
                timesteps_B_T = timesteps_B_T.unsqueeze(1)
            t_embedding_B_T_D, adaln_lora_B_T_3D = self.t_embedder(timesteps_B_T)
            t_embedding_B_T_D = self.t_embedding_norm(t_embedding_B_T_D)

        # Copilot4D-style additive per-frame ego-pose conditioning: run the raw
        # per-frame pose through the in-network zero-init head, then fold the
        # (B, T, D) pose embedding into the timestep embedding that drives every
        # block's AdaLN. The timestep embedding is usually (B, 1, D) (one sigma
        # per sample, broadcast over frames); adding a per-frame pose embedding
        # makes the whole network's AdaLN per-frame -> expand T first. Zero-init
        # head => iter-0 no-op / warm-start-safe. ``ego_pose_null_mask_B`` (True =
        # drop) zeroes the embedding per sample for pose dropout / CFG uncond.
        if ego_pose_B_T_P is not None and self.ego_pose_dim > 0:
            assert ego_pose_B_T_P.shape[-1] == self.ego_pose_dim, (
                f"ego_pose_B_T_P last dim {ego_pose_B_T_P.shape[-1]} != "
                f"ego_pose_dim {self.ego_pose_dim}"
            )
            # Feed the head in the network's COMPUTE dtype (bf16 under FSDP mixed
            # precision), not t_embedding's dtype -- t_embedding is built in an
            # fp32 autocast block above, so casting the input to it would feed
            # fp32 into bf16 weights ("mat1 and mat2 must have the same dtype").
            # ``x_B_T_H_W_D`` (the embedded hidden state) carries that compute
            # dtype and is always present; the head's own ``.parameters()`` is
            # empty here because FSDP flattens submodule params. Output is cast
            # back to t_embedding's dtype at the add below.
            ego_pose_emb_B_T_D = self.ego_pose_head(
                ego_pose_B_T_P.to(x_B_T_H_W_D.dtype)
            )  # (B, T, D)
            if ego_pose_null_mask_B is not None:
                keep = (~ego_pose_null_mask_B.to(torch.bool)).to(ego_pose_emb_B_T_D.dtype)
                ego_pose_emb_B_T_D = ego_pose_emb_B_T_D * keep.view(-1, 1, 1)
            T_pose = ego_pose_emb_B_T_D.shape[1]
            if t_embedding_B_T_D.shape[1] == 1 and T_pose != 1:
                t_embedding_B_T_D = t_embedding_B_T_D.expand(-1, T_pose, -1)
            assert ego_pose_emb_B_T_D.shape == t_embedding_B_T_D.shape, (
                f"ego_pose_emb {tuple(ego_pose_emb_B_T_D.shape)} must match "
                f"t_embedding {tuple(t_embedding_B_T_D.shape)} (after T-expand)"
            )
            t_embedding_B_T_D = t_embedding_B_T_D + ego_pose_emb_B_T_D.to(t_embedding_B_T_D.dtype)

        if self.num_dataset_embeddings > 0:
            if dataset_ids_B is None:
                raise ValueError(
                    "dataset_ids_B is required when dataset conditioning is enabled"
                )
            dataset_ids_B = dataset_ids_B.to(
                device=x_B_T_H_W_D.device, dtype=torch.long
            ).reshape(-1)
            if dataset_ids_B.shape != (x_B_T_H_W_D.shape[0],):
                raise ValueError(
                    "dataset_ids_B must have one id per sample, got "
                    f"{tuple(dataset_ids_B.shape)} for "
                    f"B={x_B_T_H_W_D.shape[0]}"
                )
            if bool(
                ((dataset_ids_B < 0) | (
                    dataset_ids_B >= self.num_dataset_embeddings
                )).any()
            ):
                raise ValueError(
                    "dataset_ids_B contains an id outside [0, "
                    f"{self.num_dataset_embeddings})"
                )
            dataset_embedding = self.dataset_embed(dataset_ids_B)
            t_embedding_B_T_D = (
                t_embedding_B_T_D
                + dataset_embedding[:, None].to(t_embedding_B_T_D.dtype)
            )
        elif dataset_ids_B is not None:
            raise ValueError(
                "dataset_ids_B was provided but dataset conditioning is disabled"
            )

        # for logging purpose
        affline_scale_log_info = {}
        affline_scale_log_info["t_embedding_B_T_D"] = t_embedding_B_T_D.detach()
        self.affline_scale_log_info = affline_scale_log_info
        self.affline_emb = t_embedding_B_T_D
        self.crossattn_emb = crossattn_emb

        if extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D is not None:
            assert x_B_T_H_W_D.shape == extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D.shape, (
                f"{x_B_T_H_W_D.shape} != {extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D.shape}"
            )

        B, T, H, W, D = x_B_T_H_W_D.shape
        # x_B_THW_D = rearrange(x_B_T_H_W_D, "b t h w d -> b (t h w) d")

        intermediate_features_outputs = []
        for i, block in enumerate(self.blocks):
            kv_cache_cfg = None
            if self_attn_kv_cache is not None:
                max_cache_tokens = None
                if self_attn_kv_max_frames is not None:
                    max_cache_tokens = (
                        int(self_attn_kv_max_frames) * H * W
                    )
                sink_cache_tokens = int(self_attn_kv_sink_frames) * H * W
                kv_cache_cfg = KVCacheConfig(
                    run_with_kv=True,
                    store_kv=store_self_attn_kv,
                    cache=self_attn_kv_cache[i],
                    max_cache_tokens=max_cache_tokens,
                    sink_cache_tokens=sink_cache_tokens,
                    detach_cache=detach_self_attn_kv,
                )
            x_B_T_H_W_D = block(
                x_B_T_H_W_D,
                t_embedding_B_T_D,
                context_input,
                rope_emb_L_1_1_D=rope_emb_L_1_1_D,
                adaln_lora_B_T_3D=adaln_lora_B_T_3D,
                extra_per_block_pos_emb=extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D,
                kv_cache_cfg=kv_cache_cfg,
                self_attn_mask=self_attn_mask,
            )
            if kv_cache_cfg is not None:
                self_attn_kv_cache[i] = kv_cache_cfg.cache
            if intermediate_feature_ids and i in intermediate_feature_ids:
                x_reshaped_for_disc = rearrange(x_B_T_H_W_D, "b tp hp wp d -> b (tp hp wp) d")
                intermediate_features_outputs.append(x_reshaped_for_disc)

        # x_B_T_H_W_D = rearrange(x_B_THW_D, "b (t h w) d -> b t h w d", t=T, h=H, w=W)
        # O = out_channels * patch_h * patch_w * temporal_patch_size
        x_B_T_H_W_O = self.final_layer(x_B_T_H_W_D, t_embedding_B_T_D, adaln_lora_B_T_3D=adaln_lora_B_T_3D)
        x_B_C_Tt_Hp_Wp = self.unpatchify(x_B_T_H_W_O)
        if intermediate_feature_ids:
            if return_self_attn_kv:
                raise ValueError(
                    "return_self_attn_kv is incompatible with intermediate "
                    "feature collection"
                )
            if len(intermediate_features_outputs) != len(intermediate_feature_ids):
                log.warning(
                    f"Collected {len(intermediate_features_outputs)} intermediate features, "
                    f"but expected {len(intermediate_feature_ids)}. "
                    f"Requested IDs: {intermediate_feature_ids}"
                )
            return x_B_C_Tt_Hp_Wp, intermediate_features_outputs

        if return_self_attn_kv:
            if self_attn_kv_cache is None:
                raise ValueError(
                    "return_self_attn_kv requires self_attn_kv_cache"
                )
            return x_B_C_Tt_Hp_Wp, self_attn_kv_cache
        return x_B_C_Tt_Hp_Wp

    def enable_selective_checkpoint(self, sac_config: SACConfig, blocks: nn.ModuleList):
        if sac_config.mode == CheckpointMode.NONE:
            return self

        log.info(
            f"Enable selective checkpoint with {sac_config.mode}, for every {sac_config.every_n_blocks} blocks. Total blocks: {len(blocks)}"
        )
        _context_fn = sac_config.get_context_fn()
        for block_id, block in blocks.named_children():
            if int(block_id) % sac_config.every_n_blocks == 0:
                log.info(f"Enable selective checkpoint for block {block_id}")
                block = ptd_checkpoint_wrapper(
                    block,
                    context_fn=_context_fn,
                    preserve_rng_state=False,
                )
                blocks.register_module(block_id, block)
        self.register_module(
            "final_layer",
            ptd_checkpoint_wrapper(
                self.final_layer,
                context_fn=_context_fn,
                preserve_rng_state=False,
            ),
        )

        return self

    def fully_shard(self, mesh, **fsdp_kwargs):
        for i, block in enumerate(self.blocks):
            reshard_after_forward = i < len(self.blocks) - 1
            fully_shard(block, mesh=mesh, reshard_after_forward=reshard_after_forward, **fsdp_kwargs)

        # TODO: (qsh 2025-01-13) shall we do it?
        fully_shard(self.final_layer, mesh=mesh, reshard_after_forward=True, **fsdp_kwargs)
        if self.extra_per_block_abs_pos_emb:
            fully_shard(self.extra_pos_embedder, mesh=mesh, reshard_after_forward=True, **fsdp_kwargs)
        fully_shard(self.t_embedder, mesh=mesh, reshard_after_forward=False, **fsdp_kwargs)
        if self.extra_image_context_dim is not None:
            fully_shard(self.img_context_proj, mesh=mesh, reshard_after_forward=False, **fsdp_kwargs)

    def disable_context_parallel(self):
        # pos_embedder
        self.pos_embedder.disable_context_parallel()
        if self.extra_per_block_abs_pos_emb:
            self.extra_pos_embedder.disable_context_parallel()

        # attention
        for block in self.blocks:
            block.set_context_parallel_group(
                process_group=None,
                ranks=None,
                stream=torch.cuda.Stream(),
            )

        self._is_context_parallel_enabled = False

    def enable_context_parallel(self, process_group: Optional[ProcessGroup] = None):
        # pos_embedder
        self.pos_embedder.enable_context_parallel(process_group=process_group)
        if self.extra_per_block_abs_pos_emb:
            self.extra_pos_embedder.enable_context_parallel(process_group=process_group)

        # attention
        cp_ranks = get_process_group_ranks(process_group)
        for block in self.blocks:
            block.set_context_parallel_group(
                process_group=process_group,
                ranks=cp_ranks,
                stream=torch.cuda.Stream(),
            )

        self._is_context_parallel_enabled = True

    @property
    def is_context_parallel_enabled(self):
        return self._is_context_parallel_enabled


def replace_selfattn_op_with_sparse_attn_op(
    model: MiniTrainDIT, n_dense_blocks: int = 0, natten_parameters: Union[dict, list] = None
) -> MiniTrainDIT:
    """
    Replace the self-attention operator with a sparse self-attention operator.

    Args:
        model: MiniTrainDIT instance
        n_dense_blocks: Number of blocks that will remain dense (not replaced with NeighborhoodAttention)
            If 0, all blocks use NeighborhoodAttention.
            If -1, return model directly without any modifications.
            Otherwise, n_dense_blocks blocks will remain dense, distributed evenly across the network.

    Returns:
        Modified instance
    """
    # Special case: return model directly without modifications
    if n_dense_blocks == -1:
        return model

    num_blocks = len(model.blocks)

    if natten_parameters is None:
        raise ValueError("Please specify natten_parameters when n_dense_blocks > -1.")

    if isinstance(natten_parameters, Sequence) and len(natten_parameters) != num_blocks:
        raise ValueError(
            "List of NATTEN parameters must be the same length as the number of blocks, "
            f"got {len(natten_parameters)=} != {num_blocks=}."
        )

    if isinstance(natten_parameters, Sequence) and n_dense_blocks > 0:
        log.warning(f"NATTEN parameters was a list; ignoring {n_dense_blocks=}.")

    if isinstance(natten_parameters, Sequence):
        natten_parameters_list = natten_parameters
    else:
        if n_dense_blocks >= num_blocks:
            raise ValueError(f"n_dense_blocks ({n_dense_blocks}) must be less than the number of blocks ({num_blocks})")

        # Determine which blocks should remain dense
        dense_indices = set()

        if n_dense_blocks > 0:
            # General rule: distribute n_dense_blocks blocks evenly across the network
            if n_dense_blocks == 1:
                # Special case: just the middle block
                dense_indices.add(num_blocks // 2)
            else:
                # For multiple blocks, distribute them evenly from start to end
                indices = np.linspace(0, num_blocks - 1, n_dense_blocks, dtype=int)
                dense_indices.update(indices.tolist())

        natten_parameters_list = [None if i in dense_indices else natten_parameters for i in range(num_blocks)]

    # Replace self-attention with NeighborhoodAttention for non-dense blocks
    for i, block in enumerate(model.blocks):
        natten_params = natten_parameters_list[i]
        if natten_params is not None:
            natten_parameters_layer = {k: v for k, v in natten_params.items()}
            natten_parameters_layer["layer_id"] = i
            if block.self_attn.backend == "minimal_a2a":
                sparse_attn_op = NattenA2AAttnOp(natten_parameters=natten_parameters_layer)
            else:
                raise NotImplementedError(
                    f"Using sparsity with attention backend {block.self_attn.backend} is not supported."
                )

            block.self_attn.register_module("attn_op", sparse_attn_op)

    return model
