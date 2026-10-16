# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adapt the Cosmos-Predict2 2B base checkpoint into a LidarVerse warm start.

Input is the ``consolidated.pt`` produced by ``scripts/diffusion/convert_dcp_to_pt.py``
(a raw ``MinimalV1LVGDiT`` state dict with ``net.*`` keys).

Why an adapter (instead of just pointing ``checkpoint.load_path`` at it)
====================================================================
The cosmos base is a 16-ch RGB-video ``MinimalV1LVGDiT`` trained with
reason1.1 cross-attention (a 100352->1024 ``crossattn_proj``). Our lidar
net (``mininet_2b`` = ``LVGDiTForLidar``, same ``MinimalV1LVGDiT`` family)
differs in exactly three places:

  * ``x_embedder``   changes from RGB-video channels to the factorized
                      LiDAR/HDMap/geometry/mask input     -> can't transfer
  * ``final_layer``  out-channels 16 -> 64                -> can't transfer
  * ``crossattn_proj``  absent in our net, which accepts 1024-dimensional
                      conditioning tokens directly. The released range-only
                      recipe supplies a zero text condition.

Everything else transfers: all 28 transformer blocks (self/cross attn,
mlp, adaln), ``t_embedder``, ``pos_embedder``, the norms. The attention
parameter names (``q_proj``/``k_proj``/``v_proj``/``q_norm``/``k_norm``/
``v_norm``) are defined before the attention-backend branch in
``minimal_v4_dit.py``, so TransformerEngine-trained weights drop straight
into the ``minimal_a2a`` net.

At train time LidarVerse loads via ``non_strict_load_model`` (because
``strict_resume=False``), which already (a) skips ``_extra_state``,
(b) *pops* any name-matched-but-shape-mismatched key, and (c) ignores
missing/unexpected keys. So this adapter only has to put the transferable
weights under the right (identity) key namespace; the loader tolerates
the rest. We still drop the three known-untransferable groups here so the
emitted file is clean and the train-time log shows ~0 surprises.

The LidarVerse FSDP loader resolves ``load_path=X.pt`` to ``X_reg_model.pt``
and ``X_ema_model.pt`` if present, else falls back to ``X.pt`` itself for
both the regular and EMA shadow. We therefore emit a single ``X.pt`` so a
fresh finetune initializes *both* reg and EMA from the cosmos weights.

Usage
=====
    python scripts/diffusion/adapt_cosmos2_to_lidar_warmstart.py \
        --ckpt   checkpoints/posttraining/cosmos_diffusion_v2/consolidated.pt \
        --out    checkpoints/posttraining/cosmos_diffusion_v2/cosmos2_2b_lidar_warmstart.pt

Optional ``--verify_against_net`` builds the target lidar net (mininet_2b
with the finetune overrides) on CPU and reports the exact transferred /
fresh-init / leftover key counts -- a strong correctness check, at the
cost of allocating the 2B params once.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from typing import Dict, Tuple

import torch

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

_WRAPPER_KEYS = ("state_dict", "model", "module")
_PREFIXES = ("module.", "_orig_mod.", "_fsdp_wrapped_module.", "net.")


def _unwrap_payload(payload) -> Dict[str, torch.Tensor]:
    state_dict = payload
    for key in _WRAPPER_KEYS:
        if (
            isinstance(state_dict, dict)
            and key in state_dict
            and isinstance(state_dict[key], dict)
        ):
            state_dict = state_dict[key]
    if not isinstance(state_dict, dict):
        raise RuntimeError(
            f"Cannot find a checkpoint state dictionary in {type(payload)}"
        )
    return state_dict


def _strip_prefixes(name: str) -> str:
    changed = True
    while changed:
        changed = False
        for prefix in _PREFIXES:
            if name.startswith(prefix):
                name = name[len(prefix):]
                changed = True
    return name


# Key groups that cannot transfer (channel / conditioning mismatch). Matched
# against the *bare* key (after stripping module./_orig_mod./net. etc.).
_DROP_BARE_PREFIXES: Tuple[str, ...] = (
    "x_embedder.",      # input channels change to factorized LiDAR conditions
    "final_layer.",     # out-channels 16 -> 64
    "crossattn_proj.",  # 100352->1024 proj baked into precomputed embeddings
)

# The LidarVerse diffusion model stores the DiT under ``model.net.*``.
# stores the DiT under: ``model.net.*``. Saved reg/ema checkpoints and
# ``non_strict_load_model`` both use this bare ``net.`` prefix.
_TARGET_PREFIX = "net."


def _adapt(raw_sd: Dict[str, object]) -> Tuple[Dict[str, torch.Tensor], Dict[str, int]]:
    out: Dict[str, torch.Tensor] = {}
    stats: Dict[str, int] = defaultdict(int)
    for k, v in raw_sd.items():
        if k.startswith("net_ema.") or "_ema." in k:
            stats["dropped_ema"] += 1
            continue
        if "_extra_state" in k:
            stats["dropped_extra_state"] += 1
            continue
        if not torch.is_tensor(v):
            stats["dropped_nontensor"] += 1
            continue
        bare = _strip_prefixes(k)
        if any(bare.startswith(p) for p in _DROP_BARE_PREFIXES):
            stats["dropped_mismatch_head"] += 1
            continue
        out[_TARGET_PREFIX + bare] = v
        stats["kept"] += 1
    return out, stats


def _build_target_net():
    """Instantiate ``mininet_2b`` with the finetune overrides (in=193,
    out=64, cosmos rope/timestep_scale) to verify key alignment."""
    import copy

    from diffusion.training.config.text2world.registry import MININET_2B_Config
    from utils.lazy_config import instantiate

    cfg = copy.deepcopy(MININET_2B_Config)
    cfg.in_channels = 193  # 64 latent + 64 HDMap + 64 geometry + mask
    cfg.out_channels = 64
    cfg.timestep_scale = 0.001       # reuse cosmos t_embedder (RF feeds t in [0,1000])
    cfg.rope_h_extrapolation_ratio = 3.0
    cfg.rope_w_extrapolation_ratio = 3.0
    cfg.rope_t_extrapolation_ratio = 1.0
    cfg.rope_enable_fps_modulation = False
    cfg.use_wan_fp32_strategy = True
    return instantiate(cfg)


def _verify(adapted: Dict[str, torch.Tensor]) -> None:
    print("\nBuilding target LiDAR net (mininet_2b, in=193/out=64) on CPU...")
    net = _build_target_net()
    # The diffusion model wraps the net at ``self.net``, so a real LidarVerse
    # reg_model.pt (and the dict ``non_strict_load_model`` compares against)
    # uses the ``net.`` prefix -- match that namespace here so the report
    # reflects the actual train-time load, not the bare standalone net.
    net_sd = {
        _TARGET_PREFIX + k: v
        for k, v in net.state_dict().items()
        if torch.is_tensor(v)
    }
    net_keys = set(net_sd)
    adapted_keys = set(adapted)

    transferred = [k for k in adapted_keys & net_keys
                   if tuple(adapted[k].shape) == tuple(net_sd[k].shape)]
    shape_bad = [k for k in adapted_keys & net_keys
                 if tuple(adapted[k].shape) != tuple(net_sd[k].shape)]
    leftover = sorted(adapted_keys - net_keys)            # in file, not in net
    fresh_init = sorted(net_keys - adapted_keys)          # net uses its own init

    print(f"  net tensor keys:        {len(net_keys)}")
    print(f"  transferred (name+shape): {len(transferred)} "
          f"({sum(adapted[k].numel() for k in transferred) / 1e9:.3f}B params)")
    print(f"  shape-mismatch (loader will pop): {len(shape_bad)}")
    for k in shape_bad[:10]:
        print(f"    {k}  file={tuple(adapted[k].shape)}  net={tuple(net_sd[k].shape)}")
    print(f"  fresh-init in net (expected: x_embedder / final_layer): {len(fresh_init)}")
    for k in fresh_init[:10]:
        print(f"    {k}  {tuple(net_sd[k].shape)}")
    if len(fresh_init) > 10:
        print(f"    ... and {len(fresh_init) - 10} more")
    print(f"  leftover in file, unused by net: {len(leftover)}")
    for k in leftover[:10]:
        print(f"    {k}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--ckpt", required=True, help="consolidated .pt (from convert_dcp_to_pt.py)")
    ap.add_argument("--out", required=True, help="output warm-start .pt (point load_path here)")
    ap.add_argument("--verify_against_net", action="store_true",
                    help="Build the target lidar net and report exact key alignment.")
    args = ap.parse_args()

    if not os.path.isfile(args.ckpt):
        print(f"ERROR: not a file: {args.ckpt}", file=sys.stderr)
        return 1

    print(f"Loading consolidated ckpt: {args.ckpt}")
    payload = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    raw_sd = _unwrap_payload(payload)
    print(f"  raw keys: {len(raw_sd)}")

    adapted, stats = _adapt(raw_sd)
    print("  adapt stats:")
    for k in sorted(stats):
        print(f"    {k:24s} {stats[k]}")
    n_params = sum(t.numel() for t in adapted.values()) / 1e9
    print(f"  kept tensors: {len(adapted)} ({n_params:.3f}B params)")

    if args.verify_against_net:
        _verify(adapted)

    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    torch.save(adapted, args.out)
    print(f"\nWrote warm-start checkpoint: {args.out}")
    print("Point the experiment's checkpoint.load_path at this file with "
          "strict_resume=False, load_training_state=False.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
