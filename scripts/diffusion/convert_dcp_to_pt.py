# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert a sharded PyTorch DCP (Distributed Checkpoint) directory into a
single consolidated ``.pt`` file that this codebase's plain ``torch.load``
based loaders can consume.

The conversion reads ONLY -- no DCP shard is modified or deleted. Output
goes to a new path specified by ``--out``.

Steps:
  1. Initialize a single-process gloo distributed group (required by the
     PyTorch 2.1.x DCP API; we're not actually distributed, just satisfying
     the API contract for ``load_state_dict``).
  2. Read the directory's ``.metadata`` file via ``FileSystemReader``;
     this is the authoritative manifest of every tensor and its
     (key, shape, dtype) -- gather correctness comes from here.
  3. Pre-allocate empty CPU tensors with the metadata shapes/dtypes.
  4. ``dcp.load_state_dict`` populates them from the shards.
  5. ``torch.save({"model": state_dict}, out)`` -- wrapped in {"model": ...}
     because this codebase's ``checkpointer.py`` expects that key.

Usage::

    python scripts/diffusion/convert_dcp_to_pt.py \\
        --dcp_dir checkpoints/posttraining/cosmos_diffusion_v2/model \\
        --out     checkpoints/posttraining/cosmos_diffusion_v2/consolidated.pt
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp


def _patch_metadata_compat() -> None:
    """Cross-version unpickle fix.

    PyTorch >=2.3 pickles ``TensorProperties.memory_format`` using the
    enum ``torch.distributed.checkpoint.metadata._MEM_FORMAT_ENCODING``,
    which doesn't exist on 2.1.2 (the symbol is local to that module's
    ``__getstate__`` closure, not a module attribute). Unpickling
    therefore dies with ``AttributeError: Can't get attribute
    '_MEM_FORMAT_ENCODING'``.

    Fix: synthesize the enum class with the exact 3 members the newer
    code defines, and inject it under BOTH names so:
      (a) pickle.find_class can resolve ``_MEM_FORMAT_ENCODING``;
      (b) the local ``TensorProperties.__setstate__`` (which compares
          against ``MEM_FORMAT_ENCODING.TORCH_*``) sees the same class
          object and decodes ``memory_format`` correctly via identity."""
    import torch.distributed.checkpoint.metadata as m
    from enum import Enum

    if hasattr(m, "_MEM_FORMAT_ENCODING"):
        return

    class _MEM_FORMAT_ENCODING(Enum):
        TORCH_CONTIGUOUS_FORMAT = 0
        TORCH_CHANNELS_LAST = 1
        TORCH_PRESERVE_FORMAT = 2

    _MEM_FORMAT_ENCODING.__module__ = m.__name__
    m._MEM_FORMAT_ENCODING = _MEM_FORMAT_ENCODING
    m.MEM_FORMAT_ENCODING = _MEM_FORMAT_ENCODING


def _init_single_process_dist() -> None:
    """Spin up a 1-rank gloo group so ``dcp.load_state_dict`` works
    outside torchrun. No-op if already initialized."""
    if dist.is_initialized():
        return
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29555")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    dist.init_process_group(backend="gloo", rank=0, world_size=1)


def _allocate_from_metadata(reader: dcp.FileSystemReader) -> dict:
    """Read ``.metadata`` and allocate empty CPU tensors for every
    tensor key it lists. Returns the empty state_dict ready for
    ``dcp.load_state_dict`` to populate in-place."""
    metadata = reader.read_metadata()
    state_dict: dict = {}
    n_tensors, n_bytes = 0, 0
    for key, info in metadata.state_dict_metadata.items():
        # Skip non-tensor metadata entries if any (BytesStorageMetadata).
        if not hasattr(info, "size"):
            continue
        # PyTorch 2.1: info is TensorStorageMetadata with .size (Size)
        # and .properties (TensorProperties with .dtype).
        size = tuple(info.size)
        dtype = info.properties.dtype
        t = torch.empty(size, dtype=dtype)
        state_dict[key] = t
        n_tensors += 1
        n_bytes += t.numel() * t.element_size()
    print(
        f"Pre-allocated {n_tensors} tensors "
        f"({n_bytes / 1e9:.2f} GB) from metadata."
    )
    return state_dict


def _quick_sanity(state_dict: dict, n_samples: int = 5) -> None:
    """Print a few representative tensor stats so we can eyeball that
    nothing is uninitialized or full of NaN/Inf."""
    print("\n=== sanity sample ===")
    keys = sorted(state_dict.keys())
    step = max(1, len(keys) // n_samples)
    for k in keys[::step][:n_samples]:
        t = state_dict[k]
        if not isinstance(t, torch.Tensor):
            continue
        finite = torch.isfinite(t).all().item()
        print(
            f"  {k:60s}  shape={tuple(t.shape)}  dtype={t.dtype}  "
            f"finite={finite}  "
            f"mean={t.float().mean().item():+.4f}  "
            f"std={t.float().std().item():.4f}  "
            f"absmax={t.float().abs().max().item():.4f}"
        )

    total_params = sum(t.numel() for t in state_dict.values()
                       if isinstance(t, torch.Tensor))
    print(f"\nTotal params: {total_params:,}  ({total_params / 1e9:.3f} B)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dcp_dir", required=True,
                        help="Directory containing __*.distcp shards + .metadata")
    parser.add_argument("--out", required=True,
                        help="Output .pt file path (must NOT overlap dcp_dir).")
    parser.add_argument("--wrap_key", default="model",
                        help="Top-level dict key to wrap the state_dict in "
                             "(default 'model', matches checkpointer.py).")
    parser.add_argument("--no_wrap", action="store_true",
                        help="Save the bare state_dict (no {'model': ...} wrap).")
    args = parser.parse_args()

    if not os.path.isdir(args.dcp_dir):
        print(f"ERROR: not a directory: {args.dcp_dir}", file=sys.stderr)
        return 1
    metadata_path = os.path.join(args.dcp_dir, ".metadata")
    if not os.path.exists(metadata_path):
        print(f"ERROR: missing .metadata in {args.dcp_dir}", file=sys.stderr)
        print("  Without this manifest the DCP loader cannot reassemble shards.",
              file=sys.stderr)
        return 1

    # Guard: refuse to write the .pt INSIDE the DCP dir (would mix with shards).
    out_abs = os.path.abspath(args.out)
    dcp_abs = os.path.abspath(args.dcp_dir)
    if out_abs.startswith(dcp_abs + os.sep):
        print(f"ERROR: --out '{args.out}' is inside --dcp_dir '{args.dcp_dir}'. "
              "Pick a path OUTSIDE the DCP directory so original shards "
              "aren't touched.", file=sys.stderr)
        return 1
    if os.path.exists(out_abs):
        print(f"ERROR: --out already exists: {out_abs}  "
              "(refusing to overwrite; delete first if intended).",
              file=sys.stderr)
        return 1
    os.makedirs(os.path.dirname(out_abs) or ".", exist_ok=True)

    print(f"DCP dir: {dcp_abs}")
    print(f"Output:  {out_abs}")

    _patch_metadata_compat()
    _init_single_process_dist()

    reader = dcp.FileSystemReader(args.dcp_dir)
    state_dict = _allocate_from_metadata(reader)

    print("\nLoading shards into pre-allocated state_dict...")
    dcp.load_state_dict(state_dict=state_dict, storage_reader=reader)
    print("Load complete.")

    _quick_sanity(state_dict)

    payload = state_dict if args.no_wrap else {args.wrap_key: state_dict}
    print(f"\nSaving to: {out_abs}")
    torch.save(payload, out_abs)
    print(f"Done. Wrote {os.path.getsize(out_abs) / 1e9:.2f} GB.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
