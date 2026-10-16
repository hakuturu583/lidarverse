#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Export a tokenizer training checkpoint as deployment-only EMA weights."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import OrderedDict
from pathlib import Path

import torch


def _is_required_buffer(name: str) -> bool:
    return name == "coords" or name.endswith((".rope.freqs_h", ".rope.freqs_w"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _export_ema_state(
    checkpoint: Path,
) -> tuple[OrderedDict[str, torch.Tensor], int | None]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("model", payload)
    if not isinstance(state, dict):
        raise TypeError(f"Expected a state dictionary in {checkpoint}")

    if payload.get("weights") == "ema":
        # Evaluation snapshots have already overlaid EMA onto network.*.
        if not state or any(
            not key.startswith("network.") or not torch.is_tensor(value)
            for key, value in state.items()
        ):
            raise ValueError("Materialized EMA snapshot must contain only network tensors")
        return OrderedDict(
            (key, value.detach().cpu().contiguous()) for key, value in state.items()
        ), payload.get("iteration")

    regular = {
        key[len("network.") :]: value
        for key, value in state.items()
        if key.startswith("network.") and torch.is_tensor(value)
    }
    ema = {
        key[len("ema.network-") :].replace("-", "."): value
        for key, value in state.items()
        if key.startswith("ema.network-") and torch.is_tensor(value)
    }
    if not regular:
        raise ValueError(f"No network.* tensors found in {checkpoint}")
    if not ema:
        raise ValueError(f"No ema.network-* tensors found in {checkpoint}")

    unknown_ema = sorted(set(ema) - set(regular))
    if unknown_ema:
        raise ValueError(f"EMA tensors have no regular counterpart: {unknown_ema[:8]}")

    regular_only = sorted(set(regular) - set(ema))
    unsupported = [name for name in regular_only if not _is_required_buffer(name)]
    if unsupported:
        raise ValueError(
            "Refusing to copy non-EMA trainable tensors into the deployment "
            f"checkpoint: {unsupported[:8]}"
        )

    output = OrderedDict()
    for name in regular:
        tensor = ema.get(name, regular[name])
        output[f"network.{name}"] = tensor.detach().cpu().contiguous()
    iteration = payload.get("iteration") if isinstance(payload, dict) else None
    return output, iteration


def _export_latent_stats(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict):
        mean, std = payload.get("mean"), payload.get("std")
    elif isinstance(payload, (tuple, list)) and len(payload) == 2:
        mean, std = payload
    else:
        raise TypeError(f"Unsupported latent-stat payload in {path}")
    if not torch.is_tensor(mean) or not torch.is_tensor(std):
        raise TypeError(f"Latent mean/std must be tensors in {path}")
    mean = mean.detach().cpu().float().contiguous()
    std = std.detach().cpu().float().contiguous()
    if mean.numel() != 64 or std.numel() != 64:
        raise ValueError(
            f"Expected 64-channel latent statistics, got {mean.shape}/{std.shape}"
        )
    if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
        raise ValueError(f"Latent statistics contain non-finite values in {path}")
    if not torch.all(std > 0):
        raise ValueError(f"Latent standard deviations must be positive in {path}")
    return {"mean": mean, "std": std}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--latent-stats", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preset", choices=("pai",),
                        help="Attach the PAI 40k architecture and sensor contract.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)

    checkpoint = args.checkpoint.resolve()
    latent_stats = args.latent_stats.resolve()
    output_dir = args.output_dir.resolve()
    output_checkpoint = output_dir / "ckpt.pt"
    output_stats = output_dir / "latent_stats.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not latent_stats.is_file():
        raise FileNotFoundError(latent_stats)
    if not args.overwrite:
        for path in (output_checkpoint, output_stats, output_dir / "config.json"):
            if path.exists():
                raise FileExistsError(f"{path} already exists; pass --overwrite")

    output_dir.mkdir(parents=True, exist_ok=True)
    state, iteration = _export_ema_state(checkpoint)
    if any(not torch.isfinite(tensor).all() for tensor in state.values()):
        raise ValueError("Checkpoint contains non-finite tensors")
    if args.preset == "pai" and iteration != 40000:
        raise ValueError("The PAI preset requires the iteration-40k EMA")
    stats = _export_latent_stats(latent_stats)
    temporary_checkpoint = output_checkpoint.with_suffix(".pt.tmp")
    torch.save(state, temporary_checkpoint)
    restored = torch.load(str(temporary_checkpoint), map_location="cpu", weights_only=True, mmap=True)
    if set(restored) != set(state) or any(
        not torch.equal(tensor, restored[key]) for key, tensor in state.items()
    ):
        raise RuntimeError("Exported weights differ from the source EMA")

    temporary_stats = output_stats.with_suffix(".pt.tmp")
    torch.save(stats, temporary_stats)
    restored_stats = torch.load(str(temporary_stats), map_location="cpu", weights_only=True)
    if set(restored_stats) != set(stats) or any(
        not torch.equal(tensor, restored_stats[key]) for key, tensor in stats.items()
    ):
        raise RuntimeError("Exported latent statistics differ from the source")
    temporary_checkpoint.replace(output_checkpoint)
    temporary_stats.replace(output_stats)

    parameter_count = sum(
        tensor.numel()
        for name, tensor in state.items()
        if not _is_required_buffer(name[len("network.") :])
    )
    buffer_count = sum(
        _is_required_buffer(name[len("network.") :]) for name in state
    )
    config = dict(
        stage="tokenizer", weights="ema", iteration=iteration,
        tensors=len(state), tensor_elements=sum(t.numel() for t in state.values()),
        parameters=parameter_count, required_buffers=buffer_count,
        sha256=_sha256(output_checkpoint),
        latent_stats_file="latent_stats.pt", latent_stats_sha256=_sha256(output_stats),
    )
    if args.preset == "pai":
        config.update(
            architecture="CrossDatasetTransformerVAE",
            tokenizer="transformer_lidar_video_tokenizer_8x8x1_z64_pandar_2p5to100_1024_ema40k",
            resolution=[128, 1024], base_resolution=[64, 1024],
            min_range_m=2.5, max_range_m=100.0, sensor_dataset_id=0,
            input_channels=2, output_channels=1, latent_channels=64,
            spatial_compression=[8, 8], temporal_compression=1,
            base_channels=128, depths=[4, 2, 4], num_heads=[4, 16, 16],
            patch_size=[2, 2], window_size=[5, 35], dilation=[1, 1, 1],
            mlp_ratio=3.0, mapping_depth=2, positional_embedding="rope_only",
            temporal_mixer="attention", temporal_downsample=[False, False, False],
            mask_as_input=True, predict_validity=True,
            encoder_dtype="float32", decoder_dtype="float32",
        )
    (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(
        f"Exported {parameter_count:,} EMA parameters and {buffer_count} required "
        f"buffers to {output_checkpoint}"
    )
    if iteration is not None:
        print(f"source training iteration: {int(iteration)}")
    print(f"checkpoint sha256: {config['sha256']}")
    print(f"latent stats sha256: {config['latent_stats_sha256']}")


if __name__ == "__main__":
    main()
