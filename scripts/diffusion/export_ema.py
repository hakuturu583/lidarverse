#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Package a PAI DiT EMA without training state or source paths."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Trusted research *_ema_model.pt file, not regular weights.")
    parser.add_argument("--output-folder", type=Path, required=True)
    parser.add_argument("--model-type", choices=("bidir", "causal", "sfdmd"), default="bidir")
    parser.add_argument("--sfdmd-voxel-m", type=float, default=0.15)
    parser.add_argument("--sfdmd-encoder-dtype", choices=("float32", "bfloat16"), default="bfloat16")
    args = parser.parse_args()
    if not math.isfinite(args.sfdmd_voxel_m) or args.sfdmd_voxel_m <= 0:
        parser.error("--sfdmd-voxel-m must be finite and positive")
    if args.model_type != "sfdmd" and (args.sfdmd_voxel_m != 0.15 or args.sfdmd_encoder_dtype != "bfloat16"):
        parser.error("SF-DMD overrides require --model-type sfdmd")
    if not args.checkpoint.name.endswith("_ema_model.pt"):
        parser.error("Pass an explicitly extracted *_ema_model.pt checkpoint")
    torch.set_num_threads(4)
    payload = torch.load(str(args.checkpoint), map_location="cpu", weights_only=False, mmap=True)
    state = payload.get("ema", payload.get("model", payload))
    if not isinstance(state, dict) or not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise ValueError("Expected a tensor-only EMA model state")
    expected = {
        "net.x_embedder.proj.1.weight": (2048, 390),
        "net.final_layer.linear.weight": (128, 2048),
        "net.dataset_embed.weight": (3, 2048),
    }
    for key, shape in expected.items():
        if key not in state or tuple(state[key].shape) != shape:
            raise ValueError(f"Not the PAI factorized P1x2 checkpoint: {key} must be {shape}")
    if any(not torch.isfinite(value).all() for value in state.values()):
        raise ValueError("Checkpoint contains non-finite tensors")
    args.output_folder.mkdir(parents=True, exist_ok=True)
    output = args.output_folder / "ema_model.pt"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    temporary = output.with_suffix(".pt.tmp")
    torch.save({"model": state}, temporary)
    restored = torch.load(str(temporary), map_location="cpu", weights_only=True, mmap=True)["model"]
    if set(restored) != set(state) or any(not torch.equal(value, restored[key]) for key, value in state.items()):
        raise RuntimeError("Exported weights differ from the source EMA")
    temporary.replace(output)
    config = dict(
        architecture="pai_factorized_bidir_p1x2", weights="ema",
        stage={"bidir": "stage1", "causal": "stage2", "sfdmd": "stage3"}[args.model_type],
        iteration=int(args.checkpoint.name.removeprefix("iter_").removesuffix("_ema_model.pt")),
        model_channels=2048, num_blocks=28, num_heads=16,
        input_channels=193, output_channels=64, patch_spatial=[1, 2],
        num_dataset_embeddings=3, sensor_dataset_id=0, resolution=[128, 1024],
        min_range_m=2.5, max_range_m=100, max_frames=21, input_has_prepend=False,
        shift=5, shift_clamp_percentiles=None, precision="bfloat16",
        tokenizer="transformer_lidar_video_tokenizer_8x8x1_z64_pandar_2p5to100_1024_ema40k",
        tokenizer_encoder_dtype="float32", tokenizer_decoder_dtype="float32",
        timestep_scale=0.001, rope_scale_positions_by_patch=True,
        rope_h_extrapolation_ratio=3.0, rope_w_extrapolation_ratio=3.0,
        rope_t_extrapolation_ratio=1.0, rope_enable_fps_modulation=False,
        tensors=len(state), tensor_elements=sum(v.numel() for v in state.values()),
        sha256=sha256(output),
    )
    if args.model_type == "causal":
        config.pop("max_frames")
        config.update(
            architecture="pai_factorized_causal_p1x2",
            temporal_block_size=4, temporal_prefix_frames=1,
            training_real_frames=21, window_frames=21,
        )
    if args.model_type == "sfdmd":
        config.pop("max_frames")
        config.update(
            architecture="pai_factorized_sfdmd_p1x2",
            temporal_block_size=4, temporal_prefix_frames=1,
            training_real_frames=21, window_frames=13,
            kv_cache_frames=9, kv_sink_frames=1,
            kv_budget_includes_sink=True, clean_x0_kv_commit=True,
            stream_condition_vae_encode=True, condition_vae_cache_frames=9,
            sampler="sde", num_steps=4, sde_eta=1.0, guidance=1.0,
            sigmas=[1.0, 15.0 / 16.0, 5.0 / 6.0, 5.0 / 8.0, 0.0],
            tokenizer_encoder_dtype=args.sfdmd_encoder_dtype,
            # Factorized rolling geometry uses blockwise_spatial_map_*, not
            # the generic sfdmd_spatial_map_* settings in the training config.
            spatial_cache=dict(
                voxel_policy="persistent_static", voxel_m=args.sfdmd_voxel_m, radius_m=100.0,
                observation_unit="frame", static_min_observations=2,
                include_latest=False, candidate_ttl_updates=1,
                free_space_clear=True, free_space_margin_m=0.75,
                free_space_misses_to_remove=2,
            ),
        )
    (args.output_folder / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"Exported and verified {len(state)} unchanged EMA tensors: {output}", flush=True)


if __name__ == "__main__":
    main()
