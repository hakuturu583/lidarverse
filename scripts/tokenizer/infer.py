#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Encode and reconstruct native-grid LiDAR samples with LidarVerse."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from utils.inference import (
    MAX_RANGE_M,
    MIN_RANGE_M,
    SENSOR_GRIDS,
    display_yaw,
    label_image,
    load_sample,
    points_from_ranges,
    render_points,
    sample_output,
    sample_paths,
    write_ply,
    write_video,
)


EXPERIMENT = (
    "transformer_vae_8x8x1_z64_cross_dataset3_pandar_pandaset_"
    "nuscenes32_gridrope_mask_pts_scratch_200k"
)
MAX_CHUNK_FRAMES = 32
PAI_TOKENIZER_NAME = (
    "transformer_lidar_video_tokenizer_8x8x1_z64_pandar_2p5to100_1024_ema40k"
)


def _normalize(ranges: np.ndarray, valid: np.ndarray) -> np.ndarray:
    clipped = np.clip(ranges, MIN_RANGE_M, MAX_RANGE_M)
    normalized = (
        2.0 * (clipped - MIN_RANGE_M) / (MAX_RANGE_M - MIN_RANGE_M) - 1.0
    )
    return np.where(valid, normalized, -1.0).astype(np.float32)


def _unnormalize(
    normalized: np.ndarray,
    validity: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    if validity is None:
        valid = normalized > -0.98
    else:
        valid = validity >= 0.5
    valid &= np.isfinite(normalized)
    ranges = (
        (normalized + 1.0) * 0.5 * (MAX_RANGE_M - MIN_RANGE_M)
        + MIN_RANGE_M
    )
    ranges = np.clip(ranges, MIN_RANGE_M, MAX_RANGE_M)
    return np.where(valid, ranges, 0.0).astype(np.float32), valid


def _reconstruct_chunks(tokenizer, ranges, valid, chunk_frames):
    reconstructed, masks = [], []
    for start in range(0, len(ranges), chunk_frames):
        end = min(start + chunk_frames, len(ranges))
        normalized = _normalize(ranges[start:end], valid[start:end])
        # Each independent window matches the training mask and frame-0 prefix.
        video = torch.from_numpy(np.stack(
            [normalized, valid[start:end].astype(np.float32)], axis=0
        ))[None].cuda()
        video = torch.cat([video[:, :, :1], video], dim=2)
        latent = tokenizer.encode(video)
        decoded, validity = tokenizer.decode(latent, return_validity=True)
        if tuple(decoded.shape[2:]) != (end - start + 1, *ranges.shape[1:]):
            raise RuntimeError(f"Unexpected reconstruction shape: {tuple(decoded.shape)}")
        if not torch.isfinite(decoded).all() or (validity is not None and not torch.isfinite(validity).all()):
            raise FloatingPointError(f"Non-finite reconstruction in frames {start}:{end}")
        values = decoded.float().cpu().numpy()[0, 0, 1:]
        confidence = None if validity is None else validity.float().cpu().numpy()[0, 0, 1:]
        restored, keep = _unnormalize(values, confidence)
        reconstructed.append(restored)
        masks.append(keep)
        del video, latent, decoded, validity
    return np.concatenate(reconstructed), np.concatenate(masks)


def _visualization_frames(ranges, valid, rays, recon_ranges, recon_valid, output, sensor, yaw_deg):
    for index in range(len(ranges)):
        gt_points = points_from_ranges(ranges[index], valid[index], rays[index])
        recon_points = points_from_ranges(recon_ranges[index], recon_valid[index], rays[index])
        write_ply(output / "ply" / f"gt_{index:03d}.ply", gt_points)
        write_ply(output / "ply" / f"reconstruction_{index:03d}.ply", recon_points)
        yield np.concatenate([
            label_image(render_points(gt_points, yaw_deg), f"{sensor} | Input"),
            label_image(render_points(recon_points, yaw_deg), f"{sensor} | Reconstruction"),
        ], axis=1)


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-path", "--sample_path", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--latent-stats", "--latent_stats", type=Path, required=True)
    parser.add_argument("--tokenizer-name", default=EXPERIMENT)
    parser.add_argument("--encoder-dtype", choices=("float32", "bfloat16"))
    parser.add_argument("--decoder-dtype", choices=("float32", "bfloat16"))
    parser.add_argument(
        "--output-folder", "--output_folder", type=Path, required=True
    )
    parser.add_argument(
        "--tokenizer-dtype",
        "--tokenizer_dtype",
        choices=("float32", "bfloat16"),
        default="float32",
    )
    parser.add_argument("--vis-pcd", "--vis_pcd", type=int, choices=(0, 1), default=1)
    parser.add_argument(
        "--chunk-frames", type=int, default=21,
        help="Real frames per independent encode/decode window (1-32); bounds GPU memory.",
    )
    parser.add_argument(
        "--fps", type=float, default=None,
        help="Playback rate override; otherwise use sample fps (default: 10).",
    )
    args = parser.parse_args()
    if not 1 <= args.chunk_frames <= MAX_CHUNK_FRAMES:
        parser.error(f"--chunk-frames must be between 1 and {MAX_CHUNK_FRAMES}")

    for required in (args.checkpoint, args.latent_stats):
        if not required.is_file():
            raise FileNotFoundError(required)
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for tokenizer inference")

    from diffusion.training.module.pretrained_vae_base import ConfigBasedVideoVAE

    use_bfloat16 = args.tokenizer_dtype == "bfloat16"
    tokenizer = ConfigBasedVideoVAE(
        experiment_name=args.tokenizer_name,
        checkpoint_path=str(args.checkpoint),
        mean_std_fp=str(args.latent_stats),
        latent_ch=64,
        is_bf16=use_bfloat16,
        encoder_is_bf16=(args.encoder_dtype == "bfloat16") if args.encoder_dtype else None,
        decoder_is_bf16=(args.decoder_dtype == "bfloat16") if args.decoder_dtype else None,
        spatial_compression=8,
        apply_validity_mask=True,
        use_ema=False,
        strict_checkpoint=True,
    ).cuda().eval()

    sample_root = args.sample_path.resolve()
    output_root = args.output_folder.resolve()
    for sample in sample_paths(sample_root):
        ranges, valid, rays, stored_rays = load_sample(sample)
        if args.tokenizer_name == PAI_TOKENIZER_NAME and ranges.shape[-2:] != (128, 1024):
            raise ValueError(f"PAI 40k tokenizer requires 128x1024 inputs: {sample}")
        with np.load(sample, allow_pickle=False) as metadata:
            fps = args.fps if args.fps is not None else float(metadata.get("fps", 10))
            sensor = str(metadata.get("sensor_name", SENSOR_GRIDS[tuple(ranges.shape[-2:])]))
            dataset = str(metadata.get("dataset", ""))
        yaw_deg = display_yaw(sample, dataset)
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError(f"{sample}: fps must be finite and positive")
        output = sample_output(sample, sample_root, output_root)
        output.mkdir(parents=True, exist_ok=True)

        print(f"Reconstructing {sensor}: {len(ranges)} frames at {fps:g} Hz", flush=True)
        recon_ranges, recon_valid = _reconstruct_chunks(
            tokenizer, ranges, valid, args.chunk_frames
        )
        np.savez_compressed(
            output / "reconstruction.npz",
            range_m=recon_ranges,
            valid_mask=recon_valid,
            ray_directions=stored_rays,
            fps=np.float32(fps),
            chunk_start_frames=np.arange(0, len(ranges), args.chunk_frames, dtype=np.int64),
        )

        if args.vis_pcd:
            print(f"Rendering {sensor}", flush=True)
            frames = _visualization_frames(
                ranges, valid, rays, recon_ranges, recon_valid, output, sensor, yaw_deg
            )
            write_video(output / "reconstruction.mp4", frames, fps)
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()
