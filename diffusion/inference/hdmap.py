# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HDMap-only diffusion loading, sampling, and output for native sensor grids."""

from __future__ import annotations

import json
import os
import time
from collections import Counter
from copy import deepcopy
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from utils import inference as common
from utils.lidar_rangemap import unnormalize_and_reduce_channels


# Dataset-ID order is a checkpoint contract, not inferred from grid dimensions.
SENSORS = {
    0: ("pandar", 128, 1024, 10.0),
    1: ("pandaset", 64, 1024, 10.0),
    2: ("nuscenes", 32, 1024, 20.0),
}


def resolve_checkpoint(path: Path) -> Path:
    if path.is_file():
        return path.resolve()
    folder = path / "checkpoints" if (path / "checkpoints").is_dir() else path
    marker = folder / "latest_checkpoint.txt"
    stem = marker.read_text().strip().removesuffix(".pt")
    if not stem.startswith("iter_") or not stem[5:].isdigit():
        raise ValueError(f"Invalid checkpoint marker in {marker}: {stem!r}")
    source = folder / f"{stem}_ema_model.pt"
    if not source.is_file() or source.stat().st_size == 0:
        raise FileNotFoundError(f"No complete EMA checkpoint: {source}")
    # Match the research launcher: pin the EMA before training retention unlinks it.
    snapshot_dir = folder.parent / "eval_snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    snapshot = snapshot_dir / source.name
    try:
        os.link(source, snapshot)
    except FileExistsError:
        pass
    return snapshot.resolve()


def load_model(checkpoint: Path, tokenizer_checkpoint: Path, latent_stats: Path):
    from diffusion.training.config.base.lidarverse_vae import get_lidarverse_tokenizer
    from diffusion.training.config.base.model import LidarModelConfig
    from diffusion.training.config.text2world.registry import (
        MININET_2B_Config, VideoConditionerFpsSizePaddingConfig,
    )
    from diffusion.training.models.hdmaptolidar_model_rf import HdmapToLidarRectifiedFlowModel
    from utils.lazy_config import LazyDict

    net = deepcopy(MININET_2B_Config)
    net.update(
        in_channels=128, out_channels=64, patch_spatial=(1, 1),
        rope_scale_positions_by_patch=True, num_dataset_embeddings=3,
        timestep_scale=0.001, rope_h_extrapolation_ratio=3.0,
        rope_w_extrapolation_ratio=3.0, rope_t_extrapolation_ratio=1.0,
        rope_enable_fps_modulation=False, use_wan_fp32_strategy=True,
    )
    net.sac_config.mode = "none"
    vae = get_lidarverse_tokenizer()
    vae.update(checkpoint_path=str(tokenizer_checkpoint), mean_std_fp=str(latent_stats),
               strict_checkpoint=True, use_ema=False)
    # No training experiment registration, dataloaders, FSDP, or extra EMA model.
    config = LidarModelConfig(
        net=net, vae=vae, conditioner=deepcopy(VideoConditionerFpsSizePaddingConfig),
        ema=LazyDict(dict(enabled=False)), latent_shape=[64, 21, 16, 128],
        hdmap_channels=3, additional_concat_ch=64, encode_hdmap_with_vae=True,
        hdmap_dropout_prob=0.1, sigma_data=1.0, shift=5,
        shift_clamp_percentiles=(0.5, 99.9), rf_prediction_type="velocity",
        loss_reduce="mean", input_video_has_prepend=True,
    )
    model = HdmapToLidarRectifiedFlowModel(config)
    model.model.to(dtype=torch.bfloat16)
    # Local/trusted PyTorch checkpoints only. Load strictly: never adapt a head.
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("model", payload)
    if isinstance(payload.get("ema"), dict):
        state = payload["ema"]
    model.model.load_state_dict(state, strict=True)
    del state, payload
    model.cuda().eval()
    model.vae.float()
    print(f"Loaded all diffusion weights strictly: {checkpoint}", flush=True)
    return model


def load_hdmap_sample(path: Path, num_frames: int) -> dict:
    with np.load(path, allow_pickle=False) as payload:
        sample = {key: payload[key] for key in payload.files}
    for key in ("hdmap", "ray_directions", "sensor_dataset_id", "fps"):
        if key not in sample:
            raise ValueError(f"{path}: missing {key}; tokenizer-only NPZs have no HDMap condition")
    dataset_id = int(sample["sensor_dataset_id"])
    if dataset_id not in SENSORS:
        raise ValueError(f"{path}: unknown sensor_dataset_id {dataset_id}")
    sensor, height, width, fps = SENSORS[dataset_id]
    hdmap = np.asarray(sample["hdmap"], dtype=np.float32)
    if hdmap.shape != (num_frames, height, width):
        raise ValueError(f"{path}: expected {num_frames} real {height}x{width} HDMap frames; got {hdmap.shape}")
    if bool(sample.get("has_prepend", False)):
        raise ValueError(f"{path}: store only real frames, not the duplicated VAE prefix")
    if str(sample.get("dataset", sensor)) != sensor or float(sample["fps"]) != fps:
        raise ValueError(f"{path}: sensor name/FPS does not match dataset ID {dataset_id}")
    if float(sample.get("min_range_m", 2.5)) != 2.5 or float(sample.get("max_range_m", 100.0)) != 100.0:
        raise ValueError(f"{path}: this checkpoint uses 2.5--100 m")
    if not np.isfinite(hdmap).all() or hdmap.min() < -1 or hdmap.max() > 1:
        raise ValueError(f"{path}: hdmap must be normalized to [-1,1], invalid cells=-1")
    rays = np.asarray(sample["ray_directions"], dtype=np.float32)
    if rays.shape not in ((height, width, 3), (num_frames, height, width, 3)):
        raise ValueError(f"{path}: rays do not match the native HDMap grid: {rays.shape}")
    if not np.isfinite(rays).all() or not np.allclose(np.linalg.norm(rays, axis=-1), 1, atol=1e-4):
        raise ValueError(f"{path}: ray_directions must be finite unit vectors")
    sample.update(hdmap=hdmap, ray_directions=rays, dataset=sensor,
                  visualization_yaw_deg=common.display_yaw(path, sensor))
    return sample


def _metric_ranges(video):
    # Same validity cutoff (including near/far buffers) as the working launcher.
    ranges, valid = unnormalize_and_reduce_channels(
        video, max_range=100.0, min_range=2.5, min_value=-1.0,
        inverse_depth=False, process_output="mono",
    )
    return ranges[0].float().cpu().numpy(), valid[0].cpu().numpy()


def _preview_frames(sample, generated, valid, output):
    hdmap, hd_valid = _metric_ranges(torch.from_numpy(sample["hdmap"])[None, None])
    rays = np.broadcast_to(sample["ray_directions"], hdmap.shape + (3,))
    for index in range(len(generated)):
        panels = []
        clouds = [("hdmap", "HDMap", hdmap, hd_valid),
                  ("generation", "Generated", generated, valid)]
        for name, title, values, mask in clouds:
            points = common.points_from_ranges(values[index], mask[index], rays[index])
            common.write_ply(output / "ply" / f"{name}_{index:03d}.ply", points)
            panels.append(common.label_image(
                common.render_points(points, yaw_deg=sample["visualization_yaw_deg"]),
                f"{sample['dataset']} | {title}",
            ))
        yield np.concatenate(panels, axis=1)


def _generate_sequence(model, sample, args, seed, console, label):
    dataset_id = int(sample["sensor_dataset_id"])
    torch.cuda.synchronize()
    phase_started = time.perf_counter()
    # The diffusion path sees HDMap + null text, never GT.
    video = torch.from_numpy(sample["hdmap"])[None, None].cuda()
    video = torch.cat([video[:, :, :1], video], dim=2)
    _, _, _, height, width = video.shape
    batch = {
        # The shared preparation API encodes a target to obtain its latent shape.
        # Supply an empty target, never GT. Its latent is not a sampling condition.
        "video": torch.full_like(video, -1),
        "lidar_hdmap": video,
        "is_preprocessed": True,
        "sensor_dataset_id": torch.tensor([dataset_id], device="cuda", dtype=torch.long),
        "fps": torch.tensor([float(sample["fps"])], device="cuda"),
        "num_frames": torch.tensor([args.sample_n_frames], device="cuda"),
        "image_size": torch.tensor([[height, width]], device="cuda"),
        "padding_mask": torch.zeros(1, 1, height, width, device="cuda"),
        "t5_text_embeddings": torch.zeros(1, 512, 1024, device="cuda"),
        "t5_text_mask": torch.zeros(1, 512, device="cuda"),
    }
    batch["t5_text_mask"][:, 0] = 1
    # The VAE samples a Gaussian posterior even in eval mode. Seed preparation
    # too, retaining the reference target-then-HDMap encode/RNG order.
    torch.manual_seed(seed)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        _, shape_latent, _ = model.get_data_and_condition(batch)
    torch.cuda.synchronize()
    encode_seconds = time.perf_counter() - phase_started
    phase_started = time.perf_counter()
    # Both supported samplers make one velocity query per step. Ordinary
    # CFG makes two DiT forwards; CFG=0/1 makes one. The hook observes only
    # completion and MUST return None so it cannot replace network outputs.
    forwards_per_step = 1 if args.guidance in (0.0, 1.0) else 2
    forwards = 0
    with tqdm(total=args.num_steps, desc=f"{label} sampling", unit="step",
              file=console, dynamic_ncols=True, leave=False,
              disable=not console.isatty()) as progress:
        def on_forward(_module, _inputs, _output):
            nonlocal forwards
            forwards += 1
            if forwards % forwards_per_step == 0:
                progress.update(1)

        hook = model.net.register_forward_hook(on_forward)
        try:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                latent = model.generate_samples_from_batch(
                    batch, state_shape=shape_latent.shape[1:], n_sample=1,
                    guidance=args.guidance, seed=seed, num_steps=args.num_steps,
                    sampler=args.sampler, sde_eta=args.sde_eta,
                )
            torch.cuda.synchronize()
        finally:
            hook.remove()
    sampling_seconds = time.perf_counter() - phase_started
    phase_started = time.perf_counter()
    decoded = model.decode(latent.float())
    if decoded.shape != video.shape or not torch.isfinite(decoded).all():
        raise RuntimeError(f"Invalid decoded range video: {tuple(decoded.shape)}")
    generated, valid = _metric_ranges(decoded[:, :, 1:])
    torch.cuda.synchronize()
    decode_seconds = time.perf_counter() - phase_started
    return generated, valid, {
        "encode_seconds": encode_seconds,
        "sampling_seconds": sampling_seconds,
        "decode_seconds": decode_seconds,
    }


@torch.inference_mode()
def run(args, console, say, started):
    sample_root, output_root = args.sample_path.resolve(), args.output_folder
    # Count selected clips without decompressing their range videos. Preserve
    # the original file order and per-sensor seed/index convention.
    samples, selected_counts = [], Counter()
    for path in common.sample_paths(sample_root):
        with np.load(path, allow_pickle=False) as payload:
            dataset_id = int(payload["sensor_dataset_id"])
        if dataset_id not in SENSORS:
            raise ValueError(f"{path}: unknown sensor_dataset_id {dataset_id}")
        if selected_counts[dataset_id] < args.num_samples:
            samples.append(path)
            selected_counts[dataset_id] += 1
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required; use the diffusion-capable container")
    checkpoint = resolve_checkpoint(args.checkpoint.resolve())
    model = load_model(checkpoint, args.tokenizer_checkpoint.resolve(), args.latent_stats.resolve())
    torch.cuda.synchronize()
    startup_seconds = time.perf_counter() - started
    say(f"Ready in {startup_seconds:.1f}s | {checkpoint.name} | {len(samples)} clips, {args.sample_n_frames} frames each")
    metadata = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    metadata.update(resolved_checkpoint=str(checkpoint), shift=5, dataset_ids={k: v[0] for k, v in SENSORS.items()})
    (output_root / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    counts = {key: 0 for key in SENSORS}
    sensor_seconds = Counter()
    records = []
    timing_path = output_root / "timings.json"
    for clip_index, path in enumerate(samples, 1):
        clip_started = time.perf_counter()
        sample = load_hdmap_sample(path, args.sample_n_frames)
        dataset_id = int(sample["sensor_dataset_id"])
        seed = args.seed + int(sample.get("sample_index", counts[dataset_id]))
        counts[dataset_id] += 1
        label = f"[{clip_index:02d}/{len(samples):02d}] {sample['dataset']}/{path.stem}"
        say(f"{label} | encoding, {sample['hdmap'].shape[1]}x{sample['hdmap'].shape[2]}, seed {seed}")
        generated, valid, generation_timing = _generate_sequence(
            model, sample, args, seed, console, label
        )
        encode_seconds = generation_timing["encode_seconds"]
        sampling_seconds = generation_timing["sampling_seconds"]
        decode_seconds = generation_timing["decode_seconds"]
        phase_started = time.perf_counter()
        output = common.sample_output(path, sample_root, output_root)
        output.mkdir(parents=True, exist_ok=True)
        saved = {key: sample[key] for key in (
            "ray_directions", "fps", "sensor_dataset_id", "dataset", "clip_name",
            "lidar_to_world", "pose_sensor_start_end", "frame_indices",
        ) if key in sample}
        np.savez_compressed(output / "generation.npz", range_m=generated, valid_mask=valid,
                            seed=np.int64(seed), **saved)
        save_seconds = time.perf_counter() - phase_started
        phase_started = time.perf_counter()
        if args.vis_pcd:
            with tqdm(_preview_frames(sample, generated, valid, output), total=len(generated),
                      desc=f"{label} rendering", unit="frame", file=console,
                      dynamic_ncols=True, leave=False, disable=not console.isatty()) as preview:
                common.write_video(output / "generation.mp4", preview, float(sample["fps"]))
        render_seconds = time.perf_counter() - phase_started if args.vis_pcd else 0.0
        total_seconds = time.perf_counter() - clip_started
        sensor_seconds[dataset_id] += total_seconds
        remaining = selected_counts[dataset_id] - counts[dataset_id]
        eta = sensor_seconds[dataset_id] / counts[dataset_id] * remaining
        generation_seconds = encode_seconds + sampling_seconds + decode_seconds
        records.append(dict(sample=str(path), dataset=sample["dataset"], seed=seed,
                            encode_seconds=encode_seconds, sampling_seconds=sampling_seconds,
                            decode_seconds=decode_seconds, save_seconds=save_seconds,
                            render_seconds=render_seconds, total_seconds=total_seconds,
                            generation_fps=args.sample_n_frames / generation_seconds))
        report = dict(startup_seconds=startup_seconds, elapsed_seconds=time.perf_counter() - started,
                      completed_clips=len(records), selected_clips=len(samples), samples=records)
        timing_path.with_suffix(".json.tmp").write_text(json.dumps(report, indent=2) + "\n")
        timing_path.with_suffix(".json.tmp").replace(timing_path)
        say(f"  Done {total_seconds:.1f}s | encode {encode_seconds:.1f}s | sample {sampling_seconds:.1f}s"
            f" | decode {decode_seconds:.1f}s | save/render {save_seconds + render_seconds:.1f}s"
            f" | {sample['dataset']} remaining ~{timedelta(seconds=round(eta))}")
        model._store_hdmap_latent(None)
    say(f"Finished {len(records)} clips in {timedelta(seconds=round(time.perf_counter() - started))} (including loading and rendering).")
    say(f"Outputs: {output_root}\nTimings: {timing_path}")
