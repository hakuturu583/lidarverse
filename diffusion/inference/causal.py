# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Causal P1B4 sampling with clean K/V commits and streaming spatial memory."""

import torch
from tqdm import tqdm

from diffusion.inference.hdmap import _metric_ranges
from diffusion.inference.pai import SPATIAL_CACHE, make_batch, mode_spec
from diffusion.training.datasets.lidar_dataset.forecast_warp import TorchSpatialPointMap
from utils import misc


def _spatial_map(model, batch):
    rays, elev, azim, row_off = model._stage2_spatial_map_geometry(batch["video"])
    policy = {key: getattr(model.config, f"blockwise_spatial_map_{key}") for key in SPATIAL_CACHE}
    policy["voxel_size_m"] = policy.pop("voxel_m")
    return TorchSpatialPointMap(
        ray_dirs=rays, row_elevations_rad=elev, column_azimuths_rad=azim,
        row_azimuth_offsets_rad=row_off, device=rays.device,
        min_range=2.5, max_range=100, min_value=-1, inverse_depth=False,
        **policy,
    )


def _commit(model, latent, condition):
    model._current_sfdmd_store_self_attn_kv = True
    try:
        velocity = model._rf_network_forward(
            latent.to(**model.tensor_kwargs),
            torch.zeros((1, latent.shape[2]), device=latent.device), condition,
        )
        if not torch.isfinite(velocity).all():
            raise FloatingPointError("Clean K/V commit produced non-finite values")
    finally:
        model._current_sfdmd_store_self_attn_kv = False


def generate(model, sample, mode, args, seed, console):
    spec = mode_spec(mode, args.stage)
    length = spec["input_frames"] + spec["output_frames"]
    # Complete P1+4N blocks using the extra supplied pose, then crop for display.
    padded_length = 1 + ((length - 1 + 3) // 4) * 4
    batch = make_batch(sample, mode, args.stage, length=padded_length)
    video = batch["video"]
    known = torch.arange(padded_length, device=video.device) < spec["input_frames"]
    torch.manual_seed(seed)
    model._current_stage2_data_batch = batch
    model._current_stage2_geometry_source_mask = known
    model._current_sensor_dataset_ids = batch["sensor_dataset_id"]
    model._current_stage2_geometry_mode = "frozen_object" if mode == "resim" else "rolling_static"
    model.factorized_static_only_geometry = False
    model._hdmap_force_zero_latent = spec["hdmap"] == "null"
    model._current_sfdmd_self_attn_kv_cache = model.new_teacher_forcing_kv_cache()
    model._current_sfdmd_self_attn_kv_max_frames = args.kv_cache_frames
    model._current_sfdmd_self_attn_kv_sink_frames = 1
    model._current_sfdmd_store_self_attn_kv = False
    model._current_sfdmd_detach_self_attn_kv = True
    model._current_sfdmd_return_self_attn_kv = False

    blocks = [(0, 1)] + [(start, start + 4) for start in range(1, padded_length, 4)]
    spatial_map = _spatial_map(model, batch) if mode != "resim" else None
    frozen = None
    if mode == "resim":
        # The training builder reads points only from the observed prefix.
        # Target actor poses place the observed canonical object clouds.
        frozen, _, _ = model._build_torch_spatial_map_history(video, batch["lidar_to_world"])
    elif mode == "forecast":
        # Prefill all supplied observations, then append only newly generated sweeps.
        for start, end in blocks:
            end = min(end, spec["input_frames"])
            if start >= end:
                break
            spatial_map.append_token_block(
                video[0, :, start:end], batch["lidar_to_world"][0, start:end], start,
                input_has_prepend=False,
                token_pose_sensor_start_end=batch["pose_sensor_start_end"][0, start:end],
            )
    observed = torch.zeros((1, 64, padded_length, 16, 128), device=video.device)
    observed_cache = None
    for t in range(spec["input_frames"]):
        latent, observed_cache = model.encode_stream(
            video[:, :, t:t + 1].float(), temporal_kv_cache=observed_cache, max_cache_frames=9,
        )
        if not torch.isfinite(latent).all():
            raise FloatingPointError(f"Non-finite observed latent at frame {t}")
        observed[:, :, t:t + 1] = latent.float()
    del observed_cache

    condition_cache = decoder_cache = None
    decoded_blocks = []
    try:
        for block_index, (start, end) in enumerate(tqdm(
            blocks, desc=f"{mode} causal sampling", unit="block", file=console,
            leave=False, disable=not console.isatty(),
        )):
            poses = batch["lidar_to_world"][0, start:end]
            motion = batch["pose_sensor_start_end"][0, start:end]
            if spatial_map is not None:
                geometry, _ = spatial_map.render(poses, target_pose_sensor_start_end=motion)
                geometry = geometry[None]
            else:
                geometry = frozen[:, :, start:end]
            active = dict(batch)
            active.update(
                video=torch.full_like(video[:, :, start:end], -1),
                lidar_hdmap=batch["lidar_hdmap"][:, :, start:end],
                lidar_blockwise_warped_history=geometry,
                num_frames=batch["num_frames"].new_tensor([end - start]),
            )
            model._current_sfdmd_temporal_positions = torch.arange(start, end, device=video.device)
            model._current_stage2_known_frame_mask = known[start:end]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                condition, condition_cache = model.get_online_condition_stream(
                    active, temporal_kv_cache=condition_cache, max_cache_frames=9,
                )
                block_known = known[start:end]
                known_latents = observed[:, :, start:end]
                if bool(block_known.all()):
                    sampled = known_latents
                else:
                    noise = misc.arch_invariant_rand(
                        (1, 64, end - start, 16, 128), torch.float32, "cuda", seed + block_index,
                    )
                    sampled = model._generate_sde(
                        latents=noise, condition=condition, uncondition=None,
                        guidance=1.0, use_cfg=False, B=1, num_steps=args.num_steps,
                        shift=5, seed=seed + block_index, sde_eta=args.sde_eta,
                        known_latents=known_latents if bool(block_known.any()) else None,
                        known_mask=block_known if bool(block_known.any()) else None,
                        known_latents_mode="clean_mask",
                    )
                if not torch.isfinite(sampled).all():
                    raise FloatingPointError(f"Non-finite latents in block {block_index}")
                if bool(block_known.any()) and not torch.equal(
                    sampled[:, :, block_known].float(), known_latents[:, :, block_known].float(),
                ):
                    raise RuntimeError(f"Clean-prefix replacement failed in block {block_index}")
                # Commit once after denoising, synchronously, before the next block.
                _commit(model, sampled, condition)
            decoded, decoder_cache = model.decode_stream(
                sampled.float(), temporal_kv_cache=decoder_cache, max_cache_frames=30,
            )
            if decoded.shape != video[:, :, start:end].shape or not torch.isfinite(decoded).all():
                raise RuntimeError(f"Invalid decoded LiDAR in block {block_index}")
            decoded = torch.where(block_known[None, None, :, None, None], video[:, :, start:end], decoded)
            decoded_blocks.append(decoded)
            new_start = max(start, spec["input_frames"])
            if spatial_map is not None and new_start < end:
                # No current-block or future GT points can enter its own condition.
                offset = new_start - start
                spatial_map.append_token_block(
                    decoded[0, :, offset:], poses[offset:], new_start, input_has_prepend=False,
                    token_pose_sensor_start_end=motion[offset:],
                )
    finally:
        model._current_sfdmd_temporal_positions = None
        model._current_sfdmd_self_attn_kv_cache = None
        model._current_sfdmd_store_self_attn_kv = False
        model._current_stage2_data_batch = None
    return _metric_ranges(torch.cat(decoded_blocks, dim=2)[:, :, :length])
