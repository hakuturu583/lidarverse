# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared PAI package loading, conditioning, and quickstart outputs."""

from copy import deepcopy
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from diffusion.inference.hdmap import _metric_ranges
from utils import inference as common

# No duplicate first frame: this DiT was trained on 21 unique sweeps.
MODES = {
    "world": dict(input_frames=0, output_frames=20, geometry_frames=0,
                  hdmap="full", geometry="null"),
    "forecast": dict(input_frames=10, output_frames=11, geometry_frames=10,
                     hdmap="null", geometry="frozen_static"),
    "resim": dict(input_frames=15, output_frames=5, geometry_frames=15,
                  hdmap="null", geometry="frozen_object"),
}
SPATIAL_CACHE = dict(
    voxel_policy="persistent_static", voxel_m=0.15, radius_m=100.0,
    observation_unit="frame", static_min_observations=2, include_latest=False,
    candidate_ttl_updates=1, free_space_clear=True, free_space_margin_m=0.75,
    free_space_misses_to_remove=2,
)


def mode_spec(mode, stage=1):
    spec = dict(MODES[mode])
    if mode == "world" and stage != 1:
        spec["output_frames"] = 21
        spec["geometry"] = "rolling_generated"
    if mode == "forecast" and stage != 1:
        spec["geometry"] = "rolling_observed_then_generated"
    return spec


def load_package(checkpoint, stage):
    config_path = checkpoint.with_name("config.json")
    config = json.loads(config_path.read_text())
    architecture = {1: "bidir", 2: "causal", 3: "sfdmd"}[stage]
    expected = dict(
        architecture=f"pai_factorized_{architecture}_p1x2", stage=f"stage{stage}",
        weights="ema", input_channels=193, output_channels=64,
        patch_spatial=[1, 2], model_channels=2048, num_blocks=28, num_heads=16,
        num_dataset_embeddings=3, sensor_dataset_id=0, resolution=[128, 1024],
        min_range_m=2.5, max_range_m=100, input_has_prepend=False,
        shift=5, shift_clamp_percentiles=None, precision="bfloat16",
        tokenizer="transformer_lidar_video_tokenizer_8x8x1_z64_pandar_2p5to100_1024_ema",
        tokenizer_encoder_dtype="bfloat16" if stage == 3 else "float32",
        tokenizer_decoder_dtype="float32",
    )
    if stage != 1:
        expected.update(temporal_block_size=4, temporal_prefix_frames=1)
    if stage == 3:
        encoder_dtype = config.get("tokenizer_encoder_dtype")
        if encoder_dtype not in ("float32", "bfloat16"):
            raise ValueError(f"{config_path}: unsupported tokenizer encoder dtype {encoder_dtype!r}")
        spatial_cache = config.get("spatial_cache")
        if not isinstance(spatial_cache, dict):
            raise ValueError(f"{config_path}: missing spatial_cache configuration")
        voxel_m = spatial_cache.get("voxel_m")
        if not isinstance(voxel_m, (int, float)) or not np.isfinite(voxel_m) or voxel_m <= 0:
            raise ValueError(f"{config_path}: voxel_m must be finite and positive")
        expected.update(tokenizer_encoder_dtype=encoder_dtype,
                        spatial_cache={**SPATIAL_CACHE, "voxel_m": voxel_m}, stream_condition_vae_encode=True,
                        condition_vae_cache_frames=9, clean_x0_kv_commit=True)
    for key, value in expected.items():
        actual = config.get(key)
        if key == "tokenizer" and actual == "transformer_lidar_video_tokenizer_8x8x1_z64_pandar_2p5to100_1024_ema40k":
            actual = value
        if key not in config or actual != value:
            raise ValueError(f"{config_path}: expected {key}={value!r}, got {config.get(key)!r}")
    return config


def load_model(checkpoint, tokenizer_checkpoint, latent_stats, stage=1, package=None):
    package = load_package(checkpoint, stage) if package is None else package
    from diffusion.training.config.base.lidarverse_vae import get_lidarverse_tokenizer
    from diffusion.training.config.base.model import LidarModelConfig
    from diffusion.training.config.text2world.registry import (
        MININET_2B_Config, VideoConditionerFpsSizePaddingConfig,
    )
    from diffusion.training.models.hdmaptolidar_model_rf import (
        HdmapToLidarFactorizedBidirectionalRFModel,
        HdmapToLidarFactorizedTeacherForcingRFModel,
    )
    from tokenizer.training.configs.base.nets.lidarverse import PAI_TOKENIZER_NAME
    from utils.lazy_config import LazyDict

    net = deepcopy(MININET_2B_Config)
    net.update(
        in_channels=193, out_channels=64, patch_spatial=(1, 2),
        num_dataset_embeddings=3, num_frame_type_embeddings=0,
        temporal_block_size=0 if stage == 1 else 4,
        temporal_prefix_frames=0 if stage == 1 else 1, side_condition_channels=0,
        atten_backend="minimal_a2a" if stage == 1 else "torch-flex",
        rope_scale_positions_by_patch=True, timestep_scale=0.001,
        rope_h_extrapolation_ratio=3.0, rope_w_extrapolation_ratio=3.0,
        rope_t_extrapolation_ratio=1.0, rope_enable_fps_modulation=False,
        use_wan_fp32_strategy=True,
    )
    net.sac_config.mode = "none"
    vae = get_lidarverse_tokenizer()
    vae.update(
        experiment_name=PAI_TOKENIZER_NAME, checkpoint_path=str(tokenizer_checkpoint),
        mean_std_fp=str(latent_stats), strict_checkpoint=True, use_ema=False,
        is_bf16=False, encoder_is_bf16=package["tokenizer_encoder_dtype"] == "bfloat16", decoder_is_bf16=False,
    )
    config = LidarModelConfig(
        net=net, vae=vae, conditioner=deepcopy(VideoConditionerFpsSizePaddingConfig),
        ema=LazyDict(dict(enabled=False)), fsdp_enabled=False,
        latent_shape=[64, package.get("window_frames", 21), 16, 128], input_video_has_prepend=False,
        hdmap_channels=1, additional_concat_ch=64, encode_hdmap_with_vae=True,
        hdmap_dropout_prob=0.0, sigma_data=1.0, shift=5,
        shift_clamp_percentiles=None, enable_shift_stretch=False,
        rf_prediction_type="velocity", loss_reduce="mean",
        diffusion_forcing_block_size=21 if stage == 1 else 4,
        teacher_forcing_prefix_frames=0 if stage == 1 else 1,
        factorized_bidirectional_stage1=stage == 1,
        blockwise_warped_history_condition=True, blockwise_warped_history_concat=True,
        blockwise_warped_history_backend="torch_spatial_map",
        blockwise_spatial_map_max_range=100.0, blockwise_spatial_map_min_range=2.5,
        **{f"blockwise_spatial_map_{key}": value for key, value in package.get("spatial_cache", SPATIAL_CACHE).items()},
        blockwise_spatial_map_geometry_root_path="", blockwise_spatial_map_geometry_filter_list_path="",
        stage2_known_mask_concat=True,
        stage2_factorized_prefix_frames=[0, 1, 2, 4, 8, 10, 12, 16],
        stage2_factorized_prefix_probs=[0.5] + [1 / 14] * 7,
        stage2_factorized_hdmap_modes=["null", "full"], stage2_factorized_hdmap_probs=[0.5, 0.5],
        stage2_factorized_geometry_modes_k0=["null", "rolling_static", "frozen_object"],
        stage2_factorized_geometry_probs_k0=[0.1, 0.45, 0.45],
        stage2_factorized_geometry_modes_known=["null", "rolling_static", "frozen_object"],
        stage2_factorized_geometry_probs_known=[0.1, 0.45, 0.45],
        factorized_decouple_geometry_observations=True,
        factorized_geometry_observation_frames=[1, 2, 4, 8, 10, 12, 16, 20],
        factorized_geometry_observation_probs=[0.125] * 8,
        factorized_geometry_rolling_shutter=True, factorized_geometry_actor_box_time_mode="frame",
    )
    model_type = (HdmapToLidarFactorizedBidirectionalRFModel if stage == 1
                  else HdmapToLidarFactorizedTeacherForcingRFModel)
    model = model_type(config)
    model.model.to(dtype=torch.bfloat16)
    payload = torch.load(str(checkpoint), map_location="cpu", weights_only=True, mmap=True)
    model.model.load_state_dict(payload["model"], strict=True)
    del payload
    model.cuda().eval()
    if stage != 1:
        model.net.set_cached_inference_self_attention_backend("sdpa-flash")
    print(f"Loaded PAI Stage {stage} EMA strictly: {checkpoint}", flush=True)
    return model


def load_sample(path, modes, stage=1):
    with np.load(path, allow_pickle=False) as data:
        sample = {key: data[key] for key in data.files}
    ranges, valid, _, rays = common.load_sample(path)
    sample.update(range_m=ranges, valid_mask=valid, ray_directions=rays)
    frames = max(sum(mode_spec(mode, stage)[key] for key in ("input_frames", "output_frames")) for mode in modes)
    if stage != 1:
        frames = 1 + ((frames - 1 + 3) // 4) * 4
    if sample["range_m"].shape[1:] != (128, 1024) or len(sample["range_m"]) < frames:
        raise ValueError(f"{path}: need at least {frames} real 128x1024 sweeps")
    if int(sample.get("sensor_dataset_id", -1)) != 0 or str(sample.get("dataset", "")) != "pandar":
        raise ValueError(f"{path}: this checkpoint is NVIDIA PAI only (dataset ID 0)")
    if float(sample.get("fps", 0)) != 10 or bool(sample.get("has_prepend", False)):
        raise ValueError(f"{path}: expected 10 Hz real sweeps without a duplicated prefix")
    if float(sample.get("min_range_m", 0)) != 2.5 or float(sample.get("max_range_m", 0)) != 100:
        raise ValueError(f"{path}: expected range support [2.5,100] m")
    rays = sample["ray_directions"]
    if rays.shape != (128, 1024, 3) or not np.isfinite(rays).all() or not np.allclose(np.linalg.norm(rays, axis=-1), 1, atol=1e-4):
        raise ValueError(f"{path}: expected a calibrated 128x1024 unit-ray grid")
    if "world" in modes:
        hdmap = sample.get("hdmap")
        if hdmap is None or hdmap.shape != sample["range_m"].shape or not np.isfinite(hdmap).all() or hdmap.min() < -1 or hdmap.max() > 1:
            raise ValueError(f"{path}: world generation needs aligned HDMap ranges normalized to [-1,1]")
    poses = sample.get("pose_sensor_start_end")
    if poses is None or poses.shape != (len(sample["range_m"]), 2, 4, 4) or not np.isfinite(poses).all():
        raise ValueError(f"{path}: missing aligned sensor-to-world start/end poses")
    if "resim" in modes or ("forecast" in modes and stage == 1):
        for key in ("box_object_to_world", "box_lwh", "box_valid", "box_is_moving", "box_track_id"):
            if key not in sample or len(sample[key]) != len(sample["range_m"]):
                raise ValueError(f"{path}: forecast/resim need aligned {key}; use prepare_pai_bidir.py")
        shape = sample["box_valid"].shape
        if len(shape) != 2 or sample["box_track_id"].shape != shape or sample["box_is_moving"].shape != shape:
            raise ValueError(f"{path}: box masks and track IDs must be (T,M)")
        if sample["box_object_to_world"].shape != shape + (4, 4) or sample["box_lwh"].shape != shape + (3,):
            raise ValueError(f"{path}: box poses/sizes do not match the mask")
        active = sample["box_valid"].astype(bool)
        if not np.isfinite(sample["box_object_to_world"][active]).all() or not np.isfinite(sample["box_lwh"][active]).all():
            raise ValueError(f"{path}: non-finite actor annotations")
        if (sample["box_lwh"][active] <= 0).any() or (sample["box_track_id"][active] < 0).any():
            raise ValueError(f"{path}: actor sizes must be positive and track IDs nonnegative")
    return sample


def make_batch(sample, mode, stage=1, length=None):
    from utils.pai_batch import make_pai_batch

    spec = mode_spec(mode, stage)
    length = spec["input_frames"] + spec["output_frames"] if length is None else length
    return make_pai_batch(sample, length, input_frames=spec["input_frames"],
                          hdmap=spec["hdmap"] == "full",
                          boxes=mode == "resim" or (mode == "forecast" and stage == 1), device="cuda")


def generate_bidir(model, sample, mode, args, seed, console):
    spec = MODES[mode]
    model.stage2_factorized_force_prefix_k = spec["input_frames"]
    model.stage2_factorized_force_hdmap_mode = spec["hdmap"]
    # frozen_object + static_only is exactly the frozen_static builder branch.
    model.stage2_factorized_force_geometry_mode = "null" if mode == "world" else "frozen_object"
    model.factorized_static_only_geometry = mode == "forecast"
    model.factorized_force_geometry_observations = spec["geometry_frames"] if mode != "world" else -1
    torch.manual_seed(seed)
    batch = make_batch(sample, mode)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        _, latent, _ = model.get_data_and_condition(batch)
    known_mask = model._current_stage2_known_frame_mask
    kwargs = dict(known_latents=latent, known_mask=known_mask, known_latents_mode="clean_mask") if spec["input_frames"] else {}
    with tqdm(total=args.num_steps, desc=f"{mode} sampling", file=console,
              unit="step", leave=False, disable=not console.isatty()) as progress:
        calls = 0
        def on_forward(_module, _inputs, _output):
            nonlocal calls
            calls += 1
            if calls % (1 if args.guidance in (0, 1) else 2) == 0:
                progress.update(1)
        hook = model.net.register_forward_hook(on_forward)
        try:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                result = model.generate_samples_from_batch(
                    batch, state_shape=latent.shape[1:], n_sample=1,
                    guidance=args.guidance, seed=seed, num_steps=args.num_steps,
                    sampler=args.sampler, sde_eta=args.sde_eta, **kwargs,
                )
        finally:
            hook.remove()
    if not torch.isfinite(result).all():
        raise FloatingPointError("Sampling produced non-finite latents")
    if spec["input_frames"] and not torch.equal(result[:, :, :spec["input_frames"]], latent[:, :, :spec["input_frames"]].float()):
        raise RuntimeError("Clean-prefix latents were not preserved")
    decoded = model.decode(result.float())
    if decoded.shape != batch["video"].shape or not torch.isfinite(decoded).all():
        raise RuntimeError("Invalid decoded LiDAR sequence")
    return _metric_ranges(decoded)


def _preview_label(image, title, frame, length, input_frames, is_input):
    image = image.copy()
    cv2.rectangle(image, (0, 0), (image.shape[1] - 1, 80), (255, 255, 255), -1)
    color = (175, 90, 35) if is_input else (30, 30, 30)
    cv2.putText(image, title, (20, 33), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
    detail = f"Frame {frame} / {length - 1}"
    if input_frames:
        detail += f"  |  Input: 0-{input_frames - 1}  |  Generated: {input_frames}-{length - 1}"
    cv2.putText(image, detail, (20, 64), cv2.FONT_HERSHEY_SIMPLEX, .7, (65, 65, 65), 1, cv2.LINE_AA)
    return image


def preview(sample, generated, valid, mode, output):
    from utils.visualize.hdmap import render_hdmap

    spec = MODES[mode]
    length = len(generated)
    if mode == "world":
        source, source_valid = _metric_ranges(torch.from_numpy(sample["hdmap"][:length])[None, None])
        source_title = "Input HDMap"
    else:
        source, source_valid = sample["range_m"][:length], sample["valid_mask"][:length]
    for t in range(length):
        is_input = t < spec["input_frames"]
        if mode != "world":
            source_title = "INPUT | Observed GT" if is_input else "GT | Reference only (not input)"
        generated_title = "INPUT | Observed GT" if is_input else {
            "world": "GENERATION | HDMap-conditioned",
            "forecast": "GENERATION | Forecasting",
            "resim": "GENERATION | Resimulation",
        }[mode]
        # Display the actual provided sweep, not its tokenizer reconstruction.
        values, mask = (source[t], source_valid[t]) if is_input else (generated[t], valid[t])
        clouds = (("input" if mode == "world" else "gt", source_title, source[t], source_valid[t]),
                  ("generation", generated_title, values, mask))
        panels = []
        for name, title, ranges, hits in clouds:
            points = common.points_from_ranges(ranges, hits, sample["ray_directions"])
            common.write_ply(output / "ply" / f"{name}_{t:03d}.ply", points)
            rendered = render_hdmap(sample, t) if mode == "world" and name == "input" else common.render_points(points, yaw_deg=0)
            panels.append(_preview_label(rendered, title, t, length, spec["input_frames"], is_input))
        yield np.concatenate(panels, axis=1)


@torch.inference_mode()
def run(args, console, say):
    package = load_package(args.checkpoint, args.stage)
    modes = list(MODES) if args.mode == "all" else [args.mode]
    paths = common.sample_paths(args.sample_path.resolve())[:args.num_samples]
    if len(paths) < args.num_samples:
        raise ValueError(f"Requested {args.num_samples} clips, found only {len(paths)}")
    # Validate all requested annotations before allocating the diffusion model.
    for path in paths:
        sample = load_sample(path, modes, args.stage)
        if args.raw_hdmap_condition_rasterize and "world" in modes:
            from diffusion.inference.raw_hdmap import validate_raw_hdmap
            validate_raw_hdmap(sample)
        if args.vis_pcd and "world" in modes:
            for key in ("map_vertices", "map_offsets", "map_kinds", "box_class"):
                if key not in sample:
                    raise ValueError(f"{path}: missing {key}; rerun prepare_pai_bidir.py")
    if not torch.cuda.is_available():
        raise RuntimeError("Run this quickstart on a CUDA GPU in the diffusion environment")
    model = load_model(args.checkpoint, args.tokenizer_checkpoint, args.latent_stats, args.stage, package)
    generate = generate_bidir
    if args.stage != 1:
        from diffusion.inference.causal import generate
    metadata = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    metadata.update(checkpoint_config=package,
                    modes={mode: mode_spec(mode, args.stage) for mode in modes}, shift=5,
                    shift_clamp_percentiles=None, input_has_prepend=False,
                    tokenizer_encoder_dtype=package["tokenizer_encoder_dtype"], tokenizer_decoder_dtype="float32")
    if args.stage != 1:
        metadata.update(kv_cache_frames=args.kv_cache_frames, kv_sink_frames=1,
                        attention_backend="sdpa-flash",
                        kv_budget_includes_sink=True, clean_kv_commit="synchronous_blockwise",
                        condition_vae_cache_frames=9, decoder_cache_frames=30,
                        observed_latent_encoding="causal_observed_only_stream",
                        spatial_cache=package.get("spatial_cache", SPATIAL_CACHE))
    (args.output_folder / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    for mode in modes:
        for index, path in enumerate(paths):
            seed = args.seed + index
            say(f"{mode} [{index + 1}/{len(paths)}] {path.stem} | seed {seed}")
            sample = load_sample(path, [mode], args.stage)
            output = common.sample_output(path, args.sample_path.resolve(), args.output_folder / mode)
            output.mkdir(parents=True, exist_ok=True)
            if args.raw_hdmap_condition_rasterize and mode == "world":
                from diffusion.inference.raw_hdmap import rasterize_raw_hdmap
                say("  Rasterizing raw map and boxes with rolling-shutter poses...")
                condition, stats = rasterize_raw_hdmap(
                    sample, bbox_dilation=args.raw_hdmap_condition_bbox_dilation,
                    seg_interval=args.raw_hdmap_seg_interval,
                )
                sample["hdmap"] = condition
                np.savez_compressed(output / "hdmap_condition.npz", normalized_range=condition,
                                    frame_indices=sample["frame_indices"],
                                    pose_sensor_start_end=sample["pose_sensor_start_end"],
                                    ray_directions=sample["ray_directions"])
                (output / "raw_hdmap_condition.json").write_text(json.dumps(stats, indent=2) + "\n")
                say(f"  HDMap coverage: {stats['original_hit_fraction']:.3%} -> {stats['condition_hit_fraction']:.3%}")
            generated, valid = generate(model, sample, mode, args, seed, console)
            length = len(generated)
            np.savez_compressed(output / "generation.npz", range_m=generated, valid_mask=valid,
                                ray_directions=sample["ray_directions"], fps=np.float32(10),
                                pose_sensor_start_end=sample["pose_sensor_start_end"][:length],
                                frame_indices=sample["frame_indices"][:length], clip_name=sample["clip_name"],
                                known_frame_mask=np.arange(length) < MODES[mode]["input_frames"], seed=np.int64(seed))
            if args.vis_pcd:
                common.write_video(output / "generation.mp4", preview(sample, generated, valid, mode, output), 10)
            say(f"  Saved {output}")
    say(f"Finished. Outputs: {args.output_folder}")
