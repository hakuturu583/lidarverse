# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Open-ended HDMap-conditioned generation with a causal (Stage 2/3) package.

diffusion/inference/causal.py generates a fixed clip; a simulator asks for
one sweep after another. ``WorldStream`` runs the same blocks (one frame,
then four at a time), the same rolling K/V caches and the same spatial cache
of generated geometry, but takes each block's poses and HDMap condition when
it is generated. Several streams can share one model: each keeps its own
caches and swaps them in around its block.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from . import sensor


def load_world_model(checkpoint_dir: str | Path, stage: int = 3):
    """A released causal package (``checkpoints/`` as download_checkpoints.py
    lays it out) on the current CUDA device."""
    from diffusion.inference.pai import load_model, load_package

    if stage not in (2, 3):
        raise ValueError("streaming generation needs a causal package (stage 2 or 3)")
    root = Path(checkpoint_dir)
    package = {2: "causal", 3: "sfdmd"}[stage]
    checkpoint = root / f"lidarverse-{package}-pai" / "ema_model.pt"
    tokenizer = root / "lidarverse-tokenizer-pai" / "ckpt.pt"
    stats = root / "lidarverse-tokenizer-pai" / "latent_stats.pt"
    for path in (checkpoint, checkpoint.with_name("config.json"), tokenizer, stats):
        if not path.is_file():
            raise FileNotFoundError(f"{path} (run scripts/download_checkpoints.py)")
    return load_model(checkpoint, tokenizer, stats, stage, load_package(checkpoint, stage))


class WorldStream:
    """One sensor's generation state."""

    def __init__(self, model, *, num_steps: int = 4, sde_eta: float = 1.0, seed: int = 0,
                 kv_cache_frames: int = 9):
        if kv_cache_frames < 5 or (kv_cache_frames - 1) % 4:
            raise ValueError("the K/V budget must be 1+4N frames (including the sink), at least 5")
        self.model = model
        self.num_steps = num_steps
        self.sde_eta = sde_eta
        self.seed = seed
        self.kv_cache_frames = kv_cache_frames
        self.device = next(model.parameters()).device
        self.rays = torch.as_tensor(sensor.ray_directions(), device=self.device)
        self.frame = 0  # the next frame's index
        self.block = 0
        self.kv_cache = model.new_teacher_forcing_kv_cache()
        self.condition_cache = None
        self.decoder_cache = None
        self.spatial_map = None

    def next_block_length(self) -> int:
        return 1 if self.frame == 0 else 4

    def _batch(self, length: int, hdmap: torch.Tensor, lidar_to_world: torch.Tensor, motion: torch.Tensor) -> dict:
        d = self.device
        batch = dict(
            video=torch.full((1, 1, length, sensor.HEIGHT, sensor.WIDTH), -1.0, device=d),
            is_preprocessed=True, lidar_hdmap=hdmap[None, None].float(),
            sensor_dataset_id=torch.tensor([0], device=d),
            sensor_ray_directions=self.rays[None],
            sensor_depth_bounds=torch.tensor([[sensor.MIN_RANGE_M, sensor.MAX_RANGE_M]], device=d),
            fps=torch.tensor([sensor.FPS], device=d), num_frames=torch.tensor([length], device=d),
            image_size=torch.tensor([[sensor.HEIGHT, sensor.WIDTH]], device=d),
            padding_mask=torch.zeros(1, 1, sensor.HEIGHT, sensor.WIDTH, device=d),
            t5_text_embeddings=torch.zeros(1, 512, 1024, device=d),
            t5_text_mask=torch.zeros(1, 512, device=d),
            lidar_to_world=lidar_to_world[None], pose_sensor_start_end=motion[None],
        )
        batch["t5_text_mask"][:, 0] = 1
        return batch

    def _bind(self, batch: dict) -> None:
        """Point the model's per-run state at this stream (as causal.generate does)."""
        m = self.model
        m._current_stage2_data_batch = batch
        m._current_sensor_dataset_ids = batch["sensor_dataset_id"]
        m._current_stage2_geometry_mode = "rolling_static"
        m.factorized_static_only_geometry = False
        m._hdmap_force_zero_latent = False
        m._current_sfdmd_self_attn_kv_cache = self.kv_cache
        m._current_sfdmd_self_attn_kv_max_frames = self.kv_cache_frames
        m._current_sfdmd_self_attn_kv_sink_frames = 1
        m._current_sfdmd_store_self_attn_kv = False
        m._current_sfdmd_detach_self_attn_kv = True
        m._current_sfdmd_return_self_attn_kv = False

    def _unbind(self) -> None:
        m = self.model
        m._current_sfdmd_temporal_positions = None
        m._current_sfdmd_self_attn_kv_cache = None
        m._current_sfdmd_store_self_attn_kv = False
        m._current_stage2_data_batch = None

    @torch.inference_mode()
    def generate(self, pose_sensor_start_end: np.ndarray, hdmap: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        """The next block: ``pose_sensor_start_end`` (n, 2, 4, 4) sensor-to-world
        at each sweep's start and end (right-handed, x forward / y left / z up),
        ``hdmap`` (n, 128, 1024) the normalized condition. Returns metric ranges
        and validity, (n, 128, 1024)."""
        from diffusion.inference.causal import _commit, _spatial_map
        from diffusion.inference.hdmap import _metric_ranges
        from utils import misc

        m = self.model
        n = len(pose_sensor_start_end)
        if n != self.next_block_length():
            raise ValueError(f"block {self.block} has {self.next_block_length()} frames, got {n}")
        start, end = self.frame, self.frame + n
        poses = torch.as_tensor(np.asarray(pose_sensor_start_end, np.float32), device=self.device)
        lidar_to_world = poses[:, 0]
        motion = torch.linalg.inv(poses[:, 1]) @ poses[:, 0]
        batch = self._batch(n, hdmap.to(self.device), lidar_to_world, motion)
        self._bind(batch)
        try:
            if self.spatial_map is None:
                self.spatial_map = _spatial_map(m, batch)
            geometry, _ = self.spatial_map.render(lidar_to_world, target_pose_sensor_start_end=motion)
            batch["lidar_blockwise_warped_history"] = geometry[None]
            m._current_sfdmd_temporal_positions = torch.arange(start, end, device=self.device)
            m._current_stage2_known_frame_mask = torch.zeros(n, dtype=torch.bool, device=self.device)
            seed = self.seed + self.block
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
                condition, self.condition_cache = m.get_online_condition_stream(
                    batch, temporal_kv_cache=self.condition_cache, max_cache_frames=9,
                )
                noise = misc.arch_invariant_rand((1, 64, n, 16, 128), torch.float32, self.device.type, seed)
                sampled = m._generate_sde(
                    latents=noise, condition=condition, uncondition=None, guidance=1.0, use_cfg=False,
                    B=1, num_steps=self.num_steps, shift=5, seed=seed, sde_eta=self.sde_eta,
                    known_latents=None, known_mask=None, known_latents_mode="clean_mask",
                )
                if not torch.isfinite(sampled).all():
                    raise FloatingPointError(f"non-finite latents in block {self.block}")
                _commit(m, sampled, condition)
            decoded, self.decoder_cache = m.decode_stream(
                sampled.float(), temporal_kv_cache=self.decoder_cache, max_cache_frames=30,
            )
            self.spatial_map.append_token_block(
                decoded[0], lidar_to_world, start, input_has_prepend=False,
                token_pose_sensor_start_end=motion,
            )
        finally:
            self._unbind()
        self.frame, self.block = end, self.block + 1
        return _metric_ranges(decoded)
