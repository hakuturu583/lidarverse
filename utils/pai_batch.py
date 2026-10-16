# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared processed-PAI batch contract for training and inference."""

import numpy as np
import torch


def make_pai_batch(sample, length, *, input_frames=None, hdmap=True, boxes=True, device="cpu"):
    ranges = sample["range_m"][:length]
    valid = sample["valid_mask"][:length] & np.isfinite(ranges) & (ranges >= 2.5) & (ranges <= 100)
    video = np.where(valid, 2 * (ranges - 2.5) / 97.5 - 1, -1).astype(np.float32)
    if input_frames is not None:
        video[input_frames:] = -1  # Never expose future GT during inference.
    poses = sample["pose_sensor_start_end"][:length]
    tensor = lambda value: torch.as_tensor(value, device=device)
    batch = dict(
        video=tensor(video)[None, None], is_preprocessed=True,
        lidar_hdmap=tensor(sample["hdmap"][:length].astype(np.float32))[None, None]
                    if hdmap else torch.full((1, 1, length, 128, 1024), -1., device=device),
        sensor_dataset_id=tensor([0]).long(), sensor_ray_directions=tensor(sample["ray_directions"])[None].float(),
        sensor_depth_bounds=tensor([[2.5, 100.0]]).float(),
        lidar_to_world=tensor(poses[:, 0])[None].float(),
        pose_sensor_start_end=tensor(np.linalg.inv(poses[:, 1]) @ poses[:, 0])[None].float(),
        fps=tensor([10.0]), num_frames=tensor([length]), image_size=tensor([[128, 1024]]),
        padding_mask=torch.zeros(1, 1, 128, 1024, device=device),
        t5_text_embeddings=torch.zeros(1, 512, 1024, device=device),
        t5_text_mask=torch.zeros(1, 512, device=device),
    )
    batch["t5_text_mask"][:, 0] = 1
    if boxes:
        valid_boxes = sample["box_valid"][:length].astype(bool)
        tracks = sample["box_track_id"][:length]
        dynamic_ids = np.unique(tracks[valid_boxes & sample["box_is_moving"][:length]])
        dynamic = valid_boxes & np.isin(tracks, dynamic_ids)
        for key in ("object_to_world", "lwh", "track_id"):
            array = sample[f"box_{key}"][:length]
            packed = np.zeros((length, 256) + array.shape[2:], dtype=array.dtype)
            if key == "track_id":
                packed.fill(-1)
            for t in range(length):
                selected = array[t, dynamic[t]]
                if len(selected) > 256:
                    raise ValueError("More than 256 dynamic boxes in a sweep")
                packed[t, :len(selected)] = selected
            batch[f"moving_box_{key}"] = tensor(packed)[None]
        batch["moving_box_valid"] = tensor(np.arange(256)[None] < dynamic.sum(axis=1)[:, None])[None]
    return batch
