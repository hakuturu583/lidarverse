# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Geometry-only target-pose warping for LiDAR forecast conditioning."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch

from diffusion.training.utils.diffusion_forcing import temporal_chunk_slices
from tokenizer.training.datasets.lidar_datasets.waymo_hdmap import (
    project_world_points,
    scatter_min_range,
)


def _unnormalize_valid_depth(
    normalized: np.ndarray,
    valid: np.ndarray,
    min_range: float,
    max_range: float,
    min_value: float,
    inverse_depth: bool,
) -> np.ndarray:
    c_min = min_range if not inverse_depth else 1.0 / max_range
    c_max = max_range if not inverse_depth else 1.0 / min_range
    if min_value == -1:
        depth = (normalized + 1.0) * 0.5 * (c_max - c_min) + c_min
    else:
        depth = normalized * (c_max - c_min) + c_min
    if inverse_depth:
        depth = 1.0 / depth
    return np.where(valid, depth, 0.0).astype(np.float32)


def _normalize_metric_depth(
    metric: np.ndarray,
    min_range: float,
    max_range: float,
    min_value: float,
    inverse_depth: bool,
) -> np.ndarray:
    c_min = min_range if not inverse_depth else 1.0 / max_range
    c_max = max_range if not inverse_depth else 1.0 / min_range
    if inverse_depth:
        invalid = metric == 0
        work = metric.copy()
        work[invalid] = max_range
        work = 1.0 / work
        work[invalid] = c_min
    else:
        work = metric
    work = np.clip(work, c_min, c_max)
    if min_value == -1:
        work = (work - c_min) / (c_max - c_min) * 2.0 - 1.0
    else:
        work = (work - c_min) / (c_max - c_min)
    return work.astype(np.float32)


def _wrap_angle_torch(angle: torch.Tensor) -> torch.Tensor:
    return torch.remainder(angle + torch.pi, 2.0 * torch.pi) - torch.pi


def _fractional_transform_points_torch(
    points: torch.Tensor,
    transform: torch.Tensor,
    fractions: torch.Tensor,
) -> torch.Tensor:
    """Apply identity-to-transform SE(3) interpolation per point."""
    if points.ndim != 2 or points.shape[1:] != (3,):
        raise ValueError(f"points must be (N,3), got {tuple(points.shape)}")
    if transform.shape != (4, 4):
        raise ValueError(
            f"transform must be (4,4), got {tuple(transform.shape)}"
        )
    if fractions.shape != (points.shape[0],):
        raise ValueError(
            "fractions must align with points, got "
            f"{tuple(fractions.shape)} for {points.shape[0]} points"
        )
    if points.shape[0] == 0:
        return points

    transform = transform.to(device=points.device, dtype=points.dtype)
    fractions = fractions.to(
        device=points.device, dtype=points.dtype
    ).clamp_(0.0, 1.0)
    rotation = transform[:3, :3]
    skew_vector = torch.stack(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ]
    )
    cosine = ((torch.trace(rotation) - 1.0) * 0.5).clamp(-1.0, 1.0)
    theta = torch.acos(cosine)
    sine = torch.sin(theta)
    rotvec_scale = torch.where(
        theta < 1.0e-4,
        0.5 + theta.square() / 12.0,
        theta / (2.0 * sine.clamp_min(1.0e-12)),
    )
    rotvec = rotvec_scale * skew_vector
    base_theta = torch.linalg.vector_norm(rotvec)
    axis = rotvec / base_theta.clamp_min(1.0e-12)

    angles = fractions * base_theta
    sin_angles = torch.sin(angles).unsqueeze(1)
    cos_angles = torch.cos(angles).unsqueeze(1)
    axis_rows = axis.unsqueeze(0).expand_as(points)
    rotated = (
        points * cos_angles
        + torch.cross(axis_rows, points, dim=1) * sin_angles
        + axis_rows
        * (points * axis_rows).sum(dim=1, keepdim=True)
        * (1.0 - cos_angles)
    )
    translation = fractions.unsqueeze(1) * transform[:3, 3].unsqueeze(0)
    return rotated + translation


_ACTOR_BOX_TIME_MODES = (
    "frame",
    "current_to_next",
    "previous_to_current",
)


def _fractional_rotation_matrices_torch(
    rotation: torch.Tensor,
    fractions: torch.Tensor,
) -> torch.Tensor:
    """Interpolate identity to one rotation for each supplied fraction."""
    if rotation.shape != (3, 3):
        raise ValueError(
            f"rotation must be (3,3), got {tuple(rotation.shape)}"
        )
    if fractions.ndim != 1:
        raise ValueError(
            f"fractions must be one-dimensional, got {tuple(fractions.shape)}"
        )
    rotation = rotation.to(device=fractions.device, dtype=fractions.dtype)
    fractions = fractions.clamp(0.0, 1.0)
    skew_vector = torch.stack(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ]
    )
    cosine = ((torch.trace(rotation) - 1.0) * 0.5).clamp(-1.0, 1.0)
    theta = torch.acos(cosine)
    sine = torch.sin(theta)
    rotvec_scale = torch.where(
        theta < 1.0e-4,
        0.5 + theta.square() / 12.0,
        theta / (2.0 * sine.clamp_min(1.0e-12)),
    )
    rotvec = rotvec_scale * skew_vector
    base_theta = torch.linalg.vector_norm(rotvec)
    axis = rotvec / base_theta.clamp_min(1.0e-12)
    axis_skew = torch.zeros(
        (3, 3), device=fractions.device, dtype=fractions.dtype
    )
    axis_skew[0, 1] = -axis[2]
    axis_skew[0, 2] = axis[1]
    axis_skew[1, 0] = axis[2]
    axis_skew[1, 2] = -axis[0]
    axis_skew[2, 0] = -axis[1]
    axis_skew[2, 1] = axis[0]
    angles = fractions * base_theta
    identity = torch.eye(
        3, device=fractions.device, dtype=fractions.dtype
    ).unsqueeze(0)
    return (
        identity
        + torch.sin(angles)[:, None, None] * axis_skew.unsqueeze(0)
        + (1.0 - torch.cos(angles))[:, None, None]
        * (axis_skew @ axis_skew).unsqueeze(0)
    )


def _interpolate_pose_components_torch(
    start_pose: torch.Tensor,
    end_pose: torch.Tensor,
    fractions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-fraction object-to-world rotations and translations."""
    if start_pose.shape != (4, 4) or end_pose.shape != (4, 4):
        raise ValueError(
            "pose interpolation requires two (4,4) transforms, got "
            f"{tuple(start_pose.shape)} and {tuple(end_pose.shape)}"
        )
    start = start_pose.to(device=fractions.device, dtype=fractions.dtype)
    end = end_pose.to(device=fractions.device, dtype=fractions.dtype)
    relative_rotation = start[:3, :3].T @ end[:3, :3]
    delta_rotation = _fractional_rotation_matrices_torch(
        relative_rotation, fractions
    )
    rotations = start[:3, :3].unsqueeze(0) @ delta_rotation
    translations = (
        (1.0 - fractions[:, None]) * start[:3, 3].unsqueeze(0)
        + fractions[:, None] * end[:3, 3].unsqueeze(0)
    )
    return rotations, translations


def _extrapolate_actor_pose_before_torch(
    current_pose: torch.Tensor,
    following_pose: torch.Tensor,
) -> torch.Tensor:
    """Back-extrapolate one 10 Hz actor pose for a window's first sweep."""
    start = current_pose.clone()
    start[:3, 3] = 2.0 * current_pose[:3, 3] - following_pose[:3, 3]
    relative_rotation = current_pose[:3, :3].T @ following_pose[:3, :3]
    start[:3, :3] = current_pose[:3, :3] @ relative_rotation.T
    return start


def _extrapolate_actor_pose_after_torch(
    previous_pose: torch.Tensor,
    current_pose: torch.Tensor,
) -> torch.Tensor:
    """Forward-extrapolate one actor pose from the preceding pair."""
    end = current_pose.clone()
    end[:3, 3] = 2.0 * current_pose[:3, 3] - previous_pose[:3, 3]
    relative_rotation = previous_pose[:3, :3].T @ current_pose[:3, :3]
    end[:3, :3] = current_pose[:3, :3] @ relative_rotation
    return end


def _track_box_index_torch(
    valid_boxes: torch.Tensor,
    track_ids: torch.Tensor,
    frame_idx: int,
    track_id: int,
) -> int | None:
    matches = (
        valid_boxes[frame_idx]
        & (track_ids[frame_idx] == int(track_id))
    ).nonzero(as_tuple=False).flatten()
    if matches.numel() == 0:
        return None
    if matches.numel() != 1:
        raise ValueError(
            f"track {track_id} has {matches.numel()} boxes at frame {frame_idx}"
        )
    return int(matches.item())


def _actor_pose_interval_torch(
    object_to_world: torch.Tensor,
    valid_boxes: torch.Tensor,
    track_ids: torch.Tensor,
    frame_idx: int,
    box_idx: int,
    actor_box_time_mode: str,
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    """Resolve the actor pose interval spanning one LiDAR sweep."""
    if actor_box_time_mode not in _ACTOR_BOX_TIME_MODES:
        raise ValueError(
            f"unsupported actor box time mode {actor_box_time_mode!r}; "
            f"expected {_ACTOR_BOX_TIME_MODES}"
        )
    current = object_to_world[frame_idx, box_idx]
    if actor_box_time_mode == "frame":
        return current, current, False

    track_id = int(track_ids[frame_idx, box_idx].item())
    if track_id < 0:
        return current, current, False
    if actor_box_time_mode == "current_to_next":
        if frame_idx + 1 < object_to_world.shape[0]:
            following_idx = _track_box_index_torch(
                valid_boxes, track_ids, frame_idx + 1, track_id
            )
            if following_idx is not None:
                return (
                    current,
                    object_to_world[frame_idx + 1, following_idx],
                    True,
                )
        if frame_idx > 0:
            previous_idx = _track_box_index_torch(
                valid_boxes, track_ids, frame_idx - 1, track_id
            )
            if previous_idx is not None:
                end = _extrapolate_actor_pose_after_torch(
                    object_to_world[frame_idx - 1, previous_idx], current
                )
                return current, end, True
        return current, current, False
    if frame_idx > 0:
        previous_idx = _track_box_index_torch(
            valid_boxes, track_ids, frame_idx - 1, track_id
        )
        if previous_idx is not None:
            return object_to_world[frame_idx - 1, previous_idx], current, True
    if frame_idx + 1 < object_to_world.shape[0]:
        following_idx = _track_box_index_torch(
            valid_boxes, track_ids, frame_idx + 1, track_id
        )
        if following_idx is not None:
            start = _extrapolate_actor_pose_before_torch(
                current, object_to_world[frame_idx + 1, following_idx]
            )
            return start, current, True
    return current, current, False


def _world_points_to_actor_local_torch(
    points_world: torch.Tensor,
    start_pose: torch.Tensor,
    end_pose: torch.Tensor,
    fractions: torch.Tensor,
) -> torch.Tensor:
    rotations, translations = _interpolate_pose_components_torch(
        start_pose, end_pose, fractions
    )
    return torch.einsum(
        "ni,nij->nj", points_world - translations, rotations
    )


def infer_pose_sensor_start_end_from_lidar_poses(
    lidar_to_world: torch.Tensor,
) -> torch.Tensor:
    """Approximate within-sweep motion from adjacent sweep-start poses."""
    poses = torch.as_tensor(lidar_to_world)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(
            f"lidar_to_world must be (T,4,4), got {tuple(poses.shape)}"
        )
    relative = torch.eye(
        4, device=poses.device, dtype=poses.dtype
    ).repeat(poses.shape[0], 1, 1)
    if poses.shape[0] > 1:
        relative[:-1] = torch.linalg.inv(poses[1:]) @ poses[:-1]
        relative[-1] = relative[-2]
    return relative


def apply_planar_novel_view_offset(
    lidar_to_world: np.ndarray,
    pose_sensor_start_end: np.ndarray | None,
    *,
    start_frame: int,
    forward_m: float,
    left_m: float,
    sensor_to_ego_rotation: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Translate a target trajectory in the ego plane without changing yaw."""
    poses = np.asarray(lidar_to_world)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(
            f"lidar_to_world must be (T,4,4), got {poses.shape}"
        )
    if not 0 <= int(start_frame) < poses.shape[0]:
        raise ValueError(
            f"start_frame must be in [0,{poses.shape[0]}), got {start_frame}"
        )
    if not np.isfinite(forward_m) or not np.isfinite(left_m):
        raise ValueError("forward_m and left_m must be finite")

    sensor_to_ego = (
        np.eye(3, dtype=poses.dtype)
        if sensor_to_ego_rotation is None
        else np.asarray(sensor_to_ego_rotation, dtype=poses.dtype)
    )
    if sensor_to_ego.shape != (3, 3):
        raise ValueError(
            "sensor_to_ego_rotation must be (3,3), got "
            f"{sensor_to_ego.shape}"
        )
    if not np.isfinite(sensor_to_ego).all():
        raise ValueError("sensor_to_ego_rotation must be finite")

    offset = np.eye(4, dtype=poses.dtype)
    ego_offset = np.array(
        [float(forward_m), float(left_m), 0.0],
        dtype=poses.dtype,
    )
    try:
        offset[:3, 3] = np.linalg.solve(sensor_to_ego, ego_offset)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "sensor_to_ego_rotation must be invertible"
        ) from exc
    shifted_poses = poses.copy()
    shifted_poses[start_frame:] = shifted_poses[start_frame:] @ offset

    if pose_sensor_start_end is None:
        return shifted_poses, None
    motion = np.asarray(pose_sensor_start_end)
    if motion.shape != poses.shape:
        raise ValueError(
            "pose_sensor_start_end must match lidar_to_world, got "
            f"{motion.shape} and {poses.shape}"
        )
    shifted_motion = motion.copy()
    inverse_offset = np.linalg.inv(offset)
    shifted_motion[start_frame:] = (
        inverse_offset @ shifted_motion[start_frame:] @ offset
    )
    return shifted_poses, shifted_motion


def apply_lateral_novel_view_offset(
    lidar_to_world: np.ndarray,
    pose_sensor_start_end: np.ndarray | None,
    *,
    start_frame: int,
    left_m: float,
    sensor_to_ego_rotation: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Shift a target trajectory along the ego-frame left axis."""
    return apply_planar_novel_view_offset(
        lidar_to_world,
        pose_sensor_start_end,
        start_frame=start_frame,
        forward_m=0.0,
        left_m=left_m,
        sensor_to_ego_rotation=sensor_to_ego_rotation,
    )


def _normalize_metric_depth_torch(
    metric: torch.Tensor,
    min_range: float,
    max_range: float,
    min_value: float,
    inverse_depth: bool,
) -> torch.Tensor:
    c_min = min_range if not inverse_depth else 1.0 / max_range
    c_max = max_range if not inverse_depth else 1.0 / min_range
    if inverse_depth:
        invalid = metric == 0
        work = torch.where(invalid, metric.new_full((), max_range), metric)
        work = work.reciprocal()
        work = torch.where(invalid, work.new_full((), c_min), work)
    else:
        work = metric
    work = work.clamp(c_min, c_max)
    if min_value == -1:
        return (work - c_min) / (c_max - c_min) * 2.0 - 1.0
    return (work - c_min) / (c_max - c_min)


def compose_normalized_range_channels(
    channels: torch.Tensor,
    *,
    channel_dim: int,
    min_value: float,
    inverse_depth: bool,
) -> torch.Tensor:
    """Compose semantic range channels with the nearest valid return."""
    if channels.ndim < 3:
        raise ValueError(
            "range channels must include channel and spatial dimensions, got "
            f"{tuple(channels.shape)}"
        )
    valid = torch.isfinite(channels) & (
        channels > float(min_value) + 1.0e-5
    )
    if inverse_depth:
        work = channels.masked_fill(~valid, -torch.inf)
        composed = work.amax(dim=channel_dim, keepdim=True)
    else:
        work = channels.masked_fill(~valid, torch.inf)
        composed = work.amin(dim=channel_dim, keepdim=True)
    return torch.where(
        torch.isfinite(composed),
        composed,
        composed.new_full((), float(min_value)),
    )


def compose_normalized_range_feature_layers(
    ranges: torch.Tensor,
    features: torch.Tensor,
    *,
    min_value: float,
    inverse_depth: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compose attributed returns while retaining the nearest return's feature."""
    if ranges.ndim != 3 or features.ndim != 4:
        raise ValueError("ranges/features must be (N,H,W)/(N,F,H,W)")
    if ranges.shape[0] != features.shape[0] or ranges.shape[1:] != features.shape[2:]:
        raise ValueError(
            f"range/feature layers do not align: {ranges.shape}/{features.shape}"
        )
    valid = torch.isfinite(ranges) & (
        ranges > float(min_value) + 1.0e-5
    )
    if inverse_depth:
        work = ranges.masked_fill(~valid, -torch.inf)
        winner = work.argmax(dim=0)
        composed_range = work.amax(dim=0)
    else:
        work = ranges.masked_fill(~valid, torch.inf)
        winner = work.argmin(dim=0)
        composed_range = work.amin(dim=0)
    hit = valid.any(dim=0)
    height, width = ranges.shape[1:]
    flat_winner = winner.reshape(-1)
    flat_pixel = torch.arange(
        height * width, device=ranges.device, dtype=torch.long
    )
    feature_count = features.shape[1]
    flattened = features.permute(0, 2, 3, 1).reshape(
        ranges.shape[0], height * width, feature_count
    )
    composed_features = flattened[flat_winner, flat_pixel].reshape(
        height, width, feature_count
    ).permute(2, 0, 1)
    composed_range = torch.where(
        hit,
        composed_range,
        composed_range.new_full((), float(min_value)),
    )
    composed_features = torch.where(
        hit.unsqueeze(0),
        composed_features,
        composed_features.new_full((), float(min_value)),
    )
    return composed_range.unsqueeze(0), composed_features, hit


@torch.no_grad()
def remove_points_inside_moving_boxes(
    video: torch.Tensor,
    lidar_to_world: torch.Tensor,
    box_object_to_world: torch.Tensor,
    box_lwh: torch.Tensor,
    box_valid: torch.Tensor,
    box_track_id: torch.Tensor | None = None,
    *,
    ray_dirs: torch.Tensor,
    min_range: float,
    max_range: float,
    min_value: float,
    inverse_depth: bool,
    pose_sensor_start_end: torch.Tensor | None = None,
    actor_box_time_mode: str = "frame",
    point_chunk_size: int = 8192,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Remove LiDAR returns lying inside supplied moving 3-D boxes."""
    if video.ndim != 4 or video.shape[0] < 1:
        raise ValueError(
            "dynamic filtering requires (C,T,H,W) with range in channel 0, got "
            f"{tuple(video.shape)}"
        )
    num_frames, height, width = map(int, video.shape[1:])
    poses = torch.as_tensor(
        lidar_to_world, device=video.device, dtype=torch.float32
    )
    if poses.shape != (num_frames, 4, 4):
        raise ValueError(
            "lidar_to_world must align one pose per frame, got "
            f"{tuple(poses.shape)} for T={num_frames}"
        )
    motion = None
    if pose_sensor_start_end is not None:
        motion = torch.as_tensor(
            pose_sensor_start_end,
            device=video.device,
            dtype=torch.float32,
        )
        if motion.shape != (num_frames, 4, 4):
            raise ValueError(
                "pose_sensor_start_end must align one transform per frame, "
                f"got {tuple(motion.shape)} for T={num_frames}"
            )
    if (
        box_object_to_world.ndim != 4
        or box_object_to_world.shape[:2] != box_lwh.shape[:2]
        or box_object_to_world.shape[:2] != box_valid.shape
        or box_object_to_world.shape[0] != num_frames
        or box_object_to_world.shape[2:] != (4, 4)
        or box_lwh.shape[2:] != (3,)
    ):
        raise ValueError(
            "moving boxes must be aligned as (T,M,4,4)/(T,M,3)/(T,M), "
            f"got {tuple(box_object_to_world.shape)}, "
            f"{tuple(box_lwh.shape)}, {tuple(box_valid.shape)}"
        )
    if actor_box_time_mode not in _ACTOR_BOX_TIME_MODES:
        raise ValueError(
            f"unsupported actor box time mode {actor_box_time_mode!r}; "
            f"expected {_ACTOR_BOX_TIME_MODES}"
        )
    track_ids = None
    if box_track_id is not None:
        track_ids = torch.as_tensor(
            box_track_id, device=video.device, dtype=torch.long
        )
        if track_ids.shape != box_valid.shape:
            raise ValueError(
                "box_track_id must align with box_valid, got "
                f"{tuple(track_ids.shape)} and {tuple(box_valid.shape)}"
            )
    if actor_box_time_mode != "frame" and track_ids is None:
        raise ValueError(
            "ray-time actor box interpolation requires box_track_id"
        )
    object_to_world_all = torch.as_tensor(
        box_object_to_world, device=video.device, dtype=torch.float32
    )
    lwh_all = torch.as_tensor(
        box_lwh, device=video.device, dtype=torch.float32
    )
    valid_boxes_all = torch.as_tensor(
        box_valid, device=video.device, dtype=torch.bool
    )
    rays = torch.as_tensor(
        ray_dirs, device=video.device, dtype=torch.float32
    )
    if rays.shape != (height, width, 3):
        raise ValueError(
            "ray grid must match the video spatial shape, got "
            f"{tuple(rays.shape)} vs {(height, width, 3)}"
        )
    if point_chunk_size <= 0:
        raise ValueError("point_chunk_size must be positive")

    filtered = video.clone()
    removed_count = 0
    valid_count = 0
    c_min = min_range if not inverse_depth else 1.0 / max_range
    c_max = max_range if not inverse_depth else 1.0 / min_range
    for frame_idx in range(num_frames):
        sweep = video[0, frame_idx].float()
        valid = torch.isfinite(sweep) & (
            sweep > float(min_value) + 1.0e-5
        )
        flat_indices = valid.reshape(-1).nonzero(as_tuple=False).flatten()
        valid_count += int(flat_indices.numel())
        frame_boxes = box_valid[frame_idx].to(
            device=video.device, dtype=torch.bool
        )
        if flat_indices.numel() == 0 or not bool(frame_boxes.any()):
            continue

        if min_value == -1:
            depth = (sweep + 1.0) * 0.5 * (c_max - c_min) + c_min
        else:
            depth = sweep * (c_max - c_min) + c_min
        if inverse_depth:
            depth = depth.reciprocal()
        points_sensor = (
            depth[..., None] * rays
        ).reshape(-1, 3).index_select(0, flat_indices)
        columns = torch.remainder(flat_indices, width).to(
            dtype=points_sensor.dtype
        )
        fractions = columns / float(max(width - 1, 1))
        if motion is not None:
            points_sensor = _fractional_transform_points_torch(
                points_sensor,
                torch.linalg.inv(motion[frame_idx]),
                fractions,
            )

        pose = poses[frame_idx]
        points_world = (
            points_sensor @ pose[:3, :3].T + pose[:3, 3]
        )
        remove = torch.zeros(
            points_world.shape[0],
            device=video.device,
            dtype=torch.bool,
        )
        if actor_box_time_mode == "frame":
            object_to_world = object_to_world_all[frame_idx, frame_boxes]
            half_lwh = 0.5 * lwh_all[frame_idx, frame_boxes]
            centers = object_to_world[:, :3, 3]
            rotations = object_to_world[:, :3, :3]
            for start in range(0, points_world.shape[0], point_chunk_size):
                end = min(start + point_chunk_size, points_world.shape[0])
                delta = points_world[start:end, None, :] - centers[None]
                points_object = torch.einsum(
                    "nmi,mij->nmj", delta, rotations
                )
                remove[start:end] = (
                    points_object.abs() <= half_lwh[None] + 1.0e-3
                ).all(dim=-1).any(dim=-1)
        else:
            assert track_ids is not None
            for box_idx in valid_boxes_all[frame_idx].nonzero(
                as_tuple=False
            ).flatten().tolist():
                start_pose, end_pose, _ = _actor_pose_interval_torch(
                    object_to_world_all,
                    valid_boxes_all,
                    track_ids,
                    frame_idx,
                    box_idx,
                    actor_box_time_mode,
                )
                for start in range(0, points_world.shape[0], point_chunk_size):
                    end = min(start + point_chunk_size, points_world.shape[0])
                    points_object = _world_points_to_actor_local_torch(
                        points_world[start:end],
                        start_pose,
                        end_pose,
                        fractions[start:end],
                    )
                    remove[start:end] |= (
                        points_object.abs()
                        <= 0.5 * lwh_all[frame_idx, box_idx] + 1.0e-3
                    ).all(dim=1)

        remove_indices = flat_indices[remove]
        removed_count += int(remove_indices.numel())
        filtered[:, frame_idx].reshape(video.shape[0], -1)[
            :, remove_indices
        ] = float(min_value)

    fraction = filtered.new_tensor(
        float(removed_count) / float(max(valid_count, 1)),
        dtype=torch.float32,
    )
    return filtered, fraction


def local_hit_density_confidence(
    hit: torch.Tensor,
    *,
    kernel_size: int = 5,
    saturation_hits: float = 5.0,
) -> torch.Tensor:
    """Return circular-azimuth local hit density in ``[0, 1]``."""
    if hit.ndim < 2:
        raise ValueError(f"hit must end in (H,W), got {tuple(hit.shape)}")
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError(
            f"kernel_size must be a positive odd integer, got {kernel_size}"
        )
    if saturation_hits <= 0.0:
        raise ValueError(
            f"saturation_hits must be positive, got {saturation_hits}"
        )
    density_input = hit.float().reshape(
        -1, 1, hit.shape[-2], hit.shape[-1]
    )
    pad = kernel_size // 2
    if pad:
        density_input = torch.cat(
            [
                density_input[..., -pad:],
                density_input,
                density_input[..., :pad],
            ],
            dim=-1,
        )
    density_counts = torch.nn.functional.conv2d(
        density_input,
        torch.ones(
            (1, 1, kernel_size, kernel_size),
            device=density_input.device,
            dtype=density_input.dtype,
        ),
        padding=(pad, 0),
    ).reshape_as(hit)
    return (
        density_counts / float(saturation_hits)
    ).clamp_(0.0, 1.0) * hit.float()


def spatial_warp_window(
    global_warp: torch.Tensor,
    window_start: int,
    window_frames: int,
    *,
    input_has_prepend: bool = True,
) -> torch.Tensor:
    """Build a model-aligned window from a global warp stream."""
    if global_warp.ndim != 5:
        raise ValueError(
            f"global_warp must be (B,C,T,H,W), got {global_warp.shape}"
        )
    if window_start < 0 or window_frames <= 0:
        raise ValueError(
            f"invalid warp window start={window_start}, frames={window_frames}"
        )
    if not input_has_prepend:
        result = global_warp[
            :, :, window_start : window_start + window_frames
        ]
        if result.shape[2] != window_frames:
            raise ValueError(
                "global warp does not contain the requested unique-frame "
                f"window: start={window_start}, frames={window_frames}, "
                f"available_T={global_warp.shape[2]}"
            )
        return result.contiguous()
    if window_start == 0:
        result = global_warp[:, :, : window_frames + 1]
    else:
        first_real_token = window_start + 1
        real = global_warp[
            :, :, first_real_token : first_real_token + window_frames
        ]
        if real.shape[2] != window_frames:
            raise ValueError(
                "global warp does not contain the requested real-frame window: "
                f"start={window_start}, frames={window_frames}, "
                f"available_T={global_warp.shape[2]}"
            )
        result = torch.cat([real[:, :, :1], real], dim=2)
    if result.shape[2] != window_frames + 1:
        raise ValueError(
            f"warp window has T={result.shape[2]}, expected {window_frames + 1}"
        )
    return result.contiguous()


def lidar_poses_for_global_tokens(
    lidar_to_world: torch.Tensor,
    global_start: int,
    block_size: int,
    *,
    input_has_prepend: bool = True,
) -> torch.Tensor:
    """Map global token indices to their real LiDAR target poses."""
    if lidar_to_world.ndim != 3 or lidar_to_world.shape[1:] != (4, 4):
        raise ValueError(
            "lidar_to_world must be (S,4,4), got "
            f"{tuple(lidar_to_world.shape)}"
        )
    token_indices = torch.arange(
        global_start,
        global_start + block_size,
        device=lidar_to_world.device,
        dtype=torch.long,
    )
    pose_indices = (token_indices - int(input_has_prepend)).clamp(
        min=0, max=lidar_to_world.shape[0] - 1
    )
    return lidar_to_world.index_select(0, pose_indices)


@torch.autocast("cuda", enabled=False)
def sensor_tables_from_ray_directions(
    ray_dirs: Any,
    *,
    allow_approximate: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Recover coarse elevation/azimuth tables from calibrated rays.

    Native spinning-LiDAR grids encode azimuth as one shared column table plus
    a constant per-row offset. Some calibrated grids are not exactly separable;
    callers that retain the exact ray grid can allow an approximate table for
    coarse lookup and refine against the original rays during rasterization.
    """
    rays = torch.as_tensor(ray_dirs, dtype=torch.float32)
    if rays.ndim != 3 or rays.shape[-1] != 3:
        raise ValueError(
            f"ray_dirs must be (H,W,3), got {tuple(rays.shape)}"
        )
    norm = torch.linalg.vector_norm(rays, dim=-1, keepdim=True)
    if not torch.isfinite(norm).all() or bool((norm <= 0).any()):
        raise ValueError("ray_dirs must contain finite nonzero vectors")
    rays = rays / norm

    elevation_grid = torch.atan2(
        rays[..., 2],
        torch.linalg.vector_norm(rays[..., :2], dim=-1),
    )
    elevations = elevation_grid.mean(dim=1)
    azimuth_grid = torch.atan2(rays[..., 1], rays[..., 0])
    azimuths = azimuth_grid[0].contiguous()
    relative = _wrap_angle_torch(azimuth_grid - azimuths[None])
    row_offsets = torch.atan2(
        torch.sin(relative).mean(dim=1),
        torch.cos(relative).mean(dim=1),
    )

    reconstructed = torch.stack(
        (
            torch.cos(elevations)[:, None]
            * torch.cos(azimuths[None] + row_offsets[:, None]),
            torch.cos(elevations)[:, None]
            * torch.sin(azimuths[None] + row_offsets[:, None]),
            torch.sin(elevations)[:, None].expand(-1, rays.shape[1]),
        ),
        dim=-1,
    )
    max_error = torch.linalg.vector_norm(
        reconstructed - rays, dim=-1
    ).max()
    if not torch.isfinite(max_error):
        raise ValueError("reconstructed sensor rays must be finite")
    if float(max_error) > 1e-3 and not allow_approximate:
        raise ValueError(
            "sensor rays are not representable by a spinning-LiDAR row/column "
            f"grid (max direction error={float(max_error):.6g})"
        )
    return elevations, azimuths, row_offsets


class TorchSpatialPointMap:
    """Persistent pose-registered voxel map and local LiDAR rasterizer.

    Points remain on the caller's device and are compacted to one newest point
    per globally anchored voxel when new sweeps are appended. The optional
    ``persistent_static`` policy promotes a voxel only after it is occupied in
    enough selected observation units. ``block`` counts at most once per
    committed block, while ``frame`` counts at most once per distinct frame
    within that block. The newest sweep can remain as an ephemeral layer so
    current dynamic objects stay visible without leaving long trails. Optional
    free-space clearing removes a confirmed voxel only
    after repeated in-range current rays pass through it to farther returns;
    nearer returns, missing rays, and out-of-range voxels are treated as
    unobservable. Rendering queries a GPU-resident coarse XY cell index,
    applies an exact target-local radius mask to that temporary subset, and
    leaves the global map unchanged.
    """

    @torch.autocast("cuda", enabled=False)
    def __init__(
        self,
        *,
        ray_dirs: Any,
        row_elevations_rad: Any,
        column_azimuths_rad: Any,
        row_azimuth_offsets_rad: Any,
        min_range: float,
        max_range: float,
        min_value: float = -1.0,
        inverse_depth: bool = False,
        radius_m: float = 75.0,
        voxel_size_m: float = 0.2,
        voxel_policy: str = "newest",
        observation_unit: str = "block",
        static_min_observations: int = 2,
        candidate_ttl_updates: int = 1,
        include_latest: bool = True,
        free_space_clear: bool = False,
        free_space_margin_m: float = 0.75,
        free_space_misses_to_remove: int = 2,
        feature_channels: int = 0,
        device: torch.device | str | None = None,
    ):
        if radius_m <= 0.0:
            raise ValueError(f"radius_m must be positive, got {radius_m}")
        if voxel_size_m <= 0.0:
            raise ValueError(
                f"voxel_size_m must be positive, got {voxel_size_m}"
            )
        if voxel_policy not in ("newest", "persistent_static"):
            raise ValueError(
                f"unsupported voxel policy {voxel_policy!r}; expected "
                "'newest' or 'persistent_static'"
            )
        if observation_unit not in ("block", "frame"):
            raise ValueError(
                f"unsupported observation unit {observation_unit!r}; expected "
                "'block' or 'frame'"
            )
        if static_min_observations < 1:
            raise ValueError(
                "static_min_observations must be positive, got "
                f"{static_min_observations}"
            )
        if candidate_ttl_updates < 1:
            raise ValueError(
                "candidate_ttl_updates must be positive, got "
                f"{candidate_ttl_updates}"
            )
        if free_space_clear and voxel_policy != "persistent_static":
            raise ValueError(
                "free_space_clear requires voxel_policy='persistent_static'"
            )
        if free_space_margin_m <= 0.0:
            raise ValueError(
                "free_space_margin_m must be positive, got "
                f"{free_space_margin_m}"
            )
        if free_space_misses_to_remove < 1:
            raise ValueError(
                "free_space_misses_to_remove must be positive, got "
                f"{free_space_misses_to_remove}"
            )
        self.ray_dirs = torch.as_tensor(
            ray_dirs, device=device, dtype=torch.float32
        ).contiguous()
        if self.ray_dirs.ndim != 3 or self.ray_dirs.shape[-1] != 3:
            raise ValueError(
                f"ray_dirs must be (H,W,3), got {self.ray_dirs.shape}"
            )
        ray_norm = torch.linalg.vector_norm(
            self.ray_dirs, dim=-1, keepdim=True
        )
        if not torch.isfinite(ray_norm).all() or bool((ray_norm <= 0).any()):
            raise ValueError("ray_dirs must contain finite nonzero vectors")
        self.ray_dirs = (self.ray_dirs / ray_norm).contiguous()
        self.height, self.width = map(int, self.ray_dirs.shape[:2])
        self.elev = torch.as_tensor(
            row_elevations_rad, device=self.ray_dirs.device, dtype=torch.float32
        ).contiguous()
        self.azim = torch.as_tensor(
            column_azimuths_rad,
            device=self.ray_dirs.device,
            dtype=torch.float32,
        ).contiguous()
        self.row_off = torch.as_tensor(
            row_azimuth_offsets_rad,
            device=self.ray_dirs.device,
            dtype=torch.float32,
        ).contiguous()
        if self.elev.shape != (self.height,):
            raise ValueError(
                f"row elevations must be {(self.height,)}, got {self.elev.shape}"
            )
        if self.row_off.shape != (self.height,):
            raise ValueError(
                f"row offsets must be {(self.height,)}, got {self.row_off.shape}"
            )
        if self.azim.shape != (self.width,):
            raise ValueError(
                f"column azimuths must be {(self.width,)}, got {self.azim.shape}"
            )

        self.elev_order = torch.argsort(self.elev, stable=True)
        self.elev_sorted = self.elev[self.elev_order].contiguous()
        azim_wrapped = _wrap_angle_torch(self.azim)
        self.azim_order = torch.argsort(azim_wrapped, stable=True)
        self.azim_sorted = azim_wrapped[self.azim_order].contiguous()
        reconstructed_dirs = torch.stack(
            (
                torch.cos(self.elev)[:, None]
                * torch.cos(self.azim[None] + self.row_off[:, None]),
                torch.cos(self.elev)[:, None]
                * torch.sin(self.azim[None] + self.row_off[:, None]),
                torch.sin(self.elev)[:, None].expand(-1, self.width),
            ),
            dim=-1,
        )
        table_error = torch.linalg.vector_norm(
            reconstructed_dirs - self.ray_dirs, dim=-1
        ).max()
        if not torch.isfinite(table_error):
            raise ValueError("sensor lookup tables must reconstruct finite rays")
        self.refine_exact_beam_lookup = bool(float(table_error) > 1e-3)
        self.min_range = float(min_range)
        self.max_range = float(max_range)
        self.min_value = float(min_value)
        self.inverse_depth = bool(inverse_depth)
        self.radius_m = float(radius_m)
        self.voxel_size_m = float(voxel_size_m)
        self.voxel_policy = voxel_policy
        self.observation_unit = observation_unit
        self.static_min_observations = int(static_min_observations)
        self.candidate_ttl_updates = int(candidate_ttl_updates)
        self.include_latest = bool(include_latest)
        self.free_space_clear = bool(free_space_clear)
        self.free_space_margin_m = float(free_space_margin_m)
        self.free_space_misses_to_remove = int(free_space_misses_to_remove)
        self.feature_channels = int(feature_channels)
        if self.feature_channels < 0:
            raise ValueError("feature_channels must be non-negative")
        # A current beam must pass through the stored voxel, up to 10 cm of
        # calibration/pose slack, before a farther return counts as free space.
        self.free_space_lateral_tolerance_m = (
            0.5 * math.sqrt(3.0) * self.voxel_size_m + 0.1
        )
        # Four cells per radius keeps the coarse square query close to the
        # exact target disk while requiring only a fixed 9x9 lookup stencil.
        self.spatial_cell_size_m = max(
            self.voxel_size_m, self.radius_m / 4.0
        )
        cell_radius = int(
            math.ceil(self.radius_m / self.spatial_cell_size_m)
        )
        cell_axis = torch.arange(
            -cell_radius,
            cell_radius + 1,
            device=self.ray_dirs.device,
            dtype=torch.int64,
        )
        cell_x, cell_y = torch.meshgrid(
            cell_axis, cell_axis, indexing="ij"
        )
        self.query_cell_offsets = torch.stack(
            [cell_x.reshape(-1), cell_y.reshape(-1)], dim=1
        ).contiguous()
        self.points_world = torch.empty(
            (0, 3), device=self.ray_dirs.device, dtype=torch.float32
        )
        self.point_features = torch.empty(
            (0, self.feature_channels),
            device=self.ray_dirs.device,
            dtype=torch.float32,
        )
        self.timestamps = torch.empty(
            (0,), device=self.ray_dirs.device, dtype=torch.long
        )
        self.observation_counts = torch.empty(
            (0,), device=self.ray_dirs.device, dtype=torch.long
        )
        self.last_seen_updates = torch.empty(
            (0,), device=self.ray_dirs.device, dtype=torch.long
        )
        self.free_miss_counts = torch.empty(
            (0,), device=self.ray_dirs.device, dtype=torch.long
        )
        self.latest_points_world = torch.empty(
            (0, 3), device=self.ray_dirs.device, dtype=torch.float32
        )
        self.latest_point_features = torch.empty(
            (0, self.feature_channels),
            device=self.ray_dirs.device,
            dtype=torch.float32,
        )
        self.latest_timestamp = -1
        self._update_index = -1
        self.last_free_space_cleared = 0
        self.total_free_space_cleared = 0
        # One packed coarse-XY key per point, sorted for searchsorted lookup.
        self.cell_keys = torch.empty(
            (0,), device=self.ray_dirs.device, dtype=torch.int64
        )

    @property
    def num_points(self) -> int:
        return int(self.points_world.shape[0])

    @property
    def confirmed_mask(self) -> torch.Tensor:
        if self.voxel_policy != "persistent_static":
            return torch.ones(
                self.points_world.shape[0],
                device=self.points_world.device,
                dtype=torch.bool,
            )
        if self.observation_counts.shape[0] != self.points_world.shape[0]:
            raise RuntimeError("persistent-static observation state is stale")
        return self.observation_counts >= self.static_min_observations

    @property
    def confirmed_points_world(self) -> torch.Tensor:
        return self.points_world[self.confirmed_mask]

    @property
    def confirmed_point_features(self) -> torch.Tensor:
        return self.point_features[self.confirmed_mask]

    @property
    def num_confirmed_points(self) -> int:
        return int(self.confirmed_mask.sum().item())

    @property
    def num_candidate_points(self) -> int:
        return self.num_points - self.num_confirmed_points

    @property
    def num_latest_points(self) -> int:
        return int(self.latest_points_world.shape[0])

    def _normalized_sweep_to_metric(
        self, normalized_range: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sweep = normalized_range.to(
            device=self.ray_dirs.device, dtype=torch.float32
        )
        if sweep.ndim == 3 and sweep.shape[0] == 1:
            sweep = sweep[0]
        if sweep.shape != (self.height, self.width):
            raise ValueError(
                "normalized sweep must match the ray grid: "
                f"got {tuple(sweep.shape)}, expected "
                f"{(self.height, self.width)}"
            )
        valid = torch.isfinite(sweep) & (
            sweep > self.min_value + 1.0e-5
        )
        c_min = (
            self.min_range
            if not self.inverse_depth
            else 1.0 / self.max_range
        )
        c_max = (
            self.max_range
            if not self.inverse_depth
            else 1.0 / self.min_range
        )
        if self.min_value == -1:
            depth = (sweep + 1.0) * 0.5 * (c_max - c_min) + c_min
        else:
            depth = sweep * (c_max - c_min) + c_min
        if self.inverse_depth:
            depth = depth.reciprocal()
        return depth, valid

    def _sweep_to_world(
        self,
        normalized_range: torch.Tensor,
        lidar_to_world: torch.Tensor,
        pose_sensor_start_end: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pose = lidar_to_world.to(
            device=self.ray_dirs.device, dtype=torch.float32
        )
        if pose.shape != (4, 4):
            raise ValueError(f"lidar_to_world must be (4,4), got {pose.shape}")

        depth, valid = self._normalized_sweep_to_metric(normalized_range)
        points_local = (depth[..., None] * self.ray_dirs)[valid]
        if points_local.numel() == 0:
            return self.points_world.new_empty((0, 3))
        if pose_sensor_start_end is not None:
            relative = torch.as_tensor(
                pose_sensor_start_end,
                device=points_local.device,
                dtype=points_local.dtype,
            )
            if relative.shape != (4, 4):
                raise ValueError(
                    "pose_sensor_start_end must be (4,4), got "
                    f"{tuple(relative.shape)}"
                )
            columns = torch.arange(
                self.width,
                device=points_local.device,
                dtype=points_local.dtype,
            ).view(1, self.width).expand(self.height, self.width)[valid]
            fractions = columns / float(max(self.width - 1, 1))
            points_local = _fractional_transform_points_torch(
                points_local,
                torch.linalg.inv(relative),
                fractions,
            )
        return (
            points_local @ pose[:3, :3].T + pose[:3, 3].unsqueeze(0)
        )

    def _sweep_to_world_with_features(
        self,
        normalized_range: torch.Tensor,
        point_features: torch.Tensor | None,
        lidar_to_world: torch.Tensor,
        pose_sensor_start_end: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Lift one sweep and preserve attributes on the same valid returns."""
        points_world = self._sweep_to_world(
            normalized_range,
            lidar_to_world,
            pose_sensor_start_end,
        )
        if self.feature_channels == 0:
            if (
                point_features is not None
                and torch.as_tensor(point_features).numel() > 0
            ):
                raise ValueError("this spatial map was created without features")
            return points_world, self.point_features.new_empty(
                (points_world.shape[0], 0)
            )
        if point_features is None:
            raise ValueError(
                f"spatial map requires {self.feature_channels} point features"
            )
        features = torch.as_tensor(
            point_features,
            device=self.ray_dirs.device,
            dtype=torch.float32,
        )
        if features.ndim == 2 and self.feature_channels == 1:
            features = features.unsqueeze(0)
        if features.shape != (
            self.feature_channels,
            self.height,
            self.width,
        ):
            raise ValueError(
                "point features must be (F,H,W), got "
                f"{tuple(features.shape)}"
            )
        _, valid = self._normalized_sweep_to_metric(normalized_range)
        selected = features[:, valid].T.contiguous()
        if selected.shape[0] != points_world.shape[0]:
            raise RuntimeError("range points and point features lost alignment")
        return points_world, selected

    @torch.autocast("cuda", enabled=False)
    def _nearest_beam_indices(
        self, points_sensor: torch.Tensor, ranges: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        point_elev = torch.asin(
            (points_sensor[:, 2] / ranges).clamp(-1.0, 1.0)
        )
        elev_idx = torch.searchsorted(self.elev_sorted, point_elev)
        elev_idx0 = (elev_idx - 1).clamp(0, self.height - 1)
        elev_idx1 = elev_idx.clamp(0, self.height - 1)
        elev_dist0 = (point_elev - self.elev_sorted[elev_idx0]).abs()
        elev_dist1 = (point_elev - self.elev_sorted[elev_idx1]).abs()
        choose0 = (elev_dist0 < elev_dist1) | (
            (elev_dist0 == elev_dist1)
            & (
                self.elev_order[elev_idx0]
                <= self.elev_order[elev_idx1]
            )
        )
        nearest_sorted = torch.where(choose0, elev_idx0, elev_idx1)
        rows = self.elev_order[nearest_sorted]

        point_azim = _wrap_angle_torch(
            torch.atan2(points_sensor[:, 1], points_sensor[:, 0])
            - self.row_off[rows]
        )
        azim_idx = torch.searchsorted(self.azim_sorted, point_azim)
        azim_idx0 = torch.remainder(azim_idx - 1, self.width)
        azim_idx1 = torch.remainder(azim_idx, self.width)
        azim_dist0 = _wrap_angle_torch(
            point_azim - self.azim_sorted[azim_idx0]
        ).abs()
        azim_dist1 = _wrap_angle_torch(
            point_azim - self.azim_sorted[azim_idx1]
        ).abs()
        nearest_azim = torch.where(
            azim_dist0 <= azim_dist1, azim_idx0, azim_idx1
        )
        columns = self.azim_order[nearest_azim]
        if not self.refine_exact_beam_lookup or points_sensor.shape[0] == 0:
            return rows, columns

        unit_directions = points_sensor / ranges[:, None]
        refined_rows = torch.empty_like(rows)
        refined_columns = torch.empty_like(columns)
        row_deltas = torch.arange(
            -3, 4, device=rows.device, dtype=torch.long
        )
        column_deltas = torch.arange(
            -4, 4, device=rows.device, dtype=torch.long
        )
        # Bound temporary candidate tensors for long persistent-map queries.
        for start in range(0, points_sensor.shape[0], 32768):
            stop = min(start + 32768, points_sensor.shape[0])
            directions = unit_directions[start:stop]
            row_sorted = nearest_sorted[start:stop, None] + row_deltas[None]
            row_sorted = row_sorted.clamp(0, self.height - 1)
            candidate_rows = self.elev_order[row_sorted]

            point_azimuth = torch.atan2(directions[:, 1], directions[:, 0])
            adjusted_azimuth = _wrap_angle_torch(
                point_azimuth[:, None] - self.row_off[candidate_rows]
            ).contiguous()
            column_sorted = torch.searchsorted(
                self.azim_sorted, adjusted_azimuth
            )
            column_sorted = torch.remainder(
                column_sorted[..., None] + column_deltas,
                self.width,
            )
            candidate_columns = self.azim_order[column_sorted]
            candidate_rays = self.ray_dirs[
                candidate_rows[..., None], candidate_columns
            ]
            similarity = (
                candidate_rays * directions[:, None, None]
            ).sum(dim=-1)
            best = similarity.flatten(1).argmax(dim=1)
            best_row = torch.div(
                best, column_deltas.numel(), rounding_mode="floor"
            )
            best_column = torch.remainder(best, column_deltas.numel())
            batch = torch.arange(
                stop - start, device=rows.device, dtype=torch.long
            )
            refined_rows[start:stop] = candidate_rows[batch, best_row]
            refined_columns[start:stop] = candidate_columns[
                batch, best_row, best_column
            ]
        return refined_rows, refined_columns

    def _apply_free_space_clearing(
        self,
        normalized_range: torch.Tensor,
        lidar_to_world: torch.Tensor,
        *,
        pose_sensor_start_end: torch.Tensor | None = None,
    ) -> int:
        """Remove confirmed voxels contradicted by repeated visible free space."""
        if (
            not self.free_space_clear
            or self.points_world.shape[0] == 0
        ):
            return 0
        if self.free_miss_counts.shape[0] != self.points_world.shape[0]:
            raise RuntimeError("persistent-static free-space state is stale")

        pose = lidar_to_world.to(
            device=self.points_world.device, dtype=torch.float32
        )
        depth, valid = self._normalized_sweep_to_metric(normalized_range)
        map_indices = self._local_point_indices_for_targets(
            pose.unsqueeze(0), confirmed_only=True
        )
        if map_indices.numel() == 0:
            return 0
        points_world = self.points_world[map_indices]
        if pose_sensor_start_end is None:
            points_sensor = (
                points_world - pose[:3, 3].unsqueeze(0)
            ) @ pose[:3, :3]
            ranges = torch.linalg.vector_norm(points_sensor, dim=1)
            rows, cols = self._nearest_beam_indices(
                points_sensor, ranges.clamp_min(1.0e-8)
            )
        else:
            points_sensor, ranges, rows, cols = (
                self._points_world_to_rolling_sensor(
                    pose,
                    points_world,
                    pose_sensor_start_end,
                )
            )
        in_observable_range = (
            (ranges >= self.min_range)
            & (ranges <= self.max_range)
        )
        if not bool(in_observable_range.any()):
            return 0

        map_indices = map_indices[in_observable_range]
        points_sensor = points_sensor[in_observable_range]
        ranges = ranges[in_observable_range]
        rows = rows[in_observable_range]
        cols = cols[in_observable_range]
        nearest_dirs = self.ray_dirs[rows, cols]
        along_ray = (points_sensor * nearest_dirs).sum(dim=1)
        lateral_sq = (
            ranges.square() - along_ray.square()
        ).clamp_min(0.0)
        near_real_beam = (
            (along_ray > 0.0)
            & (
                lateral_sq
                <= self.free_space_lateral_tolerance_m**2
            )
        )
        if not bool(near_real_beam.any()):
            return 0

        map_indices = map_indices[near_real_beam]
        ranges = ranges[near_real_beam]
        rows = rows[near_real_beam]
        cols = cols[near_real_beam]
        offsets = torch.arange(
            -1, 2, device=rows.device, dtype=torch.long
        )
        row_grid, col_grid = torch.meshgrid(
            offsets, offsets, indexing="ij"
        )
        neighbor_rows = (
            rows[:, None] + row_grid.reshape(1, -1)
        ).clamp(0, self.height - 1)
        neighbor_cols = torch.remainder(
            cols[:, None] + col_grid.reshape(1, -1),
            self.width,
        )
        neighbor_depth = depth[neighbor_rows, neighbor_cols]
        neighbor_valid = valid[neighbor_rows, neighbor_cols]
        range_delta = neighbor_depth - ranges[:, None]
        reobserved = (
            neighbor_valid
            & (range_delta.abs() <= self.free_space_margin_m)
        ).any(dim=1)
        has_valid_return = neighbor_valid.any(dim=1)
        all_valid_returns_farther = (
            (~neighbor_valid)
            | (range_delta > self.free_space_margin_m)
        ).all(dim=1)
        contradicted = (
            ~reobserved & has_valid_return & all_valid_returns_farther
        )

        miss_counts = self.free_miss_counts[map_indices]
        miss_counts = torch.where(
            reobserved, torch.zeros_like(miss_counts), miss_counts
        )
        miss_counts = torch.where(
            contradicted, miss_counts + 1, miss_counts
        )
        self.free_miss_counts[map_indices] = miss_counts
        remove_indices = map_indices[
            miss_counts >= self.free_space_misses_to_remove
        ]
        if remove_indices.numel() == 0:
            return 0

        keep = torch.ones(
            self.points_world.shape[0],
            device=self.points_world.device,
            dtype=torch.bool,
        )
        keep[remove_indices] = False
        self.points_world = self.points_world[keep]
        self.point_features = self.point_features[keep]
        self.timestamps = self.timestamps[keep]
        self.observation_counts = self.observation_counts[keep]
        self.last_seen_updates = self.last_seen_updates[keep]
        self.free_miss_counts = self.free_miss_counts[keep]
        cleared = int(remove_indices.numel())
        self.total_free_space_cleared += cleared
        self._rebuild_cell_index()
        return cleared

    def append_block(
        self,
        normalized_block: torch.Tensor,
        lidar_to_world: torch.Tensor,
        global_start: int,
        *,
        point_feature_block: torch.Tensor | None = None,
        input_has_prepend: bool = True,
        pose_sensor_start_end: torch.Tensor | None = None,
    ) -> None:
        """Append committed real sweeps using the configured token layout."""
        token_poses = lidar_poses_for_global_tokens(
            lidar_to_world.to(device=self.ray_dirs.device),
            global_start,
            normalized_block.shape[-3],
            input_has_prepend=input_has_prepend,
        )
        token_motion = (
            lidar_poses_for_global_tokens(
                pose_sensor_start_end.to(device=self.ray_dirs.device),
                global_start,
                normalized_block.shape[-3],
                input_has_prepend=input_has_prepend,
            )
            if pose_sensor_start_end is not None
            else None
        )
        self.append_token_block(
            normalized_block=normalized_block,
            point_feature_block=point_feature_block,
            token_lidar_to_world=token_poses,
            token_pose_sensor_start_end=token_motion,
            global_start=global_start,
            input_has_prepend=input_has_prepend,
        )

    def append_token_block(
        self,
        normalized_block: torch.Tensor,
        token_lidar_to_world: torch.Tensor,
        global_start: int,
        *,
        point_feature_block: torch.Tensor | None = None,
        input_has_prepend: bool = True,
        token_pose_sensor_start_end: torch.Tensor | None = None,
    ) -> None:
        """Append committed sweeps with one explicit pose per global token."""
        block = normalized_block
        if block.ndim == 4:
            if block.shape[0] != 1:
                raise ValueError(
                    "normalized block with channels must have C=1, got "
                    f"{tuple(block.shape)}"
                )
            block = block[0]
        if block.ndim != 3:
            raise ValueError(
                f"normalized_block must be (T,H,W), got {tuple(block.shape)}"
            )
        feature_block = None
        if self.feature_channels > 0:
            if point_feature_block is None:
                raise ValueError(
                    f"spatial map requires {self.feature_channels} features"
                )
            feature_block = torch.as_tensor(
                point_feature_block,
                device=self.ray_dirs.device,
                dtype=torch.float32,
            )
            if feature_block.shape != (
                self.feature_channels,
                *block.shape,
            ):
                raise ValueError(
                    "point_feature_block must be (F,T,H,W), got "
                    f"{tuple(feature_block.shape)}"
                )
        elif (
            point_feature_block is not None
            and torch.as_tensor(point_feature_block).numel() > 0
        ):
            raise ValueError("this spatial map was created without features")
        poses = token_lidar_to_world.to(device=self.ray_dirs.device)
        if poses.shape != (block.shape[0], 4, 4):
            raise ValueError(
                "token_lidar_to_world must be (T,4,4) aligned with the block, "
                f"got {tuple(poses.shape)} for T={block.shape[0]}"
            )
        motion = None
        if token_pose_sensor_start_end is not None:
            motion = token_pose_sensor_start_end.to(
                device=self.ray_dirs.device, dtype=torch.float32
            )
            if motion.shape != (block.shape[0], 4, 4):
                raise ValueError(
                    "token_pose_sensor_start_end must be (T,4,4) aligned "
                    f"with the block, got {tuple(motion.shape)} for "
                    f"T={block.shape[0]}"
                )
        new_points = []
        new_features = []
        new_times = []
        latest_points = self.points_world.new_empty((0, 3))
        latest_features = self.point_features.new_empty(
            (0, self.feature_channels)
        )
        latest_timestamp = -1
        latest_sweep = None
        latest_pose = None
        latest_motion = None
        self.last_free_space_cleared = 0
        self._update_index += 1
        update_index = self._update_index
        for local_idx in range(block.shape[0]):
            global_idx = global_start + local_idx
            if input_has_prepend and global_idx == 0:
                continue
            pose = poses[local_idx]
            motion_i = motion[local_idx] if motion is not None else None
            points, features = self._sweep_to_world_with_features(
                block[local_idx],
                (
                    None
                    if feature_block is None
                    else feature_block[:, local_idx]
                ),
                pose,
                motion_i,
            )
            if global_idx >= latest_timestamp:
                latest_points = points
                latest_features = features
                latest_timestamp = global_idx
                latest_sweep = block[local_idx]
                latest_pose = pose
                latest_motion = motion_i
            if points.shape[0] > 0:
                new_points.append(points)
                new_features.append(features)
                new_times.append(
                    torch.full(
                        (points.shape[0],),
                        global_idx,
                        device=points.device,
                        dtype=torch.long,
                    )
                )
        self.latest_points_world = latest_points
        self.latest_point_features = latest_features
        self.latest_timestamp = latest_timestamp
        if latest_sweep is not None and latest_pose is not None:
            self.last_free_space_cleared = self._apply_free_space_clearing(
                latest_sweep,
                latest_pose,
                pose_sensor_start_end=latest_motion,
            )
        if not new_points:
            if self.voxel_policy == "persistent_static":
                self._compact_global(current_update=update_index)
            return
        self.points_world = torch.cat(
            [self.points_world, *new_points], dim=0
        )
        self.point_features = torch.cat(
            [self.point_features, *new_features], dim=0
        )
        self.timestamps = torch.cat(
            [self.timestamps, *new_times], dim=0
        )
        if self.voxel_policy == "persistent_static":
            if self.observation_counts.shape[0] != (
                self.points_world.shape[0] - sum(x.shape[0] for x in new_points)
            ):
                raise RuntimeError("persistent-static observation state is stale")
            num_new = sum(x.shape[0] for x in new_points)
            self.observation_counts = torch.cat(
                [
                    self.observation_counts,
                    torch.ones(
                        num_new,
                        device=self.points_world.device,
                        dtype=torch.long,
                    ),
                ]
            )
            self.last_seen_updates = torch.cat(
                [
                    self.last_seen_updates,
                    torch.full(
                        (num_new,),
                        update_index,
                        device=self.points_world.device,
                        dtype=torch.long,
                    ),
                ]
            )
            self.free_miss_counts = torch.cat(
                [
                    self.free_miss_counts,
                    torch.zeros(
                        num_new,
                        device=self.points_world.device,
                        dtype=torch.long,
                    ),
                ]
            )
            self._compact_global(current_update=update_index)
        else:
            self._compact_global()

    def prune_outside_radius(
        self,
        target_lidar_to_world: torch.Tensor,
        radius_m: float,
    ) -> None:
        """Discard map state outside a bounded radius of the target poses."""
        if radius_m <= 0.0:
            raise ValueError(f"radius_m must be positive, got {radius_m}")
        poses = target_lidar_to_world.to(
            device=self.points_world.device, dtype=torch.float32
        )
        if poses.ndim != 3 or poses.shape[1:] != (4, 4):
            raise ValueError(
                f"target poses must be (T,4,4), got {tuple(poses.shape)}"
            )
        if self.points_world.shape[0] == 0:
            return
        keep = torch.zeros(
            self.points_world.shape[0],
            device=self.points_world.device,
            dtype=torch.bool,
        )
        radius_sq = float(radius_m) ** 2
        for center in poses[:, :2, 3]:
            delta = self.points_world[:, :2] - center.unsqueeze(0)
            keep |= delta.square().sum(dim=1) <= radius_sq
        self.points_world = self.points_world[keep]
        self.point_features = self.point_features[keep]
        self.timestamps = self.timestamps[keep]
        if self.observation_counts.shape[0] == keep.shape[0]:
            self.observation_counts = self.observation_counts[keep]
            self.last_seen_updates = self.last_seen_updates[keep]
            self.free_miss_counts = self.free_miss_counts[keep]
        self._rebuild_cell_index()

    @staticmethod
    def _pack_xy_cells(cells: torch.Tensor) -> torch.Tensor:
        """Pack signed int32 XY cells losslessly into one signed int64 key."""
        cells64 = cells.to(torch.int64)
        x_bits = cells64[..., 0] & 0xFFFFFFFF
        y_bits = cells64[..., 1] & 0xFFFFFFFF
        return (x_bits << 32) | y_bits

    def _rebuild_cell_index(self) -> None:
        if self.points_world.shape[0] == 0:
            self.cell_keys = self.cell_keys.new_empty((0,))
            return
        xy_cells = torch.floor(
            self.points_world[:, :2] / self.spatial_cell_size_m
        ).to(torch.int64)
        cell_keys = self._pack_xy_cells(xy_cells)
        order = torch.argsort(cell_keys, stable=True)
        self.points_world = self.points_world[order]
        self.point_features = self.point_features[order]
        self.timestamps = self.timestamps[order]
        if self.voxel_policy == "persistent_static":
            self.observation_counts = self.observation_counts[order]
            self.last_seen_updates = self.last_seen_updates[order]
            self.free_miss_counts = self.free_miss_counts[order]
        self.cell_keys = cell_keys[order].contiguous()

    def _compact_global(self, current_update: int | None = None) -> None:
        """Keep the newest point per global voxel and rebuild the cell index."""
        if self.points_world.shape[0] == 0:
            self.cell_keys = self.cell_keys.new_empty((0,))
            self.point_features = self.point_features.new_empty(
                (0, self.feature_channels)
            )
            self.observation_counts = self.observation_counts.new_empty((0,))
            self.last_seen_updates = self.last_seen_updates.new_empty((0,))
            self.free_miss_counts = self.free_miss_counts.new_empty((0,))
            return

        voxels = torch.floor(
            self.points_world / self.voxel_size_m
        ).to(torch.int64)
        unique_voxels, inverse = torch.unique(
            voxels, dim=0, sorted=True, return_inverse=True
        )
        num_voxels = unique_voxels.shape[0]
        latest_time = torch.full(
            (num_voxels,),
            torch.iinfo(torch.int64).min,
            device=inverse.device,
            dtype=torch.int64,
        )
        latest_time.scatter_reduce_(
            0, inverse, self.timestamps, reduce="amax", include_self=True
        )
        point_indices = torch.arange(
            self.points_world.shape[0],
            device=inverse.device,
            dtype=torch.int64,
        )
        candidates = torch.where(
            self.timestamps == latest_time[inverse],
            point_indices,
            point_indices.new_full((), -1),
        )
        selected = torch.full(
            (num_voxels,), -1, device=inverse.device, dtype=torch.int64
        )
        selected.scatter_reduce_(
            0, inverse, candidates, reduce="amax", include_self=True
        )
        if self.voxel_policy == "persistent_static":
            if (
                self.observation_counts.shape[0] != self.points_world.shape[0]
                or self.last_seen_updates.shape[0] != self.points_world.shape[0]
                or self.free_miss_counts.shape[0]
                != self.points_world.shape[0]
            ):
                raise RuntimeError(
                    "persistent-static point metadata must align before compaction"
                )
            selected_free_misses = self.free_miss_counts[selected]
            if current_update is None:
                merged_counts = torch.zeros(
                    num_voxels,
                    device=inverse.device,
                    dtype=torch.long,
                )
                merged_counts.scatter_reduce_(
                    0,
                    inverse,
                    self.observation_counts,
                    reduce="amax",
                    include_self=True,
                )
            else:
                is_current = self.last_seen_updates == current_update
                old_present = torch.zeros(
                    num_voxels,
                    device=inverse.device,
                    dtype=torch.long,
                )
                old_present.scatter_reduce_(
                    0,
                    inverse,
                    (~is_current).to(torch.long),
                    reduce="amax",
                    include_self=True,
                )
                current_observations = torch.zeros_like(old_present)
                if self.observation_unit == "frame":
                    current_pairs = torch.stack(
                        [inverse[is_current], self.timestamps[is_current]],
                        dim=1,
                    )
                    if current_pairs.shape[0] > 0:
                        unique_current_pairs = torch.unique(
                            current_pairs, dim=0
                        )
                        current_observations.scatter_add_(
                            0,
                            unique_current_pairs[:, 0],
                            torch.ones(
                                unique_current_pairs.shape[0],
                                device=inverse.device,
                                dtype=torch.long,
                            ),
                        )
                else:
                    current_observations.scatter_reduce_(
                        0,
                        inverse,
                        is_current.to(torch.long),
                        reduce="amax",
                        include_self=True,
                    )
                old_counts = torch.zeros_like(old_present)
                old_counts.scatter_reduce_(
                    0,
                    inverse,
                    torch.where(
                        is_current,
                        self.observation_counts.new_zeros(()),
                        self.observation_counts,
                    ),
                    reduce="amax",
                    include_self=True,
                )
                merged_counts = torch.where(
                    old_present.bool(),
                    old_counts + current_observations,
                    current_observations,
                )
            merged_last_seen = torch.full(
                (num_voxels,),
                torch.iinfo(torch.int64).min,
                device=inverse.device,
                dtype=torch.long,
            )
            merged_last_seen.scatter_reduce_(
                0,
                inverse,
                self.last_seen_updates,
                reduce="amax",
                include_self=True,
            )
            keep = merged_counts >= self.static_min_observations
            if current_update is not None:
                oldest_candidate_update = (
                    current_update - self.candidate_ttl_updates + 1
                )
                keep |= merged_last_seen >= oldest_candidate_update
            selected = selected[keep]
            self.observation_counts = merged_counts[keep]
            self.last_seen_updates = merged_last_seen[keep]
            self.free_miss_counts = selected_free_misses[keep]
        self.points_world = self.points_world[selected]
        self.point_features = self.point_features[selected]
        self.timestamps = self.timestamps[selected]
        self._rebuild_cell_index()

    def _points_within_target_radius(
        self, points: torch.Tensor, poses: torch.Tensor
    ) -> torch.Tensor:
        if points.shape[0] == 0:
            return points
        keep = torch.zeros(
            points.shape[0], device=points.device, dtype=torch.bool
        )
        radius_sq = self.radius_m * self.radius_m
        for center in poses[:, :2, 3]:
            delta_xy = points[:, :2] - center.unsqueeze(0)
            keep |= delta_xy.square().sum(dim=1) <= radius_sq
        return points[keep]

    def _local_point_indices_for_targets(
        self,
        target_poses: torch.Tensor,
        *,
        confirmed_only: bool,
    ) -> torch.Tensor:
        """Query indexed cells and return exact-radius point indices."""
        if self.points_world.shape[0] == 0:
            return torch.empty(
                (0,), device=self.points_world.device, dtype=torch.long
            )
        if self.cell_keys.shape[0] != self.points_world.shape[0]:
            raise RuntimeError(
                "spatial cell index is stale; points must be appended through "
                "append_block"
            )
        poses = target_poses.to(
            device=self.points_world.device, dtype=torch.float32
        )
        center_cells = torch.floor(
            poses[:, :2, 3] / self.spatial_cell_size_m
        ).to(torch.int64)
        query_cells = (
            center_cells[:, None, :] + self.query_cell_offsets[None, :, :]
        ).reshape(-1, 2)
        query_keys = torch.unique(
            self._pack_xy_cells(query_cells), sorted=True
        )
        left = torch.searchsorted(self.cell_keys, query_keys, right=False)
        right = torch.searchsorted(self.cell_keys, query_keys, right=True)
        counts = right - left
        max_count = int(counts.max().item())
        if max_count == 0:
            return torch.empty(
                (0,), device=self.points_world.device, dtype=torch.long
            )
        offsets = torch.arange(
            max_count, device=self.points_world.device, dtype=torch.int64
        )
        candidate_indices = left[:, None] + offsets[None, :]
        valid_indices = offsets[None, :] < counts[:, None]
        candidate_indices = candidate_indices[valid_indices]
        if confirmed_only and self.voxel_policy == "persistent_static":
            candidate_indices = candidate_indices[
                self.observation_counts[candidate_indices]
                >= self.static_min_observations
            ]
        candidates = self.points_world[candidate_indices]
        keep = torch.zeros(
            candidates.shape[0], device=candidates.device, dtype=torch.bool
        )
        radius_sq = self.radius_m * self.radius_m
        for center in poses[:, :2, 3]:
            delta_xy = candidates[:, :2] - center.unsqueeze(0)
            keep |= delta_xy.square().sum(dim=1) <= radius_sq
        return candidate_indices[keep]

    def _local_points_for_targets(
        self, target_poses: torch.Tensor
    ) -> torch.Tensor:
        """Query nearby indexed cells, then apply the exact XY-radius mask."""
        candidate_indices = self._local_point_indices_for_targets(
            target_poses, confirmed_only=True
        )
        return self.points_world[candidate_indices]

    @staticmethod
    def _temporal_confidence_from_counts(
        observation_counts: torch.Tensor,
    ) -> torch.Tensor:
        """Map distinct selected-unit observations to geometry confidence."""
        counts = observation_counts.to(torch.float32)
        return torch.where(
            counts >= 4,
            torch.ones_like(counts),
            torch.where(
                counts >= 3,
                counts.new_full((), 0.8),
                torch.where(
                    counts >= 2,
                    counts.new_full((), 0.6),
                    counts.new_full((), 0.2),
                ),
            ),
        )

    def _project_one(
        self,
        target_pose: torch.Tensor,
        points_world: torch.Tensor,
        point_confidence: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        metric = self.points_world.new_full(
            (self.height * self.width,), torch.inf
        )
        confidence = self.points_world.new_zeros(
            (self.height * self.width,)
        )
        if points_world.shape[0] == 0:
            hit = torch.zeros_like(metric, dtype=torch.bool)
            return (
                metric.new_zeros((self.height, self.width)),
                hit.view(self.height, self.width),
                confidence.view(self.height, self.width),
            )
        if (
            point_confidence is not None
            and point_confidence.shape != (points_world.shape[0],)
        ):
            raise ValueError(
                "point confidence must align with points: "
                f"got {tuple(point_confidence.shape)} for "
                f"{points_world.shape[0]} points"
            )

        pose = target_pose.to(
            device=self.points_world.device, dtype=torch.float32
        )
        ps = (points_world - pose[:3, 3].unsqueeze(0)) @ pose[
            :3, :3
        ]
        ranges = torch.linalg.vector_norm(ps, dim=1)
        keep = (ranges >= self.min_range) & (ranges <= self.max_range)
        ps = ps[keep]
        ranges = ranges[keep]
        if point_confidence is not None:
            point_confidence = point_confidence[keep]
        if ps.shape[0] == 0:
            hit = torch.zeros_like(metric, dtype=torch.bool)
            return (
                metric.new_zeros((self.height, self.width)),
                hit.view(self.height, self.width),
                confidence.view(self.height, self.width),
            )

        rows, cols = self._nearest_beam_indices(ps, ranges)
        flat_indices = rows * self.width + cols
        metric.scatter_reduce_(
            0, flat_indices, ranges, reduce="amin", include_self=True
        )
        if point_confidence is not None:
            nearest = ranges == metric[flat_indices]
            confidence.scatter_reduce_(
                0,
                flat_indices[nearest],
                point_confidence[nearest],
                reduce="amax",
                include_self=True,
            )
        hit = torch.isfinite(metric)
        metric = torch.where(hit, metric, metric.new_zeros(()))
        return (
            metric.view(self.height, self.width),
            hit.view(self.height, self.width),
            confidence.view(self.height, self.width),
        )

    def _gather_nearest_features(
        self,
        flat_indices: torch.Tensor,
        ranges: torch.Tensor,
        point_features: torch.Tensor,
        metric: torch.Tensor,
    ) -> torch.Tensor:
        """Gather attributes from the same point that wins the range z-buffer."""
        features = torch.as_tensor(
            point_features,
            device=self.points_world.device,
            dtype=torch.float32,
        )
        if features.shape != (ranges.shape[0], self.feature_channels):
            raise ValueError(
                "point features must align with projected ranges, got "
                f"{tuple(features.shape)} for {ranges.shape[0]} ranges"
            )
        output = features.new_full(
            (self.height * self.width, self.feature_channels),
            float(self.min_value),
        )
        if ranges.shape[0] == 0 or self.feature_channels == 0:
            return output.T.reshape(
                self.feature_channels, self.height, self.width
            )
        metric_flat = metric.reshape(-1)
        nearest = ranges == metric_flat[flat_indices]
        source_indices = torch.arange(
            ranges.shape[0], device=ranges.device, dtype=torch.long
        )
        sentinel = ranges.shape[0]
        winners = torch.full(
            (self.height * self.width,),
            sentinel,
            device=ranges.device,
            dtype=torch.long,
        )
        winners.scatter_reduce_(
            0,
            flat_indices[nearest],
            source_indices[nearest],
            reduce="amin",
            include_self=True,
        )
        hit = winners < sentinel
        output[hit] = features[winners[hit]]
        return output.T.reshape(
            self.feature_channels, self.height, self.width
        )

    def _project_one_with_features(
        self,
        target_pose: torch.Tensor,
        points_world: torch.Tensor,
        point_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        metric, hit, _ = self._project_one(target_pose, points_world)
        pose = target_pose.to(
            device=self.points_world.device, dtype=torch.float32
        )
        points = torch.as_tensor(
            points_world, device=self.points_world.device, dtype=torch.float32
        )
        features = torch.as_tensor(
            point_features,
            device=self.points_world.device,
            dtype=torch.float32,
        )
        if points.ndim != 2 or points.shape[1:] != (3,):
            raise ValueError(
                f"points_world must be (N,3), got {tuple(points.shape)}"
            )
        if features.shape != (points.shape[0], self.feature_channels):
            raise ValueError(
                "point features must align with points_world, got "
                f"{tuple(features.shape)} and {tuple(points.shape)}"
            )
        ps = (points - pose[:3, 3].unsqueeze(0)) @ pose[:3, :3]
        ranges = torch.linalg.vector_norm(ps, dim=1)
        keep = (ranges >= self.min_range) & (ranges <= self.max_range)
        ps = ps[keep]
        ranges = ranges[keep]
        features = features[keep]
        if ranges.shape[0] == 0:
            feature_image = features.new_full(
                (self.feature_channels, self.height, self.width),
                float(self.min_value),
            )
        else:
            rows, cols = self._nearest_beam_indices(ps, ranges)
            feature_image = self._gather_nearest_features(
                rows * self.width + cols,
                ranges,
                features,
                metric,
            )
        return metric, hit, feature_image

    def _points_world_to_rolling_sensor(
        self,
        target_pose: torch.Tensor,
        points_world: torch.Tensor,
        pose_sensor_start_end: torch.Tensor,
        *,
        projection_iterations: int = 3,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Project world points into their time-dependent target sensor frame."""
        if projection_iterations <= 0:
            raise ValueError("projection_iterations must be positive")
        pose = target_pose.to(
            device=self.points_world.device, dtype=torch.float32
        )
        relative = torch.as_tensor(
            pose_sensor_start_end,
            device=self.points_world.device,
            dtype=torch.float32,
        )
        if relative.shape != (4, 4):
            raise ValueError(
                "pose_sensor_start_end must be (4,4), got "
                f"{tuple(relative.shape)}"
            )
        points_start = (
            points_world - pose[:3, 3].unsqueeze(0)
        ) @ pose[:3, :3]
        if points_start.shape[0] == 0:
            empty_long = torch.empty(
                (0,), device=points_start.device, dtype=torch.long
            )
            return (
                points_start,
                points_start.new_empty((0,)),
                empty_long,
                empty_long,
            )

        start_ranges = torch.linalg.vector_norm(points_start, dim=1)
        _, columns = self._nearest_beam_indices(
            points_start, start_ranges.clamp_min(1.0e-8)
        )
        for _ in range(projection_iterations):
            fractions = columns.to(points_start.dtype) / float(
                max(self.width - 1, 1)
            )
            points_sensor = _fractional_transform_points_torch(
                points_start, relative, fractions
            )
            ranges = torch.linalg.vector_norm(points_sensor, dim=1)
            _, columns = self._nearest_beam_indices(
                points_sensor, ranges.clamp_min(1.0e-8)
            )

        fractions = columns.to(points_start.dtype) / float(
            max(self.width - 1, 1)
        )
        points_sensor = _fractional_transform_points_torch(
            points_start, relative, fractions
        )
        ranges = torch.linalg.vector_norm(points_sensor, dim=1)
        rows, columns = self._nearest_beam_indices(
            points_sensor, ranges.clamp_min(1.0e-8)
        )
        return points_sensor, ranges, rows, columns

    def _project_one_rolling(
        self,
        target_pose: torch.Tensor,
        points_world: torch.Tensor,
        pose_sensor_start_end: torch.Tensor,
        point_confidence: torch.Tensor | None = None,
        *,
        projection_iterations: int = 3,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Rasterize world points using the target sweep's per-column pose."""
        metric = self.points_world.new_full(
            (self.height * self.width,), torch.inf
        )
        confidence = self.points_world.new_zeros(
            (self.height * self.width,)
        )
        if points_world.shape[0] == 0:
            hit = torch.zeros_like(metric, dtype=torch.bool)
            return (
                metric.new_zeros((self.height, self.width)),
                hit.view(self.height, self.width),
                confidence.view(self.height, self.width),
            )
        if (
            point_confidence is not None
            and point_confidence.shape != (points_world.shape[0],)
        ):
            raise ValueError(
                "point confidence must align with points: "
                f"got {tuple(point_confidence.shape)} for "
                f"{points_world.shape[0]} points"
            )

        _, ranges, rows, cols = self._points_world_to_rolling_sensor(
            target_pose,
            points_world,
            pose_sensor_start_end,
            projection_iterations=projection_iterations,
        )
        keep = (ranges >= self.min_range) & (ranges <= self.max_range)
        ranges = ranges[keep]
        rows = rows[keep]
        cols = cols[keep]
        if point_confidence is not None:
            point_confidence = point_confidence[keep]
        if ranges.shape[0] == 0:
            hit = torch.zeros_like(metric, dtype=torch.bool)
            return (
                metric.new_zeros((self.height, self.width)),
                hit.view(self.height, self.width),
                confidence.view(self.height, self.width),
            )

        flat_indices = rows * self.width + cols
        metric.scatter_reduce_(
            0, flat_indices, ranges, reduce="amin", include_self=True
        )
        if point_confidence is not None:
            nearest = ranges == metric[flat_indices]
            confidence.scatter_reduce_(
                0,
                flat_indices[nearest],
                point_confidence[nearest],
                reduce="amax",
                include_self=True,
            )
        hit = torch.isfinite(metric)
        metric = torch.where(hit, metric, metric.new_zeros(()))
        return (
            metric.view(self.height, self.width),
            hit.view(self.height, self.width),
            confidence.view(self.height, self.width),
        )

    def _project_one_rolling_with_features(
        self,
        target_pose: torch.Tensor,
        points_world: torch.Tensor,
        point_features: torch.Tensor,
        pose_sensor_start_end: torch.Tensor,
        *,
        projection_iterations: int = 3,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        metric, hit, _ = self._project_one_rolling(
            target_pose,
            points_world,
            pose_sensor_start_end,
            projection_iterations=projection_iterations,
        )
        _, ranges, rows, cols = self._points_world_to_rolling_sensor(
            target_pose,
            points_world,
            pose_sensor_start_end,
            projection_iterations=projection_iterations,
        )
        features = torch.as_tensor(
            point_features,
            device=self.points_world.device,
            dtype=torch.float32,
        )
        if features.shape != (ranges.shape[0], self.feature_channels):
            raise ValueError(
                "point features must align with rolling-shutter points, got "
                f"{tuple(features.shape)} for {ranges.shape[0]} points"
            )
        keep = (ranges >= self.min_range) & (ranges <= self.max_range)
        ranges = ranges[keep]
        rows = rows[keep]
        cols = cols[keep]
        features = features[keep]
        feature_image = self._gather_nearest_features(
            rows * self.width + cols,
            ranges,
            features,
            metric,
        )
        return metric, hit, feature_image

    def _project_moving_object_one(
        self,
        target_pose: torch.Tensor,
        local_points: torch.Tensor,
        actor_start_pose: torch.Tensor,
        actor_end_pose: torch.Tensor,
        pose_sensor_start_end: torch.Tensor | None,
        *,
        projection_iterations: int = 3,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Rasterize one canonical actor with ray-time actor and ego poses."""
        if projection_iterations <= 0:
            raise ValueError("projection_iterations must be positive")
        metric = self.points_world.new_full(
            (self.height * self.width,), torch.inf
        )
        points = torch.as_tensor(
            local_points, device=self.points_world.device, dtype=torch.float32
        )
        if points.ndim != 2 or points.shape[1:] != (3,):
            raise ValueError(
                f"local actor points must be (N,3), got {tuple(points.shape)}"
            )
        if points.shape[0] == 0:
            return (
                metric.new_zeros((self.height, self.width)),
                torch.zeros(
                    (self.height, self.width),
                    device=metric.device,
                    dtype=torch.bool,
                ),
            )

        pose = target_pose.to(device=metric.device, dtype=torch.float32)
        start_actor = actor_start_pose.to(
            device=metric.device, dtype=torch.float32
        )
        end_actor = actor_end_pose.to(
            device=metric.device, dtype=torch.float32
        )
        relative = (
            None
            if pose_sensor_start_end is None
            else torch.as_tensor(
                pose_sensor_start_end,
                device=metric.device,
                dtype=torch.float32,
            )
        )
        if relative is not None and relative.shape != (4, 4):
            raise ValueError(
                "pose_sensor_start_end must be (4,4), got "
                f"{tuple(relative.shape)}"
            )

        world_end = (
            points @ end_actor[:3, :3].T + end_actor[:3, 3].unsqueeze(0)
        )
        points_start = (
            world_end - pose[:3, 3].unsqueeze(0)
        ) @ pose[:3, :3]
        ranges = torch.linalg.vector_norm(points_start, dim=1)
        _, columns = self._nearest_beam_indices(
            points_start, ranges.clamp_min(1.0e-8)
        )

        for _ in range(projection_iterations):
            fractions = columns.to(points.dtype) / float(
                max(self.width - 1, 1)
            )
            actor_rotations, actor_translations = (
                _interpolate_pose_components_torch(
                    start_actor, end_actor, fractions
                )
            )
            points_world = (
                torch.einsum(
                    "ni,nji->nj", points, actor_rotations
                )
                + actor_translations
            )
            points_start = (
                points_world - pose[:3, 3].unsqueeze(0)
            ) @ pose[:3, :3]
            points_sensor = (
                points_start
                if relative is None
                else _fractional_transform_points_torch(
                    points_start, relative, fractions
                )
            )
            ranges = torch.linalg.vector_norm(points_sensor, dim=1)
            _, columns = self._nearest_beam_indices(
                points_sensor, ranges.clamp_min(1.0e-8)
            )

        fractions = columns.to(points.dtype) / float(
            max(self.width - 1, 1)
        )
        actor_rotations, actor_translations = (
            _interpolate_pose_components_torch(
                start_actor, end_actor, fractions
            )
        )
        points_world = (
            torch.einsum("ni,nji->nj", points, actor_rotations)
            + actor_translations
        )
        points_start = (
            points_world - pose[:3, 3].unsqueeze(0)
        ) @ pose[:3, :3]
        points_sensor = (
            points_start
            if relative is None
            else _fractional_transform_points_torch(
                points_start, relative, fractions
            )
        )
        ranges = torch.linalg.vector_norm(points_sensor, dim=1)
        rows, columns = self._nearest_beam_indices(
            points_sensor, ranges.clamp_min(1.0e-8)
        )
        keep = (ranges >= self.min_range) & (ranges <= self.max_range)
        ranges = ranges[keep]
        rows = rows[keep]
        columns = columns[keep]
        if ranges.shape[0] == 0:
            return (
                metric.new_zeros((self.height, self.width)),
                torch.zeros(
                    (self.height, self.width),
                    device=metric.device,
                    dtype=torch.bool,
                ),
            )
        flat_indices = rows * self.width + columns
        metric.scatter_reduce_(
            0, flat_indices, ranges, reduce="amin", include_self=True
        )
        hit = torch.isfinite(metric)
        metric = torch.where(hit, metric, metric.new_zeros(()))
        return (
            metric.view(self.height, self.width),
            hit.view(self.height, self.width),
        )

    def render(
        self,
        target_lidar_to_world: torch.Tensor,
        *,
        include_latest: bool | None = None,
        target_pose_sensor_start_end: torch.Tensor | None = None,
        rolling_shutter_iterations: int = 3,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Render a temporary local query of the persistent global map."""
        poses = target_lidar_to_world.to(
            device=self.ray_dirs.device, dtype=torch.float32
        )
        if poses.ndim != 3 or poses.shape[1:] != (4, 4):
            raise ValueError(
                f"target poses must be (T,4,4), got {tuple(poses.shape)}"
            )
        target_motion = None
        if target_pose_sensor_start_end is not None:
            target_motion = torch.as_tensor(
                target_pose_sensor_start_end,
                device=self.ray_dirs.device,
                dtype=torch.float32,
            )
            if target_motion.shape != (poses.shape[0], 4, 4):
                raise ValueError(
                    "target_pose_sensor_start_end must be (T,4,4), got "
                    f"{tuple(target_motion.shape)} for T={poses.shape[0]}"
                )
        local_points = self._local_points_for_targets(poses)
        render_latest = (
            self.include_latest
            if include_latest is None
            else bool(include_latest)
        )
        if (
            self.voxel_policy == "persistent_static"
            and render_latest
            and self.latest_points_world.shape[0] > 0
        ):
            latest_points = self._points_within_target_radius(
                self.latest_points_world, poses
            )
            local_points = torch.cat([local_points, latest_points], dim=0)
        metric_frames = []
        hit_frames = []
        for frame_idx, pose in enumerate(poses):
            if target_motion is None:
                metric, hit, _ = self._project_one(pose, local_points)
            else:
                metric, hit, _ = self._project_one_rolling(
                    pose,
                    local_points,
                    target_motion[frame_idx],
                    projection_iterations=rolling_shutter_iterations,
                )
            metric_frames.append(metric)
            hit_frames.append(hit)
        metric_video = torch.stack(metric_frames, dim=0)
        hit_video = torch.stack(hit_frames, dim=0)
        normalized = _normalize_metric_depth_torch(
            metric_video,
            self.min_range,
            self.max_range,
            self.min_value,
            self.inverse_depth,
        )
        return normalized.unsqueeze(0), hit_video

    def render_with_features(
        self,
        target_lidar_to_world: torch.Tensor,
        *,
        include_latest: bool | None = None,
        target_pose_sensor_start_end: torch.Tensor | None = None,
        rolling_shutter_iterations: int = 3,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Render range and aligned attributes from one shared z-buffer."""
        if self.feature_channels == 0:
            normalized, hit = self.render(
                target_lidar_to_world,
                include_latest=include_latest,
                target_pose_sensor_start_end=target_pose_sensor_start_end,
                rolling_shutter_iterations=rolling_shutter_iterations,
            )
            empty = normalized.new_empty(
                (0, normalized.shape[1], self.height, self.width)
            )
            return normalized, empty, hit
        poses = torch.as_tensor(
            target_lidar_to_world,
            device=self.ray_dirs.device,
            dtype=torch.float32,
        )
        if poses.ndim != 3 or poses.shape[1:] != (4, 4):
            raise ValueError(
                f"target poses must be (T,4,4), got {tuple(poses.shape)}"
            )
        target_motion = None
        if target_pose_sensor_start_end is not None:
            target_motion = torch.as_tensor(
                target_pose_sensor_start_end,
                device=self.ray_dirs.device,
                dtype=torch.float32,
            )
            if target_motion.shape != (poses.shape[0], 4, 4):
                raise ValueError(
                    "target_pose_sensor_start_end must align with target poses"
                )
        candidate_indices = self._local_point_indices_for_targets(
            poses, confirmed_only=True
        )
        points = self.points_world[candidate_indices]
        features = self.point_features[candidate_indices]
        render_latest = (
            self.include_latest
            if include_latest is None
            else bool(include_latest)
        )
        if (
            self.voxel_policy == "persistent_static"
            and render_latest
            and self.latest_points_world.shape[0] > 0
        ):
            latest_keep = torch.zeros(
                self.latest_points_world.shape[0],
                device=self.points_world.device,
                dtype=torch.bool,
            )
            radius_sq = self.radius_m * self.radius_m
            for center in poses[:, :2, 3]:
                delta = self.latest_points_world[:, :2] - center.unsqueeze(0)
                latest_keep |= delta.square().sum(dim=1) <= radius_sq
            points = torch.cat(
                [points, self.latest_points_world[latest_keep]], dim=0
            )
            features = torch.cat(
                [features, self.latest_point_features[latest_keep]], dim=0
            )
        metric_frames = []
        feature_frames = []
        hit_frames = []
        for frame_idx, pose in enumerate(poses):
            if target_motion is None:
                metric, hit, feature_image = self._project_one_with_features(
                    pose, points, features
                )
            else:
                metric, hit, feature_image = (
                    self._project_one_rolling_with_features(
                        pose,
                        points,
                        features,
                        target_motion[frame_idx],
                        projection_iterations=rolling_shutter_iterations,
                    )
                )
            metric_frames.append(metric)
            feature_frames.append(feature_image)
            hit_frames.append(hit)
        normalized = _normalize_metric_depth_torch(
            torch.stack(metric_frames, dim=0),
            self.min_range,
            self.max_range,
            self.min_value,
            self.inverse_depth,
        ).unsqueeze(0)
        feature_video = torch.stack(feature_frames, dim=1)
        return normalized, feature_video, torch.stack(hit_frames, dim=0)

    def render_points_world(
        self,
        points_world: torch.Tensor,
        target_lidar_to_world: torch.Tensor,
        *,
        target_pose_sensor_start_end: torch.Tensor | None = None,
        rolling_shutter_iterations: int = 3,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Render an explicit world-space cloud without mutating the map."""
        points = torch.as_tensor(
            points_world, device=self.ray_dirs.device, dtype=torch.float32
        )
        poses = torch.as_tensor(
            target_lidar_to_world,
            device=self.ray_dirs.device,
            dtype=torch.float32,
        )
        if points.ndim != 2 or points.shape[1:] != (3,):
            raise ValueError(
                f"points_world must be (N,3), got {tuple(points.shape)}"
            )
        if poses.ndim != 3 or poses.shape[1:] != (4, 4):
            raise ValueError(
                f"target poses must be (T,4,4), got {tuple(poses.shape)}"
            )
        target_motion = None
        if target_pose_sensor_start_end is not None:
            target_motion = torch.as_tensor(
                target_pose_sensor_start_end,
                device=self.ray_dirs.device,
                dtype=torch.float32,
            )
            if target_motion.shape != (poses.shape[0], 4, 4):
                raise ValueError(
                    "target_pose_sensor_start_end must be (T,4,4), got "
                    f"{tuple(target_motion.shape)} for T={poses.shape[0]}"
                )
        metric_frames = []
        hit_frames = []
        for frame_idx, pose in enumerate(poses):
            if target_motion is None:
                metric, hit, _ = self._project_one(pose, points)
            else:
                metric, hit, _ = self._project_one_rolling(
                    pose,
                    points,
                    target_motion[frame_idx],
                    projection_iterations=rolling_shutter_iterations,
                )
            metric_frames.append(metric)
            hit_frames.append(hit)
        metric_video = torch.stack(metric_frames, dim=0)
        hit_video = torch.stack(hit_frames, dim=0)
        normalized = _normalize_metric_depth_torch(
            metric_video,
            self.min_range,
            self.max_range,
            self.min_value,
            self.inverse_depth,
        )
        return normalized.unsqueeze(0), hit_video

    def render_points_world_with_features(
        self,
        points_world: torch.Tensor,
        point_features: torch.Tensor,
        target_lidar_to_world: torch.Tensor,
        *,
        target_pose_sensor_start_end: torch.Tensor | None = None,
        rolling_shutter_iterations: int = 3,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Render an explicit attributed world cloud without mutating the map."""
        points = torch.as_tensor(
            points_world, device=self.ray_dirs.device, dtype=torch.float32
        )
        features = torch.as_tensor(
            point_features, device=self.ray_dirs.device, dtype=torch.float32
        )
        if points.ndim != 2 or points.shape[1:] != (3,):
            raise ValueError(
                f"points_world must be (N,3), got {tuple(points.shape)}"
            )
        if features.shape != (points.shape[0], self.feature_channels):
            raise ValueError(
                "point_features must align with points_world, got "
                f"{tuple(features.shape)} and {tuple(points.shape)}"
            )
        poses = torch.as_tensor(
            target_lidar_to_world,
            device=self.ray_dirs.device,
            dtype=torch.float32,
        )
        if poses.ndim != 3 or poses.shape[1:] != (4, 4):
            raise ValueError(
                f"target poses must be (T,4,4), got {tuple(poses.shape)}"
            )
        target_motion = None
        if target_pose_sensor_start_end is not None:
            target_motion = torch.as_tensor(
                target_pose_sensor_start_end,
                device=self.ray_dirs.device,
                dtype=torch.float32,
            )
            if target_motion.shape != (poses.shape[0], 4, 4):
                raise ValueError(
                    "target_pose_sensor_start_end must be (T,4,4), got "
                    f"{tuple(target_motion.shape)} for T={poses.shape[0]}"
                )
        metric_frames = []
        feature_frames = []
        hit_frames = []
        for frame_idx, pose in enumerate(poses):
            if target_motion is None:
                metric, hit, feature_image = self._project_one_with_features(
                    pose, points, features
                )
            else:
                metric, hit, feature_image = (
                    self._project_one_rolling_with_features(
                        pose,
                        points,
                        features,
                        target_motion[frame_idx],
                        projection_iterations=rolling_shutter_iterations,
                    )
                )
            metric_frames.append(metric)
            feature_frames.append(feature_image)
            hit_frames.append(hit)
        normalized = _normalize_metric_depth_torch(
            torch.stack(metric_frames, dim=0),
            self.min_range,
            self.max_range,
            self.min_value,
            self.inverse_depth,
        ).unsqueeze(0)
        return (
            normalized,
            torch.stack(feature_frames, dim=1),
            torch.stack(hit_frames, dim=0),
        )

    def render_soft_proposal(
        self,
        target_lidar_to_world: torch.Tensor,
        *,
        confirmed_only: bool = False,
        target_pose_sensor_start_end: torch.Tensor | None = None,
        rolling_shutter_iterations: int = 3,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Render confidence-scored candidate or confirmed map voxels."""
        poses = target_lidar_to_world.to(
            device=self.ray_dirs.device, dtype=torch.float32
        )
        if poses.ndim != 3 or poses.shape[1:] != (4, 4):
            raise ValueError(
                f"target poses must be (T,4,4), got {tuple(poses.shape)}"
            )
        target_motion = None
        if target_pose_sensor_start_end is not None:
            target_motion = torch.as_tensor(
                target_pose_sensor_start_end,
                device=self.ray_dirs.device,
                dtype=torch.float32,
            )
            if target_motion.shape != (poses.shape[0], 4, 4):
                raise ValueError(
                    "target_pose_sensor_start_end must be (T,4,4), got "
                    f"{tuple(target_motion.shape)} for T={poses.shape[0]}"
                )
        point_indices = self._local_point_indices_for_targets(
            poses, confirmed_only=confirmed_only
        )
        local_points = self.points_world[point_indices]
        if self.voxel_policy == "persistent_static":
            temporal_confidence = self._temporal_confidence_from_counts(
                self.observation_counts[point_indices]
            )
        else:
            temporal_confidence = local_points.new_ones(
                (local_points.shape[0],)
            )

        metric_frames = []
        hit_frames = []
        confidence_frames = []
        for frame_idx, pose in enumerate(poses):
            if target_motion is None:
                metric, hit, confidence = self._project_one(
                    pose, local_points, temporal_confidence
                )
            else:
                metric, hit, confidence = self._project_one_rolling(
                    pose,
                    local_points,
                    target_motion[frame_idx],
                    temporal_confidence,
                    projection_iterations=rolling_shutter_iterations,
                )
            metric_frames.append(metric)
            hit_frames.append(hit)
            confidence_frames.append(confidence)
        metric_video = torch.stack(metric_frames, dim=0)
        hit_video = torch.stack(hit_frames, dim=0)
        confidence_video = torch.stack(confidence_frames, dim=0)
        normalized = _normalize_metric_depth_torch(
            metric_video,
            self.min_range,
            self.max_range,
            self.min_value,
            self.inverse_depth,
        )
        return normalized.unsqueeze(0), hit_video, confidence_video


@torch.no_grad()
def build_lagen_sde_conditions(
    previous_video: torch.Tensor,
    lidar_to_world: torch.Tensor,
    box_object_to_world: torch.Tensor,
    box_lwh: torch.Tensor,
    box_valid: torch.Tensor,
    box_type: torch.Tensor,
    *,
    ray_dirs: Any,
    row_elevations_rad: Any,
    column_azimuths_rad: Any,
    row_azimuth_offsets_rad: Any,
    min_range: float,
    max_range: float,
    min_value: float,
    inverse_depth: bool,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Build LaGen foreground/background SDE conditions for one frame pair.

    Previous returns inside an annotated box are transformed through the
    nearest same-class box at the target frame. Remaining returns stay fixed
    in world coordinates and are re-rendered from the target ego pose.
    """
    if previous_video.ndim != 4 or previous_video.shape[:2] != (1, 1):
        raise ValueError(
            "LaGen SDE expects previous_video=(1,1,H,W), got "
            f"{tuple(previous_video.shape)}"
        )
    poses = torch.as_tensor(
        lidar_to_world, device=previous_video.device, dtype=torch.float32
    )
    object_to_world = torch.as_tensor(
        box_object_to_world,
        device=previous_video.device,
        dtype=torch.float32,
    )
    lwh = torch.as_tensor(
        box_lwh, device=previous_video.device, dtype=torch.float32
    )
    valid_boxes = torch.as_tensor(
        box_valid, device=previous_video.device, dtype=torch.bool
    )
    types = torch.as_tensor(
        box_type, device=previous_video.device, dtype=torch.long
    )
    if poses.shape != (2, 4, 4):
        raise ValueError(
            f"LaGen SDE requires two lidar poses, got {tuple(poses.shape)}"
        )
    if (
        object_to_world.ndim != 4
        or object_to_world.shape[0] != 2
        or object_to_world.shape[2:] != (4, 4)
        or lwh.shape != (*object_to_world.shape[:2], 3)
        or valid_boxes.shape != object_to_world.shape[:2]
        or types.shape != object_to_world.shape[:2]
    ):
        raise ValueError("LaGen SDE box tensors must align over two frames")

    projector = TorchSpatialPointMap(
        ray_dirs=ray_dirs,
        row_elevations_rad=row_elevations_rad,
        column_azimuths_rad=column_azimuths_rad,
        row_azimuth_offsets_rad=row_azimuth_offsets_rad,
        min_range=min_range,
        max_range=max_range,
        min_value=min_value,
        inverse_depth=inverse_depth,
        radius_m=max_range + 1.0,
        voxel_size_m=0.15,
        voxel_policy="newest",
        observation_unit="frame",
        static_min_observations=2,
        candidate_ttl_updates=1,
        include_latest=False,
        free_space_clear=False,
        device=previous_video.device,
    )
    depth, valid_returns = projector._normalized_sweep_to_metric(
        previous_video[0, 0]
    )
    flat_indices = valid_returns.reshape(-1).nonzero(
        as_tuple=False
    ).flatten()
    if flat_indices.numel() == 0:
        empty = previous_video.new_full(
            previous_video.shape, float(min_value)
        )
        zero = previous_video.new_zeros((), dtype=torch.float32)
        return empty, empty.clone(), {
            "foreground_fraction": zero,
            "matched_objects": zero,
        }
    points_sensor = (
        depth[..., None] * projector.ray_dirs
    ).reshape(-1, 3).index_select(0, flat_indices)
    points_world = (
        points_sensor @ poses[0, :3, :3].T + poses[0, :3, 3]
    )

    claimed = torch.zeros(
        points_world.shape[0],
        device=points_world.device,
        dtype=torch.bool,
    )
    foreground_world = []
    matched_objects = 0
    for source_idx in valid_boxes[0].nonzero(
        as_tuple=False
    ).flatten().tolist():
        source_pose = object_to_world[0, source_idx]
        local = (
            points_world - source_pose[:3, 3]
        ) @ source_pose[:3, :3]
        inside = (
            local.abs() <= 0.5 * lwh[0, source_idx] + 1.0e-3
        ).all(dim=1)
        inside &= ~claimed
        if not bool(inside.any()):
            continue
        # Object points that have no same-class target match should disappear
        # rather than leaking into the static background as a ghost.
        claimed |= inside
        candidates = valid_boxes[1] & (
            types[1] == types[0, source_idx]
        )
        candidate_indices = candidates.nonzero(
            as_tuple=False
        ).flatten()
        if candidate_indices.numel() == 0:
            continue
        target_centers = object_to_world[
            1, candidate_indices, :3, 3
        ]
        nearest = torch.linalg.vector_norm(
            target_centers - source_pose[:3, 3], dim=1
        ).argmin()
        target_idx = int(candidate_indices[nearest].item())
        target_pose = object_to_world[1, target_idx]
        foreground_world.append(
            local[inside] @ target_pose[:3, :3].T
            + target_pose[:3, 3]
        )
        matched_objects += 1

    background_points = points_world[~claimed]
    foreground_points = (
        torch.cat(foreground_world, dim=0)
        if foreground_world
        else points_world.new_empty((0, 3))
    )
    background, _ = projector.render_points_world(
        background_points, poses[1:2]
    )
    foreground, _ = projector.render_points_world(
        foreground_points, poses[1:2]
    )
    return foreground.to(previous_video.dtype), background.to(
        previous_video.dtype
    ), {
        "foreground_fraction": previous_video.new_tensor(
            float(claimed.sum().item())
            / float(max(points_world.shape[0], 1))
        ),
        "matched_objects": previous_video.new_tensor(
            float(matched_objects)
        ),
    }


@torch.no_grad()
def warp_frozen_observed_history_with_spatial_map(
    video: torch.Tensor,
    lidar_to_world: torch.Tensor,
    observed_mask: torch.Tensor,
    *,
    ray_dirs: Any,
    row_elevations_rad: Any,
    column_azimuths_rad: Any,
    row_azimuth_offsets_rad: Any,
    min_range: float,
    max_range: float,
    min_value: float = -1.0,
    inverse_depth: bool = False,
    radius_m: float = 75.0,
    voxel_size_m: float = 0.2,
    pose_sensor_start_end: torch.Tensor | None = None,
    target_lidar_to_world: torch.Tensor | None = None,
    target_pose_sensor_start_end: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Accumulate selected GT frames once, then render from target poses."""
    if video.ndim != 4 or video.shape[0] < 1:
        raise ValueError(
            "frozen observed history requires (C,T,H,W), got "
            f"{tuple(video.shape)}"
        )
    num_frames, height, width = map(int, video.shape[1:])
    poses = torch.as_tensor(
        lidar_to_world, device=video.device, dtype=torch.float32
    )
    if poses.shape != (num_frames, 4, 4):
        raise ValueError(
            "frozen observed history requires one pose per unique frame, got "
            f"{tuple(poses.shape)} for T={num_frames}"
        )
    target_poses = (
        poses
        if target_lidar_to_world is None
        else torch.as_tensor(
            target_lidar_to_world,
            device=video.device,
            dtype=torch.float32,
        )
    )
    if target_poses.shape != (num_frames, 4, 4):
        raise ValueError(
            "frozen observed history target poses must align one pose per "
            f"frame, got {tuple(target_poses.shape)} for T={num_frames}"
        )
    motion = None
    if pose_sensor_start_end is not None:
        motion = torch.as_tensor(
            pose_sensor_start_end,
            device=video.device,
            dtype=torch.float32,
        )
        if motion.shape != (num_frames, 4, 4):
            raise ValueError(
                "frozen observed history requires one start/end transform "
                f"per frame, got {tuple(motion.shape)} for T={num_frames}"
            )
    target_motion = (
        motion
        if target_pose_sensor_start_end is None
        else torch.as_tensor(
            target_pose_sensor_start_end,
            device=video.device,
            dtype=torch.float32,
        )
    )
    if (
        target_motion is not None
        and target_motion.shape != (num_frames, 4, 4)
    ):
        raise ValueError(
            "frozen observed history target motion must align one transform "
            f"per frame, got {tuple(target_motion.shape)} for T={num_frames}"
        )
    selected = torch.as_tensor(
        observed_mask, device=video.device, dtype=torch.bool
    )
    if selected.shape != (num_frames,):
        raise ValueError(
            f"observed_mask must be ({num_frames},), got {tuple(selected.shape)}"
        )

    spatial_map = TorchSpatialPointMap(
        ray_dirs=ray_dirs,
        row_elevations_rad=row_elevations_rad,
        column_azimuths_rad=column_azimuths_rad,
        row_azimuth_offsets_rad=row_azimuth_offsets_rad,
        min_range=min_range,
        max_range=max_range,
        min_value=min_value,
        inverse_depth=inverse_depth,
        radius_m=radius_m,
        voxel_size_m=voxel_size_m,
        voxel_policy="newest",
        observation_unit="frame",
        static_min_observations=2,
        candidate_ttl_updates=1,
        include_latest=False,
        free_space_clear=False,
        feature_channels=video.shape[0] - 1,
        device=video.device,
    )
    if (spatial_map.height, spatial_map.width) != (height, width):
        raise ValueError(
            "video and frozen spatial-map ray grid must match: "
            f"video={(height, width)}, "
            f"rays={(spatial_map.height, spatial_map.width)}"
        )
    for frame_idx in selected.nonzero(as_tuple=False).flatten().tolist():
        spatial_map.append_token_block(
            normalized_block=video[:1, frame_idx : frame_idx + 1],
            point_feature_block=video[1:, frame_idx : frame_idx + 1],
            token_lidar_to_world=poses[frame_idx : frame_idx + 1],
            token_pose_sensor_start_end=(
                motion[frame_idx : frame_idx + 1]
                if motion is not None
                else None
            ),
            global_start=int(frame_idx),
            input_has_prepend=False,
        )
    if video.shape[0] == 1:
        return spatial_map.render(
            target_poses,
            include_latest=False,
            target_pose_sensor_start_end=target_motion,
        )
    rendered_range, rendered_features, hit = spatial_map.render_with_features(
        target_poses,
        include_latest=False,
        target_pose_sensor_start_end=target_motion,
    )
    return torch.cat([rendered_range, rendered_features], dim=0), hit


@torch.no_grad()
def warp_frozen_object_history_with_spatial_map(
    video: torch.Tensor,
    lidar_to_world: torch.Tensor,
    observed_mask: torch.Tensor,
    box_object_to_world: torch.Tensor,
    box_lwh: torch.Tensor,
    box_valid: torch.Tensor,
    box_track_id: torch.Tensor,
    *,
    ray_dirs: Any,
    row_elevations_rad: Any,
    column_azimuths_rad: Any,
    row_azimuth_offsets_rad: Any,
    min_range: float,
    max_range: float,
    min_value: float = -1.0,
    inverse_depth: bool = False,
    radius_m: float = 75.0,
    voxel_size_m: float = 0.2,
    pose_sensor_start_end: torch.Tensor | None = None,
    target_lidar_to_world: torch.Tensor | None = None,
    target_pose_sensor_start_end: torch.Tensor | None = None,
    actor_box_time_mode: str = "frame",
    return_components: bool = False,
) -> (
    tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]
    | tuple[
        torch.Tensor,
        torch.Tensor,
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
    ]
):
    """Render frozen static GT plus canonical per-track dynamic clouds."""
    if video.ndim != 4 or video.shape[0] < 1:
        raise ValueError(
            "frozen object history requires (C,T,H,W), got "
            f"{tuple(video.shape)}"
        )
    num_frames, height, width = map(int, video.shape[1:])
    poses = torch.as_tensor(
        lidar_to_world, device=video.device, dtype=torch.float32
    )
    selected = torch.as_tensor(
        observed_mask, device=video.device, dtype=torch.bool
    )
    object_to_world = torch.as_tensor(
        box_object_to_world, device=video.device, dtype=torch.float32
    )
    lwh = torch.as_tensor(
        box_lwh, device=video.device, dtype=torch.float32
    )
    valid_boxes = torch.as_tensor(
        box_valid, device=video.device, dtype=torch.bool
    )
    track_ids = torch.as_tensor(
        box_track_id, device=video.device, dtype=torch.long
    )
    if poses.shape != (num_frames, 4, 4):
        raise ValueError(
            f"lidar_to_world must be ({num_frames},4,4), got {tuple(poses.shape)}"
        )
    target_poses = (
        poses
        if target_lidar_to_world is None
        else torch.as_tensor(
            target_lidar_to_world,
            device=video.device,
            dtype=torch.float32,
        )
    )
    if target_poses.shape != (num_frames, 4, 4):
        raise ValueError(
            "frozen-object target poses must align one pose per frame, got "
            f"{tuple(target_poses.shape)} for T={num_frames}"
        )
    motion = None
    if pose_sensor_start_end is not None:
        motion = torch.as_tensor(
            pose_sensor_start_end,
            device=video.device,
            dtype=torch.float32,
        )
        if motion.shape != (num_frames, 4, 4):
            raise ValueError(
                "pose_sensor_start_end must align one transform per frame, "
                f"got {tuple(motion.shape)} for T={num_frames}"
            )
    target_motion = (
        motion
        if target_pose_sensor_start_end is None
        else torch.as_tensor(
            target_pose_sensor_start_end,
            device=video.device,
            dtype=torch.float32,
        )
    )
    if (
        target_motion is not None
        and target_motion.shape != (num_frames, 4, 4)
    ):
        raise ValueError(
            "frozen-object target motion must align one transform per frame, "
            f"got {tuple(target_motion.shape)} for T={num_frames}"
        )
    if selected.shape != (num_frames,):
        raise ValueError(
            f"observed_mask must be ({num_frames},), got {tuple(selected.shape)}"
        )
    if (
        object_to_world.ndim != 4
        or object_to_world.shape[0] != num_frames
        or object_to_world.shape[2:] != (4, 4)
        or lwh.shape != (*object_to_world.shape[:2], 3)
        or valid_boxes.shape != object_to_world.shape[:2]
        or track_ids.shape != object_to_world.shape[:2]
    ):
        raise ValueError(
            "dynamic boxes must be aligned (T,M,4,4)/(T,M,3)/(T,M): "
            f"o2w={tuple(object_to_world.shape)}, lwh={tuple(lwh.shape)}, "
            f"valid={tuple(valid_boxes.shape)}, ids={tuple(track_ids.shape)}"
        )
    if bool(valid_boxes.any()) and not bool(
        ((track_ids >= 0) & valid_boxes).any()
    ):
        raise ValueError(
            "frozen-object geometry received moving boxes but no track IDs"
        )
    if actor_box_time_mode not in _ACTOR_BOX_TIME_MODES:
        raise ValueError(
            f"unsupported actor box time mode {actor_box_time_mode!r}; "
            f"expected {_ACTOR_BOX_TIME_MODES}"
        )

    projector = TorchSpatialPointMap(
        ray_dirs=ray_dirs,
        row_elevations_rad=row_elevations_rad,
        column_azimuths_rad=column_azimuths_rad,
        row_azimuth_offsets_rad=row_azimuth_offsets_rad,
        min_range=min_range,
        max_range=max_range,
        min_value=min_value,
        inverse_depth=inverse_depth,
        radius_m=radius_m,
        voxel_size_m=voxel_size_m,
        voxel_policy="newest",
        observation_unit="frame",
        static_min_observations=2,
        candidate_ttl_updates=1,
        include_latest=False,
        free_space_clear=False,
        feature_channels=video.shape[0] - 1,
        device=video.device,
    )
    if (projector.height, projector.width) != (height, width):
        raise ValueError(
            "video and ray grid must match: "
            f"video={(height, width)}, rays="
            f"{(projector.height, projector.width)}"
        )

    static_video = video.clone()
    object_local_parts: dict[int, list[torch.Tensor]] = {}
    object_feature_parts: dict[int, list[torch.Tensor]] = {}
    observed_points = 0
    removed_points = 0
    untracked_points = 0
    interpolated_boxes = 0
    fallback_boxes = 0
    for frame_idx in selected.nonzero(as_tuple=False).flatten().tolist():
        _, valid = projector._normalized_sweep_to_metric(
            video[0, frame_idx]
        )
        flat_indices = valid.reshape(-1).nonzero(
            as_tuple=False
        ).flatten()
        observed_points += int(flat_indices.numel())
        if flat_indices.numel() == 0:
            continue
        pose = poses[frame_idx]
        points_world = projector._sweep_to_world(
            video[0, frame_idx],
            pose,
            motion[frame_idx] if motion is not None else None,
        )
        feature_channels = int(video.shape[0] - 1)
        if feature_channels == 0:
            point_features = video.new_empty((flat_indices.numel(), 0))
        else:
            point_features = video[1:, frame_idx].reshape(
                feature_channels, -1
            )[:, flat_indices].T.contiguous()
        columns = torch.remainder(flat_indices, width).to(
            device=video.device, dtype=points_world.dtype
        )
        fractions = columns / float(max(width - 1, 1))
        claimed = torch.zeros(
            points_world.shape[0],
            device=video.device,
            dtype=torch.bool,
        )
        for box_idx in valid_boxes[frame_idx].nonzero(
            as_tuple=False
        ).flatten().tolist():
            if actor_box_time_mode == "frame":
                box_pose = object_to_world[frame_idx, box_idx]
                local = (
                    points_world - box_pose[:3, 3]
                ) @ box_pose[:3, :3]
            else:
                start_pose, end_pose, interpolated = (
                    _actor_pose_interval_torch(
                        object_to_world,
                        valid_boxes,
                        track_ids,
                        frame_idx,
                        box_idx,
                        actor_box_time_mode,
                    )
                )
                if interpolated:
                    interpolated_boxes += 1
                else:
                    fallback_boxes += 1
                local = _world_points_to_actor_local_torch(
                    points_world,
                    start_pose,
                    end_pose,
                    fractions,
                )
            inside = (
                local.abs() <= 0.5 * lwh[frame_idx, box_idx] + 1.0e-3
            ).all(dim=1)
            inside &= ~claimed
            if not bool(inside.any()):
                continue
            claimed |= inside
            count = int(inside.sum().item())
            track_id = int(track_ids[frame_idx, box_idx].item())
            if track_id >= 0:
                object_local_parts.setdefault(track_id, []).append(
                    local[inside]
                )
                object_feature_parts.setdefault(track_id, []).append(
                    point_features[inside]
                )
            else:
                untracked_points += count
        removed_points += int(claimed.sum().item())
        static_video[:, frame_idx].reshape(video.shape[0], -1)[
            :, flat_indices[claimed]
        ] = float(min_value)

    static_condition, static_hit = (
        warp_frozen_observed_history_with_spatial_map(
            static_video,
            poses,
            selected,
            ray_dirs=ray_dirs,
            row_elevations_rad=row_elevations_rad,
            column_azimuths_rad=column_azimuths_rad,
            row_azimuth_offsets_rad=row_azimuth_offsets_rad,
            min_range=min_range,
            max_range=max_range,
            min_value=min_value,
            inverse_depth=inverse_depth,
            radius_m=radius_m,
            voxel_size_m=voxel_size_m,
            pose_sensor_start_end=motion,
            target_lidar_to_world=target_poses,
            target_pose_sensor_start_end=target_motion,
        )
    )
    object_local = {
        track_id: torch.cat(parts, dim=0)
        for track_id, parts in object_local_parts.items()
    }
    object_features = {
        track_id: torch.cat(parts, dim=0)
        for track_id, parts in object_feature_parts.items()
    }
    dynamic_frames = []
    dynamic_hits = []
    for frame_idx in range(num_frames):
        if actor_box_time_mode != "frame":
            if video.shape[0] > 1:
                raise NotImplementedError(
                    "attributed frozen-object geometry currently requires "
                    "actor_box_time_mode='frame'"
                )
            actor_frames = []
            actor_hits = []
            emitted: set[int] = set()
            for box_idx in valid_boxes[frame_idx].nonzero(
                as_tuple=False
            ).flatten().tolist():
                track_id = int(track_ids[frame_idx, box_idx].item())
                if track_id < 0 or track_id in emitted:
                    continue
                local = object_local.get(track_id)
                if local is None or local.numel() == 0:
                    continue
                start_pose, end_pose, _ = _actor_pose_interval_torch(
                    object_to_world,
                    valid_boxes,
                    track_ids,
                    frame_idx,
                    box_idx,
                    actor_box_time_mode,
                )
                metric, hit = projector._project_moving_object_one(
                    target_poses[frame_idx],
                    local,
                    start_pose,
                    end_pose,
                    (
                        target_motion[frame_idx]
                        if target_motion is not None
                        else None
                    ),
                )
                actor_frames.append(
                    _normalize_metric_depth_torch(
                        metric,
                        min_range,
                        max_range,
                        min_value,
                        inverse_depth,
                    )
                )
                actor_hits.append(hit)
                emitted.add(track_id)
            if actor_frames:
                dynamic_frames.append(
                    compose_normalized_range_channels(
                        torch.stack(actor_frames, dim=0),
                        channel_dim=0,
                        min_value=min_value,
                        inverse_depth=inverse_depth,
                    )
                )
                dynamic_hits.append(torch.stack(actor_hits, dim=0).any(dim=0))
            else:
                dynamic_frames.append(
                    video.new_full((1, height, width), float(min_value))
                )
                dynamic_hits.append(
                    torch.zeros(
                        (height, width),
                        device=video.device,
                        dtype=torch.bool,
                    )
                )
            continue

        world_parts = []
        feature_parts = []
        emitted: set[int] = set()
        for box_idx in valid_boxes[frame_idx].nonzero(
            as_tuple=False
        ).flatten().tolist():
            track_id = int(track_ids[frame_idx, box_idx].item())
            if track_id < 0 or track_id in emitted:
                continue
            local = object_local.get(track_id)
            if local is None or local.numel() == 0:
                continue
            box_pose = object_to_world[frame_idx, box_idx]
            world_parts.append(
                local @ box_pose[:3, :3].T + box_pose[:3, 3]
            )
            feature_parts.append(object_features[track_id])
            emitted.add(track_id)
        points_world = (
            torch.cat(world_parts, dim=0)
            if world_parts
            else video.new_empty((0, 3), dtype=torch.float32)
        )
        target_motion_i = (
            target_motion[frame_idx : frame_idx + 1]
            if target_motion is not None
            else None
        )
        if video.shape[0] == 1:
            rendered, hit = projector.render_points_world(
                points_world,
                target_poses[frame_idx : frame_idx + 1],
                target_pose_sensor_start_end=target_motion_i,
            )
            dynamic_frames.append(rendered[:, 0])
        else:
            point_features = (
                torch.cat(feature_parts, dim=0)
                if feature_parts
                else video.new_empty((0, video.shape[0] - 1))
            )
            rendered, rendered_features, hit = (
                projector.render_points_world_with_features(
                    points_world,
                    point_features,
                    target_poses[frame_idx : frame_idx + 1],
                    target_pose_sensor_start_end=target_motion_i,
                )
            )
            dynamic_frames.append(
                torch.cat(
                    [rendered[:, 0], rendered_features[:, 0]], dim=0
                )
            )
        dynamic_hits.append(hit[0])
    dynamic_condition = torch.stack(dynamic_frames, dim=1).to(video.dtype)
    dynamic_hit = torch.stack(dynamic_hits, dim=0)
    if video.shape[0] == 1:
        condition = compose_normalized_range_channels(
            torch.cat([static_condition, dynamic_condition], dim=0),
            channel_dim=0,
            min_value=min_value,
            inverse_depth=inverse_depth,
        )
    else:
        composed_frames = []
        for frame_idx in range(num_frames):
            composed_range, composed_features, _ = (
                compose_normalized_range_feature_layers(
                    torch.stack(
                        [
                            static_condition[0, frame_idx],
                            dynamic_condition[0, frame_idx],
                        ],
                        dim=0,
                    ),
                    torch.stack(
                        [
                            static_condition[1:, frame_idx],
                            dynamic_condition[1:, frame_idx],
                        ],
                        dim=0,
                    ),
                    min_value=min_value,
                    inverse_depth=inverse_depth,
                )
            )
            composed_frames.append(
                torch.cat([composed_range, composed_features], dim=0)
            )
        condition = torch.stack(composed_frames, dim=1).to(video.dtype)
    diagnostics = {
        "dynamic_removed_fraction": video.new_tensor(
            float(removed_points) / float(max(observed_points, 1))
        ),
        "dynamic_untracked_fraction": video.new_tensor(
            float(untracked_points) / float(max(removed_points, 1))
        ),
        "dynamic_tracks": video.new_tensor(float(len(object_local))),
        "dynamic_object_points": video.new_tensor(
            float(sum(points.shape[0] for points in object_local.values()))
        ),
        "dynamic_actor_interpolated_boxes": video.new_tensor(
            float(interpolated_boxes)
        ),
        "dynamic_actor_fallback_boxes": video.new_tensor(
            float(fallback_boxes)
        ),
    }
    if return_components:
        static_range = static_condition[0]
        dynamic_range = dynamic_condition[0]
        if inverse_depth:
            dynamic_visible = dynamic_hit & (
                ~static_hit | (dynamic_range > static_range)
            )
        else:
            dynamic_visible = dynamic_hit & (
                ~static_hit | (dynamic_range < static_range)
            )
        components = {
            "static_condition": static_condition,
            "static_hit": static_hit,
            "dynamic_condition": dynamic_condition,
            "dynamic_hit": dynamic_hit,
            "dynamic_visible": dynamic_visible,
        }
        return condition, static_hit | dynamic_hit, diagnostics, components
    return condition, static_hit | dynamic_hit, diagnostics


@torch.no_grad()
def build_stage2_factorized_geometry(
    video: torch.Tensor,
    lidar_to_world: torch.Tensor,
    geometry_source_mask: torch.Tensor,
    geometry_mode: str,
    *,
    pose_sensor_start_end: torch.Tensor | None = None,
    geometry_source_lidar_to_world: torch.Tensor | None = None,
    geometry_source_pose_sensor_start_end: torch.Tensor | None = None,
    actor_box_time_mode: str = "frame",
    static_only_geometry: bool = False,
    box_object_to_world: torch.Tensor | None,
    box_lwh: torch.Tensor | None,
    box_valid: torch.Tensor | None,
    box_track_id: torch.Tensor | None,
    block_size: int,
    prefix_frames: int,
    input_has_prepend: bool,
    ray_dirs: Any,
    row_elevations_rad: Any,
    column_azimuths_rad: Any,
    row_azimuth_offsets_rad: Any,
    min_range: float,
    max_range: float,
    min_value: float,
    inverse_depth: bool,
    radius_m: float,
    voxel_size_m: float,
    voxel_policy: str,
    observation_unit: str,
    static_min_observations: int,
    candidate_ttl_updates: int,
    include_latest: bool,
    free_space_clear: bool,
    free_space_margin_m: float,
    free_space_misses_to_remove: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Build the exact Stage-2 geometry tensor for one factorized mode."""
    modes = ("null", "rolling_static", "frozen_raw", "frozen_object")
    if geometry_mode not in modes:
        raise ValueError(
            f"unsupported geometry mode {geometry_mode!r}; expected {modes}"
        )
    if video.ndim != 4 or video.shape[0] < 1:
        raise ValueError(
            "factorized geometry requires (C,T,H,W), got "
            f"{tuple(video.shape)}"
        )
    num_frames, height, width = map(int, video.shape[1:])
    selected = torch.as_tensor(
        geometry_source_mask, device=video.device, dtype=torch.bool
    )
    if selected.shape != (num_frames,):
        raise ValueError(
            "geometry_source_mask must be "
            f"({num_frames},), got {tuple(selected.shape)}"
        )
    zero = video.new_zeros((), dtype=torch.float32)
    diagnostics = {
        "dynamic_removed_fraction": zero,
        "dynamic_untracked_fraction": zero,
        "dynamic_tracks": zero,
        "dynamic_object_points": zero,
        "dynamic_actor_interpolated_boxes": zero,
        "dynamic_actor_fallback_boxes": zero,
    }
    if geometry_mode == "null":
        return (
            video.new_full(video.shape, float(min_value)),
            torch.zeros(
                (num_frames, height, width),
                device=video.device,
                dtype=torch.bool,
            ),
            diagnostics,
        )
    if geometry_mode in ("frozen_raw", "frozen_object") and not bool(
        selected.any()
    ):
        raise ValueError(
            f"{geometry_mode} requires at least one observed geometry frame"
        )
    poses = torch.as_tensor(
        lidar_to_world, device=video.device, dtype=torch.float32
    )
    source_poses = (
        poses
        if geometry_source_lidar_to_world is None
        else torch.as_tensor(
            geometry_source_lidar_to_world,
            device=video.device,
            dtype=torch.float32,
        )
    )
    source_motion = (
        pose_sensor_start_end
        if geometry_source_pose_sensor_start_end is None
        else torch.as_tensor(
            geometry_source_pose_sensor_start_end,
            device=video.device,
            dtype=torch.float32,
        )
    )
    if geometry_mode == "frozen_raw":
        condition, hit = warp_frozen_observed_history_with_spatial_map(
            video,
            source_poses,
            selected,
            ray_dirs=ray_dirs,
            row_elevations_rad=row_elevations_rad,
            column_azimuths_rad=column_azimuths_rad,
            row_azimuth_offsets_rad=row_azimuth_offsets_rad,
            min_range=min_range,
            max_range=max_range,
            min_value=min_value,
            inverse_depth=inverse_depth,
            radius_m=radius_m,
            voxel_size_m=voxel_size_m,
            pose_sensor_start_end=source_motion,
            target_lidar_to_world=poses,
            target_pose_sensor_start_end=pose_sensor_start_end,
        )
        return condition, hit, diagnostics

    required_boxes = {
        "box_object_to_world": box_object_to_world,
        "box_lwh": box_lwh,
        "box_valid": box_valid,
    }
    missing = [name for name, value in required_boxes.items() if value is None]
    if missing:
        raise KeyError(
            f"{geometry_mode} geometry is missing box tensors {missing}"
        )
    assert box_object_to_world is not None
    assert box_lwh is not None
    assert box_valid is not None
    if geometry_mode == "frozen_object":
        if static_only_geometry:
            static_video, removed_fraction = remove_points_inside_moving_boxes(
                video,
                source_poses,
                box_object_to_world,
                box_lwh,
                box_valid,
                box_track_id,
                ray_dirs=torch.as_tensor(
                    ray_dirs, device=video.device, dtype=torch.float32
                ),
                min_range=min_range,
                max_range=max_range,
                min_value=min_value,
                inverse_depth=inverse_depth,
                pose_sensor_start_end=source_motion,
                actor_box_time_mode=actor_box_time_mode,
            )
            condition, hit = warp_frozen_observed_history_with_spatial_map(
                static_video,
                source_poses,
                selected,
                ray_dirs=ray_dirs,
                row_elevations_rad=row_elevations_rad,
                column_azimuths_rad=column_azimuths_rad,
                row_azimuth_offsets_rad=row_azimuth_offsets_rad,
                min_range=min_range,
                max_range=max_range,
                min_value=min_value,
                inverse_depth=inverse_depth,
                radius_m=radius_m,
                voxel_size_m=voxel_size_m,
                pose_sensor_start_end=source_motion,
                target_lidar_to_world=poses,
                target_pose_sensor_start_end=pose_sensor_start_end,
            )
            diagnostics["dynamic_removed_fraction"] = removed_fraction
            return condition, hit, diagnostics
        if box_track_id is None:
            raise KeyError(
                "frozen_object geometry requires box_track_id"
            )
        return warp_frozen_object_history_with_spatial_map(
            video,
            source_poses,
            selected,
            box_object_to_world,
            box_lwh,
            box_valid,
            box_track_id,
            ray_dirs=ray_dirs,
            row_elevations_rad=row_elevations_rad,
            column_azimuths_rad=column_azimuths_rad,
            row_azimuth_offsets_rad=row_azimuth_offsets_rad,
            min_range=min_range,
            max_range=max_range,
            min_value=min_value,
            inverse_depth=inverse_depth,
            radius_m=radius_m,
            voxel_size_m=voxel_size_m,
            pose_sensor_start_end=source_motion,
            target_lidar_to_world=poses,
            target_pose_sensor_start_end=pose_sensor_start_end,
            actor_box_time_mode=actor_box_time_mode,
        )

    static_video, removed_fraction = remove_points_inside_moving_boxes(
        video,
        poses,
        box_object_to_world,
        box_lwh,
        box_valid,
        box_track_id,
        ray_dirs=torch.as_tensor(
            ray_dirs, device=video.device, dtype=torch.float32
        ),
        min_range=min_range,
        max_range=max_range,
        min_value=min_value,
        inverse_depth=inverse_depth,
        pose_sensor_start_end=pose_sensor_start_end,
        actor_box_time_mode=actor_box_time_mode,
    )
    condition, hit = warp_blockwise_history_with_spatial_map(
        video=static_video,
        lidar_to_world=poses,
        block_size=block_size,
        prefix_frames=prefix_frames,
        input_has_prepend=input_has_prepend,
        pose_sensor_start_end=pose_sensor_start_end,
        ray_dirs=ray_dirs,
        row_elevations_rad=row_elevations_rad,
        column_azimuths_rad=column_azimuths_rad,
        row_azimuth_offsets_rad=row_azimuth_offsets_rad,
        min_range=min_range,
        max_range=max_range,
        min_value=min_value,
        inverse_depth=inverse_depth,
        radius_m=radius_m,
        voxel_size_m=voxel_size_m,
        voxel_policy=voxel_policy,
        observation_unit=observation_unit,
        static_min_observations=static_min_observations,
        candidate_ttl_updates=candidate_ttl_updates,
        include_latest=include_latest,
        free_space_clear=free_space_clear,
        free_space_margin_m=free_space_margin_m,
        free_space_misses_to_remove=free_space_misses_to_remove,
    )
    diagnostics["dynamic_removed_fraction"] = removed_fraction
    return condition, hit, diagnostics


@torch.no_grad()
def warp_blockwise_history_with_spatial_map(
    video: torch.Tensor,
    lidar_to_world: Any,
    block_size: int,
    *,
    prefix_frames: int = 0,
    input_has_prepend: bool = True,
    pose_sensor_start_end: Any | None = None,
    ray_dirs: Any,
    row_elevations_rad: Any,
    column_azimuths_rad: Any,
    row_azimuth_offsets_rad: Any,
    min_range: float,
    max_range: float,
    min_value: float = -1.0,
    inverse_depth: bool = False,
    radius_m: float = 75.0,
    voxel_size_m: float = 0.2,
    voxel_policy: str = "newest",
    observation_unit: str = "block",
    static_min_observations: int = 2,
    candidate_ttl_updates: int = 1,
    include_latest: bool = True,
    free_space_clear: bool = False,
    free_space_margin_m: float = 0.75,
    free_space_misses_to_remove: int = 2,
    return_soft_source: bool = False,
    soft_density_kernel: int = 5,
    soft_density_saturation_hits: float = 5.0,
) -> (
    tuple[torch.Tensor, torch.Tensor]
    | tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]
):
    """Build a causal blockwise condition with a transient spatial map.

    With ``input_has_prepend=True``, the input is aligned as
    ``[dup(f0), f0, f1, ...]`` while poses contain only real sweeps. Otherwise
    every token maps directly to the pose at the same index. Each active block
    is rendered before its own GT returns are appended, so no target-block
    LiDAR can leak into its condition. Frame-level evidence changes promotion
    counts only; newly promoted voxels remain invisible until the next block.
    """
    if not isinstance(video, torch.Tensor):
        raise TypeError(
            "Torch spatial-map history requires video to be a torch.Tensor"
        )
    if video.ndim != 4:
        raise ValueError(f"video must be (C,T,H,W), got {tuple(video.shape)}")
    if video.shape[0] < 1:
        raise ValueError("video must contain at least one depth channel")
    if return_soft_source and video.shape[0] != 1:
        raise ValueError(
            "soft geometry proposals currently support range-only video"
        )
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")

    poses = torch.as_tensor(
        lidar_to_world, device=video.device, dtype=torch.float32
    )
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(
            "lidar_to_world must be (S,4,4), got "
            f"{tuple(poses.shape)}"
        )
    num_frames, height, width = map(int, video.shape[1:])
    required_poses = max(0, num_frames - int(input_has_prepend))
    if poses.shape[0] < required_poses:
        raise ValueError(
            f"{num_frames} tokens with input_has_prepend="
            f"{input_has_prepend} require at least {required_poses} poses, "
            f"got {poses.shape[0]}"
        )
    motion = None
    if pose_sensor_start_end is not None:
        motion = torch.as_tensor(
            pose_sensor_start_end,
            device=video.device,
            dtype=torch.float32,
        )
        if motion.shape != poses.shape:
            raise ValueError(
                "pose_sensor_start_end must align with lidar_to_world, got "
                f"{tuple(motion.shape)} and {tuple(poses.shape)}"
            )

    spatial_map = TorchSpatialPointMap(
        ray_dirs=ray_dirs,
        row_elevations_rad=row_elevations_rad,
        column_azimuths_rad=column_azimuths_rad,
        row_azimuth_offsets_rad=row_azimuth_offsets_rad,
        min_range=min_range,
        max_range=max_range,
        min_value=min_value,
        inverse_depth=inverse_depth,
        radius_m=radius_m,
        voxel_size_m=voxel_size_m,
        voxel_policy=voxel_policy,
        observation_unit=observation_unit,
        static_min_observations=static_min_observations,
        candidate_ttl_updates=candidate_ttl_updates,
        include_latest=include_latest,
        free_space_clear=free_space_clear,
        free_space_margin_m=free_space_margin_m,
        free_space_misses_to_remove=free_space_misses_to_remove,
        feature_channels=video.shape[0] - 1,
        device=video.device,
    )
    if (spatial_map.height, spatial_map.width) != (height, width):
        raise ValueError(
            "video and spatial-map ray grid must match: "
            f"video={(height, width)}, "
            f"rays={(spatial_map.height, spatial_map.width)}"
        )

    warped = video.new_full(video.shape, float(min_value))
    hit = torch.zeros(
        (num_frames, height, width),
        device=video.device,
        dtype=torch.bool,
    )
    soft_source = None
    soft_hit = None
    soft_confidence = None
    if return_soft_source:
        soft_source = torch.full_like(warped, float(min_value))
        soft_hit = torch.zeros_like(hit)
        soft_confidence = torch.zeros(
            (num_frames, height, width),
            device=video.device,
            dtype=torch.float32,
        )
    if prefix_frames:
        chunks = temporal_chunk_slices(
            num_frames,
            block_size,
            prefix_frames,
        )
    else:
        # Preserve the legacy diagnostic behavior, which permits a short
        # final block. Imaginaire matching uses the strict prefixed layout.
        chunks = [
            (start, min(start + block_size, num_frames))
            for start in range(0, num_frames, block_size)
        ]
    for block_start, block_end in chunks:
        if block_start > 0:
            target_poses = lidar_poses_for_global_tokens(
                poses,
                block_start,
                block_end - block_start,
                input_has_prepend=input_has_prepend,
            )
            target_motion = (
                lidar_poses_for_global_tokens(
                    motion,
                    block_start,
                    block_end - block_start,
                    input_has_prepend=input_has_prepend,
                )
                if motion is not None
                else None
            )
            if video.shape[0] == 1:
                rendered, rendered_hit = spatial_map.render(
                    target_poses,
                    target_pose_sensor_start_end=target_motion,
                )
            else:
                rendered_range, rendered_features, rendered_hit = (
                    spatial_map.render_with_features(
                        target_poses,
                        target_pose_sensor_start_end=target_motion,
                    )
                )
                rendered = torch.cat(
                    [rendered_range, rendered_features], dim=0
                )
            warped[:, block_start:block_end] = rendered.to(video.dtype)
            hit[block_start:block_end] = rendered_hit
            if return_soft_source:
                assert soft_source is not None
                assert soft_hit is not None
                assert soft_confidence is not None
                (
                    rendered_soft,
                    rendered_soft_hit,
                    temporal_confidence,
                ) = spatial_map.render_soft_proposal(
                    target_poses,
                    target_pose_sensor_start_end=target_motion,
                )
                density_confidence = local_hit_density_confidence(
                    rendered_soft_hit,
                    kernel_size=soft_density_kernel,
                    saturation_hits=soft_density_saturation_hits,
                )
                soft_source[:, block_start:block_end] = (
                    rendered_soft.to(video.dtype)
                )
                soft_hit[block_start:block_end] = rendered_soft_hit
                soft_confidence[block_start:block_end] = (
                    temporal_confidence * density_confidence
                )
        spatial_map.append_block(
            normalized_block=video[:1, block_start:block_end],
            point_feature_block=video[1:, block_start:block_end],
            lidar_to_world=poses,
            global_start=block_start,
            input_has_prepend=input_has_prepend,
            pose_sensor_start_end=motion,
        )
    if return_soft_source:
        assert soft_source is not None
        assert soft_hit is not None
        assert soft_confidence is not None
        return warped, hit, soft_source, soft_hit, soft_confidence
    return warped, hit


def range_sweep_to_world_points(
    normalized_range: Any,
    lidar_to_world: Any,
    ray_dirs: np.ndarray,
    min_range: float,
    max_range: float,
    min_value: float = -1.0,
    inverse_depth: bool = False,
    pose_sensor_start_end: Any | None = None,
) -> np.ndarray:
    """Convert one normalized range sweep into world-space points.

    When ``pose_sensor_start_end`` is provided, each azimuth column is first
    ego-deskewed into the sweep-start sensor frame using its normalized scan
    time. The final world transform remains ``lidar_to_world`` at sweep start.
    """
    if hasattr(normalized_range, "detach"):
        sweep = normalized_range.detach().float().cpu().numpy()
    else:
        sweep = np.asarray(normalized_range)
    if hasattr(lidar_to_world, "detach"):
        pose = lidar_to_world.detach().double().cpu().numpy()
    else:
        pose = np.asarray(lidar_to_world, dtype=np.float64)
    if sweep.ndim == 3 and sweep.shape[0] == 1:
        sweep = sweep[0]
    if sweep.ndim != 2:
        raise ValueError(
            f"normalized_range must be (H,W) or (1,H,W), got {sweep.shape}"
        )
    if pose.shape != (4, 4):
        raise ValueError(f"lidar_to_world must be (4,4), got {pose.shape}")
    dirs = np.asarray(ray_dirs, dtype=np.float32)
    if dirs.shape != (*sweep.shape, 3):
        raise ValueError(
            f"ray_dirs must be {(*sweep.shape, 3)}, got {dirs.shape}"
        )

    sweep = sweep.astype(np.float32, copy=False)
    valid = np.isfinite(sweep) & (sweep > float(min_value) + 1.0e-5)
    depth = _unnormalize_valid_depth(
        sweep,
        valid,
        min_range,
        max_range,
        min_value,
        inverse_depth,
    )
    points_local = (depth[..., None] * dirs)[valid]
    if points_local.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    if pose_sensor_start_end is not None:
        relative_pose = (
            pose_sensor_start_end.detach().double().cpu().numpy()
            if hasattr(pose_sensor_start_end, "detach")
            else np.asarray(pose_sensor_start_end, dtype=np.float64)
        )
        if relative_pose.shape != (4, 4):
            raise ValueError(
                "pose_sensor_start_end must be (4,4), got "
                f"{relative_pose.shape}"
            )
        from utils.lidar_rangemap import apply_motion_compensation_impl

        columns = np.broadcast_to(
            np.arange(sweep.shape[1], dtype=np.float64), sweep.shape
        )[valid]
        scan_fraction = columns / float(max(sweep.shape[1] - 1, 1))
        points_local = apply_motion_compensation_impl(
            points_local.astype(np.float64),
            relative_pose,
            [0.0, 1.0],
            scan_fraction,
        )
    return (
        points_local.astype(np.float64) @ pose[:3, :3].T
        + pose[:3, 3][None]
    )


def generated_block_to_world_points(
    normalized_block: Any,
    lidar_to_world: Any,
    global_start: int,
    ray_dirs: np.ndarray,
    min_range: float,
    max_range: float,
    min_value: float = -1.0,
    inverse_depth: bool = False,
) -> list[np.ndarray]:
    """Convert newly generated tokens to world points, skipping token-0 prepend."""
    if hasattr(normalized_block, "detach"):
        block = normalized_block.detach().float().cpu().numpy()
    else:
        block = np.asarray(normalized_block)
    if hasattr(lidar_to_world, "detach"):
        poses = lidar_to_world.detach().double().cpu().numpy()
    else:
        poses = np.asarray(lidar_to_world, dtype=np.float64)
    if block.ndim == 4:
        block = block[0]
    if block.ndim != 3:
        raise ValueError(
            f"normalized_block must be (C,T,H,W) or (T,H,W), got {block.shape}"
        )
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"lidar_to_world must be (S,4,4), got {poses.shape}")

    parts = []
    for local_idx in range(block.shape[0]):
        global_idx = global_start + local_idx
        if global_idx == 0:
            continue
        pose_idx = global_idx - 1
        if pose_idx >= poses.shape[0]:
            raise ValueError(
                f"generated token {global_idx} has no real pose in {poses.shape}"
            )
        points = range_sweep_to_world_points(
            normalized_range=block[local_idx],
            lidar_to_world=poses[pose_idx],
            ray_dirs=ray_dirs,
            min_range=min_range,
            max_range=max_range,
            min_value=min_value,
            inverse_depth=inverse_depth,
        )
        if points.shape[0] > 0:
            parts.append(points)
    return parts


def render_world_points_to_range_video(
    points_world: Any,
    target_lidar_to_world: Any,
    ray_dirs: np.ndarray,
    row_elevations_rad: np.ndarray,
    column_azimuths_rad: np.ndarray,
    row_azimuth_offsets_rad: np.ndarray,
    min_range: float,
    max_range: float,
    min_value: float = -1.0,
    inverse_depth: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Rasterize a persistent world-space cloud into target LiDAR poses."""
    if hasattr(points_world, "detach"):
        points = points_world.detach().double().cpu().numpy()
    else:
        points = np.asarray(points_world, dtype=np.float64)
    if hasattr(target_lidar_to_world, "detach"):
        poses = target_lidar_to_world.detach().double().cpu().numpy()
    else:
        poses = np.asarray(target_lidar_to_world, dtype=np.float64)
    if points.ndim != 2 or points.shape[1:] != (3,):
        raise ValueError(f"points_world must be (N,3), got {points.shape}")
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(
            f"target_lidar_to_world must be (T,4,4), got {poses.shape}"
        )

    dirs = np.asarray(ray_dirs)
    if dirs.ndim != 3 or dirs.shape[2] != 3:
        raise ValueError(f"ray_dirs must be (H,W,3), got {dirs.shape}")
    height, width = dirs.shape[:2]
    warped = np.full(
        (1, poses.shape[0], height, width),
        float(min_value),
        dtype=np.float32,
    )
    hit = np.zeros((poses.shape[0], height, width), dtype=bool)
    if points.shape[0] == 0:
        return warped, hit

    elev = np.asarray(row_elevations_rad, dtype=np.float64)
    azim = np.asarray(column_azimuths_rad, dtype=np.float64)
    row_off = np.asarray(row_azimuth_offsets_rad, dtype=np.float64)
    for target_idx, target_pose in enumerate(poses):
        rows, cols, ranges = project_world_points(
            points,
            target_pose,
            elev,
            azim,
            row_off,
            width,
            width,
            min_range,
            max_range,
        )
        metric = scatter_min_range(height, width, rows, cols, ranges)
        hit[target_idx] = metric > 0.0
        warped[0, target_idx] = _normalize_metric_depth(
            metric,
            min_range,
            max_range,
            min_value,
            inverse_depth,
        )
    return warped, hit


def _nearest_beam_indices_numpy(
    points_sensor: np.ndarray,
    ranges: np.ndarray,
    row_elevations_rad: np.ndarray,
    column_azimuths_rad: np.ndarray,
    row_azimuth_offsets_rad: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Match sensor-space points to the nearest native LiDAR beams."""
    points = np.asarray(points_sensor, dtype=np.float64)
    radii = np.asarray(ranges, dtype=np.float64)
    elev = np.asarray(row_elevations_rad, dtype=np.float64)
    azim = np.asarray(column_azimuths_rad, dtype=np.float64)
    row_off = np.asarray(row_azimuth_offsets_rad, dtype=np.float64)
    if points.ndim != 2 or points.shape[1:] != (3,):
        raise ValueError(
            f"points_sensor must be (N,3), got {points.shape}"
        )
    if radii.shape != (points.shape[0],):
        raise ValueError(
            f"ranges must be ({points.shape[0]},), got {radii.shape}"
        )
    if points.shape[0] == 0:
        empty = np.empty((0,), dtype=np.int64)
        return empty, empty

    point_elev = np.arcsin(
        np.clip(points[:, 2] / np.maximum(radii, 1.0e-12), -1.0, 1.0)
    )
    elev_order = np.argsort(elev, kind="stable")
    elev_sorted = elev[elev_order]
    elev_idx = np.searchsorted(elev_sorted, point_elev)
    elev_idx0 = np.clip(elev_idx - 1, 0, elev.shape[0] - 1)
    elev_idx1 = np.clip(elev_idx, 0, elev.shape[0] - 1)
    elev_dist0 = np.abs(point_elev - elev_sorted[elev_idx0])
    elev_dist1 = np.abs(point_elev - elev_sorted[elev_idx1])
    nearest_sorted = np.where(
        elev_dist0 < elev_dist1,
        elev_idx0,
        np.where(
            elev_dist1 < elev_dist0,
            elev_idx1,
            np.where(
                elev_order[elev_idx0] <= elev_order[elev_idx1],
                elev_idx0,
                elev_idx1,
            ),
        ),
    )
    rows = elev_order[nearest_sorted]

    def _wrap(angle: np.ndarray) -> np.ndarray:
        return (angle + np.pi) % (2.0 * np.pi) - np.pi

    point_azim = _wrap(
        np.arctan2(points[:, 1], points[:, 0]) - row_off[rows]
    )
    azim_wrapped = _wrap(azim)
    azim_order = np.argsort(azim_wrapped, kind="stable")
    azim_sorted = azim_wrapped[azim_order]
    azim_idx = np.searchsorted(azim_sorted, point_azim)
    azim_idx0 = (azim_idx - 1) % azim.shape[0]
    azim_idx1 = azim_idx % azim.shape[0]
    azim_dist0 = np.abs(_wrap(point_azim - azim_sorted[azim_idx0]))
    azim_dist1 = np.abs(_wrap(point_azim - azim_sorted[azim_idx1]))
    columns = azim_order[
        np.where(azim_dist0 <= azim_dist1, azim_idx0, azim_idx1)
    ]
    return rows.astype(np.int64), columns.astype(np.int64)


def render_world_points_to_rolling_range_video(
    points_world: Any,
    target_lidar_to_world: Any,
    target_pose_sensor_start_end: Any,
    ray_dirs: np.ndarray,
    row_elevations_rad: np.ndarray,
    column_azimuths_rad: np.ndarray,
    row_azimuth_offsets_rad: np.ndarray,
    min_range: float,
    max_range: float,
    min_value: float = -1.0,
    inverse_depth: bool = False,
    projection_iterations: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Rasterize a world cloud using each target sweep's per-column pose."""
    if projection_iterations <= 0:
        raise ValueError("projection_iterations must be positive")
    points = (
        points_world.detach().double().cpu().numpy()
        if hasattr(points_world, "detach")
        else np.asarray(points_world, dtype=np.float64)
    )
    poses = (
        target_lidar_to_world.detach().double().cpu().numpy()
        if hasattr(target_lidar_to_world, "detach")
        else np.asarray(target_lidar_to_world, dtype=np.float64)
    )
    target_motion = (
        target_pose_sensor_start_end.detach().double().cpu().numpy()
        if hasattr(target_pose_sensor_start_end, "detach")
        else np.asarray(target_pose_sensor_start_end, dtype=np.float64)
    )
    if points.ndim != 2 or points.shape[1:] != (3,):
        raise ValueError(f"points_world must be (N,3), got {points.shape}")
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(
            f"target_lidar_to_world must be (T,4,4), got {poses.shape}"
        )
    if target_motion.shape != (poses.shape[0], 4, 4):
        raise ValueError(
            "target_pose_sensor_start_end must be (T,4,4), got "
            f"{target_motion.shape} for T={poses.shape[0]}"
        )

    dirs = np.asarray(ray_dirs)
    if dirs.ndim != 3 or dirs.shape[2] != 3:
        raise ValueError(f"ray_dirs must be (H,W,3), got {dirs.shape}")
    height, width = dirs.shape[:2]
    warped = np.full(
        (1, poses.shape[0], height, width),
        float(min_value),
        dtype=np.float32,
    )
    hit = np.zeros((poses.shape[0], height, width), dtype=bool)
    if points.shape[0] == 0:
        return warped, hit

    from utils.lidar_rangemap import apply_motion_compensation_impl

    for target_idx, target_pose in enumerate(poses):
        rotation = target_pose[:3, :3].astype(np.float64)
        translation = target_pose[:3, 3].astype(np.float64)
        points_start = (points - translation[None]) @ rotation
        start_ranges = np.linalg.norm(points_start, axis=1)
        _, columns = _nearest_beam_indices_numpy(
            points_start,
            start_ranges,
            row_elevations_rad,
            column_azimuths_rad,
            row_azimuth_offsets_rad,
        )
        inverse_relative = np.linalg.inv(target_motion[target_idx])
        for _ in range(projection_iterations):
            fractions = columns.astype(np.float64) / float(
                max(width - 1, 1)
            )
            points_sensor = apply_motion_compensation_impl(
                points_start,
                inverse_relative,
                [0.0, 1.0],
                fractions,
            )
            ranges = np.linalg.norm(points_sensor, axis=1)
            _, columns = _nearest_beam_indices_numpy(
                points_sensor,
                ranges,
                row_elevations_rad,
                column_azimuths_rad,
                row_azimuth_offsets_rad,
            )

        fractions = columns.astype(np.float64) / float(max(width - 1, 1))
        points_sensor = apply_motion_compensation_impl(
            points_start,
            inverse_relative,
            [0.0, 1.0],
            fractions,
        )
        ranges = np.linalg.norm(points_sensor, axis=1)
        rows, columns = _nearest_beam_indices_numpy(
            points_sensor,
            ranges,
            row_elevations_rad,
            column_azimuths_rad,
            row_azimuth_offsets_rad,
        )
        keep = (ranges >= min_range) & (ranges <= max_range)
        metric = scatter_min_range(
            height,
            width,
            rows[keep],
            columns[keep],
            ranges[keep].astype(np.float32),
        )
        hit[target_idx] = metric > 0.0
        warped[0, target_idx] = _normalize_metric_depth(
            metric,
            min_range,
            max_range,
            min_value,
            inverse_depth,
        )
    return warped, hit


def warp_latest_past_sweep_to_video(
    video: Any,
    lidar_to_world: Any,
    n_past: int,
    ray_dirs: np.ndarray,
    row_elevations_rad: np.ndarray,
    column_azimuths_rad: np.ndarray,
    row_azimuth_offsets_rad: np.ndarray,
    min_range: float,
    max_range: float,
    min_value: float = -1.0,
    inverse_depth: bool = False,
    history_mode: str = "latest",
) -> tuple[np.ndarray, np.ndarray]:
    """Warp observed sweep(s) into every prepend-aligned frame pose.

    Args:
        video: Normalized range video ``(C, S+1, H, W)`` with frame-0 prepend.
        lidar_to_world: Sampled sweep-start poses ``(S, 4, 4)`` without prepend.
        n_past: Number of observed sampled sweeps.
        history_mode: ``"latest"`` uses video index ``n_past`` and pose index
            ``n_past - 1``. ``"all"`` uses every unique past video index
            ``1..n_past`` and merges their target-pose projections with one
            nearest-return z-buffer.

    Returns:
        ``(warped_video, hit_mask)`` with shapes ``(1, S+1, H, W)`` and
        ``(S+1, H, W)``. Future lidar never enters the construction. Empty
        target pixels retain the dataset's normalized invalid sentinel.
    """
    if hasattr(video, "detach"):
        video_np = video.detach().cpu().numpy()
    else:
        video_np = np.asarray(video)
    if hasattr(lidar_to_world, "detach"):
        l2w = lidar_to_world.detach().cpu().numpy()
    else:
        l2w = np.asarray(lidar_to_world)

    if video_np.ndim != 4 or video_np.shape[0] < 1:
        raise ValueError(f"video must be (C,T,H,W), got {video_np.shape}")
    if l2w.ndim != 3 or l2w.shape[1:] != (4, 4):
        raise ValueError(f"lidar_to_world must be (S,4,4), got {l2w.shape}")
    if video_np.shape[1] != l2w.shape[0] + 1:
        raise ValueError(
            f"prepend mismatch: video T={video_np.shape[1]} but poses S={l2w.shape[0]}"
        )
    if not 0 < n_past <= l2w.shape[0]:
        raise ValueError(f"n_past={n_past} is invalid for S={l2w.shape[0]}")
    if history_mode not in ("latest", "all"):
        raise ValueError(
            f"history_mode must be 'latest' or 'all', got {history_mode!r}"
        )

    _, T, H, W = video_np.shape
    dirs = np.asarray(ray_dirs, dtype=np.float32)
    if dirs.shape != (H, W, 3):
        raise ValueError(f"ray_dirs must be {(H, W, 3)}, got {dirs.shape}")
    elev = np.asarray(row_elevations_rad, dtype=np.float64)
    azim = np.asarray(column_azimuths_rad, dtype=np.float64)
    row_off = np.asarray(row_azimuth_offsets_rad, dtype=np.float64)
    if elev.shape != (H,) or row_off.shape != (H,) or azim.shape != (W,):
        raise ValueError(
            "beam-table shapes must match video grid: "
            f"elev={elev.shape}, azim={azim.shape}, row_off={row_off.shape}, "
            f"grid={(H, W)}"
        )

    source_indices = [n_past] if history_mode == "latest" else list(
        range(1, n_past + 1)
    )
    world_parts = []
    for video_idx in source_indices:
        source_norm = video_np[0, video_idx].astype(np.float32, copy=True)
        valid = np.isfinite(source_norm) & (
            source_norm > float(min_value) + 1.0e-5
        )
        source_depth = _unnormalize_valid_depth(
            source_norm,
            valid,
            min_range,
            max_range,
            min_value,
            inverse_depth,
        )
        points_source = (source_depth[..., None] * dirs)[valid]
        source_pose = l2w[video_idx - 1].astype(np.float64)
        world_parts.append(
            points_source.astype(np.float64) @ source_pose[:3, :3].T
            + source_pose[:3, 3][None]
        )
    points_world = np.concatenate(world_parts, axis=0)
    poses_prepend = np.concatenate([l2w[:1], l2w], axis=0).astype(np.float64)

    warped = np.full((1, T, H, W), float(min_value), dtype=np.float32)
    hit = np.zeros((T, H, W), dtype=bool)
    for t in range(T):
        rows, cols, ranges = project_world_points(
            points_world,
            poses_prepend[t],
            elev,
            azim,
            row_off,
            W,
            W,
            min_range,
            max_range,
        )
        metric = scatter_min_range(H, W, rows, cols, ranges)
        hit[t] = metric > 0.0
        warped[0, t] = _normalize_metric_depth(
            metric,
            min_range,
            max_range,
            min_value,
            inverse_depth,
        )
    return warped, hit


def warp_blockwise_history_to_video(
    video: Any,
    lidar_to_world: Any,
    block_size: int,
    ray_dirs: np.ndarray,
    row_elevations_rad: np.ndarray,
    column_azimuths_rad: np.ndarray,
    row_azimuth_offsets_rad: np.ndarray,
    min_range: float,
    max_range: float,
    min_value: float = -1.0,
    inverse_depth: bool = False,
    history_mode: str = "all",
) -> tuple[np.ndarray, np.ndarray]:
    """Build a leakage-free history warp for each causal temporal block.

    The input video is prepend-aligned: ``[dup(f0), f0, f1, ...]``. Block zero
    has no earlier observations and remains invalid. For each later block
    beginning at video index ``s``, only unique lidar frames at indices
    ``1..s-1`` are accumulated and rendered into that block's target poses.
    Frames inside the target block never enter their own conditioning signal.
    """
    if hasattr(video, "detach"):
        video_np = video.detach().cpu().numpy()
    else:
        video_np = np.asarray(video)
    if video_np.ndim != 4:
        raise ValueError(f"video must be (C,T,H,W), got {video_np.shape}")
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")

    _, num_frames, height, width = video_np.shape
    warped = np.full(
        (1, num_frames, height, width),
        float(min_value),
        dtype=np.float32,
    )
    hit = np.zeros((num_frames, height, width), dtype=bool)

    for block_start in range(block_size, num_frames, block_size):
        # video index 0 duplicates f0, so the number of unique observed sweeps
        # before this block is exactly block_start - 1.
        n_past = block_start - 1
        block_warped, block_hit = warp_latest_past_sweep_to_video(
            video=video_np,
            lidar_to_world=lidar_to_world,
            n_past=n_past,
            ray_dirs=ray_dirs,
            row_elevations_rad=row_elevations_rad,
            column_azimuths_rad=column_azimuths_rad,
            row_azimuth_offsets_rad=row_azimuth_offsets_rad,
            min_range=min_range,
            max_range=max_range,
            min_value=min_value,
            inverse_depth=inverse_depth,
            history_mode=history_mode,
        )
        block_end = min(block_start + block_size, num_frames)
        warped[:, block_start:block_end] = block_warped[
            :, block_start:block_end
        ]
        hit[block_start:block_end] = block_hit[block_start:block_end]

    return warped, hit
