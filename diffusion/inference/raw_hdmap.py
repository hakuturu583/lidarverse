# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inference-only raw-vector HDMap conditioning for prepared PAI samples."""

import numpy as np
import torch
import torch.nn.functional as F


def validate_raw_hdmap(sample):
    required = ("map_vertices", "map_offsets", "box_valid", "box_lwh", "box_object_to_world")
    for key in required:
        if key not in sample:
            raise ValueError(f"Missing {key}; rerun scripts/data/prepare_pai_bidir.py")
    vertices, offsets = sample["map_vertices"], sample["map_offsets"]
    if vertices.ndim != 2 or vertices.shape[1:] != (3,) or not np.isfinite(vertices).all():
        raise ValueError("Raw map vertices must be finite (N,3) world points")
    if (offsets.ndim != 1 or len(offsets) < 2 or offsets.dtype.kind not in "iu"
            or offsets[0] != 0 or offsets[-1] != len(vertices) or (np.diff(offsets) < 2).any()):
        raise ValueError("Raw map offsets must delimit nonempty polylines of at least two vertices")
    shape = sample["box_valid"].shape
    if len(shape) != 2 or shape[0] != len(sample["range_m"]):
        raise ValueError("Raw box masks must be (T,M), aligned with the real sweeps")
    if sample["box_lwh"].shape != shape + (3,) or sample["box_object_to_world"].shape != shape + (4, 4):
        raise ValueError("Raw box sizes/poses do not match the box mask")
    active = sample["box_valid"].astype(bool)
    if (not np.isfinite(sample["box_object_to_world"][active]).all()
            or not np.isfinite(sample["box_lwh"][active]).all()
            or (sample["box_lwh"][active] <= 0).any()):
        raise ValueError("Active boxes must have finite poses and positive finite sizes")


def _sample_polyline(vertices, spacing):
    # Match LiDARGen's raw-MADS densification, including endpoint handling.
    vertices = np.asarray(vertices, dtype=np.float64)
    points = []
    for start, end in zip(vertices[:-1], vertices[1:]):
        distance = np.linalg.norm(end - start)
        count = int(np.clip(np.floor(distance / spacing), 2, 10000))
        fractions = np.linspace(0, 1, count)[:-1]
        points.append(start + fractions[:, None] * (end - start))
    points.append(vertices[-1:])
    return np.concatenate(points).astype(np.float32)


def _box_points(sample, frame, spacing):
    signs = np.array([[1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1],
                      [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1]], dtype=np.float64)
    wireframe = [0, 1, 2, 3, 0, 4, 5, 6, 7, 4, 5, 1, 2, 6, 7, 3]
    points = []
    for index in np.flatnonzero(sample["box_valid"][frame]):
        corners = signs * sample["box_lwh"][frame, index] / 2
        pose = sample["box_object_to_world"][frame, index].astype(np.float64)
        corners = corners @ pose[:3, :3].T + pose[:3, 3]
        points.append(_sample_polyline(corners[wireframe], spacing))
    return np.concatenate(points) if points else np.empty((0, 3), np.float32)


def _dilate_nearest(normalized, kernel_size):
    """Widen bbox pixels without averaging ranges or filling empty neighborhoods."""
    if kernel_size < 1 or kernel_size % 2 != 1:
        raise ValueError("BBox dilation must be a positive odd integer")
    if kernel_size == 1:
        return normalized
    valid = torch.isfinite(normalized) & (normalized > -1 + 1e-5)
    work = normalized[:, None].masked_fill(~valid[:, None], torch.inf)
    nearest = -F.max_pool2d(-work, kernel_size, stride=1, padding=kernel_size // 2)
    return torch.where(torch.isfinite(nearest), nearest, -1.)[:, 0]


@torch.inference_mode()
def rasterize_raw_hdmap(sample, *, bbox_dilation=1, seg_interval=0.1):
    """Return a (T,H,W) normalized condition and provenance, without GT LiDAR."""
    from diffusion.training.datasets.lidar_dataset.forecast_warp import (
        TorchSpatialPointMap, compose_normalized_range_channels,
    )
    from diffusion.training.models.hdmaptolidar_model_rf import _sensor_tables_from_ray_directions

    validate_raw_hdmap(sample)
    if not np.isfinite(seg_interval) or seg_interval <= 0:
        raise ValueError("Raw HDMap segment spacing must be positive and finite")
    if bbox_dilation < 1 or bbox_dilation % 2 != 1:
        raise ValueError("BBox dilation must be a positive odd integer")
    rays = torch.as_tensor(sample["ray_directions"], dtype=torch.float32, device="cuda")
    elevations, azimuths, offsets = _sensor_tables_from_ray_directions(rays, allow_approximate=True)
    renderer = TorchSpatialPointMap(
        ray_dirs=rays, row_elevations_rad=elevations, column_azimuths_rad=azimuths,
        row_azimuth_offsets_rad=offsets, min_range=2.5, max_range=100,
        min_value=-1, inverse_depth=False, radius_m=100, voxel_size_m=0.2, device="cuda",
    )
    vertices, offsets = sample["map_vertices"], sample["map_offsets"]
    static_points = np.concatenate([_sample_polyline(vertices[a:b], seg_interval)
                                    for a, b in zip(offsets[:-1], offsets[1:])])
    poses = sample["pose_sensor_start_end"].astype(np.float32)
    motions = np.linalg.inv(poses[:, 1]) @ poses[:, 0]
    static, static_hit = renderer.render_points_world(
        static_points, poses[:, 0], target_pose_sensor_start_end=motions,
    )
    boxes, box_hits = [], []
    for t, pose in enumerate(poses):
        ranges, hits = renderer.render_points_world(
            _box_points(sample, t, seg_interval), pose[None, 0],
            target_pose_sensor_start_end=motions[t:t + 1],
        )
        boxes.append(ranges[0, 0])
        box_hits.append(hits[0])
    thick_boxes = _dilate_nearest(torch.stack(boxes), bbox_dilation)
    condition = compose_normalized_range_channels(
        torch.stack((static[0], thick_boxes)), channel_dim=0, min_value=-1, inverse_depth=False,
    )[0].cpu().numpy()
    stats = dict(
        source="prepared_raw_vectors", bbox_dilation_kernel=bbox_dilation,
        map_dilation_kernel=1, segment_spacing_m=seg_interval,
        rolling_shutter=True, min_range_m=2.5, max_range_m=100,
        static_hit_fraction=float(static_hit.float().mean()),
        bbox_hit_fraction_before_dilation=float(torch.stack(box_hits).float().mean()),
        bbox_hit_fraction_after_dilation=float((thick_boxes > -1 + 1e-5).float().mean()),
        original_hit_fraction=float((sample["hdmap"] > -1 + 1e-5).mean()),
        condition_hit_fraction=float((condition > -1 + 1e-5).mean()),
    )
    return condition, stats
