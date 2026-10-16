# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""PandaSet Pandar64 range-image projection and cache reader.

The official raw release preserves Pandar64 firing rows and physical azimuths
and is the source of truth for the 64-row image.  The processed release stores
motion-compensated world points; paired raw/processed points recover and verify
the unpublished Pandar64-to-pose-reference extrinsic when a cache is built.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np


PANDASET_CACHE_VERSION = "pandaset_pandar64_64x1024_v6"
PANDASET_HEIGHT = 64
PANDASET_WIDTH = 1024
PANDASET_NATIVE_WIDTH = 1800
PANDASET_MIN_RANGE = 1.0
PANDASET_MAX_RANGE = 100.0
PANDASET_CALIBRATION_FRAMES = 3
PANDASET_CALIBRATION_MAX_POINTS_PER_FRAME = 50_000
PANDASET_CALIBRATION_MEDIAN_ERROR_MAX_M = 0.05
PANDASET_CALIBRATION_P90_ERROR_MAX_M = 0.15
PANDASET_AZIMUTH_GRID_P90_ERROR_MAX_DEG = 0.25
PANDASET_AZIMUTH_PHASE_QUANTIZATION_DEG = 0.01

# Exact values from the official devkit's
# tutorials/raw_depth_projection/pandar64_channel_distribution.csv.
PANDAR64_ELEVATION_DEG = np.asarray(
    [
        14.870, 11.020, 8.047, 5.045, 3.028, 2.016, 1.848, 1.676,
        1.510, 1.339, 1.172, 1.001, 0.834, 0.663, 0.496, 0.325,
        0.157, -0.012, -0.181, -0.349, -0.520, -0.687, -0.857,
        -1.025, -1.196, -1.363, -1.534, -1.700, -1.872, -2.040,
        -2.210, -2.377, -2.548, -2.712, -2.885, -3.052, -3.222,
        -3.387, -3.560, -3.724, -3.896, -4.062, -4.233, -4.397,
        -4.570, -4.732, -4.904, -5.069, -5.241, -5.403, -5.577,
        -5.738, -5.910, -6.073, -7.075, -8.071, -9.072, -9.897,
        -11.044, -12.018, -12.986, -13.942, -18.901, -24.909,
    ],
    dtype=np.float64,
)
PANDAR64_ELEVATION_RAD = np.deg2rad(PANDAR64_ELEVATION_DEG)
PANDAR64_HORIZONTAL_OFFSET_DEG = np.asarray(
    [
        -1.042, -1.042, -1.042, -1.042, -1.042, -1.042,
        1.042, 3.125, 5.208, -5.208, -3.125, -1.042,
        1.042, 3.125, 5.208, -5.208, -3.125, -1.042,
        1.042, 3.125, 5.208, -5.208, -3.125, -1.042,
        1.042, 3.125, 5.208, -5.208, -3.125, -1.042,
        1.042, 3.125, 5.208, -5.208, -3.125, -1.042,
        1.042, 3.125, 5.208, -5.208, -3.125, -1.042,
        1.042, 3.125, 5.208, -5.208, -3.125, -1.042,
        1.042, 3.125, 5.208, -5.208, -3.125, -1.042,
        -1.042, -1.042, -1.042, -1.042, -1.042, -1.042,
        -1.042, -1.042, -1.042, -1.042,
    ],
    dtype=np.float64,
)
# PandaSet ego axes are x-right/y-forward/z-up.  The repository convention is
# x-forward/y-left/z-up.
PANDASET_EGO_TO_REPO = np.asarray(
    [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
    dtype=np.float64,
)


def pandaset_ray_directions(
    width: int = PANDASET_WIDTH,
    *,
    azimuth_y_sign: int = 1,
    azimuth_phase_rad: float = 0.0,
    azimuth_column_sign: int = 1,
) -> np.ndarray:
    """Return Pandar64 rays on the calibrated scan-ordered azimuth grid.

    ``azimuth_y_sign`` is recovered together with the rigid Pandar64-to-pose
    calibration because the raw release does not publish that extrinsic or its
    azimuth handedness. ``azimuth_phase_rad`` and ``azimuth_column_sign`` map
    the official devkit's corrected scan-relative columns to physical azimuth.
    Keeping scan-relative column order preserves the rolling-shutter time axis.
    """
    width = int(width)
    if width <= 0:
        raise ValueError(f"PandaSet ray width must be positive, got {width}.")
    if int(azimuth_y_sign) not in (-1, 1):
        raise ValueError(
            f"PandaSet azimuth_y_sign must be -1 or 1, got {azimuth_y_sign}."
        )
    if int(azimuth_column_sign) not in (-1, 1):
        raise ValueError(
            "PandaSet azimuth_column_sign must be -1 or 1, got "
            f"{azimuth_column_sign}."
        )
    azimuth = float(azimuth_phase_rad) + int(azimuth_column_sign) * (
        np.arange(width, dtype=np.float64) + 0.5
    ) * (
        2.0 * np.pi / width
    )
    elevation = PANDAR64_ELEVATION_RAD[:, None]
    cos_elevation = np.cos(elevation)
    rays = np.stack(
        [
            cos_elevation * np.cos(azimuth)[None, :],
            int(azimuth_y_sign) * cos_elevation * np.sin(azimuth)[None, :],
            np.sin(elevation) * np.ones((1, width), dtype=np.float64),
        ],
        axis=-1,
    )
    return rays.astype(np.float32)


def quaternion_wxyz_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    """Convert normalized ``[..., w, x, y, z]`` quaternions to rotations."""
    quaternion = np.asarray(quaternion, dtype=np.float64)
    quaternion = quaternion / np.linalg.norm(quaternion, axis=-1, keepdims=True)
    w, x, y, z = np.moveaxis(quaternion, -1, 0)
    return np.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ],
        axis=-1,
    ).reshape(quaternion.shape[:-1] + (3, 3))


def _normalize_quaternion(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float64)
    return quaternion / np.linalg.norm(quaternion, axis=-1, keepdims=True)


def interpolate_poses(
    query_timestamps: np.ndarray,
    pose_timestamps: np.ndarray,
    positions: np.ndarray,
    quaternions_wxyz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Linearly interpolate translation and SLERP rotation at query times."""
    query_timestamps = np.asarray(query_timestamps, dtype=np.float64)
    pose_timestamps = np.asarray(pose_timestamps, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)
    quaternions_wxyz = _normalize_quaternion(quaternions_wxyz)
    if pose_timestamps.ndim != 1 or len(pose_timestamps) < 2:
        raise ValueError("PandaSet pose interpolation requires at least two poses.")
    if np.any(np.diff(pose_timestamps) <= 0):
        raise ValueError("PandaSet pose timestamps must be strictly increasing.")

    right = np.searchsorted(pose_timestamps, query_timestamps, side="right")
    right = np.clip(right, 1, len(pose_timestamps) - 1)
    left = right - 1
    denom = pose_timestamps[right] - pose_timestamps[left]
    # PandaSet frame timestamps mark sweep ends, so frame-0 points precede the
    # first available pose by up to one sweep.  Extrapolate from the nearest
    # pose pair at either boundary instead of collapsing those points onto the
    # first/last pose.
    alpha = (query_timestamps - pose_timestamps[left]) / denom

    translation = (
        positions[left] * (1.0 - alpha[:, None])
        + positions[right] * alpha[:, None]
    )
    q0 = quaternions_wxyz[left]
    q1 = quaternions_wxyz[right]
    dot = np.sum(q0 * q1, axis=-1)
    q1 = np.where((dot < 0.0)[:, None], -q1, q1)
    dot = np.abs(dot)
    linear = dot > 0.9995
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_theta = np.sin(theta)
    safe_sin = np.where(linear, 1.0, sin_theta)
    weight0 = np.sin((1.0 - alpha) * theta) / safe_sin
    weight1 = np.sin(alpha * theta) / safe_sin
    quaternion = weight0[:, None] * q0 + weight1[:, None] * q1
    quaternion[linear] = (
        (1.0 - alpha[linear, None]) * q0[linear]
        + alpha[linear, None] * q1[linear]
    )
    return translation, _normalize_quaternion(quaternion)


def pose_components_from_json(
    poses: Sequence[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray]:
    """Extract translations and wxyz quaternions from PandaSet pose JSON."""
    positions = np.asarray(
        [[p["position"]["x"], p["position"]["y"], p["position"]["z"]] for p in poses],
        dtype=np.float64,
    )
    quaternions = np.asarray(
        [
            [
                p["heading"]["w"],
                p["heading"]["x"],
                p["heading"]["y"],
                p["heading"]["z"],
            ]
            for p in poses
        ],
        dtype=np.float64,
    )
    return positions, quaternions


def _poses_to_pandar64_lidar_to_world(
    positions: np.ndarray,
    quaternions_wxyz: np.ndarray,
    pandar64_to_pose: np.ndarray,
) -> np.ndarray:
    """Compose PandaSet pose-reference motion with the recovered lidar extrinsic."""
    world_from_pose = np.broadcast_to(
        np.eye(4, dtype=np.float64), (len(positions), 4, 4)
    ).copy()
    world_from_pose[:, :3, :3] = quaternion_wxyz_to_matrix(quaternions_wxyz)
    world_from_pose[:, :3, 3] = positions
    return (world_from_pose @ np.asarray(pandar64_to_pose)).astype(np.float32)


def resolve_point_timestamps(
    point_timestamps: np.ndarray, frame_timestamp: float
) -> np.ndarray:
    """Resolve PandaSet point timestamps to absolute Unix seconds."""
    point_timestamps = np.asarray(point_timestamps, dtype=np.float64)
    finite = point_timestamps[np.isfinite(point_timestamps)]
    if finite.size == 0:
        return np.full(point_timestamps.shape, frame_timestamp, dtype=np.float64)
    if np.median(np.abs(finite)) > 1.0e8:
        return point_timestamps
    return frame_timestamp + point_timestamps


def world_points_to_repo_sensor(
    points_world: np.ndarray,
    query_timestamps: np.ndarray,
    pose_timestamps: np.ndarray,
    positions: np.ndarray,
    quaternions_wxyz: np.ndarray,
) -> np.ndarray:
    """Motion-compensate world points into their instantaneous lidar frames."""
    translations, quaternions = interpolate_poses(
        query_timestamps, pose_timestamps, positions, quaternions_wxyz
    )
    rotations = quaternion_wxyz_to_matrix(quaternions)
    points_pandaset_ego = np.einsum(
        "nij,nj->ni", rotations.transpose(0, 2, 1), points_world - translations
    )
    return points_pandaset_ego @ PANDASET_EGO_TO_REPO.T


def world_points_to_pose_frame(
    points_world: np.ndarray,
    query_timestamps: np.ndarray,
    pose_timestamps: np.ndarray,
    positions: np.ndarray,
    quaternions_wxyz: np.ndarray,
) -> np.ndarray:
    """Motion-compensate world points into PandaSet's pose-reference frame."""
    translations, quaternions = interpolate_poses(
        query_timestamps, pose_timestamps, positions, quaternions_wxyz
    )
    rotations = quaternion_wxyz_to_matrix(quaternions)
    return np.einsum(
        "nij,nj->ni", rotations.transpose(0, 2, 1), points_world - translations
    )


def raw_pandar64_points(
    distance: np.ndarray,
    elevation_deg: np.ndarray,
    azimuth_deg: np.ndarray,
    *,
    azimuth_y_sign: int,
) -> np.ndarray:
    """Convert official raw firing angles and ranges to Pandar64-frame XYZ."""
    distance = np.asarray(distance, dtype=np.float64)
    elevation = np.deg2rad(np.asarray(elevation_deg, dtype=np.float64))
    azimuth = np.deg2rad(np.mod(np.asarray(azimuth_deg, dtype=np.float64), 360.0))
    cos_elevation = np.cos(elevation)
    return distance[:, None] * np.stack(
        [
            cos_elevation * np.cos(azimuth),
            int(azimuth_y_sign) * cos_elevation * np.sin(azimuth),
            np.sin(elevation),
        ],
        axis=-1,
    )


def raw_pandar64_scan_columns(
    laser_id: np.ndarray, column_id: np.ndarray
) -> np.ndarray:
    """Apply official offsets to obtain the scan-relative spherical columns."""
    laser_id = np.asarray(laser_id, dtype=np.int64)
    column_id = np.asarray(column_id, dtype=np.int64)
    if laser_id.shape != column_id.shape:
        raise ValueError(
            f"laser_id/column_id shapes differ: {laser_id.shape}/{column_id.shape}."
        )
    if np.any((laser_id < 0) | (laser_id >= PANDASET_HEIGHT)):
        raise ValueError("PandaSet laser_id lies outside [0, 64).")
    column_shift = (
        PANDASET_NATIVE_WIDTH / 360.0 * PANDAR64_HORIZONTAL_OFFSET_DEG
    ).astype(np.int64)
    scan_column = column_id - int(column_id.min())
    return np.mod(
        scan_column + column_shift[laser_id] + PANDASET_NATIVE_WIDTH,
        PANDASET_NATIVE_WIDTH,
    ).astype(np.int64)


def recover_pandar64_azimuth_grid(
    scan_columns: np.ndarray,
    corrected_azimuth_deg: np.ndarray,
) -> tuple[float, int, float]:
    """Fit physical azimuth phase/direction to official scan-relative columns."""
    scan_columns = np.asarray(scan_columns, dtype=np.float64)
    azimuth = np.deg2rad(
        np.mod(np.asarray(corrected_azimuth_deg, dtype=np.float64), 360.0)
    )
    if scan_columns.shape != azimuth.shape:
        raise ValueError(
            f"PandaSet scan-column/azimuth shapes differ: "
            f"{scan_columns.shape}/{azimuth.shape}."
        )
    finite = np.isfinite(scan_columns) & np.isfinite(azimuth)
    scan_columns, azimuth = scan_columns[finite], azimuth[finite]
    if scan_columns.size < 1_000:
        raise ValueError("PandaSet azimuth-grid fit requires 1,000 samples.")

    def _wrap(angle: np.ndarray) -> np.ndarray:
        return (angle + np.pi) % (2.0 * np.pi) - np.pi

    candidates = []
    scan_angle = scan_columns * (2.0 * np.pi / PANDASET_NATIVE_WIDTH)
    for column_sign in (-1, 1):
        phase_samples = _wrap(azimuth - column_sign * scan_angle)
        phase = float(
            np.angle(np.mean(np.exp(1j * phase_samples), dtype=np.complex128))
        )
        error_deg = np.rad2deg(
            np.abs(_wrap(azimuth - (phase + column_sign * scan_angle)))
        )
        candidates.append(
            (
                float(np.percentile(error_deg, 90.0)),
                column_sign,
                phase,
            )
        )
    _, column_sign, phase = min(candidates)
    phase_deg = (
        round(
            np.rad2deg(phase) / PANDASET_AZIMUTH_PHASE_QUANTIZATION_DEG
        )
        * PANDASET_AZIMUTH_PHASE_QUANTIZATION_DEG
    )
    phase = float(np.deg2rad(phase_deg))
    error_deg = np.rad2deg(
        np.abs(_wrap(azimuth - (phase + column_sign * scan_angle)))
    )
    p90_error_deg = float(np.percentile(error_deg, 90.0))
    if p90_error_deg > PANDASET_AZIMUTH_GRID_P90_ERROR_MAX_DEG:
        raise ValueError(
            "Official PandaSet corrected azimuths do not form a stable "
            f"scan-relative grid: p90={p90_error_deg:.4f} deg, expected <= "
            f"{PANDASET_AZIMUTH_GRID_P90_ERROR_MAX_DEG:.3f} deg."
        )
    return phase, int(column_sign), p90_error_deg


def _fit_rigid_transform(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Least-squares proper rigid transform ``target ~= T @ source``."""
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError(
            f"Rigid-fit source/target must both be (N,3), got "
            f"{source.shape} and {target.shape}."
        )
    if source.shape[0] < 3:
        raise ValueError("Rigid fit requires at least three correspondences.")
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    covariance = (source - source_center).T @ (target - target_center)
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1.0
        rotation = vt.T @ u.T
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = target_center - rotation @ source_center
    return transform


def _transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return (
        np.asarray(points, dtype=np.float64) @ transform[:3, :3].T
        + transform[:3, 3]
    )


def recover_pandar64_to_pose(
    raw_by_sign: dict[int, np.ndarray],
    pose_points: np.ndarray,
) -> tuple[np.ndarray, int, float, float]:
    """Recover the unpublished raw-Pandar64-to-pose rigid calibration.

    The raw and processed PandaSet releases preserve point order for Pandar64.
    The processed points are first motion-compensated into the pose-reference
    frame.  We then robustly fit both possible raw azimuth handednesses and
    retain the proper rigid transform with the lower residual.
    """
    pose_points = np.asarray(pose_points, dtype=np.float64)
    candidates = []
    for sign in (-1, 1):
        source = np.asarray(raw_by_sign[sign], dtype=np.float64)
        if source.shape != pose_points.shape:
            raise ValueError(
                f"PandaSet calibration shape mismatch for sign {sign}: "
                f"{source.shape} versus {pose_points.shape}."
            )
        keep = np.ones(source.shape[0], dtype=bool)
        transform = np.eye(4, dtype=np.float64)
        for _ in range(4):
            transform = _fit_rigid_transform(source[keep], pose_points[keep])
            residual = np.linalg.norm(
                _transform_points(source, transform) - pose_points, axis=-1
            )
            median = float(np.median(residual[keep]))
            mad = float(np.median(np.abs(residual[keep] - median)))
            threshold = max(median + 6.0 * max(mad, 1e-4), 0.05)
            updated = residual <= threshold
            if updated.sum() < 1_000 or np.array_equal(updated, keep):
                break
            keep = updated
        residual = np.linalg.norm(
            _transform_points(source, transform) - pose_points, axis=-1
        )
        candidates.append(
            (
                float(np.median(residual)),
                float(np.percentile(residual, 90.0)),
                sign,
                transform,
            )
        )
    median_error, p90_error, sign, transform = min(
        candidates, key=lambda item: (item[0], item[1])
    )
    if median_error > PANDASET_CALIBRATION_MEDIAN_ERROR_MAX_M or (
        p90_error > PANDASET_CALIBRATION_P90_ERROR_MAX_M
    ):
        raise ValueError(
            "Recovered PandaSet Pandar64 calibration failed registration: "
            f"median={median_error:.4f} m, p90={p90_error:.4f} m; expected "
            f"<= {PANDASET_CALIBRATION_MEDIAN_ERROR_MAX_M:.3f} m and "
            f"<= {PANDASET_CALIBRATION_P90_ERROR_MAX_M:.3f} m."
        )
    return transform, int(sign), median_error, p90_error


def spherical_bins(
    points_repo: np.ndarray, *, width: int = PANDASET_WIDTH
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assign processed points to calibrated rows and wrapped azimuth columns."""
    width = int(width)
    if width <= 0:
        raise ValueError(f"PandaSet projection width must be positive, got {width}.")
    points_repo = np.asarray(points_repo, dtype=np.float64)
    distance = np.linalg.norm(points_repo, axis=-1)
    elevation = np.arcsin(
        np.divide(
            points_repo[:, 2],
            distance,
            out=np.zeros_like(distance),
            where=distance > 0,
        )
    )
    row = np.argmin(
        np.abs(elevation[:, None] - PANDAR64_ELEVATION_RAD[None, :]), axis=1
    ).astype(np.int64)
    azimuth = np.arctan2(points_repo[:, 1], points_repo[:, 0])
    column = np.floor(
        np.mod(azimuth + np.pi, 2.0 * np.pi)
        * (width / (2.0 * np.pi))
    ).astype(np.int64)
    column = np.clip(column, 0, width - 1)
    return row, column, distance


def project_processed_points(
    points_repo: np.ndarray,
    intensity: np.ndarray,
    *,
    min_range: float = PANDASET_MIN_RANGE,
    max_range: float = PANDASET_MAX_RANGE,
    width: int = PANDASET_WIDTH,
) -> tuple[np.ndarray, np.ndarray]:
    """Nearest-return z-buffer processed points into a 64-row range image."""
    width = int(width)
    row, column, distance = spherical_bins(points_repo, width=width)
    intensity = np.asarray(intensity, dtype=np.float32)
    valid = (
        np.isfinite(points_repo).all(axis=-1)
        & np.isfinite(distance)
        & np.isfinite(intensity)
        & (distance >= min_range)
        & (distance <= max_range)
    )
    row, column = row[valid], column[valid]
    distance = distance[valid].astype(np.float32)
    intensity = intensity[valid]

    # Descending range means the closest return is written last.
    order = np.argsort(-distance, kind="stable")
    range_map = np.zeros((PANDASET_HEIGHT, width), dtype=np.float32)
    intensity_map = np.zeros_like(range_map)
    range_map[row[order], column[order]] = distance[order]
    intensity_map[row[order], column[order]] = intensity[order]
    return range_map, intensity_map


def project_raw_pandar64(
    laser_id: np.ndarray,
    column_id: np.ndarray,
    distance: np.ndarray,
    intensity: np.ndarray,
    *,
    min_range: float = PANDASET_MIN_RANGE,
    max_range: float = PANDASET_MAX_RANGE,
    width: int = PANDASET_WIDTH,
) -> tuple[np.ndarray, np.ndarray]:
    """Project raw firings on the calibrated scan-ordered azimuth grid."""
    laser_id = np.asarray(laser_id, dtype=np.int64)
    column_id = np.asarray(column_id, dtype=np.int64)
    distance = np.asarray(distance, dtype=np.float32)
    intensity = np.asarray(intensity, dtype=np.float32)
    if not (
        laser_id.shape == column_id.shape == distance.shape == intensity.shape
    ):
        raise ValueError("PandaSet raw firing arrays must have identical shapes.")

    valid = (
        (laser_id >= 0)
        & (laser_id < PANDASET_HEIGHT)
        & np.isfinite(distance)
        & np.isfinite(intensity)
        & (distance >= min_range)
        & (distance <= max_range)
    )
    row = laser_id[valid]
    scan_column = raw_pandar64_scan_columns(laser_id, column_id)[valid]
    column = scan_column * int(width) // PANDASET_NATIVE_WIDTH
    distance, intensity = distance[valid], intensity[valid]

    order = np.argsort(-distance, kind="stable")
    range_map = np.zeros((PANDASET_HEIGHT, int(width)), dtype=np.float32)
    intensity_map = np.zeros_like(range_map)
    range_map[row[order], column[order]] = distance[order]
    intensity_map[row[order], column[order]] = intensity[order]
    return range_map, intensity_map


def build_pandaset_sequence_cache(
    sequence_dir: str | Path,
    *,
    packet_sequence_dir: str | Path | None = None,
    min_range: float = PANDASET_MIN_RANGE,
    max_range: float = PANDASET_MAX_RANGE,
) -> dict[str, np.ndarray]:
    """Build one physically registered cache from raw and processed PandaSet."""
    try:
        import pandas as pd
    except ImportError as error:
        raise ImportError(
            "Building PandaSet caches requires pandas; cache-backed training does not."
        ) from error

    sequence_dir = Path(sequence_dir)
    if packet_sequence_dir is None:
        raise ValueError(
            "A raw PandaSet Pandar64 packet directory is required. The processed "
            "release cannot recover firing rows, physical azimuths, or the "
            "unpublished Pandar64-to-pose calibration."
        )
    lidar_dir = sequence_dir / "lidar"
    with (lidar_dir / "poses.json").open("r") as handle:
        poses = json.load(handle)
    with (lidar_dir / "timestamps.json").open("r") as handle:
        pose_timestamps = np.asarray(json.load(handle), dtype=np.float64)
    positions, quaternions = pose_components_from_json(poses)
    if not (len(poses) == len(pose_timestamps)):
        raise ValueError(
            f"{sequence_dir.name}: {len(poses)} poses != "
            f"{len(pose_timestamps)} timestamps."
        )

    frame_paths = sorted(lidar_dir.glob("*.pkl.gz"))
    if not frame_paths:
        frame_paths = sorted(lidar_dir.glob("*.pkl"))
    if len(frame_paths) != len(poses):
        raise ValueError(
            f"{sequence_dir.name}: {len(frame_paths)} lidar frames != "
            f"{len(poses)} poses."
        )
    packet_sequence_dir = Path(packet_sequence_dir)
    packet_lidar_dir = (
        packet_sequence_dir / "lidar"
        if (packet_sequence_dir / "lidar").is_dir()
        else packet_sequence_dir
    )
    packet_paths = sorted(packet_lidar_dir.glob("*.pkl.gz"))
    if not packet_paths:
        packet_paths = sorted(packet_lidar_dir.glob("*.pkl"))
    if len(packet_paths) != len(frame_paths):
        raise ValueError(
            f"{sequence_dir.name}: {len(packet_paths)} raw packet frames != "
            f"{len(frame_paths)} processed frames."
        )

    raw_calibration = {-1: [], 1: []}
    pose_calibration = []
    azimuth_grid_columns = []
    azimuth_grid_angles_deg = []
    intensity_match_fractions = []
    # Frame 0 points precede the first available end-of-sweep pose and require
    # pose extrapolation.  That is adequate for cache metadata but can bias the
    # rigid sensor calibration on sequences that begin while accelerating.
    # Recover the fixed extrinsic only from sweeps bracketed by measured poses.
    calibration_start = 1 if len(frame_paths) > PANDASET_CALIBRATION_FRAMES else 0
    calibration_frame_indices = np.arange(
        calibration_start,
        min(
            calibration_start + PANDASET_CALIBRATION_FRAMES,
            len(frame_paths),
        ),
        dtype=np.int32,
    )
    calibration_frames = len(calibration_frame_indices)
    for frame_index in calibration_frame_indices:
        frame = pd.read_pickle(frame_paths[frame_index])
        required_processed = {"x", "y", "z", "i", "t", "d"}
        missing_processed = required_processed.difference(frame.columns)
        if missing_processed:
            raise ValueError(
                f"{frame_paths[frame_index]}: missing PandaSet columns "
                f"{sorted(missing_processed)}."
            )
        frame = frame.loc[frame["d"].to_numpy() == 0]
        packet = pd.read_pickle(packet_paths[frame_index])
        required_packet = {
            "laser_id",
            "column_id",
            "elevation",
            "azimuth_col_corrected",
            "distance",
            "intensity",
        }
        missing_packet = required_packet.difference(packet.columns)
        if missing_packet:
            raise ValueError(
                f"{packet_paths[frame_index]}: missing PandaSet raw columns "
                f"{sorted(missing_packet)}."
            )
        if len(frame) != len(packet):
            raise ValueError(
                f"{sequence_dir.name} frame {frame_index}: processed/raw Pandar64 "
                f"point counts differ ({len(frame)} != {len(packet)})."
            )
        processed_intensity = frame["i"].to_numpy(dtype=np.float64)
        packet_intensity = packet["intensity"].to_numpy(dtype=np.float64)
        intensity_match = np.isclose(
            processed_intensity, packet_intensity, atol=0.5, rtol=0.0
        )
        intensity_match_fractions.append(float(intensity_match.mean()))
        point_timestamps = resolve_point_timestamps(
            frame["t"].to_numpy(dtype=np.float64), pose_timestamps[frame_index]
        )
        pose_points = world_points_to_pose_frame(
            frame[["x", "y", "z"]].to_numpy(dtype=np.float64),
            point_timestamps,
            pose_timestamps,
            positions,
            quaternions,
        )
        packet_distance = packet["distance"].to_numpy(dtype=np.float64)
        select = (
            intensity_match
            & np.isfinite(pose_points).all(axis=-1)
            & np.isfinite(packet_distance)
            & (packet_distance >= 3.0)
            & (packet_distance <= 80.0)
        )
        selected = np.flatnonzero(select)
        if selected.size > PANDASET_CALIBRATION_MAX_POINTS_PER_FRAME:
            selected = selected[
                np.linspace(
                    0,
                    selected.size - 1,
                    PANDASET_CALIBRATION_MAX_POINTS_PER_FRAME,
                    dtype=np.int64,
                )
            ]
        if selected.size < 1_000:
            raise ValueError(
                f"{sequence_dir.name} frame {frame_index}: only {selected.size} "
                "usable raw/processed calibration correspondences."
            )
        pose_calibration.append(pose_points[selected])
        azimuth_grid_columns.append(
            raw_pandar64_scan_columns(
                packet["laser_id"].to_numpy(dtype=np.int64),
                packet["column_id"].to_numpy(dtype=np.int64),
            )
        )
        azimuth_grid_angles_deg.append(
            packet["azimuth_col_corrected"].to_numpy(dtype=np.float64)
        )
        for sign in (-1, 1):
            raw_calibration[sign].append(
                raw_pandar64_points(
                    packet_distance[selected],
                    packet["elevation"].to_numpy(dtype=np.float64)[selected],
                    packet["azimuth_col_corrected"].to_numpy(dtype=np.float64)[
                        selected
                    ],
                    azimuth_y_sign=sign,
                )
            )
    intensity_match_fraction = float(np.mean(intensity_match_fractions))
    if intensity_match_fraction < 0.95:
        raise ValueError(
            f"{sequence_dir.name}: raw/processed point-order validation failed; "
            f"only {intensity_match_fraction:.2%} of intensities match."
        )
    try:
        pandar64_to_pose, azimuth_y_sign, calibration_median, calibration_p90 = (
            recover_pandar64_to_pose(
                {
                    sign: np.concatenate(raw_calibration[sign], axis=0)
                    for sign in (-1, 1)
                },
                np.concatenate(pose_calibration, axis=0),
            )
        )
    except ValueError as error:
        raise ValueError(f"{sequence_dir.name}: {error}") from error
    azimuth_phase_rad, azimuth_column_sign, azimuth_grid_p90_error_deg = (
        recover_pandar64_azimuth_grid(
            np.concatenate(azimuth_grid_columns),
            np.concatenate(azimuth_grid_angles_deg),
        )
    )
    ray_directions = pandaset_ray_directions(
        PANDASET_WIDTH,
        azimuth_y_sign=azimuth_y_sign,
        azimuth_phase_rad=azimuth_phase_rad,
        azimuth_column_sign=azimuth_column_sign,
    )

    range_frames, intensity_frames = [], []
    start_timestamps, end_timestamps = [], []
    start_positions, start_quaternions = [], []
    end_positions, end_quaternions = [], []
    for frame_index, frame_path in enumerate(frame_paths):
        frame = pd.read_pickle(frame_path)
        required = {"x", "y", "z", "i", "t", "d"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(
                f"{frame_path}: missing PandaSet columns {sorted(missing)}."
            )
        frame = frame.loc[frame["d"].to_numpy() == 0]
        point_timestamps = resolve_point_timestamps(
            frame["t"].to_numpy(dtype=np.float64), pose_timestamps[frame_index]
        )
        finite_timestamps = point_timestamps[np.isfinite(point_timestamps)]
        if finite_timestamps.size == 0:
            start_timestamp = end_timestamp = pose_timestamps[frame_index]
        else:
            start_timestamp = float(finite_timestamps.min())
            end_timestamp = float(finite_timestamps.max())

        packet = pd.read_pickle(packet_paths[frame_index])
        required_packet = {
            "laser_id",
            "azimuth_col_corrected",
            "distance",
            "intensity",
        }
        missing_packet = required_packet.difference(packet.columns)
        if missing_packet:
            raise ValueError(
                f"{packet_paths[frame_index]}: missing PandaSet raw columns "
                f"{sorted(missing_packet)}."
            )
        range_map, intensity_map = project_raw_pandar64(
            packet["laser_id"].to_numpy(),
            packet["column_id"].to_numpy(),
            packet["distance"].to_numpy(),
            packet["intensity"].to_numpy(),
            min_range=min_range,
            max_range=max_range,
        )
        range_frames.append(range_map)
        intensity_frames.append(intensity_map)
        start_timestamps.append(start_timestamp)
        end_timestamps.append(end_timestamp)
        start_position, start_quaternion = interpolate_poses(
            np.asarray([start_timestamp]),
            pose_timestamps,
            positions,
            quaternions,
        )
        end_position, end_quaternion = interpolate_poses(
            np.asarray([end_timestamp]),
            pose_timestamps,
            positions,
            quaternions,
        )
        start_positions.append(start_position[0])
        start_quaternions.append(start_quaternion[0])
        end_positions.append(end_position[0])
        end_quaternions.append(end_quaternion[0])

    lidar_to_world = _poses_to_pandar64_lidar_to_world(
        np.asarray(start_positions),
        np.asarray(start_quaternions),
        pandar64_to_pose,
    )
    lidar_to_world_end = _poses_to_pandar64_lidar_to_world(
        np.asarray(end_positions),
        np.asarray(end_quaternions),
        pandar64_to_pose,
    )
    pose_sensor_start_end = (
        np.linalg.inv(lidar_to_world_end) @ lidar_to_world
    ).astype(np.float32)
    return {
        "cache_version": np.asarray(PANDASET_CACHE_VERSION),
        "projection_mode": np.asarray("raw_pandar64_calibrated_scan_order"),
        "geometry_calibration_source": np.asarray(
            "raw_processed_correspondence_rigid_fit"
        ),
        "pandar64_to_pose": pandar64_to_pose.astype(np.float32),
        "azimuth_y_sign": np.asarray(azimuth_y_sign, dtype=np.int8),
        "azimuth_phase_rad": np.asarray(azimuth_phase_rad, dtype=np.float64),
        "azimuth_column_sign": np.asarray(azimuth_column_sign, dtype=np.int8),
        "azimuth_grid_p90_error_deg": np.asarray(
            azimuth_grid_p90_error_deg, dtype=np.float32
        ),
        "calibration_frame_count": np.asarray(calibration_frames, dtype=np.int32),
        "calibration_frame_indices": calibration_frame_indices,
        "calibration_intensity_match_fraction": np.asarray(
            intensity_match_fraction, dtype=np.float32
        ),
        "calibration_median_error_m": np.asarray(
            calibration_median, dtype=np.float32
        ),
        "calibration_p90_error_m": np.asarray(calibration_p90, dtype=np.float32),
        "sequence": np.asarray(sequence_dir.name),
        "range": np.stack(range_frames, axis=0),
        "intensity": np.stack(intensity_frames, axis=0),
        "lidar_to_world": lidar_to_world,
        "pose_sensor_start_end": pose_sensor_start_end,
        "timestamp_start_us": np.rint(
            np.asarray(start_timestamps) * 1.0e6
        ).astype(np.int64),
        "timestamp_end_us": np.rint(
            np.asarray(end_timestamps) * 1.0e6
        ).astype(np.int64),
        "ray_directions": ray_directions,
        "elevation_rad": PANDAR64_ELEVATION_RAD.astype(np.float32),
        "min_range": np.asarray(min_range, dtype=np.float32),
        "max_range": np.asarray(max_range, dtype=np.float32),
        "height": np.asarray(PANDASET_HEIGHT, dtype=np.int32),
        "width": np.asarray(PANDASET_WIDTH, dtype=np.int32),
    }


def validate_pandaset_cache(
    payload: Any, *, path: str | Path = "<cache>"
) -> None:
    """Reject stale or incompatible PandaSet caches."""
    required = {
        "cache_version",
        "projection_mode",
        "geometry_calibration_source",
        "pandar64_to_pose",
        "azimuth_y_sign",
        "azimuth_phase_rad",
        "azimuth_column_sign",
        "azimuth_grid_p90_error_deg",
        "calibration_frame_count",
        "calibration_frame_indices",
        "calibration_intensity_match_fraction",
        "calibration_median_error_m",
        "calibration_p90_error_m",
        "range",
        "intensity",
        "lidar_to_world",
        "pose_sensor_start_end",
        "timestamp_start_us",
        "timestamp_end_us",
        "ray_directions",
        "min_range",
        "max_range",
        "height",
        "width",
    }
    missing = required.difference(payload.files)
    if missing:
        raise ValueError(f"{path}: PandaSet cache is missing {sorted(missing)}.")
    version = str(np.asarray(payload["cache_version"]).item())
    if version != PANDASET_CACHE_VERSION:
        raise ValueError(
            f"{path}: stale PandaSet cache version {version!r}; "
            f"expected {PANDASET_CACHE_VERSION!r}."
        )
    projection_mode = str(np.asarray(payload["projection_mode"]).item())
    if projection_mode != "raw_pandar64_calibrated_scan_order":
        raise ValueError(f"{path}: unknown projection mode {projection_mode!r}.")
    calibration_source = str(
        np.asarray(payload["geometry_calibration_source"]).item()
    )
    if calibration_source != "raw_processed_correspondence_rigid_fit":
        raise ValueError(
            f"{path}: unsupported PandaSet calibration source "
            f"{calibration_source!r}."
        )
    shape = tuple(payload["range"].shape)
    if len(shape) != 3 or shape[1:] != (PANDASET_HEIGHT, PANDASET_WIDTH):
        raise ValueError(
            f"{path}: expected range shape (F, {PANDASET_HEIGHT}, "
            f"{PANDASET_WIDTH}), got {shape}."
        )
    if payload["intensity"].shape != payload["range"].shape:
        raise ValueError(f"{path}: intensity/range shape mismatch.")
    min_range = float(np.asarray(payload["min_range"]).item())
    max_range = float(np.asarray(payload["max_range"]).item())
    if not np.isclose(min_range, PANDASET_MIN_RANGE) or not np.isclose(
        max_range, PANDASET_MAX_RANGE
    ):
        raise ValueError(
            f"{path}: expected PandaSet bounds "
            f"[{PANDASET_MIN_RANGE}, {PANDASET_MAX_RANGE}] m, got "
            f"[{min_range}, {max_range}] m."
        )
    frame_count = shape[0]
    for key in (
        "lidar_to_world",
        "pose_sensor_start_end",
    ):
        if payload[key].shape != (frame_count, 4, 4):
            raise ValueError(f"{path}: invalid {key} shape {payload[key].shape}.")
    if payload["ray_directions"].shape != (
        PANDASET_HEIGHT,
        PANDASET_WIDTH,
        3,
    ):
        raise ValueError(
            f"{path}: invalid ray_directions shape "
            f"{payload['ray_directions'].shape}."
        )
    azimuth_y_sign = int(np.asarray(payload["azimuth_y_sign"]).item())
    azimuth_phase_rad = float(np.asarray(payload["azimuth_phase_rad"]).item())
    azimuth_column_sign = int(
        np.asarray(payload["azimuth_column_sign"]).item()
    )
    if not np.isfinite(azimuth_phase_rad):
        raise ValueError(f"{path}: azimuth_phase_rad is non-finite.")
    expected_rays = pandaset_ray_directions(
        PANDASET_WIDTH,
        azimuth_y_sign=azimuth_y_sign,
        azimuth_phase_rad=azimuth_phase_rad,
        azimuth_column_sign=azimuth_column_sign,
    )
    ray_error = float(
        np.max(np.abs(np.asarray(payload["ray_directions"]) - expected_rays))
    )
    if ray_error > 1e-6:
        raise ValueError(
            f"{path}: cached PandaSet rays do not match the calibrated "
            f"scan-ordered azimuth contract (max error {ray_error})."
        )
    azimuth_grid_p90_error_deg = float(
        np.asarray(payload["azimuth_grid_p90_error_deg"]).item()
    )
    if azimuth_grid_p90_error_deg > PANDASET_AZIMUTH_GRID_P90_ERROR_MAX_DEG:
        raise ValueError(
            f"{path}: PandaSet azimuth-grid p90 error is "
            f"{azimuth_grid_p90_error_deg:.4f} deg."
        )
    if payload["pandar64_to_pose"].shape != (4, 4):
        raise ValueError(
            f"{path}: invalid pandar64_to_pose shape "
            f"{payload['pandar64_to_pose'].shape}."
        )
    pandar64_to_pose = np.asarray(payload["pandar64_to_pose"], dtype=np.float64)
    rotation = pandar64_to_pose[:3, :3]
    if (
        not np.isfinite(pandar64_to_pose).all()
        or not np.allclose(pandar64_to_pose[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6)
        or not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-5)
        or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5)
    ):
        raise ValueError(f"{path}: pandar64_to_pose is not a proper SE(3) transform.")
    calibration_frame_count = int(
        np.asarray(payload["calibration_frame_count"]).item()
    )
    if calibration_frame_count < 1:
        raise ValueError(f"{path}: calibration_frame_count must be positive.")
    calibration_frame_indices = np.asarray(
        payload["calibration_frame_indices"], dtype=np.int32
    )
    if (
        calibration_frame_indices.shape != (calibration_frame_count,)
        or np.any(calibration_frame_indices < 0)
        or np.any(np.diff(calibration_frame_indices) <= 0)
    ):
        raise ValueError(
            f"{path}: invalid calibration_frame_indices "
            f"{calibration_frame_indices.tolist()}."
        )
    median_error = float(np.asarray(payload["calibration_median_error_m"]).item())
    p90_error = float(np.asarray(payload["calibration_p90_error_m"]).item())
    intensity_match = float(
        np.asarray(payload["calibration_intensity_match_fraction"]).item()
    )
    if median_error > PANDASET_CALIBRATION_MEDIAN_ERROR_MAX_M or (
        p90_error > PANDASET_CALIBRATION_P90_ERROR_MAX_M
    ):
        raise ValueError(
            f"{path}: failed PandaSet calibration residuals "
            f"(median={median_error:.4f} m, p90={p90_error:.4f} m)."
        )
    if intensity_match < 0.95:
        raise ValueError(
            f"{path}: raw/processed intensity correspondence is only "
            f"{intensity_match:.2%}."
        )


class PandaSetCachedLidar:
    """Cache-backed low-level reader matching the NCore reader contract."""

    MAX_CACHED_SEQUENCES = 2

    def __init__(
        self,
        root_path: str,
        filter_list_path: Optional[str] = None,
        num_views: int = -1,
        metadata_folder: Optional[str] = None,
        custom_folders: Optional[list[str]] = None,
        custom_fields: Optional[list[str]] = None,
        view_indices: Optional[list[int]] = None,
        load_intensity: bool = False,
    ):
        del metadata_folder, custom_folders, view_indices
        self.root_path = root_path
        self.num_views = num_views
        self.n_views = num_views
        self.custom_fields = custom_fields if custom_fields is not None else ["lidar"]
        self.load_intensity = bool(load_intensity)
        self.filter_set = self._build_filter_set(filter_list_path)
        if not self.filter_set:
            raise RuntimeError(
                f"No PandaSet caches found (root={root_path}, "
                f"filter_list_path={filter_list_path})."
            )
        self._cache: dict[str, dict[str, np.ndarray]] = {}

    def _build_filter_set(self, filter_list_path: Optional[str]) -> list[str]:
        if filter_list_path and os.path.isfile(filter_list_path):
            with open(filter_list_path, "r") as handle:
                return [
                    Path(line.strip()).stem
                    for line in handle
                    if line.strip()
                ]
        root = Path(self.root_path)
        return sorted(path.stem for path in root.glob("*.npz"))

    def _load(self, stem: str) -> dict[str, np.ndarray]:
        if stem in self._cache:
            return self._cache[stem]
        path = Path(self.root_path) / f"{stem}.npz"
        with np.load(path, allow_pickle=False) as payload:
            validate_pandaset_cache(payload, path=path)
            cache = {key: np.asarray(payload[key]) for key in payload.files}
        if self._cache:
            reference = next(iter(self._cache.values()))["ray_directions"]
            ray_error = float(np.max(np.abs(cache["ray_directions"] - reference)))
            if ray_error > 1e-6:
                raise ValueError(
                    f"{path}: PandaSet azimuth handedness differs from the other "
                    f"loaded sequence (ray max error {ray_error})."
                )
        if len(self._cache) >= self.MAX_CACHED_SEQUENCES:
            self._cache.pop(next(iter(self._cache)))
        self._cache[stem] = cache
        return cache

    def sensor_ray_directions(self) -> np.ndarray:
        cache = self._load(self.filter_set[0])
        return cache["ray_directions"].astype(np.float32, copy=False)

    def size_of_filter_set(self) -> int:
        return len(self.filter_set)

    @staticmethod
    def _clamp_frame_indices(
        frame_indices: np.ndarray, frame_count: int
    ) -> np.ndarray:
        frame_indices = np.asarray(frame_indices, dtype=np.int64).copy()
        out_of_bounds = (frame_indices < 0) | (frame_indices >= frame_count)
        if out_of_bounds.any():
            frame_indices[out_of_bounds] = np.random.randint(
                0, frame_count, size=int(out_of_bounds.sum())
            )
        return frame_indices

    def _read_rgb_and_lidar(self, lidar_idx, lidar_frame_idx, data_fields):
        stem = self.filter_set[int(lidar_idx)]
        cache = self._load(stem)
        frame_indices = self._clamp_frame_indices(
            lidar_frame_idx, cache["range"].shape[0]
        )
        output: dict[str, Any] = {
            "__key__": stem,
            "timestamp_list": cache["timestamp_end_us"][frame_indices],
            "pose_sensor_start_end": cache["pose_sensor_start_end"][frame_indices],
            "lidar_to_world": cache["lidar_to_world"][frame_indices],
            "rgb_frame_indices": frame_indices.tolist(),
        }
        for data_field in data_fields:
            if data_field != "custom":
                raise NotImplementedError(
                    f"PandaSetCachedLidar does not handle data field {data_field}."
                )
            custom = {"lidar": cache["range"][frame_indices]}
            if self.load_intensity:
                custom["lidar_intensity"] = cache["intensity"][frame_indices]
            output[data_field] = custom
        return output
