# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact track-ID join for legacy Waymo HDMap box dumps."""

from __future__ import annotations

from typing import Any

import numpy as np


_NCORE_CLASS_TO_BOX_TYPE = {
    "vehicle": 0,
    "pedestrian": 1,
    "cyclist": 2,
}


def resolve_waymo_box_track_ids_from_ncore(
    *,
    hdmap_composer,
    sequence_loader,
    lidar_sensor,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Attach exact NCore cuboid track IDs to legacy HDMap box rows."""
    box_frame = np.asarray(hdmap_composer.box_frame, dtype=np.int64)
    existing = np.asarray(hdmap_composer.box_track_id, dtype=np.int64)
    if existing.shape != box_frame.shape:
        raise ValueError(
            "box_track_id must align with box_frame, got "
            f"{existing.shape} and {box_frame.shape}"
        )
    if existing.size == 0 or np.all(existing >= 0):
        return existing.copy(), {
            "track_id_source": "hdmap_npz",
            "matched_box_rows": int(existing.shape[0]),
            "max_box_center_error_m": 0.0,
            "max_box_size_error_m": 0.0,
        }

    from ncore.impl.data.v4.components import CuboidsComponent

    component_reader = getattr(sequence_loader, "_reader", None)
    if component_reader is None:
        raise ValueError(
            "the NCore sequence loader does not expose its component reader"
        )
    cuboid_readers = component_reader.open_component_readers(
        CuboidsComponent.Reader
    )
    if "default" not in cuboid_readers:
        raise ValueError(
            "the NCore sequence has no default cuboid component; exact dynamic "
            "actor composition is unavailable"
        )
    observations = list(cuboid_readers["default"].get_observations())

    frame_timestamps = np.asarray(
        lidar_sensor.get_frames_timestamps_us(), dtype=np.int64
    )
    timestamp_to_frame = {
        int(timestamp): frame_index
        for frame_index, timestamp in enumerate(frame_timestamps)
    }
    frame_indices = np.arange(frame_timestamps.shape[0], dtype=np.int64)
    lidar_to_world = np.asarray(
        lidar_sensor.get_frames_T_sensor_target("world", frame_indices),
        dtype=np.float64,
    )
    sensor_id = str(lidar_sensor.sensor_id)

    candidates: dict[
        tuple[int, int], list[tuple[str, np.ndarray, np.ndarray]]
    ] = {}
    for observation in observations:
        class_name = str(observation.class_id).split(".")[-1].lower()
        box_type = _NCORE_CLASS_TO_BOX_TYPE.get(class_name)
        if box_type is None:
            continue
        if str(observation.reference_frame_id) != sensor_id:
            raise ValueError(
                "expected NCore cuboids in the LiDAR frame, got "
                f"{observation.reference_frame_id!r}"
            )
        frame_index = timestamp_to_frame.get(
            int(observation.reference_frame_timestamp_us)
        )
        if frame_index is None:
            continue
        pose = lidar_to_world[frame_index]
        center_sensor = np.asarray(
            observation.bbox3.centroid, dtype=np.float64
        )
        center_world = center_sensor @ pose[:3, :3].T + pose[:3, 3]
        candidates.setdefault((frame_index, box_type), []).append(
            (
                str(observation.track_id),
                center_world,
                np.asarray(observation.bbox3.dim, dtype=np.float64),
            )
        )

    box_o2w = np.asarray(hdmap_composer.box_o2w, dtype=np.float64)
    box_lwh = np.asarray(hdmap_composer.box_lwh, dtype=np.float64)
    box_type = np.asarray(hdmap_composer.box_type, dtype=np.int64)
    matched_track_names: list[str | None] = [None] * box_frame.shape[0]
    center_errors: list[float] = []
    size_errors: list[float] = []

    for key in sorted(set(zip(box_frame.tolist(), box_type.tolist()))):
        rows = np.nonzero(
            (box_frame == key[0]) & (box_type == key[1])
        )[0]
        frame_candidates = candidates.get(key, [])
        if len(frame_candidates) != int(rows.shape[0]):
            raise ValueError(
                "NCore cuboids do not align with HDMap boxes for "
                f"frame/type={key}: boxes={rows.shape[0]}, "
                f"cuboids={len(frame_candidates)}"
            )

        candidate_centers = np.stack(
            [candidate[1] for candidate in frame_candidates]
        )
        candidate_sizes = np.stack(
            [candidate[2] for candidate in frame_candidates]
        )
        row_centers = box_o2w[rows, :3, 3]
        row_sizes = box_lwh[rows]
        center_distance = np.linalg.norm(
            row_centers[:, None] - candidate_centers[None], axis=-1
        )
        size_distance = np.linalg.norm(
            row_sizes[:, None] - candidate_sizes[None], axis=-1
        )
        cost = center_distance + size_distance

        row_assignment = np.full(rows.shape[0], -1, dtype=np.int64)
        used_candidates = np.zeros(len(frame_candidates), dtype=bool)
        for flat_index in np.argsort(cost, axis=None):
            local_row, candidate_index = np.unravel_index(
                int(flat_index), cost.shape
            )
            if (
                row_assignment[local_row] >= 0
                or used_candidates[candidate_index]
            ):
                continue
            row_assignment[local_row] = candidate_index
            used_candidates[candidate_index] = True
        if (row_assignment < 0).any():
            raise RuntimeError(
                f"failed to join all HDMap boxes for frame/type={key}"
            )

        for local_row, candidate_index in enumerate(row_assignment):
            center_error = float(
                center_distance[local_row, candidate_index]
            )
            size_error = float(size_distance[local_row, candidate_index])
            if center_error > 0.05 or size_error > 0.01:
                raise ValueError(
                    "NCore cuboid does not exactly match its HDMap box: "
                    f"frame/type={key}, center_error={center_error:.6f}m, "
                    f"size_error={size_error:.6f}m"
                )
            matched_track_names[int(rows[local_row])] = frame_candidates[
                int(candidate_index)
            ][0]
            center_errors.append(center_error)
            size_errors.append(size_error)

    if any(track_name is None for track_name in matched_track_names):
        raise RuntimeError("not every HDMap box received an NCore track ID")
    unique_names = sorted(set(matched_track_names))
    name_to_id = {
        track_name: track_id
        for track_id, track_name in enumerate(unique_names)
    }
    resolved = np.asarray(
        [name_to_id[track_name] for track_name in matched_track_names],
        dtype=np.int64,
    )
    return resolved, {
        "track_id_source": "ncore_cuboids_exact_geometry_join",
        "matched_box_rows": int(resolved.shape[0]),
        "cuboid_tracks": int(len(unique_names)),
        "max_box_center_error_m": max(center_errors, default=0.0),
        "max_box_size_error_m": max(size_errors, default=0.0),
    }
