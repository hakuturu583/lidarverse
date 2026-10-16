# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""On-the-fly HDMap range-image composer for the Waymo NCore dataloader.

Consumes the per-clip NPZ written by ``scripts/data/dump_waymo_hdmap.py`` (static map
polylines pre-sampled to world-frame points + per-frame dynamic boxes) and
rasterizes a single-channel metric-depth (range) image for a given frame, using
the SAME beam grid + uniform-azimuth rebin as the lidar range image
(``WaymoNCoreLidar._scatter_frame``), so hdmap and lidar pixels line up.

Projection (world point -> range pixel), matching the lidar grid convention:
  p_s   = R^T (p_w - t)                 # world -> sweep-start sensor frame
  r     = ||p_s||                        # metric range
  row   = argmin_r |elev[r] - asin(z/r)| # nearest beam (elevation table)
  az    = atan2(y, x) - row_offset[row]  # per-row azimuth offset (row-offset model)
  col   = round((az - azim0)/daz) % C    # native column (uniform azimuths)
          then (col * out_width)//C      # SAME rebin the lidar scatter uses
On a (row,col) collision the NEAREST return wins (scatter-min over range).

All static map points are pose-independent; only the ego ``lidar_to_world`` (and
per-frame box poses) change with the frame.
"""

from typing import Optional, Sequence, Tuple

import numpy as np
import torch

# ----------------------------------------------------------------------------- #
# Box wireframe geometry
# ----------------------------------------------------------------------------- #
# Corner order: bit index = 4*xi + 2*yi + zi  (xi,yi,zi in {0:+, 1:-})
_BOX_SIGNS = np.array(
    [[sx, sy, sz] for sx in (1.0, -1.0) for sy in (1.0, -1.0) for sz in (1.0, -1.0)],
    dtype=np.float64,
)  # (8, 3)
_BOX_EDGES = [
    (i, j) for i in range(8) for j in range(i + 1, 8) if bin(i ^ j).count("1") == 1
]  # 12 edges


def sample_box_edges(o2w: np.ndarray, lwh: np.ndarray, spacing: float) -> np.ndarray:
    """Sample the 12 wireframe edges of one box to world-frame points (P, 3)."""
    half = 0.5 * np.asarray(lwh, dtype=np.float64)  # (3,)
    corners = _BOX_SIGNS * half[None, :]  # (8, 3) object frame
    pts = []
    for i, j in _BOX_EDGES:
        a, b = corners[i], corners[j]
        L = float(np.linalg.norm(b - a))
        n = max(int(np.ceil(L / max(spacing, 1e-3))), 2)
        ts = np.linspace(0.0, 1.0, n)[:, None]
        pts.append(a[None, :] + ts * (b - a)[None, :])
    obj_pts = np.concatenate(pts, axis=0)  # (P, 3)
    return obj_pts @ o2w[:3, :3].T + o2w[:3, 3]


# ----------------------------------------------------------------------------- #
# Projection
# ----------------------------------------------------------------------------- #
def project_world_points(
    pts_world: np.ndarray,
    lidar_to_world: np.ndarray,
    elev: np.ndarray,
    azim_native: np.ndarray,
    row_off: np.ndarray,
    C_native: int,
    out_width: int,
    min_range: float,
    max_range: float,
    max_horizontal: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """World points -> (rows, cols, ranges) on the (R, out_width) range grid.

    ``max_horizontal`` (m), if given, additionally drops points whose BEV (xy)
    distance exceeds it -- a natural "map crop" radius for the lane graph."""
    if pts_world.shape[0] == 0:
        z = np.zeros((0,), np.int64)
        return z, z, np.zeros((0,), np.float32)
    Rm = lidar_to_world[:3, :3].astype(np.float64)
    t = lidar_to_world[:3, 3].astype(np.float64)
    ps = (pts_world.astype(np.float64) - t[None, :]) @ Rm  # == R^T (p - t)
    r = np.linalg.norm(ps, axis=1)
    keep = (r >= min_range) & (r <= max_range)
    if max_horizontal is not None:
        keep = keep & (np.hypot(ps[:, 0], ps[:, 1]) <= max_horizontal)
    ps, r = ps[keep], r[keep]
    if ps.shape[0] == 0:
        z = np.zeros((0,), np.int64)
        return z, z, np.zeros((0,), np.float32)
    el = np.arcsin(np.clip(ps[:, 2] / r, -1.0, 1.0))
    # Memory-bounded nearest-beam lookup. The old ``(P,R)`` broadcast costs
    # hundreds of MB for an all-history union (up to 6*64*1024 points).
    elev_order = np.argsort(elev, kind="stable")
    elev_sorted = np.asarray(elev)[elev_order]
    ei = np.searchsorted(elev_sorted, el)
    ei0 = np.clip(ei - 1, 0, elev_sorted.shape[0] - 1)
    ei1 = np.clip(ei, 0, elev_sorted.shape[0] - 1)
    ed0 = np.abs(el - elev_sorted[ei0])
    ed1 = np.abs(el - elev_sorted[ei1])
    nearest_ei = np.where(
        ed0 < ed1,
        ei0,
        np.where(
            ed1 < ed0,
            ei1,
            # Preserve np.argmin's original-index tie break.
            np.where(elev_order[ei0] <= elev_order[ei1], ei0, ei1),
        ),
    )
    row = elev_order[nearest_ei]

    # Native column = the column whose azimuth best matches (point az - row off).
    # The azimuth table is ROLLED (starts mid-circle, wraps through +/-pi), so a
    # linear (az - az0)/daz inversion is invalid; use a circular nearest search.
    def _wrap(a):
        return (a + np.pi) % (2.0 * np.pi) - np.pi

    az = _wrap(np.arctan2(ps[:, 1], ps[:, 0]) - row_off[row])  # (P,)
    azn = _wrap(np.asarray(azim_native))  # (C,)
    order = np.argsort(azn, kind="stable")
    azn_sorted = azn[order]
    i = np.searchsorted(azn_sorted, az)
    i0 = (i - 1) % C_native
    i1 = i % C_native
    d0 = np.abs(_wrap(az - azn_sorted[i0]))
    d1 = np.abs(_wrap(az - azn_sorted[i1]))
    col_native = order[np.where(d0 <= d1, i0, i1)]
    col = np.minimum((col_native * out_width) // C_native, out_width - 1)
    return row.astype(np.int64), col.astype(np.int64), r.astype(np.float32)


def scatter_min_range(
    R: int, out_width: int, rows: np.ndarray, cols: np.ndarray, ranges: np.ndarray
) -> np.ndarray:
    """Scatter ranges into (R, out_width); nearest (smallest) wins on collision."""
    rmap = np.zeros((R, out_width), dtype=np.float32)
    if rows.shape[0] == 0:
        return rmap
    order = np.argsort(-ranges, kind="stable")  # write smallest LAST -> wins
    rmap[rows[order], cols[order]] = ranges[order]
    return rmap


# ----------------------------------------------------------------------------- #
# Semantic channel layout
# ----------------------------------------------------------------------------- #
# Channel order for the multi-channel rasterizer. Static map classes first
# (indexed by NPZ ``map_type`` 0..3), then dynamic box classes (NPZ
# ``box_type`` 0..2). One range-image channel per class; 0 == empty.
MAP_CLASS_NAMES = ["lane", "road_line", "road_edge", "crosswalk"]  # map_type 0..3
BOX_CLASS_NAMES = ["vehicle", "pedestrian", "cyclist"]  # box_type 0..2
CLASS_NAMES = MAP_CLASS_NAMES + BOX_CLASS_NAMES  # 7 channels
NUM_CLASSES = len(CLASS_NAMES)


# ----------------------------------------------------------------------------- #
# Composer
# ----------------------------------------------------------------------------- #
class WaymoHDMapComposer:
    """Per-clip HDMap rasterizer loaded from ``<hdmap_root>/<clip_id>.npz``."""

    def __init__(
        self,
        npz_path: str,
        elev: np.ndarray,
        azim_native: np.ndarray,
        row_off: np.ndarray,
        C_native: int,
        box_edge_spacing: float = 0.15,
        map_max_dist: Optional[float] = None,
    ):
        d = np.load(npz_path, allow_pickle=False)
        self.map_points = d["map_points"].astype(np.float32)  # (M,3) world
        self.map_type = d["map_type"].astype(np.int8)
        self.box_frame = d["box_frame"].astype(np.int64)  # (K,)
        self.box_o2w = d["box_o2w"].astype(np.float32)  # (K,4,4)
        self.box_lwh = d["box_lwh"].astype(np.float32)  # (K,3)
        self.box_type = d["box_type"].astype(np.int8)
        self.box_moving = d["box_moving"].astype(bool)
        self.box_track_id = (
            d["box_track_id"].astype(np.int64)
            if "box_track_id" in d.files
            else np.full(self.box_frame.shape, -1, dtype=np.int64)
        )
        self.box_extrapolated = np.zeros(self.box_frame.shape, dtype=bool)
        self.num_frames = int(d["num_frames"])
        self.vehicle_to_world_f0 = d["vehicle_to_world_f0"].astype(np.float32)

        self.elev = np.asarray(elev, dtype=np.float64)
        self.azim_native = np.asarray(azim_native, dtype=np.float64)
        self.row_off = np.asarray(row_off, dtype=np.float64)
        self.C_native = int(C_native)
        self.R = int(self.elev.shape[0])
        self.box_edge_spacing = float(box_edge_spacing)
        self.map_max_dist = None if map_max_dist is None else float(map_max_dist)

    def persist_static_box_tracks_across_timeline(
        self,
        track_ids: np.ndarray,
        through_frame: Optional[int] = None,
    ) -> dict[str, int | str]:
        """Hold every non-moving track at its latest pose for all frames."""
        track_ids = np.asarray(track_ids, dtype=np.int64)
        if track_ids.shape != self.box_frame.shape:
            raise ValueError(
                "track_ids must align with box rows, got "
                f"{track_ids.shape} and {self.box_frame.shape}"
            )
        if np.any(track_ids < 0):
            raise ValueError("static box persistence requires valid track IDs")
        target_frame = (
            self.num_frames - 1
            if through_frame is None
            else int(through_frame)
        )
        if not 0 <= target_frame < self.num_frames:
            raise ValueError(
                f"through_frame must lie in [0,{self.num_frames - 1}], "
                f"got {target_frame}"
            )

        keep = np.ones(self.box_frame.shape[0], dtype=bool)
        persistent_frame = []
        persistent_o2w = []
        persistent_lwh = []
        persistent_type = []
        persistent_moving = []
        persistent_track_id = []
        persistent_extrapolated = []
        static_tracks = 0
        original_static_rows = 0
        filled_rows = 0
        for track_id in np.unique(track_ids):
            rows = np.nonzero(track_ids == int(track_id))[0]
            if self.box_moving[rows].any():
                continue
            static_tracks += 1
            original_static_rows += int(rows.shape[0])
            keep[rows] = False
            order = np.argsort(self.box_frame[rows], kind="stable")
            canonical_row = int(rows[order[-1]])
            annotated_frames = set(
                int(frame) for frame in self.box_frame[rows].tolist()
            )
            for frame_idx in range(target_frame + 1):
                persistent_frame.append(frame_idx)
                persistent_o2w.append(self.box_o2w[canonical_row])
                persistent_lwh.append(self.box_lwh[canonical_row])
                persistent_type.append(self.box_type[canonical_row])
                persistent_moving.append(False)
                persistent_track_id.append(int(track_id))
                extrapolated = frame_idx not in annotated_frames
                persistent_extrapolated.append(extrapolated)
                filled_rows += int(extrapolated)

        if persistent_frame:
            self.box_frame = np.concatenate(
                [
                    self.box_frame[keep],
                    np.asarray(persistent_frame, dtype=np.int64),
                ]
            )
            self.box_o2w = np.concatenate(
                [
                    self.box_o2w[keep],
                    np.stack(persistent_o2w).astype(np.float32),
                ],
                axis=0,
            )
            self.box_lwh = np.concatenate(
                [
                    self.box_lwh[keep],
                    np.stack(persistent_lwh).astype(np.float32),
                ],
                axis=0,
            )
            self.box_type = np.concatenate(
                [
                    self.box_type[keep],
                    np.asarray(persistent_type, dtype=self.box_type.dtype),
                ]
            )
            self.box_moving = np.concatenate(
                [
                    self.box_moving[keep],
                    np.asarray(persistent_moving, dtype=bool),
                ]
            )
            self.box_track_id = np.concatenate(
                [
                    track_ids[keep],
                    np.asarray(persistent_track_id, dtype=np.int64),
                ]
            )
            self.box_extrapolated = np.concatenate(
                [
                    self.box_extrapolated[keep],
                    np.asarray(persistent_extrapolated, dtype=bool),
                ]
            )
            order = np.lexsort((self.box_track_id, self.box_frame))
            self.box_frame = self.box_frame[order]
            self.box_o2w = self.box_o2w[order]
            self.box_lwh = self.box_lwh[order]
            self.box_type = self.box_type[order]
            self.box_moving = self.box_moving[order]
            self.box_track_id = self.box_track_id[order]
            self.box_extrapolated = self.box_extrapolated[order]
        else:
            self.box_track_id = track_ids.copy()

        return {
            "policy": "static_latest_pose_hold_full_timeline",
            "through_frame": target_frame,
            "static_tracks": static_tracks,
            "original_static_rows": original_static_rows,
            "persistent_static_rows": len(persistent_frame),
            "filled_static_rows": filled_rows,
        }

    def extrapolate_box_tracks_forward(
        self,
        track_ids: np.ndarray,
        through_frame: Optional[int] = None,
    ) -> dict[str, int | str]:
        """Extend every box track after its final annotated frame."""
        track_ids = np.asarray(track_ids, dtype=np.int64)
        if track_ids.shape != self.box_frame.shape:
            raise ValueError(
                "track_ids must align with box rows, got "
                f"{track_ids.shape} and {self.box_frame.shape}"
            )
        if np.any(track_ids < 0):
            raise ValueError("box extrapolation requires valid track IDs")
        target_frame = (
            self.num_frames - 1
            if through_frame is None
            else int(through_frame)
        )
        if not 0 <= target_frame < self.num_frames:
            raise ValueError(
                f"through_frame must lie in [0,{self.num_frames - 1}], "
                f"got {target_frame}"
            )

        new_frame = []
        new_o2w = []
        new_lwh = []
        new_type = []
        new_moving = []
        new_track_id = []
        linear_tracks = 0
        held_tracks = 0
        single_observation_tracks = 0
        for track_id in np.unique(track_ids):
            rows = np.nonzero(track_ids == int(track_id))[0]
            order = np.argsort(self.box_frame[rows], kind="stable")
            rows = rows[order]
            last_row = int(rows[-1])
            last_frame = int(self.box_frame[last_row])
            if last_frame >= target_frame:
                continue

            last_pose = self.box_o2w[last_row].astype(np.float64)
            track_moving = bool(self.box_moving[rows].any())
            if rows.shape[0] >= 2 and track_moving:
                previous_row = int(rows[-2])
                previous_frame = int(self.box_frame[previous_row])
                frame_delta = last_frame - previous_frame
                if frame_delta <= 0:
                    raise ValueError(
                        f"track {track_id} has non-increasing box frames"
                    )
                translation_per_frame = (
                    last_pose[:3, 3]
                    - self.box_o2w[previous_row, :3, 3].astype(np.float64)
                ) / frame_delta
                previous_rotation = self.box_o2w[
                    previous_row, :3, :3
                ].astype(np.float64)
                relative_rotation = last_pose[:3, :3] @ previous_rotation.T
                yaw_per_frame = np.arctan2(
                    relative_rotation[1, 0], relative_rotation[0, 0]
                ) / frame_delta
                linear_tracks += 1
            else:
                translation_per_frame = np.zeros(3, dtype=np.float64)
                yaw_per_frame = 0.0
                held_tracks += 1
                single_observation_tracks += int(rows.shape[0] == 1)

            for frame_idx in range(last_frame + 1, target_frame + 1):
                elapsed = frame_idx - last_frame
                yaw = elapsed * yaw_per_frame
                c, s = np.cos(yaw), np.sin(yaw)
                yaw_rotation = np.asarray(
                    [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
                    dtype=np.float64,
                )
                pose = last_pose.copy()
                pose[:3, :3] = yaw_rotation @ last_pose[:3, :3]
                pose[:3, 3] = (
                    last_pose[:3, 3] + elapsed * translation_per_frame
                )
                new_frame.append(frame_idx)
                new_o2w.append(pose.astype(np.float32))
                new_lwh.append(self.box_lwh[last_row])
                new_type.append(self.box_type[last_row])
                new_moving.append(track_moving)
                new_track_id.append(int(track_id))

        original_rows = int(self.box_frame.shape[0])
        if new_frame:
            self.box_frame = np.concatenate(
                [self.box_frame, np.asarray(new_frame, dtype=np.int64)]
            )
            self.box_o2w = np.concatenate(
                [self.box_o2w, np.stack(new_o2w).astype(np.float32)], axis=0
            )
            self.box_lwh = np.concatenate(
                [self.box_lwh, np.stack(new_lwh).astype(np.float32)], axis=0
            )
            self.box_type = np.concatenate(
                [self.box_type, np.asarray(new_type, dtype=self.box_type.dtype)]
            )
            self.box_moving = np.concatenate(
                [self.box_moving, np.asarray(new_moving, dtype=bool)]
            )
            self.box_track_id = np.concatenate(
                [track_ids, np.asarray(new_track_id, dtype=np.int64)]
            )
            self.box_extrapolated = np.concatenate(
                [
                    self.box_extrapolated,
                    np.ones(len(new_frame), dtype=bool),
                ]
            )
            order = np.lexsort((self.box_track_id, self.box_frame))
            self.box_frame = self.box_frame[order]
            self.box_o2w = self.box_o2w[order]
            self.box_lwh = self.box_lwh[order]
            self.box_type = self.box_type[order]
            self.box_moving = self.box_moving[order]
            self.box_track_id = self.box_track_id[order]
            self.box_extrapolated = self.box_extrapolated[order]
        else:
            self.box_track_id = track_ids.copy()

        return {
            "policy": "moving_last_two_pose_linear_yaw_static_hold",
            "through_frame": target_frame,
            "original_box_rows": original_rows,
            "extrapolated_box_rows": len(new_frame),
            "extrapolated_tracks": linear_tracks + held_tracks,
            "linear_velocity_tracks": linear_tracks,
            "constant_pose_tracks": held_tracks,
            "single_observation_tracks": single_observation_tracks,
        }

    def _box_world_points(
        self, frame_idx: int, moving_only: bool, box_type: Optional[int] = None
    ) -> np.ndarray:
        sel = self.box_frame == int(frame_idx)
        if moving_only:
            sel = sel & self.box_moving
        if box_type is not None:
            sel = sel & (self.box_type == int(box_type))
        idx = np.nonzero(sel)[0]
        if idx.shape[0] == 0:
            return np.zeros((0, 3), np.float32)
        return np.concatenate(
            [
                sample_box_edges(self.box_o2w[k], self.box_lwh[k], self.box_edge_spacing)
                for k in idx
            ],
            axis=0,
        )

    def _project(
        self,
        pts: np.ndarray,
        lidar_to_world: np.ndarray,
        out_width: int,
        min_range: float,
        max_range: float,
        max_horizontal: Optional[float] = None,
    ):
        return project_world_points(
            pts,
            lidar_to_world,
            self.elev,
            self.azim_native,
            self.row_off,
            self.C_native,
            out_width,
            min_range,
            max_range,
            max_horizontal=max_horizontal,
        )

    def rasterize(
        self,
        frame_idx: int,
        lidar_to_world: np.ndarray,
        out_width: int,
        min_range: float,
        max_range: float,
        include_boxes: bool = True,
        moving_only: bool = False,
        return_parts: bool = False,
    ):
        """Compose the hdmap range image for ``frame_idx``.

        Returns ``(R, out_width)`` float32 (metres; 0 == empty). If
        ``return_parts`` also returns a dict with separate ``map`` / ``boxes``
        range images (for visualization)."""

        def _proj(pts, max_horizontal=None):
            return self._project(
                pts, lidar_to_world, out_width, min_range, max_range, max_horizontal
            )

        mr, mc, md = _proj(self.map_points, max_horizontal=self.map_max_dist)
        rows, cols, rng = [mr], [mc], [md]
        parts = {}
        if return_parts:
            parts["map"] = scatter_min_range(self.R, out_width, mr, mc, md)
        if include_boxes:
            bpts = self._box_world_points(frame_idx, moving_only)
            br, bc, bd = _proj(bpts)
            rows.append(br)
            cols.append(bc)
            rng.append(bd)
            if return_parts:
                parts["boxes"] = scatter_min_range(self.R, out_width, br, bc, bd)
        full = scatter_min_range(
            self.R,
            out_width,
            np.concatenate(rows),
            np.concatenate(cols),
            np.concatenate(rng),
        )
        if return_parts:
            return full, parts
        return full

    def _to_sensor(
        self,
        pts_world: np.ndarray,
        Rm: np.ndarray,
        t: np.ndarray,
        min_range: float,
        max_range: float,
        max_horizontal: Optional[float] = None,
    ) -> np.ndarray:
        """World points -> top-sensor frame (P, 3), range/BEV-cropped (no projection)."""
        if pts_world.shape[0] == 0:
            return np.zeros((0, 3), np.float32)
        ps = (pts_world.astype(np.float64) - t[None, :]) @ Rm  # == R^T (p - t)
        r = np.linalg.norm(ps, axis=1)
        keep = (r >= min_range) & (r <= max_range)
        if max_horizontal is not None:
            keep = keep & (np.hypot(ps[:, 0], ps[:, 1]) <= max_horizontal)
        return ps[keep].astype(np.float32)

    def sensor_points(
        self,
        frame_idx: int,
        lidar_to_world: np.ndarray,
        min_range: float,
        max_range: float,
        include_boxes: bool = True,
        moving_only: bool = False,
    ) -> np.ndarray:
        """HDMap points in the TOP-SENSOR frame ``(P, 3)`` metres.

        Same world-point set :meth:`rasterize` composes (static map polylines +
        dynamic box wireframes), transformed ``p_s = R^T (p_w - t)`` and
        range/BEV-cropped exactly like the rasterizer -- but WITHOUT the range-
        image projection, so the metric points can be fed straight to a BEV
        voxelizer (e.g. TransformerVAEDecompBEV's ``extra_points`` path) instead
        of being rasterized -> unprojected (which puts ground-plane map elements
        on grazing rays). Static map points are BEV-cropped by ``map_max_dist``;
        box points use the full range (mirrors :meth:`rasterize`)."""
        Rm = lidar_to_world[:3, :3].astype(np.float64)
        t = lidar_to_world[:3, 3].astype(np.float64)
        parts = [
            self._to_sensor(
                self.map_points, Rm, t, min_range, max_range, self.map_max_dist
            )
        ]
        if include_boxes:
            bpts = self._box_world_points(frame_idx, moving_only)
            parts.append(self._to_sensor(bpts, Rm, t, min_range, max_range, None))
        return np.concatenate(parts, axis=0)

    def _box_indices(
        self, frame_idx: int, moving_only: bool, box_type: Optional[int] = None
    ) -> np.ndarray:
        sel = self.box_frame == int(frame_idx)
        if moving_only:
            sel = sel & self.box_moving
        if box_type is not None:
            sel = sel & (self.box_type == int(box_type))
        return np.nonzero(sel)[0]

    @staticmethod
    def _fill_quad_bev(
        out_ch: np.ndarray, poly: np.ndarray, grid: Tuple[float, float, float, float, int, int]
    ) -> None:
        """Fill a convex quad (4, 2) into ``out_ch`` (Y, X). ``grid`` =
        (x_min, y_min, vx, vy, Y, X). Cell centre inside-test over the quad's
        bbox (CCW half-plane intersection) -> solid/closed box footprint."""
        x_min, y_min, vx, vy, Y, X = grid
        ixs = np.floor((poly[:, 0] - x_min) / vx)
        iys = np.floor((poly[:, 1] - y_min) / vy)
        ix0, ix1 = max(int(ixs.min()), 0), min(int(ixs.max()) + 1, X)
        iy0, iy1 = max(int(iys.min()), 0), min(int(iys.max()) + 1, Y)
        if ix1 <= ix0 or iy1 <= iy0:
            return
        xs = x_min + (np.arange(ix0, ix1) + 0.5) * vx
        ys = y_min + (np.arange(iy0, iy1) + 0.5) * vy
        gx, gy = np.meshgrid(xs, ys)  # (h, w)
        # Orient CCW (positive signed area) so the half-plane test is consistent.
        area = 0.0
        n = poly.shape[0]
        for i in range(n):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % n]
            area += x1 * y2 - x2 * y1
        pp = poly if area > 0 else poly[::-1]
        inside = np.ones(gx.shape, dtype=bool)
        for i in range(n):
            x1, y1 = pp[i]
            x2, y2 = pp[(i + 1) % n]
            inside &= ((x2 - x1) * (gy - y1) - (y2 - y1) * (gx - x1)) >= 0
        out_ch[iy0:iy1, ix0:ix1][inside] = 1.0

    def rasterize_bev(
        self,
        frame_idx: int,
        lidar_to_world: np.ndarray,
        bev_size: Tuple[int, int],
        pc_range: Tuple[float, float, float, float, float, float],
        include_boxes: bool = True,
        moving_only: bool = False,
    ) -> np.ndarray:
        """Multi-channel BEV occupancy raster in the TOP-SENSOR frame.

        Returns ``(NUM_CLASSES, Y, X)`` float32 in {0, 1}, one channel per
        semantic class (:data:`CLASS_NAMES`: lane, road_line, road_edge,
        crosswalk, vehicle, pedestrian, cyclist). Static map classes splat their
        pre-sampled polyline points into BEV cells; dynamic box classes fill the
        CLOSED (solid) rotated BEV footprint of each box. Uses the SAME
        ``x,y -> cell`` mapping as ``PillarVoxelizer`` (row = iy over y, col = ix
        over x) so the raster is spatially aligned with the VAE's BEV latent
        grid. ``bev_size`` = (Y, X); ``pc_range`` = (x_min, x_max, y_min, y_max,
        z_min, z_max) (z unused here -- pure top-down)."""
        Y, X = int(bev_size[0]), int(bev_size[1])
        x_min, x_max, y_min, y_max = (float(v) for v in pc_range[:4])
        vx = (x_max - x_min) / X
        vy = (y_max - y_min) / Y
        out = np.zeros((NUM_CLASSES, Y, X), dtype=np.float32)
        Rm = lidar_to_world[:3, :3].astype(np.float64)
        t = lidar_to_world[:3, 3].astype(np.float64)

        # Static map: splat sampled polyline points, one channel per map_type.
        for ci in range(len(MAP_CLASS_NAMES)):
            pw = self.map_points[self.map_type == ci]
            if pw.shape[0] == 0:
                continue
            ps = (pw.astype(np.float64) - t[None, :]) @ Rm  # (P, 3) sensor
            ix = np.floor((ps[:, 0] - x_min) / vx).astype(np.int64)
            iy = np.floor((ps[:, 1] - y_min) / vy).astype(np.int64)
            keep = (ix >= 0) & (ix < X) & (iy >= 0) & (iy < Y)
            out[ci, iy[keep], ix[keep]] = 1.0

        # Dynamic boxes: fill the solid rotated footprint, one channel per box_type.
        if include_boxes:
            grid = (x_min, y_min, vx, vy, Y, X)
            xy_signs = np.array([[1, 1], [1, -1], [-1, -1], [-1, 1]], dtype=np.float64)
            for bi in range(len(BOX_CLASS_NAMES)):
                ch = len(MAP_CLASS_NAMES) + bi
                for k in self._box_indices(frame_idx, moving_only, box_type=bi):
                    o2w = self.box_o2w[k].astype(np.float64)
                    half = 0.5 * self.box_lwh[k].astype(np.float64)
                    corners_obj = np.concatenate(
                        [xy_signs * half[:2], np.zeros((4, 1))], axis=1
                    )  # (4, 3) object frame, z=0 (footprint)
                    corners_w = corners_obj @ o2w[:3, :3].T + o2w[:3, 3]
                    poly = ((corners_w - t[None, :]) @ Rm)[:, :2]  # (4, 2) sensor xy
                    self._fill_quad_bev(out[ch], poly, grid)

        return out

    def rasterize_multichannel(
        self,
        frame_idx: int,
        lidar_to_world: np.ndarray,
        out_width: int,
        min_range: float,
        max_range: float,
        moving_only: bool = False,
    ) -> np.ndarray:
        """One metric-range channel per semantic class -> ``(NUM_CLASSES, R, out_width)``.

        Channel order is :data:`CLASS_NAMES` (lane, road_line, road_edge,
        crosswalk, vehicle, pedestrian, cyclist). Each channel holds the nearest
        return's range in metres (0 == empty). Static-map channels are BEV-cropped
        by ``self.map_max_dist``; box channels use the full range."""
        out = np.zeros((NUM_CLASSES, self.R, out_width), dtype=np.float32)

        # Static map: one channel per map_type (0..3).
        for ci in range(len(MAP_CLASS_NAMES)):
            pts = self.map_points[self.map_type == ci]
            r, c, d = self._project(
                pts, lidar_to_world, out_width, min_range, max_range, self.map_max_dist
            )
            out[ci] = scatter_min_range(self.R, out_width, r, c, d)

        # Dynamic boxes: one channel per box_type (0..2), offset by the map channels.
        for bi in range(len(BOX_CLASS_NAMES)):
            pts = self._box_world_points(frame_idx, moving_only, box_type=bi)
            r, c, d = self._project(pts, lidar_to_world, out_width, min_range, max_range)
            out[len(MAP_CLASS_NAMES) + bi] = scatter_min_range(self.R, out_width, r, c, d)

        return out


class TorchWaymoHDMapRasterizer:
    """GPU-resident single-channel counterpart to :meth:`rasterize`.

    Static map points, beam lookup tables, and all selected per-frame box
    wireframes are uploaded once in FP64 to preserve the CPU composer's pixel
    assignment. Calls accept CUDA poses and return FP32 CUDA metric range
    images without a per-block host/device transfer.
    """

    def __init__(
        self,
        composer: WaymoHDMapComposer,
        device: torch.device,
        include_boxes: bool = True,
        moving_only: bool = False,
    ):
        self.device = torch.device(device)
        self.compute_dtype = torch.float64
        self.height = composer.R
        self.native_width = composer.C_native
        self.map_max_dist = composer.map_max_dist
        self.static_points = torch.as_tensor(
            composer.map_points,
            dtype=self.compute_dtype,
            device=self.device,
        )

        elev = torch.as_tensor(
            composer.elev,
            dtype=self.compute_dtype,
            device=self.device,
        )
        self.elev_order = torch.argsort(elev, stable=True)
        self.elev_sorted = elev[self.elev_order]
        azim = self._wrap_angle(
            torch.as_tensor(
                composer.azim_native,
                dtype=self.compute_dtype,
                device=self.device,
            )
        )
        self.azim_order = torch.argsort(azim, stable=True)
        self.azim_sorted = azim[self.azim_order]
        self.row_off = torch.as_tensor(
            composer.row_off,
            dtype=self.compute_dtype,
            device=self.device,
        )

        box_parts = []
        box_frame_offsets = [0]
        for frame_idx in range(composer.num_frames):
            points = (
                composer._box_world_points(frame_idx, moving_only)
                if include_boxes
                else np.zeros((0, 3), dtype=np.float64)
            )
            box_parts.append(points.astype(np.float64, copy=False))
            box_frame_offsets.append(
                box_frame_offsets[-1] + int(points.shape[0])
            )
        box_points = (
            np.concatenate(box_parts, axis=0)
            if box_frame_offsets[-1] > 0
            else np.zeros((0, 3), dtype=np.float64)
        )
        self.box_points = torch.as_tensor(
            box_points,
            dtype=self.compute_dtype,
            device=self.device,
        )
        self.box_frame_offsets = tuple(box_frame_offsets)

    @staticmethod
    def _wrap_angle(angle: torch.Tensor) -> torch.Tensor:
        return torch.remainder(angle + np.pi, 2.0 * np.pi) - np.pi

    @property
    def resident_bytes(self) -> int:
        tensors = (
            self.static_points,
            self.box_points,
            self.elev_order,
            self.elev_sorted,
            self.azim_order,
            self.azim_sorted,
            self.row_off,
        )
        return sum(t.numel() * t.element_size() for t in tensors)

    def _project_points(
        self,
        points_world: torch.Tensor,
        pose: torch.Tensor,
        out_width: int,
        min_range: float,
        max_range: float,
        max_horizontal: Optional[float],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if points_world.shape[0] == 0:
            empty_indices = torch.empty(
                (0,), dtype=torch.int64, device=self.device
            )
            return empty_indices, points_world.new_empty((0,))

        points_sensor = (
            points_world - pose[:3, 3].unsqueeze(0)
        ) @ pose[:3, :3]
        ranges = torch.linalg.vector_norm(points_sensor, dim=1)
        keep = (ranges >= min_range) & (ranges <= max_range)
        if max_horizontal is not None:
            keep &= (
                points_sensor[:, :2].square().sum(dim=1)
                <= max_horizontal * max_horizontal
            )
        points_sensor = points_sensor[keep]
        ranges = ranges[keep]
        if points_sensor.shape[0] == 0:
            empty_indices = torch.empty(
                (0,), dtype=torch.int64, device=self.device
            )
            return empty_indices, ranges

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
            & (self.elev_order[elev_idx0] <= self.elev_order[elev_idx1])
        )
        rows = self.elev_order[
            torch.where(choose0, elev_idx0, elev_idx1)
        ]

        point_azim = self._wrap_angle(
            torch.atan2(points_sensor[:, 1], points_sensor[:, 0])
            - self.row_off[rows]
        )
        azim_idx = torch.searchsorted(self.azim_sorted, point_azim)
        azim_idx0 = torch.remainder(azim_idx - 1, self.native_width)
        azim_idx1 = torch.remainder(azim_idx, self.native_width)
        azim_dist0 = self._wrap_angle(
            point_azim - self.azim_sorted[azim_idx0]
        ).abs()
        azim_dist1 = self._wrap_angle(
            point_azim - self.azim_sorted[azim_idx1]
        ).abs()
        nearest_azim = torch.where(
            azim_dist0 <= azim_dist1, azim_idx0, azim_idx1
        )
        native_cols = self.azim_order[nearest_azim]
        cols = torch.clamp(
            torch.div(
                native_cols * out_width,
                self.native_width,
                rounding_mode="floor",
            ),
            max=out_width - 1,
        )
        return rows * out_width + cols, ranges

    @torch.inference_mode()
    def rasterize(
        self,
        frame_indices: Sequence[int],
        lidar_to_world: torch.Tensor,
        out_width: int,
        min_range: float,
        max_range: float,
    ) -> torch.Tensor:
        poses = lidar_to_world.to(
            device=self.device,
            dtype=self.compute_dtype,
        )
        if poses.ndim != 3 or poses.shape[1:] != (4, 4):
            raise ValueError(
                "lidar_to_world must have shape [T,4,4], got "
                f"{tuple(poses.shape)}"
            )
        if len(frame_indices) != poses.shape[0]:
            raise ValueError(
                "frame_indices must align with lidar poses: "
                f"{len(frame_indices)} vs {poses.shape[0]}"
            )

        rasters = []
        for frame_idx, pose in zip(frame_indices, poses):
            frame_idx = int(frame_idx)
            if not 0 <= frame_idx < len(self.box_frame_offsets) - 1:
                raise IndexError(f"HDMap frame index out of range: {frame_idx}")
            metric = torch.full(
                (self.height * out_width,),
                torch.inf,
                dtype=self.compute_dtype,
                device=self.device,
            )
            flat_indices, ranges = self._project_points(
                self.static_points,
                pose,
                out_width,
                min_range,
                max_range,
                self.map_max_dist,
            )
            metric.scatter_reduce_(
                0, flat_indices, ranges, reduce="amin", include_self=True
            )
            box_start = self.box_frame_offsets[frame_idx]
            box_end = self.box_frame_offsets[frame_idx + 1]
            flat_indices, ranges = self._project_points(
                self.box_points[box_start:box_end],
                pose,
                out_width,
                min_range,
                max_range,
                None,
            )
            metric.scatter_reduce_(
                0, flat_indices, ranges, reduce="amin", include_self=True
            )
            metric.masked_fill_(~torch.isfinite(metric), 0.0)
            rasters.append(metric.view(self.height, out_width))
        return torch.stack(rasters, dim=0).to(dtype=torch.float32)
