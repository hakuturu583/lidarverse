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

"""NCore-backed nuScenes lidar reader (the RadymLidar analog for NCore V4 data).

This is the nuScenes sibling of ``waymo_ncore.py``. nuScenes NCore stores the
lidar (a 32-beam Velodyne HDL-32E spinning sensor) as the SAME *structured
sparse ray bundle* per frame that Waymo uses -- each ray carries a
``model_element`` (row, col) index into the sensor grid plus per-return
``distance_m`` / ``intensity`` / ``valid_mask``. The sensor grid is defined by a
``RowOffsetStructuredSpinningLidarModel`` whose ``row_elevations_rad`` (32) and
``column_azimuths_rad`` tables give the beam geometry.

Because the ray-bundle API and the structured-spinning model are identical to
Waymo's (only the row/column counts differ), this module REUSES the verified,
sensor-agnostic geometry helpers from ``waymo_ncore`` (``get_waymo_model_tables``
etc.) and only diverges on:

  * the default ``lidar_id`` / camera-name map (nuScenes 6-camera rig),
  * hdmap rasterized from a per-scene NPZ (scripts/data/dump_nuscenes_hdmap.py) but
    projected with the nuScenes per-sweep ``lidar_to_global`` stored in the NPZ
    (the NCore pose graph drifts from nuScenes' ego_pose), NOT the NCore pose,
  * NO dumped blind-spot extra lidars (nuScenes has a single lidar).

The reader returns the same dict keys ``LidarRangeMapSampler`` depends on, so the
rest of the pipeline works unchanged. All reads are strictly read-only.
"""

import os
import threading
from typing import Any, List, Optional, Tuple

import numpy as np
import torch
import torch.utils.data
from lru import LRU

from tokenizer.training.datasets.lidar_datasets.data_fileds import DataField

# Sensor-agnostic NCore geometry helpers (defined for the Waymo reader but they
# only read a RowOffsetStructuredSpinningLidarModel's angle tables, so they apply
# to any NCore structured-spinning lidar, nuScenes included).
from tokenizer.training.datasets.lidar_datasets.waymo_ncore import (
    get_waymo_model_tables as get_ncore_model_tables,
    load_waymo_lidar_geometry,
    waymo_range_map_to_ray_directions as ncore_range_map_to_ray_directions,
    waymo_resampled_azimuth_table as ncore_resampled_azimuth_table,
)

try:
    from ncore.data.v4 import SequenceComponentGroupsReader, SequenceLoaderV4
    from ncore.impl.data.types import FrameTimepoint

    _NCORE_AVAILABLE = True
    _NCORE_IMPORT_ERROR: Optional[Exception] = None
except Exception as _e:  # pragma: no cover - import guard for CPU-only envs
    _NCORE_AVAILABLE = False
    _NCORE_IMPORT_ERROR = _e
    FrameTimepoint = None  # type: ignore


# Camera view index -> NCore camera instance name (nuScenes 6-camera rig).
NUSCENES_VIEW_IDX_TO_CAMERA = {
    0: "camera_front",
    1: "camera_front_left",
    2: "camera_front_right",
    3: "camera_back",
    4: "camera_back_left",
    5: "camera_back_right",
}


def canonicalize_vehicle_forward_columns(
    values: np.ndarray,
    ray_directions: np.ndarray,
    lidar_to_rig: np.ndarray,
    column_axis: int = -1,
) -> np.ndarray:
    """Put rig-forward at W/2 with azimuth decreasing left-to-right."""
    width = values.shape[column_axis]
    if width != ray_directions.shape[1]:
        raise ValueError(
            f"values width {width} != ray width "
            f"{ray_directions.shape[1]}"
        )
    rotation = np.asarray(lidar_to_rig, dtype=np.float64)[:3, :3]
    rays_rig = np.asarray(ray_directions, dtype=np.float64) @ rotation.T
    horizontal = rays_rig[..., :2]
    horizontal /= np.linalg.norm(horizontal, axis=-1, keepdims=True).clip(
        min=1e-12
    )
    column_xy = horizontal.mean(axis=0)
    column_azimuth = np.arctan2(column_xy[:, 1], column_xy[:, 0])
    front_column = int(np.argmax(column_xy[:, 0]))
    next_column = (front_column + 1) % column_azimuth.shape[0]
    azimuth_step = np.arctan2(
        np.sin(column_azimuth[next_column] - column_azimuth[front_column]),
        np.cos(column_azimuth[next_column] - column_azimuth[front_column]),
    )

    output = np.asarray(values)
    if azimuth_step > 0:
        output = np.flip(output, axis=column_axis)
        front_column = width - 1 - front_column
    output = np.roll(output, width // 2 - front_column, axis=column_axis)
    return np.ascontiguousarray(output)


def _lidar_to_rig(loader, lidar, lidar_id: str) -> np.ndarray:
    timestamp = np.asarray(
        [lidar.frames_timestamps_us[0, FrameTimepoint.START.value]],
        dtype=np.uint64,
    )
    return loader.pose_graph.evaluate_poses(
        lidar_id, "rig", timestamp
    )[0].astype(np.float64)


def load_nuscenes_lidar_geometry(
    root_path: str,
    filter_list_path: Optional[str] = None,
    lidar_id: str = "lidar_top",
    resample_width: int = -1,
    itar_index_tail_read_size: int = 1 << 20,
    center_vehicle_forward: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Return native ray directions and the LiDAR-to-rig calibration.

    Opens the FIRST clip in the filter list ONCE to read the sensor's
    ``row_elevations_rad`` / ``column_azimuths_rad`` / ``row_azimuth_offsets_rad``
    tables (all nuScenes top-lidar scenes share the same structured spinning
    model). When ``resample_width > 0`` the azimuth axis is the uniform-azimuth
    rebin grid the reader scatters onto; otherwise the native column grid.

    Strictly read-only. Mirrors ``load_waymo_ray_directions``.
    """
    if not _NCORE_AVAILABLE:
        raise ImportError(
            "load_nuscenes_lidar_geometry requires the `ncore` package (pip "
            f"install nvidia-ncore). Import failed with: {_NCORE_IMPORT_ERROR}"
        )
    stems: List[str] = []
    if filter_list_path is not None and os.path.isfile(filter_list_path):
        with open(filter_list_path, "r") as f:
            stems = [line.strip() for line in f.readlines() if line.strip()]
        stems = [s.split(".ncore4")[0].split(".json")[0] for s in stems]
    if not stems and os.path.isdir(root_path):
        for name in sorted(os.listdir(root_path)):
            d = os.path.join(root_path, name)
            if os.path.isdir(d) and os.path.isfile(
                os.path.join(d, f"{name}.ncore4-{lidar_id}.zarr.itar")
            ):
                stems.append(name)
                break
    if not stems:
        raise RuntimeError(
            f"load_nuscenes_lidar_geometry: no clips found (root={root_path}, "
            f"filter_list_path={filter_list_path})."
        )
    stem = stems[0]
    clip_dir = os.path.join(root_path, stem)
    paths = [
        os.path.join(clip_dir, f"{stem}.ncore4.zarr.itar"),
        os.path.join(clip_dir, f"{stem}.ncore4-{lidar_id}.zarr.itar"),
    ]
    reader = SequenceComponentGroupsReader(
        paths, itar_index_tail_read_size=itar_index_tail_read_size
    )
    loader = SequenceLoaderV4(
        reader, masks_component_group_name=None, cuboids_component_group_name=None
    )
    lidar = loader.get_lidar_sensor(lidar_id)
    elev, azim, row_off = get_ncore_model_tables(lidar.model_parameters)
    if resample_width and int(resample_width) > 0:
        azim = ncore_resampled_azimuth_table(azim, int(resample_width))
    dirs = ncore_range_map_to_ray_directions(elev, azim, row_off)  # (R, W, 3)
    lidar_to_rig = _lidar_to_rig(loader, lidar, lidar_id)
    if center_vehicle_forward:
        dirs = canonicalize_vehicle_forward_columns(
            dirs,
            dirs,
            lidar_to_rig,
            column_axis=-2,
        )
    del loader, reader
    return dirs, lidar_to_rig


def load_nuscenes_ray_directions(
    root_path: str,
    filter_list_path: Optional[str] = None,
    lidar_id: str = "lidar_top",
    resample_width: int = -1,
    itar_index_tail_read_size: int = 1 << 20,
    center_vehicle_forward: bool = True,
) -> np.ndarray:
    """Per-pixel unit ray directions ``(R, W, 3)`` for the nuScenes range image."""
    dirs, _ = load_nuscenes_lidar_geometry(
        root_path=root_path,
        filter_list_path=filter_list_path,
        lidar_id=lidar_id,
        resample_width=resample_width,
        itar_index_tail_read_size=itar_index_tail_read_size,
        center_vehicle_forward=center_vehicle_forward,
    )
    return dirs


# ----------------------------------------------------------------------------- #
# Reader
# ----------------------------------------------------------------------------- #
class NuScenesNCoreLidar:
    """RadymLidar-compatible reader backed by nuScenes NCore V4 ``.zarr.itar``.

    Mirrors ``WaymoNCoreLidar`` (same surface ``LidarRangeMapSampler`` depends on:
    ``custom_fields``, ``size_of_filter_set()``,
    ``_read_rgb_and_lidar(lidar_idx, lidar_frame_idx, data_fields)``). hdmap is
    supported via a dumped NPZ (nuScenes-pose-projected); blind-spot extra lidars
    are not (nuScenes has a single lidar).
    """

    MAX_LOADERS = 4  # per-worker cache of opened NCore sequences (.itar handles)

    def __init__(
        self,
        root_path: str,
        filter_list_path: Optional[str] = None,
        num_views: int = -1,
        metadata_folder: Optional[str] = None,
        custom_folders: Optional[List[str]] = None,
        custom_fields: Optional[List[str]] = None,
        view_indices: Optional[List[int]] = None,
        load_intensity: bool = False,
        min_range: float = 1.0,
        max_range: float = 100.0,
        # --- NCore specifics ---
        lidar_id: str = "lidar_top",
        return_index: int = 0,
        load_cameras: bool = False,
        itar_index_tail_read_size: int = 1 << 20,
        resample_width: int = -1,
        center_vehicle_forward: bool = True,
        # --- hdmap (on-the-fly rasterized from dump_nuscenes_hdmap.py NPZs) ---
        hdmap_root: Optional[str] = None,
        hdmap_box_edge_spacing: float = 0.15,
        hdmap_moving_only: bool = False,
        hdmap_include_boxes: bool = True,
        hdmap_map_max_dist: Optional[float] = None,
        hdmap_multichannel: bool = False,
        hdmap_bev: bool = False,
        hdmap_bev_size: Tuple[int, int] = (256, 256),
        hdmap_bev_pc_range: Tuple[float, float, float, float, float, float] = (
            -50.0, 50.0, -50.0, 50.0, -5.0, 5.0,
        ),
        # --- optional nuScenes-layout -> Waymo-sensor canonicalization ---
        hdmap_sensor_template: str = "native",
        hdmap_template_root_path: Optional[str] = None,
        hdmap_template_filter_list_path: Optional[str] = None,
        hdmap_virtual_sensor_height_m: Optional[float] = None,
        hdmap_virtual_sensor_yaw_deg: float = 0.0,
        canonical_lidar_mode: str = "native",
        hdmap_pose_for_lidar_to_world: bool = False,
        **_ignored: Any,
    ):
        if not _NCORE_AVAILABLE:
            raise ImportError(
                "NuScenesNCoreLidar requires the `ncore` package (pip install "
                f"nvidia-ncore). Import failed with: {_NCORE_IMPORT_ERROR}"
            )

        self.root_path = root_path
        self.num_views = num_views
        self.n_views = num_views
        self.view_indices = view_indices
        self.custom_folders = custom_folders
        self.custom_fields = custom_fields if custom_fields is not None else ["lidar"]
        self.metadata_folder = metadata_folder
        self.load_intensity = load_intensity

        self.min_range = float(min_range)
        self.max_range = float(max_range)

        self.lidar_id = lidar_id
        self.return_index = return_index
        self.load_cameras = load_cameras
        self.itar_index_tail_read_size = itar_index_tail_read_size
        # >0 -> rebin the native column/azimuth axis onto a uniform grid of this
        # width. <=0 -> keep the native grid. Rows (beams) are never resampled.
        self.resample_width = int(resample_width)
        self.center_vehicle_forward = bool(center_vehicle_forward)

        # HDMap: no native NCore component; rasterized on the fly from the per-scene
        # NPZ written by scripts/data/dump_nuscenes_hdmap.py. Unlike Waymo, the composer
        # projects with the nuScenes per-sweep lidar_to_global stored in the NPZ (the
        # NCore pose graph drifts from nuScenes' ego_pose), NOT the NCore lidar pose.
        self.hdmap_root = hdmap_root
        self.hdmap_box_edge_spacing = float(hdmap_box_edge_spacing)
        self.hdmap_moving_only = bool(hdmap_moving_only)
        self.hdmap_include_boxes = bool(hdmap_include_boxes)
        self.hdmap_multichannel = bool(hdmap_multichannel)
        self.hdmap_bev = bool(hdmap_bev)
        self.hdmap_bev_size = tuple(hdmap_bev_size)
        self.hdmap_bev_pc_range = tuple(hdmap_bev_pc_range)
        self.hdmap_map_max_dist = (
            self.max_range if hdmap_map_max_dist is None else float(hdmap_map_max_dist)
        )
        if hdmap_sensor_template not in ("native", "waymo"):
            raise ValueError(
                "hdmap_sensor_template must be 'native' or 'waymo', got "
                f"{hdmap_sensor_template!r}"
            )
        if canonical_lidar_mode not in ("native", "empty"):
            raise ValueError(
                "canonical_lidar_mode must be 'native' or 'empty', got "
                f"{canonical_lidar_mode!r}"
            )
        if hdmap_sensor_template == "waymo" and canonical_lidar_mode != "empty":
            raise ValueError(
                "Waymo-template HDMap projection requires canonical_lidar_mode='empty' "
                "so video and conditioning share the 64-row Waymo tensor shape."
            )
        self.hdmap_sensor_template = hdmap_sensor_template
        self.hdmap_virtual_sensor_height_m = (
            None if hdmap_virtual_sensor_height_m is None
            else float(hdmap_virtual_sensor_height_m)
        )
        self.hdmap_virtual_sensor_yaw_deg = float(
            hdmap_virtual_sensor_yaw_deg
        )
        self.canonical_lidar_mode = canonical_lidar_mode
        self.hdmap_pose_for_lidar_to_world = bool(
            hdmap_pose_for_lidar_to_world
        )
        self._template_geometry = None
        if self.hdmap_sensor_template == "waymo":
            if not hdmap_template_root_path:
                raise ValueError(
                    "hdmap_template_root_path is required when "
                    "hdmap_sensor_template='waymo'"
                )
            _, elev, azim, row_off = load_waymo_lidar_geometry(
                root_path=hdmap_template_root_path,
                filter_list_path=hdmap_template_filter_list_path,
                lidar_id="lidar_top",
                resample_width=-1,
                itar_index_tail_read_size=self.itar_index_tail_read_size,
            )
            self._template_geometry = (elev, azim, row_off, int(azim.shape[0]))
        self.composers: dict[str, Any] = {}

        # nuScenes has a single spinning lidar (no dumped blind-spot extras).
        if "extra_lidar" in self.custom_fields:
            raise NotImplementedError(
                "NuScenesNCoreLidar does not support custom_fields ['extra_lidar'] "
                "(nuScenes has a single lidar). Use custom_fields=['lidar'] or "
                "['lidar', 'hdmap']."
            )
        if "hdmap" in self.custom_fields and not self.hdmap_root:
            raise NotImplementedError(
                "nuScenes NCore has no native hdmap component. Set hdmap_root=<dir of "
                "dump_nuscenes_hdmap.py NPZs> to rasterize one on the fly, or remove "
                "'hdmap' from custom_fields."
            )
        if (
            self.hdmap_pose_for_lidar_to_world
            and "hdmap" not in self.custom_fields
        ):
            raise ValueError(
                "hdmap_pose_for_lidar_to_world requires the HDMap composer"
            )

        # Build the clip list (stems). Each clip is a subdir <root>/<stem>/ that
        # holds <stem>.ncore4*.zarr.itar + <stem>.json.
        self.filter_set: List[str] = self._build_filter_set(filter_list_path)
        if len(self.filter_set) == 0:
            raise RuntimeError(
                f"No nuScenes NCore clips found (root={root_path}, "
                f"filter_list_path={filter_list_path})."
            )

        # Per-worker handle cache (opened lazily inside the worker process).
        self.worker_id: Optional[int] = None
        self._lock = threading.Lock()
        self.loaders = LRU(self.MAX_LOADERS, callback=self._evict_loader)

    def canonicalize_columns(
        self, loader, lidar, values: np.ndarray
    ) -> np.ndarray:
        if (
            not self.center_vehicle_forward
            or self.canonical_lidar_mode != "native"
        ):
            return np.ascontiguousarray(values)
        elev, azim, row_off = get_ncore_model_tables(lidar.model_parameters)
        if self.resample_width > 0:
            azim = ncore_resampled_azimuth_table(azim, self.resample_width)
        rays = ncore_range_map_to_ray_directions(elev, azim, row_off)
        return canonicalize_vehicle_forward_columns(
            values, rays, _lidar_to_rig(loader, lidar, self.lidar_id)
        )

    # --------------------------------------------------------------------- #
    @staticmethod
    def _evict_loader(_key, value):
        loader, reader = value
        del loader, reader

    def _build_filter_set(self, filter_list_path: Optional[str]) -> List[str]:
        if filter_list_path is not None and os.path.isfile(filter_list_path):
            with open(filter_list_path, "r") as f:
                stems = [line.strip() for line in f.readlines() if line.strip()]
            stems = [s.split(".ncore4")[0].split(".json")[0] for s in stems]
            return stems
        # Fallback: scan root for clip subdirs containing the lidar itar.
        stems = []
        if os.path.isdir(self.root_path):
            for name in sorted(os.listdir(self.root_path)):
                d = os.path.join(self.root_path, name)
                if os.path.isdir(d) and os.path.isfile(
                    os.path.join(d, f"{name}.ncore4-{self.lidar_id}.zarr.itar")
                ):
                    stems.append(name)
        return stems

    def _check_worker_id(self):
        info = torch.utils.data.get_worker_info()
        if info is not None:
            if self.worker_id is not None:
                assert self.worker_id == info.id, "Worker id mismatch"
            else:
                self.worker_id = info.id

    def _clip_component_paths(self, stem: str) -> List[str]:
        clip_dir = os.path.join(self.root_path, stem)
        paths = [
            os.path.join(clip_dir, f"{stem}.ncore4.zarr.itar"),
            os.path.join(clip_dir, f"{stem}.ncore4-{self.lidar_id}.zarr.itar"),
        ]
        if self.load_cameras:
            view_indices = (
                self.view_indices
                if self.view_indices is not None
                else range(max(self.num_views, 0))
            )
            for v in view_indices:
                cam_name = NUSCENES_VIEW_IDX_TO_CAMERA[v]
                paths.append(
                    os.path.join(clip_dir, f"{stem}.ncore4-{cam_name}.zarr.itar")
                )
        return paths

    def _get_loader(self, stem: str):
        """Open (and cache) the NCore sequence loader for a clip."""
        self._check_worker_id()
        if stem in self.loaders:
            return self.loaders[stem][0]
        paths = self._clip_component_paths(stem)
        reader = SequenceComponentGroupsReader(
            paths, itar_index_tail_read_size=self.itar_index_tail_read_size
        )
        loader = SequenceLoaderV4(
            reader,
            masks_component_group_name=None,
            cuboids_component_group_name=None,
        )
        self.loaders[stem] = (loader, reader)
        return loader

    def _get_composer(self, stem: str, lidar):
        """Open (and cache) the per-scene HDMap composer (nuScenes-pose-projected)."""
        if stem in self.composers:
            return self.composers[stem]
        from tokenizer.training.datasets.lidar_datasets.nuscenes_hdmap import (
            NuScenesHDMapComposer,
        )

        npz_path = os.path.join(self.hdmap_root, f"{stem}.npz")
        if not os.path.isfile(npz_path):
            raise FileNotFoundError(
                f"HDMap NPZ not found: {npz_path} (run scripts/data/dump_nuscenes_hdmap.py "
                f"--scene {stem} --out_root {self.hdmap_root})."
            )
        if self._template_geometry is None:
            elev, azim, row_off = get_ncore_model_tables(lidar.model_parameters)
            native_columns = int(lidar.model_parameters.n_columns)
        else:
            elev, azim, row_off, native_columns = self._template_geometry
        comp = NuScenesHDMapComposer(
            npz_path,
            elev,
            azim,
            row_off,
            native_columns,
            box_edge_spacing=self.hdmap_box_edge_spacing,
            map_max_dist=self.hdmap_map_max_dist,
            virtual_sensor_height_m=self.hdmap_virtual_sensor_height_m,
            virtual_sensor_yaw_deg=self.hdmap_virtual_sensor_yaw_deg,
        )
        # Frame indices are 1:1 with NCore lidar frames (same ordered sweep list).
        if comp.lidar_to_global.shape[0] != int(lidar.frames_count):
            raise RuntimeError(
                f"[{stem}] hdmap NPZ has {comp.lidar_to_global.shape[0]} frames but "
                f"NCore lidar has {int(lidar.frames_count)} -- re-dump with the "
                "matching nuScenes version so frame indices align."
            )
        self.composers[stem] = comp
        return comp

    # --------------------------------------------------------------------- #
    def size_of_filter_set(self) -> int:
        return len(self.filter_set)

    @staticmethod
    def _clamp_frame_indices(frame_idx: np.ndarray, frames_count: int) -> np.ndarray:
        """Replace out-of-range indices with random valid ones (mirrors
        RadymLidar.parse_range_map_tar)."""
        frame_idx = np.asarray(frame_idx).astype(np.int64).copy()
        oob = frame_idx > frames_count - 1
        if oob.any():
            frame_idx[oob] = np.random.randint(0, frames_count - 1, size=int(oob.sum()))
        return frame_idx

    def _scatter_frame(
        self, lidar, fi: int, R: int, C: int
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Build a dense (R, outW) range map (and optional intensity) for frame
        fi. ``outW = resample_width`` if >0 (uniform-azimuth rebin), else ``C``.
        On a collision (multiple native columns -> one target bin) the NEAREST
        return wins (smallest range = closest surface)."""
        elem = lidar.get_frame_ray_bundle_model_element(fi)  # (N, 2) row,col
        if elem is None:
            raise RuntimeError(
                f"Lidar frame {fi} has no model_element (cannot rasterize)."
            )
        dist = lidar.get_frame_ray_bundle_return_distance_m(fi, return_index=self.return_index)
        valid = lidar.get_frame_ray_bundle_return_valid_mask(fi, return_index=self.return_index)
        mask = valid & np.isfinite(dist) & (dist > 0)
        rows = elem[mask, 0].astype(np.int64)
        cols = elem[mask, 1].astype(np.int64)
        d = dist[mask].astype(np.float32)
        inten = None
        if self.load_intensity:
            inten = lidar.get_frame_ray_bundle_return_intensity(
                fi, return_index=self.return_index
            )[mask].astype(np.float32)

        if self.resample_width > 0:
            outW = self.resample_width
            cols = np.minimum((cols * outW) // C, outW - 1)
            # nearest-range wins on collision: sort descending so the smallest
            # range is written LAST (overwrites).
            order = np.argsort(-d, kind="stable")
            rows, cols, d = rows[order], cols[order], d[order]
            if inten is not None:
                inten = inten[order]
        else:
            outW = C

        range_map = np.zeros((R, outW), dtype=np.float32)
        range_map[rows, cols] = d

        intensity_map = None
        if inten is not None:
            intensity_map = np.zeros((R, outW), dtype=np.float32)
            intensity_map[rows, cols] = inten
        return range_map, intensity_map

    def _poses_for_frames(self, loader, lidar, frame_idx: np.ndarray):
        """Return (lidar_to_world (N,4,4), sensor_pose_start_end (N,4,4),
        timestamp_list (N,))."""
        fts = lidar.frames_timestamps_us  # (F, 2) uint64 [START, END]
        t_start = fts[frame_idx, FrameTimepoint.START.value].astype(np.uint64)
        t_end = fts[frame_idx, FrameTimepoint.END.value].astype(np.uint64)
        pg = loader.pose_graph
        l2w_start = pg.evaluate_poses(self.lidar_id, "world", t_start).astype(np.float32)
        l2w_end = pg.evaluate_poses(self.lidar_id, "world", t_end).astype(np.float32)
        # Mirror LidarGEN: relative within-sweep motion = inv(end) @ start.
        sensor_pose_start_end = (np.linalg.inv(l2w_end) @ l2w_start).astype(np.float32)
        return l2w_start, sensor_pose_start_end, t_end.astype(np.int64)

    def _read_rgb_and_lidar(self, lidar_idx, lidar_frame_idx, data_fields):
        """Mirror RadymLidar._read_rgb_and_lidar, NCore-backed (nuScenes)."""
        stem = self.filter_set[lidar_idx]
        loader = self._get_loader(stem)
        lidar = loader.get_lidar_sensor(self.lidar_id)
        frames_count = lidar.frames_count
        frame_idx = self._clamp_frame_indices(lidar_frame_idx, frames_count)

        mp = lidar.model_parameters
        R, C = int(mp.n_rows), int(mp.n_columns)
        if self._template_geometry is None:
            output_rows = R
        else:
            output_rows = int(self._template_geometry[0].shape[0])
        output_width = self.resample_width if self.resample_width > 0 else (
            int(self._template_geometry[3]) if self._template_geometry is not None else C
        )

        output_dict: dict[str | DataField, Any] = {"__key__": stem}

        l2w_start, sensor_pose_start_end, ts = self._poses_for_frames(loader, lidar, frame_idx)
        if self.canonical_lidar_mode == "empty":
            comp = self._get_composer(stem, lidar)
            l2w_start = np.stack(
                [comp.pose_for_frame(int(fi)) for fi in frame_idx], axis=0
            ).astype(np.float32)
            sensor_pose_start_end = np.repeat(
                np.eye(4, dtype=np.float32)[None], len(frame_idx), axis=0
            )
            ts = comp.frame_timestamps_us[frame_idx].astype(np.int64)
        elif self.hdmap_pose_for_lidar_to_world:
            # Keep accumulated-geometry registration in the same nuScenes
            # global frame used to render the map. NCore's start/end relative
            # pose remains the within-sweep rolling-shutter correction.
            comp = self._get_composer(stem, lidar)
            l2w_start = np.stack(
                [comp.pose_for_frame(int(fi)) for fi in frame_idx], axis=0
            ).astype(np.float32)
            ts = comp.frame_timestamps_us[frame_idx].astype(np.int64)
        output_dict["timestamp_list"] = ts
        output_dict["pose_sensor_start_end"] = sensor_pose_start_end
        output_dict["lidar_to_world"] = l2w_start  # (N, 4, 4), sweep-start
        output_dict["rgb_frame_indices"] = frame_idx.tolist()

        for data_field in data_fields:
            if data_field == "custom":
                output_dict[data_field] = {}
                range_list, intensity_list = [], []
                if self.canonical_lidar_mode == "empty":
                    range_list = [
                        np.zeros((output_rows, output_width), dtype=np.float32)
                        for _ in frame_idx
                    ]
                else:
                    for fi in frame_idx:
                        rmap, imap = self._scatter_frame(lidar, int(fi), R, C)
                        range_list.append(
                            self.canonicalize_columns(loader, lidar, rmap)
                        )
                        if imap is not None:
                            intensity_list.append(
                                self.canonicalize_columns(loader, lidar, imap)
                            )
                output_dict[data_field]["lidar"] = np.stack(range_list, axis=0)  # [N, R, C]
                if self.load_intensity:
                    if self.canonical_lidar_mode == "empty":
                        intensity_list = [
                            np.zeros((output_rows, output_width), dtype=np.float32)
                            for _ in frame_idx
                        ]
                    output_dict[data_field]["lidar_intensity"] = np.stack(intensity_list, axis=0)
                if "hdmap" in self.custom_fields:
                    comp = self._get_composer(stem, lidar)
                    outW = output_width
                    # Project with the nuScenes per-sweep pose (NOT the NCore l2w):
                    # map/boxes live in the nuScenes global frame.
                    hdmap_list = []
                    for fi in frame_idx:
                        pose = comp.pose_for_frame(int(fi))
                        if self.hdmap_multichannel:
                            hdmap = comp.rasterize_multichannel(
                                int(fi),
                                pose,
                                outW,
                                self.min_range,
                                self.max_range,
                                moving_only=self.hdmap_moving_only,
                            )
                        else:
                            hdmap = comp.rasterize(
                                int(fi),
                                pose,
                                outW,
                                self.min_range,
                                self.max_range,
                                include_boxes=self.hdmap_include_boxes,
                                moving_only=self.hdmap_moving_only,
                            )
                        hdmap_list.append(
                            self.canonicalize_columns(loader, lidar, hdmap)
                        )
                    # multichannel -> (N, NUM_CLASSES, R, W); else (N, R, W).
                    output_dict[data_field]["hdmap"] = np.stack(hdmap_list, axis=0)
                    if self.hdmap_bev:
                        bev_list = [
                            comp.rasterize_bev(
                                int(fi), comp.pose_for_frame(int(fi)),
                                self.hdmap_bev_size, self.hdmap_bev_pc_range,
                                include_boxes=self.hdmap_include_boxes,
                                moving_only=self.hdmap_moving_only,
                            )
                            for fi in frame_idx
                        ]
                        output_dict[data_field]["hdmap_bev"] = np.stack(bev_list, axis=0)
            elif data_field == DataField.IMAGE_RGB:
                # Camera RGB conditioning: read each requested view's image
                # nearest (in time) to every sampled lidar frame, as (N, 3, H, W)
                # in [0, 1] -- the format the sampler's get_rgb()/process_rgb()
                # expects. Cameras must have been opened (load_cameras=True).
                output_dict[data_field] = {}
                view_indices = (
                    self.view_indices
                    if self.view_indices is not None
                    else list(range(max(self.num_views, 0)))
                )
                for c_view_idx, view_idx in enumerate(view_indices):
                    cam_name = NUSCENES_VIEW_IDX_TO_CAMERA[view_idx]
                    cam = loader.get_camera_sensor(cam_name)
                    frames = []
                    for t_us in ts:
                        ci = cam.get_closest_frame_index(int(t_us))
                        img = np.asarray(cam.get_frame_image_array(ci))  # (H,W,3) uint8
                        frames.append(img)
                    rgb = np.stack(frames, axis=0)  # (N, H, W, 3)
                    rgb = (
                        torch.from_numpy(rgb).permute(0, 3, 1, 2).contiguous().float()
                        / 255.0
                    )  # (N, 3, H, W) in [0, 1]
                    output_dict[data_field][c_view_idx] = rgb
            elif data_field in (
                DataField.CAMERA_INTRINSICS,
                DataField.CAMERA_C2LIDAR_TRANSFORM,
            ):
                # Image-conditioned RF models only VAE-encode the RGB as context;
                # they never project points, so intrinsics / extrinsics are
                # unused. Emit empty dicts (sampler treats as no-op).
                output_dict[data_field] = {}
            else:
                raise NotImplementedError(
                    f"NuScenesNCoreLidar does not yet handle data field {data_field}."
                )

        return output_dict
