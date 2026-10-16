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

"""NCore-backed Waymo lidar reader (the RadymLidar analog for NCore V4 data).

Waymo NCore stores lidar as a *structured sparse ray bundle* per frame, NOT a
dense range image and NOT a flat xyz cloud. Each ray carries:

  * ``model_element``  (N, 2) uint16  -> (row, col) index into the sensor grid
  * ``direction``      (N, 3) float32 -> unit ray direction (sensor frame)
  * per-return ``distance_m`` / ``intensity`` / ``valid_mask``

The sensor grid for the Waymo top lidar is 64 rows x ~2650 cols, defined by a
``RowOffsetStructuredSpinningLidarModel`` whose ``row_elevations_rad`` (64) and
``column_azimuths_rad`` (~2650) tables give the exact beam geometry (the stored
``direction`` equals the angle-table ray, verified cos == 1.0).

This module scatters ``distance_m`` into ``model_element`` to losslessly build a
dense ``(R, C)`` range image (missing grid cells == no-return == range 0, i.e.
raydrop is implicit). That is exactly the ``(row, col, range)`` sparse-triplet
form the existing LidarGEN tar pipeline (``load_each_frame_from_tar_data``)
already consumes, so the rest of ``LidarRangeMapSampler`` works unchanged.

All reads are strictly read-only; the dataset on disk is never modified.
"""

import os
import threading
from typing import Any, List, Optional, Tuple

import numpy as np
import torch
import torch.utils.data
from lru import LRU

from tokenizer.training.datasets.lidar_datasets.data_fileds import DataField

try:
    from ncore.data.v4 import SequenceComponentGroupsReader, SequenceLoaderV4
    from ncore.impl.data.types import FrameTimepoint

    _NCORE_AVAILABLE = True
    _NCORE_IMPORT_ERROR: Optional[Exception] = None
except Exception as _e:  # pragma: no cover - import guard for CPU-only envs
    _NCORE_AVAILABLE = False
    _NCORE_IMPORT_ERROR = _e
    FrameTimepoint = None  # type: ignore


# ----------------------------------------------------------------------------- #
# Geometry helpers (native Waymo grid; uses the model's actual angle tables)
# ----------------------------------------------------------------------------- #
def get_waymo_model_tables(model_parameters) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (row_elevations_rad (R,), column_azimuths_rad (C,),
    row_azimuth_offsets_rad (R,)) from a
    ``RowOffsetStructuredSpinningLidarModelParameters``.

    The per-row azimuth offset is part of the geometry (this is a *row-offset*
    spinning model, like the Hesai P128): the azimuth of cell (r, c) is
    ``column_azimuths_rad[c] + row_azimuth_offsets_rad[r]`` -- see NCore's
    ``LidarModel.elements_to_sensor_angles``."""
    elev = np.asarray(model_parameters.row_elevations_rad, dtype=np.float64)
    azim = np.asarray(model_parameters.column_azimuths_rad, dtype=np.float64)
    row_az_off = np.asarray(model_parameters.row_azimuth_offsets_rad, dtype=np.float64)
    return elev, azim, row_az_off


def waymo_range_map_to_ray_directions(
    row_elevations_rad: np.ndarray,
    column_azimuths_rad: np.ndarray,
    row_azimuth_offsets_rad: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Per-pixel unit ray directions for the native Waymo grid.

    Mirrors NCore's canonical (row,col)->direction EXACTLY (verified against
    ``impl/sensors/lidar.py`` ``elements_to_sensor_angles`` +
    ``sensor_angles_to_sensor_rays``):

        az(r,c) = column_azimuths_rad[c] + row_azimuth_offsets_rad[r]
        el(r,c) = row_elevations_rad[r]
        x = cos(az)cos(el),  y = sin(az)cos(el),  z = sin(el)

    The spinning direction (cw/ccw) is already baked into the stored azimuth
    values, so there is NO extra sign flip. Unlike
    ``utils.lidar_rangemap.range_map_to_ray_directions`` (uniform linspace
    azimuth + degree elevations for Pandar128), this uses the sensor's measured
    tables + per-row offset, so column ``c`` / row ``r`` map to exactly the
    cells ``model_element`` scatters into.

    Returns (R, C, 3) float32 unit vectors in the lidar sensor frame.
    """
    elev = np.asarray(row_elevations_rad, dtype=np.float64)  # (R,)
    azim = np.asarray(column_azimuths_rad, dtype=np.float64)  # (C,)
    if row_azimuth_offsets_rad is None:
        row_off = np.zeros_like(elev)
    else:
        row_off = np.asarray(row_azimuth_offsets_rad, dtype=np.float64)  # (R,)
    # az(r,c) = column_azimuths[c] + row_offset[r]; el(r,c) = elev[r]
    azim_grid = azim[None, :] + row_off[:, None]  # (R, C)
    elev_grid = np.broadcast_to(elev[:, None], azim_grid.shape)  # (R, C)
    x = np.cos(azim_grid) * np.cos(elev_grid)
    y = np.sin(azim_grid) * np.cos(elev_grid)
    z = np.sin(elev_grid)
    return np.stack([x, y, z], axis=-1).astype(np.float32)  # (R, C, 3)


def waymo_resampled_azimuth_table(column_azimuths_rad: np.ndarray, width: int) -> np.ndarray:
    """Azimuth (rad) of each column on a ``width``-wide uniform-azimuth grid that
    the native ``C``-column azimuths are rebinned onto.

    The reader maps a native column ``c`` to target bin ``t = (c * width) // C``.
    The representative azimuth of bin ``t`` is the native azimuth at the bin
    centre native column ``floor((t + 0.5) * C / width)``. Because the native
    Waymo column azimuths are uniform (~0.1359 deg step), this is a uniform
    azimuth table -- it matches LidarDM's 64x2048 Waymo range-image convention
    (``WAYMO_RANGE_CONFIG``: width 2048 vs the 0.13585 deg / 2650-col raycast).

    Returns (width,) float64 azimuths in radians.
    """
    azim = np.asarray(column_azimuths_rad, dtype=np.float64)  # (C,)
    C = azim.shape[0]
    idx = np.minimum(((np.arange(width) + 0.5) * C / width).astype(np.int64), C - 1)
    return azim[idx]


def load_waymo_lidar_geometry(
    root_path: str,
    filter_list_path: Optional[str] = None,
    lidar_id: str = "lidar_top",
    resample_width: int = -1,
    itar_index_tail_read_size: int = 1 << 20,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load fixed Waymo beam geometry as ``(dirs, elev, azim, row_offset)``.

    Opens the FIRST clip in the filter list ONCE to read the sensor's
    ``row_elevations_rad`` / ``column_azimuths_rad`` / ``row_azimuth_offsets_rad``
    tables. All Waymo top-lidar clips share the same structured spinning model, so
    a single representative read suffices -- this mirrors the existing single
    fixed Pandar128 elevation table used for LidarGEN (the loss assumes one sensor
    geometry across the dataset). When ``resample_width > 0`` the azimuth axis is
    the uniform-azimuth rebin grid (``waymo_resampled_azimuth_table``) that the
    reader scatters returns onto; otherwise the native column grid.

    When ``resample_width > 0``, ``azim`` and ``dirs`` describe the same rebinned
    azimuth grid emitted by :class:`WaymoNCoreLidar`. The returned tables are also
    sufficient for projecting arbitrary sensor/world points back to that grid.
    Strictly read-only.
    """
    if not _NCORE_AVAILABLE:
        raise ImportError(
            "load_waymo_lidar_geometry requires the `ncore` package (pip install "
            f"nvidia-ncore). Import failed with: {_NCORE_IMPORT_ERROR}"
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
            f"load_waymo_lidar_geometry: no clips found (root={root_path}, "
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
    elev, azim, row_off = get_waymo_model_tables(lidar.model_parameters)
    if resample_width and int(resample_width) > 0:
        azim = waymo_resampled_azimuth_table(azim, int(resample_width))
    dirs = waymo_range_map_to_ray_directions(elev, azim, row_off)  # (R, W, 3)
    del loader, reader
    return dirs, elev, azim, row_off


def load_waymo_ray_directions(
    root_path: str,
    filter_list_path: Optional[str] = None,
    lidar_id: str = "lidar_top",
    resample_width: int = -1,
    itar_index_tail_read_size: int = 1 << 20,
) -> np.ndarray:
    """Per-pixel unit ray directions ``(R, W, 3)`` for the Waymo range image."""
    dirs, _, _, _ = load_waymo_lidar_geometry(
        root_path=root_path,
        filter_list_path=filter_list_path,
        lidar_id=lidar_id,
        resample_width=resample_width,
        itar_index_tail_read_size=itar_index_tail_read_size,
    )
    return dirs


# ----------------------------------------------------------------------------- #
# Reader
# ----------------------------------------------------------------------------- #
class WaymoNCoreLidar:
    """RadymLidar-compatible reader backed by NCore V4 ``.zarr.itar`` stores.

    Mirrors the surface that ``LidarRangeMapSampler`` depends on:
      * ``custom_fields`` attribute
      * ``size_of_filter_set()``
      * ``_read_rgb_and_lidar(lidar_idx, lidar_frame_idx, data_fields)`` -> dict

    Args mirror ``RadymLidar`` (extra args are accepted/ignored so the sampler
    can build either backend with the same call), plus a few NCore specifics.
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
        # --- NCore specifics ---
        lidar_id: str = "lidar_top",
        return_index: int = 0,
        load_cameras: bool = False,
        itar_index_tail_read_size: int = 1 << 20,
        resample_width: int = -1,
        # --- hdmap (on-the-fly rasterized from dump_waymo_hdmap.py NPZs) ---
        hdmap_root: Optional[str] = None,
        min_range: float = 2.5,
        max_range: float = 75.0,
        hdmap_box_edge_spacing: float = 0.15,
        hdmap_moving_only: bool = False,
        hdmap_include_boxes: bool = True,
        hdmap_map_max_dist: Optional[float] = None,
        hdmap_multichannel: bool = False,
        # --- BEV hdmap: multi-channel top-down occupancy raster (one channel per
        #     semantic class, solid box footprints) for BEV-latent conditioning ---
        hdmap_bev: bool = False,
        hdmap_bev_size: Tuple[int, int] = (256, 256),
        hdmap_bev_pc_range: Tuple[float, float, float, float, float, float] = (
            -75.0, 75.0, -75.0, 75.0, -3.0, 5.0,
        ),
    ):
        if not _NCORE_AVAILABLE:
            raise ImportError(
                "WaymoNCoreLidar requires the `ncore` package (pip install "
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

        self.lidar_id = lidar_id
        self.return_index = return_index
        self.load_cameras = load_cameras
        self.itar_index_tail_read_size = itar_index_tail_read_size
        # >0 -> rebin the native column/azimuth axis onto a uniform grid of this
        # width (e.g. 2048, LidarDM's Waymo range-image convention). The native
        # 2650-col raycast grid is finer than the ~2200 samples/beam, so it has
        # structural gaps; pooling onto 2048 densifies while keeping real
        # raydrop. <=0 -> keep the native grid. Rows (beams) are never resampled.
        self.resample_width = int(resample_width)

        # HDMap: no native NCore component; rasterized on the fly from the
        # per-clip NPZ written by scripts/data/dump_waymo_hdmap.py.
        self.hdmap_root = hdmap_root
        self.min_range = float(min_range)
        self.max_range = float(max_range)
        self.hdmap_box_edge_spacing = float(hdmap_box_edge_spacing)
        self.hdmap_moving_only = bool(hdmap_moving_only)
        self.hdmap_include_boxes = bool(hdmap_include_boxes)
        # One range-image channel per semantic class (lane/road_line/road_edge/
        # crosswalk/vehicle/pedestrian/cyclist) -> hdmap is (N, NUM_CLASSES, R, W).
        # False -> single composited depth channel -> (N, R, W).
        self.hdmap_multichannel = bool(hdmap_multichannel)
        # BEV hdmap raster (top-down, one channel per semantic class).
        self.hdmap_bev = bool(hdmap_bev)
        self.hdmap_bev_size = tuple(hdmap_bev_size)
        self.hdmap_bev_pc_range = tuple(hdmap_bev_pc_range)
        # BEV (xy) crop radius for the static lane graph; None -> use max_range.
        self.hdmap_map_max_dist = (
            self.max_range if hdmap_map_max_dist is None else float(hdmap_map_max_dist)
        )
        self.composers: dict[str, Any] = {}
        if "hdmap" in self.custom_fields and not self.hdmap_root:
            raise NotImplementedError(
                "Waymo NCore has no native hdmap component. Set hdmap_root=<dir of "
                "dump_waymo_hdmap.py NPZs> to rasterize one on the fly, or remove "
                "'hdmap' from custom_fields."
            )

        # Build the clip list (stems). Each clip is a subdir <root>/<stem>/ that
        # holds <stem>.ncore4*.zarr.itar + <stem>.json.
        self.filter_set: List[str] = self._build_filter_set(filter_list_path)
        if len(self.filter_set) == 0:
            raise RuntimeError(
                f"No Waymo NCore clips found (root={root_path}, "
                f"filter_list_path={filter_list_path})."
            )

        # Per-worker handle cache. Opened lazily inside the worker process; the
        # callback closes evicted sequence readers' underlying .itar stores.
        self.worker_id: Optional[int] = None
        self._lock = threading.Lock()
        self.loaders = LRU(self.MAX_LOADERS, callback=self._evict_loader)

    # --------------------------------------------------------------------- #
    @staticmethod
    def _evict_loader(_key, value):
        # IndexedTarStore file objects are closed when the reader is GC'd; we
        # just drop the references here so the LRU can reclaim them.
        loader, reader = value
        del loader, reader

    def _build_filter_set(self, filter_list_path: Optional[str]) -> List[str]:
        if filter_list_path is not None and os.path.isfile(filter_list_path):
            with open(filter_list_path, "r") as f:
                stems = [line.strip() for line in f.readlines() if line.strip()]
            # tolerate ".json"/extension suffixes in the list
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
                cam_name = WAYMO_VIEW_IDX_TO_CAMERA[v]
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
        """Open (and cache) the per-clip HDMap composer."""
        if stem in self.composers:
            return self.composers[stem]
        from tokenizer.training.datasets.lidar_datasets.waymo_hdmap import (
            WaymoHDMapComposer,
        )

        npz_path = os.path.join(self.hdmap_root, f"{stem}.npz")
        if not os.path.isfile(npz_path):
            raise FileNotFoundError(
                f"HDMap NPZ not found: {npz_path} (run scripts/data/dump_waymo_hdmap.py "
                f"--clip {stem} --out_root {self.hdmap_root})."
            )
        elev, azim, row_off = get_waymo_model_tables(lidar.model_parameters)
        comp = WaymoHDMapComposer(
            npz_path,
            elev,
            azim,
            row_off,
            int(lidar.model_parameters.n_columns),
            box_edge_spacing=self.hdmap_box_edge_spacing,
            map_max_dist=self.hdmap_map_max_dist,
        )
        # Cap the cache so a full-dataset scan doesn't grow it unbounded.
        if len(self.composers) >= self.MAX_LOADERS:
            self.composers.pop(next(iter(self.composers)))
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
            # range is written LAST (overwrites). Native grid has no collisions
            # (each ray is a unique cell), so this is a no-op there.
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
        """Mirror RadymLidar._read_rgb_and_lidar, NCore-backed."""
        stem = self.filter_set[lidar_idx]
        loader = self._get_loader(stem)
        lidar = loader.get_lidar_sensor(self.lidar_id)
        frames_count = lidar.frames_count
        frame_idx = self._clamp_frame_indices(lidar_frame_idx, frames_count)

        mp = lidar.model_parameters
        R, C = int(mp.n_rows), int(mp.n_columns)

        output_dict: dict[str | DataField, Any] = {"__key__": stem}

        l2w_start, sensor_pose_start_end, ts = self._poses_for_frames(loader, lidar, frame_idx)
        output_dict["timestamp_list"] = ts
        output_dict["pose_sensor_start_end"] = sensor_pose_start_end
        output_dict["lidar_to_world"] = l2w_start  # (N, 4, 4), sweep-start
        output_dict["rgb_frame_indices"] = frame_idx.tolist()

        for data_field in data_fields:
            if data_field == "custom":
                output_dict[data_field] = {}
                range_list, intensity_list = [], []
                for fi in frame_idx:
                    rmap, imap = self._scatter_frame(lidar, int(fi), R, C)
                    range_list.append(rmap)
                    if imap is not None:
                        intensity_list.append(imap)
                output_dict[data_field]["lidar"] = np.stack(range_list, axis=0)  # [N, R, C]
                if self.load_intensity:
                    output_dict[data_field]["lidar_intensity"] = np.stack(intensity_list, axis=0)
                if "hdmap" in self.custom_fields:
                    comp = self._get_composer(stem, lidar)
                    outW = self.resample_width if self.resample_width > 0 else C
                    hdmap_list = []
                    for k, fi in enumerate(frame_idx):
                        if self.hdmap_multichannel:
                            hdmap_list.append(
                                comp.rasterize_multichannel(
                                    int(fi),
                                    l2w_start[k],
                                    outW,
                                    self.min_range,
                                    self.max_range,
                                    moving_only=self.hdmap_moving_only,
                                )
                            )
                        else:
                            hdmap_list.append(
                                comp.rasterize(
                                    int(fi),
                                    l2w_start[k],
                                    outW,
                                    self.min_range,
                                    self.max_range,
                                    include_boxes=self.hdmap_include_boxes,
                                    moving_only=self.hdmap_moving_only,
                                )
                            )
                    # multichannel -> (N, NUM_CLASSES, R, W); else (N, R, W).
                    output_dict[data_field]["hdmap"] = np.stack(hdmap_list, axis=0)
                    if self.hdmap_bev:
                        bev_list = [
                            comp.rasterize_bev(
                                int(fi),
                                l2w_start[k],
                                self.hdmap_bev_size,
                                self.hdmap_bev_pc_range,
                                include_boxes=self.hdmap_include_boxes,
                                moving_only=self.hdmap_moving_only,
                            )
                            for k, fi in enumerate(frame_idx)
                        ]
                        # (N, NUM_CLASSES, Y, X)
                        output_dict[data_field]["hdmap_bev"] = np.stack(bev_list, axis=0)
            elif data_field == DataField.IMAGE_RGB:
                # Camera RGB conditioning: read each requested view's image
                # nearest (in time) to every sampled lidar frame, as a
                # (N, 3, H, W) float tensor in [0, 1] -- the exact format the
                # sampler's get_rgb() -> process_rgb() expects (matches
                # RadymLidar). Cameras must have been opened (load_cameras=True
                # -> _clip_component_paths added the camera zarrs).
                output_dict[data_field] = {}
                view_indices = (
                    self.view_indices
                    if self.view_indices is not None
                    else list(range(max(self.num_views, 0)))
                )
                for c_view_idx, view_idx in enumerate(view_indices):
                    cam_name = WAYMO_VIEW_IDX_TO_CAMERA[view_idx]
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
                # The image-conditioned RF models only VAE-encode the RGB as
                # cross-attention / canvas context -- they never project points,
                # so intrinsics / camera->lidar extrinsics are unused. Emit empty
                # dicts (the sampler's get_camera_* returns {} -> sample.update
                # is a no-op) instead of fabricating values. Populate from the
                # NCore CameraModel here if a projection-based waymo model needs
                # them later.
                output_dict[data_field] = {}
            else:
                raise NotImplementedError(
                    f"WaymoNCoreLidar does not yet handle data field {data_field}."
                )

        return output_dict


# Camera view index -> NCore camera instance name (Waymo 5-camera rig).
WAYMO_VIEW_IDX_TO_CAMERA = {
    0: "camera_front_50fov",
    1: "camera_front_left_50fov",
    2: "camera_front_right_50fov",
    3: "camera_side_left_50fov",
    4: "camera_side_right_50fov",
}
