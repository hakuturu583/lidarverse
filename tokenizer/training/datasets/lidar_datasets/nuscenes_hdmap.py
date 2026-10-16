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

"""nuScenes on-the-fly HDMap composer.

Thin subclass of ``WaymoHDMapComposer``: the rasterization / projection / box
geometry is dataset-agnostic and reused verbatim. The ONLY nuScenes-specific
piece is the projection frame. NCore's re-derived pose graph drifts from
nuScenes' ``ego_pose`` trajectory, so instead of the NCore ``lidar_to_world`` we
project the map/boxes (kept in the nuScenes GLOBAL frame by
``scripts/data/dump_nuscenes_hdmap.py``) with the per-sweep nuScenes
``lidar_to_global`` stored alongside them -- nuScenes map + ego_pose are
self-consistent, so the map lands on the actual returns.

Frame indices are 1:1 with the NCore lidar frames (verified: same ordered sweep
list, same count), so a sampled NCore ``frame_idx`` indexes this NPZ directly.
"""

import numpy as np

from tokenizer.training.datasets.lidar_datasets.waymo_hdmap import WaymoHDMapComposer


class NuScenesHDMapComposer(WaymoHDMapComposer):
    """Per-scene nuScenes HDMap rasterizer (map/boxes in the nuScenes global frame)."""

    def __init__(
        self,
        *args,
        virtual_sensor_height_m=None,
        virtual_sensor_yaw_deg=0.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # npz_path is the first positional (parent already validated + loaded it).
        npz_path = args[0] if args else kwargs["npz_path"]
        d = np.load(npz_path, allow_pickle=False)
        # per-sweep nuScenes lidar->global; the frame the composer projects from.
        self.lidar_to_global = d["lidar_to_global"].astype(np.float32)  # (F,4,4)
        self.frame_timestamps_us = d["frame_timestamps_us"].astype(np.int64)  # (F,)
        self.virtual_sensor_height_m = (
            None if virtual_sensor_height_m is None
            else float(virtual_sensor_height_m)
        )
        self.virtual_sensor_yaw_deg = float(virtual_sensor_yaw_deg)

    def pose_for_frame(self, frame_idx: int) -> np.ndarray:
        """nuScenes lidar->global for sweep ``frame_idx`` (the projection frame)."""
        pose = self.lidar_to_global[int(frame_idx)]
        if (
            self.virtual_sensor_height_m is None
            and self.virtual_sensor_yaw_deg == 0.0
        ):
            return pose
        pose = pose.copy()
        if self.virtual_sensor_yaw_deg != 0.0:
            # nuScenes LIDAR_TOP uses x-right/y-forward. The canonical
            # nuScenes->Waymo config supplies the yaw needed to map the NCore
            # Waymo TOP sensor axes into these native lidar axes before the
            # stored lidar->global transform.
            yaw = np.deg2rad(self.virtual_sensor_yaw_deg)
            c, s = np.cos(yaw), np.sin(yaw)
            correction = np.array(
                [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
                dtype=np.float32,
            )
            pose[:3, :3] = pose[:3, :3] @ correction
        if self.virtual_sensor_height_m is not None:
            # nuScenes maps are 2D at global z=0. Canonical generation therefore
            # uses an explicit virtual sensor height instead of inheriting the
            # nuScenes HDL-32E height (~1.84 m).
            pose[2, 3] = self.virtual_sensor_height_m
        return pose
