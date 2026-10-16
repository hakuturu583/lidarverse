# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Public HDMap-to-LiDAR dataset contracts for LidarVerse.

The released training recipe uses NVIDIA PAI and nuScenes. The Waymo entries
are a generic NCore adapter for users who independently prepare that dataset;
they are not part of the released checkpoint's training lineage.
"""

import os


def _path(name: str, default: str) -> str:
    return os.path.expanduser(os.environ.get(name, default))


def _base_hdmap_video(
    *,
    data_name: str,
    root: str,
    split: str,
    hdmap_root: str,
    backend: str,
    height: int,
    width: int,
    lidar_length: int,
    sensor_fps: float,
    sensor_name: str,
) -> dict:
    custom_folders = [root, root] if backend == "radym" else [root]
    return {
        "data_name": data_name,
        "root_path": root,
        "filter_list_path": split,
        "t5_embedding_path": "",
        "custom_folders": custom_folders,
        "custom_fields": ["lidar", "hdmap"],
        "metadata_folder": "",
        "max_range": 100.0,
        "min_range": 2.5,
        "downsample_factor_row": 1,
        "downsample_factor_col": 1,
        "repeat_row": 1,
        "repeat_col": 1,
        "repeat_temporal": 1,
        "lidar_crop_size": [-1, -1],
        "rgb_crop_size": [-1, -1],
        "rgb_resize_size": [-1, -1],
        "lidar_size": [height, width],
        "lidar_resize_size": [-1, -1],
        "downsample_method": "scatter_min",
        "min_value": -1.0,
        "num_views": 0,
        "sample_n_frames": 21,
        "lidar_length": lidar_length,
        "frame_sampling_method": "sequential",
        "frame_sampling_every_n_frames": 1,
        "sample_lidar_prob": 1.0,
        "is_video_tokenizer": False,
        "load_lidar_only": True,
        "load_rgb_only": False,
        "load_lidar_and_rgb": False,
        "load_lidar_or_rgb": False,
        "RGB_FPS": 30,
        "LIDAR_FPS": int(round(sensor_fps)),
        "load_t5_embeddings": False,
        "inverse_depth": False,
        "process_input": "mono",
        "load_intensity": False,
        "dataset_backend": backend,
        "resample_width": width if backend.endswith("_ncore") else -1,
        "center_vehicle_forward": backend == "nuscenes_ncore",
        "hdmap_root": hdmap_root,
        "hdmap_multichannel": False,
        "hdmap_moving_only": False,
        "hdmap_include_boxes": True,
        "hdmap_map_max_dist": 100.0,
        "hdmap_pose_for_lidar_to_world": backend == "nuscenes_ncore",
        "sensor_fps": sensor_fps,
        "sensor_dataset_name": sensor_name,
        "sensor_dataset_id": -1,
    }


_PAI_ROOT = _path("LIDARVERSE_NVIDIA_PAI_ROOT", "data/nvidia_pai")
_NUSCENES_ROOT = _path("LIDARVERSE_NUSCENES_ROOT", "data/nuscenes_ncore")
_WAYMO_TRAIN_ROOT = _path(
    "LIDARVERSE_WAYMO_ROOT", "data/waymo_ncore/training"
)
_WAYMO_VAL_ROOT = _path(
    "LIDARVERSE_WAYMO_VAL_ROOT", "data/waymo_ncore/validation"
)

COMMONDATA_CONFIG = {
    "lidarverse_nvidia_pai_train": _base_hdmap_video(
        data_name="mads_range_map",
        root=_PAI_ROOT,
        split=_path(
            "LIDARVERSE_NVIDIA_PAI_TRAIN_LIST",
            "data/splits/nvidia_pai_train.lst",
        ),
        hdmap_root=_PAI_ROOT,
        backend="radym",
        height=128,
        width=1200,
        lidar_length=200,
        sensor_fps=10.0,
        sensor_name="nvidia_pai",
    ),
    "lidarverse_nuscenes_train": _base_hdmap_video(
        data_name="nuscenes_lidar_range_map",
        root=_NUSCENES_ROOT,
        split=_path(
            "LIDARVERSE_NUSCENES_TRAIN_LIST",
            "data/splits/nuscenes_train.lst",
        ),
        hdmap_root=_path(
            "LIDARVERSE_NUSCENES_HDMAP_ROOT", "data/nuscenes_hdmap"
        ),
        backend="nuscenes_ncore",
        height=32,
        width=1024,
        lidar_length=380,
        sensor_fps=2.0,
        sensor_name="nuscenes",
    ),
    "lidarverse_waymo_train": _base_hdmap_video(
        data_name="waymo_lidar_range_map",
        root=_WAYMO_TRAIN_ROOT,
        split=_path(
            "LIDARVERSE_WAYMO_TRAIN_LIST", "data/splits/waymo_train.lst"
        ),
        hdmap_root=_path(
            "LIDARVERSE_WAYMO_HDMAP_ROOT", "data/waymo_hdmap/training"
        ),
        backend="waymo_ncore",
        height=64,
        width=1024,
        lidar_length=199,
        sensor_fps=10.0,
        sensor_name="waymo",
    ),
}

for dataset, env_name, default in (
    (
        "nvidia_pai",
        "LIDARVERSE_NVIDIA_PAI_VAL_LIST",
        "data/splits/nvidia_pai_val.lst",
    ),
    (
        "nuscenes",
        "LIDARVERSE_NUSCENES_VAL_LIST",
        "data/splits/nuscenes_val.lst",
    ),
):
    source = f"lidarverse_{dataset}_train"
    target = f"lidarverse_{dataset}_val"
    COMMONDATA_CONFIG[target] = dict(COMMONDATA_CONFIG[source])
    COMMONDATA_CONFIG[target]["filter_list_path"] = _path(env_name, default)

COMMONDATA_CONFIG["lidarverse_waymo_val"] = _base_hdmap_video(
    data_name="waymo_lidar_range_map",
    root=_WAYMO_VAL_ROOT,
    split=_path(
        "LIDARVERSE_WAYMO_VAL_LIST", "data/splits/waymo_val.lst"
    ),
    hdmap_root=_path(
        "LIDARVERSE_WAYMO_VAL_HDMAP_ROOT", "data/waymo_hdmap/validation"
    ),
    backend="waymo_ncore",
    height=64,
    width=1024,
    lidar_length=199,
    sensor_fps=10.0,
    sensor_name="waymo",
)

# Native PAI columns are reduced to the trained 1200-column grid.
for split in ("train", "val"):
    config = COMMONDATA_CONFIG[f"lidarverse_nvidia_pai_{split}"]
    config["metadata_folder"] = _path(
        "LIDARVERSE_NVIDIA_PAI_METADATA", "data/nvidia_pai/metadata"
    )
    config["downsample_factor_col"] = 3
