# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Public native-grid LiDAR dataset contracts used by LidarVerse.

No split files are distributed with the repository. Paths can be changed with
the environment variables below or by adapting these dictionaries to another
dataset that exposes the same range-map sampler contract.
"""

import os


def _path(name: str, default: str) -> str:
    return os.path.expanduser(os.environ.get(name, default))


def _base_range_video(
    *,
    data_name: str,
    root: str,
    split: str,
    backend: str,
    height: int,
    width: int,
    lidar_length: int,
    lidar_fps: int,
) -> dict:
    return {
        "data_name": data_name,
        "root_path": root,
        "filter_list_path": split,
        "t5_embedding_path": "",
        "custom_folders": [root],
        "custom_fields": ["lidar"],
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
        # The video sampler prepends its first sweep, producing nine frames.
        "sample_n_frames": 8,
        "lidar_length": lidar_length,
        "frame_sampling_method": "sequential",
        "frame_sampling_every_n_frames": 1,
        "sample_lidar_prob": 1.0,
        "is_video_tokenizer": True,
        "load_lidar_only": True,
        "load_rgb_only": False,
        "load_lidar_and_rgb": False,
        "load_lidar_or_rgb": False,
        "RGB_FPS": 30,
        "LIDAR_FPS": lidar_fps,
        "load_t5_embeddings": False,
        "inverse_depth": False,
        "process_input": "mono",
        "process_output": "mono",
        "load_intensity": False,
        "emit_valid_mask": True,
        "dataset_backend": backend,
        "resample_width": width if backend.endswith("_ncore") else -1,
        "center_vehicle_forward": backend == "nuscenes_ncore",
    }


_PAI_ROOT = _path("LIDARVERSE_NVIDIA_PAI_ROOT", "data/nvidia_pai")
_PANDASET_ROOT = _path(
    "LIDARVERSE_PANDASET_ROOT", "data/pandaset_rangemap_64x1024"
)
_NUSCENES_ROOT = _path("LIDARVERSE_NUSCENES_ROOT", "data/nuscenes_ncore")

COMMONDATA_CONFIG = {
    "nvidia_pai_range_video_train": _base_range_video(
        data_name="mads_range_map",
        root=_PAI_ROOT,
        split=_path(
            "LIDARVERSE_NVIDIA_PAI_TRAIN_LIST",
            "data/splits/nvidia_pai_train.lst",
        ),
        backend="radym",
        height=128,
        width=1200,
        lidar_length=200,
        lidar_fps=10,
    ),
    "pandaset_range_video_train": _base_range_video(
        data_name="pandaset_lidar_range_map",
        root=_PANDASET_ROOT,
        split=_path(
            "LIDARVERSE_PANDASET_TRAIN_LIST",
            "data/splits/pandaset_train.lst",
        ),
        backend="pandaset",
        height=64,
        width=1024,
        lidar_length=80,
        lidar_fps=10,
    ),
    "nuscenes_range_video_train": _base_range_video(
        data_name="nuscenes_lidar_range_map",
        root=_NUSCENES_ROOT,
        split=_path(
            "LIDARVERSE_NUSCENES_TRAIN_LIST",
            "data/splits/nuscenes_train.lst",
        ),
        backend="nuscenes_ncore",
        height=32,
        width=1024,
        lidar_length=380,
        lidar_fps=20,
    ),
}

for dataset, env_name, default in (
    (
        "nvidia_pai",
        "LIDARVERSE_NVIDIA_PAI_VAL_LIST",
        "data/splits/nvidia_pai_val.lst",
    ),
    (
        "pandaset",
        "LIDARVERSE_PANDASET_VAL_LIST",
        "data/splits/pandaset_val.lst",
    ),
    (
        "nuscenes",
        "LIDARVERSE_NUSCENES_VAL_LIST",
        "data/splits/nuscenes_val.lst",
    ),
):
    source = f"{dataset}_range_video_train"
    target = f"{dataset}_range_video_val"
    COMMONDATA_CONFIG[target] = dict(COMMONDATA_CONFIG[source])
    COMMONDATA_CONFIG[target]["filter_list_path"] = _path(env_name, default)

# The public PAI layout stores metadata beside its range-map payloads.
for split in ("train", "val"):
    config = COMMONDATA_CONFIG[f"nvidia_pai_range_video_{split}"]
    config["metadata_folder"] = _path(
        "LIDARVERSE_NVIDIA_PAI_METADATA", "data/nvidia_pai/metadata"
    )
    # Native PAI columns are reduced to the trained 1200-column grid.
    config["downsample_factor_col"] = 3


LIDAR_DATASET_NAMES = {
    "nvidia_pai_train": "nvidia_pai_range_video_train",
    "nvidia_pai_val": "nvidia_pai_range_video_val",
    "pandaset_train": "pandaset_range_video_train",
    "pandaset_val": "pandaset_range_video_val",
    "nuscenes_train": "nuscenes_range_video_train",
    "nuscenes_val": "nuscenes_range_video_val",
}
