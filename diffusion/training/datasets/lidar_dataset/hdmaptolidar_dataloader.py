# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dataloader for hdmap-to-lidar diffusion training.

Loads lidar range-map + HD-map conditioning, reusing the diffusion
``LidarRangeMapSampler`` infrastructure and adding the metadata fields
expected by the Cosmos diffusion training loop (t5 embeddings, fps, etc.).
"""

import gc
import json
import os

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset, Sampler

from diffusion.training.datasets.lidar_dataset.configs import COMMONDATA_CONFIG
from diffusion.training.datasets.lidar_dataset.imagetolidar_dataloader import (
    LidarRangeMapSampler,
    dict_collation_fn,
    frames_as_batch_collation_fn,
)
from diffusion.training.datasets.lidar_dataset.forecast_warp import (
    compose_normalized_range_channels,
    warp_blockwise_history_to_video,
)
from diffusion.training.datasets.lidar_dataset.waymo_track_ids import (
    resolve_waymo_box_track_ids_from_ncore,
)
from tokenizer.training.datasets.lidar_datasets.data_fileds import DataField
from tokenizer.training.datasets.lidar_datasets.waymo_ncore import (
    load_waymo_lidar_geometry,
)
from utils import log
from utils.lazy_config import LazyCall as L

try:
    from megatron.core import parallel_state
except ImportError:
    pass


class HdmapLidarSampler(LidarRangeMapSampler):
    """Dataset that returns lidar range-map + HD-map with diffusion metadata.

    Inherits all lidar/hdmap loading logic from the existing
    ``LidarRangeMapSampler`` and adds the metadata fields required by the
    Cosmos diffusion training loop.
    """

    def __init__(
        self,
        blockwise_warped_history_condition: bool = False,
        blockwise_warped_history_block_size: int = 3,
        blockwise_warped_history_mode: str = "all",
        autoregressive_pair: bool = False,
        autoregressive_previous_root: str = "",
        autoregressive_generated_previous_prob: float = 0.0,
        autoregressive_include_static_boxes: bool = False,
        synthesize_null_hdmap: bool = False,
        return_hdmap_semantic: bool = False,
        return_moving_boxes: bool = False,
        resolve_moving_box_track_ids: bool = False,
        return_full_lidar_metadata: bool = False,
        moving_box_limit: int = 256,
        moving_box_root: str = "",
        sensor_fps: float = 24.0,
        emit_sensor_metadata: bool = False,
        sensor_dataset_id: int = -1,
        sensor_dataset_name: str = "",
        **kwargs,
    ):
        self.sensor_fps = float(sensor_fps)
        if self.sensor_fps <= 0.0:
            raise ValueError(
                f"sensor_fps must be positive, got {self.sensor_fps}"
            )
        self.emit_sensor_metadata = bool(emit_sensor_metadata)
        self.sensor_dataset_id = int(sensor_dataset_id)
        self.sensor_dataset_name = str(
            sensor_dataset_name or kwargs.get("data_name", "")
        )
        self._sensor_root_path = kwargs.get("root_path")
        self._sensor_filter_list_path = kwargs.get("filter_list_path")
        self._sensor_center_vehicle_forward = bool(
            kwargs.get("center_vehicle_forward", True)
        )
        self._sensor_ray_directions = None
        self.autoregressive_pair = bool(autoregressive_pair)
        self.autoregressive_target_frames = int(
            kwargs.get("sample_n_frames", 1)
        )
        if self.autoregressive_pair:
            if self.autoregressive_target_frames != 1:
                raise ValueError(
                    "autoregressive_pair currently requires "
                    "sample_n_frames=1"
                )
            if bool(kwargs.get("is_video_tokenizer", False)):
                raise ValueError(
                    "autoregressive_pair requires a per-frame tokenizer"
                )
            kwargs["sample_n_frames"] = 2
        if kwargs.get("dataset_backend", "radym") == "radym":
            custom_folders = kwargs.get("custom_folders")
            custom_fields = kwargs.get("custom_fields")
            if (
                custom_folders is not None
                and custom_fields is not None
                and len(custom_folders) != len(custom_fields)
            ):
                raise ValueError(
                    "Radym custom_folders and custom_fields must have the "
                    "same length, got "
                    f"{len(custom_folders)} and {len(custom_fields)}"
                )
        super().__init__(**kwargs)
        self.autoregressive_previous_root = str(
            autoregressive_previous_root
        )
        self.autoregressive_generated_previous_prob = float(
            autoregressive_generated_previous_prob
        )
        self.autoregressive_include_static_boxes = bool(
            autoregressive_include_static_boxes
        )
        if not 0.0 <= self.autoregressive_generated_previous_prob <= 1.0:
            raise ValueError(
                "autoregressive_generated_previous_prob must be in [0,1]"
            )
        if (
            self.autoregressive_generated_previous_prob > 0.0
            and not self.autoregressive_previous_root
        ):
            raise ValueError(
                "generated previous-frame sampling requires "
                "autoregressive_previous_root"
            )
        self.synthesize_null_hdmap = bool(synthesize_null_hdmap)
        self.return_hdmap_semantic = bool(return_hdmap_semantic)
        self.return_moving_boxes = bool(return_moving_boxes)
        self.resolve_moving_box_track_ids = bool(
            resolve_moving_box_track_ids
        )
        self.return_full_lidar_metadata = bool(
            return_full_lidar_metadata
        )
        self.moving_box_limit = int(moving_box_limit)
        self.moving_box_root = str(moving_box_root)
        if (
            self.return_hdmap_semantic
            and self.dataset_backend != "waymo_ncore"
        ):
            raise ValueError(
                "semantic HDMap output requires "
                "dataset_backend='waymo_ncore'"
            )
        if self.return_hdmap_semantic and not bool(
            self._hdmap_kwargs["hdmap_multichannel"]
        ):
            raise ValueError(
                "return_hdmap_semantic requires hdmap_multichannel=True"
            )
        if self.return_moving_boxes and self.moving_box_limit <= 0:
            raise ValueError("moving_box_limit must be positive")
        if (
            self.return_moving_boxes
            and self.dataset_backend == "radym"
            and not self.moving_box_root
        ):
            raise ValueError(
                "radym moving-box output requires moving_box_root"
            )
        if (
            self.resolve_moving_box_track_ids
            and self.dataset_backend != "waymo_ncore"
        ):
            raise ValueError(
                "NCore track-ID resolution is only defined for Waymo"
            )
        if self.synthesize_null_hdmap:
            # Preserve the LiDAR + HDMap conditioner view count even though the
            # underlying NCore reader was asked to load only LiDAR.
            self.plus = 2
        self.blockwise_warped_history_condition = bool(
            blockwise_warped_history_condition
        )
        self.blockwise_warped_history_block_size = int(
            blockwise_warped_history_block_size
        )
        self.blockwise_warped_history_mode = str(
            blockwise_warped_history_mode
        )
        self._blockwise_warp_geometry = None
        self._blockwise_warp_geometry_args = None
        if self.blockwise_warped_history_condition:
            if self.dataset_backend != "waymo_ncore":
                raise ValueError(
                    "blockwise warped-history conditioning requires "
                    f"dataset_backend='waymo_ncore', got {self.dataset_backend!r}"
                )
            if self.process_input != "mono":
                raise ValueError(
                    "blockwise warped-history conditioning requires "
                    f"process_input='mono', got {self.process_input!r}"
                )
            if not self.is_video_tokenizer:
                raise ValueError(
                    "blockwise warped-history conditioning requires a video "
                    "tokenizer with frame-0 prepend"
                )
            if self.blockwise_warped_history_block_size <= 0:
                raise ValueError(
                    "blockwise_warped_history_block_size must be positive"
                )
            if self.blockwise_warped_history_mode not in ("latest", "all"):
                raise ValueError(
                    "blockwise_warped_history_mode must be 'latest' or 'all', "
                    f"got {self.blockwise_warped_history_mode!r}"
                )
            self._blockwise_warp_geometry_args = dict(
                root_path=kwargs["root_path"],
                filter_list_path=kwargs.get("filter_list_path"),
                resample_width=int(kwargs.get("resample_width", -1)),
            )

    def get_sensor_metadata(self, height: int, width: int) -> dict:
        """Return the calibrated native ray grid for cross-dataset training."""
        if not self.emit_sensor_metadata:
            return {}
        if self._sensor_ray_directions is None:
            if self.dataset_backend == "waymo_ncore":
                from tokenizer.training.datasets.lidar_datasets.waymo_ncore import (
                    load_waymo_ray_directions,
                )

                rays = load_waymo_ray_directions(
                    root_path=self._sensor_root_path,
                    filter_list_path=self._sensor_filter_list_path,
                    lidar_id="lidar_top",
                    resample_width=self.resample_width,
                )
            elif self.dataset_backend == "nuscenes_ncore":
                from tokenizer.training.datasets.lidar_datasets.nuscenes_ncore import (
                    load_nuscenes_ray_directions,
                )

                rays = load_nuscenes_ray_directions(
                    root_path=self._sensor_root_path,
                    filter_list_path=self._sensor_filter_list_path,
                    lidar_id="lidar_top",
                    resample_width=self.resample_width,
                    center_vehicle_forward=self._sensor_center_vehicle_forward,
                )
            else:
                from utils.lidar_rangemap import (
                    load_pandar128_elevations,
                    range_map_to_ray_directions,
                )

                elevations = torch.tensor(
                    load_pandar128_elevations(), dtype=torch.float32
                )
                rays = range_map_to_ray_directions(
                    width // self.repeat_col, elevations
                )
            rays = torch.as_tensor(rays, dtype=torch.float32)
            rays = rays.repeat_interleave(self.repeat_row, dim=0)
            rays = rays.repeat_interleave(self.repeat_col, dim=1)
            if tuple(rays.shape) != (height, width, 3):
                raise ValueError(
                    "sensor rays must match the repeated native grid: "
                    f"rays={tuple(rays.shape)}, sample={(height, width, 3)}"
                )
            self._sensor_ray_directions = rays.contiguous()
        return {
            "sensor_ray_directions": self._sensor_ray_directions,
            "sensor_depth_bounds": torch.tensor(
                [self.min_distance, self.max_distance], dtype=torch.float32
            ),
            "sensor_dataset_id": self.sensor_dataset_id,
            "sensor_dataset_name": self.sensor_dataset_name,
        }

    def _radym_moving_box_arrays(self, example):
        key = str(example["__key__"])
        frame_names = []
        for value in example["rgb_frame_indices"]:
            frame_name = str(value)
            if frame_name.isdigit():
                frame_name = f"{int(frame_name):06d}"
            frame_names.append(frame_name)
        tar_handle = self.dataset._get_tar_handle(
            self.moving_box_root, "all_object_info", key
        )
        frame_payloads = []
        actor_names = set()
        for frame_name in frame_names:
            member_name = (
                f"{key}.{frame_name}.all_object_info.json"
            )
            member = tar_handle.extractfile(member_name)
            if member is None:
                raise FileNotFoundError(
                    f"missing moving-box metadata {member_name}"
                )
            payload = json.load(member)
            frame_payloads.append(payload)
            actor_names.update(payload)
        ordered_names = sorted(
            actor_names,
            key=lambda value: (
                0, int(value)
            ) if str(value).isdigit() else (1, str(value)),
        )
        track_ids = {
            actor_name: index
            for index, actor_name in enumerate(ordered_names)
        }
        type_ids = {"Car": 0, "Pedestrian": 1, "Cyclist": 2}
        box_frame = []
        box_o2w = []
        box_lwh = []
        box_type = []
        box_moving = []
        box_track_id = []
        for local_frame, payload in enumerate(frame_payloads):
            for actor_name, actor in payload.items():
                box_frame.append(local_frame)
                box_o2w.append(actor["object_to_world"])
                box_lwh.append(actor["object_lwh"])
                box_type.append(type_ids.get(actor["object_type"], -1))
                box_moving.append(bool(actor["object_is_moving"]))
                box_track_id.append(track_ids[actor_name])
        if box_frame:
            return (
                np.asarray(box_frame, dtype=np.int64),
                np.asarray(box_o2w, dtype=np.float32),
                np.asarray(box_lwh, dtype=np.float32),
                np.asarray(box_type, dtype=np.int64),
                np.asarray(box_moving, dtype=bool),
                np.asarray(box_track_id, dtype=np.int64),
                np.arange(len(frame_names), dtype=np.int64),
            )
        return (
            np.zeros((0,), dtype=np.int64),
            np.zeros((0, 4, 4), dtype=np.float32),
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=bool),
            np.zeros((0,), dtype=np.int64),
            np.arange(len(frame_names), dtype=np.int64),
        )

    def _moving_boxes_for_sample(self, example):
        if self.dataset_backend == "radym":
            (
                box_frame,
                box_o2w,
                box_lwh,
                box_type,
                box_moving,
                track_ids,
                frame_indices,
            ) = self._radym_moving_box_arrays(example)
        else:
            composer = self.dataset.composers.get(example["__key__"])
            if composer is None:
                raise RuntimeError(
                    "moving-box output requires a cached HDMap composer"
                )
            track_ids = getattr(
                composer, "_resolved_box_track_ids", None
            )
            if track_ids is None and self.resolve_moving_box_track_ids:
                loader = self.dataset._get_loader(example["__key__"])
                lidar_sensor = loader.get_lidar_sensor(
                    self.dataset.lidar_id
                )
                track_ids, metadata = (
                    resolve_waymo_box_track_ids_from_ncore(
                        hdmap_composer=composer,
                        sequence_loader=loader,
                        lidar_sensor=lidar_sensor,
                    )
                )
                composer._resolved_box_track_ids = track_ids
                composer._resolved_box_track_id_metadata = metadata
            if track_ids is None:
                track_ids = composer.box_track_id
            box_frame = np.asarray(composer.box_frame, dtype=np.int64)
            box_o2w = np.asarray(composer.box_o2w, dtype=np.float32)
            box_lwh = np.asarray(composer.box_lwh, dtype=np.float32)
            box_type = np.asarray(composer.box_type, dtype=np.int64)
            box_moving = np.asarray(composer.box_moving, dtype=bool)
            track_ids = np.asarray(track_ids, dtype=np.int64)
            frame_indices = np.asarray(
                example["rgb_frame_indices"], dtype=np.int64
            )
        if bool((box_moving & (track_ids < 0)).any()):
            raise ValueError(
                "moving actors have no persistent track IDs; refresh the "
                "dataset's HDMap/box metadata"
            )
        moving_rows = box_moving
        dynamic_track_ids = np.unique(track_ids[moving_rows])
        dynamic_rows = np.isin(track_ids, dynamic_track_ids)
        selected_rows = (
            np.ones_like(dynamic_rows)
            if self.autoregressive_include_static_boxes
            else dynamic_rows
        )
        num_frames = int(frame_indices.shape[0])
        object_to_world = np.zeros(
            (num_frames, self.moving_box_limit, 4, 4),
            dtype=np.float32,
        )
        lwh = np.zeros(
            (num_frames, self.moving_box_limit, 3), dtype=np.float32
        )
        valid = np.zeros(
            (num_frames, self.moving_box_limit), dtype=bool
        )
        sampled_track_ids = np.full(
            (num_frames, self.moving_box_limit), -1, dtype=np.int64
        )
        sampled_box_types = np.full(
            (num_frames, self.moving_box_limit), -1, dtype=np.int64
        )
        truncated = 0
        for local_idx, frame_idx in enumerate(frame_indices):
            indices = np.nonzero(
                (box_frame == int(frame_idx))
                & selected_rows
            )[0]
            truncated += max(0, int(indices.shape[0]) - self.moving_box_limit)
            indices = indices[: self.moving_box_limit]
            count = int(indices.shape[0])
            if count == 0:
                continue
            object_to_world[local_idx, :count] = box_o2w[indices]
            lwh[local_idx, :count] = box_lwh[indices]
            sampled_track_ids[local_idx, :count] = track_ids[indices]
            sampled_box_types[local_idx, :count] = box_type[indices]
            valid[local_idx, :count] = True
        return (
            torch.from_numpy(object_to_world),
            torch.from_numpy(lwh),
            torch.from_numpy(valid),
            torch.from_numpy(sampled_track_ids),
            torch.from_numpy(sampled_box_types),
            truncated,
        )

    def _generated_previous_frame(
        self, clip_key: str, native_frame_idx: int
    ) -> torch.Tensor:
        path = os.path.join(
            self.autoregressive_previous_root, f"{clip_key}.npz"
        )
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"missing DAgger rollout for {clip_key!r}: {path}"
            )
        with np.load(path, allow_pickle=False) as payload:
            frames = np.asarray(payload["range"], dtype=np.float32)
            frame_indices = np.asarray(
                payload["frame_indices"], dtype=np.int64
            )
        matches = np.nonzero(frame_indices == int(native_frame_idx))[0]
        if matches.shape[0] != 1:
            raise RuntimeError(
                f"{path} must contain native frame {native_frame_idx} "
                f"exactly once, found {matches.shape[0]}"
            )
        frame = frames[int(matches[0])]
        if frame.shape != tuple(self.lidar_size):
            raise ValueError(
                f"DAgger frame shape {frame.shape} does not match "
                f"lidar_size={tuple(self.lidar_size)}"
            )
        return torch.from_numpy(frame)[None, None]

    def _sample_generated_previous_frame(
        self, clip_key: str
    ) -> tuple[int, torch.Tensor]:
        path = os.path.join(
            self.autoregressive_previous_root, f"{clip_key}.npz"
        )
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"missing DAgger rollout for {clip_key!r}: {path}"
            )
        with np.load(path, allow_pickle=False) as payload:
            frames = np.asarray(payload["range"], dtype=np.float32)
            frame_indices = np.asarray(
                payload["frame_indices"], dtype=np.int64
            )
        if frames.shape[0] != frame_indices.shape[0] or frames.shape[0] == 0:
            raise ValueError(
                f"invalid DAgger bank shapes in {path}: "
                f"range={frames.shape}, frame_indices={frame_indices.shape}"
            )
        selected = int(np.random.randint(0, frames.shape[0]))
        frame = frames[selected]
        if frame.shape != tuple(self.lidar_size):
            raise ValueError(
                f"DAgger frame shape {frame.shape} does not match "
                f"lidar_size={tuple(self.lidar_size)}"
            )
        return int(frame_indices[selected]), torch.from_numpy(frame)[None, None]

    def __getitem__(self, idx):
        # Deterministic parity mode: caller-provided (clip_idx, native rows) via
        # ``self._explicit_windows`` (set e.g. by scripts/dump_hdmap_gen_ncore.py
        # to align with an r2flow manifest). Mirrors the base
        # ``LidarRangeMapSampler.__getitem__`` -- this override otherwise drops
        # the explicit-windows branch and silently falls back to the random /
        # frame-index path, ignoring the requested (clip, frame).
        if self._explicit_windows is not None:
            lidar_data_idx, lidar_frame_indices = self._explicit_windows[idx]
            lidar_frame_indices = np.asarray(lidar_frame_indices, dtype=np.int64)
        else:
            lidar_data_idx = (idx % self.n_data_per_node) + self.data_start_idx
            lidar_frame_indices = self.sample_frame_indices(load_lidar=True)
        use_generated_previous = (
            self.autoregressive_pair
            and self.autoregressive_generated_previous_prob > 0.0
            and np.random.random()
            < self.autoregressive_generated_previous_prob
        )
        generated_previous = None
        if use_generated_previous:
            if self._explicit_windows is None:
                clip_key = self.dataset.filter_set[lidar_data_idx]
                (
                    generated_frame_idx,
                    generated_previous,
                ) = self._sample_generated_previous_frame(clip_key)
                lidar_frame_indices = np.asarray(
                    [generated_frame_idx, generated_frame_idx + 1],
                    dtype=np.int64,
                )
            else:
                generated_previous = self._generated_previous_frame(
                    self.dataset.filter_set[lidar_data_idx],
                    int(lidar_frame_indices[0]),
                )

        # When cameras are requested, additionally pull RGB + intrinsics +
        # Camera-to-LiDAR transforms supplied by the configured dataset root.
        data_fields = ["custom"]
        if self.load_lidar_and_rgb:
            data_fields = [
                "custom",
                DataField.CAMERA_C2LIDAR_TRANSFORM,
                DataField.CAMERA_INTRINSICS,
                DataField.IMAGE_RGB,
            ]

        try:
            example = self.dataset._read_rgb_and_lidar(
                lidar_data_idx, lidar_frame_indices, data_fields
            )
        except Exception as e:
            log.info(f"Error loading data for index {idx}: {e}", rank0_only=False)
            return self.__getitem__(np.random.randint(0, self.__len__()))

        lidar_rangemap = self.get_lidar(example)
        if lidar_rangemap is None:
            log.info(f"Lidar rangemap missing for index {idx}, retrying", rank0_only=False)
            return self.__getitem__(np.random.randint(0, self.__len__()))

        lidar_hdmap = self.get_lidar_hdmap(example)
        if lidar_hdmap is None:
            if self.synthesize_null_hdmap:
                lidar_hdmap = torch.full_like(
                    lidar_rangemap[:1], self.min_value
                )
            else:
                log.info(
                    f"HDmap missing for index {idx}, retrying",
                    rank0_only=False,
                )
                return self.__getitem__(np.random.randint(0, self.__len__()))
        autoregressive_previous = None
        autoregressive_lidar_to_world = None
        autoregressive_frame_indices = None
        if self.autoregressive_pair:
            if lidar_rangemap.shape[1] != 2 or lidar_hdmap.shape[1] != 2:
                raise RuntimeError(
                    "autoregressive pair must load exactly two frames, got "
                    f"lidar={tuple(lidar_rangemap.shape)}, "
                    f"hdmap={tuple(lidar_hdmap.shape)}"
                )
            autoregressive_previous = lidar_rangemap[:, :1].clone()
            autoregressive_lidar_to_world = np.asarray(
                example["lidar_to_world"], dtype=np.float32
            )
            autoregressive_frame_indices = np.asarray(
                example["rgb_frame_indices"], dtype=np.int64
            )
            if generated_previous is not None:
                autoregressive_previous = generated_previous
            lidar_rangemap = lidar_rangemap[:, 1:2]
            lidar_hdmap = lidar_hdmap[:, 1:2]
        lidar_hdmap_semantic = None
        if lidar_hdmap.shape[0] > 1:
            lidar_hdmap_semantic = lidar_hdmap
            lidar_hdmap = compose_normalized_range_channels(
                lidar_hdmap_semantic,
                channel_dim=0,
                min_value=self.min_value,
                inverse_depth=self.inverse_depth,
            )
        if self.return_hdmap_semantic and lidar_hdmap_semantic is None:
            raise RuntimeError(
                "semantic HDMap output requested but the reader returned a "
                "single composited channel"
            )

        try:
            t5_embed = self.prepare_t5_embeddings(example, lidar_frame_indices)
        except (FileNotFoundError, RuntimeError) as e:
            log.warning(
                f"T5 embedding load failed for index {idx} "
                f"(key={example['__key__']}): {e}; retrying with new index",
                rank0_only=False,
            )
            return self.__getitem__(np.random.randint(0, self.__len__()))

        bev_hdmap = self.get_lidar_hdmap_bev(example)

        metadata_slice = (
            slice(None)
            if self.return_full_lidar_metadata
            else slice(-1, None)
        )
        sample = {
            "video": lidar_rangemap,
            "lidar_hdmap": lidar_hdmap,
            "__key__": example["__key__"],
            "clip_name": "%s-%s-%d" % (self.data_name, example["__key__"], idx),
            "data_type": "lidar",
            "num_frames": (
                self.autoregressive_target_frames
                if self.autoregressive_pair
                else self.sample_n_frames
            ),
            "is_preprocessed": True,
            "image_size": torch.tensor(self.lidar_size),
            "padding_mask": torch.zeros(1, self.lidar_size[0], self.lidar_size[1]),
            "fps": self.sensor_fps,
            "sensor_dataset_id": int(self.sensor_dataset_id),
            "sensor_dataset_name": str(self.sensor_dataset_name),
            "t5_text_embeddings": t5_embed["t5_text_embeddings"],
            "t5_text_mask": t5_embed["t5_text_mask"],
            # Sweep-start (and start/end) ego poses for the *actually sampled*
            # frames (sequential sampling uses a random start_idx) -- forwarded
            # so --save_poses dumps poses aligned to the dumped range maps
            # instead of re-deriving from metadata assuming start frame 0.
            "lidar_to_world": np.asarray(
                example["lidar_to_world"][metadata_slice],
                dtype=np.float32,
            ),
            "pose_sensor_start_end": np.asarray(
                example["pose_sensor_start_end"][metadata_slice],
                dtype=np.float32,
            ),
            # Actual sampled lidar frame indices (post frame-sampling). Mirrors
            # the base image sampler; consumed by verification viz that re-reads
            # the raw cloud/camera for the SAME frames (viz_waymo_cam5_overlay).
            "rgb_frame_indices": np.asarray(
                example["rgb_frame_indices"][metadata_slice],
                dtype=np.int64,
            ),
        }
        sample.update(
            self.get_sensor_metadata(*lidar_rangemap.shape[-2:])
        )
        if self.autoregressive_pair:
            sample["autoregressive_previous_video"] = (
                autoregressive_previous
            )
            sample["autoregressive_lidar_to_world"] = (
                autoregressive_lidar_to_world
            )
            sample["autoregressive_frame_indices"] = (
                autoregressive_frame_indices
            )
        if bev_hdmap is not None:
            sample["bev_hdmap"] = bev_hdmap
        if self.return_hdmap_semantic:
            sample["lidar_hdmap_semantic"] = lidar_hdmap_semantic
        if self.return_moving_boxes:
            (
                moving_box_object_to_world,
                moving_box_lwh,
                moving_box_valid,
                moving_box_track_id,
                moving_box_type,
                moving_box_truncated,
            ) = self._moving_boxes_for_sample(example)
            sample["moving_box_object_to_world"] = (
                moving_box_object_to_world
            )
            sample["moving_box_lwh"] = moving_box_lwh
            sample["moving_box_valid"] = moving_box_valid
            sample["moving_box_track_id"] = moving_box_track_id
            sample["moving_box_type"] = moving_box_type
            sample["moving_box_truncated"] = moving_box_truncated

        if self.blockwise_warped_history_condition:
            if self._blockwise_warp_geometry is None:
                self._blockwise_warp_geometry = load_waymo_lidar_geometry(
                    **self._blockwise_warp_geometry_args
                )
            ray_dirs, elev, azim, row_off = self._blockwise_warp_geometry
            warped, _ = warp_blockwise_history_to_video(
                video=sample["video"],
                lidar_to_world=sample["lidar_to_world"],
                block_size=self.blockwise_warped_history_block_size,
                ray_dirs=ray_dirs,
                row_elevations_rad=elev,
                column_azimuths_rad=azim,
                row_azimuth_offsets_rad=row_off,
                min_range=self.min_distance,
                max_range=self.max_distance,
                min_value=self.min_value,
                inverse_depth=self.inverse_depth,
                history_mode=self.blockwise_warped_history_mode,
            )
            sample["lidar_blockwise_warped_history"] = torch.from_numpy(warped)

        if self.load_lidar_and_rgb:
            # image_rgb_<v> / camera_intrinsics_<v> / camera_c2lidar_transform_<v>
            for getter in (
                self.get_rgb,
                self.get_camera_intrinsics,
                self.get_camera_c2lidar_transform,
            ):
                cam_dict = getter(example)
                if cam_dict is not None:
                    sample.update(cam_dict)

        del example
        gc.collect()
        return sample


class HdmapLidarDataset:
    """Lazy-constructable dataset config that builds ``HdmapLidarSampler``."""

    def __init__(self, dataset_name: str, **kwargs):
        self.dataset_name = dataset_name
        self.dataset_config = OmegaConf.create(COMMONDATA_CONFIG[dataset_name])
        self.extra_config = OmegaConf.create(kwargs)

    def build_dataset(self):
        if self.extra_config:
            merged_config = OmegaConf.merge(self.dataset_config, self.extra_config)
        else:
            merged_config = self.dataset_config
        config_dict = OmegaConf.to_container(merged_config, resolve=True)
        return HdmapLidarSampler(**config_dict)


class HdmapLidarDataLoader(DataLoader):
    def __init__(self, dataset, batch_size: int = 1, *args, **kw):
        dataset_obj = dataset.build_dataset()
        if "dataloaders" in kw:
            kw.pop("dataloaders")
        super().__init__(dataset_obj, batch_size, collate_fn=dict_collation_fn, *args, **kw)


class CrossDatasetHdmapLidarDataset:
    """Lazy native-grid HDMap/LiDAR mixture with homogeneous batches."""

    def __init__(
        self,
        dataset_names,
        shared_overrides=None,
        per_dataset_overrides=None,
    ):
        self.dataset_names = list(dataset_names)
        self.shared_overrides = dict(shared_overrides or {})
        self.per_dataset_overrides = dict(per_dataset_overrides or {})

    def build_dataset(self):
        children = []
        for dataset_id, dataset_name in enumerate(self.dataset_names):
            dataset_config = OmegaConf.create(COMMONDATA_CONFIG[dataset_name])
            overrides = OmegaConf.create(
                {
                    **self.shared_overrides,
                    **self.per_dataset_overrides.get(dataset_name, {}),
                    "emit_sensor_metadata": True,
                    "sensor_dataset_id": dataset_id,
                    "sensor_dataset_name": dataset_name,
                }
            )
            config_dict = OmegaConf.to_container(
                OmegaConf.merge(dataset_config, overrides), resolve=True
            )
            children.append(HdmapLidarSampler(**config_dict))
        return HomogeneousMultiHdmapDataset(children)


class HomogeneousMultiHdmapDataset(Dataset):
    def __init__(self, children):
        self.children = list(children)
        if not self.children:
            raise ValueError("At least one child dataset is required")

    def __len__(self):
        return sum(len(child) for child in self.children)

    def __getitem__(self, index):
        dataset_id, local_index = index
        return self.children[dataset_id][local_index]


class HomogeneousHdmapBatchSampler(Sampler):
    def __init__(
        self,
        dataset,
        batch_size,
        validation=False,
        mix_datasets_across_ranks=False,
        dataset_sampling_weights=None,
        seed=0,
    ):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.validation = bool(validation)
        self.mix_datasets_across_ranks = bool(mix_datasets_across_ranks)
        self.seed = int(seed)
        if "parallel_state" in globals() and parallel_state.is_initialized():
            self.rank = parallel_state.get_data_parallel_rank()
            self.world_size = parallel_state.get_data_parallel_world_size()
        elif torch.distributed.is_initialized():
            self.rank = torch.distributed.get_rank()
            self.world_size = torch.distributed.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1
        self.dataset_sampling_weights = None
        if dataset_sampling_weights is not None:
            weights = torch.as_tensor(
                dataset_sampling_weights, dtype=torch.float64
            )
            if weights.numel() != len(dataset.children):
                raise ValueError(
                    "dataset_sampling_weights must have one value per dataset; "
                    f"got {weights.numel()} values for {len(dataset.children)}"
                )
            if (
                not torch.isfinite(weights).all()
                or (weights < 0).any()
                or weights.sum() <= 0
            ):
                raise ValueError(
                    "dataset_sampling_weights must be finite, non-negative, "
                    "and sum positive"
                )
            if self.mix_datasets_across_ranks and not torch.allclose(
                weights, weights.new_full(weights.shape, weights.mean())
            ):
                raise ValueError(
                    "Non-uniform dataset sampling is incompatible with "
                    "mix_datasets_across_ranks=True"
                )
            self.dataset_sampling_weights = weights / weights.sum()
        self.num_batches = max(
            1,
            sum(len(child) for child in dataset.children)
            // (self.batch_size * self.world_size),
        )

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed)
        cursors = [0] * len(self.dataset.children)
        for step in range(self.num_batches):
            if self.validation:
                dataset_id = step % len(self.dataset.children)
            elif self.mix_datasets_across_ranks:
                dataset_id = (self.rank + step) % len(self.dataset.children)
            elif self.dataset_sampling_weights is None:
                dataset_id = int(
                    torch.randint(
                        len(self.dataset.children), (1,), generator=generator
                    ).item()
                )
            else:
                dataset_id = int(
                    torch.multinomial(
                        self.dataset_sampling_weights,
                        1,
                        generator=generator,
                    ).item()
                )
            child_len = len(self.dataset.children[dataset_id])
            start = cursors[dataset_id] + self.rank * self.batch_size
            batch = [
                (dataset_id, (start + offset) % child_len)
                for offset in range(self.batch_size)
            ]
            cursors[dataset_id] = (
                cursors[dataset_id] + self.world_size * self.batch_size
            ) % child_len
            yield batch


class HomogeneousMultiHdmapDataLoader(DataLoader):
    def __init__(
        self,
        dataset,
        batch_size=1,
        validation=False,
        mix_datasets_across_ranks=False,
        dataset_sampling_weights=None,
        seed=0,
        *args,
        **kwargs,
    ):
        dataset_obj = dataset.build_dataset()
        batch_sampler = HomogeneousHdmapBatchSampler(
            dataset_obj,
            batch_size=batch_size,
            validation=validation,
            mix_datasets_across_ranks=mix_datasets_across_ranks,
            dataset_sampling_weights=dataset_sampling_weights,
            seed=seed,
        )
        if "dataloaders" in kwargs:
            kwargs.pop("dataloaders")
        if "drop_last" in kwargs:
            kwargs.pop("drop_last")
        super().__init__(
            dataset_obj,
            batch_sampler=batch_sampler,
            collate_fn=dict_collation_fn,
            *args,
            **kwargs,
        )


def get_hdmap_lidar_dataset(dataset_name: str, **kwargs):
    """Build the raw dataset (no DataLoader wrapping)."""
    dataset_config = OmegaConf.create(COMMONDATA_CONFIG[dataset_name])
    if kwargs:
        dataset_config = OmegaConf.merge(dataset_config, OmegaConf.create(kwargs))
    config_dict = OmegaConf.to_container(dataset_config, resolve=True)
    return HdmapLidarSampler(**config_dict)
