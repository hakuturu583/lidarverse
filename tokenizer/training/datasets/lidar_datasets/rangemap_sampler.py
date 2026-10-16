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

import gc
import os

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from utils import log

try:
    from megatron.core import parallel_state
    USE_MEGATRON = True
except ImportError:
    USE_MEGATRON = False

from torch.utils.data import Dataset


def _safe_pickle_load(path: str):
    """Load a legacy .pkl file using a restricted unpickler.

    Only allows basic Python types and numpy arrays to be deserialized,
    blocking arbitrary code execution.
    """
    import pickle as _pickle

    _SAFE_MODULES = frozenset({
        "builtins", "collections", "collections.abc",
        "numpy", "numpy.core.multiarray", "numpy.core._multiarray_umath",
        "numpy._core.multiarray", "numpy._core._multiarray_umath",
    })

    class _SafeUnpickler(_pickle.Unpickler):
        def find_class(self, module: str, name: str):
            if module in _SAFE_MODULES:
                return super().find_class(module, name)
            raise _pickle.UnpicklingError(
                f"Blocked unpickling of {module}.{name} — only basic types "
                f"and numpy arrays are allowed."
            )

    with open(path, "rb") as f:
        return _SafeUnpickler(f).load()

from tokenizer.training.datasets.lidar_datasets.datasets import RadymLidar
from tokenizer.training.datasets.lidar_datasets.data_fileds import DataField
from utils.lidar_rangemap import RangeMapDownsampler, normalize_range_map, RANGE_BANDS, SinusoidalEncoder
from utils.per_row_norm import get_per_row_normalizer, apply_per_row_norm


class LidarRangeMapSampler(Dataset):
    def __init__(
        self,
        data_name,
        root_path,
        filter_list_path,
        t5_embedding_path,
        custom_folders,
        custom_fields,
        metadata_folder,
        max_range,
        min_range,
        downsample_factor_row,
        downsample_factor_col,
        repeat_row,
        repeat_col,
        repeat_temporal,
        lidar_crop_size,
        rgb_crop_size,
        rgb_resize_size,
        lidar_size,
        downsample_method,  # for range map
        min_value,
        num_views,  # the number of RGB cameras
        sample_n_frames,
        lidar_length,
        frame_sampling_method,
        frame_sampling_every_n_frames,
        sample_lidar_prob,
        is_video_tokenizer,
        sequential_start_idx=0,
        random_n_frames=False,
        load_lidar_only=True,
        load_rgb_only=False,
        load_lidar_and_rgb=False,  # load both lidar and rgb
        load_lidar_or_rgb=False,
        RGB_FPS=30,
        LIDAR_FPS=10,
        load_t5_embeddings=False,
        inverse_depth=False,
        process_input="repeat",   # "repeat", "concat", "band", "log", or "mono" — normalizing / model input
        process_output="repeat",  # unnormalizing and loss target; "mono" = single channel [N,1,H,W]
        num_bands=3,
        fourier_max_freq=10,
        inv_depth_threshold=20,
        view_indices=None,
        load_intensity=False,
        intensity_only=False,
        intensity_transform="legacy",
        intensity_raw_max=1.0,
        emit_sensor_metadata=False,
        sensor_dataset_id=-1,
        sensor_dataset_name="",
        lidar_resize_size=(-1, -1),
        per_row_normalize=False,
        per_row_stats_path="",
        per_row_smooth_sigma=5.0,
        emit_valid_mask=False,
        dataset_backend="radym",  # "radym", "*_ncore", or cached "pandaset"
        resample_width=-1,  # waymo_ncore only: rebin azimuth to this width (e.g. 2048); <=0 native
        center_vehicle_forward=False,
        # --- waymo_ncore hdmap (on-the-fly rasterized from dump_waymo_hdmap NPZs) ---
        hdmap_root=None,
        hdmap_multichannel=False,
        hdmap_moving_only=False,
        hdmap_include_boxes=True,
        hdmap_box_edge_spacing=0.15,
        hdmap_map_max_dist=None,
        hdmap_sensor_template="native",
        hdmap_template_root_path=None,
        hdmap_template_filter_list_path=None,
        hdmap_virtual_sensor_height_m=None,
        hdmap_virtual_sensor_yaw_deg=0.0,
        canonical_lidar_mode="native",
        hdmap_pose_for_lidar_to_world=False,
    ):
        super().__init__()
        self.data_name = data_name
        self.load_intensity = load_intensity
        self.intensity_only = bool(intensity_only)
        if self.intensity_only and not self.load_intensity:
            raise ValueError("intensity_only=True requires load_intensity=True")
        self.intensity_transform = intensity_transform
        if self.intensity_transform not in ("legacy", "log1p"):
            raise ValueError(f"Unknown intensity_transform: {self.intensity_transform!r}")
        self.intensity_raw_max = float(intensity_raw_max)
        if self.intensity_raw_max <= 0:
            raise ValueError(
                f"intensity_raw_max must be positive, got {self.intensity_raw_max}"
            )
        self.emit_sensor_metadata = bool(emit_sensor_metadata)
        self.sensor_dataset_id = int(sensor_dataset_id)
        self.sensor_dataset_name = str(sensor_dataset_name or data_name)
        self._sensor_root_path = root_path
        self._sensor_filter_list_path = filter_list_path
        self._sensor_ray_directions = None
        self.dataset_backend = dataset_backend
        self.resample_width = resample_width
        self.center_vehicle_forward = bool(center_vehicle_forward)
        # waymo_ncore hdmap rasterization knobs (consumed in _build_dataset).
        self._hdmap_kwargs = dict(
            hdmap_root=hdmap_root,
            min_range=min_range,
            max_range=max_range,
            hdmap_multichannel=hdmap_multichannel,
            hdmap_moving_only=hdmap_moving_only,
            hdmap_include_boxes=hdmap_include_boxes,
            hdmap_box_edge_spacing=hdmap_box_edge_spacing,
            hdmap_map_max_dist=hdmap_map_max_dist,
        )
        self._nuscenes_canonical_kwargs = dict(
            hdmap_sensor_template=hdmap_sensor_template,
            hdmap_template_root_path=hdmap_template_root_path,
            hdmap_template_filter_list_path=hdmap_template_filter_list_path,
            hdmap_virtual_sensor_height_m=hdmap_virtual_sensor_height_m,
            hdmap_virtual_sensor_yaw_deg=hdmap_virtual_sensor_yaw_deg,
            canonical_lidar_mode=canonical_lidar_mode,
            center_vehicle_forward=self.center_vehicle_forward,
            hdmap_pose_for_lidar_to_world=hdmap_pose_for_lidar_to_world,
        )
        # build the dataset
        self.dataset = self._build_dataset(
            root_path=root_path,
            filter_list_path=filter_list_path,
            num_views=num_views,
            metadata_folder=metadata_folder,
            custom_folders=custom_folders,
            custom_fields=custom_fields,
            view_indices=view_indices,
            load_intensity=load_intensity,
        )
        self.plus = 1 if "hdmap" not in self.dataset.custom_fields else 2
        self.num_views = num_views

        self.LIDAR_FPS = LIDAR_FPS
        self.RGB_FPS = RGB_FPS

        self.sample_n_frames = sample_n_frames
        self.random_n_frames = random_n_frames
        self.lidar_length = lidar_length
        self.rgb_length = lidar_length * (RGB_FPS // LIDAR_FPS)
        self.frame_sampling_method = frame_sampling_method
        assert self.frame_sampling_method in ["random", "sequential", "sequential_from_zero"]
        self.frame_sampling_every_n_frames = frame_sampling_every_n_frames
        # Chunk offset for sequential_from_zero: render frames [start, start+n).
        # Default 0 preserves existing behavior bit-for-bit.
        self.sequential_start_idx = int(sequential_start_idx)

        self.inverse_depth = inverse_depth
        self.min_distance = min_range
        self.max_distance = max_range
        self.max_range = max_range if not inverse_depth else 1 / min_range
        self.min_range = min_range if not inverse_depth else 1 / max_range

        self.min_value = min_value  # min value after normalisation, if 0 then [0, 1],if -1, then [-1, 1]

        self.repeat_row = repeat_row
        self.repeat_col = repeat_col
        self.repeat_temporal = repeat_temporal
        self.process_input = process_input
        self.process_output = process_output
        assert self.process_input in ["repeat", "concat", "band", "log", "mono", "fourier"]
        assert self.process_output in ["repeat", "concat", "band", "log", "mono"]
        self.num_bands = num_bands
        assert self.num_bands in RANGE_BANDS.keys()

        # Per-row depth normalization (post min-max; mono only). Built once here
        # from training-set stats; None when disabled (no-op downstream).
        self.per_row_normalizer = get_per_row_normalizer(
            per_row_normalize,
            per_row_stats_path,
            min_range=self.min_distance,
            max_range=self.max_distance,
            min_value=self.min_value,
            smooth_sigma=per_row_smooth_sigma,
            log_depth=False,
        )
        if self.per_row_normalizer is not None:
            assert not self.inverse_depth, "per_row_normalize requires inverse_depth=False"
            assert self.process_input == "mono" and self.process_output == "mono", (
                "per_row_normalize is only supported with process_input/output='mono'"
            )

        # Emit an explicit per-pixel valid (return) mask computed from the
        # min-max depth BEFORE per-row normalization. Required when per-row norm
        # is on AND the network/loss use a validity head: per-row z-scoring
        # shifts the no-return sentinel by a row-dependent amount, so the
        # downstream ``depth > min_value + eps`` derivation no longer separates
        # returns from raydrop. The mask (1 = return) is carried as
        # ``sample["loss_mask"]`` -> MASK_KEY (losses) and threaded into the
        # network's ``mask_as_input`` concat. mono only.
        self.emit_valid_mask = emit_valid_mask
        if self.emit_valid_mask:
            assert self.process_input == "mono", (
                "emit_valid_mask is only supported with process_input='mono'"
            )

        self.lidar_size = lidar_size
        self.lidar_crop_size = lidar_crop_size
        self.rgb_crop_size = rgb_crop_size
        self.rgb_resize_size = rgb_resize_size
        # Optional resize of the final range map to a fixed [H, W] (use -1 to
        # keep a dim). Done with nearest-exact interpolation (NOT bilinear):
        # range/depth pixels must not be averaged across depth discontinuities
        # (that invents phantom points / blurs thin structure). Mirrors
        # r2flow's F.interpolate(..., mode="nearest-exact") resize.
        self.lidar_resize_size = lidar_resize_size

        self.sample_lidar_prob = sample_lidar_prob if not load_rgb_only else -1
        self.load_lidar_only = load_lidar_only
        self.load_rgb_only = load_rgb_only
        self.load_lidar_and_rgb = load_lidar_and_rgb
        self.load_lidar_or_rgb = load_lidar_or_rgb

        assert sum([self.load_lidar_only, self.load_rgb_only, self.load_lidar_and_rgb, self.load_lidar_or_rgb]) == 1

        # t5 text embedding
        self.load_t5_embeddings = load_t5_embeddings
        self.t5_embedding_path = t5_embedding_path

        self.length = self.dataset.size_of_filter_set()
        log.critical(f"load_lidar_only: {self.load_lidar_only}")
        log.critical(f"dataset size: {self.length}")

        self.is_video_tokenizer = is_video_tokenizer  # if true, then the frame sampling method is sequential, also we need to upsample the lidar to the rgb fps
        if self.is_video_tokenizer:
            assert self.frame_sampling_method == "sequential" or self.frame_sampling_method == "sequential_from_zero"

        self.range_map_downsampler = RangeMapDownsampler(row_factor = downsample_factor_row, col_factor = downsample_factor_col, method = downsample_method)

        # image transform
        self.img_transform = transforms.Compose(
            [
                transforms.Resize(
                    self.rgb_resize_size, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True
                ),
                transforms.CenterCrop(self.rgb_crop_size),
            ]
        )
        self.norm_image = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)

        # Prepare infinite dataset logic
        self.n_data = self.length
        if parallel_state.is_initialized():
            dp_group_id = parallel_state.get_data_parallel_rank()
            dp_world_size = parallel_state.get_data_parallel_world_size()
            log.critical(
                f"Using parallelism size CP :{parallel_state.get_context_parallel_world_size()}, "
                + f"TP :{parallel_state.get_tensor_model_parallel_world_size()} for video dataset, "
                + f"DP: {dp_group_id}, DP World size: {dp_world_size}"
            )
        else:
            dp_world_size = 1
            dp_group_id = 0

        # Data partitioning. Clamp to >=1 so that a split smaller than the
        # data-parallel world size (e.g. the 20-sequence Waymo val set on 32
        # GPUs) doesn't divide to zero -- ranks beyond the split then replicate
        # samples (start index wrapped mod n_data). For every split with
        # n_data >= dp_world_size this is a no-op (bit-identical to before).
        self.n_data_per_node = max(1, self.n_data // dp_world_size)
        self.data_start_idx = (dp_group_id * self.n_data_per_node) % self.n_data
        self.dp_group_id = dp_group_id

        # Infinite loop multiplier
        maximum_iter = 2000000  # hack to create infinite loop
        self.multiplier = max(1, maximum_iter // self.n_data_per_node)
        # self.multiplier = 1

        # Fourier encoder
        self.fourier_max_freq = fourier_max_freq
        self.fourier_encoder = SinusoidalEncoder(
            n_input_dims=1,
            min_deg=0,
            max_deg=self.fourier_max_freq,
            enable_identity=True,
        )

    def _build_dataset(self, **kwargs):
        """Construct the selected low-level reader."""
        if self.dataset_backend == "waymo_ncore":
            from tokenizer.training.datasets.lidar_datasets.waymo_ncore import (
                WaymoNCoreLidar,
            )

            return WaymoNCoreLidar(
                resample_width=self.resample_width,
                **self._hdmap_kwargs,
                **kwargs,
            )
        if self.dataset_backend == "nuscenes_ncore":
            from tokenizer.training.datasets.lidar_datasets.nuscenes_ncore import (
                NuScenesNCoreLidar,
            )

            return NuScenesNCoreLidar(
                resample_width=self.resample_width,
                **self._hdmap_kwargs,
                **self._nuscenes_canonical_kwargs,
                **kwargs,
            )
        if self.dataset_backend == "pandaset":
            from tokenizer.training.datasets.lidar_datasets.pandaset import (
                PandaSetCachedLidar,
            )

            return PandaSetCachedLidar(**kwargs)
        return RadymLidar(**kwargs)

    def __len__(self):
        """
        Return the artificially multiplied length
        """
        return self.multiplier * self.n_data_per_node

    def sample_random_frames(self, n_frames, total_frames):
        # sample self.sample_n_frames frames from the video
        # return the frame indices
        frame_indices = np.random.randint(0, total_frames, size=n_frames)
        return frame_indices

    def sample_sequential_frames(self, n_frames, total_frames):
        # sample every self.frame_sampling_every_n_frames frames
        if self.frame_sampling_method == "sequential_from_zero":
            start_idx = self.sequential_start_idx
        else:
            start_idx = np.random.randint(0, total_frames - n_frames * self.frame_sampling_every_n_frames)
        frame_indices = np.arange(start_idx, start_idx + n_frames * self.frame_sampling_every_n_frames, self.frame_sampling_every_n_frames)
        return frame_indices
    

    def _draw_n_frames(self):
        """Return sample_n_frames, or a random T in [1, sample_n_frames] if random_n_frames is set."""
        if not self.random_n_frames:
            return self.sample_n_frames
        return np.random.randint(2, self.sample_n_frames+1)

    def sample_frame_indices(self, load_lidar):
        n_frames = self._draw_n_frames()
        if load_lidar:
            if self.frame_sampling_method == "random":
                frame_indices = self.sample_random_frames(n_frames, self.lidar_length)
            else:
                frame_indices = self.sample_sequential_frames(
                    n_frames // self.repeat_temporal, self.lidar_length
                )

        else:  # load rgb
            if self.frame_sampling_method == "random":
                frame_indices = self.sample_random_frames(n_frames, self.rgb_length)
            else:
                frame_indices = self.sample_sequential_frames(
                    n_frames + 1, self.rgb_length
                )  # + 1 because the video tokenizer will need one more additional frame
        return frame_indices

    def normalise_and_process_range_map_with_mode(self, range_maps, mode):
        # Input: [N, H, W]. Output: [N, C, H, W] with C=3 for repeat/concat/band/log, C=1 for mono.
        # mode: "repeat", "concat", "band", "log", or "mono"
        if mode == "repeat":
            range_maps = normalize_range_map(
                range_maps, self.max_distance, self.min_distance, self.min_value, self.inverse_depth
            )  # shape: [N, H, W]
            range_maps = range_maps[:, None, :, :].repeat(3, axis=1)  # shape: [N, 3, 128, 1800]
        elif mode == "mono":
            range_maps = normalize_range_map(
                range_maps, self.max_distance, self.min_distance, self.min_value, self.inverse_depth
            )  # shape: [N, H, W]
            # Per-row z-score on top of min-max (no-op when normalizer is None).
            range_maps = apply_per_row_norm(range_maps, self.per_row_normalizer)
            range_maps = range_maps[:, None, :, :]  # shape: [N, 1, H, W]
        elif mode == "concat":
            inverse_depth = normalize_range_map(
                range_maps, self.max_distance, self.min_distance, self.min_value, True
            )  # shape: [N, H, W]
            depth = normalize_range_map(
                range_maps, self.max_distance, self.min_distance, self.min_value, False
            )  # shape: [N, H, W]
            range_maps = np.concatenate(
                [inverse_depth[:, None, :, :], depth[:, None, :, :], depth[:, None, :, :]], axis=1
            )  # shape: [N, 3, 128, 1800]
        elif mode == "band":
            range_maps_list = []
            for i in range(self.num_bands):
                band_range_map = normalize_range_map(
                    range_maps, 
                    RANGE_BANDS[self.num_bands][i+1], 
                    RANGE_BANDS[self.num_bands][i], 
                    self.min_value, 
                    self.inverse_depth
                )[:, None, :, :] # shape: [N, 1, H, W]
                range_maps_list.append(band_range_map)
            range_maps = np.concatenate(range_maps_list, axis=1)  # shape: [N, 3, 128, 1800]
        elif mode == "log":
            max_scale = np.log2(self.max_distance + 1)
            range_maps = np.clip(range_maps, self.min_distance, self.max_distance) # 5-100
            range_maps = np.log2(range_maps + 1) / max_scale # 0-1
            if self.min_value == -1:
                range_maps = range_maps * 2 - 1 # -1-1
            range_maps = range_maps[:, None, :, :].repeat(3, axis=1)  # shape: [N, 3, 128, 1800]
        elif mode == "fourier":
            range_maps = normalize_range_map(
                range_maps, self.max_distance, self.min_distance, self.min_value, self.inverse_depth
            )  # shape: [N, H, W]
            x = torch.from_numpy(range_maps).float().unsqueeze(-1)  # [N, H, W, 1]
            encoded = self.fourier_encoder(x)  # [N, H, W, n_output_dims]
            range_maps = encoded.permute(0, 3, 1, 2).numpy()  # [N, n_output_dims, H, W]
        return range_maps

    def normalise_and_convert_range_map_to_three_channels(self, range_maps):
        """Uses process_input."""
        return self.normalise_and_process_range_map_with_mode(
            range_maps, self.process_input
        )

    def _apply_repeat_crop_temporal(self, range_maps_np, start_col_idx=None):
        # range_maps_np: [N, C, H, W] (C=3 or 1 for mono). When two tensors use same crop, pass shared start_col_idx.
        range_maps_np = range_maps_np.repeat(self.repeat_row, axis=2)
        range_maps_np = range_maps_np.repeat(self.repeat_col, axis=3)
        if self.lidar_crop_size[1] != -1:
            if self.lidar_crop_size[1] > 1600:
                start_col_idx = (range_maps_np.shape[-1] - self.lidar_crop_size[1]) // 2
                range_maps_np = range_maps_np[:, :, :, start_col_idx : start_col_idx + self.lidar_crop_size[1]]
            else:
                if start_col_idx is None:
                    start_col_idx = np.random.randint(0, range_maps_np.shape[-1] - self.lidar_crop_size[1])
                range_maps_np = range_maps_np[:, :, :, start_col_idx : start_col_idx + self.lidar_crop_size[1]]
        if self.frame_sampling_method in ("sequential", "sequential_from_zero") and self.is_video_tokenizer:
            first_range_map = range_maps_np[0]
            range_maps_np = range_maps_np.repeat(self.repeat_temporal, axis=0)
            range_maps_np = np.concatenate([first_range_map[None], range_maps_np], axis=0)
        range_maps_t = torch.from_numpy(range_maps_np).float()
        # Optional fixed-size resize (nearest-exact, see __init__). Applied to
        # the [N, C, H, W] tensor so depth values are resampled, never blended.
        if self.lidar_resize_size is not None and (
            self.lidar_resize_size[0] != -1 or self.lidar_resize_size[1] != -1
        ):
            _, _, H, W = range_maps_t.shape
            target_h = self.lidar_resize_size[0] if self.lidar_resize_size[0] != -1 else H
            target_w = self.lidar_resize_size[1] if self.lidar_resize_size[1] != -1 else W
            if (target_h, target_w) != (H, W):
                range_maps_t = F.interpolate(
                    range_maps_t, size=(target_h, target_w), mode="nearest-exact"
                )
        return range_maps_t

    def _normalize_intensity(self, intensity_maps):
        """Apply the selected intensity transform and add a channel dimension."""
        intensity_maps = intensity_maps.copy()
        if self.intensity_transform == "log1p":
            intensity_maps = np.clip(
                intensity_maps / self.intensity_raw_max, 0.0, 1.0
            )
            intensity_maps = np.log1p(intensity_maps) / np.log(2.0)
            intensity_maps = intensity_maps * 2.0 - 1.0
            return intensity_maps[:, None, :, :]
        if self.min_value == -1:
            intensity_maps = intensity_maps * 2 - 1
        return intensity_maps[:, None, :, :]  # [N, 1, H, W]

    def get_sensor_metadata(self, height, width):
        """Return true sensor geometry for opt-in cross-dataset training."""
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
                    center_vehicle_forward=self.center_vehicle_forward,
                )
            elif self.dataset_backend == "pandaset":
                rays = self.dataset.sensor_ray_directions()
            else:
                from utils.lidar_rangemap import (
                    load_pandar128_elevations,
                    range_map_to_ray_directions,
                )

                elevations = torch.tensor(load_pandar128_elevations(), dtype=torch.float32)
                rays = range_map_to_ray_directions(
                    width // self.repeat_col, elevations
                )
            rays = torch.as_tensor(rays, dtype=torch.float32)
            rays = rays.repeat_interleave(self.repeat_row, dim=0)
            rays = rays.repeat_interleave(self.repeat_col, dim=1)
            if tuple(rays.shape[:2]) != (height, width):
                raise ValueError(
                    "True-ray metadata must match the repeated sensor grid; "
                    f"got rays {tuple(rays.shape[:2])} and sample {(height, width)}"
                )
            self._sensor_ray_directions = rays
        return {
            "sensor_ray_directions": self._sensor_ray_directions,
            "sensor_depth_bounds": torch.tensor(
                [self.min_distance, self.max_distance], dtype=torch.float32
            ),
            "sensor_dataset_id": self.sensor_dataset_id,
            "sensor_dataset_name": self.sensor_dataset_name,
        }

    def _compute_valid_mask_np(self, range_maps):
        # range_maps: [N, H, W] metric depth (post-downsample, pre per-row).
        # Returns [N, 1, H, W] float {0,1}: 1 = real return, 0 = no-return.
        # Use raw metric support rather than a normalized epsilon. A threshold
        # at ``min_value + 0.02`` incorrectly removes the first 1% of every
        # sensor's configured range (e.g. valid Pandar returns at 5--5.95 m).
        valid = (
            np.isfinite(range_maps)
            & (range_maps >= self.min_distance)
            & (range_maps <= self.max_distance)
        ).astype(np.float32)
        return valid[:, None, :, :]  # [N, 1, H, W]

    def process_lidar(self, range_maps, intensity_maps=None):
        # Returns (input_tensor, supervision_tensor, valid_tensor). Same crop/temporal for all.
        # valid_tensor is None unless ``emit_valid_mask`` (then [N, 1, H, W] {0,1}).
        # When intensity_maps is provided, it is appended as extra channel(s) to both tensors.
        if intensity_maps is not None:
            range_maps, [intensity_maps] = self.range_map_downsampler.downsample(range_maps, [intensity_maps])
        else:
            range_maps = self.range_map_downsampler.downsample(range_maps)  # shape: [N, H, W]

        # Replace every unsupported metric depth with the raw no-return
        # sentinel before normalization. Otherwise np.clip maps +inf and
        # finite >max returns to +1, while NaNs survive normalization. Keep the
        # optional explicit mask from this exact same support predicate.
        metric_valid_np = self._compute_valid_mask_np(range_maps)
        valid_np = metric_valid_np if self.emit_valid_mask else None
        range_maps = np.where(
            metric_valid_np[:, 0].astype(bool), range_maps, 0.0
        ).astype(np.float32, copy=False)
        if intensity_maps is not None:
            intensity_maps = np.where(
                metric_valid_np[:, 0].astype(bool), intensity_maps, 0.0
            ).astype(np.float32, copy=False)
        intensity_np = self._normalize_intensity(intensity_maps) if intensity_maps is not None else None

        if self.intensity_only:
            if intensity_np is None:
                raise ValueError(
                    "intensity_only=True but this sample has no intensity map"
                )
            # The depth image remains available only to define return validity.
            # Model input and reconstruction target are strictly one-channel
            # intensity; no-return pixels are excluded by ``valid_tensor``.
            start_col_idx = None
            if self.lidar_crop_size[1] != -1 and self.lidar_crop_size[1] <= 1600:
                w = intensity_np.shape[-1] * self.repeat_col
                start_col_idx = np.random.randint(
                    0, w - self.lidar_crop_size[1]
                )
            intensity_tensor = self._apply_repeat_crop_temporal(
                intensity_np, start_col_idx
            )
            valid_tensor = (
                self._apply_repeat_crop_temporal(valid_np, start_col_idx)
                if valid_np is not None
                else None
            )
            return intensity_tensor, intensity_tensor, valid_tensor

        if self.process_output != self.process_input:
            range_maps_input = self.normalise_and_process_range_map_with_mode(
                range_maps, self.process_input
            )
            range_maps_supervision = self.normalise_and_process_range_map_with_mode(
                range_maps, self.process_output
            )
            if intensity_np is not None:
                range_maps_input = np.concatenate([range_maps_input, intensity_np], axis=1)
                range_maps_supervision = np.concatenate([range_maps_supervision, intensity_np], axis=1)
            # Share random crop when applicable
            start_col_idx = None
            if self.lidar_crop_size[1] != -1 and self.lidar_crop_size[1] <= 1600:
                w = range_maps_input.shape[-1] * self.repeat_col
                start_col_idx = np.random.randint(0, w - self.lidar_crop_size[1])
            input_tensor = self._apply_repeat_crop_temporal(range_maps_input, start_col_idx)
            supervision_tensor = self._apply_repeat_crop_temporal(range_maps_supervision, start_col_idx)
            valid_tensor = (
                self._apply_repeat_crop_temporal(valid_np, start_col_idx)
                if valid_np is not None else None
            )
            return input_tensor, supervision_tensor, valid_tensor
        else:
            range_maps = self.normalise_and_convert_range_map_to_three_channels(range_maps)
            if intensity_np is not None:
                range_maps = np.concatenate([range_maps, intensity_np], axis=1)
            # Generate the (possibly random) crop offset once so depth and the
            # validity mask are cropped identically.
            start_col_idx = None
            if (
                valid_np is not None
                and self.lidar_crop_size[1] != -1
                and self.lidar_crop_size[1] <= 1600
            ):
                w = range_maps.shape[-1] * self.repeat_col
                start_col_idx = np.random.randint(0, w - self.lidar_crop_size[1])
            input_tensor = self._apply_repeat_crop_temporal(range_maps, start_col_idx)
            valid_tensor = (
                self._apply_repeat_crop_temporal(valid_np, start_col_idx)
                if valid_np is not None else None
            )
            return input_tensor, input_tensor, valid_tensor

    def process_rgb(self, rgb_images):
        rgb_images = self.img_transform(rgb_images)
        rgb_images = self.norm_image(rgb_images)  # N C H W
        return rgb_images

    def prepare_dummy_data(self):
        data = {}
        dummy_text_embedding = torch.zeros(512, 1024)
        dummy_text_mask = torch.zeros(512)
        dummy_text_mask[0] = 1
        data["t5_text_embeddings"] = dummy_text_embedding
        data["t5_text_mask"] = dummy_text_mask

        return data

    def prepare_t5_embeddings(self, example, frame_indices):
        # determine which chunk to use based on the frame indices
        mid_indice = frame_indices[len(frame_indices) // 2]
        chunk_id = mid_indice // int(self.lidar_length / 5)

        if self.load_t5_embeddings:
            # t5_embed_path = os.path.join(self.t5_embedding_path, example["__key__"] + f"_{chunk_id}.pkl")
            t5_embed_path = os.path.join(self.t5_embedding_path, example["__key__"] + ".pkl")
            if os.path.exists(t5_embed_path):
                t5_embed_pickle = _safe_pickle_load(t5_embed_path)
                t5_embed = {}
                text_embedding = t5_embed_pickle["pickle"]["ground_truth"]["embeddings"]["t5_xxl"]
                dummy_text_embedding = torch.zeros(512, 1024)
                dummy_text_mask = torch.zeros(512)
                n_text = text_embedding.shape[0]  # 512
                dummy_text_embedding[:n_text] = torch.from_numpy(text_embedding)
                dummy_text_mask[:n_text] = 1
                t5_embed["t5_text_embeddings"] = dummy_text_embedding
                t5_embed["t5_text_mask"] = dummy_text_mask
                del t5_embed_pickle
            else:
                log.info(f"t5 embedding not found for {example['__key__']}", rank0_only=False)
                t5_embed = self.prepare_dummy_data()
        else:
            t5_embed = self.prepare_dummy_data()
            
        if self.num_views > 1:
            t5_embed["t5_text_embeddings"] = torch.cat([t5_embed["t5_text_embeddings"]] * (self.num_views+self.plus), dim=0)
            t5_embed["t5_text_mask"] = torch.cat([t5_embed["t5_text_mask"]] * (self.num_views+self.plus), dim=0)

        return t5_embed

    def get_lidar(self, example, lidar_key="lidar"):
        """Returns (input_tensor, recon_target, valid_mask). First two [C, N, H, W].
        valid_mask is [1, N, H, W] (or None when emit_valid_mask is off)."""
        if lidar_key in example["custom"]:
            range_maps = example["custom"][lidar_key]  # shape: [N, 128, 3600]
            intensity_maps = example["custom"].get("lidar_intensity") if self.load_intensity else None
            input_tensor, recon_target, valid_mask = self.process_lidar(range_maps, intensity_maps)
            input_tensor = input_tensor.permute(1, 0, 2, 3).contiguous()  # shape: [C, N, H, W]
            recon_target = recon_target.permute(1, 0, 2, 3).contiguous()
            if valid_mask is not None:
                valid_mask = valid_mask.permute(1, 0, 2, 3).contiguous()  # [1, N, H, W]
            return input_tensor, recon_target, valid_mask
        return None, None, None

    def process_hdmap_multichannel(self, hdmap):
        """Per-class semantic hdmap [N, K, H, W] (metric range, 0=empty) -> torch
        [K, N', H, W]. Each class channel is normalized like a mono depth map
        (empty -> min_value sentinel) and shares the lidar's downsample / repeat /
        temporal-prepend / resize so channels stay pixel-aligned with the lidar."""
        N, K, H, W = hdmap.shape
        chans = []
        for k in range(K):
            rm = self.range_map_downsampler.downsample(hdmap[:, k])  # [N, H, W]
            rm = normalize_range_map(
                np.asarray(rm, dtype=np.float32).copy(),
                self.max_distance,
                self.min_distance,
                self.min_value,
                self.inverse_depth,
            )  # [N, H, W] in [min_value, 1]
            chans.append(rm[:, None, :, :])  # [N, 1, H, W]
        arr = np.concatenate(chans, axis=1)  # [N, K, H, W]
        t = self._apply_repeat_crop_temporal(arr)  # [N', K, H, W]
        return t.permute(1, 0, 2, 3).contiguous()  # [K, N', H, W]

    def get_lidar_hdmap(self, example, hdmap_key="hdmap"):
        if hdmap_key not in example["custom"]:
            return None
        hd = example["custom"][hdmap_key]
        # Multi-channel semantic hdmap (one range-image channel per class) ->
        # [N, K, H, W]; normalize per channel. Single-channel depth hdmap ->
        # [N, H, W]; reuse the lidar pipeline.
        if np.asarray(hd).ndim == 4:
            return self.process_hdmap_multichannel(hd)
        hdmap_input, _, _ = self.process_lidar(hd)
        return hdmap_input.permute(1, 0, 2, 3).contiguous()  # shape: [C, N, H, W]

    def get_rgb(self, example):
        key = DataField.IMAGE_RGB
        if key in example.keys():
            rgb_images_dict = dict()
            view_keys = example[key].keys()
            for view_key in view_keys:
                rgb_images = example[key][view_key]
                rgb_images = self.process_rgb(rgb_images)
                rgb_images = rgb_images.permute(1, 0, 2, 3).contiguous()  # shape: [C, N, H, W]
                rgb_images_dict[key.value + "_" + str(view_key)] = rgb_images
        else:
            rgb_images_dict = None
        return rgb_images_dict

    def get_rgb_hdmap(self, example):
        key = DataField.IMAGE_HDMAP
        if key in example.keys():
            rgb_hdmap_dict = dict()
            view_keys = example[key].keys()
            for view_key in view_keys:
                rgb_hdmap = example[key][view_key]
                rgb_hdmap = self.process_rgb(rgb_hdmap)
                rgb_hdmap = rgb_hdmap.permute(1, 0, 2, 3).contiguous()  # shape: [C, N, H, W]
                rgb_hdmap_dict[key.value + "_" + str(view_key)] = rgb_hdmap
        else:
            rgb_hdmap_dict = None
        return rgb_hdmap_dict
    
    def get_camera_c2lidar_transform(self, example):
        key = DataField.CAMERA_C2LIDAR_TRANSFORM
        if key in example.keys():
            camera_c2lidar_transform_dict = dict()
            view_keys = example[key].keys()
            for view_key in view_keys:
                camera_c2lidar_transform = example[key][view_key]
                camera_c2lidar_transform_dict[key.value + "_" + str(view_key)] = camera_c2lidar_transform
        else:
            camera_c2lidar_transform_dict = None
        return camera_c2lidar_transform_dict
    
    def get_camera_intrinsics(self, example):
        key = DataField.CAMERA_INTRINSICS
        if key in example.keys():
            view_keys = example[key].keys()
            camera_intrinsics_dict = dict()
            for view_key in view_keys:
                camera_intrinsics = example[key][view_key]
                camera_intrinsics_dict[key.value + "_" + str(view_key)] = camera_intrinsics
        else:
            camera_intrinsics_dict = None
        return camera_intrinsics_dict
    
    

    def __getitem__(self, idx):
        if self.load_rgb_only:
            raise NotImplementedError
        if self.load_lidar_or_rgb:
            raise NotImplementedError

        # sample the video index
        lidar_data_idx = (idx % self.n_data_per_node) + self.data_start_idx  # frame idx
        lidar_frame_indices = self.sample_frame_indices(load_lidar=True)

        # load the data
        try:
            if self.load_lidar_and_rgb:  # we also load the camera intrinsics and camera-to-lidar transformation matrix as well as the timestamp for motion compensation
                # example = self.dataset._read_rgb_and_lidar(lidar_data_idx, lidar_frame_indices, ["custom", DataField.CAMERA_C2LIDAR_TRANSFORM, DataField.CAMERA_INTRINSICS, DataField.IMAGE_HDMAP, DataField.IMAGE_RGB])
                example = self.dataset._read_rgb_and_lidar(lidar_data_idx, lidar_frame_indices, ["custom", DataField.CAMERA_C2LIDAR_TRANSFORM, DataField.CAMERA_INTRINSICS, DataField.IMAGE_RGB])
            else:
                example = self.dataset._read_rgb_and_lidar(lidar_data_idx, lidar_frame_indices, ["custom"])
        except Exception as e:
            log.info(f"Error loading data for index {idx}: {e}", rank0_only=False)
            return self.__getitem__(np.random.randint(0, self.__len__()))

        lidar_rangemap, recon_target, lidar_valid_mask = self.get_lidar(example)
        lidar_hdmap = self.get_lidar_hdmap(example)
        rgb_images = self.get_rgb(example)
        rgb_hdmap = self.get_rgb_hdmap(example)
        t5_embed = self.prepare_t5_embeddings(example, lidar_frame_indices)
        
        
        cam_intrinsics = self.get_camera_intrinsics(example)
        cam_c2lidar_transform = self.get_camera_c2lidar_transform(example)

        key = "video"

        sample = {}
        sample[key] = lidar_rangemap
        sample["recon_target"] = recon_target
        if lidar_valid_mask is not None:
            # Explicit per-pixel return mask (pre per-row). Routed to MASK_KEY in
            # model.{training,validation}_step and into the mask_as_input concat.
            sample["loss_mask"] = lidar_valid_mask
            sample["loss_mask_is_validity"] = True
        sample["__key__"] = example["__key__"]
        sample["clip_name"] = "%s-%s-%d" % (self.data_name, example["__key__"], idx)
        sample["data_type"] = "lidar"
        sample["num_frames"] = self.sample_n_frames + 1 if self.is_video_tokenizer else self.sample_n_frames
        sample["is_preprocessed"] = True
        sample["image_size"] = torch.tensor(self.lidar_size)
        sample["padding_mask"] = torch.zeros(1, self.lidar_size[0], self.lidar_size[1])
        sample["fps"] = 24  # unsure why this value
        sample['num_views'] = self.num_views
        sample["view_indices"] = torch.arange(self.num_views + self.plus) # +1 for the lidar view
        sample["timestamp_list"] = example["timestamp_list"]
        sample["pose_sensor_start_end"] = example["pose_sensor_start_end"]
        sample["rgb_frame_indices"]=example["rgb_frame_indices"]
        # Ego poses (lidar->world, sweep-start) aligned to the *video* temporal
        # layout: mirror the repeat_temporal + frame-0 prepend applied to the
        # range maps in `_apply_repeat_crop_temporal` so pose[t] indexes the
        # same frame as video[:, t]. Shape: (T, 4, 4).
        lidar_to_world = np.asarray(example["lidar_to_world"], dtype=np.float32)  # (N, 4, 4)
        if self.is_video_tokenizer and self.frame_sampling_method in (
            "sequential",
            "sequential_from_zero",
        ):
            lidar_to_world = np.concatenate(
                [lidar_to_world[:1], np.repeat(lidar_to_world, self.repeat_temporal, axis=0)],
                axis=0,
            )
        sample["lidar_to_world"] = torch.from_numpy(lidar_to_world).float()
        sample.update(t5_embed)

        if lidar_hdmap is not None:
            sample["lidar_hdmap"] = lidar_hdmap
        if rgb_images is not None:
            sample.update(rgb_images)
        if rgb_hdmap is not None:
            sample.update(rgb_hdmap)
        if self.load_lidar_and_rgb:
            
            sample.update(cam_intrinsics)
            sample.update(cam_c2lidar_transform)
            
        
        del example
        gc.collect()
        # from imaginaire.utils import distributed, log
        # rank = distributed.get_rank()
        # print(f"RANK {rank}, {data_idx}: Working on clip {sample['clip_name']}.")
        return sample
