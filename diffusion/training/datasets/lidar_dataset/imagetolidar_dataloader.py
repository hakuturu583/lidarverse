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

import numpy as np
import torch

import os
import pickle

import einops
import omegaconf
import torchvision.transforms as transforms

from utils.lazy_config import LazyCall as L
from utils import log

try:
    from megatron.core import parallel_state
    USE_MEGATRON = True
except ImportError:
    USE_MEGATRON = False

from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset

from tokenizer.training.datasets.lidar_datasets.datasets import RadymLidar
from tokenizer.training.datasets.lidar_datasets.data_fileds import DataField
from utils.lidar_rangemap import RangeMapDownsampler, normalize_range_map

from diffusion.training.datasets.lidar_dataset.configs import COMMONDATA_CONFIG

def dict_collation_fn(samples, combine_tensors=True, combine_scalars=True, ignore_keys=None):
    """Take a list  of samples (as dictionary) and create a batch, preserving the keys.
    If `tensors` is True, `ndarray` objects are combined into
    tensor batches.
    :param dict samples: list of samples
    :param bool tensors: whether to turn lists of ndarrays into a single ndarray
    :returns: single sample consisting of a batch
    :rtype: dict
    """

    batched = {key: [] for key in samples[0]}
    # assert isinstance(samples[0][first_key], (list, tuple)), type(samples[first_key])

    for s in samples:
        [batched[key].append(s[key]) for key in batched if not (ignore_keys is not None and key in ignore_keys)]

    result = {}
    for key in batched:
        if ignore_keys and key in ignore_keys:
            continue
        try:
            if isinstance(batched[key][0], bool):
                assert key == "is_preprocessed"
                result[key] = batched[key][0]  # this is a hack to align with cosmos data
            elif isinstance(batched[key][0], (int, float)):
                if combine_scalars:
                    result[key] = torch.from_numpy(np.array(list(batched[key])))
            elif isinstance(batched[key][0], torch.Tensor):
                if combine_tensors:
                    result[key] = torch.stack(list(batched[key]))
            elif isinstance(batched[key][0], np.ndarray):
                if combine_tensors:
                    result[key] = np.array(list(batched[key]))
            elif isinstance(batched[key][0], list) and isinstance(batched[key][0][0], int):
                result[key] = [torch.Tensor(elems).long() for elems in zip(*batched[key])]
            else:
                result[key] = list(batched[key])
        except Exception as e:
            print(key)
            raise e
        # result.append(b)
    del batched
    return result


def frames_as_batch_collation_fn(
    samples, combine_tensors=True, combine_scalars=True, ignore_keys=None,
    video_key="video", hdmap_key="lidar_hdmap",
):
    """Flatten the per-sample temporal dim (N>1) into the batch dim.

    This is intended for image-style training (``is_video_tokenizer=False``)
    where each ``__getitem__`` call returns a tensor of shape ``[C, N, H, W]``
    representing N independent frames from the same clip. After the normal
    collation produces ``[B, C, N, H, W]`` for video/hdmap, we reshape to
    ``[B*N, C, 1, H, W]`` so every frame becomes its own training example, and
    repeat per-sample metadata N times so alignment is preserved.

    If N == 1 this is a no-op and the output is identical to ``dict_collation_fn``.
    """
    batch = dict_collation_fn(samples, combine_tensors, combine_scalars, ignore_keys)

    vid = batch.get(video_key)
    if not (isinstance(vid, torch.Tensor) and vid.ndim == 5):
        return batch

    B, C, N, H, W = vid.shape
    if N <= 1:
        return batch

    def _flatten_frames(t: torch.Tensor) -> torch.Tensor:
        Bt, Ct, Nt, Ht, Wt = t.shape
        return t.permute(0, 2, 1, 3, 4).reshape(Bt * Nt, Ct, 1, Ht, Wt).contiguous()

    batch[video_key] = _flatten_frames(vid)
    _flatten_keys = {video_key, hdmap_key, "bev_hdmap"}
    for fk in (hdmap_key, "bev_hdmap"):
        if isinstance(batch.get(fk), torch.Tensor) and batch[fk].ndim == 5:
            batch[fk] = _flatten_frames(batch[fk])

    for key, v in list(batch.items()):
        if key in _flatten_keys:
            continue
        if isinstance(v, bool):
            continue
        if isinstance(v, torch.Tensor) and v.ndim >= 1 and v.shape[0] == B:
            batch[key] = v.repeat_interleave(N, dim=0)
        elif isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == B:
            batch[key] = np.repeat(v, N, axis=0)
        elif isinstance(v, list) and len(v) == B:
            expanded = []
            for item in v:
                expanded.extend([item] * N)
            batch[key] = expanded

    return batch


class MyDataLoader(DataLoader):
    def __init__(self, dataset, batch_size: int = 1, *args, **kw):
        dataset_obj = dataset.build_dataset()
        print("dl", kw.get("dataloaders"))
        if "dataloaders" in kw:
            kw.pop("dataloaders")  # HACK?
        print("dl2", kw.get("dataloaders"))
        super().__init__(dataset_obj, batch_size, collate_fn=dict_collation_fn, *args, **kw)


class LidarRangeMapDataset:
    def __init__(self, dataset_name, filter_list_path=None, **kwargs):
        self.dataset_name = dataset_name
        self.dataset_config = OmegaConf.create(COMMONDATA_CONFIG[dataset_name])
        self.extra_config = OmegaConf.create(kwargs)
        
        if filter_list_path is not None:
            self.dataset_config["filter_list_path"] = filter_list_path

    def build_dataset(self):
        config_dict = OmegaConf.to_container(self.dataset_config, resolve=True)
        return LidarRangeMapSampler(**config_dict)
    
    
def get_lidar_dataset(dataset_name, filter_list_path=None):
    dataset_config = OmegaConf.create(COMMONDATA_CONFIG[dataset_name])
    
    if filter_list_path is not None:
        dataset_config["filter_list_path"] = filter_list_path
    
    config_dict = OmegaConf.to_container(dataset_config, resolve=True)
    return LidarRangeMapSampler(**config_dict)


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
        load_lidar_only=True,
        load_rgb_only=False,
        load_lidar_and_rgb=False,  # load both lidar and rgb
        load_lidar_or_rgb=False,
        RGB_FPS=30,
        LIDAR_FPS=10,
        load_t5_embeddings=False,
        inverse_depth=False,
        to_three_channels="repeat",  # legacy: "repeat" or "concat" (3 channels)
        process_input=None,  # if set, overrides to_three_channels. One of "repeat", "concat", "mono".
        load_intensity=False,  # if True, append per-frame intensity as an extra channel (dual tokenizer)
        intensity_transform="legacy",
        intensity_raw_max=1.0,
        inv_depth_threshold=20,
        # Optional fixed-size resize of the final range map to [H, W] (use -1 to
        # keep a dim), done with nearest-exact interpolation (NOT bilinear) so
        # depth values are resampled, never blended across depth discontinuities.
        # Mirrors the tokenizer dataset's resize -- use it to feed a VAE at its
        # exact training resolution when downsample_factor_col can't hit it.
        lidar_resize_size=(-1, -1),
        view_indices=None,
        # Which frame of the sampled clip drives the T5 "chunk_id" lookup
        # in prepare_t5_embeddings. The current on-disk layout is one
        # `<__key__>.pkl` per clip (chunk-agnostic), so all three options
        # resolve to the same pickle today, but this kwarg lets us pin
        # the semantic intent ("the caption that describes the first
        # frame of the window") so we get correct behavior if per-chunk
        # pickles ever ship under `<__key__>_<chunk_id>.pkl`.
        #   "middle" (default) -- frame_indices[len(...)//2]  (legacy)
        #   "first"            -- frame_indices[0]
        #   "last"             -- frame_indices[-1]
        t5_chunk_anchor="middle",
        # --- backend dispatch (mirrors the tokenizer LidarRangeMapSampler) --- #
        # "radym" (LidarGEN tar) or "waymo_ncore" (NCore .itar). The waymo backend
        # reads the native 64x2650 grid + rasterizes hdmap on the fly.
        dataset_backend="radym",
        resample_width=-1,  # waymo_ncore only: rebin azimuth to this width; <=0 native
        center_vehicle_forward=False,
        # waymo_ncore hdmap (on-the-fly rasterized from dump_waymo_hdmap NPZs)
        hdmap_root=None,
        hdmap_multichannel=False,
        hdmap_moving_only=False,
        hdmap_include_boxes=True,
        hdmap_box_edge_spacing=0.15,
        hdmap_map_max_dist=None,
        # BEV multi-channel hdmap raster (for BEV-latent conditioning).
        hdmap_bev=False,
        hdmap_bev_size=(256, 256),
        hdmap_bev_pc_range=(-75.0, 75.0, -75.0, 75.0, -3.0, 5.0),
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
        self.intensity_transform = str(intensity_transform)
        if self.intensity_transform not in ("legacy", "log1p"):
            raise ValueError(
                f"Unknown intensity_transform: {self.intensity_transform!r}"
            )
        self.intensity_raw_max = float(intensity_raw_max)
        if self.intensity_raw_max <= 0.0:
            raise ValueError("intensity_raw_max must be positive")
        self.dataset_backend = dataset_backend
        self.resample_width = resample_width
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
            hdmap_bev=hdmap_bev,
            hdmap_bev_size=hdmap_bev_size,
            hdmap_bev_pc_range=hdmap_bev_pc_range,
        )
        self._nuscenes_canonical_kwargs = dict(
            hdmap_sensor_template=hdmap_sensor_template,
            hdmap_template_root_path=hdmap_template_root_path,
            hdmap_template_filter_list_path=hdmap_template_filter_list_path,
            hdmap_virtual_sensor_height_m=hdmap_virtual_sensor_height_m,
            hdmap_virtual_sensor_yaw_deg=hdmap_virtual_sensor_yaw_deg,
            canonical_lidar_mode=canonical_lidar_mode,
            center_vehicle_forward=bool(center_vehicle_forward),
            hdmap_pose_for_lidar_to_world=bool(hdmap_pose_for_lidar_to_world),
        )
        # waymo_ncore only: open the per-view camera zarrs when RGB is requested
        # (the radym backend ignores this; it reads RGB from separate tars).
        self._load_cameras = bool(
            load_lidar_and_rgb or load_rgb_only or load_lidar_or_rgb
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

        self.sample_n_frames = sample_n_frames  # the "actual" target sequence length
        self.lidar_length = lidar_length
        self.rgb_length = lidar_length * (RGB_FPS // LIDAR_FPS)
        self.frame_sampling_method = frame_sampling_method
        assert self.frame_sampling_method in ["random", "sequential", "sequential_from_zero"]
        self.frame_sampling_every_n_frames = frame_sampling_every_n_frames
        # Chunk offset for sequential_from_zero: render frames [start, start+n).
        # Default 0 preserves existing behavior bit-for-bit. Mirrors the tokenizer
        # LidarRangeMapSampler; used by infer_diffusion.py --chunk_start tiling.
        self.sequential_start_idx = int(sequential_start_idx)
        # Optional deterministic-window override: list of (clip_idx, frame_rows).
        # When set, __getitem__/__len__ use it instead of the random clip+start
        # logic, so an external driver can request the EXACT (clip, native rows)
        # window it needs (used by the parity eval that mirrors waymo_ncore).
        self._explicit_windows = None

        self.inverse_depth = inverse_depth
        self.min_distance = min_range
        self.max_distance = max_range
        self.max_range = max_range if not inverse_depth else 1 / min_range
        self.min_range = min_range if not inverse_depth else 1 / max_range

        self.min_value = min_value  # min value after normalisation, if 0 then [0, 1],if -1, then [-1, 1]

        self.repeat_row = repeat_row
        self.repeat_col = repeat_col
        self.repeat_temporal = repeat_temporal
        self.lidar_resize_size = lidar_resize_size
        self.to_three_channels = to_three_channels
        assert self.to_three_channels in ["repeat", "concat"]
        # process_input takes precedence over the legacy to_three_channels arg.
        # "mono" produces a single-channel range map; required for the dual
        # (range + intensity) tokenizer.
        self.process_input = process_input if process_input is not None else to_three_channels
        assert self.process_input in ["repeat", "concat", "mono"], (
            f"Unsupported process_input={self.process_input!r}; "
            "expected one of 'repeat', 'concat', 'mono'."
        )
        if self.load_intensity and self.process_input != "mono":
            log.warning(
                f"[lidar dataset] load_intensity=True with process_input="
                f"{self.process_input!r}; intensity will be appended as an extra "
                f"channel, but this combination is unusual. "
                f"Use process_input='mono' for the dual tokenizer.",
                rank0_only=True,
            )

        self.lidar_size = lidar_size
        self.lidar_crop_size = lidar_crop_size
        self.rgb_crop_size = rgb_crop_size
        self.rgb_resize_size = rgb_resize_size

        self.sample_lidar_prob = sample_lidar_prob if not load_rgb_only else -1
        self.load_lidar_only = load_lidar_only
        self.load_rgb_only = load_rgb_only
        self.load_lidar_and_rgb = load_lidar_and_rgb
        self.load_lidar_or_rgb = load_lidar_or_rgb

        assert sum([self.load_lidar_only, self.load_rgb_only, self.load_lidar_and_rgb, self.load_lidar_or_rgb]) == 1

        # t5 text embedding
        self.load_t5_embeddings = load_t5_embeddings
        self.t5_embedding_path = t5_embedding_path
        if t5_chunk_anchor not in ("first", "middle", "last"):
            raise ValueError(
                f"t5_chunk_anchor must be one of 'first'/'middle'/'last', "
                f"got {t5_chunk_anchor!r}."
            )
        self.t5_chunk_anchor = t5_chunk_anchor
        # One-shot probe flag so prepare_t5_embeddings logs the first real
        # embedding it loads (per dataloader-worker process). Each fork gets
        # its own copy, so we'll see one probe line per worker.
        self._t5_probe_logged = False

        self.length = self.dataset.size_of_filter_set()
        log.critical(f"load_lidar_only: {self.load_lidar_only}")
        log.critical(f"dataset size: {self.length}")
        log.critical(f"load_t5_embeddings: {self.load_t5_embeddings}")
        if self.load_t5_embeddings:
            log.critical(f"t5_embedding_path: {self.t5_embedding_path}")
            if not os.path.isdir(self.t5_embedding_path):
                log.warning(
                    f"[t5] embedding directory does NOT exist: "
                    f"{self.t5_embedding_path} -- prepare_t5_embeddings "
                    f"will fall back to zero embeddings",
                    rank0_only=False,
                )

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

        # Data partitioning
        self.n_data_per_node = self.n_data // dp_world_size
        self.data_start_idx = dp_group_id * self.n_data_per_node
        self.dp_group_id = dp_group_id

        # Infinite loop multiplier. Guard against n_data_per_node == 0, which
        # happens when a (tiny, e.g. overfit) filter set has fewer clips than
        # dp_world_size (n_data // dp_world_size == 0). Such datasets are only
        # ever eagerly constructed at registry import and never iterated by the
        # current experiment, so a 0 multiplier (=> len 0) is harmless; without
        # the guard the bare division kills every job that imports the registry
        # at higher GPU counts.
        maximum_iter = 2000000  # hack to create infinite loop
        self.multiplier = maximum_iter // max(1, self.n_data_per_node)
        # self.multiplier = 1

    def __len__(self):
        """
        Return the artificially multiplied length
        """
        if self._explicit_windows is not None:
            return len(self._explicit_windows)
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
    

    def sample_frame_indices(self, load_lidar):
        if load_lidar:
            if self.frame_sampling_method == "random":
                frame_indices = self.sample_random_frames(self.sample_n_frames, self.lidar_length)
            else:
                frame_indices = self.sample_sequential_frames(
                    self.sample_n_frames // self.repeat_temporal, self.lidar_length
                )

        else:  # load rgb
            if self.frame_sampling_method == "random":
                frame_indices = self.sample_random_frames(self.sample_n_frames, self.rgb_length)
            else:
                frame_indices = self.sample_sequential_frames(
                    self.sample_n_frames + 1, self.rgb_length
                )  # + 1 because the video tokenizer will need one more additional frame
        return frame_indices

    def normalise_and_convert_range_map_to_three_channels(self, range_maps):
        """Normalize the range map and lift to ``[N, C, H, W]``.

        ``C`` is 3 for ``repeat``/``concat`` (legacy 3-channel paths) and 1 for
        ``mono`` (used by the dual range+intensity tokenizer).
        """
        if self.process_input == "repeat":
            range_maps = normalize_range_map(
                range_maps, self.max_distance, self.min_distance, self.min_value, self.inverse_depth
            )  # shape: [N, H, W]
            range_maps = range_maps[:, None, :, :].repeat(3, axis=1)  # shape: [N, 3, H, W]
        elif self.process_input == "mono":
            range_maps = normalize_range_map(
                range_maps, self.max_distance, self.min_distance, self.min_value, self.inverse_depth
            )  # shape: [N, H, W]
            range_maps = range_maps[:, None, :, :]  # shape: [N, 1, H, W]
        elif self.process_input == "concat":
            inverse_depth = normalize_range_map(
                range_maps, self.max_distance, self.min_distance, self.min_value, True
            )  # shape: [N, H, W]
            depth = normalize_range_map(
                range_maps, self.max_distance, self.min_distance, self.min_value, False
            )  # shape: [N, H, W]
            range_maps = np.concatenate(
                [inverse_depth[:, None, :, :], depth[:, None, :, :], depth[:, None, :, :]], axis=1
            )  # shape: [N, 3, H, W]

        return range_maps

    def _normalize_intensity(self, intensity_maps):
        """Apply the intensity tokenizer's input transform."""
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

    def _build_dataset(self, **kwargs):
        """Construct the low-level reader. Defaults to the LidarGEN tar-backed
        ``RadymLidar``; ``dataset_backend='waymo_ncore'`` swaps in the NCore
        reader (native 64x2650 grid + on-the-fly hdmap rasterization). Mirrors
        the tokenizer ``LidarRangeMapSampler._build_dataset``."""
        if self.dataset_backend == "waymo_ncore":
            from tokenizer.training.datasets.lidar_datasets.waymo_ncore import (
                WaymoNCoreLidar,
            )

            return WaymoNCoreLidar(
                resample_width=self.resample_width,
                load_cameras=self._load_cameras,
                **self._hdmap_kwargs,
                **kwargs,
            )
        if self.dataset_backend == "nuscenes_ncore":
            from tokenizer.training.datasets.lidar_datasets.nuscenes_ncore import (
                NuScenesNCoreLidar,
            )

            return NuScenesNCoreLidar(
                resample_width=self.resample_width,
                load_cameras=self._load_cameras,
                **self._hdmap_kwargs,
                **self._nuscenes_canonical_kwargs,
                **kwargs,
            )
        return RadymLidar(**kwargs)

    def process_lidar(self, range_maps, intensity_maps=None):
        """Downsample, normalize, optionally append intensity, then repeat/crop/temporal-pad.

        When ``intensity_maps`` is provided it is downsampled jointly with the range
        map (so the same valid pixels are kept), normalized, and concatenated as an
        extra channel after ``normalise_and_convert_range_map_to_three_channels``.
        """
        if intensity_maps is not None:
            range_maps, [intensity_maps] = self.range_map_downsampler.downsample(
                range_maps, [intensity_maps]
            )  # range_maps: [N, H, W]; intensity: [N, H, W]
        else:
            range_maps = self.range_map_downsampler.downsample(range_maps)  # [N, H, W]

        range_maps = self.normalise_and_convert_range_map_to_three_channels(
            range_maps
        )  # [N, C, H, W] with C=3 (repeat/concat) or C=1 (mono)

        if intensity_maps is not None:
            intensity_np = self._normalize_intensity(intensity_maps)  # [N, 1, H, W]
            range_maps = np.concatenate([range_maps, intensity_np], axis=1)  # [N, C+1, H, W]

        range_maps = range_maps.repeat(self.repeat_row, axis=2)
        range_maps = range_maps.repeat(self.repeat_col, axis=3)

        # only crop along the column dimension
        if self.lidar_crop_size[1] != -1:
            if self.lidar_crop_size[1] > 1600: # we do center crop
                start_col_idx = (range_maps.shape[-1] - self.lidar_crop_size[1]) // 2
                range_maps = range_maps[:, :, :, start_col_idx : start_col_idx + self.lidar_crop_size[1]]
            else:
                start_col_idx = np.random.randint(0, range_maps.shape[-1] - self.lidar_crop_size[1])
                range_maps = range_maps[:, :, :, start_col_idx : start_col_idx + self.lidar_crop_size[1]]

        # if it comes from sequential sampling, then we need to repeat the range maps for  sequence
        if self.frame_sampling_method in ("sequential", "sequential_from_zero") and self.is_video_tokenizer:
            first_range_map = range_maps[
                0
            ]  # let's just repeat the first frame to make task easier. This is the same practice as when I evaluate the tokenizer
            range_maps = range_maps.repeat(self.repeat_temporal, axis=0)
            range_maps = np.concatenate(
                [first_range_map[None], range_maps], axis=0
            )  # this one will be repeated later on in the encoder

        # to torch tensor
        range_maps = torch.from_numpy(range_maps).float()  # [N, C(+1), H, W]

        # Optional fixed-size resize (nearest-exact). Applied to [N, C, H, W] so
        # depth values are resampled, never blended. Matches the tokenizer's
        # ``lidar_resize_size`` so a VAE can be fed at its training resolution.
        if self.lidar_resize_size is not None and (
            self.lidar_resize_size[0] != -1 or self.lidar_resize_size[1] != -1
        ):
            _, _, H, W = range_maps.shape
            target_h = self.lidar_resize_size[0] if self.lidar_resize_size[0] != -1 else H
            target_w = self.lidar_resize_size[1] if self.lidar_resize_size[1] != -1 else W
            if (target_h, target_w) != (H, W):
                range_maps = torch.nn.functional.interpolate(
                    range_maps, size=(target_h, target_w), mode="nearest-exact"
                )

        return range_maps

    def process_lidar_multichannel(self, range_maps):
        """Process ``(N,K,H,W)`` semantic range maps without mixing K."""
        if range_maps.ndim != 4:
            raise ValueError(
                "multi-channel HDMap must be (N,K,H,W), got "
                f"{tuple(range_maps.shape)}"
            )
        if self.process_input != "mono":
            raise ValueError(
                "multi-channel HDMap processing requires process_input='mono'"
            )
        num_frames, num_channels = range_maps.shape[:2]
        flat = range_maps.reshape(
            num_frames * num_channels, *range_maps.shape[2:]
        )
        flat = self.range_map_downsampler.downsample(flat)
        flat = self.normalise_and_convert_range_map_to_three_channels(flat)
        if flat.shape[1] != 1:
            raise RuntimeError(
                "semantic HDMap normalization must preserve one depth channel"
            )
        range_maps = flat.reshape(
            num_frames, num_channels, *flat.shape[2:]
        )
        range_maps = range_maps.repeat(self.repeat_row, axis=2)
        range_maps = range_maps.repeat(self.repeat_col, axis=3)

        if self.lidar_crop_size[1] != -1:
            crop_width = self.lidar_crop_size[1]
            if crop_width > 1600:
                start_col_idx = (range_maps.shape[-1] - crop_width) // 2
            else:
                start_col_idx = np.random.randint(
                    0, range_maps.shape[-1] - crop_width
                )
            range_maps = range_maps[
                :, :, :, start_col_idx : start_col_idx + crop_width
            ]

        if (
            self.frame_sampling_method
            in ("sequential", "sequential_from_zero")
            and self.is_video_tokenizer
        ):
            first_range_map = range_maps[0]
            range_maps = range_maps.repeat(self.repeat_temporal, axis=0)
            range_maps = np.concatenate(
                [first_range_map[None], range_maps], axis=0
            )

        range_maps = torch.from_numpy(range_maps).float()
        if self.lidar_resize_size is not None and (
            self.lidar_resize_size[0] != -1
            or self.lidar_resize_size[1] != -1
        ):
            n, k, height, width = range_maps.shape
            target_h = (
                self.lidar_resize_size[0]
                if self.lidar_resize_size[0] != -1
                else height
            )
            target_w = (
                self.lidar_resize_size[1]
                if self.lidar_resize_size[1] != -1
                else width
            )
            if (target_h, target_w) != (height, width):
                range_maps = torch.nn.functional.interpolate(
                    range_maps.reshape(n * k, 1, height, width),
                    size=(target_h, target_w),
                    mode="nearest-exact",
                ).reshape(n, k, target_h, target_w)
        return range_maps

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
        if self.t5_chunk_anchor == "first":
            anchor_indice = frame_indices[0]
        elif self.t5_chunk_anchor == "last":
            anchor_indice = frame_indices[-1]
        else:  # "middle" (legacy default)
            anchor_indice = frame_indices[len(frame_indices) // 2]
        chunk_id = anchor_indice // int(self.lidar_length / 5)

        if self.load_t5_embeddings:
            # t5_embed_path = os.path.join(self.t5_embedding_path, example["__key__"] + f"_{chunk_id}.pkl")
            t5_embed_path = os.path.join(self.t5_embedding_path, example["__key__"] + ".pkl")
            if not os.path.exists(t5_embed_path):
                # Mirror the lidar/hdmap behavior: signal __getitem__ to draw a
                # different sample instead of silently feeding zero embeddings.
                raise FileNotFoundError(
                    f"t5 embedding missing for {example['__key__']}: {t5_embed_path}"
                )
            # Truncated/corrupt .pkl files exist in the wild. Re-raise so the
            # caller can retry with a different index (same pattern as
            # hdmap/lidar load failures).
            try:
                with open(t5_embed_path, "rb") as fh:
                    t5_embed_pickle = pickle.load(fh)
                text_embedding = t5_embed_pickle["pickle"]["ground_truth"]["embeddings"]["t5_xxl"]
            except (EOFError, pickle.UnpicklingError, OSError, KeyError) as e:
                raise RuntimeError(
                    f"[t5] failed to load {t5_embed_path} "
                    f"({type(e).__name__}: {e})"
                ) from e

            t5_embed = {}
            dummy_text_embedding = torch.zeros(512, 1024)
            dummy_text_mask = torch.zeros(512)
            n_text = text_embedding.shape[0]  # 512
            dummy_text_embedding[:n_text] = torch.from_numpy(text_embedding)
            dummy_text_mask[:n_text] = 1
            t5_embed["t5_text_embeddings"] = dummy_text_embedding
            t5_embed["t5_text_mask"] = dummy_text_mask
            # One-shot probe: confirm to the training log that real
            # (non-zero) T5 embeddings are flowing through this worker.
            if not self._t5_probe_logged:
                nz_rows = int(
                    (dummy_text_embedding.abs().sum(dim=-1) > 0).sum()
                )
                log.critical(
                    f"[t5 probe] pid={os.getpid()} loaded {t5_embed_path} "
                    f"shape={tuple(text_embedding.shape)} "
                    f"nonzero_rows={nz_rows}/{dummy_text_embedding.shape[0]}",
                    rank0_only=False,
                )
                self._t5_probe_logged = True
            del t5_embed_pickle
        else:
            t5_embed = self.prepare_dummy_data()
            
        if self.num_views > 1:
            t5_embed["t5_text_embeddings"] = torch.cat([t5_embed["t5_text_embeddings"]] * (self.num_views+self.plus), dim=0)
            t5_embed["t5_text_mask"] = torch.cat([t5_embed["t5_text_mask"]] * (self.num_views+self.plus), dim=0)

        return t5_embed

    def get_lidar(self, example, lidar_key = "lidar"):
        if lidar_key in example["custom"]:
            range_maps = example["custom"][lidar_key]  # shape: [N, 128, 3600]
            intensity_maps = (
                example["custom"].get("lidar_intensity") if self.load_intensity else None
            )
            range_maps = self.process_lidar(range_maps, intensity_maps=intensity_maps)
            range_maps = range_maps.permute(1, 0, 2, 3).contiguous()  # shape: [C, N, H, W]
        else:
            range_maps = None

        return range_maps

    def get_lidar_hdmap(self, example, hdmap_key = "hdmap"):
        if hdmap_key in example["custom"]:
            hdmap = example["custom"][hdmap_key]  # shape: [N, 128, 3600]
            if hdmap.ndim == 4:
                hdmap = self.process_lidar_multichannel(hdmap)
            else:
                hdmap = self.process_lidar(hdmap)
            hdmap = hdmap.permute(1, 0, 2, 3).contiguous()
        else:
            hdmap = None
        return hdmap

    def get_lidar_hdmap_bev(self, example, key="hdmap_bev"):
        """Multi-channel BEV hdmap raster -> (C_cls, N, Y, X) tensor.

        Already a {0,1} top-down occupancy grid (one channel per semantic
        class); NOT a range map, so it skips ``process_lidar`` (no depth
        normalize / azimuth resize / repeat-crop) -- just tensor + permute to
        the (C, N, H, W) convention the collation flattens."""
        if key not in example["custom"]:
            return None
        bev = example["custom"][key]  # (N, C_cls, Y, X) float32 in {0, 1}
        if not isinstance(bev, torch.Tensor):
            bev = torch.from_numpy(bev)
        return bev.float().permute(1, 0, 2, 3).contiguous()  # (C_cls, N, Y, X)

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
        if self._explicit_windows is not None:
            # Deterministic parity mode: caller-provided (clip_idx, native rows).
            lidar_data_idx, lidar_frame_indices = self._explicit_windows[idx]
            lidar_frame_indices = np.asarray(lidar_frame_indices, dtype=np.int64)
        else:
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

        lidar_rangemap = self.get_lidar(example)
        lidar_hdmap = self.get_lidar_hdmap(example)
        bev_hdmap = self.get_lidar_hdmap_bev(example)
        rgb_images = self.get_rgb(example)
        rgb_hdmap = self.get_rgb_hdmap(example)
        try:
            t5_embed = self.prepare_t5_embeddings(example, lidar_frame_indices)
        except (FileNotFoundError, RuntimeError) as e:
            log.warning(
                f"T5 embedding load failed for index {idx} "
                f"(key={example['__key__']}): {e}; retrying with new index",
                rank0_only=False,
            )
            return self.__getitem__(np.random.randint(0, self.__len__()))
        
        
        cam_intrinsics = self.get_camera_intrinsics(example)
        cam_c2lidar_transform = self.get_camera_c2lidar_transform(example)

        key = "video"

        sample = {}
        sample[key] = lidar_rangemap
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
        # Sweep-start ego poses for the *actually sampled* frames (sequential
        # sampling uses a random start_idx) -- forwarded so --save_poses dumps
        # poses aligned to the dumped range maps rather than re-deriving from
        # metadata assuming start frame 0.
        sample["lidar_to_world"] = np.asarray(
            example["lidar_to_world"], dtype=np.float32)
        sample["rgb_frame_indices"]=example["rgb_frame_indices"]
        sample.update(t5_embed)

        if lidar_hdmap is not None:
            sample["lidar_hdmap"] = lidar_hdmap
        if bev_hdmap is not None:
            sample["bev_hdmap"] = bev_hdmap
        if rgb_images is not None:
            sample.update(rgb_images)
        if rgb_hdmap is not None:
            sample.update(rgb_hdmap)
        if self.load_lidar_and_rgb:
            
            sample.update(cam_intrinsics)
            sample.update(cam_c2lidar_transform)
            
        
        del example
        gc.collect()
        return sample
