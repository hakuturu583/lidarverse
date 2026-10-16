# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn
from omegaconf import OmegaConf

from utils.lidar_rangemap import (
    range_map_to_ray_directions,
    load_pandar128_elevations,
    undo_row_col_temporal_repeat,
)
from utils.lidar_rangemap import unnormalize_and_reduce_channels
from utils.per_row_norm import get_per_row_normalizer, invert_per_row_norm
from tokenizer.training.datasets.lidar_datasets.configs import COMMONDATA_CONFIG

class GeoConverter(nn.Module):
    def __init__(self, data_config=dict()):
        super().__init__()

        if data_config.get("uses_sensor_metadata", False):
            shared = OmegaConf.to_container(
                OmegaConf.create(data_config.get("shared_overrides", {})),
                resolve=True,
            )
            self.config = {
                "repeat_row": 1,
                "repeat_col": 1,
                "repeat_temporal": 1,
                **shared,
            }
            if any(
                self.config[key] != 1
                for key in ("repeat_row", "repeat_col", "repeat_temporal")
            ):
                raise ValueError(
                    "Metadata-driven geometry requires un-repeated native grids"
                )
            self.depth_min = self.depth_max = None
            self.process_output = self.config.get("process_output", "mono")
            self.has_intensity = self.config.get("load_intensity", False)
            self.per_row_normalizer = None
            self.ray_directions = self.ray_origins = None
            return

        dataset_config = OmegaConf.create(COMMONDATA_CONFIG[data_config.dataset_name])
        extra_config = OmegaConf.create(data_config)
        config = OmegaConf.merge(dataset_config, extra_config)
        self.config = OmegaConf.to_container(config, resolve=True)

        self.depth_min, self.depth_max = self.config["min_range"], self.config["max_range"]
        self.process_output = self.config["process_output"]
        self.has_intensity = self.config.get("load_intensity", False)

        # Per-row normalizer (None when disabled): inverted before unnormalize.
        self.per_row_normalizer = get_per_row_normalizer(
            self.config.get("per_row_normalize", False),
            self.config.get("per_row_stats_path", ""),
            min_range=self.depth_min,
            max_range=self.depth_max,
            min_value=self.config["min_value"],
            smooth_sigma=self.config.get("per_row_smooth_sigma", 5.0),
            log_depth=False,
        )

        self.init_ray_directions()

    def init_ray_directions(self):
        # Waymo NCore backend: use the sensor's measured elevation/azimuth tables
        # (+ per-row offset) instead of the Pandar128 uniform-linspace geometry,
        # otherwise the back-projection is geometrically wrong AND the ray grid
        # shape (Pandar128 128xWidth) would not even match the 64xW Waymo range
        # map. Built once from the first clip (all top-lidar clips share the model).
        _backend = self.config.get("dataset_backend", "radym")
        if _backend in ("waymo_ncore", "nuscenes_ncore"):
            if _backend == "waymo_ncore":
                from tokenizer.training.datasets.lidar_datasets.waymo_ncore import (
                    load_waymo_ray_directions as load_ncore_ray_directions,
                )
            else:
                from tokenizer.training.datasets.lidar_datasets.nuscenes_ncore import (
                    load_nuscenes_ray_directions as load_ncore_ray_directions,
                )
            ray_kwargs = {
                "root_path": self.config["root_path"],
                "filter_list_path": self.config.get("filter_list_path", None),
                "lidar_id": self.config.get("lidar_id", "lidar_top"),
                "resample_width": self.config.get("resample_width", -1),
            }
            if _backend == "nuscenes_ncore":
                ray_kwargs["center_vehicle_forward"] = self.config.get(
                    "center_vehicle_forward", False
                )
            ray_np = load_ncore_ray_directions(**ray_kwargs)
            ray_directions = torch.from_numpy(ray_np).float()  # (H, W, 3)
            self.ray_directions = ray_directions[None, None, ...].cuda()
            self.ray_origins = torch.zeros_like(self.ray_directions).cuda()
            return

        lidar_width = 3600 // self.config["downsample_factor_col"]
        # If the range map is resized to a fixed width (nearest-exact, see
        # rangemap_sampler), build the azimuth grid at that width so the
        # per-column ray directions line up with the resized range image.
        # The FOV is unchanged (uniform over the full azimuth), so a linspace
        # over the resized width is the correct correspondence.
        resize = self.config.get("lidar_resize_size", None)
        if resize is not None and resize[1] != -1:
            lidar_width = int(resize[1])
        elevation_angles = torch.tensor(load_pandar128_elevations(), dtype=torch.float32)
        
        ray_directions = torch.from_numpy(
            range_map_to_ray_directions(lidar_width, elevation_angles)
        ).float() # shape: (H, W, 3)
        
        self.ray_directions = ray_directions[None, None, ...].cuda() # shape: (1, 1, H, W, 3)
        self.ray_origins = torch.zeros_like(self.ray_directions).cuda() # shape: (1, 1, H, W, 3)

    def _forward_sensor_metadata(
        self,
        input: torch.Tensor,
        ray_directions: torch.Tensor,
        depth_bounds: torch.Tensor,
        return_valid: bool,
    ):
        """Back-project a homogeneous batch with its exact sensor metadata."""
        depth_norm = input[:, :1]
        depth_norm = undo_row_col_temporal_repeat(
            depth_norm,
            self.config["repeat_row"],
            self.config["repeat_col"],
            self.config["repeat_temporal"],
        )
        if depth_norm.ndim == 4:
            depth_norm = depth_norm.unsqueeze(2)
        depth_norm = depth_norm[:, 0]

        bounds = depth_bounds.to(device=input.device, dtype=input.dtype)
        if bounds.ndim == 1:
            bounds = bounds.unsqueeze(0)
        if bounds.shape != (input.shape[0], 2):
            raise ValueError(
                f"sensor_depth_bounds must have shape (B,2), got {tuple(bounds.shape)}"
            )
        depth_min = bounds[:, 0, None, None, None]
        depth_max = bounds[:, 1, None, None, None]
        depth = (depth_norm.clamp(-1, 1) + 1.0) * 0.5
        depth = depth * (depth_max - depth_min) + depth_min

        rays = ray_directions.to(device=input.device, dtype=input.dtype)
        if rays.ndim == 3:
            rays = rays.unsqueeze(0)
        if rays.shape != (input.shape[0], *depth.shape[-2:], 3):
            raise ValueError(
                "sensor_ray_directions must match (B,H,W,3): "
                f"got {tuple(rays.shape)} for depth {tuple(depth.shape)}"
            )
        points = depth[..., None] * rays[:, None]
        valid = (depth > depth_min) & (depth < depth_max)
        if return_valid:
            return points, valid
        return points

    def forward(
        self,
        input,
        return_valid: bool = False,
        ray_directions: torch.Tensor | None = None,
        depth_bounds: torch.Tensor | None = None,
    ):
        """
        input: [N, C, H, W] or [N, C, T, H, W] in normalized space (matching
            the dataset's ``min_value`` convention -- typically [-1, 1]).

        Returns:
            points: [N, 1, H, W, 3] (4D input) or [N, T, H, W, 3] (5D input)
                in metric meters (i.e. unnormalized to [min_range, max_range]).
            valid_mask (only if return_valid=True): same leading shape as
                ``points`` minus the trailing 3 (i.e. [N, 1, H, W] / [N, T, H, W])
                bool tensor marking pixels whose recovered depth lies within
                [min_range, max_range].
        """
        if (ray_directions is None) != (depth_bounds is None):
            raise ValueError(
                "ray_directions and depth_bounds must be provided together"
            )
        if ray_directions is not None:
            return self._forward_sensor_metadata(
                input, ray_directions, depth_bounds, return_valid
            )

        # Invert per-row z-score back to [-1, 1] space BEFORE the clamp (no-op
        # when disabled). Operates on the H dim, broadcasting over N/C/T.
        input = invert_per_row_norm(input, self.per_row_normalizer)
        input = input.clamp(-1, 1)
        input = undo_row_col_temporal_repeat(
            input,
            self.config["repeat_row"],
            self.config["repeat_col"],
            self.config["repeat_temporal"],
        )

        unnorm_result = unnormalize_and_reduce_channels(
            input,
            self.depth_max,
            self.depth_min,
            self.config["min_value"],
            self.config["inverse_depth"],
            self.process_output,
            near_buffer=0,
            far_buffer=0,
            inv_depth_threshold=self.config["inv_depth_threshold"],
            num_bands=self.config["num_bands"],
            apply_valid_mask=False,
            has_intensity=self.has_intensity,
            intensity_transform=self.config.get("intensity_transform", "legacy"),
        )
        if self.has_intensity:
            depth, valid_mask, _intensity = unnorm_result
        else:
            depth, valid_mask = unnorm_result
        # depth / valid_mask shape: (N, H, W) or (N, T, H, W)

        assert depth.shape[-2:] == self.ray_directions.shape[-3:-1], \
            f"Input shape {depth.shape} does not match ray directions shape {self.ray_directions.shape}"

        if depth.ndim == 3:
            depth = depth.unsqueeze(1)
            valid_mask = valid_mask.unsqueeze(1)

        points = self.ray_origins + self.ray_directions * depth[..., None]

        if return_valid:
            return points, valid_mask
        return points
