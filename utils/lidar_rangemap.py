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


import os
import numpy as np
import torch
import yaml
import re
from einops import rearrange
from matplotlib import cm
import mediapy as media
import matplotlib.pyplot as plt
from utils.camera.ftheta import FThetaCamera
from utils.misc import make_sure_numpy
from scipy import spatial, interpolate
from scipy.spatial.transform import Rotation as R

RANGE_BANDS = {
    3: [5, 9.898, 21.813, 100],
    4: [5, 7.933, 15.109, 26.813, 100],
    8: [5, 5.762, 7.934, 11.023, 15.109, 19.906, 26.813, 40.656, 100],
    16: [
        0, 4.973, 5.762, 6.715, 7.934, 9.367, 11.023, 12.930, 15.109, 17.422,
        19.906, 22.891, 26.813, 32.219, 40.656, 56.125, 100
    ],
    32: [
        0, 4.563, 4.973, 5.352, 5.762, 6.207, 6.715, 7.289, 7.934, 8.625,
        9.367, 10.172, 11.023, 11.938, 12.930, 13.992, 15.109, 16.250, 17.422,
        18.625, 19.906, 21.313, 22.891, 24.719, 26.813, 29.297, 32.219, 35.906,
        40.656, 47.063, 56.125, 71.125, 100
    ],
}

def transform_point_cloud(pc, T):
    """Transform the point cloud with the provided transformation matrix,
        support torch.Tensor and np.ndarry.
    Args:
        pc (np.array): point cloud coordinates (x,y,z) [num_pts, 3] or [bs, num_pts, 3]
        T (np.array): se3 transformation matrix  [4, 4] or [bs, 4, 4]

    Out:
        (np array): transformed point cloud coordinated [num_pts, 3] or [bs, num_pts, 3]
    """
    if len(pc.shape) == 3:
        if isinstance(pc, np.ndarray):
            trans_pts = T[:, :3, :3] @ pc.transpose(0, 2, 1) + T[:, :3, 3:4]
            return trans_pts.transpose(0, 2, 1)
        else:
            trans_pts = T[:, :3, :3] @ pc.permute(0, 2, 1) + T[:, :3, 3:4]
            return trans_pts.permute(0, 2, 1)

    else:
        trans_pts = T[:3, :3] @ pc.transpose() + T[:3, 3:4]
        return trans_pts.transpose()

class PoseInterpolator:
    """
    Interpolates the poses to the desired time stamps. The translation component is interpolated linearly,
    while spherical linear interpolation (SLERP) is used for the rotations. https://en.wikipedia.org/wiki/Slerp

    Args:
        poses (np.array): poses at given timestamps in a se3 representation [n,4,4]
        timestamps (np.array): timestamps of the known poses [n]
        ts_target (np.array): timestamps for which the poses will be interpolated [m]
    Out:
        (np.array): interpolated poses in se3 representation [m,4,4]
    """

    def __init__(self, poses, timestamps):
        self.slerp = spatial.transform.Slerp(timestamps, R.from_matrix(poses[:, :3, :3]))
        self.f_x = interpolate.interp1d(timestamps, poses[:, 0, 3])
        self.f_y = interpolate.interp1d(timestamps, poses[:, 1, 3])
        self.f_z = interpolate.interp1d(timestamps, poses[:, 2, 3])

        self.last_row = np.array([0, 0, 0, 1], dtype=np.float32).reshape(1, 1, -1)

        self.start_timestamp = timestamps[0]
        self.end_timestamp = timestamps[-1]

    def in_range(self, ts) -> bool:
        """Returns true if all provided timestamps (scalar or array-like) are within the interpolation range"""
        return (
            np.logical_and(self.start_timestamp <= (ts_array := np.asarray(ts)), ts_array <= self.end_timestamp)
            .all()
            .item()
        )

    def interpolate_to_timestamps(self, ts_target):
        x_interp = self.f_x(ts_target).reshape(-1, 1, 1).astype(np.float32)
        y_interp = self.f_y(ts_target).reshape(-1, 1, 1).astype(np.float32)
        z_interp = self.f_z(ts_target).reshape(-1, 1, 1).astype(np.float32)
        R_interp = self.slerp(ts_target).as_matrix().reshape(-1, 3, 3).astype(np.float32)

        t_interp = np.concatenate([x_interp, y_interp, z_interp], axis=-2)

        return np.concatenate(
            (np.concatenate([R_interp, t_interp], axis=-1), np.tile(self.last_row, (R_interp.shape[0], 1, 1))), axis=1
        )

def load_each_frame_from_tar_data(tar_data, frame_idx, n_rows=128, n_cols=3600, load_intensity=False):
    """
    Load range map (and optionally intensity map) of each frame.
    When load_intensity=True, returns (range_map, intensity_map).
    """
    lidar_row = np.load(tar_data.extractfile(f'{frame_idx}.lidar_row.npz'))['arr_0']
    lidar_col = np.load(tar_data.extractfile(f'{frame_idx}.lidar_col.npz'))['arr_0']
    lidar_range = np.load(tar_data.extractfile(f'{frame_idx}.lidar_range.npz'))['arr_0']
    
    # convert to range map
    range_map = np.zeros((n_rows, n_cols), dtype=np.float32)
    range_map[lidar_row, lidar_col] = lidar_range.astype(np.float32)
    
    if load_intensity:
        lidar_intensity = np.load(tar_data.extractfile(f'{frame_idx}.lidar_intensity.npz'))['arr_0']
        intensity_map = np.zeros((n_rows, n_cols), dtype=np.float32)
        intensity_map[lidar_row, lidar_col] = lidar_intensity.astype(np.float32)
        return range_map, intensity_map

    return range_map

def load_range_map(tar_file, n_rows=128, n_cols=3600):
    frame_idx_list = np.array([x.strip(".lidar_row.npz") for x in tar_file.getnames() if "lidar_row" in x])
        
    range_map_list = []    
    for frame_idx in frame_idx_list:
        range_map = load_each_frame_from_tar_data(tar_file, frame_idx, n_rows, n_cols)
        range_map_list.append(range_map)
    range_map = np.stack(range_map_list, axis=0)
    return range_map


def load_pandar128_elevations():
    with open("assets/lidar/lidar_model.yaml", "r") as file:
        config = yaml.safe_load(file)

    # Access the inclinations_deg from the HESAI-Pandar128 model
    elevations = config["lidar_models"]["spinning"]["HESAI-Pandar128"]["inclinations_deg"]
    return elevations  # load the elevation angles


def undo_row_col_temporal_repeat(range_map, repeat_row, repeat_col, repeat_temporal=None):
    """
    range_map: numpy array of shape (N, 3, H, W) / N, 3, T, H, W
    """
    dim = range_map.ndim
    assert dim in [4, 5], f"Range map dimension must be 4 or 5, but got {dim}"
    if dim == 4:  # undo row and col repeat
        range_map = range_map[:, :, repeat_row // 2 :: repeat_row, repeat_col // 2 :: repeat_col]
    elif dim == 5:  # undo row, col and temporal repeat
        range_map = range_map[:, :, :, repeat_row // 2 :: repeat_row, repeat_col // 2 :: repeat_col]
        assert repeat_temporal is not None, "repeat_temporal must be provided for 5D range map"
        if repeat_temporal > 1:
            range_map = torch.cat(
                (range_map[:, :, 0:1, :, :], range_map[:, :, 1 + repeat_temporal // 2 :: repeat_temporal, :, :]), dim=2
            )  # add 1+ to skip the first reconstructed frame

    return range_map

def normalize_range_map(range_map, max_range, min_range, min_value, inverse_depth):
    """
    Normalize the range map from [min_range, max_range] to [-1, 1]/[0, 1]
    Input:
        range_map: numpy array of shape (N, H, W) / (N, 3, H, W)
        max_range: float
        min_range: float
        min_value: float
        inverse_depth: bool
    Output:
        range_map: numpy array of shape (N, H, W) / (N, 3, H, W)
    """
    c_min_range = min_range if not inverse_depth else 1 / max_range
    c_max_range = max_range if not inverse_depth else 1 / min_range

    if inverse_depth:
        mask = range_map == 0
        range_map[mask] = max_range
        range_map = 1 / range_map
        range_map[mask] = c_min_range

    # normalise the range map
    if isinstance(range_map, torch.Tensor):
        range_map = torch.clamp(range_map, c_min_range, c_max_range)
    else:
        range_map = np.clip(range_map, c_min_range, c_max_range)

    if min_value == -1:
        range_map = (range_map - c_min_range) / (c_max_range - c_min_range) * 2 - 1
    else:
        range_map = (range_map - c_min_range) / (c_max_range - c_min_range)

    return range_map


def unnormalize_range_map(
    range_map, max_range, min_range, min_value, inverse_depth, near_buffer, far_buffer, valid_mask=None, apply_valid_mask=True
):
    """
    Unnormalize the range map from [-1, 1]/[0, 1] to [min_range, max_range]
    Input:
        range_map: numpy array of shape (N, H, W)
    Output:
        range_map: numpy array of shape (N, H, W)
    """
    c_min_range = min_range if not inverse_depth else 1 / max_range
    c_max_range = max_range if not inverse_depth else 1 / min_range

    if min_value == -1:
        # range_map = range_map.clip(-1, 1)
        range_map = (range_map + 1) / 2 * (c_max_range - c_min_range) + c_min_range
    else:
        # range_map = range_map.clip(0, 1)
        range_map = range_map * (c_max_range - c_min_range) + c_min_range

    if inverse_depth:
        range_map = 1 / range_map

    if valid_mask is None:
        valid_mask = (range_map > min_range + near_buffer) & (
            range_map < max_range - far_buffer
        )  # add buffer to deal with low precision of bfloat16

    if apply_valid_mask:
        if isinstance(range_map, torch.Tensor):
            range_map = torch.where(valid_mask, range_map, torch.zeros_like(range_map))
        else:
            range_map = range_map.copy()
            range_map[~valid_mask] = 0

    return range_map, valid_mask


def unnormalize_intensity(intensity_map, min_value, transform="legacy"):
    """Invert the selected intensity transform.

    ``legacy`` preserves the historical [min_value, 1] <-> [0, 1] contract.
    ``log1p`` stores normalized log-intensity in [-1, 1] and inverts it back
    to normalized linear intensity in [0, 1]. The latter is opt-in for
    cross-dataset tokenizers.
    """
    if transform == "log1p":
        if isinstance(intensity_map, torch.Tensor):
            log_intensity = (intensity_map.clamp(-1, 1) + 1.0) * (
                0.5 * np.log(2.0)
            )
            return torch.expm1(log_intensity)
        log_intensity = (np.clip(intensity_map, -1, 1) + 1.0) * (
            0.5 * np.log(2.0)
        )
        return np.expm1(log_intensity)
    if transform != "legacy":
        raise ValueError(f"Unknown intensity transform: {transform!r}")
    if min_value == -1:
        return (intensity_map + 1) / 2
    return intensity_map


def unnormalize_and_reduce_channels(
    range_map,
    max_range,
    min_range,
    min_value,
    inverse_depth,
    process_output,
    near_buffer=0.5,
    far_buffer=0.5,
    inv_depth_threshold=20,
    num_bands=3,
    apply_valid_mask=True,
    has_intensity=False,
    intensity_transform="legacy",
    channel_reduce=None,
):
    """
    Unnormalize the range map and reduce the channels to 1 (output processing).
    Input:
        range_map: tensor of shape (N, C, H, W) or (N, C, T, H, W)
        has_intensity: if True, the last channel is intensity and will be stripped
                       before processing range channels. The unnormalized intensity
                       is returned as the third element of the tuple.
        channel_reduce: how to collapse the repeated range channels ("mean" or
                        "median"). Defaults to the ``LIDAR_CHANNEL_REDUCE`` env
                        var (falling back to "mean"), so existing callers are
                        unchanged unless that env var is set.
    Output:
        Without intensity: (final_range_map, valid_mask)
        With intensity:    (final_range_map, valid_mask, intensity_map)
    """
    if channel_reduce is None:
        channel_reduce = os.environ.get("LIDAR_CHANNEL_REDUCE", "mean")

    def _reduce_channels(x):
        if channel_reduce == "median":
            return x.median(dim=1).values
        return x.mean(dim=1)

    intensity_map = None
    if has_intensity:
        intensity_map = unnormalize_intensity(
            range_map[:, -1], min_value, transform=intensity_transform
        )
        range_map = range_map[:, :-1]

    if process_output == "repeat":
        range_map = _reduce_channels(range_map)
        final_range_map, valid_mask = unnormalize_range_map(
            range_map, max_range, min_range, min_value, inverse_depth, near_buffer, far_buffer, apply_valid_mask=apply_valid_mask
        )
    elif process_output == "mono":
        # Input: (N, 1, H, W) or (N, 1, T, H, W); single channel, just unnormalize
        range_map = range_map.squeeze(1)
        final_range_map, valid_mask = unnormalize_range_map(
            range_map, max_range, min_range, min_value, inverse_depth, near_buffer, far_buffer, apply_valid_mask=apply_valid_mask
        )
    elif process_output == "concat":
        # the first channel is inverse depth, the second and third channels are depths
        inverse_depth = range_map[:, 0]  # shape: (N, H, W) or (N, T, H, W)
        depth = range_map[:, 1:].mean(dim=1)  # shape: (N, H, W) or (N, T, H, W)

        inverse_depth, inv_depth_valid_mask = unnormalize_range_map(
            inverse_depth, max_range, min_range, min_value, True, near_buffer, far_buffer, apply_valid_mask=apply_valid_mask
        )
        depth, depth_valid_mask = unnormalize_range_map(
            depth, max_range, min_range, min_value, False, near_buffer, far_buffer, apply_valid_mask=apply_valid_mask
        )

        # for the valid pixels, if the inverse depth value is below 20m, then use the inverse depth value, otherwise use the depth value
        inv_depth_valid_mask = inv_depth_valid_mask & (inverse_depth < inv_depth_threshold)
        final_range_map = torch.where(inv_depth_valid_mask, inverse_depth, depth)
        valid_mask = depth_valid_mask | inv_depth_valid_mask
    elif process_output == "band":
        assert range_map.shape[1] == num_bands, f"Range map {range_map.shape} must have {num_bands} channels"
        valid_mask = torch.ones_like(range_map[:, 0], dtype=torch.bool)
        
        final_range_map = torch.zeros_like(range_map[:, 0])
        for i in range(num_bands):
            if i == 0:
                band_range_map, _ = unnormalize_range_map(
                    range_map[:, i], 
                    RANGE_BANDS[num_bands][i+1], 
                    RANGE_BANDS[num_bands][i], 
                    min_value, 
                    inverse_depth, 
                    near_buffer, far_buffer, apply_valid_mask=False
                )
            else:
                band_range_map, _ = unnormalize_range_map(
                    range_map[:, i], 
                    RANGE_BANDS[num_bands][i+1] - RANGE_BANDS[num_bands][i], 
                    0, 
                    min_value, 
                    inverse_depth, 
                    near_buffer, far_buffer, apply_valid_mask=False
                )
            final_range_map += band_range_map

        valid_mask = (final_range_map > min_range + near_buffer) & (
            final_range_map < max_range - far_buffer
        )
        if apply_valid_mask:
            final_range_map = torch.where(valid_mask, final_range_map, torch.zeros_like(final_range_map))
    else:
        raise ValueError(f"Invalid process_output: {process_output}")

    if has_intensity:
        return final_range_map, valid_mask, intensity_map
    return final_range_map, valid_mask

def range_map_to_ray_directions(n_cols, sensor_elevation_angles):
    """
    Generate ray directions for each pixel in a range map.

    Args:
        n_cols: number of columns in the range map
        sensor_elevation_angles: list/array of elevation angles in degrees for each scan line
                                 (determines the number of rows)

    Returns:
        ray_directions: numpy array of shape (n_rows, n_cols, 3) containing unit vectors
                        for each pixel's ray direction
    """
    # Create arrays for azimuth and elevation angles
    # Azimuth angles are uniformly distributed around 360 degrees
    azimuth_angles = np.linspace(np.pi, -np.pi, n_cols, endpoint=False)

    # Convert elevation angles from degrees to radians
    elevation_angles_rad = np.radians(sensor_elevation_angles)

    # Create meshgrid for all combinations of angles
    elevation_grid, azimuth_grid = np.meshgrid(elevation_angles_rad, azimuth_angles, indexing="ij")

    x = np.cos(elevation_grid) * np.cos(azimuth_grid)  # shape: (H, W)
    y = np.cos(elevation_grid) * np.sin(azimuth_grid)
    z = np.sin(elevation_grid)

    # Stack to create ray directions
    ray_directions = np.stack([x, y, z], axis=-1)  # shape: (H, W, 3)

    return ray_directions


def build_decomp_ray_directions(resolution, ray_dirs_path: str = ""):
    """Per-pixel unit ray directions ``(H, W, 3)`` for the decomp-BEV networks.

    When ``ray_dirs_path`` is a non-empty path to a ``.npy`` of shape ``(H, W, 3)``
    (e.g. the precomputed Waymo table from ``scripts/dump_waymo_ray_dirs.py``,
    built from the sensor's measured elevation/azimuth tables + per-row offset),
    it is loaded and shape-checked against ``resolution``. Otherwise the legacy
    Pandar128 geometry (uniform-linspace azimuth + the 128-beam inclination table)
    is rebuilt -- so every existing config stays bit-identical.

    Returns a float32 ``torch.Tensor`` of shape ``(H, W, 3)``.
    """
    H, W = int(resolution[0]), int(resolution[1])
    if ray_dirs_path:
        dirs = np.load(ray_dirs_path)
        if dirs.shape != (H, W, 3):
            raise ValueError(
                f"ray_dirs_path={ray_dirs_path} has shape {dirs.shape}, expected "
                f"({H}, {W}, 3) to match resolution {(H, W)}. Regenerate the asset "
                "with the matching resample_width (W) and lidar."
            )
        return torch.from_numpy(dirs).float()
    elevations = torch.tensor(load_pandar128_elevations(), dtype=torch.float32)
    return torch.from_numpy(range_map_to_ray_directions(W, elevations)).float()


def range_map_to_point_cloud(range_map):
    """
    Convert the range map to a point cloud.
    Input:
        range_map: numpy array of shape (H, W), unnormalized range values
    Output:
        point_cloud: numpy array of shape (N, 3)
    """
    elevation_angles = load_pandar128_elevations()
    assert len(elevation_angles) == range_map.shape[0]

    ray_directions = range_map_to_ray_directions(range_map.shape[1], elevation_angles)  # shape: (H, W, 3)
    
    points = range_map[..., np.newaxis] * ray_directions  # shape: (H, W, 3)
    points = points[range_map > 0]    
    return points

def apply_color_map_to_image(
    x,
    mask=None,
    color_map="Spectral",
):
    if color_map == "gray":
        mapped = np.repeat(x.detach().clip(min=0, max=1).cpu().numpy()[..., np.newaxis], 3, axis=-1)
    else:
        cmap = cm.get_cmap(color_map)
        # Convert to NumPy so that Matplotlib color maps can be used.
        mapped = cmap(x.detach().float().clip(min=0, max=1).cpu().numpy())[..., :3]
    image = torch.tensor(mapped, device=x.device, dtype=x.dtype)
    if mask is not None:
        image[mask] = torch.tensor([0.82, 0.82, 0.82], device=x.device, dtype=x.dtype)  # Set masked areas to light gray

    return rearrange(image, "... h w c -> ... c h w")


def apply_motion_compensation_impl(
    xyz: np.ndarray, T_sensor_end_sensor_start: np.ndarray, timestamps_startend_us: list, timestamp_us: np.ndarray
) -> np.ndarray:
    """
    undo motion-compensation to bring ray's into time-dependent sensor-frame
    Here the input lidar points are motion compensated to the beginning of the spin. 

    Args:
        xyz (np.array): points from the sensor space [n,3]
        T_sensor_end_sensor_start (np.array): relative pose from end-of-frame to start-of-frame in sensor space [4,4]
        timestamps_startend_us (list): contains [start timestamp, end timestamp]
        timestamp_us (np.array): recoding target per-point timestamps [n]
    Out:
        (np.array): points after undo motion-compensation[n,3]
    """
    
    xyz = make_sure_numpy(xyz)
    T_sensor_end_sensor_start = make_sure_numpy(T_sensor_end_sensor_start)
    timestamp_us = make_sure_numpy(timestamp_us)

    # )
    pose_interpolator = PoseInterpolator(
        np.stack([np.eye(4, dtype=np.float32),np.linalg.inv(T_sensor_end_sensor_start)]),
        timestamps_startend_us,
    )

    # Note: this interpolation will fail if the point's timestamps are outside of the frame's start/end times - issue dedicated error in that case
    assert (
        (timestamps_startend_us[0] <= timestamp_us).all() and (timestamp_us <= timestamps_startend_us[1]).all()
    ), f"Lidar point timestamps out of frame timestamp bounds - this is an inconsistency in the dataset's internal data and needs to be fixed at dataset creation time"
    T_sensor_end_sensor_pointtime = pose_interpolator.interpolate_to_timestamps(timestamp_us)

    xyz = transform_point_cloud(xyz[:, np.newaxis, :], T_sensor_end_sensor_pointtime).squeeze(1)

    return xyz



def draw_lidar_points(lidar_points, camera_model, camera_to_lidar):
    cmap = plt.colormaps['rainbow']
    lidar_points_z = lidar_points[:, 2]
    lidar_points_z = lidar_points_z.clip(-2, 4)
    lidar_points_z = (lidar_points_z + 2) / 6
    lidar_z_colors = cmap(lidar_points_z)[:, :3]
    lidar_z_colors = (lidar_z_colors * 255).astype(np.uint8)
    
    # cmap = plt.colormaps['rainbow']
    # depth = np.linalg.norm(lidar_points, axis=1)
    # depth = torch.from_numpy(depth).float().log()
    # near = depth[depth > 0].quantile(0.02)
    # far = depth[depth > 0].quantile(0.98)
    # depth = 1 - (depth - near) / (far - near)
    # depth = depth.clip(0, 1).numpy()
    # lidar_z_colors = cmap(depth)[:, :3]
    # lidar_z_colors = (lidar_z_colors * 255).astype(np.uint8)
    
    visual_lidar_points = camera_model.draw_points(
        camera_to_lidar,
        lidar_points,
        colors=lidar_z_colors,
        radius=2,
    )
    return visual_lidar_points



def project_lidar_to_rgb_impl(camera_intrinsics, camera_2_lidar, input_points_compensated, recon_points_compensated):
    # init camera model
    camera_model = FThetaCamera.from_numpy(camera_intrinsics.cpu().numpy(), device='cpu')
    
    # project points to rgb
    visual_lidar_input = draw_lidar_points(input_points_compensated, camera_model, camera_2_lidar).astype(np.float32) / 255.0 # [H, W, 3], 0-1
    visual_lidar_recon = draw_lidar_points(recon_points_compensated, camera_model, camera_2_lidar).astype(np.float32) / 255.0 # [H, W, 3], 0-1
    
    return visual_lidar_input, visual_lidar_recon


def colorcode_depth_maps(result, near=None, far=None, cmap="turbo"):
    """
    Input: B x H x W
    Output: B x 3 x H x W, normalized to [0, 1]
    """
    mask = result == 0
    n_frames = result.shape[0]
    if far is None:
        far = result[n_frames // 2].view(-1).quantile(0.99).log()
    if near is None:
        try:
            near = result[n_frames // 2][result[n_frames // 2] > 0].quantile(0.01).log()
        except:
            print("No valid depth values found.")
            near = torch.zeros_like(far)

    result = result.log()
    result = 1 - (result - near) / (far - near)
    return apply_color_map_to_image(result, mask, cmap)


def save_depth_maps_to_video(depth_maps, save_path, cmap="Spectral", fps=10):
    """
    depth_maps: B x H x W
    save_path: str
    fps: int
    """
    # Convert depth maps to uint8
    depth_maps = colorcode_depth_maps(depth_maps, cmap=cmap)  # B x 3 x H x W
    depth_maps = (depth_maps * 255 + 0.5).permute(0, 2, 3, 1).numpy().astype(np.uint8)  # B x H x W x 3
    media.write_video(save_path, depth_maps, fps=fps)

class RangeMapDownsampler:
    """
    A class to handle range map downsampling operations along with associated data.
    """

    def __init__(self, row_factor=1, col_factor=1, method="scatter_min"):
        """
        Initialize the downsampler with factors and method.
        
        Args:
            row_factor: Factor to downsample rows by (default: 1)
            col_factor: Factor to downsample columns by (default: 1) 
            method: Downsampling method - "scatter_min", "scatter_max", or "every_n" (default: "scatter_min")
        """
        self.row_factor = row_factor
        self.col_factor = col_factor
        self.method = method
        
        if method not in ["scatter_min", "scatter_max", "every_n"]:
            raise ValueError(f"Invalid method: {method}")

    def _downsample_vertical(self, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Downsample vertically using the configured method."""
        if self.row_factor == 1:
            return data, np.zeros_like(data)
            
        N, H, W = data.shape
        assert H % self.row_factor == 0, f"Height {H} must be divisible by row_factor {self.row_factor}"
        
        if self.method == "every_n":
            return data[:, ::self.row_factor], np.zeros((N, H // self.row_factor, W))
        
        # Reshape to compare adjacent rows
        groups = data.reshape(N, H // self.row_factor, self.row_factor, W)
        
        if self.method == "scatter_min":
            # Handle zeros by setting them to large values
            groups_clean = np.where(groups == 0, 1e3, groups)
            min_values = np.min(groups_clean, axis=2)
            min_indices = np.argmin(groups_clean, axis=2)
            # Restore zeros
            min_values = np.where(min_values == 1e3, 0, min_values)
        else:  # scatter_max
            min_values = np.max(groups, axis=2)
            min_indices = np.argmax(groups, axis=2)
            
        return min_values, min_indices

    def _downsample_horizontal(self, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Downsample horizontally using the configured method."""
        if self.col_factor == 1:
            return data, np.zeros_like(data)
            
        N, H, W = data.shape
        assert W % self.col_factor == 0, f"Width {W} must be divisible by col_factor {self.col_factor}"
        
        if self.method == "every_n":
            return data[:, :, ::self.col_factor], np.zeros((N, H, W // self.col_factor))
        
        # Reshape to compare adjacent columns
        groups = data.reshape(N, H, W // self.col_factor, self.col_factor)
        
        if self.method == "scatter_min":
            # Handle zeros by setting them to large values
            groups_clean = np.where(groups == 0, 1e3, groups)
            min_values = np.min(groups_clean, axis=3)
            min_indices = np.argmin(groups_clean, axis=3)
            # Restore zeros
            min_values = np.where(min_values == 1e3, 0, min_values)
        else:  # scatter_max
            min_values = np.max(groups, axis=3)
            min_indices = np.argmax(groups, axis=3)
            
        return min_values, min_indices

    def _apply_indices(self, data: np.ndarray, indices: np.ndarray, axis: int) -> np.ndarray:
        """Apply downsampling indices to data along specified axis."""
        if indices is None or np.all(indices == 0):
            return data
            
        N, H, W = data.shape[:3]
        scale_factor = H // indices.shape[1] if axis == 1 else W // indices.shape[2]
        
        if axis == 1:  # vertical
            reshaped = data.reshape(N, H // scale_factor, scale_factor, W, *data.shape[3:])
            batch_idx = np.arange(N)[:, None, None]
            height_idx = np.arange(H // scale_factor)[None, :, None]
            width_idx = np.arange(W)[None, None, :]
            return reshaped[batch_idx, height_idx, indices, width_idx]
        else:  # horizontal
            reshaped = data.reshape(N, H, W // scale_factor, scale_factor, *data.shape[3:])
            batch_idx = np.arange(N)[:, None, None]
            height_idx = np.arange(H)[None, :, None]
            width_idx = np.arange(W // scale_factor)[None, None, :]
            # Broadcast indices for advanced indexing
            batch_idx = np.broadcast_to(batch_idx, (N, H, W // scale_factor))
            height_idx = np.broadcast_to(height_idx, (N, H, W // scale_factor))
            width_idx = np.broadcast_to(width_idx, (N, H, W // scale_factor))
            return reshaped[batch_idx, height_idx, width_idx, indices]

    def downsample(self, range_map: np.ndarray, extra_maps: list = None) -> tuple:
        """
        Downsample the range map and any extra maps using the configured factors and method.
        
        Args:
            range_map: Array of shape (N, H, W)
            extra_maps: Optional list of arrays to downsample along with range_map
            
        Returns:
            Tuple of (downsampled_range_map, downsampled_extra_maps) or just downsampled_range_map
        """
        current_range = range_map.copy()
        
        # Downsample vertically first
        current_range, v_indices = self._downsample_vertical(current_range)
        if extra_maps is not None:
            extra_maps = [self._apply_indices(em, v_indices, axis=1) for em in extra_maps]
        
        # Then downsample horizontally
        current_range, h_indices = self._downsample_horizontal(current_range)
        if extra_maps is not None:
            extra_maps = [self._apply_indices(em, h_indices, axis=2) for em in extra_maps]
        
        if extra_maps is not None:
            return current_range, extra_maps
        return current_range

    def __call__(self, range_map: np.ndarray, extra_maps: list = None) -> tuple:
        """Convenience method to call downsample directly."""
        return self.downsample(range_map, extra_maps)

class SinusoidalEncoder(torch.nn.Module):
    """Sinusoidal Positional Encoder used in Nerf."""

    def __init__(
        self,
        n_input_dims: int = 3,
        min_deg: int = 0,
        max_deg: int = 10,
        enable_identity: bool = True,
    ):
        super().__init__()
        self.n_input_dims = n_input_dims
        self.min_deg = min_deg
        self.max_deg = max_deg
        self.enable_identity = enable_identity
        self.scales = torch.tensor([2**i for i in range(min_deg, max_deg + 1)], dtype=torch.get_default_dtype())

    @property
    def n_output_dims(self) -> int:
        return (
            int(self.enable_identity) + (self.max_deg - self.min_deg + 1) * 2
        ) * self.n_input_dims

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [..., n_input_dims]
        Returns:
            encoded: [..., n_output_dims]
        """
        if self.max_deg == self.min_deg:
            return x
        xb = torch.reshape(
            (x[..., None, :] * self.scales[:, None]),
            list(x.shape[:-1])
            + [(self.max_deg - self.min_deg + 1) * self.n_input_dims],
        )
        encoded = torch.sin(torch.cat([xb, xb + 0.5 * torch.pi], dim=-1))
        if self.enable_identity:
            encoded = torch.cat([x] + [encoded], dim=-1)
        return encoded