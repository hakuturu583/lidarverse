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

import io
import os
import torch
import numpy as np
from PIL import Image
# open3d is only needed by the PLY-save / statistical-filter helpers below, NOT by
# the pure-numpy ``visualize_point_cloud`` proj renderer. Importing it eagerly pulls
# ``open3d.ml`` -> sklearn -> pandas, which crashes on numpy-ABI-mismatched containers
# ("numpy.dtype size changed"). Guard it so the proj render path (infer_diffusion /
# infer_forecast) works everywhere; the o3d-using helpers fail only if actually called.
try:
    import open3d as o3d
except Exception:  # noqa: BLE001
    o3d = None

# Configure environment for better Kaleido performance
os.environ["KALEIDO_SCOPE_TIMEOUT"] = "60"  # Increase timeout to 60 seconds

# eye: camera position
# center: view direction
CAMERA_VIEWS = {
    "front_view_1": {
        "eye": {"x": -0.3, "y": 0, "z": 0.2},
        "center": {"x": 0.1, "y": 0, "z": 0},
    },
    "paper_view_down": {
        "eye": {"x": -0.3, "y": 0, "z": 0.26},
        "center": {"x": 0.1, "y": 0, "z": -0.03},
    },
    "paper_view_down_strong": {
        "eye": {"x": -0.3, "y": 0, "z": 0.3},
        "center": {"x": 0.1, "y": 0, "z": -0.08},
    },
    "paper_side_view": {
        "eye": {"x": 0.0, "y": -0.12, "z": 0.04},
        "center": {"x": 0.0, "y": 0.0, "z": 0.0},
    },
    "front_view_2": {
        "eye": {"x": -0.1, "y": 0, "z": 0.05},
        "center": {"x": 0.1, "y": 0, "z": 0},
    },
    "front_view_3": {
        "eye": {"x": -0.05, "y": 0, "z": 0.05},
        "center": {"x": 0.1, "y": 0, "z": 0},
    },
    "front_view_4": {
        "eye": {"x": -0.01, "y": 0, "z": 0.01},
        "center": {"x": 0.1, "y": 0, "z": 0},
    },
    "front_view_5": {
        "eye": {"x": -0.12, "y": 0, "z": 0.05},
        "center": {"x": 0.1, "y": 0, "z": 0},
    },
    "front_view_6": {
        "eye": {"x": -0.075, "y": 0, "z": 0.05},
        "center": {"x": 0.075, "y": 0, "z": 0.0},
    },
    "top_down_view_1": {
        "eye": {"x": 0, "y": -0.05, "z": 0.5},
        "center": {"x": 0, "y": -0.05, "z": 0},
    },
}


VIZ_KWARGS = {
    "point_size": 0.3,
    "camera_position": CAMERA_VIEWS["front_view_1"],
    "width": 1280,
    "height": 720,
    "bgcolor": [0.0, 0.0, 0.0],
    "range": 200,
}


def visualize_point_cloud(
    point_cloud,
    colors,
    point_size,
    camera_position,
    width,
    height,
    range=80,
    opacity=1.0,
    bgcolor=(0, 0, 0),
    fov_deg=60.0,
    antialias_point_size=False,
    principal_y_offset=0.0,
    depth_sort=False,
    depth_sort_bin=0.0,
    point_opacity=None,
):
    """
    Ultra-fast 2D projection visualization (no rendering overhead).
    """
    # Plotly camera system: eye, center, up are in normalized domain coordinates
    # Scale eye and center to match the point cloud's coordinate system
    point_cloud_center = np.array([0, 0, 0])  # Lidar origin is always at (0,0,0)
    scale = range
    
    # Scale eye and center from normalized coordinates to match point cloud scale
    # eye: camera position (scaled to point cloud coordinates)
    # center: point the camera looks at (scaled to point cloud coordinates)
    # up: up direction on the page (normalized vector, default [0, 0, 1])
    eye = np.array([camera_position["eye"]["x"], 
                    camera_position["eye"]["y"], 
                    camera_position["eye"]["z"]]) * scale
    center = np.array([camera_position["center"]["x"], 
                       camera_position["center"]["y"], 
                       camera_position["center"]["z"]]) * scale
    
    # Get up vector from camera_position if provided, otherwise use default [0, 0, 1]
    # Up is a direction vector (normalized), not scaled by range
    if "up" in camera_position:
        up_world = np.array([camera_position["up"]["x"], 
                            camera_position["up"]["y"], 
                            camera_position["up"]["z"]])
    else:
        up_world = np.array([0, 0, 1])  # Plotly default
    
    # Calculate view direction (from eye to center)
    forward = center - eye
    forward = forward / np.linalg.norm(forward)
    
    # Calculate right vector (perpendicular to forward and up_world)
    right = np.cross(forward, up_world)
    right_norm = np.linalg.norm(right)
    
    # Handle edge case when forward is parallel to up_world
    if right_norm < 1e-6:
        # Use a different reference vector
        up_world = np.array([0, 1, 0])
        right = np.cross(forward, up_world)
        right_norm = np.linalg.norm(right)
    
    right = right / right_norm
    
    # Calculate camera up vector (perpendicular to forward and right)
    # This ensures up is in the plane defined by forward and up_world
    up = np.cross(right, forward)
    up = up / np.linalg.norm(up)
    
    # Project points to 2D using a perspective projection (Plotly uses perspective)
    # Camera coordinate system: right (x), up (y), forward (z)
    points_relative = point_cloud - eye  # Points relative to camera position
    x_cam = np.dot(points_relative, right)
    y_cam = np.dot(points_relative, up)
    z_cam = np.dot(points_relative, forward)

    if point_opacity is None:
        point_opacity = np.ones(len(point_cloud), dtype=np.float32)
    else:
        point_opacity = np.broadcast_to(
            np.asarray(point_opacity, dtype=np.float32).reshape(-1),
            (len(point_cloud),),
        )
    
    # Keep only points in front of the camera
    eps = 1e-6
    depth_mask = z_cam > eps
    x_cam = x_cam[depth_mask]
    y_cam = y_cam[depth_mask]
    z_cam = z_cam[depth_mask]
    colors = colors[depth_mask]
    point_opacity = point_opacity[depth_mask]
    
    # Perspective projection
    # Use fov_deg as vertical field of view, zoom scales the FOV
    fov_rad = np.deg2rad(fov_deg)
    tan_half_fov = np.tan(fov_rad / 2.0)
    aspect = width / height
    
    x_norm = (x_cam / (z_cam * tan_half_fov * aspect)) * 0.5 + 0.5
    y_norm = (
        (y_cam / (z_cam * tan_half_fov)) * 0.5
        + 0.5
        + float(principal_y_offset)
    )
    
    # Create image
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = np.array(bgcolor) * 255
    
    # Convert to pixel coordinates
    x_pix = (x_norm * width).astype(int)
    y_pix = ((1 - y_norm) * height).astype(int)  # Flip Y
    
    # Filter valid pixels
    valid = (x_pix >= 0) & (x_pix < width) & (y_pix >= 0) & (y_pix < height)
    x_pix = x_pix[valid]
    y_pix = y_pix[valid]
    # reshape(-1, 3) keeps an empty/1-D color array broadcastable into the (0, 3)
    # pixel slice (e.g. when every point is filtered out -> nothing to draw).
    colors_valid = (colors[valid] * 255).astype(np.uint8).reshape(-1, 3)
    opacity_valid = np.clip(point_opacity[valid], 0.0, 1.0)
    if depth_sort:
        # Painter's algorithm: far points first, near points last. Stable sorting
        # preserves caller ordering for equal-depth temporal composites.
        sort_depth = z_cam[valid]
        if depth_sort_bin > 0.0:
            sort_depth = np.floor(sort_depth / float(depth_sort_bin))
        order = np.argsort(-sort_depth, kind="stable")
        x_pix = x_pix[order]
        y_pix = y_pix[order]
        colors_valid = colors_valid[order]
        opacity_valid = opacity_valid[order]

    # Draw points. The legacy path quantizes the requested radius to an integer.
    # The opt-in antialiased path blends the boundary pixels by their fractional
    # disk coverage, then restores every projected center at full opacity.
    if antialias_point_size:
        radius = max(0.0, float(point_size))
        extent = int(np.ceil(radius + 0.5))
        for dy in np.arange(-extent, extent + 1):
            for dx in np.arange(-extent, extent + 1):
                if dx == 0 and dy == 0:
                    continue
                coverage = float(np.clip(radius + 0.5 - np.hypot(dx, dy), 0.0, 1.0))
                if coverage == 0.0:
                    continue
                shifted_x = x_pix + dx
                shifted_y = y_pix + dy
                shifted_valid = (
                    (shifted_x >= 0)
                    & (shifted_x < width)
                    & (shifted_y >= 0)
                    & (shifted_y < height)
                )
                target_x = shifted_x[shifted_valid]
                target_y = shifted_y[shifted_valid]
                foreground = colors_valid[shifted_valid].astype(np.float32)
                background = img[target_y, target_x].astype(np.float32)
                alpha = coverage * opacity_valid[shifted_valid, None]
                img[target_y, target_x] = np.rint(
                    alpha * foreground + (1.0 - alpha) * background
                ).astype(np.uint8)
        foreground = colors_valid.astype(np.float32)
        background = img[y_pix, x_pix].astype(np.float32)
        alpha = opacity_valid[:, None]
        img[y_pix, x_pix] = np.rint(
            alpha * foreground + (1.0 - alpha) * background
        ).astype(np.uint8)
    else:
        radius_px = max(0, int(round(float(point_size))))
        if radius_px == 0:
            foreground = colors_valid.astype(np.float32)
            background = img[y_pix, x_pix].astype(np.float32)
            alpha = opacity_valid[:, None]
            img[y_pix, x_pix] = np.rint(
                alpha * foreground + (1.0 - alpha) * background
            ).astype(np.uint8)
        else:
            for dy in np.arange(-radius_px, radius_px + 1):
                for dx in np.arange(-radius_px, radius_px + 1):
                    if dx * dx + dy * dy > radius_px * radius_px:
                        continue
                    shifted_x = x_pix + dx
                    shifted_y = y_pix + dy
                    shifted_valid = (
                        (shifted_x >= 0)
                        & (shifted_x < width)
                        & (shifted_y >= 0)
                        & (shifted_y < height)
                    )
                    target = (
                        shifted_y[shifted_valid],
                        shifted_x[shifted_valid],
                    )
                    foreground = colors_valid[shifted_valid].astype(np.float32)
                    background = img[target].astype(np.float32)
                    alpha = opacity_valid[shifted_valid, None]
                    img[target] = np.rint(
                        alpha * foreground + (1.0 - alpha) * background
                    ).astype(np.uint8)

    return img


def vis_point_cloud_to_numpy(point_cloud, colors, point_size=None):
    kwargs = dict(VIZ_KWARGS)
    if point_size is not None:
        kwargs["point_size"] = point_size
    img = visualize_point_cloud(point_cloud, colors, **kwargs)
    return img


def vis_point_cloud(point_cloud, colors, save_path):
    img = visualize_point_cloud(point_cloud, colors, **VIZ_KWARGS)
    img = Image.fromarray(img)
    img.save(save_path)

def o3d_statistical_filter(pts, nb_neighbors=20, std_ratio=1.0, return_ind=False):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)

    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)

    inlier_cloud = pcd.select_by_index(ind)

    # get the numpy point cloud
    inlier_pts = np.asarray(inlier_cloud.points)
    if return_ind:
        # ``ind`` maps inlier rows back to the input order so callers can keep
        # per-point attributes (e.g. intensity) aligned with the kept points.
        return inlier_pts, np.asarray(ind, dtype=np.int64)
    return inlier_pts



def save_range_map_to_ply(
    range_map_input, save_path
):
    """
    Save the range map to a point cloud
    Input:
        range_map_input: numpy array of shape (H, W), unnormalized range values
    """

    from utils.lidar_rangemap import range_map_to_point_cloud

    # get the ray directions and ray directions
    input_points = range_map_to_point_cloud(range_map_input)

    n_pts = input_points.shape[0]
    input_colors = np.array([[1, 0.706, 0]] * n_pts, dtype=np.float32)


    pcd_ref = o3d.geometry.PointCloud()
    pcd_ref.points = o3d.utility.Vector3dVector(input_points)
    pcd_ref.colors = o3d.utility.Vector3dVector(input_colors)
    
    o3d.io.write_point_cloud(save_path, pcd_ref)
    
    
def render_range_map_to_point_cloud(
    range_map_input, 
    range_map_recon, 
    filter_outlier=False,
    save_path=None,
    point_size=None,
    ray_directions=None,
    filter_nb_neighbors=20,
    filter_std_ratio=1.0,
):
    """
    Render the range map to a point cloud.
    Input:
        range_map_input: numpy array of shape (N, H, W), unnormalized range values
        range_map_recon: numpy array of shape (N, H, W), unnormalized range values
        ray_directions: optional (H, W, 3) unit ray-direction grid. When given it
            overrides the default Pandar128 geometry (e.g. the Waymo sensor's
            measured elevation/azimuth tables); must match the range-map H, W.
    Output:
        point_cloud: numpy array of shape (N, 3, H, W)
    """
    # unnormalize the range map
    valid_mask_input = range_map_input > 0
    valid_mask_recon = range_map_recon > 0

    # get the ray directions (default Pandar128 unless an override is supplied)
    if ray_directions is None:
        from utils.lidar_rangemap import (
            load_pandar128_elevations,
            range_map_to_ray_directions,
        )

        elevation_angles = load_pandar128_elevations()
        assert len(elevation_angles) == range_map_input.shape[1]
        ray_directions = range_map_to_ray_directions(range_map_input.shape[2], elevation_angles)  # shape: (H, W, 3)
    else:
        ray_directions = np.asarray(ray_directions, dtype=np.float32)
        assert ray_directions.shape[:2] == range_map_input.shape[1:3], (
            f"ray_directions {ray_directions.shape} does not match range map "
            f"{range_map_input.shape}"
        )
    ray_origins = np.zeros_like(ray_directions)  # shape: (H, W, 3)

    # get the point cloud
    bs = range_map_input.shape[0]

    vis_list = []
    for batch_idx in range(bs):
        c_range_map_input = range_map_input[batch_idx]  # shape: (H, W)
        c_range_map_recon = range_map_recon[batch_idx]  # shape: (H, W)
        c_ray_directions = ray_directions  # shape: (H, W, 3)
        c_ray_origins = ray_origins  # shape: (H, W, 3)

        c_valid_mask_input = valid_mask_input[batch_idx]
        c_valid_mask_recon = valid_mask_recon[batch_idx]

        input_points = (
            c_ray_origins[c_valid_mask_input]
            + c_ray_directions[c_valid_mask_input] * c_range_map_input[c_valid_mask_input][:, np.newaxis]
        )  # shape: (N, 3)
        recon_points = (
            c_ray_origins[c_valid_mask_recon]
            + c_ray_directions[c_valid_mask_recon] * c_range_map_recon[c_valid_mask_recon][:, np.newaxis]
        )  # shape: (N, 3)

        if filter_outlier:
            recon_points = o3d_statistical_filter(
                recon_points,
                nb_neighbors=filter_nb_neighbors,
                std_ratio=filter_std_ratio,
            )

        n_pts = input_points.shape[0]
        input_colors = np.array([[1, 0.706, 0]] * n_pts, dtype=np.float32)
        n_pts = recon_points.shape[0]
        recon_colors = np.array([[0, 0.651, 0.929]] * n_pts, dtype=np.float32)

        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)
            o3d.io.write_point_cloud(
                f"{save_path}/input_pcd_{batch_idx}.ply", 
                o3d.geometry.PointCloud(
                    points=o3d.utility.Vector3dVector(input_points)
                )
            )
            o3d.io.write_point_cloud(
                f"{save_path}/recon_pcd_{batch_idx}.ply", 
                o3d.geometry.PointCloud(
                    points=o3d.utility.Vector3dVector(recon_points)
                )
            )

        input_img = vis_point_cloud_to_numpy(input_points, input_colors, point_size=point_size)  # shape: (H, W, 3)
        recon_img = vis_point_cloud_to_numpy(recon_points, recon_colors, point_size=point_size)  # shape: (H, W, 3)

        recon_images = np.concatenate([input_img, recon_img], axis=1)  # shape: (H, 2W, 3)
        vis_list.append(recon_images)

    vis_list = np.array(vis_list)  # shape: (bs, H, 2W, 3)

    return vis_list

def render_range_map_to_point_cloud_with_intensity(
    range_map_input,
    range_map_recon,
    intensity_input,
    intensity_recon,
    filter_outlier=False,
    ray_directions=None,
    filter_nb_neighbors=20,
    filter_std_ratio=1.0,
):
    """
    Render point clouds colored by intensity (viridis colormap).
    Input:
        range_map_input/recon: numpy array of shape (N, H, W), unnormalized range
        intensity_input/recon: numpy array of shape (N, H, W), values in [0, 1]
        ray_directions: optional (H, W, 3) unit ray-direction grid. When given it
            overrides the default Pandar128 geometry (e.g. the Waymo sensor's
            measured elevation/azimuth tables); must match the range-map H, W.
    Output:
        vis_list: numpy array of shape (N, H, 2W, 3)
    """
    from matplotlib import cm
    cmap = cm.get_cmap("viridis")

    valid_mask_input = range_map_input > 0
    valid_mask_recon = range_map_recon > 0

    if ray_directions is None:
        from utils.lidar_rangemap import (
            load_pandar128_elevations,
            range_map_to_ray_directions,
        )

        elevation_angles = load_pandar128_elevations()
        assert len(elevation_angles) == range_map_input.shape[1]
        ray_directions = range_map_to_ray_directions(range_map_input.shape[2], elevation_angles)
    else:
        ray_directions = np.asarray(ray_directions, dtype=np.float32)
        assert ray_directions.shape[:2] == range_map_input.shape[1:3], (
            f"ray_directions {ray_directions.shape} does not match range map "
            f"{range_map_input.shape}"
        )

    bs = range_map_input.shape[0]
    vis_list = []
    for b in range(bs):
        mask_in = valid_mask_input[b]
        mask_re = valid_mask_recon[b]

        pts_in = ray_directions[mask_in] * range_map_input[b][mask_in][:, np.newaxis]
        pts_re = ray_directions[mask_re] * range_map_recon[b][mask_re][:, np.newaxis]

        if filter_outlier:
            pts_re = o3d_statistical_filter(
                pts_re,
                nb_neighbors=filter_nb_neighbors,
                std_ratio=filter_std_ratio,
            )

        colors_in = cmap(np.clip(intensity_input[b][mask_in], 0, 1))[:, :3].astype(np.float32)
        colors_re = cmap(np.clip(intensity_recon[b][mask_re], 0, 1))[:, :3].astype(np.float32)

        img_in = vis_point_cloud_to_numpy(pts_in, colors_in)
        img_re = vis_point_cloud_to_numpy(pts_re, colors_re)
        vis_list.append(np.concatenate([img_in, img_re], axis=1))

    return np.array(vis_list)


def range_map_to_structured_point_cloud(range_map: torch.Tensor):
    """
    Convert the range map to a structured point cloud.
    Input:
        range_map: torch tensor of shape (N, H, W), unnormalized range values
    Output:
        point_cloud: torch tensor of shape (N, 3, H, W)
    """
    from utils.lidar_rangemap import (
        load_pandar128_elevations,
        range_map_to_ray_directions,
    )

    elevation_angles = load_pandar128_elevations()
    assert len(elevation_angles) == range_map.shape[1]

    valid_mask = range_map > 0

    ray_directions = range_map_to_ray_directions(range_map.shape[2], elevation_angles)  # shape: (H, W, 3)
    ray_origins = np.zeros_like(ray_directions)  # shape: (H, W, 3)

    points = range_map[..., np.newaxis] * ray_directions  # shape: (H, W, 3)
    points = points[range_map > 0]    
    return points
