# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared sample I/O and preview rendering for tokenizer and diffusion inference."""

from __future__ import annotations

import json
import shutil
import subprocess
from itertools import chain
from pathlib import Path
from tempfile import TemporaryDirectory, TemporaryFile

import cv2
import numpy as np
from matplotlib import colormaps

from utils.visualize.point_cloud import CAMERA_VIEWS, VIZ_KWARGS, visualize_point_cloud


MIN_RANGE_M = 2.5
MAX_RANGE_M = 100.0
SENSOR_GRIDS = {
    (128, 1024): "NVIDIA PAI",
    (128, 1200): "NVIDIA PAI",
    (64, 1024): "PandaSet",
    (32, 1024): "nuScenes",
}
DISPLAY_YAW_DEG = {
    "nvidia_pai": 0.0,
    "pandar": 0.0,
    "nuscenes": -90.0,
    "pandaset": 180.0,
    "kitti": 0.0,
}


def load_sample(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path) as payload:
        if "range_m" not in payload or "ray_directions" not in payload:
            raise ValueError(
                f"{path} must contain range_m and ray_directions arrays"
            )
        ranges = np.asarray(payload["range_m"], dtype=np.float32)
        stored_rays = np.asarray(payload["ray_directions"], dtype=np.float32)
        valid = np.asarray(
            payload.get("valid_mask", ranges > 0), dtype=np.bool_
        )

    if ranges.ndim == 2:
        ranges = ranges[None]
    if ranges.ndim != 3:
        raise ValueError(f"range_m must have shape (T,H,W), got {ranges.shape}")
    if valid.ndim == 2:
        valid = valid[None]
    if valid.shape != ranges.shape:
        raise ValueError(
            f"valid_mask shape {valid.shape} does not match range_m {ranges.shape}"
        )
    if ranges.shape[0] < 1:
        raise ValueError(f"Empty sequence in {path}")
    grid = tuple(ranges.shape[-2:])
    if grid not in SENSOR_GRIDS:
        expected = ", ".join(f"{h}x{w}" for h, w in SENSOR_GRIDS)
        raise ValueError(f"Unsupported input grid {grid}; expected one of {expected}")

    if stored_rays.ndim == 3:
        rays = np.broadcast_to(stored_rays[None], ranges.shape + (3,))
    elif stored_rays.ndim == 4:
        rays = stored_rays
    else:
        raise ValueError(
            "ray_directions must have shape (H,W,3) or (T,H,W,3), "
            f"got {stored_rays.shape}"
        )
    if rays.shape != ranges.shape + (3,):
        raise ValueError(
            f"ray_directions shape {rays.shape} does not match {ranges.shape}"
        )
    if not np.isfinite(rays).all():
        raise ValueError(f"ray_directions contains non-finite values in {path}")
    norms = np.linalg.norm(rays, axis=-1, keepdims=True)
    if np.any(norms <= 1e-8):
        raise ValueError(f"ray_directions contains zero-length rays in {path}")
    rays = rays / norms
    valid &= (
        np.isfinite(ranges)
        & (ranges >= MIN_RANGE_M)
        & (ranges <= MAX_RANGE_M)
    )
    ranges = np.where(valid, ranges, 0.0).astype(np.float32)
    rays = rays.astype(np.float32)
    output_rays = rays[0] if stored_rays.ndim == 3 else rays
    return ranges, valid, rays, output_rays


def points_from_ranges(
    ranges: np.ndarray, valid: np.ndarray, rays: np.ndarray
) -> np.ndarray:
    return (rays[valid] * ranges[valid][:, None]).astype(np.float32)


def _height_colors(points: np.ndarray) -> np.ndarray:
    if not len(points):
        return np.empty((0, 3), dtype=np.uint8)
    scale = np.clip((points[:, 2] + 3.0) / 8.0, 0.0, 1.0)
    return (colormaps["viridis"](scale)[:, :3] * 255).astype(np.uint8)


def write_ply(path: Path, points: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    colors = _height_colors(points)
    vertices = np.empty(
        len(points),
        dtype=[
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    for axis, name in enumerate(("x", "y", "z")):
        vertices[name] = points[:, axis]
    for channel, name in enumerate(("red", "green", "blue")):
        vertices[name] = colors[:, channel]
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(points)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    with path.open("wb") as handle:
        handle.write(header)
        vertices.tofile(handle)


def render_points(points: np.ndarray, yaw_deg: float = 0.0) -> np.ndarray:
    colors = _height_colors(points).astype(np.float32) / 255.0
    angle = np.deg2rad(yaw_deg)
    c, s = np.cos(angle), np.sin(angle)
    rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], np.float32)
    # Display-only rotation; the exported PLYs remain in the original sensor frame.
    displayed_points = points @ rotation.T
    camera = CAMERA_VIEWS["paper_view_down"]
    center = camera["center"]
    eye = {axis: center[axis] + 0.6 * (camera["eye"][axis] - center[axis])
           for axis in ("x", "y", "z")}
    kwargs = dict(VIZ_KWARGS)
    kwargs.update(
        camera_position={"eye": eye, "center": dict(center)},
        bgcolor=(1.0, 1.0, 1.0), point_size=0.8,
        antialias_point_size=True, principal_y_offset=0.08, depth_sort=True,
    )
    rgb = visualize_point_cloud(displayed_points, colors, **kwargs)
    return np.ascontiguousarray(rgb[..., ::-1])


def label_image(image: np.ndarray, text: str) -> np.ndarray:
    output = image.copy()
    cv2.putText(
        output,
        text,
        (20, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (30, 30, 30),
        2,
        cv2.LINE_AA,
    )
    return output


def write_video(path: Path, frames, fps: float) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("Install system ffmpeg to write H.264 video previews")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("Video fps must be finite and positive")
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = iter(frames)
    first = next(frames, None)
    if first is None:
        raise ValueError(f"No video frames for {path}")
    height, width = first.shape[:2]
    # Match the quickstart grid's browser-compatible encoding. Publish only
    # after FFmpeg has finalized the container, preserving any previous video.
    with TemporaryDirectory(dir=path.parent, prefix=f".{path.stem}-") as temporary:
        encoded = Path(temporary) / "video.mp4"
        command = [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s:v", f"{width}x{height}", "-r", str(fps), "-i", "pipe:0",
            "-an", "-c:v", "libx264", "-threads", "4",
            "-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(encoded),
        ]
        with TemporaryFile() as errors:
            process = subprocess.Popen(
                command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                stderr=errors,
            )
            try:
                with process.stdin:
                    for frame in chain((first,), frames):
                        if frame.shape != (height, width, 3) or frame.dtype != np.uint8:
                            raise ValueError("Video frames must be equally sized uint8 BGR images")
                        process.stdin.write(frame.tobytes())
                returncode = process.wait()
            except BrokenPipeError:
                returncode = process.wait()
                errors.seek(0)
                raise RuntimeError(
                    f"FFmpeg failed for {path}: {errors.read().decode(errors='replace')}"
                ) from None
            except BaseException:
                process.kill()
                process.wait()
                raise
            if returncode:
                errors.seek(0)
                raise RuntimeError(
                    f"FFmpeg failed for {path}: {errors.read().decode(errors='replace')}"
                )
        encoded.replace(path)


def display_yaw(path: Path, dataset: str) -> float:
    sidecar = path.with_suffix(".json")
    display = json.loads(sidecar.read_text()) if sidecar.is_file() else {}
    yaw_deg = float(display.get("visualization_yaw_deg", DISPLAY_YAW_DEG.get(dataset, 0.0)))
    if not np.isfinite(yaw_deg):
        raise ValueError(f"{path}: non-finite display yaw")
    return yaw_deg


def sample_paths(path: Path) -> list[Path]:
    if path.is_file():
        if path.suffix.lower() != ".npz":
            raise ValueError(f"Expected an NPZ sample, got {path}")
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(path)
    samples = sorted(path.rglob("*.npz"))
    if not samples:
        raise FileNotFoundError(f"No NPZ samples found under {path}")
    return samples


def sample_output(sample: Path, sample_root: Path, output_root: Path) -> Path:
    if sample_root.is_file():
        return output_root
    return output_root / sample.relative_to(sample_root).with_suffix("")
