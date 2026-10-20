# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The released models' sensor: NVIDIA PAI's Hesai Pandar128 on a 128x1024 grid.

Rows are the Pandar128 lasers (assets/lidar/lidar_model.yaml), columns the
azimuth from +180 deg (behind) clockwise seen from above; directions are in
the sensor's x-forward, y-left, z-up frame, as scripts/data/prepare_pai_example.py
builds them for the quickstart samples.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml

HEIGHT, WIDTH = 128, 1024
MIN_RANGE_M, MAX_RANGE_M = 2.5, 100.0
FPS = 10.0
LIDAR_MODEL = Path(__file__).resolve().parents[1] / "assets" / "lidar" / "lidar_model.yaml"


@lru_cache(maxsize=1)
def elevations_deg() -> np.ndarray:
    """(128,) the lasers' elevations, row order."""
    with LIDAR_MODEL.open() as handle:
        config = yaml.safe_load(handle)
    values = np.asarray(config["lidar_models"]["spinning"]["HESAI-Pandar128"]["inclinations_deg"], np.float64)
    if values.shape != (HEIGHT,):
        raise ValueError(f"{LIDAR_MODEL}: expected {HEIGHT} Pandar128 inclinations")
    return values


def column_azimuths_rad() -> np.ndarray:
    """(1024,) each column's azimuth, left positive: pi down to -pi."""
    return np.linspace(np.pi, -np.pi, WIDTH, endpoint=False)


def ray_directions() -> np.ndarray:
    """(128, 1024, 3) float32 unit rays, x forward / y left / z up."""
    phi, theta = np.meshgrid(np.radians(elevations_deg()), column_azimuths_rad(), indexing="ij")
    return np.stack([np.cos(phi) * np.cos(theta), np.cos(phi) * np.sin(theta), np.sin(phi)],
                    axis=-1).astype(np.float32)


def grid_cells(directions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The grid cell each ray (``(N, 3)``, x forward / y left / z up) reads.

    Returns rows, columns and whether the ray is inside the sensor's vertical
    field of view (half a laser spacing beyond the outermost lasers)."""
    d = np.asarray(directions, np.float64)
    elevation = np.degrees(np.arctan2(d[:, 2], np.hypot(d[:, 0], d[:, 1])))
    azimuth = np.arctan2(d[:, 1], d[:, 0])
    lasers = elevations_deg()
    order = np.argsort(lasers)
    sorted_lasers = lasers[order]
    position = np.clip(np.searchsorted(sorted_lasers, elevation), 1, len(sorted_lasers) - 1)
    lower, upper = sorted_lasers[position - 1], sorted_lasers[position]
    nearest = np.where(elevation - lower < upper - elevation, position - 1, position)
    rows = order[nearest]
    low_margin = (sorted_lasers[1] - sorted_lasers[0]) / 2
    high_margin = (sorted_lasers[-1] - sorted_lasers[-2]) / 2
    inside = (elevation >= sorted_lasers[0] - low_margin) & (elevation <= sorted_lasers[-1] + high_margin)
    columns = np.rint((np.pi - azimuth) / (2 * np.pi) * WIDTH).astype(np.int64) % WIDTH
    return rows, columns, inside
