# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The raw vector map LidarVerse conditions on, from CARLA's OpenDRIVE.

The released models were conditioned on RDS-HQ vector layers (lanes, lane
lines, road boundaries, crosswalks, wait lines, traffic lights and signs; see
utils/visualize/hdmap.py). roadgen reads the map's OpenDRIVE (what
``carla.Map.to_opendrive()`` returns, which carla_canvas sends with
``LoadWorld``) and writes those layers as a ClipGT clip; this module turns
them into polylines in OpenDRIVE's frame, which is carla_canvas's
``right_handed`` frame (CARLA's with y negated).
"""

from __future__ import annotations

import logging
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from diffusion.inference.raw_hdmap import box_wireframe_points, sample_polylines

log = logging.getLogger("lidarverse_canvas")

# ClipGT layer -> how its rows become polylines.
POLYLINE_LAYERS = {
    "lane": ("left_rail", "right_rail"),
    "lane_line": ("line_rail",),
    "road_boundary": ("location",),
    "crosswalk": ("location",),
    "wait_line": ("location",),
}
CLOSED_LAYERS = {"crosswalk"}
CUBOID_LAYERS = ("traffic_light", "traffic_sign")


@dataclass
class VectorMap:
    """Map polylines (world, right-handed) and their densified points."""

    vertices: np.ndarray  # (N, 3) float32
    offsets: np.ndarray  # (P + 1,) int64: polyline i is vertices[offsets[i]:offsets[i+1]]
    points: np.ndarray  # (M, 3) float32: edges sampled every `spacing` meters
    spacing: float

    @staticmethod
    def from_polylines(polylines: list[np.ndarray], spacing: float = 0.1) -> "VectorMap":
        polylines = [np.asarray(p, np.float32) for p in polylines if len(p) >= 2]
        offsets = np.zeros(len(polylines) + 1, np.int64)
        offsets[1:] = np.cumsum([len(p) for p in polylines])
        vertices = np.concatenate(polylines) if polylines else np.empty((0, 3), np.float32)
        return VectorMap(vertices, offsets, sample_polylines(vertices, offsets, spacing), spacing)

    @staticmethod
    def from_opendrive(xodr: str, spacing: float = 0.1, sampling: float = 0.5) -> "VectorMap":
        """Read an OpenDRIVE document (its text) through roadgen's ClipGT export."""
        import roadgen

        with tempfile.TemporaryDirectory(prefix="lidarverse_canvas_") as tmp:
            path = Path(tmp) / "map.xodr"
            path.write_text(xodr)
            road_map = roadgen.read_opendrive(str(path), sampling=sampling)
            for note in road_map.read_warnings():
                log.info("OpenDRIVE: %s", note)
            road_map.export_clipgt(str(Path(tmp) / "clipgt"), trace=False)
            polylines = read_clipgt(Path(tmp) / "clipgt")
        vector_map = VectorMap.from_polylines(polylines, spacing)
        vector_map.check_against_header(xodr)
        return vector_map

    @staticmethod
    def from_rdshq(root: str | Path, clip: str, spacing: float = 0.1) -> "VectorMap":
        """An RDS-HQ clip's 3d_* layers (e.g. roadgen's export_rdshq, or a dataset's)."""
        from utils.visualize.hdmap import load_vectors

        vectors = load_vectors(Path(root), clip)
        offsets = vectors["map_offsets"]
        polylines = [vectors["map_vertices"][a:b] for a, b in zip(offsets[:-1], offsets[1:])]
        return VectorMap.from_polylines(polylines, spacing)

    def near(self, centers: np.ndarray, radius: float) -> np.ndarray:
        """Points within ``radius`` (horizontally) of any of ``centers`` (K, 3)."""
        if not len(self.points):
            return self.points
        keep = np.zeros(len(self.points), bool)
        xy = self.points[:, :2]
        for center in np.asarray(centers, np.float32).reshape(-1, 3):
            keep |= ((xy - center[:2]) ** 2).sum(1) <= radius * radius
        return self.points[keep]

    def check_against_header(self, xodr: str) -> None:
        """Warn when the polylines do not overlap the bounds the file declares."""
        try:
            header = ET.fromstring(xodr).find("header")
            bounds = {k: float(header.attrib[k]) for k in ("north", "south", "east", "west")}
        except (ET.ParseError, AttributeError, KeyError, ValueError):
            return
        if not len(self.vertices) or bounds["north"] <= bounds["south"] or bounds["east"] <= bounds["west"]:
            return
        lo, hi = self.vertices[:, :2].min(0), self.vertices[:, :2].max(0)
        if hi[0] < bounds["west"] or lo[0] > bounds["east"] or hi[1] < bounds["south"] or lo[1] > bounds["north"]:
            log.warning("the vector map (x %.0f..%.0f, y %.0f..%.0f) lies outside the OpenDRIVE header's bounds; "
                        "check renderer.options.map_offset", lo[0], hi[0], lo[1], hi[1])


def _xyz(items) -> np.ndarray:
    return np.asarray([(p["x"], p["y"], p["z"]) for p in items], np.float32).reshape(-1, 3)


def _cuboid_pose(row: dict) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    q = row["orientation"]
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_quat([q["x"], q["y"], q["z"], q["w"]]).as_matrix()
    pose[:3, 3] = [row["center"]["x"], row["center"]["y"], row["center"]["z"]]
    return pose


def read_clipgt(directory: Path) -> list[np.ndarray]:
    """Polylines of a ClipGT clip's map layers (cuboids as wireframes)."""
    import pyarrow.parquet as pq

    polylines: list[np.ndarray] = []
    for path in sorted(Path(directory).glob("*.parquet")):
        layer = path.name.split(".")[-2]
        if layer not in POLYLINE_LAYERS and layer not in CUBOID_LAYERS:
            continue
        for record in pq.read_table(path).column(layer).to_pylist():
            if layer in CUBOID_LAYERS:
                d = record["dimensions"]
                corners = box_wireframe_points([_cuboid_pose(record)], [(d["x"], d["y"], d["z"])], spacing=1e9)
                polylines.append(corners)
                continue
            for field in POLYLINE_LAYERS[layer]:
                line = _xyz(record[field])
                if layer in CLOSED_LAYERS and len(line) > 2:
                    line = np.concatenate([line, line[:1]])
                polylines.append(line)
    return polylines
