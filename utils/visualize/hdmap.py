# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Raw-map polylines and translucent actor boxes for PAI quickstart previews.

Map vertices and actors are in the clip reference frame; sensor poses bring
both into the same ego frame as the generated cloud. This is display-only.
"""

import json
import tarfile

import cv2
import numpy as np

from utils.visualize.point_cloud import CAMERA_VIEWS, VIZ_KWARGS

MAP_KINDS = ("lanes", "lanelines", "road_boundaries", "crosswalks", "wait_lines",
             "road_markings", "poles", "traffic_signs", "traffic_lights")
MAP_COLORS = ((168, 96, 224), (232, 86, 232), (226, 42, 214), (46, 204, 113),
              (255, 214, 44), (255, 214, 44), (150, 92, 224), (131, 175, 155), (252, 157, 155))
BOX_COLORS = ((74, 144, 240), (52, 106, 204), (46, 204, 113), (255, 196, 44), (176, 184, 196))
EDGES = ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7))


def object_class(name):
    aliases = {"Bus": "Truck", "Heavy_truck": "Truck", "Train_or_tram_car": "Truck",
               "Trolley_bus": "Truck", "Trailer": "Truck", "Vehicle": "Car",
               "Automobile": "Car", "Other_vehicle": "Car", "Person": "Pedestrian", "Rider": "Cyclist"}
    return {"Car": 0, "Truck": 1, "Pedestrian": 2, "Cyclist": 3}.get(aliases.get(name, name), 4)


def _vertices(node):
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "vertices":
                yield np.asarray(value, dtype=np.float32)
            else:
                yield from _vertices(value)
    elif isinstance(node, list):
        for item in node:
            yield from _vertices(item)


def load_vectors(root, clip):
    points, offsets, kinds = [], [0], []
    for kind_id, kind in enumerate(MAP_KINDS):
        with tarfile.open(root / f"3d_{kind}" / f"{clip}.tar") as archive:
            members = [m for m in archive.getmembers() if m.isfile() and m.name.endswith(".json")]
            if len(members) != 1:
                raise ValueError(f"{clip}/{kind}: expected one vector-map JSON")
            with archive.extractfile(members[0]) as handle:
                data = json.load(handle)
        for vertices in _vertices(data):
            if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
                raise ValueError(f"{clip}/{kind}: invalid vertices")
            if len(vertices) < 2:
                continue
            if kind in ("traffic_signs", "traffic_lights") and len(vertices) == 8:
                vertices = vertices[[0, 1, 2, 3, 0, 4, 5, 6, 7, 4, 5, 1, 2, 6, 7, 3]]
            elif kind in ("crosswalks", "road_markings"):
                vertices = np.concatenate((vertices, vertices[:1]))
            points.append(vertices)
            offsets.append(offsets[-1] + len(vertices))
            kinds.append(kind_id)
    return dict(map_vertices=np.concatenate(points) if points else np.empty((0, 3), np.float32),
                map_offsets=np.asarray(offsets, np.int64), map_kinds=np.asarray(kinds, np.int64))


def render_hdmap(sample, frame):
    """Return a white-background BGR frame matching the point-cloud camera."""
    for key in ("map_vertices", "map_offsets", "map_kinds", "box_class"):
        if key not in sample:
            raise ValueError(f"Missing {key}; rerun scripts/data/prepare_pai_bidir.py for raw-map previews")
    width, height = VIZ_KWARGS["width"], VIZ_KWARGS["height"]
    camera = CAMERA_VIEWS["paper_view_down"]
    center = np.array([camera["center"][a] for a in "xyz"]) * VIZ_KWARGS["range"]
    eye = np.array([camera["eye"][a] for a in "xyz"]) * VIZ_KWARGS["range"]
    eye = center + .6 * (eye - center)
    forward = center - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0, 0, 1])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    transform = np.linalg.inv(sample["pose_sensor_start_end"][frame, 0])
    tangent = np.tan(np.deg2rad(VIZ_KWARGS.get("fov_deg", 60)) / 2)

    def project(points, world=True):
        local = points @ transform[:3, :3].T + transform[:3, 3] if world else points
        relative = local - eye
        depth = relative @ forward
        x = (relative @ right / (np.maximum(depth, 1e-3) * tangent * width / height) * .5 + .5) * width
        y = (1 - (relative @ up / (np.maximum(depth, 1e-3) * tangent) * .5 + .5 + .08)) * height
        return np.clip(np.stack((x, y), -1), -100000, 100000).astype(np.int32), depth, local

    image = np.full((height, width, 3), 255, np.uint8)
    map_lines = []
    box_lines = []
    for start, end, kind in zip(sample["map_offsets"][:-1], sample["map_offsets"][1:], sample["map_kinds"]):
        pixels, depth, local = project(sample["map_vertices"][start:end])
        for i in range(len(pixels) - 1):
            if min(depth[i:i+2]) > .1 and np.linalg.norm(local[i:i+2], axis=1).min() < 100:
                map_lines.append((pixels[i], pixels[i+1], MAP_COLORS[kind][::-1]))

    def box(size, pose, color, world=True):
        corners = np.array([[1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1],
                            [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1]]) * size / 2
        pixels, depth, local = project(corners @ pose[:3, :3].T + pose[:3, 3], world)
        if depth.min() <= .1 or np.linalg.norm(local.mean(0)) > 100:
            return
        overlay = image.copy()
        hull = cv2.convexHull(pixels)
        cv2.fillConvexPoly(overlay, hull, color, cv2.LINE_AA)
        cv2.addWeighted(overlay, .25, image, .75, 0, dst=image)
        box_lines.extend((pixels[a], pixels[b], color) for a, b in EDGES)

    active = np.flatnonzero(sample["box_valid"][frame])
    # Paint distant boxes first. Outlines are drawn after all translucent fills.
    origin = sample["pose_sensor_start_end"][frame, 0, :3, 3]
    active = sorted(active, key=lambda i: -np.linalg.norm(sample["box_object_to_world"][frame, i, :3, 3] - origin))
    for i in active:
        box(sample["box_lwh"][frame, i], sample["box_object_to_world"][frame, i],
            BOX_COLORS[int(sample["box_class"][frame, i])][::-1])
    ego = np.eye(4)
    ego[2, 3] = -1.1
    box(np.array([4.5, 2., 1.6]), ego, (40, 122, 240), world=False)
    for a, b, color in map_lines:
        cv2.line(image, tuple(a), tuple(b), color, 1, cv2.LINE_AA)
    # Confine the soft halo to box edges; map polylines stay crisp and thin.
    halo = np.zeros_like(image, dtype=np.float32)
    coverage = np.zeros((height, width), dtype=np.float32)
    for a, b, color in box_lines:
        cv2.line(halo, tuple(a), tuple(b), color, 3)
        cv2.line(coverage, tuple(a), tuple(b), 1., 3)
    halo = cv2.GaussianBlur(halo, (5, 5), .8)
    coverage = cv2.GaussianBlur(coverage, (5, 5), .8)[..., None]
    image = np.clip(image * (1 - .12 * coverage) + .12 * halo, 0, 255).astype(np.uint8)
    for a, b, color in box_lines:
        cv2.line(image, tuple(a), tuple(b), color, 1, cv2.LINE_AA)
    return image
