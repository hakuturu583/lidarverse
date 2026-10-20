# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LidarVerse as a carla_canvas renderer: CARLA's LiDARs from the world model.

carla_canvas (https://github.com/hakuturu583/carla_canvas) swaps CARLA's
sensor rendering behind the CARLA API. This backend answers its ray-cast
LiDARs with LidarVerse's HDMap-conditioned generation ("world" mode of a
causal package, by default Stage 3, the four-step self-forcing model):

* the map's OpenDRIVE (LoadWorld) becomes the raw vector map the models were
  conditioned on (vector_map.py), and the actors' bounding boxes its boxes;
* every 0.1 s of simulation time (the model's 10 Hz) one 360-degree sweep is
  due: frame 0 alone, then blocks of four, each rasterized into the HDMap
  range condition from the LiDAR's poses at the sweeps' start and end, and
  generated with the rolling K/V and spatial caches (stream.py). A block's
  later poses are extrapolated from the last ticks (trajectory.py);
* each CARLA measurement reads its rays from the sweep covering its time, at
  the nearest grid cell (sensor.py). carla_canvas applies CARLA's LiDAR model
  around it (partial sweeps per tick, drop-off, noise, intensity).

The released packages are range-only and trained on NVIDIA PAI's Pandar128
(128 lasers, +15..-25 deg, 2.5-100 m): rays outside that field of view or
range return nothing, whatever the CARLA blueprint's channels. Semantic
LiDARs and cameras are not rendered (combine with another backend for those).

Configuration (``renderer`` section of the carla_canvas file)::

    renderer:
      backend: lidarverse_canvas:LidarverseRenderer
      options:
        checkpoints: checkpoints        # scripts/download_checkpoints.py output
        stage: 3                        # 3 (SF-DMD, 4 steps) or 2 (causal)
        num_steps: 4
        seed: 0
        actors: ["vehicle.*", "walker.pedestrian.*"]
"""

from __future__ import annotations

import hashlib
import logging
import math
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from carla_canvas import CameraFrame, CameraView, Renderer, WorldInfo
from carla_canvas.geometry import Transform, to_frame
from carla_canvas.renderer import LidarView, RayHits

from . import sensor
from .trajectory import Track
from .vector_map import VectorMap

log = logging.getLogger("lidarverse_canvas")

REPO_ROOT = Path(__file__).resolve().parents[1]
SWEEP_SECONDS = 1.0 / sensor.FPS
MAP_RADIUS_M = sensor.MAX_RANGE_M + 10.0


@dataclass
class _Actor:
    track: Track
    lwh: np.ndarray


@dataclass
class _Sensor:
    stream: Any
    t0: float | None = None
    last_time: float | None = None
    ego: Track = field(default_factory=Track)
    actors: dict[int, _Actor] = field(default_factory=dict)
    frames: dict[int, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)


class LidarverseRenderer(Renderer):
    name = "lidarverse"
    version = "0.1.0"
    camera_outputs: frozenset[str] = frozenset()
    coordinate_frame = "right_handed"  # the models' and OpenDRIVE's: x forward, y left, z up
    semantic_lidar = False

    def __init__(self, options: dict[str, Any] | None = None) -> None:
        super().__init__(options)
        o = self.options
        self.stage = int(o.get("stage", 3))
        self.num_steps = int(o.get("num_steps", 4 if self.stage == 3 else 35))
        self.sde_eta = float(o.get("sde_eta", 1.0))
        self.seed = int(o.get("seed", 0))
        self.kv_cache_frames = int(o.get("kv_cache_frames", 9 if self.stage == 3 else 21))
        self.segment_spacing = float(o.get("segment_spacing", 0.1))
        self.bbox_dilation = int(o.get("bbox_dilation", 3))
        self.actor_patterns = list(o.get("actors", ["vehicle.*", "walker.pedestrian.*"]))
        self.map_offset = np.asarray(o.get("map_offset", [0.0, 0.0, 0.0]), np.float32)
        self.reset_gap = float(o.get("reset_after_seconds", 1.0))
        self.device = str(o.get("device", "cuda"))
        self.warmup = bool(o.get("warmup", True))
        self.vector_map: VectorMap | None = None
        self._map_key: str | None = None
        self._model = None
        self._sensors: dict[int, _Sensor] = {}
        self._lock = threading.Lock()

    # -- carla_canvas -----------------------------------------------------------

    def load_world(self, world: WorldInfo) -> None:
        with self._lock:
            self._sensors.clear()
            hdmap = self.options.get("hdmap")
            if hdmap:
                key = f"rdshq:{hdmap['rdshq']}:{hdmap['clip']}"
                if key != self._map_key:
                    self.vector_map = VectorMap.from_rdshq(world.resolve(hdmap["rdshq"]), hdmap["clip"],
                                                           self.segment_spacing)
            else:
                if not world.opendrive:
                    raise RuntimeError("lidarverse needs the map's OpenDRIVE (carla_canvas protocol 2) "
                                       "or renderer.options.hdmap")
                key = "xodr:" + hashlib.sha256(world.opendrive.encode()).hexdigest()
                if key != self._map_key:
                    self.vector_map = VectorMap.from_opendrive(world.opendrive, self.segment_spacing)
            if key != self._map_key and self.map_offset.any():
                self.vector_map.points = self.vector_map.points + self.map_offset
                self.vector_map.vertices = self.vector_map.vertices + self.map_offset
            self._map_key = key
            log.info("%s: %d map polylines, %d condition points", world.map_basename,
                     len(self.vector_map.offsets) - 1, len(self.vector_map.points))
            if self._model is None:
                self._model = self._load_model(world)
                if self.warmup:
                    self._warm_up()

    def render_camera(self, view: CameraView) -> CameraFrame:
        raise NotImplementedError("lidarverse renders LiDARs only")

    def cast_lidar_rays(self, view: LidarView) -> RayHits:
        with self._lock:
            ranges, valid = self._sweep(view)
        rows, columns, inside = sensor.grid_cells(view.body_directions())
        distance = ranges[rows, columns]
        hit = inside & valid[rows, columns] & (distance <= view.max_distance)
        return RayHits(distance=np.where(hit, distance, np.inf))

    def close(self) -> None:
        with self._lock:
            self._sensors.clear()
            self._model = None

    # -- overridable for tests ----------------------------------------------------

    def _load_model(self, world: WorldInfo):
        from .stream import load_world_model

        checkpoints = self.options.get("checkpoints")
        root = world.resolve(checkpoints) if checkpoints else REPO_ROOT / "checkpoints"
        log.info("loading LidarVerse stage %d from %s", self.stage, root)
        return load_world_model(root, self.stage)

    def _new_stream(self):
        from .stream import WorldStream

        return WorldStream(self._model, num_steps=self.num_steps, sde_eta=self.sde_eta,
                           seed=self.seed, kv_cache_frames=self.kv_cache_frames)

    def _condition(self, pose_sensor_start_end: np.ndarray, box_points: list[np.ndarray]):
        from diffusion.inference.raw_hdmap import rasterize_points_condition

        centers = pose_sensor_start_end[:, :, :3, 3].reshape(-1, 3)
        condition, *_ = rasterize_points_condition(
            self.vector_map.near(centers, MAP_RADIUS_M), box_points, pose_sensor_start_end,
            sensor.ray_directions(), bbox_dilation=self.bbox_dilation, device=self.device,
        )
        return condition

    def _warm_up(self) -> None:
        """One throwaway frame and block, so that compiling the attention
        kernels happens here (LoadWorld's deadline is renderer.startup_timeout)
        rather than in the first ticks (renderer.timeout)."""
        import torch

        stream = self._new_stream()
        pose = np.broadcast_to(np.eye(4, dtype=np.float32), (2, 4, 4))
        for n in (1, 4):
            empty = torch.full((n, sensor.HEIGHT, sensor.WIDTH), -1.0)
            stream.generate(np.stack([pose] * n), empty)
        log.info("warm-up done")

    # -- sweeps -------------------------------------------------------------------

    def _sweep(self, view: LidarView) -> tuple[np.ndarray, np.ndarray]:
        if self.vector_map is None:
            raise RuntimeError("load_world has not been called")
        t = float(view.scene.elapsed_seconds)
        state = self._sensors.get(view.sensor_id)
        if state is not None and state.last_time is not None and (
                t < state.last_time - 1e-6 or t - state.last_time > self.reset_gap):
            log.info("sensor %d: time jumped %.3f -> %.3f s, restarting its stream",
                     view.sensor_id, state.last_time, t)
            state = None
        if state is None:
            state = self._sensors[view.sensor_id] = _Sensor(self._new_stream())
        state.last_time = t
        if state.t0 is None:
            state.t0 = t
        state.ego.add(t, view.pose_matrix())
        self._track_actors(state, view, t)
        frame = max(0, math.ceil((t - state.t0) / SWEEP_SECONDS - 1e-6))
        while state.stream.frame <= frame:
            self._generate_block(state)
        for old in [k for k in state.frames if k < frame - 1]:
            del state.frames[old]
        return state.frames[frame]

    def _track_actors(self, state: _Sensor, view: LidarView, t: float) -> None:
        scene = view.scene
        skip = set(view.ignore_actors) | {view.sensor_id} | set(scene.ancestors(view.sensor_id))
        present = set()
        for info, transform in scene.bodies(exclude=skip):
            if not any(info.matches(p) for p in self.actor_patterns):
                continue
            b = info.bounding_box
            box_to_world = to_frame(transform.matrix() @ Transform(b.location, b.rotation).matrix(), self.coordinate_frame)
            actor = state.actors.get(info.id)
            if actor is None:
                actor = state.actors[info.id] = _Actor(Track(maxlen=4), 2 * np.asarray(b.extent, np.float64))
            actor.track.add(t, box_to_world)
            present.add(info.id)
        for gone in set(state.actors) - present:
            del state.actors[gone]

    def _generate_block(self, state: _Sensor) -> None:
        from diffusion.inference.raw_hdmap import box_wireframe_points

        first, n = state.stream.frame, state.stream.next_block_length()
        ends = state.t0 + (first + np.arange(n)) * SWEEP_SECONDS
        poses = np.stack([[state.ego.at(e - SWEEP_SECONDS), state.ego.at(e)] for e in ends]).astype(np.float32)
        actors = list(state.actors.values())
        box_points = [
            box_wireframe_points([a.track.at(e) for a in actors], [a.lwh for a in actors], self.segment_spacing)
            for e in ends
        ]
        ranges, valid = state.stream.generate(poses, self._condition(poses, box_points))
        for i in range(n):
            state.frames[first + i] = (np.asarray(ranges[i], np.float32), np.asarray(valid[i], bool))
