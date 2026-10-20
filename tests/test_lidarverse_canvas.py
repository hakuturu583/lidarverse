# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The carla_canvas backend on CPU, without checkpoints: the generator is
replaced by one that returns its HDMap condition, so a scan shows the map
where the backend placed it."""

import math

import numpy as np
import pytest
import torch

roadgen = pytest.importorskip("roadgen")

from carla_canvas.geometry import Transform  # noqa: E402
from carla_canvas.lidar import LidarSimulator  # noqa: E402
from carla_canvas.renderer import WorldInfo  # noqa: E402
from carla_canvas.scene import ActorInfo, ActorState, BoundingBox, Scene  # noqa: E402
from carla_canvas.sensors import LidarSpec  # noqa: E402

from lidarverse_canvas import LidarverseRenderer, VectorMap, sensor  # noqa: E402
from lidarverse_canvas.trajectory import Track  # noqa: E402


def opendrive(y=10.0):
    """A two-lane road along OpenDRIVE x at OpenDRIVE y = ``y``."""
    m = roadgen.Map()
    m.add_road(start=(0.0, y, 0.0), end=(100.0, y, 0.0),
               lanes=[roadgen.Lane(width=3.5, direction="forward"), roadgen.Lane(width=3.5, direction="backward")])
    return m.to_opendrive_xml()


class EchoStream:
    """Returns the HDMap condition's ranges as the generated sweep."""

    def __init__(self):
        self.frame = 0
        self.blocks = []

    def next_block_length(self):
        return 1 if self.frame == 0 else 4

    def generate(self, poses, hdmap):
        from diffusion.inference.hdmap import _metric_ranges

        assert len(poses) == self.next_block_length() and hdmap.shape == (len(poses), 128, 1024)
        self.blocks.append(poses)
        self.frame += len(poses)
        return _metric_ranges(hdmap[None, None])


class EchoRenderer(LidarverseRenderer):
    def _load_model(self, world):
        return object()

    def _new_stream(self):
        self.stream = EchoStream()
        return self.stream


def test_grid_cells_round_trip():
    rays = sensor.ray_directions().reshape(-1, 3)
    rows, columns, inside = sensor.grid_cells(rays)
    assert inside.all()
    np.testing.assert_array_equal(rows.reshape(128, 1024), np.repeat(np.arange(128)[:, None], 1024, 1))
    np.testing.assert_array_equal(columns.reshape(128, 1024), np.repeat(np.arange(1024)[None], 128, 0))
    # columns run clockwise from behind: left (+y) 256, ahead 512, right 768; far above the lasers is outside
    _, c, ok = sensor.grid_cells(np.array([[0, 1.0, 0], [1.0, 0, 0], [0, -1.0, 0], [0.5, 0, 1.0]]))
    assert c[:3].tolist() == [256, 512, 768] and ok.tolist() == [True, True, True, False]


def test_vector_map_from_opendrive_keeps_its_frame():
    vm = VectorMap.from_opendrive(opendrive(10.0))
    ys = np.unique(np.round(vm.vertices[:, 1], 3))
    assert {6.5, 10.0, 13.5} <= set(ys.tolist())
    assert vm.vertices[:, 0].min() == pytest.approx(0.0) and vm.vertices[:, 0].max() == pytest.approx(100.0)
    near = vm.near(np.array([[0.0, 10.0, 0.0]]), 5.0)
    assert len(near) and np.abs(near[:, 0]).max() <= 5.0 + 1e-4


def test_track_interpolates_and_extrapolates():
    track = Track()
    for t in (0.0, 0.05):
        pose = np.eye(4)
        pose[:3, :3] = Transform((0, 0, 0), (0, math.degrees(t), 0)).rotation_matrix()
        pose[0, 3] = 10 * t
        track.add(t, pose)
    np.testing.assert_allclose(track.at(0.025)[:3, 3], [0.25, 0, 0], atol=1e-12)
    ahead = track.at(0.15)
    assert ahead[0, 3] == pytest.approx(1.5, abs=0.01)
    np.testing.assert_allclose(ahead[:3, :3], Transform((0, 0, 0), (0, math.degrees(0.15), 0)).rotation_matrix(),
                               atol=1e-9)
    np.testing.assert_allclose(track.at(-1.0), track.samples[0][1])


def scene_at(t, sensor_location):
    scene = Scene()
    scene.elapsed_seconds = t
    scene.add_actor(ActorInfo(id=9, type_id="sensor.lidar.ray_cast"))
    scene.states[9] = ActorState(Transform(sensor_location))
    scene.add_actor(ActorInfo(id=5, type_id="vehicle.tesla.model3",
                              bounding_box=BoundingBox((0, 0, 0.75), (2.3, 1.0, 0.75))))
    scene.states[5] = ActorState(Transform((sensor_location[0] + 12.0, -10.0, 0.0)))
    return scene


def test_scan_places_the_map_in_carla_frame_and_keeps_the_cadence():
    renderer = EchoRenderer({"device": "cpu", "bbox_dilation": 1})
    renderer.load_world(WorldInfo("Carla/Maps/Test", 1, {}, opendrive=opendrive(10.0)))
    spec = LidarSpec(channels=64, range=100.0, points_per_second=1_280_000, upper_fov=10.0, lower_fov=-30.0,
                     dropoff_general_rate=0.0, dropoff_intensity_limit=0.0)
    lidar = LidarSimulator(spec, sensor_id=9)
    location = (50.0, 0.0, 2.0)  # CARLA: the road is at y = -10 (OpenDRIVE y negated)
    scan = lidar.scan(renderer, scene_at(0.0, location), Transform(location), 0.1)
    assert [len(b) for b in renderer.stream.blocks] == [1]
    points = scan.points
    off_car = np.abs(points[:, 0] - 12) > 3
    ground = points[(points[:, 2] < -1.5) & (np.linalg.norm(points[:, :2], axis=1) < 30) & off_car]
    assert len(ground) > 50
    np.testing.assert_allclose(ground[:, 2], -2.0, atol=0.3)
    # the lines are at CARLA y = -6.5, -10, -13.5 (sensor y right): never on the right
    distance_to_line = np.abs(ground[:, 1][:, None] - np.array([-6.5, -10.0, -13.5])[None]).min(1)
    assert np.median(distance_to_line) < 0.3 and (ground[:, 1] < -5).all()
    # the car 12 m ahead at y = -10 is in the condition too
    car = points[(np.abs(points[:, 0] - 12) < 2.5) & (points[:, 2] > -1.5)]
    assert len(car) and np.abs(car[:, 1] + 10).max() < 1.5

    # 0.05 s later: still frame 1's sweep after one block of four, generated once
    lidar.scan(renderer, scene_at(0.05, location), Transform(location), 0.05)
    lidar.scan(renderer, scene_at(0.1, location), Transform(location), 0.05)
    assert [len(b) for b in renderer.stream.blocks] == [1, 4]
    first = renderer.stream.blocks[1]
    np.testing.assert_allclose(first[:, 1, :3, 3], [[50, 0, 2]] * 4, atol=1e-5)  # right-handed: y stays 0

    # a backwards jump (new episode without load_world) restarts the stream
    lidar.scan(renderer, scene_at(0.0, location), Transform(location), 0.05)
    assert [len(b) for b in renderer.stream.blocks] == [1]


def test_cameras_are_not_claimed():
    renderer = EchoRenderer({})
    assert renderer.lidar_types == ["sensor.lidar.ray_cast"]
    assert not renderer.camera_outputs


def test_compat_rmsnorm_loads_transformer_engine_state():
    from utils.compat import TorchRMSNorm

    norm = TorchRMSNorm(8, eps=1e-6)
    norm.load_state_dict({"weight": torch.full((8,), 2.0), "_extra_state": torch.zeros(0)}, strict=True)
    x = torch.randn(3, 8, dtype=torch.bfloat16)
    expected = (x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6) * 2).bfloat16()
    torch.testing.assert_close(norm(x), expected)
