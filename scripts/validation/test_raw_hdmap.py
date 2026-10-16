# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused checks for the inference-only raw HDMap helpers (no model needed)."""

import unittest

import numpy as np
import torch

from diffusion.inference.raw_hdmap import _box_points, _dilate_nearest, _sample_polyline


class RawHDMapTests(unittest.TestCase):
    def test_empty_and_identity(self):
        values = torch.full((2, 5, 7), -1.)
        torch.testing.assert_close(_dilate_nearest(values, 3), values)
        values[0, 2, 3] = .5
        torch.testing.assert_close(_dilate_nearest(values, 1), values)

    def test_nearest_depth_wins(self):
        values = torch.full((1, 5, 7), -1.)
        values[0, 2, 2] = .5
        values[0, 2, 4] = .1
        expected = torch.full_like(values, -1.)
        expected[0, 1:4, 1:4] = .5
        expected[0, 1:4, 3:6] = .1
        torch.testing.assert_close(_dilate_nearest(values, 3), expected)

    def test_no_frame_or_boundary_wrap(self):
        values = torch.full((2, 5, 7), -1.)
        values[0, 0, 0] = .5
        expected = torch.full_like(values, -1.)
        expected[0, :2, :2] = .5
        torch.testing.assert_close(_dilate_nearest(values, 3), expected)

    def test_invalid_kernel(self):
        for kernel in (0, 2, -1):
            with self.assertRaises(ValueError):
                _dilate_nearest(torch.zeros(1, 3, 3), kernel)

    def test_polyline_preserves_endpoints(self):
        points = _sample_polyline(np.array([[0, 0, 0], [1, 0, 0]]), .1)
        np.testing.assert_allclose(points[0], [0, 0, 0])
        np.testing.assert_allclose(points[-1], [1, 0, 0])
        self.assertEqual(len(points), 10)

    def test_boxes_use_world_pose_and_mask(self):
        pose = np.eye(4)
        pose[:3, 3] = [10, 20, 30]
        sample = dict(box_valid=np.array([[True, False]]),
                      box_lwh=np.array([[[4, 2, 6], [100, 100, 100]]]),
                      box_object_to_world=np.array([[pose, np.eye(4)]]))
        points = _box_points(sample, 0, .1)
        np.testing.assert_allclose(points.min(0), [8, 19, 27])
        np.testing.assert_allclose(points.max(0), [12, 21, 33])
        sample["box_valid"][:] = False
        self.assertEqual(_box_points(sample, 0, .1).shape, (0, 3))


if __name__ == "__main__":
    unittest.main()
