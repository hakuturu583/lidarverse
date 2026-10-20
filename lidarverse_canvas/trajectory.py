# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Poses between and after the simulator's ticks.

A block of sweeps is generated when its first sweep is due, so the poses of
its later sweeps lie ahead of the simulation: they are extrapolated at the
constant (screw) velocity of the last two ticks, as are the actors' boxes.
"""

from __future__ import annotations

from collections import deque

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


class Track:
    """Recent (time, 4x4 pose) samples of one body."""

    def __init__(self, maxlen: int = 8):
        self.samples: deque[tuple[float, np.ndarray]] = deque(maxlen=maxlen)

    def add(self, t: float, pose: np.ndarray) -> None:
        if self.samples and t <= self.samples[-1][0] + 1e-9:
            self.samples[-1] = (t, np.asarray(pose, np.float64))  # same tick again
            return
        self.samples.append((t, np.asarray(pose, np.float64)))

    @property
    def latest_time(self) -> float:
        return self.samples[-1][0]

    def at(self, t: float) -> np.ndarray:
        if not self.samples:
            raise ValueError("no samples")
        times = [s[0] for s in self.samples]
        if len(self.samples) == 1 or t <= times[0]:
            return self.samples[0][1].copy() if t <= times[0] else self.samples[-1][1].copy()
        if t >= times[-1]:
            (t0, a), (t1, b) = self.samples[-2], self.samples[-1]
            return _screw(a, b, (t - t0) / (t1 - t0))
        i = int(np.searchsorted(times, t))
        (t0, a), (t1, b) = self.samples[i - 1], self.samples[i]
        return _interpolate(a, b, (t - t0) / (t1 - t0))


def _interpolate(a: np.ndarray, b: np.ndarray, f: float) -> np.ndarray:
    out = np.eye(4)
    slerp = Slerp([0.0, 1.0], Rotation.from_matrix(np.stack([a[:3, :3], b[:3, :3]])))
    out[:3, :3] = slerp([f]).as_matrix()[0]
    out[:3, 3] = (1 - f) * a[:3, 3] + f * b[:3, 3]
    return out


def _screw(a: np.ndarray, b: np.ndarray, f: float) -> np.ndarray:
    """a moved by f times the body-frame motion from a to b (f may exceed 1)."""
    delta = np.linalg.inv(a) @ b
    step = np.eye(4)
    step[:3, :3] = Rotation.from_rotvec(Rotation.from_matrix(delta[:3, :3]).as_rotvec() * f).as_matrix()
    step[:3, 3] = delta[:3, 3] * f
    return a @ step
