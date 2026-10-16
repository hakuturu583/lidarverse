# -----------------------------------------------------------------------------
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# This codebase constitutes NVIDIA proprietary technology and is strictly
# confidential. Any unauthorized reproduction, distribution, or disclosure
# of this code, in whole or in part, outside NVIDIA is strictly prohibited
# without prior written consent.
#
# For inquiries regarding the use of this code in other NVIDIA proprietary
# projects, please contact the Deep Imagination Research Team at
# dir@exchange.nvidia.com.
# -----------------------------------------------------------------------------

from typing import Callable, Optional, Tuple

import numpy as np
import torch

from diffusion.training.schedulers.shift_schedule import (
    sample_stretched_shifted_u,
    stretch_shifted_t_to_unit,
    stretch_shifts_to_terminal,
)


class TrainTimeWeight:
    def __init__(
        self,
        noise_scheduler,
        weight: str = "uniform",
    ):
        # Map reweighting -> uniform to support inference for existing checkpoints.
        if weight == "reweighting":
            weight = "uniform"

        self.weight = weight
        self.noise_scheduler = noise_scheduler

        assert self.weight == "uniform", "Only uniform loss weight is supported in RF"

    def __call__(self, t, tensor_kwargs) -> torch.Tensor:
        if self.weight == "uniform":
            wts = torch.ones_like(t)
        else:
            raise NotImplementedError(f"Time weight '{self.weight}' is not implemented.")

        return wts


class TrainTimeSampler:
    def __init__(
        self,
        distribution: str = "uniform",
    ):
        self.distribution = distribution

    @torch.no_grad()
    def __call__(
        self,
        batch_size: int,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """
        Sample time tensor for training

        Returns:
            torch.Tensor: Time tensor, shape (batch_size,)
        """
        if self.distribution == "uniform":
            t = torch.rand((batch_size,)).to(device=device, dtype=dtype)
        elif self.distribution == "logitnormal":
            t = torch.sigmoid(torch.randn((batch_size,))).to(device=device, dtype=dtype)
        elif self.distribution.startswith("waver_mode_"):
            s = float(self.distribution.split("_")[-1])
            assert s - 1.29 < 1e-4, "Waver's mode distribution is only supported with s = 1.29"
            u = torch.rand((batch_size,), dtype=torch.float32)
            t = 1.0 - u - s * (torch.cos(torch.pi / 2.0 * u) ** 2 - 1 + u)
            t = t.to(device=device, dtype=dtype)
        else:
            raise NotImplementedError(f"Time distribution '{self.distribution}' is not implemented.")

        return t


class RectifiedFlow:
    def __init__(
        self,
        velocity_field: Callable,
        train_time_distribution: TrainTimeSampler | str = "uniform",
        train_time_weight_method: str = "uniform",
        use_dynamic_shift: bool = False,
        shift: int = 3,
        shift_clamp_percentiles: Optional[Tuple[float, float]] = None,
        shift_stretch_terminal: Optional[float] = None,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ):
        r"""Initialize the RectifiedFlow class.

        Args:
            velocity_field (`Callable`):
                A function that predicts the velocity given the current state and time.
            train_time_distribution (`TrainTimeSampler` or `str`, *optional*, defaults to `"uniform"`):
                Distribution for sampling training times.
                Can be an instance of `TrainTimeSampler` or a string specifying the distribution type.
            train_time_weight (`TrainTimeWeight` or `str`, *optional*, defaults to `"uniform"`):
                Weight applied to training times.
                Can be an instance of `TrainTimeWeight` or a string specifying the weight type.
        """
        self.velocity_field = velocity_field
        self.train_time_sampler: TrainTimeSampler = (
            train_time_distribution
            if isinstance(train_time_distribution, TrainTimeSampler)
            else TrainTimeSampler(train_time_distribution)
        )

        assert use_dynamic_shift is False, "Dynamic shift is not supported in RectifiedFlow"
        self.shift = shift
        self.shift_clamp_percentiles = shift_clamp_percentiles
        self.shift_stretch_terminal = shift_stretch_terminal
        self.num_train_timesteps = 1000
        self.train_time_weight = TrainTimeWeight(None, train_time_weight_method)

        self.device = torch.device(device) if isinstance(device, str) else device
        self.dtype = torch.dtype(dtype) if isinstance(dtype, str) else dtype

    def sample_train_time(self, batch_size: int):
        r"""This method calls the `TrainTimeSampler` to sample training times.

        Returns:
            t (`torch.Tensor`):
                A tensor of sampled training times with shape `(batch_size,)`,
                matching the class specified `device` and `dtype`.
        """
        if self.shift_clamp_percentiles is not None:
            return sample_stretched_shifted_u(
                batch_size,
                float(self.shift),
                tuple(self.shift_clamp_percentiles),
                device=self.device,
                dtype=self.dtype,
            )
        return self.train_time_sampler(batch_size, device=self.device, dtype=self.dtype)

    def apply_shift_reparam(self, u: torch.Tensor) -> torch.Tensor:
        """Map ``u ∈ [0, 1]`` through shift (+ optional LTX stretch affine to ``[0, 1]``)."""
        t = self.shift * u / (1 + (self.shift - 1) * u)
        if self.shift_clamp_percentiles is not None:
            t = stretch_shifted_t_to_unit(
                t, float(self.shift), tuple(self.shift_clamp_percentiles),
            )
        return t

    def stretch_inference_sigmas(self, sigmas: np.ndarray) -> np.ndarray:
        """LTX stretch on a noise→clean sigma schedule (``sigmas[-1]`` → terminal)."""
        if self.shift_stretch_terminal is None:
            return sigmas
        return stretch_shifts_to_terminal(sigmas, self.shift_stretch_terminal)

    def get_discrete_timestamp(self, u, tensor_kwargs):
        r"""This method map time from 0,1 to discrete steps"""
        u = u.squeeze()
        timesteps = self.apply_shift_reparam(u)
        timesteps = timesteps * self.num_train_timesteps  # [0, 1] to [0, 1000]
        return timesteps.unsqueeze(0) if timesteps.ndim == 0 else timesteps

    def get_sigmas(self, timesteps, tensor_kwargs):
        sigmas = (timesteps.to(**tensor_kwargs)) / self.num_train_timesteps
        return sigmas

    def get_interpolation(
        self,
        x_0: torch.Tensor,
        x_1: torch.Tensor,
        t: torch.Tensor,
    ):
        r"""
        This method computes interpolation `X_t` and their time derivatives `dotX_t` at the specified time points `t`.
        Note that `x_0` is the noise, and `x_1` is the clean data. This is aligned with the notation in the recified flow community,
        but different from the notation in the diffusion community.

        Args:
            x_0 (`torch.Tensor`):
                noise, shape `(B, D1, D2, ..., Dn)`, where `B` is the batch size, and `D1, D2, ..., Dn` are the data dimensions.
            x_1 (`torch.Tensor`):
                clean data, with the same shape as `x_0`
            t (`torch.Tensor`):
                A tensor of time steps, with shape `(B,)`, where each value is in `[0, 1]`.

        Returns:
            (x_t, dot_x_t) (`Tuple[torch.Tensor, torch.Tensor]`):
                - x_t (`torch.Tensor`): The interpolated state, with shape `(B, D1, D2, ..., Dn)`.
                - dot_x_t (torch.Tensor): The time derivative of the interpolated state, with the same shape as `x_t`.
        """
        assert x_0.shape == x_1.shape, "x_0 and x_1 must have the same shape."
        assert x_0.shape[0] == x_1.shape[0], "Batch size of x_0 and x_1 must match."
        assert t.shape[0] == x_1.shape[0], "Batch size of t must match x_1."
        # Reshape t to match dimensions of x_1
        t = t.view(t.shape[0], *([1] * (len(x_1.shape) - 1)))
        x_t = x_0 * t + x_1 * (1 - t)
        dot_x_t = x_0 - x_1
        return x_t, dot_x_t
