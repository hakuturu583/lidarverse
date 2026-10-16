# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 TSAIL Team and The HuggingFace Team. All rights reserved.
# NVIDIA modifications: LiDAR rectified-flow sampling integration.

# Adapted from https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_dpmsolver_multistep.py
# Specialized for FLOW MATCHING (rectified flow): prediction_type="flow_prediction",
# flow sigmas (alpha_t = 1 - sigma, sigma_t = sigma), and the SAME shift / LTX-stretch
# sigma ladder as ``FlowUniPCMultistepScheduler`` so it is a drop-in sibling of the
# UniPC scheduler already used by the RF models.
#
# Supports the "dpmsolver++" (deterministic, high-order ODE) and "sde-dpmsolver++"
# (stochastic) algorithm types -- i.e. "Flow-DPM-Solver++". Orders 1-3, midpoint /
# heun second-order variants.

import math
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import (
    KarrasDiffusionSchedulers,
    SchedulerMixin,
    SchedulerOutput,
)

from diffusion.training.schedulers.shift_schedule import stretch_shifts_to_terminal


class FlowDPMSolverMultistepScheduler(SchedulerMixin, ConfigMixin):
    """Flow-matching DPM-Solver++ (and SDE-DPM-Solver++) multistep scheduler.

    Mirrors :class:`FlowUniPCMultistepScheduler` (same flow sigma ladder,
    ``set_timesteps`` shift/stretch interface, ``step`` signature) but the
    update rule is DPM-Solver++ (data-prediction exponential integrator).

    Args:
        solver_order: 1, 2, or 3. Effective ODE order. 2 is a good default for
            guided sampling; 3 for unconditional.
        algorithm_type: ``"dpmsolver++"`` (deterministic) or
            ``"sde-dpmsolver++"`` (stochastic). The SDE variant injects fresh
            noise each step (the marginal-preserving Langevin term) and tends to
            beat the ODE for imperfect models at enough steps.
        solver_type: ``"midpoint"`` or ``"heun"`` (2nd-order coefficient variant).
        lower_order_final / euler_at_final: drop to lower order on the last
            step(s) for stability at small step counts (recommended).
    """

    _compatibles = [e.name for e in KarrasDiffusionSchedulers]
    order = 1

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        solver_order: int = 2,
        prediction_type: str = "flow_prediction",
        shift: Optional[float] = 1.0,
        use_dynamic_shifting: bool = False,
        thresholding: bool = False,
        dynamic_thresholding_ratio: float = 0.995,
        sample_max_value: float = 1.0,
        algorithm_type: str = "dpmsolver++",
        solver_type: str = "midpoint",
        lower_order_final: bool = True,
        euler_at_final: bool = False,
        final_sigmas_type: str = "zero",  # "zero" | "sigma_min"
    ):
        if algorithm_type not in ["dpmsolver++", "sde-dpmsolver++"]:
            raise NotImplementedError(
                f"{algorithm_type} is not implemented for {self.__class__}. "
                "Use 'dpmsolver++' or 'sde-dpmsolver++'."
            )
        if solver_type not in ["midpoint", "heun"]:
            if solver_type in ["logrho", "bh1", "bh2"]:
                self.register_to_config(solver_type="midpoint")
            else:
                raise NotImplementedError(
                    f"{solver_type} is not implemented for {self.__class__}"
                )

        # Flow sigma ladder (sigma in [0,1]); mirrors FlowUniPCMultistepScheduler.
        self.num_inference_steps = None
        alphas = np.linspace(1, 1 / num_train_timesteps, num_train_timesteps)[::-1].copy()
        sigmas = 1.0 - alphas
        sigmas = torch.from_numpy(sigmas).to(dtype=torch.float32)
        if not use_dynamic_shifting:
            sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)

        self.sigmas = sigmas
        self.timesteps = sigmas * num_train_timesteps

        self.model_outputs = [None] * solver_order
        self.lower_order_nums = 0
        self._step_index = None
        self._begin_index = None

        self.sigmas = self.sigmas.to("cpu")
        self.sigma_min = self.sigmas[-1].item()
        self.sigma_max = self.sigmas[0].item()

    @property
    def step_index(self):
        return self._step_index

    @property
    def begin_index(self):
        return self._begin_index

    def set_begin_index(self, begin_index: int = 0):
        self._begin_index = begin_index

    def _sigma_to_alpha_sigma_t(self, sigma):
        # Flow matching: x_t = (1 - sigma) * x0 + sigma * eps.
        return 1 - sigma, sigma

    def time_shift(self, mu: float, sigma: float, t: torch.Tensor):
        return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)

    def set_timesteps(
        self,
        num_inference_steps: Union[int, None] = None,
        device: Union[str, torch.device] = None,
        sigmas: Optional[List[float]] = None,
        mu: Optional[float] = None,
        shift: Optional[float] = None,
        stretch_terminal: Optional[float] = None,
    ):
        """Build the flow sigma ladder (shift + optional LTX stretch), append the
        terminal sigma, and reset the multistep buffers. Signature matches
        ``FlowUniPCMultistepScheduler.set_timesteps`` so it's a drop-in swap."""
        if self.config.use_dynamic_shifting and mu is None:
            raise ValueError("pass `mu` when `use_dynamic_shifting=True`")

        if sigmas is None:
            sigmas = np.linspace(self.sigma_max, self.sigma_min, num_inference_steps + 1).copy()[:-1]

        if self.config.use_dynamic_shifting:
            sigmas = self.time_shift(mu, 1.0, sigmas)
        else:
            if shift is None:
                shift = self.config.shift
            sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
            if stretch_terminal is not None:
                sigmas = stretch_shifts_to_terminal(
                    np.asarray(sigmas, dtype=np.float64), float(stretch_terminal)
                )

        if self.config.final_sigmas_type == "sigma_min":
            sigma_last = self.sigma_min
        elif self.config.final_sigmas_type == "zero":
            sigma_last = 0
        else:
            raise ValueError(
                f"`final_sigmas_type` must be 'zero' or 'sigma_min', got "
                f"{self.config.final_sigmas_type}"
            )

        timesteps = sigmas * self.config.num_train_timesteps
        sigmas = np.concatenate([sigmas, [sigma_last]]).astype(np.float32)

        self.sigmas = torch.from_numpy(sigmas)
        self.timesteps = torch.from_numpy(timesteps).to(device=device, dtype=torch.int64)

        self.num_inference_steps = len(timesteps)
        self.model_outputs = [None] * self.config.solver_order
        self.lower_order_nums = 0
        self._step_index = None
        self._begin_index = None
        self.sigmas = self.sigmas.to("cpu")

    def _threshold_sample(self, sample: torch.Tensor) -> torch.Tensor:
        dtype = sample.dtype
        batch_size, channels, *remaining_dims = sample.shape
        if dtype not in (torch.float32, torch.float64):
            sample = sample.float()
        sample = sample.reshape(batch_size, channels * int(np.prod(remaining_dims)))
        abs_sample = sample.abs()
        s = torch.quantile(abs_sample, self.config.dynamic_thresholding_ratio, dim=1)
        s = torch.clamp(s, min=1, max=self.config.sample_max_value)
        s = s.unsqueeze(1)
        sample = torch.clamp(sample, -s, s) / s
        sample = sample.reshape(batch_size, channels, *remaining_dims)
        return sample.to(dtype)

    def convert_model_output(
        self, model_output: torch.Tensor, *args, sample: torch.Tensor = None, **kwargs
    ) -> torch.Tensor:
        """Flow velocity -> predicted clean latent x0 (DPM-Solver++ is a data-
        prediction integrator). x_t = (1-sigma)x0 + sigma*eps, v = eps - x0
        => x0 = x_t - sigma * v."""
        if sample is None:
            raise ValueError("missing `sample` as a required keyword argument")
        if self.config.prediction_type != "flow_prediction":
            raise ValueError(
                f"prediction_type {self.config.prediction_type} must be "
                "'flow_prediction' for FlowDPMSolverMultistepScheduler."
            )
        sigma_t = self.sigmas[self.step_index]
        x0_pred = sample - sigma_t * model_output
        if self.config.thresholding:
            x0_pred = self._threshold_sample(x0_pred)
        return x0_pred

    def dpm_solver_first_order_update(
        self, model_output: torch.Tensor, *args,
        sample: torch.Tensor = None, noise: Optional[torch.Tensor] = None, **kwargs,
    ) -> torch.Tensor:
        if sample is None:
            raise ValueError("missing `sample` as a required keyword argument")
        sigma_t, sigma_s = self.sigmas[self.step_index + 1], self.sigmas[self.step_index]
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t)
        alpha_s, sigma_s = self._sigma_to_alpha_sigma_t(sigma_s)
        lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
        lambda_s = torch.log(alpha_s) - torch.log(sigma_s)
        h = lambda_t - lambda_s
        if self.config.algorithm_type == "dpmsolver++":
            x_t = (sigma_t / sigma_s) * sample - (alpha_t * (torch.exp(-h) - 1.0)) * model_output
        elif self.config.algorithm_type == "sde-dpmsolver++":
            assert noise is not None
            x_t = (
                (sigma_t / sigma_s * torch.exp(-h)) * sample
                + (alpha_t * (1 - torch.exp(-2.0 * h))) * model_output
                + sigma_t * torch.sqrt(1.0 - torch.exp(-2 * h)) * noise
            )
        return x_t

    def multistep_dpm_solver_second_order_update(
        self, model_output_list: List[torch.Tensor], *args,
        sample: torch.Tensor = None, noise: Optional[torch.Tensor] = None, **kwargs,
    ) -> torch.Tensor:
        if sample is None:
            raise ValueError("missing `sample` as a required keyword argument")
        sigma_t, sigma_s0, sigma_s1 = (
            self.sigmas[self.step_index + 1],
            self.sigmas[self.step_index],
            self.sigmas[self.step_index - 1],
        )
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t)
        alpha_s0, sigma_s0 = self._sigma_to_alpha_sigma_t(sigma_s0)
        alpha_s1, sigma_s1 = self._sigma_to_alpha_sigma_t(sigma_s1)
        lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
        lambda_s0 = torch.log(alpha_s0) - torch.log(sigma_s0)
        lambda_s1 = torch.log(alpha_s1) - torch.log(sigma_s1)

        m0, m1 = model_output_list[-1], model_output_list[-2]
        h, h_0 = lambda_t - lambda_s0, lambda_s0 - lambda_s1
        r0 = h_0 / h
        D0, D1 = m0, (1.0 / r0) * (m0 - m1)
        if self.config.algorithm_type == "dpmsolver++":
            if self.config.solver_type == "midpoint":
                x_t = (
                    (sigma_t / sigma_s0) * sample
                    - (alpha_t * (torch.exp(-h) - 1.0)) * D0
                    - 0.5 * (alpha_t * (torch.exp(-h) - 1.0)) * D1
                )
            else:  # heun
                x_t = (
                    (sigma_t / sigma_s0) * sample
                    - (alpha_t * (torch.exp(-h) - 1.0)) * D0
                    + (alpha_t * ((torch.exp(-h) - 1.0) / h + 1.0)) * D1
                )
        elif self.config.algorithm_type == "sde-dpmsolver++":
            assert noise is not None
            if self.config.solver_type == "midpoint":
                x_t = (
                    (sigma_t / sigma_s0 * torch.exp(-h)) * sample
                    + (alpha_t * (1 - torch.exp(-2.0 * h))) * D0
                    + 0.5 * (alpha_t * (1 - torch.exp(-2.0 * h))) * D1
                    + sigma_t * torch.sqrt(1.0 - torch.exp(-2 * h)) * noise
                )
            else:  # heun
                x_t = (
                    (sigma_t / sigma_s0 * torch.exp(-h)) * sample
                    + (alpha_t * (1 - torch.exp(-2.0 * h))) * D0
                    + (alpha_t * ((1.0 - torch.exp(-2.0 * h)) / (-2.0 * h) + 1.0)) * D1
                    + sigma_t * torch.sqrt(1.0 - torch.exp(-2 * h)) * noise
                )
        return x_t

    def multistep_dpm_solver_third_order_update(
        self, model_output_list: List[torch.Tensor], *args,
        sample: torch.Tensor = None, noise: Optional[torch.Tensor] = None, **kwargs,
    ) -> torch.Tensor:
        if sample is None:
            raise ValueError("missing `sample` as a required keyword argument")
        sigma_t, sigma_s0, sigma_s1, sigma_s2 = (
            self.sigmas[self.step_index + 1],
            self.sigmas[self.step_index],
            self.sigmas[self.step_index - 1],
            self.sigmas[self.step_index - 2],
        )
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t)
        alpha_s0, sigma_s0 = self._sigma_to_alpha_sigma_t(sigma_s0)
        alpha_s1, sigma_s1 = self._sigma_to_alpha_sigma_t(sigma_s1)
        alpha_s2, sigma_s2 = self._sigma_to_alpha_sigma_t(sigma_s2)
        lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
        lambda_s0 = torch.log(alpha_s0) - torch.log(sigma_s0)
        lambda_s1 = torch.log(alpha_s1) - torch.log(sigma_s1)
        lambda_s2 = torch.log(alpha_s2) - torch.log(sigma_s2)

        m0, m1, m2 = model_output_list[-1], model_output_list[-2], model_output_list[-3]
        h, h_0, h_1 = lambda_t - lambda_s0, lambda_s0 - lambda_s1, lambda_s1 - lambda_s2
        r0, r1 = h_0 / h, h_1 / h
        D0 = m0
        D1_0, D1_1 = (1.0 / r0) * (m0 - m1), (1.0 / r1) * (m1 - m2)
        D1 = D1_0 + (r0 / (r0 + r1)) * (D1_0 - D1_1)
        D2 = (1.0 / (r0 + r1)) * (D1_0 - D1_1)
        if self.config.algorithm_type == "dpmsolver++":
            x_t = (
                (sigma_t / sigma_s0) * sample
                - (alpha_t * (torch.exp(-h) - 1.0)) * D0
                + (alpha_t * ((torch.exp(-h) - 1.0) / h + 1.0)) * D1
                - (alpha_t * ((torch.exp(-h) - 1.0 + h) / h**2 - 0.5)) * D2
            )
        elif self.config.algorithm_type == "sde-dpmsolver++":
            assert noise is not None
            x_t = (
                (sigma_t / sigma_s0 * torch.exp(-h)) * sample
                + (alpha_t * (1.0 - torch.exp(-2.0 * h))) * D0
                + (alpha_t * ((1.0 - torch.exp(-2.0 * h)) / (-2.0 * h) + 1.0)) * D1
                + (alpha_t * ((1.0 - torch.exp(-2.0 * h) - 2.0 * h) / (2.0 * h) ** 2 - 0.5)) * D2
                + sigma_t * torch.sqrt(1.0 - torch.exp(-2 * h)) * noise
            )
        return x_t

    def index_for_timestep(self, timestep, schedule_timesteps=None):
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps
        indices = (schedule_timesteps == timestep).nonzero()
        pos = 1 if len(indices) > 1 else 0
        return indices[pos].item()

    def _init_step_index(self, timestep):
        if self.begin_index is None:
            if isinstance(timestep, torch.Tensor):
                timestep = timestep.to(self.timesteps.device)
            self._step_index = self.index_for_timestep(timestep)
        else:
            self._step_index = self._begin_index

    def step(
        self,
        model_output: torch.Tensor,
        timestep: Union[int, torch.Tensor],
        sample: torch.Tensor,
        generator=None,
        variance_noise: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[SchedulerOutput, Tuple]:
        """One multistep DPM-Solver++ update. Signature matches
        ``FlowUniPCMultistepScheduler.step`` (extra ``variance_noise`` kwarg for
        the SDE variant is optional)."""
        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', run 'set_timesteps' first."
            )
        if self.step_index is None:
            self._init_step_index(timestep)

        # Lower the order on the final step(s) for stability (and to land exactly
        # on x0 when the terminal sigma is 0).
        is_last = self.step_index == len(self.timesteps) - 1
        lower_order_final = is_last and (
            self.config.euler_at_final
            or (self.config.lower_order_final and len(self.timesteps) < 15)
            or self.config.final_sigmas_type == "zero"
        )
        lower_order_second = (
            (self.step_index == len(self.timesteps) - 2)
            and self.config.lower_order_final
            and len(self.timesteps) < 15
        )

        model_output = self.convert_model_output(model_output, sample=sample)
        for i in range(self.config.solver_order - 1):
            self.model_outputs[i] = self.model_outputs[i + 1]
        self.model_outputs[-1] = model_output

        sample = sample.to(torch.float32)
        if self.config.algorithm_type == "sde-dpmsolver++":
            if variance_noise is None:
                noise = torch.randn(
                    model_output.shape, generator=generator,
                    device=model_output.device, dtype=torch.float32,
                )
            else:
                noise = variance_noise.to(device=model_output.device, dtype=torch.float32)
        else:
            noise = None

        if self.config.solver_order == 1 or self.lower_order_nums < 1 or lower_order_final:
            prev_sample = self.dpm_solver_first_order_update(
                model_output, sample=sample, noise=noise
            )
        elif self.config.solver_order == 2 or self.lower_order_nums < 2 or lower_order_second:
            prev_sample = self.multistep_dpm_solver_second_order_update(
                self.model_outputs, sample=sample, noise=noise
            )
        else:
            prev_sample = self.multistep_dpm_solver_third_order_update(
                self.model_outputs, sample=sample, noise=noise
            )

        if self.lower_order_nums < self.config.solver_order:
            self.lower_order_nums += 1

        prev_sample = prev_sample.to(model_output.dtype)
        self._step_index += 1

        if not return_dict:
            return (prev_sample, model_output)
        return SchedulerOutput(prev_sample=prev_sample)

    def scale_model_input(self, sample: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return sample

    def __len__(self):
        return self.config.num_train_timesteps
