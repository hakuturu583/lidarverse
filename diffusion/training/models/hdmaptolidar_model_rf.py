# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rectified-flow variant of the Hdmap-to-Lidar model.

Replaces EDM training (SDE + preconditioning) with flow-matching velocity
prediction and uses FlowUniPCMultistepScheduler for inference.

Inherits all hdmap conditioning, VAE, FSDP, and data-loading logic from
``HdmapToLidarModel``.  To disable the EDM ``sigma_data`` latent scaling
set ``sigma_data: 1.0`` in the experiment config.
"""

from dataclasses import fields
from typing import Dict, Optional, Tuple

import torch
from einops import rearrange
from torch import Tensor, nn

from diffusion.conditioner import DataType
from diffusion.functional.batch_ops import batch_mul
from diffusion.training.models.hdmaptolidar_model import (
    HdmapToLidarModel,
    diffusion_fsdp_class_decorator,
)
from diffusion.training.datasets.lidar_dataset.forecast_warp import (
    build_lagen_sde_conditions,
    build_stage2_factorized_geometry,
    compose_normalized_range_channels,
    remove_points_inside_moving_boxes,
    warp_blockwise_history_with_spatial_map,
    warp_frozen_observed_history_with_spatial_map,
)
from diffusion.training.models.fm_solvers_unipc import FlowUniPCMultistepScheduler
from diffusion.training.models.fm_solvers_dpm import FlowDPMSolverMultistepScheduler
from diffusion.training.schedulers.rectified_flow import RectifiedFlow
from diffusion.training.utils.diffusion_forcing import (
    artifixer_flow_schedule,
    build_clean_context_timesteps,
    expand_chunkwise,
    temporal_chunk_slices,
    validate_prefix_mask,
)
from diffusion.training.utils.sf_dmd import (
    confidence_composite_geometry,
    confidence_weighted_geometry_source,
)
from tokenizer.training.datasets.lidar_datasets.waymo_ncore import (
    load_waymo_lidar_geometry,
)
from utils import log, misc


@torch.autocast("cuda", enabled=False)
def _sensor_tables_from_ray_directions(
    ray_dirs: Tensor,
    *,
    allow_approximate: bool = False,
) -> tuple[Tensor, Tensor, Tensor]:
    """Recover coarse elevation and azimuth tables from calibrated rays."""
    rays = torch.as_tensor(ray_dirs, dtype=torch.float32)
    if rays.ndim != 3 or rays.shape[-1] != 3:
        raise ValueError(
            f"ray_dirs must be (H,W,3), got {tuple(rays.shape)}"
        )
    norm = torch.linalg.vector_norm(rays, dim=-1, keepdim=True)
    if not torch.isfinite(norm).all() or bool((norm <= 0).any()):
        raise ValueError("ray_dirs must contain finite nonzero vectors")
    rays = rays / norm

    elevation_grid = torch.atan2(
        rays[..., 2],
        torch.linalg.vector_norm(rays[..., :2], dim=-1),
    )
    elevations = elevation_grid.mean(dim=1)
    azimuth_grid = torch.atan2(rays[..., 1], rays[..., 0])
    azimuths = azimuth_grid[0].contiguous()
    relative = torch.remainder(
        azimuth_grid - azimuths[None] + torch.pi,
        2.0 * torch.pi,
    ) - torch.pi
    row_offsets = torch.atan2(
        torch.sin(relative).mean(dim=1),
        torch.cos(relative).mean(dim=1),
    )

    reconstructed = torch.stack(
        (
            torch.cos(elevations)[:, None]
            * torch.cos(azimuths[None] + row_offsets[:, None]),
            torch.cos(elevations)[:, None]
            * torch.sin(azimuths[None] + row_offsets[:, None]),
            torch.sin(elevations)[:, None].expand(-1, rays.shape[1]),
        ),
        dim=-1,
    )
    max_error = torch.linalg.vector_norm(
        reconstructed - rays, dim=-1
    ).max()
    if not torch.isfinite(max_error):
        raise ValueError("reconstructed sensor rays must be finite")
    if float(max_error) > 1e-3 and not allow_approximate:
        raise ValueError(
            "sensor rays are not representable by a spinning-LiDAR row/column "
            f"grid (max direction error={float(max_error):.6g})"
        )
    return elevations, azimuths, row_offsets


def _sample_world_synchronized_bernoulli(
    probability: float,
    device: torch.device,
) -> Tensor:
    """Sample one branch decision shared by every distributed rank."""
    draw = torch.empty((), device=device, dtype=torch.float32)
    distributed = (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
    )
    if not distributed or torch.distributed.get_rank() == 0:
        draw.uniform_()
    if distributed:
        torch.distributed.broadcast(draw, src=0)
    return draw < probability


def _sample_world_synchronized_categorical(
    probabilities: Tensor,
) -> int:
    """Sample one categorical branch shared by every distributed rank."""
    if probabilities.ndim != 1 or probabilities.numel() == 0:
        raise ValueError("probabilities must be a non-empty 1-D tensor")
    distributed = (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
    )
    if not distributed or torch.distributed.get_rank() == 0:
        sampled = torch.multinomial(probabilities, 1)
    else:
        sampled = torch.zeros(
            1, device=probabilities.device, dtype=torch.long
        )
    if distributed:
        torch.distributed.broadcast(sampled, src=0)
    return int(sampled.item())


class _RelativePoseFiLM(nn.Module):
    """LaGen relative-pose FiLM for the previous-frame latent."""

    def __init__(self, latent_channels: int, hidden_channels: int = 256):
        super().__init__()
        self.latent_channels = int(latent_channels)
        self.net = nn.Sequential(
            nn.Linear(16, int(hidden_channels)),
            nn.SiLU(),
            nn.Linear(int(hidden_channels), 2 * self.latent_channels),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, latent: Tensor, relative_pose: Tensor) -> Tensor:
        if relative_pose.shape != (latent.shape[0], 4, 4):
            raise ValueError(
                "relative pose must be (B,4,4), got "
                f"{tuple(relative_pose.shape)}"
            )
        pose = relative_pose.flatten(1).to(
            device=latent.device,
            dtype=self.net[0].weight.dtype,
        )
        scale, shift = self.net(pose).chunk(2, dim=1)
        shape = (latent.shape[0], self.latent_channels) + (1,) * (
            latent.ndim - 2
        )
        return latent * (1.0 + scale.view(shape)) + shift.view(shape)


class HdmapToLidarRectifiedFlowModel(HdmapToLidarModel):
    """Rectified-flow version of HdmapToLidarModel.

    Training: velocity-matching loss on the RF interpolation.
    Inference: FlowUniPCMultistepScheduler (multi-step ODE solver).
    """

    def __init__(self, config):
        super().__init__(config)
        self.tensor_kwargs_fp32 = {"device": "cuda", "dtype": torch.float32}

        shift = getattr(config, "shift", 5)
        clamp_p = getattr(config, "shift_clamp_percentiles", None)
        if clamp_p is not None:
            clamp_p = tuple(clamp_p)
        stretch_terminal = (
            float(getattr(config, "shift_stretch_terminal", 0.1))
            if getattr(config, "enable_shift_stretch", False)
            else None
        )
        self.rectified_flow = RectifiedFlow(
            velocity_field=self.net,
            train_time_distribution=getattr(config, "train_time_distribution", "logitnormal"),
            train_time_weight_method=getattr(config, "train_time_weight", "uniform"),
            shift=shift,
            shift_clamp_percentiles=clamp_p,
            shift_stretch_terminal=stretch_terminal,
            device=torch.device("cuda"),
            dtype=torch.float32,
        )
        self.sample_scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=1000, shift=1, use_dynamic_shifting=False,
        )
        self._rf_shift = float(shift)
        # RF prediction parameterization: "velocity" (legacy) regresses
        # v = noise - clean; "clean" (VGGT-World z-prediction) regresses the
        # clean latent x0 directly and converts to velocity for the sampler.
        self.rf_prediction_type = getattr(config, "rf_prediction_type", "velocity")
        assert self.rf_prediction_type in ("velocity", "clean"), (
            f"rf_prediction_type must be 'velocity' or 'clean', got {self.rf_prediction_type}"
        )
        # When ``generate_samples_from_batch`` is called with
        # ``early_stop_t > 0`` (LTX-style decoder hand-off), the actual
        # exit sigma -- which may not exactly equal early_stop_t because
        # the UniPC schedule is discrete -- is recorded here so callers
        # can pass it to ``model.decode(z, t=model.last_sampling_exit_t)``.
        # Always in [0, 1]. None when no early stop was applied.
        self.last_sampling_exit_t: Optional[float] = None
        log.info(
            f"HdmapToLidarRectifiedFlowModel: shift={shift}, "
            f"shift_clamp_percentiles={clamp_p}, "
            f"enable_shift_stretch={getattr(config, 'enable_shift_stretch', False)}, "
            f"shift_stretch_terminal={self.rectified_flow.shift_stretch_terminal}, "
            f"sigma_data={self.sigma_data}, "
            f"rf_prediction_type={self.rf_prediction_type}"
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def training_step(
        self, data_batch: dict[str, Tensor], iteration: int
    ) -> tuple[dict[str, Tensor], Tensor]:
        return self._rectified_flow_forward(data_batch)

    def _rectified_flow_forward(
        self, data_batch: dict[str, Tensor]
    ) -> tuple[dict[str, Tensor], Tensor]:
        raw_state, x0, condition = self.get_data_and_condition(data_batch)

        B = x0.shape[0]
        epsilon = torch.randn(x0.size(), **self.tensor_kwargs_fp32)

        t_B = self.rectified_flow.sample_train_time(B).to(**self.tensor_kwargs_fp32)
        t_B = rearrange(t_B, "b -> b 1")

        timesteps = self.rectified_flow.get_discrete_timestamp(t_B, self.tensor_kwargs_fp32)
        sigmas = self.rectified_flow.get_sigmas(timesteps, self.tensor_kwargs_fp32)
        timesteps = rearrange(timesteps, "b -> b 1")
        sigmas = rearrange(sigmas, "b -> b 1")

        xt, vt_target = self.rectified_flow.get_interpolation(epsilon, x0, sigmas)

        net_pred = self._rf_network_forward(
            xt.to(**self.tensor_kwargs), timesteps, condition,
        ).float()

        # "clean": regress x0 directly (VGGT-World z-prediction); uniform-weighted
        # clean-target MSE (VGGT eq. 12). "velocity": regress v = noise - clean.
        if self.rf_prediction_type == "clean":
            target = x0.float()
        else:
            target = vt_target.float()

        time_weights = self.rectified_flow.train_time_weight(
            timesteps, self.tensor_kwargs_fp32
        )
        per_instance_loss = torch.mean(
            (net_pred - target) ** 2,
            dim=list(range(1, net_pred.dim())),
        )
        loss = torch.mean(time_weights.squeeze(-1) * per_instance_loss)

        output_batch = {
            "x0": x0,
            "xt": xt,
            "sigma": sigmas,
            "model_pred": net_pred,
            "edm_loss": loss,
            "timesteps": timesteps,
            "per_instance_loss": per_instance_loss,
        }
        return output_batch, loss

    # ------------------------------------------------------------------
    # Network forward (no EDM preconditioning)
    # ------------------------------------------------------------------

    def _rf_network_forward(
        self,
        xt: Tensor,
        timesteps: Tensor,
        condition,
    ) -> Tensor:
        """Run the network without EDM c_in/c_out/c_skip preconditioning.

        Handles hdmap channel-concat and optional dummy temporal dim,
        mirroring the parent ``denoise`` method structure.
        """
        if getattr(self.config, "use_dummy_temporal_dim", False):
            xt = xt.unsqueeze(2)

        net_input = xt

        if self._current_hdmap is not None:
            hdmap = self._current_hdmap.to(net_input.dtype)
            if net_input.shape[0] != hdmap.shape[0]:
                repeats = net_input.shape[0] // hdmap.shape[0]
                hdmap = hdmap.repeat(repeats, *([1] * (hdmap.ndim - 1)))
            # Unconditional CFG pass: replace hdmap with zeros (the "null"
            # condition the model saw during training with hdmap dropout).
            force_null_mask = getattr(
                self, "_hdmap_force_null_mask", None
            )
            if force_null_mask is not None:
                if (
                    force_null_mask.ndim != 1
                    or force_null_mask.shape[0] != hdmap.shape[0]
                ):
                    raise ValueError(
                        "HDMap CFG null mask must have one entry per batch "
                        f"element: mask={tuple(force_null_mask.shape)}, "
                        f"hdmap={tuple(hdmap.shape)}"
                    )
                mask_shape = (hdmap.shape[0],) + (1,) * (
                    hdmap.ndim - 1
                )
                hdmap = hdmap.masked_fill(
                    force_null_mask.to(
                        device=hdmap.device, dtype=torch.bool
                    ).view(mask_shape),
                    0,
                )
            elif self._hdmap_force_null:
                hdmap = torch.zeros_like(hdmap)
            net_input = torch.cat([net_input, hdmap], dim=1)

        geometry_concat = bool(
            getattr(self, "blockwise_warped_history_concat", False)
        )
        if geometry_concat:
            geometry = getattr(
                self, "_current_blockwise_warped_history", None
            )
            if geometry is None:
                raise RuntimeError(
                    "geometry concat requires an encoded warped-history latent"
                )
            geometry = geometry.to(
                device=net_input.device,
                dtype=net_input.dtype,
            )
            if net_input.shape[0] != geometry.shape[0]:
                repeats = net_input.shape[0] // geometry.shape[0]
                geometry = geometry.repeat(
                    repeats, *([1] * (geometry.ndim - 1))
                )
            force_null_mask = getattr(
                self, "_geometry_force_null_mask", None
            )
            if force_null_mask is not None:
                if (
                    force_null_mask.ndim != 1
                    or force_null_mask.shape[0] != geometry.shape[0]
                ):
                    raise ValueError(
                        "geometry CFG null mask must have one entry per batch "
                        f"element: mask={tuple(force_null_mask.shape)}, "
                        f"geometry={tuple(geometry.shape)}"
                    )
                mask_shape = (geometry.shape[0],) + (1,) * (
                    geometry.ndim - 1
                )
                geometry = geometry.masked_fill(
                    force_null_mask.to(
                        device=geometry.device, dtype=torch.bool
                    ).view(mask_shape),
                    0,
                )
            elif getattr(self, "_geometry_force_null", False):
                geometry = torch.zeros_like(geometry)
            if geometry.shape[2:] != net_input.shape[2:]:
                raise ValueError(
                    "geometry concat must align with the DiT input: "
                    f"geometry={tuple(geometry.shape)}, "
                    f"input={tuple(net_input.shape)}"
                )
            net_input = torch.cat([net_input, geometry], dim=1)

        autoregressive_condition = getattr(
            self, "_current_autoregressive_condition", None
        )
        if autoregressive_condition is not None:
            autoregressive_condition = autoregressive_condition.to(
                device=net_input.device, dtype=net_input.dtype
            )
            if net_input.shape[0] != autoregressive_condition.shape[0]:
                repeats = (
                    net_input.shape[0]
                    // autoregressive_condition.shape[0]
                )
                autoregressive_condition = autoregressive_condition.repeat(
                    repeats,
                    *([1] * (autoregressive_condition.ndim - 1)),
                )
            if autoregressive_condition.shape[2:] != net_input.shape[2:]:
                raise ValueError(
                    "autoregressive condition must align with DiT input: "
                    f"condition={tuple(autoregressive_condition.shape)}, "
                    f"input={tuple(net_input.shape)}"
                )
            net_input = torch.cat(
                [net_input, autoregressive_condition], dim=1
            )

        known_mask_concat = bool(
            getattr(self.config, "stage2_known_mask_concat", False)
            or getattr(
                self.config,
                "stage2_multitask_known_mask_concat",
                False,
            )
        )
        if known_mask_concat:
            if net_input.ndim != 5:
                raise ValueError(
                    "known-frame mask concat requires video-shaped DiT input"
                )
            known_mask = getattr(
                self, "_current_stage2_known_frame_mask", None
            )
            if known_mask is None:
                known_mask = torch.zeros(
                    (net_input.shape[0], net_input.shape[2]),
                    device=net_input.device,
                    dtype=net_input.dtype,
                )
            else:
                known_mask = known_mask.to(
                    device=net_input.device, dtype=net_input.dtype
                )
                if known_mask.ndim == 1:
                    known_mask = known_mask[None]
                if known_mask.ndim != 2:
                    raise ValueError(
                        "known-frame mask must be (T,) or (B,T), got "
                        f"{tuple(known_mask.shape)}"
                    )
                if known_mask.shape[0] != net_input.shape[0]:
                    repeats = net_input.shape[0] // known_mask.shape[0]
                    known_mask = known_mask.repeat(repeats, 1)
            if known_mask.shape != (
                net_input.shape[0],
                net_input.shape[2],
            ):
                raise ValueError(
                    "known-frame mask must align with the DiT input, got "
                    f"{tuple(known_mask.shape)} for "
                    f"{tuple(net_input.shape)}"
                )
            known_channel = known_mask[:, None, :, None, None].expand(
                -1, 1, -1, net_input.shape[3], net_input.shape[4]
            )
            net_input = torch.cat([net_input, known_channel], dim=1)

        # Legacy RF passes (B,1) and the DiT accepts (B,). Diffusion Forcing
        # passes a real per-frame (B,T) tensor which must remain 2-D.
        t_for_net = (
            timesteps.squeeze(-1)
            if timesteps.ndim == 2 and timesteps.shape[1] == 1
            else timesteps
        )
        t_for_net = t_for_net.to(net_input.dtype)

        net_kwargs = condition.to_dict()
        side_condition = getattr(
            self, "_current_blockwise_warped_history", None
        )
        if side_condition is not None and not geometry_concat:
            side_condition = side_condition.to(
                device=net_input.device,
                dtype=net_input.dtype,
            )
            if net_input.shape[0] != side_condition.shape[0]:
                repeats = net_input.shape[0] // side_condition.shape[0]
                side_condition = side_condition.repeat(
                    repeats, *([1] * (side_condition.ndim - 1))
                )
            net_kwargs["side_condition"] = side_condition

        teacher_forcing_clean_frames = getattr(
            self, "_current_teacher_forcing_clean_frames", None
        )
        if teacher_forcing_clean_frames is not None:
            if getattr(self, "_current_sfdmd_temporal_positions", None) is not None:
                raise RuntimeError(
                    "one-pass teacher forcing and SF-DMD streaming cannot be "
                    "active in the same network call"
                )
            net_kwargs["teacher_forcing_clean_frames"] = int(
                teacher_forcing_clean_frames
            )
            net_kwargs["temporal_positions"] = getattr(
                self, "_current_teacher_forcing_temporal_positions"
            )
        frame_type_ids = getattr(
            self, "_current_stage2_task_frame_type_ids", None
        )
        if frame_type_ids is not None:
            if frame_type_ids.shape != (net_input.shape[2],):
                raise ValueError(
                    "Stage-2 task ids must align with the DiT temporal axis, "
                    f"got {tuple(frame_type_ids.shape)} for "
                    f"T={net_input.shape[2]}"
                )
            net_kwargs["frame_type_ids"] = frame_type_ids

        dataset_ids = getattr(
            self, "_current_sensor_dataset_ids", None
        )
        if dataset_ids is not None:
            dataset_ids = dataset_ids.to(
                device=net_input.device, dtype=torch.long
            )
            if dataset_ids.shape[0] != net_input.shape[0]:
                if net_input.shape[0] % dataset_ids.shape[0] != 0:
                    raise ValueError(
                        "dataset ids cannot be repeated to match the DiT batch: "
                        f"ids={tuple(dataset_ids.shape)}, B={net_input.shape[0]}"
                    )
                dataset_ids = dataset_ids.repeat(
                    net_input.shape[0] // dataset_ids.shape[0]
                )
            net_kwargs["dataset_ids_B"] = dataset_ids

        # SF-DMD's non-overlapping autoregressive path supplies global RoPE
        # positions and an external per-layer self-attention KV cache. These
        # attributes are absent in every ordinary training/inference path.
        temporal_positions = getattr(
            self, "_current_sfdmd_temporal_positions", None
        )
        return_self_attn_kv = False
        if temporal_positions is not None:
            net_kwargs["temporal_positions"] = temporal_positions
            net_kwargs["self_attn_kv_cache"] = (
                self._current_sfdmd_self_attn_kv_cache
            )
            net_kwargs["store_self_attn_kv"] = bool(
                self._current_sfdmd_store_self_attn_kv
            )
            net_kwargs["self_attn_kv_max_frames"] = int(
                self._current_sfdmd_self_attn_kv_max_frames
            )
            net_kwargs["self_attn_kv_sink_frames"] = int(
                getattr(self, "_current_sfdmd_self_attn_kv_sink_frames", 0)
            )
            net_kwargs["detach_self_attn_kv"] = bool(
                getattr(self, "_current_sfdmd_detach_self_attn_kv", True)
            )
            return_self_attn_kv = bool(
                getattr(
                    self,
                    "_current_sfdmd_return_self_attn_kv",
                    False,
                )
            )
            net_kwargs["return_self_attn_kv"] = return_self_attn_kv

        network = self.net
        if getattr(network, "skip_crossattn_when_zero", False):
            network.validate_zero_crossattn_skip(
                net_kwargs["crossattn_emb"],
                net_kwargs.get("img_context_emb"),
            )

        net_output = network(
            x=net_input,
            timesteps=t_for_net,
            **net_kwargs,
        )
        if return_self_attn_kv:
            net_output, updated_cache = net_output
            self._current_sfdmd_updated_self_attn_kv_cache = updated_cache

        if getattr(self.config, "use_dummy_temporal_dim", False):
            net_output = net_output.squeeze(2)

        return net_output.float()

    @staticmethod
    def _batch_cfg_conditions(condition, uncondition):
        """Concatenate matching condition dataclasses along batch."""
        if type(condition) is not type(uncondition):
            raise TypeError(
                "batched CFG requires matching condition types, got "
                f"{type(condition).__name__} and "
                f"{type(uncondition).__name__}"
            )

        values = {}
        for field in fields(condition):
            cond_value = getattr(condition, field.name)
            uncond_value = getattr(uncondition, field.name)
            if isinstance(cond_value, torch.Tensor):
                if not isinstance(uncond_value, torch.Tensor):
                    raise TypeError(
                        f"CFG condition field {field.name!r} is a tensor "
                        "only on the conditional branch"
                    )
                if cond_value.ndim == 0 or uncond_value.ndim == 0:
                    raise ValueError(
                        f"CFG condition field {field.name!r} must have a "
                        "batch dimension"
                    )
                if cond_value.shape[1:] != uncond_value.shape[1:]:
                    raise ValueError(
                        f"CFG condition field {field.name!r} has "
                        f"incompatible shapes {tuple(cond_value.shape)} and "
                        f"{tuple(uncond_value.shape)}"
                    )
                values[field.name] = torch.cat(
                    [cond_value, uncond_value], dim=0
                )
            elif cond_value is uncond_value or cond_value == uncond_value:
                values[field.name] = cond_value
            else:
                raise ValueError(
                    f"CFG condition field {field.name!r} differs between "
                    "branches but is not batchable"
                )
        return type(condition)(**values)

    def _cfg_null_modalities(self) -> tuple[bool, bool]:
        """Return whether the CFG unconditional branch nulls HDMap/geometry."""
        geometry_only_cfg = bool(
            getattr(self, "_geometry_only_cfg", False)
        )
        hdmap_only_cfg = bool(
            getattr(self, "_hdmap_only_cfg", False)
        )
        if geometry_only_cfg and hdmap_only_cfg:
            raise ValueError(
                "geometry-only and HDMap-only CFG are mutually exclusive"
            )
        drop_hdmap_uncond = hdmap_only_cfg or (
            self.hdmap_dropout_prob > 0.0 and not geometry_only_cfg
        )
        return drop_hdmap_uncond, geometry_only_cfg

    def _rf_network_forward_batched_cfg(
        self,
        xt: Tensor,
        timesteps: Tensor,
        condition,
        uncondition,
    ) -> tuple[Tensor, Tensor]:
        """Run conditional and unconditional CFG branches as one batch."""
        batch_size = xt.shape[0]
        batched_condition = self._batch_cfg_conditions(
            condition, uncondition
        )
        batched_xt = torch.cat([xt, xt], dim=0)
        batched_timesteps = torch.cat([timesteps, timesteps], dim=0)

        previous_mask = getattr(
            self, "_hdmap_force_null_mask", None
        )
        (
            drop_hdmap_uncond,
            drop_geometry_uncond,
        ) = self._cfg_null_modalities()
        branch_null_mask = torch.zeros(
            2 * batch_size,
            device=xt.device,
            dtype=torch.bool,
        )
        if drop_hdmap_uncond:
            branch_null_mask[batch_size:] = True
        previous_geometry_mask = getattr(
            self, "_geometry_force_null_mask", None
        )
        geometry_branch_null_mask = torch.zeros_like(branch_null_mask)
        if drop_geometry_uncond:
            geometry_branch_null_mask[batch_size:] = True
        self._hdmap_force_null_mask = branch_null_mask
        self._geometry_force_null_mask = geometry_branch_null_mask
        try:
            batched_out = self._rf_network_forward(
                batched_xt,
                batched_timesteps,
                batched_condition,
            )
        finally:
            if previous_mask is None:
                del self._hdmap_force_null_mask
            else:
                self._hdmap_force_null_mask = previous_mask
            if previous_geometry_mask is None:
                del self._geometry_force_null_mask
            else:
                self._geometry_force_null_mask = previous_geometry_mask

        if batched_out.shape[0] != 2 * batch_size:
            raise RuntimeError(
                "batched CFG network returned an unexpected batch size: "
                f"expected {2 * batch_size}, got {batched_out.shape[0]}"
            )
        return batched_out[:batch_size], batched_out[batch_size:]

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _clean_to_velocity(
        self, x0_hat: Tensor, x_t: Tensor, t_input: Tensor,
    ) -> Tensor:
        """Convert a clean-latent prediction to a flow velocity for the sampler.

        With our interpolation ``x_t = noise*sigma + clean*(1-sigma)`` the
        velocity is ``v = noise - clean = (x_t - clean)/sigma``. ``t_input`` is
        the discrete timestep in ``[0, num_train]``; ``sigma = t_input/num_train``.
        """
        num_train = float(self.rectified_flow.num_train_timesteps)
        sigma = (t_input.float() / num_train).clamp(min=1e-3)
        sigma = sigma.view(-1, *([1] * (x0_hat.dim() - 1)))
        return (x_t.float() - x0_hat.float()) / sigma

    def _get_velocity(
        self, latents: Tensor, t_input: Tensor, condition, uncondition,
        guidance: float, use_cfg: bool, batched_cfg: bool = False,
    ) -> Tensor:
        """Single velocity query with optional CFG.

        In ``clean`` mode the network output is the clean latent x0; CFG is
        applied on that (affine, so equivalent to CFG on velocity) and the
        combined prediction is converted to velocity for the ODE sampler.

        ``guidance == 0`` is pure unconditional sampling, so skip the
        conditional query entirely. This is exactly equivalent to
        ``uncond + 0 * (cond - uncond)`` but halves the network forwards.
        """
        net_latents = latents.to(**self.tensor_kwargs)
        if use_cfg and batched_cfg:
            cond_out, uncond_out = self._rf_network_forward_batched_cfg(
                net_latents,
                t_input,
                condition,
                uncondition,
            )
            out = uncond_out + guidance * (cond_out - uncond_out)
        elif use_cfg and guidance == 0.0:
            (
                drop_hdmap_uncond,
                drop_geometry_uncond,
            ) = self._cfg_null_modalities()
            if drop_hdmap_uncond:
                self._hdmap_force_null = True
            if drop_geometry_uncond:
                self._geometry_force_null = True
            try:
                out = self._rf_network_forward(
                    net_latents, t_input, uncondition,
                )
            finally:
                self._hdmap_force_null = False
                self._geometry_force_null = False
        else:
            cond_out = self._rf_network_forward(
                net_latents, t_input, condition,
            )
            if use_cfg:
                (
                    drop_hdmap_uncond,
                    drop_geometry_uncond,
                ) = self._cfg_null_modalities()
                if drop_hdmap_uncond:
                    self._hdmap_force_null = True
                if drop_geometry_uncond:
                    self._geometry_force_null = True
                try:
                    uncond_out = self._rf_network_forward(
                        net_latents, t_input, uncondition,
                    )
                finally:
                    self._hdmap_force_null = False
                    self._geometry_force_null = False
                out = uncond_out + guidance * (cond_out - uncond_out)
            else:
                out = cond_out

        if self.rf_prediction_type == "clean":
            out = self._clean_to_velocity(out, latents, t_input)
        return out

    @torch.no_grad()
    def generate_samples_from_batch(
        self,
        data_batch: Dict,
        guidance: float = 1.0,
        seed: int = 1,
        state_shape: Optional[Tuple] = None,
        n_sample: Optional[int] = None,
        is_negative_prompt: bool = False,
        num_steps: int = 35,
        shift: Optional[float] = None,
        sampler: str = "unipc",
        sde_eta: float = 1.0,
        sde_eta_first_frac: float = 0.0,
        sde_eta_last_frac: float = 0.0,
        solver_order: int = 2,
        early_stop_t: Optional[float] = None,
        noise: Optional[Tensor] = None,
        known_latents: Optional[Tensor] = None,
        known_mask: Optional[Tensor] = None,
        known_latents_mode: Optional[str] = None,
        repaint_u: int = 1,
        store_kv_exit_query: bool = False,
        store_kv_query_sigma: Optional[float] = None,
        **kwargs,
    ) -> Tensor:
        """Generate latents from noise.

        Args:
            ``noise``: optional pre-drawn init noise ``(B,) + state_shape``.
                When given it overrides the internally seeded
                ``arch_invariant_rand`` -- used by the long-video rollout so
                overlapping windows share noise on their shared frames.
            ``early_stop_t``: if not None, must be a float in (0, 1].
                The UniPC sampler stops as soon as the next-step sigma
                drops to or below this value -- intended for hand-off to
                an LTX-style denoising decoder that handles the final
                ``[exit_sigma, 0]`` step in one forward pass. The actual
                exit sigma (always <= ``early_stop_t``) is stored on
                ``self.last_sampling_exit_t``; callers should pass that
                exact value to ``model.decode(z, t=...)`` so the decoder
                sees the same noise level it was trained on.
        """
        use_cfg = guidance != 1.0

        if use_cfg:
            if is_negative_prompt:
                condition, uncondition = self.conditioner.get_condition_with_negative_prompt(data_batch)
            else:
                condition, uncondition = self.conditioner.get_condition_uncondition(data_batch)
        else:
            condition = self.conditioner(data_batch)
            uncondition = None

        B = n_sample or data_batch[self.input_data_key].shape[0]
        state_shape = state_shape or self.state_shape

        if noise is not None:
            noise = noise.to(device="cuda", dtype=torch.float32)
            assert tuple(noise.shape) == (B,) + tuple(state_shape), (
                f"noise {tuple(noise.shape)} must be (B={B},) + "
                f"state_shape={tuple(state_shape)}"
            )
        else:
            noise = misc.arch_invariant_rand(
                (B,) + tuple(state_shape), torch.float32, "cuda", seed,
            )

        # Reset early-stop bookkeeping before each generate call so the
        # caller can't accidentally re-use a stale exit sigma from a
        # previous (early-stopped) generation when this one ran to
        # completion.
        self.last_sampling_exit_t = None

        if known_latents is not None and sampler != "sde":
            raise NotImplementedError(
                "known_latents (RePaint-style overlap conditioning) is only "
                "supported by the sde sampler."
            )
        if store_kv_exit_query and sampler not in ("sde", "unipc"):
            raise NotImplementedError(
                "selected-query K/V commitment is only supported by the SDE "
                "and UniPC samplers"
            )

        if sampler == "sde":
            sde_kwargs = dict(
                latents=noise,
                condition=condition,
                uncondition=uncondition,
                guidance=guidance,
                use_cfg=use_cfg,
                B=B,
                num_steps=num_steps,
                shift=shift,
                seed=seed,
                sde_eta=sde_eta,
                early_stop_t=early_stop_t,
                known_latents=known_latents, known_mask=known_mask,
                repaint_u=repaint_u,
                eta_first_frac=sde_eta_first_frac,
                eta_last_frac=sde_eta_last_frac,
                store_kv_exit_query=store_kv_exit_query,
                store_kv_query_sigma=store_kv_query_sigma,
            )
            if known_latents_mode is not None:
                sde_kwargs["known_latents_mode"] = known_latents_mode
            return self._generate_sde(**sde_kwargs)

        if sampler in ("euler", "heun"):
            if early_stop_t is not None:
                raise NotImplementedError(
                    "early_stop_t is only supported by the unipc/sde samplers "
                    "(euler/heun use a fixed uniform linspace and don't "
                    "have the same early-stop semantics)."
                )
            return self._generate_euler_heun(
                noise, condition, uncondition, guidance, use_cfg,
                B, num_steps, sampler,
            )

        if sampler in ("dpmpp", "dpmpp_sde"):
            if early_stop_t is not None:
                raise NotImplementedError(
                    "early_stop_t is not yet supported by the dpmpp samplers."
                )
            return self._generate_dpmpp(
                noise, condition, uncondition, guidance, use_cfg,
                B, num_steps, shift, seed,
                algorithm_type=(
                    "sde-dpmsolver++" if sampler == "dpmpp_sde" else "dpmsolver++"
                ),
                solver_order=solver_order,
            )

        return self._generate_unipc(
            noise, condition, uncondition, guidance, use_cfg,
            B, num_steps, shift, seed,
            early_stop_t=early_stop_t,
            store_kv_exit_query=store_kv_exit_query,
            store_kv_query_sigma=store_kv_query_sigma,
        )

    # ---- UniPC sampler ----

    @torch.no_grad()
    def _generate_unipc(
        self, latents, condition, uncondition, guidance, use_cfg,
        B, num_steps, shift, seed,
        early_stop_t: Optional[float] = None,
        batched_cfg: bool = False,
        store_kv_exit_query: bool = False,
        store_kv_query_sigma: Optional[float] = None,
    ) -> Tensor:
        gen = torch.Generator(device="cuda")
        gen.manual_seed(seed)

        if shift is None:
            shift = self._rf_shift
        stretch_terminal = self.rectified_flow.shift_stretch_terminal
        self.sample_scheduler.set_timesteps(
            num_steps, device="cuda", shift=shift,
            stretch_terminal=stretch_terminal,
        )
        self.sample_scheduler.set_begin_index(0)
        timesteps = self.sample_scheduler.timesteps
        sigmas = self.sample_scheduler.sigmas.float()
        num_train = float(self.sample_scheduler.config.num_train_timesteps)

        if store_kv_query_sigma is not None and not store_kv_exit_query:
            raise ValueError(
                "store_kv_query_sigma requires store_kv_exit_query=True"
            )
        deferred_store_index = None
        deferred_store_query = None
        debug_nonfinite = bool(
            getattr(self, "_debug_nonfinite_sampling", False)
        )

        def _tensor_stats(value: torch.Tensor) -> str:
            value_f = value.detach().float()
            finite = value_f[torch.isfinite(value_f)]
            if finite.numel() == 0:
                return "finite=0.000000%, no finite values"
            return (
                f"finite={float(torch.isfinite(value_f).float().mean()):.6%}, "
                f"min={float(finite.min()):.6g}, "
                f"max={float(finite.max()):.6g}, "
                f"mean={float(finite.mean()):.6g}, "
                f"std={float(finite.std()):.6g}"
            )

        if store_kv_query_sigma is not None:
            target_sigma = torch.as_tensor(
                store_kv_query_sigma,
                device=sigmas.device,
                dtype=sigmas.dtype,
            )
            deferred_store_index = int(
                torch.argmin((sigmas[:-1] - target_sigma).abs()).item()
            )

        for i, t in enumerate(timesteps):
            t_input = t.unsqueeze(0).expand(B)
            if deferred_store_index == i:
                deferred_store_query = (
                    latents.detach().clone(),
                    t_input.detach().clone(),
                )
            store_exit = (
                store_kv_exit_query
                and deferred_store_index is None
                and i == len(timesteps) - 1
            )
            previous_store = getattr(
                self, "_current_sfdmd_store_self_attn_kv", False
            )
            if store_exit:
                self._current_sfdmd_store_self_attn_kv = True
            try:
                velocity_pred = self._get_velocity(
                    latents,
                    t_input,
                    condition,
                    uncondition,
                    guidance,
                    use_cfg,
                    batched_cfg=batched_cfg,
                )
            finally:
                if store_exit:
                    self._current_sfdmd_store_self_attn_kv = previous_store
            if debug_nonfinite and not torch.isfinite(velocity_pred).all():
                raise FloatingPointError(
                    "UniPC received a non-finite velocity prediction at "
                    f"step {i + 1}/{len(timesteps)}, timestep={float(t):.6g}; "
                    f"latents: {_tensor_stats(latents)}; "
                    f"velocity: {_tensor_stats(velocity_pred)}"
                )
            result = self.sample_scheduler.step(
                velocity_pred.unsqueeze(0), t, latents.unsqueeze(0),
                return_dict=False, generator=gen,
            )
            latents = result[0].squeeze(0)
            if debug_nonfinite and not torch.isfinite(latents).all():
                raise FloatingPointError(
                    "UniPC produced non-finite latents after "
                    f"step {i + 1}/{len(timesteps)}, timestep={float(t):.6g}; "
                    f"velocity: {_tensor_stats(velocity_pred)}; "
                    f"latents: {_tensor_stats(latents)}"
                )

            # After this step the latent's noise level is approximately
            # the next schedule timestep (or 0 if this was the last step).
            # If that level is at or below ``early_stop_t`` we stop here
            # and hand off to the LTX decoder; the EXACT noise level the
            # latent is at is recorded so the decoder gets the matching
            # ``t`` rather than the user-requested target.
            if early_stop_t is not None:
                next_t = (
                    timesteps[i + 1].item()
                    if i + 1 < len(timesteps)
                    else 0.0
                )
                next_t_norm = float(next_t) / num_train
                if next_t_norm <= float(early_stop_t):
                    self.last_sampling_exit_t = next_t_norm
                    log.info(
                        f"_generate_unipc: early-stopping at step {i + 1}/"
                        f"{len(timesteps)} (sigma={next_t_norm:.4f} <= "
                        f"early_stop_t={early_stop_t})"
                    )
                    break

        if deferred_store_index is not None:
            if deferred_store_query is None:
                raise RuntimeError(
                    "UniPC sampling ended before the requested K/V query sigma"
                )
            query_latents, query_timestep = deferred_store_query
            previous_store = getattr(
                self, "_current_sfdmd_store_self_attn_kv", False
            )
            self._current_sfdmd_store_self_attn_kv = True
            try:
                self._get_velocity(
                    query_latents,
                    query_timestep,
                    condition,
                    uncondition,
                    guidance,
                    use_cfg,
                    batched_cfg=batched_cfg,
                )
            finally:
                self._current_sfdmd_store_self_attn_kv = previous_store

        return latents

    # ---- Flow-DPM-Solver++ sampler ----

    @torch.no_grad()
    def _generate_dpmpp(
        self, latents, condition, uncondition, guidance, use_cfg,
        B, num_steps, shift, seed,
        algorithm_type: str = "dpmsolver++",
        solver_order: int = 2,
    ) -> Tensor:
        """Flow-DPM-Solver++ (DPM-Solver++ specialized for rectified flow).

        High-order data-prediction exponential integrator. ``algorithm_type``
        selects the deterministic ODE (``"dpmsolver++"``) or the stochastic
        variant (``"sde-dpmsolver++"``, fresh noise injected each step). Reuses
        the SAME shift/stretch sigma ladder as UniPC/SDE. ``solver_order`` 1-3
        (2 recommended for guided sampling). Lower-order-final is on for
        stability at small step counts.
        """
        gen = torch.Generator(device="cuda")
        gen.manual_seed(seed)

        if shift is None:
            shift = self._rf_shift
        stretch_terminal = self.rectified_flow.shift_stretch_terminal

        scheduler = FlowDPMSolverMultistepScheduler(
            num_train_timesteps=1000,
            shift=1,
            use_dynamic_shifting=False,
            solver_order=solver_order,
            algorithm_type=algorithm_type,
            solver_type="midpoint",
            lower_order_final=True,
            final_sigmas_type="zero",
        )
        scheduler.set_timesteps(
            num_steps, device="cuda", shift=shift, stretch_terminal=stretch_terminal,
        )
        scheduler.set_begin_index(0)
        timesteps = scheduler.timesteps

        for t in timesteps:
            t_input = t.unsqueeze(0).expand(B)
            velocity_pred = self._get_velocity(
                latents, t_input, condition, uncondition, guidance, use_cfg,
            )
            result = scheduler.step(
                velocity_pred.unsqueeze(0), t, latents.unsqueeze(0),
                return_dict=False, generator=gen,
            )
            latents = result[0].squeeze(0)

        return latents

    # ---- LTX-style stochastic (SDE) sampler ----

    @torch.no_grad()
    def _generate_sde(
        self, latents, condition, uncondition, guidance, use_cfg,
        B, num_steps, shift, seed, sde_eta,
        early_stop_t: Optional[float] = None,
        known_latents: Optional[Tensor] = None,
        known_mask: Optional[Tensor] = None,
        known_noise: Optional[Tensor] = None,
        repaint_u: int = 1,
        eta_first_frac: float = 0.0,
        eta_last_frac: float = 0.0,
        batched_cfg: bool = False,
        store_kv_exit_query: bool = False,
        store_kv_query_sigma: Optional[float] = None,
        geometry_source: Optional[Tensor] = None,
        geometry_confidence: Optional[Tensor] = None,
        geometry_noise_floor: float = 0.0,
    ) -> Tensor:
        """LTX-Video-style stochastic (SDE) sampler for rectified flow.

        Mirrors LTX's ``RectifiedFlowScheduler`` stochastic step: at each
        step predict the clean latent ``x0`` from the velocity, then re-noise
        it forward to the next (lower) sigma with fresh Gaussian noise
        (``add_noise(x0, eps, sigma_next) = (1-sigma_next)*x0 + sigma_next*eps``).
        Reuses the same shift/stretch sigma ladder as the UniPC sampler so the
        only difference vs deterministic sampling is the per-step renoise.

        ``sde_eta`` in ``[0, 1]`` scales how much *fresh* noise is injected:
          * ``1.0`` -> full LTX stochastic renoise (fresh noise each step),
          * ``0.0`` -> deterministic Euler on the same ladder,
          * in-between -> linear blend of the implied and fresh noise.
        The final step (``sigma_next == 0``) is always deterministic.

        ``eta_first_frac`` / ``eta_last_frac`` in ``[0, 1]`` force ``eta=0``
        (deterministic) for that fraction of the *leading* / *trailing* steps
        of the ladder, leaving stochasticity active only in the middle band.
        Use ``eta_last_frac>0`` to test the "lower eta helps as t->0"
        hypothesis: a deterministic finish preserves the sharp geometry that a
        constant late-step renoise would smear (the injected noise has too few
        remaining steps to be removed).

        ``store_kv_query_sigma`` keeps selected-query cache commitment near a
        reference sigma when a rollout temporarily uses a different step count.
        The query is captured during sampling and committed in one deferred
        forward so it cannot enter the history used by later denoising steps.

        ``known_noise`` optionally pins known frames to one forward-diffusion
        path instead of drawing a fresh path at every step.
        """
        if (geometry_source is None) != (geometry_confidence is None):
            raise ValueError(
                "geometry_source and geometry_confidence must be provided "
                "together"
            )
        if geometry_source is not None:
            geometry_source = geometry_source.to(
                device=latents.device, dtype=latents.dtype
            )
            geometry_confidence = geometry_confidence.to(
                device=latents.device, dtype=torch.float32
            )
            if geometry_source.shape != latents.shape:
                raise ValueError(
                    "geometry_source must match SDE latents: "
                    f"source={tuple(geometry_source.shape)}, "
                    f"latents={tuple(latents.shape)}"
                )

        gen = torch.Generator(device="cuda")
        gen.manual_seed(seed)
        n_steps = num_steps
        first_off = int(round(max(0.0, eta_first_frac) * n_steps))
        last_off = int(round(max(0.0, eta_last_frac) * n_steps))

        if shift is None:
            shift = self._rf_shift
        stretch_terminal = self.rectified_flow.shift_stretch_terminal
        self.sample_scheduler.set_timesteps(
            num_steps, device="cuda", shift=shift,
            stretch_terminal=stretch_terminal,
        )
        # sigmas: (num_steps+1,) in [0,1] with terminal sigma appended;
        # timesteps: (num_steps,) discrete int64 in [0, num_train].
        sigmas = self.sample_scheduler.sigmas.to("cuda").float()
        timesteps = self.sample_scheduler.timesteps
        if store_kv_query_sigma is not None and not store_kv_exit_query:
            raise ValueError(
                "store_kv_query_sigma requires store_kv_exit_query=True"
            )
        deferred_store_index = None
        deferred_store_query = None
        debug_nonfinite = bool(
            getattr(self, "_debug_nonfinite_sampling", False)
        )

        def _tensor_stats(value: torch.Tensor) -> str:
            value_f = value.detach().float()
            finite_mask = torch.isfinite(value_f)
            finite = value_f[finite_mask]
            if finite.numel() == 0:
                return "finite=0.000000%, no finite values"
            return (
                f"finite={float(finite_mask.float().mean()):.6%}, "
                f"min={float(finite.min()):.6g}, "
                f"max={float(finite.max()):.6g}, "
                f"mean={float(finite.mean()):.6g}, "
                f"std={float(finite.std()):.6g}"
            )

        if store_kv_query_sigma is not None:
            target_sigma = torch.as_tensor(
                store_kv_query_sigma,
                device=sigmas.device,
                dtype=sigmas.dtype,
            )
            deferred_store_index = int(
                torch.argmin((sigmas[:-1] - target_sigma).abs()).item()
            )

        if known_noise is not None and known_latents is None:
            raise ValueError("known_noise requires known_latents")

        # RePaint-style overlap conditioning: ``known_latents`` holds the clean
        # (diffusion-space) latent for the frames a previous window already
        # generated; ``known_mask`` (T,) marks them. At the start of each step we
        # overwrite the known frames with the known latent forward-noised to the
        # current sigma, so the new frames denoise conditioned on a fixed,
        # consistent history -> no seam between windows.
        use_known = known_latents is not None
        if use_known:
            if known_mask is None:
                raise ValueError("known_mask is required with known_latents")
            known_latents = known_latents.to(
                device=latents.device, dtype=latents.dtype)
            if tuple(known_latents.shape) != tuple(latents.shape):
                raise ValueError(
                    f"known_latents {tuple(known_latents.shape)} must match "
                    f"latents {tuple(latents.shape)}"
                )
            known_mask = known_mask.to(
                device=latents.device, dtype=torch.bool
            )
            if known_mask.ndim == 1:
                if known_mask.shape[0] != latents.shape[2]:
                    raise ValueError(
                        "known_mask length must match the latent time axis"
                    )
                km = known_mask.view(1, 1, -1, 1, 1)
            elif known_mask.ndim == 2:
                if tuple(known_mask.shape) != (B, latents.shape[2]):
                    raise ValueError(
                        "known_mask must have shape (T,) or (B,T)"
                    )
                km = known_mask[:, None, :, None, None]
            else:
                raise ValueError("known_mask must have shape (T,) or (B,T)")
            if known_noise is not None:
                known_noise = known_noise.to(
                    device=latents.device, dtype=latents.dtype
                )
                if tuple(known_noise.shape) != tuple(latents.shape):
                    raise ValueError(
                        f"known_noise {tuple(known_noise.shape)} must match "
                        f"latents {tuple(latents.shape)}"
                    )

            def _inject(x, sigma):
                eps = known_noise
                if eps is None:
                    eps = torch.randn(
                        x.shape,
                        generator=gen,
                        device=x.device,
                        dtype=x.dtype,
                    )
                noised = (1.0 - sigma) * known_latents + sigma * eps
                return torch.where(km, noised, x)

        # RePaint resampling ("jump-back"): repeat each denoise step
        # ``repaint_u`` times, jumping the noise level back up from sigma_next
        # to sigma between passes so the generated frames get multiple chances
        # to harmonize with the re-injected known frames. Only meaningful with
        # overlap conditioning; a no-op (u=1) otherwise.
        n_resample = max(1, repaint_u) if use_known else 1

        for i in range(len(timesteps)):
            sigma = sigmas[i]
            sigma_next = sigmas[i + 1]
            # no jump-back on the final (sigma_next==0) step
            u_here = n_resample if float(sigma_next) > 0.0 else 1
            t_input = timesteps[i].unsqueeze(0).expand(B)

            # Mid-band churn: disable stochasticity on the leading/trailing
            # steps so the deterministic finish keeps sharp geometry.
            in_band = (i >= first_off) and (i < len(timesteps) - last_off)
            eta_here = sde_eta if in_band else 0.0

            for u in range(u_here):
                if use_known:
                    latents = _inject(latents, sigma)
                if (
                    deferred_store_index == i
                    and u == u_here - 1
                ):
                    deferred_store_query = (
                        latents.detach().clone(),
                        t_input.detach().clone(),
                    )
                store_exit = (
                    store_kv_exit_query
                    and deferred_store_index is None
                    and i == len(timesteps) - 1
                    and u == u_here - 1
                )
                previous_store = getattr(
                    self, "_current_sfdmd_store_self_attn_kv", False
                )
                if store_exit:
                    self._current_sfdmd_store_self_attn_kv = True
                try:
                    v = self._get_velocity(
                        latents,
                        t_input,
                        condition,
                        uncondition,
                        guidance,
                        use_cfg,
                        batched_cfg=batched_cfg,
                    )
                finally:
                    if store_exit:
                        self._current_sfdmd_store_self_attn_kv = previous_store
                if debug_nonfinite and not torch.isfinite(v).all():
                    raise FloatingPointError(
                        "SDE received a non-finite velocity prediction at "
                        f"step {i + 1}/{len(timesteps)}, resample={u + 1}/"
                        f"{u_here}, sigma={float(sigma):.6g}; "
                        f"latents: {_tensor_stats(latents)}; "
                        f"velocity: {_tensor_stats(v)}"
                    )
                # x_t = (1-sigma)*x0 + sigma*eps, v = eps - x0
                # => x0 = x_t - sigma*v ;  eps_implied = v + x0
                x0 = latents - sigma * v
                eps_implied = v + x0
                if eta_here > 0.0 and float(sigma_next) > 0.0:
                    eps_fresh = torch.randn(
                        latents.shape, generator=gen,
                        device=latents.device, dtype=latents.dtype,
                    )
                    if geometry_source is not None:
                        eps_fresh = confidence_weighted_geometry_source(
                            eps_fresh,
                            geometry_source,
                            geometry_confidence,
                            geometry_noise_floor,
                        )
                    eps_used = (1.0 - eta_here) * eps_implied + eta_here * eps_fresh
                else:
                    eps_used = eps_implied
                latents = (1.0 - sigma_next) * x0 + sigma_next * eps_used
                if debug_nonfinite and not torch.isfinite(latents).all():
                    raise FloatingPointError(
                        "SDE produced non-finite latents after "
                        f"step {i + 1}/{len(timesteps)}, resample={u + 1}/"
                        f"{u_here}, sigma={float(sigma):.6g}, "
                        f"sigma_next={float(sigma_next):.6g}; "
                        f"velocity: {_tensor_stats(v)}; "
                        f"latents: {_tensor_stats(latents)}"
                    )

                if u < u_here - 1:
                    # jump back up to sigma (forward-diffuse the current x0
                    # estimate) for another harmonization pass.
                    eps_jb = torch.randn(
                        latents.shape, generator=gen,
                        device=latents.device, dtype=latents.dtype,
                    )
                    if geometry_source is not None:
                        eps_jb = confidence_weighted_geometry_source(
                            eps_jb,
                            geometry_source,
                            geometry_confidence,
                            geometry_noise_floor,
                        )
                    latents = (1.0 - sigma) * x0 + sigma * eps_jb

            if early_stop_t is not None:
                next_t_norm = float(sigma_next)
                if next_t_norm <= float(early_stop_t):
                    self.last_sampling_exit_t = next_t_norm
                    log.info(
                        f"_generate_sde: early-stopping at step {i + 1}/"
                        f"{len(timesteps)} (sigma={next_t_norm:.4f} <= "
                        f"early_stop_t={early_stop_t})"
                    )
                    break

        if use_known:
            latents = torch.where(km, known_latents, latents)

        if deferred_store_index is not None:
            if deferred_store_query is None:
                raise RuntimeError(
                    "SDE sampling ended before the requested K/V query sigma"
                )
            query_latents, query_timestep = deferred_store_query
            previous_store = getattr(
                self, "_current_sfdmd_store_self_attn_kv", False
            )
            self._current_sfdmd_store_self_attn_kv = True
            try:
                self._get_velocity(
                    query_latents,
                    query_timestep,
                    condition,
                    uncondition,
                    guidance,
                    use_cfg,
                    batched_cfg=batched_cfg,
                )
            finally:
                self._current_sfdmd_store_self_attn_kv = previous_store

        return latents

    # ---- Euler / Heun sampler ----

    @torch.no_grad()
    def _generate_euler_heun(
        self, latents, condition, uncondition, guidance, use_cfg,
        B, num_steps, sampler,
    ) -> Tensor:
        # sigma goes from 1 (noise) to 0 (clean); network expects
        # discrete timesteps = sigma * 1000.
        sigmas = torch.linspace(1.0, 0.0, num_steps + 1, device="cuda")

        for i in range(num_steps):
            sigma = sigmas[i]
            sigma_next = sigmas[i + 1]
            d_sigma = sigma_next - sigma  # negative (moving toward clean)

            t_input = (sigma * 1000.0).unsqueeze(0).expand(B)
            v = self._get_velocity(
                latents, t_input, condition, uncondition, guidance, use_cfg,
            )

            if sampler == "heun" and i < num_steps - 1:
                latents_euler = latents + d_sigma * v
                t_next_input = (sigma_next * 1000.0).unsqueeze(0).expand(B)
                v_next = self._get_velocity(
                    latents_euler, t_next_input, condition, uncondition,
                    guidance, use_cfg,
                )
                latents = latents + d_sigma * 0.5 * (v + v_next)
            else:
                latents = latents + d_sigma * v

        return latents

    # ------------------------------------------------------------------
    # Validation — reuses parent's validation_step which calls
    # generate_samples_from_batch (now the RF version above).
    # ------------------------------------------------------------------


class HdmapToLidarAutoregressiveRFModel(HdmapToLidarRectifiedFlowModel):
    """Single-frame RF model conditioned on the preceding LiDAR latent."""

    def build_model(self) -> torch.nn.ModuleDict:
        model = super().build_model()
        latent_channels = int(self.config.latent_shape[0])
        condition_count = int(
            getattr(self.config, "autoregressive_condition_count", 1)
        )
        old_channels = 2 * latent_channels + 1
        new_channels = (
            (2 + condition_count) * latent_channels + 1
        )
        model["net"].x_embedder.proj[
            1
        ]._checkpoint_zero_init_input_expansion = (
            2 * latent_channels,
            old_channels,
            new_channels,
        )
        if bool(
            getattr(self.config, "autoregressive_pose_film", False)
        ):
            model["autoregressive_pose_film"] = _RelativePoseFiLM(
                latent_channels,
                int(
                    getattr(
                        self.config,
                        "autoregressive_pose_film_hidden_channels",
                        256,
                    )
                ),
            )
        return model

    def __init__(self, config):
        super().__init__(config)
        self.autoregressive_condition_count = int(
            getattr(config, "autoregressive_condition_count", 1)
        )
        self.autoregressive_pose_film = bool(
            getattr(config, "autoregressive_pose_film", False)
        )
        self.autoregressive_ego_action_only = bool(
            getattr(config, "autoregressive_ego_action_only", False)
        )
        self.autoregressive_noise_modulation_max = float(
            getattr(
                config,
                "autoregressive_noise_modulation_max",
                0.0,
            )
        )
        if self.autoregressive_condition_count <= 0:
            raise ValueError(
                "autoregressive_condition_count must be positive"
            )
        if not 0.0 <= self.autoregressive_noise_modulation_max <= 1.0:
            raise ValueError(
                "autoregressive_noise_modulation_max must be in [0,1]"
            )
        # The LaGen paper applies the same condition-noise distribution during
        # autoregressive inference. Keep that behavior opt-in so existing
        # checkpoints and sampling commands remain unchanged.
        self.autoregressive_noise_modulation_inference = False
        self._current_autoregressive_condition: Optional[Tensor] = None

    def _apply_noise_modulation(self, latent: Tensor) -> Tensor:
        if (
            self.autoregressive_noise_modulation_max == 0.0
            or (
                not self.training
                and not self.autoregressive_noise_modulation_inference
            )
        ):
            return latent
        tau = torch.rand(
            (latent.shape[0],),
            device=latent.device,
            dtype=torch.float32,
        ) * self.autoregressive_noise_modulation_max
        tau = tau.to(latent.dtype).view(
            latent.shape[0], 1, 1, 1, 1
        )
        return (1.0 - tau) * latent + tau * torch.randn_like(latent)

    def _previous_latent(
        self, data_batch: dict[str, Tensor]
    ) -> Tensor:
        previous = data_batch.get("autoregressive_previous_video")
        poses = data_batch.get("autoregressive_lidar_to_world")
        if previous is None or poses is None:
            raise KeyError(
                "autoregressive training requires previous video and poses"
            )
        previous_z = self.encode(previous)
        if self.autoregressive_pose_film:
            poses = torch.as_tensor(
                poses, device=previous_z.device, dtype=torch.float32
            )
            if poses.shape[1:] != (2, 4, 4):
                raise ValueError(
                    "autoregressive poses must be (B,2,4,4), got "
                    f"{tuple(poses.shape)}"
                )
            relative = torch.linalg.inv(poses[:, 1]) @ poses[:, 0]
            previous_z = self.model["autoregressive_pose_film"](
                previous_z, relative.to(previous_z.dtype)
            )
        return self._apply_noise_modulation(previous_z)

    def _build_autoregressive_condition(
        self, data_batch: dict[str, Tensor]
    ) -> Tensor:
        return self._previous_latent(data_batch)

    def get_data_and_condition(
        self, data_batch: dict[str, Tensor]
    ):
        raw_state, latent_state, condition = (
            super().get_data_and_condition(data_batch)
        )
        if self.autoregressive_ego_action_only:
            if self._current_hdmap is None:
                raise RuntimeError(
                    "ego-action-only autoregression requires the retained "
                    "HDMap latent slot"
                )
            self._current_hdmap = torch.zeros_like(self._current_hdmap)
        autoregressive = self._build_autoregressive_condition(
            data_batch
        )
        expected = (
            self.autoregressive_condition_count
            * int(self.config.latent_shape[0])
        )
        if autoregressive.shape[1] != expected:
            raise ValueError(
                "autoregressive condition channel mismatch: "
                f"expected {expected}, got {autoregressive.shape[1]}"
            )
        self._current_autoregressive_condition = autoregressive
        return raw_state, latent_state, condition


class HdmapToLidarLaGenSDERFModel(
    HdmapToLidarAutoregressiveRFModel
):
    """LaGen-SDE adaptation with foreground/background latent conditions."""

    def __init__(self, config):
        super().__init__(config)
        if self.autoregressive_condition_count != 3:
            raise ValueError(
                "LaGen-SDE requires autoregressive_condition_count=3"
            )
        if not self.autoregressive_pose_film:
            raise ValueError("LaGen-SDE requires autoregressive_pose_film")
        self.autoregressive_geometry_root_path = str(
            getattr(config, "autoregressive_geometry_root_path", "")
        )
        self.autoregressive_geometry_filter_list = str(
            getattr(config, "autoregressive_geometry_filter_list", "")
        )
        self.autoregressive_min_range = float(
            getattr(config, "autoregressive_min_range", 2.5)
        )
        self.autoregressive_max_range = float(
            getattr(config, "autoregressive_max_range", 75.0)
        )
        self.autoregressive_min_value = float(
            getattr(config, "autoregressive_min_value", -1.0)
        )
        self.autoregressive_inverse_depth = bool(
            getattr(config, "autoregressive_inverse_depth", False)
        )
        self._autoregressive_geometry = None

    def _geometry(self):
        if self._autoregressive_geometry is None:
            self._autoregressive_geometry = load_waymo_lidar_geometry(
                root_path=self.autoregressive_geometry_root_path,
                filter_list_path=(
                    self.autoregressive_geometry_filter_list or None
                ),
                resample_width=1024,
            )
        return self._autoregressive_geometry

    def _build_autoregressive_condition(
        self, data_batch: dict[str, Tensor]
    ) -> Tensor:
        previous_z = self._previous_latent(data_batch)
        required = (
            "autoregressive_previous_video",
            "autoregressive_lidar_to_world",
            "moving_box_object_to_world",
            "moving_box_lwh",
            "moving_box_valid",
            "moving_box_type",
        )
        missing = [key for key in required if key not in data_batch]
        if missing:
            raise KeyError(
                f"LaGen-SDE missing batch fields: {missing}"
            )
        ray_dirs, elev, azim, row_off = self._geometry()
        foreground_frames = []
        background_frames = []
        batch_size = int(previous_z.shape[0])
        for batch_idx in range(batch_size):
            foreground, background, _ = build_lagen_sde_conditions(
                data_batch["autoregressive_previous_video"][batch_idx],
                data_batch["autoregressive_lidar_to_world"][batch_idx],
                data_batch["moving_box_object_to_world"][batch_idx],
                data_batch["moving_box_lwh"][batch_idx],
                data_batch["moving_box_valid"][batch_idx],
                data_batch["moving_box_type"][batch_idx],
                ray_dirs=ray_dirs,
                row_elevations_rad=elev,
                column_azimuths_rad=azim,
                row_azimuth_offsets_rad=row_off,
                min_range=self.autoregressive_min_range,
                max_range=self.autoregressive_max_range,
                min_value=self.autoregressive_min_value,
                inverse_depth=self.autoregressive_inverse_depth,
            )
            foreground_frames.append(foreground)
            background_frames.append(background)
        foreground_z = self._apply_noise_modulation(
            self.encode(torch.stack(foreground_frames, dim=0))
        )
        background_z = self._apply_noise_modulation(
            self.encode(torch.stack(background_frames, dim=0))
        )
        return torch.cat(
            [previous_z, foreground_z, background_z], dim=1
        )


class HdmapToLidarDiffusionForcingRFModel(HdmapToLidarRectifiedFlowModel):
    """Block-causal Diffusion Forcing for HDMap-conditioned LiDAR.

    Training samples one independent RF time per temporal block and keeps the
    standard velocity-matching objective. SDE continuation treats
    ``known_latents`` as committed clean context (t=0) by default; sparse
    resimulation can explicitly select RePaint-style active-sigma clamping.
    """

    def __init__(self, config):
        super().__init__(config)
        self.df_block_size = int(
            getattr(config, "diffusion_forcing_block_size", 0)
        )
        self.df_temporal_prefix_frames = int(
            getattr(config.net, "temporal_prefix_frames", 0)
        )
        self.df_artifixer_matching = bool(
            getattr(
                config, "diffusion_forcing_artifixer_matching", False
            )
        )
        self.input_video_has_prepend = bool(
            getattr(config, "input_video_has_prepend", True)
        )
        self.blockwise_warped_history_condition = bool(
            getattr(config, "blockwise_warped_history_condition", False)
        )
        self.blockwise_warped_history_concat = bool(
            getattr(config, "blockwise_warped_history_concat", False)
        )
        self.blockwise_warped_history_backend = str(
            getattr(config, "blockwise_warped_history_backend", "dataloader")
        )
        self.blockwise_geometry_source = bool(
            getattr(config, "blockwise_geometry_source", False)
        )
        self.blockwise_geometry_source_noise_floor = float(
            getattr(config, "blockwise_geometry_source_noise_floor", 0.2)
        )
        self.blockwise_geometry_source_preencode_confidence = bool(
            getattr(
                config,
                "blockwise_geometry_source_preencode_confidence",
                False,
            )
        )
        self.blockwise_geometry_source_dropout_prob = float(
            getattr(
                config, "blockwise_geometry_source_dropout_prob", 0.0
            )
        )
        self.blockwise_geometry_source_density_kernel = int(
            getattr(config, "blockwise_geometry_source_density_kernel", 5)
        )
        self.blockwise_geometry_source_density_saturation_hits = float(
            getattr(
                config,
                "blockwise_geometry_source_density_saturation_hits",
                5.0,
            )
        )
        self._current_blockwise_warped_history: Optional[Tensor] = None
        self._current_blockwise_geometry_source: Optional[Tensor] = None
        self._current_blockwise_geometry_confidence: Optional[Tensor] = None
        self._current_blockwise_self_prediction_fraction: Optional[Tensor] = None
        self._capture_blockwise_warped_history_input = False
        self._captured_blockwise_warped_history_input: Optional[Tensor] = None
        self.blockwise_spatial_map_self_prediction_prob = float(
            getattr(
                config,
                "blockwise_spatial_map_self_prediction_prob",
                0.0,
            )
        )
        self._blockwise_spatial_map_geometry = None
        assert self.df_block_size > 0, (
            "HdmapToLidarDiffusionForcingRFModel requires "
            "diffusion_forcing_block_size > 0"
        )
        if self.df_temporal_prefix_frames < 0:
            raise ValueError(
                "net.temporal_prefix_frames must be non-negative"
            )
        if self.blockwise_warped_history_backend not in (
            "dataloader",
            "torch_spatial_map",
        ):
            raise ValueError(
                "blockwise_warped_history_backend must be 'dataloader' or "
                f"'torch_spatial_map', got "
                f"{self.blockwise_warped_history_backend!r}"
            )
        if (
            self.blockwise_warped_history_backend == "torch_spatial_map"
            and not self.blockwise_warped_history_condition
        ):
            raise ValueError(
                "torch_spatial_map backend requires "
                "blockwise_warped_history_condition=True"
            )
        if (
            self.blockwise_warped_history_concat
            and not self.blockwise_warped_history_condition
        ):
            raise ValueError(
                "blockwise_warped_history_concat requires "
                "blockwise_warped_history_condition=True"
            )
        if not 0.0 <= self.blockwise_spatial_map_self_prediction_prob <= 1.0:
            raise ValueError(
                "blockwise_spatial_map_self_prediction_prob must be in [0,1]"
            )
        if (
            self.blockwise_spatial_map_self_prediction_prob > 0.0
            and self.blockwise_warped_history_backend != "torch_spatial_map"
        ):
            raise ValueError(
                "blockwise_spatial_map_self_prediction_prob requires the "
                "torch_spatial_map history backend"
            )
        if self.blockwise_geometry_source:
            if self.blockwise_warped_history_backend != "torch_spatial_map":
                raise ValueError(
                    "blockwise_geometry_source requires the "
                    "torch_spatial_map history backend"
                )
            if not 0.0 <= self.blockwise_geometry_source_noise_floor <= 1.0:
                raise ValueError(
                    "blockwise_geometry_source_noise_floor must be in [0,1]"
                )
            if not (
                0.0
                <= self.blockwise_geometry_source_dropout_prob
                <= 1.0
            ):
                raise ValueError(
                    "blockwise_geometry_source_dropout_prob must be in [0,1]"
                )
            if (
                self.blockwise_geometry_source_density_kernel <= 0
                or self.blockwise_geometry_source_density_kernel % 2 == 0
            ):
                raise ValueError(
                    "blockwise_geometry_source_density_kernel must be a "
                    "positive odd integer"
                )
            if self.blockwise_geometry_source_density_saturation_hits <= 0.0:
                raise ValueError(
                    "blockwise_geometry_source_density_saturation_hits must "
                    "be positive"
                )
        if self.blockwise_warped_history_backend == "torch_spatial_map":
            self._blockwise_spatial_map_radius_m = float(
                getattr(config, "blockwise_spatial_map_radius_m")
            )
            self._blockwise_spatial_map_voxel_m = float(
                getattr(config, "blockwise_spatial_map_voxel_m")
            )
            self._blockwise_spatial_map_voxel_policy = str(
                getattr(config, "blockwise_spatial_map_voxel_policy")
            )
            self._blockwise_spatial_map_observation_unit = str(
                getattr(
                    config,
                    "blockwise_spatial_map_observation_unit",
                    "block",
                )
            )
            if self._blockwise_spatial_map_observation_unit not in (
                "block",
                "frame",
            ):
                raise ValueError(
                    "blockwise_spatial_map_observation_unit must be "
                    "'block' or 'frame'"
                )
            self._blockwise_spatial_map_static_min_observations = int(
                getattr(
                    config,
                    "blockwise_spatial_map_static_min_observations",
                )
            )
            self._blockwise_spatial_map_candidate_ttl_updates = int(
                getattr(
                    config,
                    "blockwise_spatial_map_candidate_ttl_updates",
                )
            )
            self._blockwise_spatial_map_include_latest = bool(
                getattr(config, "blockwise_spatial_map_include_latest")
            )
            self._blockwise_spatial_map_free_space_clear = bool(
                getattr(config, "blockwise_spatial_map_free_space_clear")
            )
            self._blockwise_spatial_map_free_space_margin_m = float(
                getattr(config, "blockwise_spatial_map_free_space_margin_m")
            )
            self._blockwise_spatial_map_free_space_misses_to_remove = int(
                getattr(
                    config,
                    "blockwise_spatial_map_free_space_misses_to_remove",
                )
            )
            self._blockwise_spatial_map_min_range = float(
                getattr(config, "blockwise_spatial_map_min_range")
            )
            self._blockwise_spatial_map_max_range = float(
                getattr(config, "blockwise_spatial_map_max_range")
            )
            self._blockwise_spatial_map_min_value = float(
                getattr(config, "blockwise_spatial_map_min_value")
            )
            self._blockwise_spatial_map_inverse_depth = bool(
                getattr(config, "blockwise_spatial_map_inverse_depth")
            )
            self._blockwise_spatial_map_geometry_root_path = str(
                getattr(config, "blockwise_spatial_map_geometry_root_path")
            )
            self._blockwise_spatial_map_geometry_filter_list_path = str(
                getattr(
                    config,
                    "blockwise_spatial_map_geometry_filter_list_path",
                )
            )
            self._blockwise_spatial_map_geometry_resample_width = int(
                getattr(
                    config,
                    "blockwise_spatial_map_geometry_resample_width",
                    -1,
                )
            )
        net_block_size = int(getattr(config.net, "temporal_block_size", 0))
        factorized_bidirectional_stage1 = bool(
            getattr(config, "factorized_bidirectional_stage1", False)
        )
        expected_net_block_size = self.df_block_size
        if factorized_bidirectional_stage1:
            expected_net_block_size = 0
        assert net_block_size == expected_net_block_size, (
            f"model diffusion_forcing_block_size={self.df_block_size} expects "
            f"net.temporal_block_size={expected_net_block_size}, got "
            f"{net_block_size}"
        )
        assert self.rf_prediction_type == "velocity", (
            "Diffusion Forcing currently supports rf_prediction_type='velocity' "
            f"only, got {self.rf_prediction_type!r}"
        )
        if (
            self.df_artifixer_matching
            and self.rectified_flow.shift_clamp_percentiles is not None
        ):
            raise ValueError(
                "ArtiFixer-matched Diffusion Forcing requires shift-only "
                "timesteps; set shift_clamp_percentiles=None"
            )
        net = getattr(self.net, "module", self.net)
        side_channels = int(getattr(net, "side_condition_channels", 0))
        expected_side_channels = (
            int(config.latent_shape[0])
            if (
                self.blockwise_warped_history_condition
                and not self.blockwise_warped_history_concat
            )
            else 0
        )
        assert side_channels == expected_side_channels, (
            "net.side_condition_channels must match the lidar latent channels "
            "when blockwise warped history is enabled, and must be zero "
            f"otherwise; got {side_channels}, expected {expected_side_channels}"
        )
        log.info(
            "HdmapToLidarDiffusionForcingRFModel: "
            f"block_size={self.df_block_size}, "
            f"artifixer_matching={self.df_artifixer_matching}, "
            f"input_video_has_prepend={self.input_video_has_prepend}, "
            "blockwise_warped_history_condition="
            f"{self.blockwise_warped_history_condition}, "
            "blockwise_warped_history_concat="
            f"{self.blockwise_warped_history_concat}, "
            "blockwise_warped_history_backend="
            f"{self.blockwise_warped_history_backend}, "
            f"blockwise_geometry_source={self.blockwise_geometry_source}, "
            "geometry_source_preencode_confidence="
            f"{self.blockwise_geometry_source_preencode_confidence}, "
            "blockwise_spatial_map_self_prediction_prob="
            f"{self.blockwise_spatial_map_self_prediction_prob}, "
            "blockwise_spatial_map_observation_unit="
            f"{getattr(self, '_blockwise_spatial_map_observation_unit', 'block')}"
        )

    @torch.no_grad()
    def _build_torch_spatial_map_history(
        self,
        raw_state: Tensor,
        lidar_to_world: Tensor,
    ) -> tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        if self._blockwise_spatial_map_geometry is None:
            self._blockwise_spatial_map_geometry = load_waymo_lidar_geometry(
                root_path=self._blockwise_spatial_map_geometry_root_path,
                filter_list_path=(
                    self._blockwise_spatial_map_geometry_filter_list_path
                ),
                resample_width=(
                    self._blockwise_spatial_map_geometry_resample_width
                ),
            )
        if raw_state.ndim != 5:
            raise ValueError(
                "Torch spatial-map history requires raw lidar video shaped "
                f"(B,C,T,H,W), got {tuple(raw_state.shape)}"
            )
        if lidar_to_world.ndim != 4:
            raise ValueError(
                "Torch spatial-map history requires lidar_to_world shaped "
                f"(B,S,4,4), got {tuple(lidar_to_world.shape)}"
            )
        if raw_state.shape[0] != lidar_to_world.shape[0]:
            raise ValueError(
                "raw lidar and pose batch sizes differ: "
                f"{raw_state.shape[0]} vs {lidar_to_world.shape[0]}"
            )

        ray_dirs, elev, azim, row_off = self._blockwise_spatial_map_geometry
        if (
            not isinstance(ray_dirs, torch.Tensor)
            or ray_dirs.device != raw_state.device
        ):
            self._blockwise_spatial_map_geometry = tuple(
                torch.as_tensor(
                    value,
                    device=raw_state.device,
                    dtype=torch.float32,
                )
                for value in (ray_dirs, elev, azim, row_off)
            )
            ray_dirs, elev, azim, row_off = (
                self._blockwise_spatial_map_geometry
            )
        warped = []
        geometry_source = []
        geometry_confidence = []
        for batch_idx in range(raw_state.shape[0]):
            result = warp_blockwise_history_with_spatial_map(
                video=raw_state[batch_idx],
                lidar_to_world=lidar_to_world[batch_idx],
                block_size=self.df_block_size,
                prefix_frames=self.df_temporal_prefix_frames,
                input_has_prepend=self.input_video_has_prepend,
                ray_dirs=ray_dirs,
                row_elevations_rad=elev,
                column_azimuths_rad=azim,
                row_azimuth_offsets_rad=row_off,
                min_range=self._blockwise_spatial_map_min_range,
                max_range=self._blockwise_spatial_map_max_range,
                min_value=self._blockwise_spatial_map_min_value,
                inverse_depth=self._blockwise_spatial_map_inverse_depth,
                radius_m=self._blockwise_spatial_map_radius_m,
                voxel_size_m=self._blockwise_spatial_map_voxel_m,
                voxel_policy=self._blockwise_spatial_map_voxel_policy,
                observation_unit=(
                    self._blockwise_spatial_map_observation_unit
                ),
                static_min_observations=(
                    self._blockwise_spatial_map_static_min_observations
                ),
                candidate_ttl_updates=(
                    self._blockwise_spatial_map_candidate_ttl_updates
                ),
                include_latest=self._blockwise_spatial_map_include_latest,
                free_space_clear=self._blockwise_spatial_map_free_space_clear,
                free_space_margin_m=(
                    self._blockwise_spatial_map_free_space_margin_m
                ),
                free_space_misses_to_remove=(
                    self._blockwise_spatial_map_free_space_misses_to_remove
                ),
                return_soft_source=self.blockwise_geometry_source,
                soft_density_kernel=(
                    self.blockwise_geometry_source_density_kernel
                ),
                soft_density_saturation_hits=(
                    self.blockwise_geometry_source_density_saturation_hits
                ),
            )
            if self.blockwise_geometry_source:
                (
                    warped_i,
                    _,
                    geometry_source_i,
                    _,
                    confidence_i,
                ) = result
                geometry_source.append(geometry_source_i)
                geometry_confidence.append(confidence_i)
            else:
                warped_i, _ = result
            warped.append(warped_i)
        source = (
            torch.stack(geometry_source, dim=0)
            if geometry_source
            else None
        )
        confidence = (
            torch.stack(geometry_confidence, dim=0)
            if geometry_confidence
            else None
        )
        return torch.stack(warped, dim=0), source, confidence

    @torch.no_grad()
    def _store_blockwise_spatial_map_history(
        self,
        warped: Tensor,
        geometry_source: Optional[Tensor],
        geometry_confidence: Optional[Tensor],
        latent_state: Tensor,
    ) -> None:
        side_history = warped
        if self.blockwise_geometry_source:
            if geometry_source is None or geometry_confidence is None:
                raise RuntimeError(
                    "geometry-source training did not produce source and confidence"
                )
            confidence = geometry_confidence[:, None]
            side_history = confidence_composite_geometry(
                geometry_source,
                confidence,
                self._blockwise_spatial_map_min_value,
            )
        self._captured_blockwise_warped_history_input = None
        if self._capture_blockwise_warped_history_input:
            self._captured_blockwise_warped_history_input = (
                side_history.detach().float().cpu().contiguous()
            )
        warped_z = self.encode(
            side_history.to(device=latent_state.device).float()
        ).contiguous()
        if warped_z.shape != latent_state.shape:
            raise ValueError(
                "blockwise warped-history latent must match lidar latent shape: "
                f"got {tuple(warped_z.shape)} vs {tuple(latent_state.shape)}"
            )
        self._current_blockwise_warped_history = warped_z.to(
            self.diffusion_dtype
        )
        self._current_blockwise_geometry_source = None
        self._current_blockwise_geometry_confidence = None
        if not self.blockwise_geometry_source:
            return
        assert geometry_source is not None
        assert geometry_confidence is not None
        source_input = (
            side_history
            if self.blockwise_geometry_source_preencode_confidence
            else geometry_source
        )
        source_z = self.encode(
            source_input.to(device=latent_state.device).float()
        ).contiguous()
        if source_z.shape != latent_state.shape:
            raise ValueError(
                "blockwise geometry-source latent must match lidar latent shape: "
                f"got {tuple(source_z.shape)} vs {tuple(latent_state.shape)}"
            )
        self._current_blockwise_geometry_source = source_z.to(
            self.diffusion_dtype
        )
        confidence_5d = geometry_confidence[:, None].to(
            device=source_z.device, dtype=torch.float32
        )
        self._current_blockwise_geometry_confidence = (
            torch.nn.functional.adaptive_avg_pool3d(
                confidence_5d, source_z.shape[2:]
            ).clamp_(0.0, 1.0)
        )

    @torch.no_grad()
    def _one_step_euler_history_prediction(
        self,
        latent_state: Tensor,
        condition,
        *,
        raw_state: Optional[Tensor] = None,
    ) -> Tensor:
        del raw_state
        epsilon = torch.randn(
            latent_state.shape, **self.tensor_kwargs_fp32
        )
        source = epsilon
        if self.blockwise_geometry_source:
            geometry = self._current_blockwise_geometry_source
            confidence = self._current_blockwise_geometry_confidence
            if geometry is None or confidence is None:
                raise RuntimeError(
                    "one-step history prediction requires geometry and confidence"
                )
            source_weight = (
                1.0 - self.blockwise_geometry_source_noise_floor
            ) * confidence.float()
            source = (
                source_weight * geometry.float()
                + (1.0 - source_weight) * epsilon
            )
        timesteps = torch.full(
            (latent_state.shape[0], latent_state.shape[2]),
            float(self.rectified_flow.num_train_timesteps),
            **self.tensor_kwargs_fp32,
        )
        velocity = self._rf_network_forward(
            source.to(**self.tensor_kwargs), timesteps, condition
        ).float()
        return self.decode(source - velocity).detach()

    def get_data_and_condition(
        self, data_batch: dict[str, Tensor],
    ):
        raw_state, latent_state, condition = super().get_data_and_condition(
            data_batch
        )
        self._current_blockwise_geometry_source = None
        self._current_blockwise_geometry_confidence = None
        self._current_blockwise_self_prediction_fraction = None
        if not self.blockwise_warped_history_condition:
            self._current_blockwise_warped_history = None
            return raw_state, latent_state, condition

        if self.blockwise_warped_history_backend == "torch_spatial_map":
            poses = data_batch.get("lidar_to_world")
            if poses is None:
                raise KeyError(
                    "torch_spatial_map history requires "
                    "data_batch['lidar_to_world']"
                )
            warped, geometry_source, geometry_confidence = (
                self._build_torch_spatial_map_history(raw_state, poses)
            )
        else:
            geometry_source = None
            geometry_confidence = None
            warped = data_batch.get("lidar_blockwise_warped_history")
            if warped is None:
                raise KeyError(
                    "blockwise_warped_history_condition=True with dataloader "
                    "backend requires "
                    "data_batch['lidar_blockwise_warped_history']"
                )
        self._store_blockwise_spatial_map_history(
            warped,
            geometry_source,
            geometry_confidence,
            latent_state,
        )
        self_prediction_prob = self.blockwise_spatial_map_self_prediction_prob
        if self.training and self_prediction_prob > 0.0:
            # This branch contains an extra FSDP forward, so every rank in the
            # process mesh must preserve the same collective ordering.
            use_self_prediction = _sample_world_synchronized_bernoulli(
                self_prediction_prob,
                raw_state.device,
            )
            self_prediction_mask = use_self_prediction.repeat(
                raw_state.shape[0]
            )
            self._current_blockwise_self_prediction_fraction = (
                self_prediction_mask.float().mean()
            )
            if bool(use_self_prediction):
                predicted = self._one_step_euler_history_prediction(
                    latent_state,
                    condition,
                    raw_state=raw_state,
                ).to(device=raw_state.device, dtype=raw_state.dtype)
                if predicted.shape != raw_state.shape:
                    raise ValueError(
                        "one-step decoded history must match raw lidar shape: "
                        f"got {tuple(predicted.shape)} vs "
                        f"{tuple(raw_state.shape)}"
                    )
                mask = self_prediction_mask.view(
                    -1, *([1] * (raw_state.ndim - 1))
                )
                history_state = torch.where(mask, predicted, raw_state)
                warped, geometry_source, geometry_confidence = (
                    self._build_torch_spatial_map_history(
                        history_state, poses
                    )
                )
                self._store_blockwise_spatial_map_history(
                    warped,
                    geometry_source,
                    geometry_confidence,
                    latent_state,
                )
        return raw_state, latent_state, condition

    def _online_condition_inputs(
        self,
        data_batch: dict[str, Tensor],
        *,
        allow_training: bool = False,
    ) -> tuple[Tensor, Tensor]:
        if self.training and not allow_training:
            raise RuntimeError("get_online_condition is inference-only")
        if not self.blockwise_warped_history_condition:
            raise RuntimeError(
                "get_online_condition requires blockwise warp conditioning"
            )
        if (
            not self.encode_hdmap_with_vae
            or self.bev_hdmap_wan is not None
            or self.bev_hdmap_pool is not None
        ):
            raise NotImplementedError(
                "online joint encoding currently requires VAE-encoded "
                "range-image HDMap conditioning"
            )

        self._normalize_video_databatch_inplace(data_batch)
        self._augment_image_dim_inplace(data_batch)
        if self.is_image_batch(data_batch):
            raise ValueError("online joint encoding requires a video batch")

        hdmap = data_batch.get("lidar_hdmap")
        warped = data_batch.get("lidar_blockwise_warped_history")
        if hdmap is None or warped is None:
            raise KeyError(
                "online joint encoding requires lidar_hdmap and "
                "lidar_blockwise_warped_history"
            )
        if (
            hdmap.shape[0] != warped.shape[0]
            or hdmap.shape[2:] != warped.shape[2:]
        ):
            raise ValueError(
                "HDMap and warped-history inputs must align for joint encode: "
                f"hdmap={tuple(hdmap.shape)}, warped={tuple(warped.shape)}"
            )
        if getattr(self, "intensity_vae", None) is None:
            if hdmap.shape[1] != warped.shape[1]:
                raise ValueError(
                    "range-only HDMap and geometry must have equal channels"
                )
        elif hdmap.shape[1] != 1 or warped.shape[1] != 2:
            raise ValueError(
                "paired intensity conditioning requires C=1 HDMap and C=2 "
                f"range/intensity geometry, got {tuple(hdmap.shape)} and "
                f"{tuple(warped.shape)}"
            )
        return hdmap, warped

    def _store_online_condition(
        self,
        data_batch: dict[str, Tensor],
        hdmap_z: Tensor,
        warped_z: Tensor,
    ):
        if (
            hdmap_z.shape[0] != warped_z.shape[0]
            or hdmap_z.shape[2:] != warped_z.shape[2:]
        ):
            raise ValueError(
                "HDMap and warp latent grids must align: "
                f"hdmap={tuple(hdmap_z.shape)}, "
                f"warped={tuple(warped_z.shape)}"
            )
        if hdmap_z.shape[1] != self._hdmap_concat_ch:
            raise ValueError(
                "HDMap condition has incompatible channels: "
                f"got {hdmap_z.shape[1]}, expected {self._hdmap_concat_ch}"
            )
        if warped_z.shape[1] != self.state_shape[0]:
            raise ValueError(
                "geometry condition has incompatible channels: "
                f"got {warped_z.shape[1]}, expected {self.state_shape[0]}"
            )
        if tuple(hdmap_z.shape[3:]) != tuple(self.state_shape[2:]):
            raise ValueError(
                "condition latent has incompatible spatial shape: "
                f"got {tuple(hdmap_z.shape[3:])}, expected "
                f"{tuple(self.state_shape[2:])}"
            )
        self._store_hdmap_latent(hdmap_z.to(self.diffusion_dtype))
        self._current_blockwise_warped_history = warped_z.to(
            self.diffusion_dtype
        )
        condition = self.conditioner(data_batch)
        condition.data_type = DataType.VIDEO
        return condition

    @torch.no_grad()
    def get_online_condition(self, data_batch: dict[str, Tensor]):
        """Encode full HDMap and generated-warp windows in one VAE call."""
        hdmap, warped = self._online_condition_inputs(data_batch)
        if getattr(self, "_hdmap_force_zero_latent", False):
            warped_z = self.encode(warped.float()).contiguous()
            hdmap_z = warped_z.new_zeros(
                warped_z.shape[0],
                self._hdmap_concat_ch,
                *warped_z.shape[2:],
            )
        else:
            hdmap_z = self.encode(hdmap.float()).contiguous()
            warped_z = self.encode(warped.float()).contiguous()
        if hdmap_z.shape[2] != self.state_shape[1]:
            raise ValueError(
                f"full condition encode returned T={hdmap_z.shape[2]}, "
                f"expected {self.state_shape[1]}"
            )
        return self._store_online_condition(
            data_batch, hdmap_z, warped_z
        )

    @torch.no_grad()
    def get_online_condition_stream(
        self,
        data_batch: dict[str, Tensor],
        temporal_kv_cache=None,
        max_cache_frames: int | None = None,
        allow_training: bool = False,
    ):
        """Causally encode only the newly active HDMap and warp frames."""
        hdmap, warped = self._online_condition_inputs(
            data_batch,
            allow_training=allow_training,
        )
        cache = temporal_kv_cache
        if getattr(self, "intensity_vae", None) is not None:
            if cache is None:
                cache = {"hdmap": None, "geometry": None}
            elif not isinstance(cache, dict) or set(cache) != {
                "hdmap",
                "geometry",
            }:
                raise ValueError(
                    "paired online condition cache must contain exactly "
                    "'hdmap' and 'geometry'"
                )
            warped_z, geometry_cache = self.encode_stream(
                warped.float(),
                temporal_kv_cache=cache["geometry"],
                max_cache_frames=max_cache_frames,
            )
            if getattr(self, "_hdmap_force_zero_latent", False):
                hdmap_z = warped_z.new_zeros(
                    warped_z.shape[0],
                    self._hdmap_concat_ch,
                    *warped_z.shape[2:],
                )
                hdmap_cache = cache["hdmap"]
            else:
                hdmap_z, hdmap_cache = self.encode_stream(
                    hdmap.float(),
                    temporal_kv_cache=cache["hdmap"],
                    max_cache_frames=max_cache_frames,
                )
            condition = self._store_online_condition(
                data_batch, hdmap_z.contiguous(), warped_z.contiguous()
            )
            return condition, {
                "hdmap": hdmap_cache,
                "geometry": geometry_cache,
            }

        batch_size = hdmap.shape[0]
        if getattr(self, "_hdmap_force_zero_latent", False):
            warped_z, new_cache = self.encode_stream(
                warped.float(),
                temporal_kv_cache=temporal_kv_cache,
                max_cache_frames=max_cache_frames,
            )
            warped_z = warped_z.contiguous()
            hdmap_z = torch.zeros_like(warped_z)
        else:
            joint_z, new_cache = self.encode_stream(
                torch.cat([hdmap.float(), warped.float()], dim=0),
                temporal_kv_cache=temporal_kv_cache,
                max_cache_frames=max_cache_frames,
            )
            joint_z = joint_z.contiguous()
            hdmap_z, warped_z = joint_z.split(batch_size, dim=0)
        condition = self._store_online_condition(
            data_batch, hdmap_z, warped_z
        )
        return condition, new_cache

    def _sample_block_timesteps(
        self, batch_size: int, num_frames: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return block-constant timesteps, sigmas, and loss weights."""
        chunks = temporal_chunk_slices(
            num_frames,
            self.df_block_size,
            self.df_temporal_prefix_frames,
        )
        num_blocks = len(chunks)
        if self.df_artifixer_matching:
            num_train_timesteps = self.rectified_flow.num_train_timesteps
            timestep_ids_B_K = torch.randint(
                0,
                num_train_timesteps,
                (batch_size, num_blocks),
                device=self.tensor_kwargs_fp32["device"],
            )
            timesteps_B_K, sigmas_B_K, weights_B_K = (
                artifixer_flow_schedule(
                    timestep_ids_B_K,
                    num_train_timesteps=num_train_timesteps,
                    shift=float(self.rectified_flow.shift),
                )
            )
            return (
                expand_chunkwise(
                    timesteps_B_K,
                    num_frames,
                    self.df_block_size,
                    self.df_temporal_prefix_frames,
                ),
                expand_chunkwise(
                    sigmas_B_K,
                    num_frames,
                    self.df_block_size,
                    self.df_temporal_prefix_frames,
                ),
                expand_chunkwise(
                    weights_B_K,
                    num_frames,
                    self.df_block_size,
                    self.df_temporal_prefix_frames,
                ),
            )

        u_B_K = self.rectified_flow.sample_train_time(
            batch_size * num_blocks
        ).reshape(batch_size, num_blocks)
        u_B_T = expand_chunkwise(
            u_B_K,
            num_frames,
            self.df_block_size,
            self.df_temporal_prefix_frames,
        )
        # ``get_discrete_timestamp`` has a legacy squeeze that drops B for B=1.
        sigmas_B_T = self.rectified_flow.apply_shift_reparam(u_B_T)
        timesteps_B_T = (
            sigmas_B_T * self.rectified_flow.num_train_timesteps
        )
        weights_B_T = self.rectified_flow.train_time_weight(
            timesteps_B_T, self.tensor_kwargs_fp32,
        )
        return timesteps_B_T, sigmas_B_T, weights_B_T

    def _rectified_flow_forward(
        self, data_batch: dict[str, Tensor],
    ) -> tuple[dict[str, Tensor], Tensor]:
        _, x0, condition = self.get_data_and_condition(data_batch)
        B, _, T = x0.shape[:3]
        epsilon = torch.randn(x0.shape, **self.tensor_kwargs_fp32)
        timesteps_B_T, sigmas_B_T, time_weights = (
            self._sample_block_timesteps(B, T)
        )

        x0_f = x0.float()
        source = epsilon
        source_weight = None
        source_dropout_fraction = None
        if self.blockwise_geometry_source:
            geometry = self._current_blockwise_geometry_source
            confidence = self._current_blockwise_geometry_confidence
            if geometry is None or confidence is None:
                raise RuntimeError(
                    "geometry-source RF requires geometry and confidence"
                )
            expected_confidence_shape = (B, 1, *x0.shape[2:])
            if (
                geometry.shape != x0.shape
                or confidence.shape != expected_confidence_shape
            ):
                raise ValueError(
                    "geometry-source latent/confidence shapes must match x0: "
                    f"geometry={tuple(geometry.shape)}, "
                    f"confidence={tuple(confidence.shape)}, "
                    f"x0={tuple(x0.shape)}"
                )
            source_weight = (
                1.0 - self.blockwise_geometry_source_noise_floor
            ) * confidence.float()
            dropout_prob = self.blockwise_geometry_source_dropout_prob
            if self.training and dropout_prob > 0.0:
                chunks = temporal_chunk_slices(
                    T,
                    self.df_block_size,
                    self.df_temporal_prefix_frames,
                )
                num_blocks = len(chunks)
                keep_B_K = (
                    torch.rand(
                        (B, num_blocks),
                        device=x0.device,
                        dtype=torch.float32,
                    )
                    >= dropout_prob
                ).float()
                keep_B_K[:, 0] = 1.0
                keep_B_T = expand_chunkwise(
                    keep_B_K,
                    T,
                    self.df_block_size,
                    self.df_temporal_prefix_frames,
                )
                source_weight = source_weight * keep_B_T[
                    :, None, :, None, None
                ]
                source_dropout_fraction = (
                    1.0 - keep_B_K[:, 1:].mean()
                    if num_blocks > 1
                    else keep_B_K.new_zeros(())
                )
            source = (
                source_weight * geometry.float()
                + (1.0 - source_weight) * epsilon
            )
        sigma_5d = sigmas_B_T[:, None, :, None, None]
        xt = source * sigma_5d + x0_f * (1.0 - sigma_5d)
        vt_target = source - x0_f

        net_pred = self._rf_network_forward(
            xt.to(**self.tensor_kwargs), timesteps_B_T, condition,
        ).float()
        per_frame_loss = torch.mean(
            (net_pred - vt_target) ** 2,
            dim=(1, 3, 4),
        )
        loss = torch.mean(time_weights * per_frame_loss)
        per_instance_loss = per_frame_loss.mean(dim=1)

        output_batch = {
            "x0": x0,
            "xt": xt,
            "sigma": sigmas_B_T,
            "model_pred": net_pred,
            "edm_loss": loss,
            "timesteps": timesteps_B_T,
            "per_frame_loss": per_frame_loss,
            "per_instance_loss": per_instance_loss,
        }
        if source_weight is not None:
            output_batch.update(
                {
                    "metric/geometry_source_confidence_mean": (
                        self._current_blockwise_geometry_confidence.mean()
                    ),
                    "metric/geometry_source_weight_mean": (
                        source_weight.mean()
                    ),
                    "metric/geometry_source_coverage": (
                        source_weight > 0.0
                    ).float().mean(),
                    "metric/geometry_source_dropout_fraction": (
                        source_dropout_fraction
                        if source_dropout_fraction is not None
                        else source_weight.new_zeros(())
                    ),
                }
            )
        self_prediction_fraction = getattr(
            self, "_current_blockwise_self_prediction_fraction", None
        )
        if self_prediction_fraction is not None:
            output_batch[
                "metric/spatial_map_self_prediction_fraction"
            ] = self_prediction_fraction
        return output_batch, loss

    @torch.no_grad()
    def _generate_sde(
        self, latents, condition, uncondition, guidance, use_cfg,
        B, num_steps, shift, seed, sde_eta,
        early_stop_t: Optional[float] = None,
        known_latents: Optional[Tensor] = None,
        known_mask: Optional[Tensor] = None,
        known_noise: Optional[Tensor] = None,
        known_latents_mode: str = "clean_context",
        repaint_u: int = 1,
        eta_first_frac: float = 0.0,
        eta_last_frac: float = 0.0,
        batched_cfg: bool = False,
        store_kv_exit_query: bool = False,
        store_kv_query_sigma: Optional[float] = None,
        geometry_source: Optional[Tensor] = None,
        geometry_confidence: Optional[Tensor] = None,
        geometry_noise_floor: float = 0.0,
    ) -> Tensor:
        if known_latents_mode == "repaint":
            return super()._generate_sde(
                latents, condition, uncondition, guidance, use_cfg,
                B, num_steps, shift, seed, sde_eta,
                early_stop_t=early_stop_t,
                known_latents=known_latents,
                known_mask=known_mask,
                known_noise=known_noise,
                repaint_u=repaint_u,
                eta_first_frac=eta_first_frac,
                eta_last_frac=eta_last_frac,
                batched_cfg=batched_cfg,
                store_kv_exit_query=store_kv_exit_query,
                store_kv_query_sigma=store_kv_query_sigma,
                geometry_source=geometry_source,
                geometry_confidence=geometry_confidence,
                geometry_noise_floor=geometry_noise_floor,
            )
        if known_latents_mode not in ("clean_context", "clean_mask"):
            raise ValueError(
                "known_latents_mode must be 'clean_context', 'clean_mask', "
                "or 'repaint'"
            )
        if known_noise is not None:
            raise ValueError(
                "known_noise is only supported with "
                "known_latents_mode='repaint'"
            )

        # The first seed window can be sampled jointly; continuation windows
        # use the clean-context schedule below.
        if known_latents is None:
            return super()._generate_sde(
                latents, condition, uncondition, guidance, use_cfg,
                B, num_steps, shift, seed, sde_eta,
                early_stop_t=early_stop_t,
                known_latents=None,
                known_mask=None,
                known_noise=None,
                repaint_u=repaint_u,
                eta_first_frac=eta_first_frac,
                eta_last_frac=eta_last_frac,
                batched_cfg=batched_cfg,
                store_kv_exit_query=store_kv_exit_query,
                store_kv_query_sigma=store_kv_query_sigma,
                geometry_source=geometry_source,
                geometry_confidence=geometry_confidence,
                geometry_noise_floor=geometry_noise_floor,
            )
        if geometry_source is not None or geometry_confidence is not None:
            raise ValueError(
                "geometry-source SDE renoise is only supported by the "
                "non-overlapping cached rollout"
            )
        if store_kv_query_sigma is not None:
            raise NotImplementedError(
                "store_kv_query_sigma is not supported with known-latent "
                "Diffusion-Forcing continuation"
            )
        if repaint_u != 1:
            raise ValueError(
                "Diffusion-Forcing continuation does not use RePaint "
                f"jump-backs; expected repaint_u=1, got {repaint_u}"
            )
        if known_mask is None:
            raise ValueError("known_mask is required with known_latents")

        T = latents.shape[2]
        mask_B_T = known_mask.to(device=latents.device, dtype=torch.bool)
        if mask_B_T.ndim == 1:
            mask_B_T = mask_B_T.unsqueeze(0).expand(B, -1)
        if tuple(mask_B_T.shape) != (B, T):
            raise ValueError(
                f"known_mask {tuple(mask_B_T.shape)} must be (T={T},) or "
                f"(B={B},T={T})"
            )
        if known_latents_mode == "clean_context":
            try:
                validate_prefix_mask(mask_B_T, self.df_block_size)
            except ValueError as exc:
                raise ValueError(
                    f"{exc}; use overlap >= "
                    f"window_frames-{self.df_block_size}"
                ) from exc

        known_latents = known_latents.to(
            device=latents.device, dtype=latents.dtype,
        )
        if tuple(known_latents.shape) != tuple(latents.shape):
            raise ValueError(
                f"known_latents {tuple(known_latents.shape)} must match "
                f"latents {tuple(latents.shape)}"
            )
        known_5d = mask_B_T[:, None, :, None, None]
        active_5d = ~known_5d
        latents = torch.where(known_5d, known_latents, latents)

        gen = torch.Generator(device="cuda")
        gen.manual_seed(seed)
        if shift is None:
            shift = self._rf_shift
        self.sample_scheduler.set_timesteps(
            num_steps,
            device="cuda",
            shift=shift,
            stretch_terminal=self.rectified_flow.shift_stretch_terminal,
        )
        sigmas = self.sample_scheduler.sigmas.to("cuda").float()
        timesteps = self.sample_scheduler.timesteps
        first_off = int(round(max(0.0, eta_first_frac) * num_steps))
        last_off = int(round(max(0.0, eta_last_frac) * num_steps))

        for i, timestep in enumerate(timesteps):
            sigma = sigmas[i]
            sigma_next = sigmas[i + 1]
            active_timestep_B = timestep.to(
                device=latents.device,
            ).expand(B)
            t_input = build_clean_context_timesteps(
                active_timestep_B,
                mask_B_T,
            )
            latents = torch.where(known_5d, known_latents, latents)
            store_exit = store_kv_exit_query and i == len(timesteps) - 1
            previous_store = getattr(
                self, "_current_sfdmd_store_self_attn_kv", False
            )
            if store_exit:
                self._current_sfdmd_store_self_attn_kv = True
            try:
                velocity = self._get_velocity(
                    latents,
                    t_input,
                    condition,
                    uncondition,
                    guidance,
                    use_cfg,
                    batched_cfg=batched_cfg,
                )
            finally:
                if store_exit:
                    self._current_sfdmd_store_self_attn_kv = previous_store
            x0 = latents - sigma * velocity
            eps_implied = velocity + x0

            in_band = (
                i >= first_off and i < len(timesteps) - last_off
            )
            eta_here = sde_eta if in_band else 0.0
            if eta_here > 0.0 and float(sigma_next) > 0.0:
                eps_fresh = torch.randn(
                    latents.shape,
                    generator=gen,
                    device=latents.device,
                    dtype=latents.dtype,
                )
                eps_used = (
                    (1.0 - eta_here) * eps_implied
                    + eta_here * eps_fresh
                )
            else:
                eps_used = eps_implied
            next_latents = (
                (1.0 - sigma_next) * x0 + sigma_next * eps_used
            )
            latents = torch.where(
                active_5d, next_latents, known_latents,
            )

            if early_stop_t is not None:
                next_t_norm = float(sigma_next)
                if next_t_norm <= float(early_stop_t):
                    self.last_sampling_exit_t = next_t_norm
                    break

        return torch.where(known_5d, known_latents, latents)


class HdmapToLidarTeacherForcingRFModel(
    HdmapToLidarDiffusionForcingRFModel
):
    """Legacy interleaved clean-prefix teacher forcing for the LiDAR DiT.

    Each noisy block queries K/V produced by clean earlier blocks. One RF time
    is shared by the complete sequence. This implementation interleaves each
    noisy query and clean cache write; the separate Imaginaire class below uses
    a literal complete clean pass followed by a noisy replay pass.
    """

    def __init__(self, config):
        super().__init__(config)
        self.teacher_forcing_detach_clean_kv = bool(
            getattr(config, "teacher_forcing_detach_clean_kv", False)
        )
        self.teacher_forcing_loss_scale = float(
            getattr(config, "teacher_forcing_loss_scale", 1.0)
        )
        if self.teacher_forcing_loss_scale <= 0.0:
            raise ValueError("teacher_forcing_loss_scale must be positive")
        log.info(
            "HdmapToLidarTeacherForcingRFModel: "
            "detach_clean_kv="
            f"{self.teacher_forcing_detach_clean_kv}, "
            f"loss_scale={self.teacher_forcing_loss_scale}"
        )

    def new_teacher_forcing_kv_cache(
        self,
    ) -> list[dict[str, Tensor] | None]:
        net = self.net
        while hasattr(net, "module"):
            net = net.module
        return [None] * len(net.blocks)

    def _teacher_forcing_block_velocity(
        self,
        x_t: Tensor,
        timesteps: Tensor,
        condition,
        *,
        block_index: int,
        total_frames: int,
        kv_cache: list[dict[str, Tensor] | None],
        store_kv: bool,
        detach_kv: bool,
    ) -> Tensor:
        start = block_index * self.df_block_size
        end = start + self.df_block_size
        return self._teacher_forcing_chunk_velocity(
            x_t,
            timesteps,
            condition,
            start=start,
            end=end,
            total_frames=total_frames,
            kv_cache=kv_cache,
            store_kv=store_kv,
            detach_kv=detach_kv,
        )

    def _teacher_forcing_chunk_velocity(
        self,
        x_t: Tensor,
        timesteps: Tensor,
        condition,
        *,
        start: int,
        end: int,
        total_frames: int,
        kv_cache: list[dict[str, Tensor] | None],
        store_kv: bool,
        detach_kv: bool,
    ) -> Tensor:
        chunk_frames = end - start
        if (
            start < 0
            or end > total_frames
            or chunk_frames <= 0
            or x_t.shape[2] != chunk_frames
        ):
            raise ValueError(
                "teacher-forcing call does not match its temporal chunk: "
                f"chunk=({start},{end}), x_t={tuple(x_t.shape)}, "
                f"total_frames={total_frames}"
            )
        if self._current_hdmap is None:
            raise RuntimeError("teacher forcing requires encoded HDMap")
        full_hdmap = self._current_hdmap
        full_side = self._current_blockwise_warped_history
        if full_hdmap.shape[2] != total_frames:
            raise ValueError(
                "teacher-forcing HDMap length must match the latent sequence: "
                f"hdmap={full_hdmap.shape[2]}, latent={total_frames}"
            )
        if full_side is not None and full_side.shape[2] != total_frames:
            raise ValueError(
                "teacher-forcing side-condition length must match the latent "
                f"sequence: side={full_side.shape[2]}, latent={total_frames}"
            )

        names = (
            "_current_hdmap",
            "_current_blockwise_warped_history",
            "_current_sfdmd_temporal_positions",
            "_current_sfdmd_self_attn_kv_cache",
            "_current_sfdmd_store_self_attn_kv",
            "_current_sfdmd_self_attn_kv_max_frames",
            "_current_sfdmd_self_attn_kv_sink_frames",
            "_current_sfdmd_detach_self_attn_kv",
            "_current_sfdmd_return_self_attn_kv",
            "_current_sfdmd_updated_self_attn_kv_cache",
        )
        missing = object()
        previous = {
            name: getattr(self, name, missing)
            for name in names
        }
        self._current_hdmap = full_hdmap[:, :, start:end].contiguous()
        self._current_blockwise_warped_history = (
            None
            if full_side is None
            else full_side[:, :, start:end].contiguous()
        )
        self._current_sfdmd_temporal_positions = torch.arange(
            start, end, device=x_t.device
        )
        self._current_sfdmd_self_attn_kv_cache = kv_cache
        self._current_sfdmd_store_self_attn_kv = store_kv
        self._current_sfdmd_self_attn_kv_max_frames = total_frames
        self._current_sfdmd_self_attn_kv_sink_frames = 0
        self._current_sfdmd_detach_self_attn_kv = detach_kv
        self._current_sfdmd_return_self_attn_kv = store_kv
        self._current_sfdmd_updated_self_attn_kv_cache = None
        try:
            velocity = self._rf_network_forward(
                x_t.to(**self.tensor_kwargs), timesteps, condition
            ).float()
            if store_kv:
                updated_cache = (
                    self._current_sfdmd_updated_self_attn_kv_cache
                )
                if updated_cache is None:
                    raise RuntimeError(
                        "teacher-forcing clean forward did not return its "
                        "updated K/V cache"
                    )
                kv_cache[:] = updated_cache
            return velocity
        finally:
            for name, value in previous.items():
                if value is missing:
                    if hasattr(self, name):
                        delattr(self, name)
                else:
                    setattr(self, name, value)

    def _teacher_forcing_chunk_slices(
        self, num_frames: int
    ) -> list[tuple[int, int]]:
        return temporal_chunk_slices(num_frames, self.df_block_size)

    def _sample_teacher_forcing_source(
        self,
        x0: Tensor,
        epsilon: Tensor,
    ) -> tuple[Tensor, Tensor | None, Tensor]:
        if not self.blockwise_geometry_source:
            return epsilon, None, epsilon.new_zeros(())
        geometry = self._current_blockwise_geometry_source
        confidence = self._current_blockwise_geometry_confidence
        if geometry is None or confidence is None:
            raise RuntimeError(
                "geometry-source teacher forcing requires geometry and "
                "confidence"
            )
        expected_confidence_shape = (x0.shape[0], 1, *x0.shape[2:])
        if (
            geometry.shape != x0.shape
            or confidence.shape != expected_confidence_shape
        ):
            raise ValueError(
                "geometry-source latent/confidence shapes must match x0: "
                f"geometry={tuple(geometry.shape)}, "
                f"confidence={tuple(confidence.shape)}, "
                f"x0={tuple(x0.shape)}"
            )
        source_weight = (
            1.0 - self.blockwise_geometry_source_noise_floor
        ) * confidence.float()
        batch_size, _, num_frames = x0.shape[:3]
        chunks = self._teacher_forcing_chunk_slices(num_frames)
        num_blocks = len(chunks)
        dropout_prob = self.blockwise_geometry_source_dropout_prob
        dropout_fraction = source_weight.new_zeros(())
        if self.training and dropout_prob > 0.0:
            keep_B_K = (
                torch.rand(
                    (batch_size, num_blocks),
                    device=x0.device,
                    dtype=torch.float32,
                )
                >= dropout_prob
            ).float()
            keep_B_K[:, 0] = 1.0
            keep_B_T = keep_B_K.new_empty((batch_size, num_frames))
            for chunk_index, (start, end) in enumerate(chunks):
                keep_B_T[:, start:end] = keep_B_K[:, chunk_index, None]
            source_weight = source_weight * keep_B_T[
                :, None, :, None, None
            ]
            if num_blocks > 1:
                dropout_fraction = 1.0 - keep_B_K[:, 1:].mean()
        source = (
            source_weight * geometry.float()
            + (1.0 - source_weight) * epsilon
        )
        return source, source_weight, dropout_fraction

    def _rectified_flow_forward(
        self, data_batch: dict[str, Tensor]
    ) -> tuple[dict[str, Tensor], Tensor]:
        _, x0, condition = self.get_data_and_condition(data_batch)
        batch_size, _, num_frames = x0.shape[:3]
        if num_frames % self.df_block_size != 0:
            raise ValueError(
                "teacher forcing requires complete causal blocks, got "
                f"T={num_frames}, block_size={self.df_block_size}"
            )

        epsilon = torch.randn(x0.shape, **self.tensor_kwargs_fp32)
        source, source_weight, source_dropout_fraction = (
            self._sample_teacher_forcing_source(x0, epsilon)
        )
        sampled_time = self.rectified_flow.sample_train_time(batch_size)
        timesteps_B = self.rectified_flow.get_discrete_timestamp(
            sampled_time, self.tensor_kwargs_fp32
        ).reshape(batch_size)
        sigmas_B = self.rectified_flow.get_sigmas(
            timesteps_B, self.tensor_kwargs_fp32
        ).reshape(batch_size)
        time_weights = self.rectified_flow.train_time_weight(
            timesteps_B[:, None], self.tensor_kwargs_fp32
        ).reshape(batch_size, -1).mean(dim=1)

        x0_float = x0.float()
        sigma_5d = sigmas_B[:, None, None, None, None]
        x_t = sigma_5d * source + (1.0 - sigma_5d) * x0_float
        target = source - x0_float
        clean_timesteps = torch.zeros(
            (batch_size, self.df_block_size),
            device=x0.device,
            dtype=torch.float32,
        )
        active_timesteps = timesteps_B[:, None].expand(
            -1, self.df_block_size
        )
        clean_cache = self.new_teacher_forcing_kv_cache()
        prediction_blocks = []
        for block_index, start in enumerate(
            range(0, num_frames, self.df_block_size)
        ):
            end = start + self.df_block_size
            prediction_blocks.append(
                self._teacher_forcing_block_velocity(
                    x_t[:, :, start:end],
                    active_timesteps,
                    condition,
                    block_index=block_index,
                    total_frames=num_frames,
                    kv_cache=clean_cache,
                    store_kv=False,
                    detach_kv=self.teacher_forcing_detach_clean_kv,
                )
            )
            self._teacher_forcing_block_velocity(
                x0[:, :, start:end],
                clean_timesteps,
                condition,
                block_index=block_index,
                total_frames=num_frames,
                kv_cache=clean_cache,
                store_kv=True,
                detach_kv=self.teacher_forcing_detach_clean_kv,
            )

        prediction = torch.cat(prediction_blocks, dim=2)
        per_frame_loss = (
            prediction - target
        ).square().mean(dim=(1, 3, 4))
        per_instance_loss = per_frame_loss.mean(dim=1)
        raw_loss = (time_weights * per_instance_loss).mean()
        loss = self.teacher_forcing_loss_scale * raw_loss
        output_batch = {
            "x0": x0,
            "xt": x_t,
            "sigma": sigmas_B,
            "model_pred": prediction,
            "edm_loss": loss,
            "teacher_forcing_loss": raw_loss,
            "timesteps": timesteps_B,
            "per_frame_loss": per_frame_loss,
            "per_instance_loss": per_instance_loss,
        }
        if source_weight is not None:
            output_batch.update(
                {
                    "metric/geometry_source_confidence_mean": (
                        self._current_blockwise_geometry_confidence.mean()
                    ),
                    "metric/geometry_source_weight_mean": (
                        source_weight.mean()
                    ),
                    "metric/geometry_source_coverage": (
                        source_weight > 0.0
                    ).float().mean(),
                    "metric/geometry_source_dropout_fraction": (
                        source_dropout_fraction
                    ),
                }
            )
        return output_batch, loss


class HdmapToLidarImaginaireTeacherForcingRFModel(
    HdmapToLidarTeacherForcingRFModel
):
    """Imaginaire-style two-pass teacher forcing with a singleton prefix."""

    def __init__(self, config):
        super().__init__(config)
        if self.input_video_has_prepend:
            raise ValueError(
                "Imaginaire teacher forcing requires unique input frames; "
                "set input_video_has_prepend=False"
            )
        if self.blockwise_geometry_source:
            raise ValueError(
                "Imaginaire teacher forcing requires pure Gaussian RF source"
            )
        map_observation_unit = getattr(
            self, "_blockwise_spatial_map_observation_unit", "block"
        )
        map_static_min_observations = getattr(
            self, "_blockwise_spatial_map_static_min_observations", 0
        )
        minimum_map_observations = (
            3 if map_observation_unit == "block" else 2
        )
        insufficient_map_observations = (
            map_static_min_observations < minimum_map_observations
        )
        if (
            not self.blockwise_warped_history_condition
            or self.blockwise_warped_history_backend != "torch_spatial_map"
            or self._blockwise_spatial_map_include_latest
            or insufficient_map_observations
        ):
            raise ValueError(
                "Imaginaire teacher forcing requires strict accumulated "
                "geometry side conditioning with no latest sweep and at "
                "least three independent map observations in block mode or "
                "two distinct-frame observations in frame mode"
            )
        self.teacher_forcing_prefix_frames = int(
            getattr(config, "teacher_forcing_prefix_frames", 1)
        )
        factorized_bidirectional_stage1 = bool(
            getattr(config, "factorized_bidirectional_stage1", False)
        )
        if self.teacher_forcing_prefix_frames < 0 or (
            self.teacher_forcing_prefix_frames == 0
            and not factorized_bidirectional_stage1
        ):
            raise ValueError(
                "Imaginaire teacher forcing requires a positive prefix unless "
                "the isolated factorized bidirectional Stage 1 is enabled"
            )
        self.imaginaire_conditioned_prefix_frames = tuple(
            int(value)
            for value in getattr(
                config,
                "imaginaire_conditioned_prefix_frames",
                [0, 1],
            )
        )
        prefix_probs = tuple(
            float(value)
            for value in getattr(
                config,
                "imaginaire_conditioned_prefix_probs",
                [0.7, 0.3],
            )
        )
        if not self.imaginaire_conditioned_prefix_frames:
            self.imaginaire_conditioned_prefix_frames = (0, 1)
        if not prefix_probs:
            prefix_probs = (0.7, 0.3)
        if len(self.imaginaire_conditioned_prefix_frames) != len(
            prefix_probs
        ):
            raise ValueError(
                "Imaginaire prefix frames/probabilities must have the same "
                "length"
            )
        if (
            any(value < 0 for value in self.imaginaire_conditioned_prefix_frames)
            or any(value < 0.0 for value in prefix_probs)
            or sum(prefix_probs) <= 0.0
        ):
            raise ValueError(
                "Imaginaire prefix frames must be non-negative and prefix "
                "probabilities must have positive mass"
            )
        self.imaginaire_conditioned_prefix_probs = torch.tensor(
            prefix_probs, dtype=torch.float32
        ) / sum(prefix_probs)
        net = self.net
        while hasattr(net, "module"):
            net = net.module
        if factorized_bidirectional_stage1:
            invalid_layout = (
                net.temporal_block_size != 0
                or net.temporal_prefix_frames != 0
            )
        else:
            invalid_layout = (
                net.temporal_block_size != self.df_block_size
                or net.temporal_prefix_frames
                != self.teacher_forcing_prefix_frames
            )
        if invalid_layout:
            raise ValueError(
                "Imaginaire teacher-forcing model/network temporal layouts "
                "must match: "
                f"model={self.teacher_forcing_prefix_frames}+N*"
                f"{self.df_block_size}, net={net.temporal_prefix_frames}+N*"
                f"{net.temporal_block_size}"
            )

    def _sample_imaginaire_conditioned_prefix(
        self,
        *,
        num_frames: int,
        device: torch.device,
    ) -> tuple[int, Tensor]:
        # Stage 2/3 keep the same graph for every K, so sample independently
        # per rank just as Imaginaire samples a plan per training example.
        index = int(
            torch.multinomial(
                self.imaginaire_conditioned_prefix_probs, 1
            ).item()
        )
        prefix_frames = self.imaginaire_conditioned_prefix_frames[index]
        if prefix_frames >= num_frames:
            raise ValueError(
                "Imaginaire conditioned prefix must leave at least one noisy "
                f"frame, got K={prefix_frames} for T={num_frames}"
            )
        noisy_mask = torch.ones(
            num_frames, device=device, dtype=torch.float32
        )
        noisy_mask[:prefix_frames] = 0.0
        return prefix_frames, noisy_mask

    def _teacher_forcing_chunk_slices(
        self, num_frames: int
    ) -> list[tuple[int, int]]:
        return temporal_chunk_slices(
            num_frames,
            self.df_block_size,
            self.teacher_forcing_prefix_frames,
        )

    @staticmethod
    def _teacher_forcing_prefix_cache(
        full_cache: list[dict[str, Tensor] | None],
        *,
        prefix_frames: int,
        total_frames: int,
    ) -> list[dict[str, Tensor] | None]:
        if prefix_frames == 0:
            return [None] * len(full_cache)
        prefix_cache = []
        for layer_index, entry in enumerate(full_cache):
            if entry is None:
                raise RuntimeError(
                    "clean teacher-forcing cache is incomplete at layer "
                    f"{layer_index}"
                )
            num_tokens = entry["k"].shape[1]
            if num_tokens % total_frames != 0:
                raise ValueError(
                    "clean K/V length is not divisible by temporal frames: "
                    f"tokens={num_tokens}, frames={total_frames}"
                )
            prefix_tokens = prefix_frames * (num_tokens // total_frames)
            prefix_cache.append(
                {
                    "k": entry["k"][:, :prefix_tokens],
                    "v": entry["v"][:, :prefix_tokens],
                }
            )
        return prefix_cache

    def _build_complete_clean_cache(
        self,
        x0: Tensor,
        condition,
    ) -> list[dict[str, Tensor] | None]:
        batch_size, _, num_frames = x0.shape[:3]
        clean_cache = self.new_teacher_forcing_kv_cache()
        for frame_index in range(num_frames):
            clean_timesteps = torch.zeros(
                (batch_size, 1),
                device=x0.device,
                dtype=torch.float32,
            )
            self._teacher_forcing_chunk_velocity(
                x0[:, :, frame_index : frame_index + 1],
                clean_timesteps,
                condition,
                start=frame_index,
                end=frame_index + 1,
                total_frames=num_frames,
                kv_cache=clean_cache,
                store_kv=True,
                detach_kv=self.teacher_forcing_detach_clean_kv,
            )
        return clean_cache

    def _reduce_teacher_forcing_per_frame_loss(
        self,
        per_frame_loss: Tensor,
        noisy_mask_T: Tensor,
    ) -> Tensor:
        del noisy_mask_T
        return per_frame_loss.mean(dim=1)

    def _rectified_flow_forward(
        self, data_batch: dict[str, Tensor]
    ) -> tuple[dict[str, Tensor], Tensor]:
        _, x0, condition = self.get_data_and_condition(data_batch)
        batch_size, _, num_frames = x0.shape[:3]
        chunks = self._teacher_forcing_chunk_slices(num_frames)
        conditioned_prefix_frames, noisy_mask_T = (
            self._sample_imaginaire_conditioned_prefix(
                num_frames=num_frames,
                device=x0.device,
            )
        )

        epsilon = torch.randn(x0.shape, **self.tensor_kwargs_fp32)
        source, source_weight, source_dropout_fraction = (
            self._sample_teacher_forcing_source(x0, epsilon)
        )
        sampled_time = self.rectified_flow.sample_train_time(batch_size)
        timesteps_B = self.rectified_flow.get_discrete_timestamp(
            sampled_time, self.tensor_kwargs_fp32
        ).reshape(batch_size)
        sigmas_B = self.rectified_flow.get_sigmas(
            timesteps_B, self.tensor_kwargs_fp32
        ).reshape(batch_size)
        time_weights = self.rectified_flow.train_time_weight(
            timesteps_B[:, None], self.tensor_kwargs_fp32
        ).reshape(batch_size, -1).mean(dim=1)

        x0_float = x0.float()
        sigma_B_T = sigmas_B[:, None] * noisy_mask_T[None]
        sigma_5d = sigma_B_T[:, None, :, None, None]
        x_t = sigma_5d * source + (1.0 - sigma_5d) * x0_float
        target = source - x0_float

        # Pass 1 builds complete clean replay K/V. Pass 2 gives each noisy
        # chunk only the clean chunks that precede it.
        clean_cache = self._build_complete_clean_cache(x0, condition)
        prediction_blocks = []
        for start, end in chunks:
            active_timesteps = (
                timesteps_B[:, None]
                * noisy_mask_T[None, start:end]
            )
            prefix_cache = self._teacher_forcing_prefix_cache(
                clean_cache,
                prefix_frames=start,
                total_frames=num_frames,
            )
            prediction_blocks.append(
                self._teacher_forcing_chunk_velocity(
                    x_t[:, :, start:end],
                    active_timesteps,
                    condition,
                    start=start,
                    end=end,
                    total_frames=num_frames,
                    kv_cache=prefix_cache,
                    store_kv=False,
                    detach_kv=self.teacher_forcing_detach_clean_kv,
                )
            )

        prediction = torch.cat(prediction_blocks, dim=2)
        per_frame_loss = (
            prediction - target
        ).square().mean(dim=(1, 3, 4)) * noisy_mask_T[None]
        per_instance_loss = self._reduce_teacher_forcing_per_frame_loss(
            per_frame_loss, noisy_mask_T
        )
        raw_loss = (time_weights * per_instance_loss).mean()
        loss = self.teacher_forcing_loss_scale * raw_loss
        output_batch = {
            "x0": x0,
            "xt": x_t,
            "sigma": sigmas_B,
            "model_pred": prediction,
            "edm_loss": loss,
            "teacher_forcing_loss": raw_loss,
            "timesteps": timesteps_B,
            "per_frame_loss": per_frame_loss,
            "per_instance_loss": per_instance_loss,
            "condition_mask": 1.0 - noisy_mask_T,
            "metric/conditioned_prefix_frames": loss.new_tensor(
                float(conditioned_prefix_frames)
            ),
        }
        if source_weight is not None:
            output_batch.update(
                {
                    "metric/geometry_source_confidence_mean": (
                        self._current_blockwise_geometry_confidence.mean()
                    ),
                    "metric/geometry_source_weight_mean": source_weight.mean(),
                    "metric/geometry_source_coverage": (
                        source_weight > 0.0
                    ).float().mean(),
                    "metric/geometry_source_dropout_fraction": (
                        source_dropout_fraction
                    ),
                }
            )
        return output_batch, loss


class HdmapToLidarImaginaireOnePassTeacherForcingRFModel(
    HdmapToLidarImaginaireTeacherForcingRFModel
):
    """Chunkwise teacher forcing in one ``[clean | noisy]`` DiT pass.

    This preserves the paper pipeline's singleton-prefix/chunk-4 visibility
    while using a packed one-pass execution: clean queries are frame-causal,
    and noisy queries see strictly earlier clean chunks plus their own noisy
    chunk. Gradients from noisy predictions flow through the clean stream
    without retaining mutable replay caches.
    """

    def _teacher_forcing_one_pass_velocity(
        self,
        x0: Tensor,
        x_t: Tensor,
        active_timesteps: Tensor,
        condition,
    ) -> Tensor:
        if x0.shape != x_t.shape:
            raise ValueError(
                "clean and noisy teacher-forcing tensors must match: "
                f"clean={tuple(x0.shape)}, noisy={tuple(x_t.shape)}"
            )
        batch_size, _, num_frames = x0.shape[:3]
        if active_timesteps.shape != (batch_size, num_frames):
            raise ValueError(
                "teacher-forcing timesteps must be (B,T), got "
                f"{tuple(active_timesteps.shape)} for B={batch_size}, "
                f"T={num_frames}"
            )
        # Validate the configured [prefix, chunk, ...] partition before the
        # compiled mask sees it.
        self._teacher_forcing_chunk_slices(num_frames)

        hdmap = self._current_hdmap
        side_condition = self._current_blockwise_warped_history
        if hdmap is None or side_condition is None:
            raise RuntimeError(
                "one-pass teacher forcing requires HDMap and geometry side "
                "conditions"
            )
        if hdmap.shape[2] != num_frames or side_condition.shape[2] != num_frames:
            raise ValueError(
                "teacher-forcing conditions must match the clean sequence: "
                f"hdmap={tuple(hdmap.shape)}, "
                f"side={tuple(side_condition.shape)}, T={num_frames}"
            )

        packed_x = torch.cat([x0, x_t.to(x0.dtype)], dim=2)
        packed_timesteps = torch.cat(
            [torch.zeros_like(active_timesteps), active_timesteps], dim=1
        )
        positions = torch.arange(num_frames, device=x0.device)

        names = (
            "_current_hdmap",
            "_current_blockwise_warped_history",
            "_current_teacher_forcing_clean_frames",
            "_current_teacher_forcing_temporal_positions",
            "_current_stage2_known_frame_mask",
            "_current_stage2_task_frame_type_ids",
        )
        missing = object()
        previous = {name: getattr(self, name, missing) for name in names}
        self._current_hdmap = torch.cat([hdmap, hdmap], dim=2)
        self._current_blockwise_warped_history = torch.cat(
            [side_condition, side_condition], dim=2
        )
        self._current_teacher_forcing_clean_frames = num_frames
        self._current_teacher_forcing_temporal_positions = torch.cat(
            [positions, positions], dim=0
        )
        known_mask = getattr(
            self, "_current_stage2_known_frame_mask", None
        )
        if known_mask is not None:
            self._current_stage2_known_frame_mask = torch.cat(
                [known_mask, known_mask], dim=-1
            )
        task_ids = getattr(
            self, "_current_stage2_task_frame_type_ids", None
        )
        if task_ids is not None:
            self._current_stage2_task_frame_type_ids = torch.cat(
                [task_ids, task_ids], dim=0
            )
        try:
            packed_prediction = self._rf_network_forward(
                packed_x.to(**self.tensor_kwargs),
                packed_timesteps,
                condition,
            )
        finally:
            for name, value in previous.items():
                if value is missing:
                    if hasattr(self, name):
                        delattr(self, name)
                else:
                    setattr(self, name, value)

        if packed_prediction.shape[2] != 2 * num_frames:
            raise RuntimeError(
                "one-pass teacher forcing returned the wrong temporal length: "
                f"{tuple(packed_prediction.shape)}"
            )
        return packed_prediction[:, :, num_frames:]

    def _rectified_flow_forward(
        self, data_batch: dict[str, Tensor]
    ) -> tuple[dict[str, Tensor], Tensor]:
        _, x0, condition = self.get_data_and_condition(data_batch)
        batch_size, _, num_frames = x0.shape[:3]
        conditioned_prefix_frames, noisy_mask_T = (
            self._sample_imaginaire_conditioned_prefix(
                num_frames=num_frames,
                device=x0.device,
            )
        )

        epsilon = torch.randn(x0.shape, **self.tensor_kwargs_fp32)
        source, source_weight, source_dropout_fraction = (
            self._sample_teacher_forcing_source(x0, epsilon)
        )
        sampled_time = self.rectified_flow.sample_train_time(batch_size)
        timesteps_B = self.rectified_flow.get_discrete_timestamp(
            sampled_time, self.tensor_kwargs_fp32
        ).reshape(batch_size)
        sigmas_B = self.rectified_flow.get_sigmas(
            timesteps_B, self.tensor_kwargs_fp32
        ).reshape(batch_size)
        time_weights = self.rectified_flow.train_time_weight(
            timesteps_B[:, None], self.tensor_kwargs_fp32
        ).reshape(batch_size, -1).mean(dim=1)

        x0_float = x0.float()
        sigma_B_T = sigmas_B[:, None] * noisy_mask_T[None]
        sigma_5d = sigma_B_T[:, None, :, None, None]
        x_t = sigma_5d * source + (1.0 - sigma_5d) * x0_float
        target = source - x0_float
        active_timesteps = timesteps_B[:, None] * noisy_mask_T[None]
        prediction = self._teacher_forcing_one_pass_velocity(
            x0,
            x_t,
            active_timesteps,
            condition,
        )

        per_frame_loss = (
            prediction - target
        ).square().mean(dim=(1, 3, 4)) * noisy_mask_T[None]
        per_instance_loss = self._reduce_teacher_forcing_per_frame_loss(
            per_frame_loss, noisy_mask_T
        )
        raw_loss = (time_weights * per_instance_loss).mean()
        loss = self.teacher_forcing_loss_scale * raw_loss
        output_batch = {
            "x0": x0,
            "xt": x_t,
            "sigma": sigmas_B,
            "model_pred": prediction,
            "edm_loss": loss,
            "teacher_forcing_loss": raw_loss,
            "timesteps": timesteps_B,
            "per_frame_loss": per_frame_loss,
            "per_instance_loss": per_instance_loss,
            "condition_mask": 1.0 - noisy_mask_T,
            "metric/conditioned_prefix_frames": loss.new_tensor(
                float(conditioned_prefix_frames)
            ),
        }
        if source_weight is not None:
            output_batch.update(
                {
                    "metric/geometry_source_confidence_mean": (
                        self._current_blockwise_geometry_confidence.mean()
                    ),
                    "metric/geometry_source_weight_mean": source_weight.mean(),
                    "metric/geometry_source_coverage": (
                        source_weight > 0.0
                    ).float().mean(),
                    "metric/geometry_source_dropout_fraction": (
                        source_dropout_fraction
                    ),
                }
            )
        return output_batch, loss


class HdmapToLidarConcatGeometryTeacherForcingRFModel(
    HdmapToLidarImaginaireOnePassTeacherForcingRFModel
):
    """Canonical Stage 2 TF with concatenated GT accumulated geometry."""

    def build_model(self) -> torch.nn.ModuleDict:
        model = super().build_model()
        latent_channels = int(self.config.latent_shape[0])
        model["net"].x_embedder.proj[
            1
        ]._checkpoint_zero_init_input_expansion = (
            2 * latent_channels,
            2 * latent_channels + 2,
            3 * latent_channels + 2,
        )
        return model

    def __init__(self, config):
        super().__init__(config)
        if not self.blockwise_warped_history_concat:
            raise ValueError(
                "concat-geometry teacher forcing requires geometry concat"
            )
        if self.blockwise_spatial_map_self_prediction_prob != 0.0:
            raise ValueError(
                "GT-accumulation teacher forcing requires "
                "blockwise_spatial_map_self_prediction_prob=0"
            )


class HdmapToLidarMultiTaskTeacherForcingRFModel(
    HdmapToLidarConcatGeometryTeacherForcingRFModel
):
    """Explicit world, forecast, and resimulation Stage-2 mixture."""

    _ALLOWED_TASKS = ("world", "forecast", "resimulation")

    def build_model(self) -> torch.nn.ModuleDict:
        model = super().build_model()
        latent_channels = int(self.config.latent_shape[0])
        model["net"].x_embedder.proj[
            1
        ]._checkpoint_zero_init_input_expansion = (
            2 * latent_channels,
            2 * latent_channels + 2,
            3 * latent_channels + 3,
        )
        return model

    def __init__(self, config):
        super().__init__(config)
        self.stage2_multitask_modes = tuple(
            str(mode) for mode in config.stage2_multitask_modes
        )
        probabilities = tuple(
            float(value) for value in config.stage2_multitask_probs
        )
        if (
            not self.stage2_multitask_modes
            or len(self.stage2_multitask_modes) != len(probabilities)
            or len(set(self.stage2_multitask_modes))
            != len(self.stage2_multitask_modes)
            or any(
                mode not in self._ALLOWED_TASKS
                for mode in self.stage2_multitask_modes
            )
            or any(value < 0.0 for value in probabilities)
            or sum(probabilities) <= 0.0
        ):
            raise ValueError(
                "Stage-2 multi-task modes must be unique members of "
                f"{self._ALLOWED_TASKS} with matching positive probability "
                "mass"
            )
        self.stage2_multitask_probs = torch.tensor(
            probabilities, dtype=torch.float32
        ) / sum(probabilities)
        self.stage2_multitask_sampling = str(
            config.stage2_multitask_sampling
        )
        if self.stage2_multitask_sampling not in (
            "categorical",
            "round_robin",
        ):
            raise ValueError(
                "stage2_multitask_sampling must be categorical or round_robin"
            )
        self._stage2_multitask_round_robin_index = 0
        self.stage2_forecast_observed_frames = int(
            config.stage2_multitask_forecast_observed_frames
        )
        self.stage2_resim_known_min = int(
            config.stage2_multitask_resim_known_min_per_block
        )
        self.stage2_resim_known_max = int(
            config.stage2_multitask_resim_known_max_per_block
        )
        self.stage2_resim_drop_block_prob = float(
            config.stage2_multitask_resim_drop_block_prob
        )
        self.stage2_world_hdmap_dropout_prob = float(
            config.stage2_multitask_world_hdmap_dropout_prob
        )
        self.stage2_hdmap_bbox_channel_start = int(
            config.stage2_multitask_hdmap_bbox_channel_start
        )
        if not config.stage2_multitask_known_mask_concat:
            raise ValueError(
                "multi-task Stage 2 requires the known-frame mask channel"
            )
        if self.hdmap_dropout_prob != 0.0:
            raise ValueError(
                "multi-task Stage 2 controls HDMap availability per task; "
                "set the generic hdmap_dropout_prob to zero"
            )
        if self.stage2_forecast_observed_frames <= 0:
            raise ValueError(
                "forecast observed-frame count must be positive"
            )
        if (
            self.stage2_resim_known_min < 0
            or self.stage2_resim_known_max < self.stage2_resim_known_min
            or self.stage2_resim_known_max > self.df_block_size
            or not 0.0 <= self.stage2_resim_drop_block_prob <= 1.0
            or not 0.0 <= self.stage2_world_hdmap_dropout_prob <= 1.0
            or self.stage2_hdmap_bbox_channel_start < 0
        ):
            raise ValueError("invalid Stage-2 multi-task mask/HDMap settings")
        net = self.net
        while hasattr(net, "module"):
            net = net.module
        if net.num_frame_type_embeddings < len(
            self.stage2_multitask_modes
        ):
            raise ValueError(
                "the DiT needs one frame-type embedding per Stage-2 task"
            )
        self._current_stage2_task_id = 0
        self._current_stage2_task_name = self.stage2_multitask_modes[0]
        self._current_stage2_known_frame_mask: Optional[Tensor] = None
        self._current_stage2_task_frame_type_ids: Optional[Tensor] = None
        self._current_stage2_data_batch = None
        self._current_stage2_dynamic_removed_fraction = torch.tensor(0.0)
        self._current_stage2_truncated_boxes = 0.0

    def _sample_stage2_task(self, device: torch.device) -> int:
        if self.stage2_multitask_sampling == "round_robin":
            task_id = (
                self._stage2_multitask_round_robin_index
                % len(self.stage2_multitask_modes)
            )
            self._stage2_multitask_round_robin_index += 1
            return task_id
        return _sample_world_synchronized_categorical(
            self.stage2_multitask_probs.to(device=device)
        )

    def _sample_stage2_known_mask(
        self,
        task_name: str,
        *,
        num_frames: int,
        device: torch.device,
    ) -> Tensor:
        known = torch.zeros(
            num_frames, device=device, dtype=torch.float32
        )
        if task_name == "world":
            return known
        if task_name == "forecast":
            if self.stage2_forecast_observed_frames >= num_frames:
                raise ValueError(
                    "forecast observations must leave at least one target "
                    f"frame, got {self.stage2_forecast_observed_frames}/"
                    f"{num_frames}"
                )
            known[: self.stage2_forecast_observed_frames] = 1.0
            return known

        chunks = self._teacher_forcing_chunk_slices(num_frames)
        known[0] = 1.0
        for start, end in chunks[1:]:
            if (
                torch.rand((), device=device)
                < self.stage2_resim_drop_block_prob
            ):
                continue
            count = int(
                torch.randint(
                    self.stage2_resim_known_min,
                    self.stage2_resim_known_max + 1,
                    (1,),
                    device=device,
                ).item()
            )
            if count == 0:
                continue
            selected = torch.randperm(
                end - start, device=device
            )[:count]
            known[start + selected] = 1.0
        return known

    def _sample_imaginaire_conditioned_prefix(
        self,
        *,
        num_frames: int,
        device: torch.device,
    ) -> tuple[int, Tensor]:
        known = self._current_stage2_known_frame_mask
        if known is None or known.shape != (num_frames,):
            raise RuntimeError(
                "Stage-2 task mask was not prepared before RF training"
            )
        known = known.to(device=device, dtype=torch.float32)
        noisy = 1.0 - known
        if not bool(noisy.any()):
            raise ValueError("Stage-2 task mask must leave a prediction target")
        return int(known.sum().item()), noisy

    def _reduce_teacher_forcing_per_frame_loss(
        self,
        per_frame_loss: Tensor,
        noisy_mask_T: Tensor,
    ) -> Tensor:
        return per_frame_loss.sum(dim=1) / noisy_mask_T.sum().clamp_min(1.0)

    def _stage2_spatial_map_geometry(
        self, raw_state: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if self._blockwise_spatial_map_geometry is None:
            self._blockwise_spatial_map_geometry = load_waymo_lidar_geometry(
                root_path=self._blockwise_spatial_map_geometry_root_path,
                filter_list_path=(
                    self._blockwise_spatial_map_geometry_filter_list_path
                ),
                resample_width=(
                    self._blockwise_spatial_map_geometry_resample_width
                ),
            )
        if not isinstance(
            self._blockwise_spatial_map_geometry[0], torch.Tensor
        ) or self._blockwise_spatial_map_geometry[0].device != raw_state.device:
            self._blockwise_spatial_map_geometry = tuple(
                torch.as_tensor(
                    value,
                    device=raw_state.device,
                    dtype=torch.float32,
                )
                for value in self._blockwise_spatial_map_geometry
            )
        return self._blockwise_spatial_map_geometry

    def _remove_dynamic_returns(
        self,
        raw_state: Tensor,
        lidar_to_world: Tensor,
        ray_dirs: Tensor,
    ) -> Tensor:
        data_batch = self._current_stage2_data_batch
        if data_batch is None:
            raise RuntimeError("Stage-2 task batch is unavailable")
        required = (
            "moving_box_object_to_world",
            "moving_box_lwh",
            "moving_box_valid",
        )
        missing = [key for key in required if key not in data_batch]
        if missing:
            raise KeyError(
                "dynamic filtering is missing dataset fields "
                f"{missing}"
            )
        filtered = []
        fractions = []
        for batch_idx in range(raw_state.shape[0]):
            filtered_i, fraction_i = remove_points_inside_moving_boxes(
                raw_state[batch_idx],
                lidar_to_world[batch_idx],
                data_batch["moving_box_object_to_world"][batch_idx],
                data_batch["moving_box_lwh"][batch_idx],
                data_batch["moving_box_valid"][batch_idx],
                ray_dirs=ray_dirs,
                min_range=self._blockwise_spatial_map_min_range,
                max_range=self._blockwise_spatial_map_max_range,
                min_value=self._blockwise_spatial_map_min_value,
                inverse_depth=self._blockwise_spatial_map_inverse_depth,
            )
            filtered.append(filtered_i)
            fractions.append(fraction_i)
        self._current_stage2_dynamic_removed_fraction = torch.stack(
            fractions
        ).mean()
        return torch.stack(filtered)

    @torch.no_grad()
    def _build_torch_spatial_map_history(
        self,
        raw_state: Tensor,
        lidar_to_world: Tensor,
    ) -> tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        ray_dirs, elev, azim, row_off = self._stage2_spatial_map_geometry(
            raw_state
        )
        task_name = self._current_stage2_task_name
        self._current_stage2_dynamic_removed_fraction = raw_state.new_tensor(
            0.0, dtype=torch.float32
        )
        if task_name == "world":
            static_state = self._remove_dynamic_returns(
                raw_state, lidar_to_world, ray_dirs
            )
            return super()._build_torch_spatial_map_history(
                static_state, lidar_to_world
            )

        observed_mask = self._current_stage2_known_frame_mask
        if observed_mask is None:
            raise RuntimeError("Stage-2 observed mask is unavailable")
        history_state = raw_state
        if task_name == "resimulation":
            history_state = self._remove_dynamic_returns(
                raw_state, lidar_to_world, ray_dirs
            )
        warped = []
        for batch_idx in range(raw_state.shape[0]):
            warped_i, _ = warp_frozen_observed_history_with_spatial_map(
                history_state[batch_idx],
                lidar_to_world[batch_idx],
                observed_mask,
                ray_dirs=ray_dirs,
                row_elevations_rad=elev,
                column_azimuths_rad=azim,
                row_azimuth_offsets_rad=row_off,
                min_range=self._blockwise_spatial_map_min_range,
                max_range=self._blockwise_spatial_map_max_range,
                min_value=self._blockwise_spatial_map_min_value,
                inverse_depth=self._blockwise_spatial_map_inverse_depth,
                radius_m=self._blockwise_spatial_map_radius_m,
                voxel_size_m=self._blockwise_spatial_map_voxel_m,
            )
            warped.append(warped_i)
        return torch.stack(warped), None, None

    def get_data_and_condition(
        self, data_batch: dict[str, Tensor]
    ):
        video = data_batch.get(self.input_data_key)
        if video is None or video.ndim != 5:
            raise ValueError(
                "multi-task Stage 2 requires video input (B,C,T,H,W)"
            )
        task_id = self._sample_stage2_task(video.device)
        task_name = self.stage2_multitask_modes[task_id]
        num_frames = int(video.shape[2])
        known_mask = self._sample_stage2_known_mask(
            task_name,
            num_frames=num_frames,
            device=video.device,
        )
        self._current_stage2_task_id = task_id
        self._current_stage2_task_name = task_name
        self._current_stage2_known_frame_mask = known_mask
        self._current_stage2_task_frame_type_ids = torch.full(
            (num_frames,),
            task_id,
            device=video.device,
            dtype=torch.long,
        )
        self._current_stage2_data_batch = data_batch

        if task_name == "resimulation":
            semantic = data_batch.get("lidar_hdmap_semantic")
            if semantic is None or semantic.ndim != 5:
                raise KeyError(
                    "resimulation task requires semantic HDMap (B,K,T,H,W)"
                )
            if self.stage2_hdmap_bbox_channel_start >= semantic.shape[1]:
                raise ValueError(
                    "bbox HDMap channel start is outside the semantic tensor"
                )
            data_batch["lidar_hdmap"] = compose_normalized_range_channels(
                semantic[:, self.stage2_hdmap_bbox_channel_start :],
                channel_dim=1,
                min_value=self._blockwise_spatial_map_min_value,
                inverse_depth=self._blockwise_spatial_map_inverse_depth,
            )

        previous_force_zero = getattr(
            self, "_hdmap_force_zero_latent", False
        )
        force_zero = task_name == "forecast"
        if task_name == "world" and self.training:
            force_zero = bool(
                torch.rand((), device=video.device)
                < self.stage2_world_hdmap_dropout_prob
            )
        self._hdmap_force_zero_latent = force_zero
        try:
            result = super().get_data_and_condition(data_batch)
            truncated = data_batch.get("moving_box_truncated")
            if truncated is not None:
                if isinstance(truncated, Tensor):
                    self._current_stage2_truncated_boxes = float(
                        truncated.float().sum().item()
                    )
                else:
                    self._current_stage2_truncated_boxes = float(
                        sum(truncated)
                    )
            return result
        finally:
            self._hdmap_force_zero_latent = previous_force_zero
            self._current_stage2_data_batch = None

    def _rectified_flow_forward(
        self, data_batch: dict[str, Tensor]
    ) -> tuple[dict[str, Tensor], Tensor]:
        output_batch, loss = super()._rectified_flow_forward(data_batch)
        for task_id, task_name in enumerate(self.stage2_multitask_modes):
            output_batch[f"metric/stage2_task_{task_name}"] = loss.new_tensor(
                float(task_id == self._current_stage2_task_id)
            )
        known = self._current_stage2_known_frame_mask
        if known is not None:
            output_batch["metric/stage2_known_frames"] = known.sum()
            output_batch["metric/stage2_known_fraction"] = known.mean()
        output_batch["metric/stage2_dynamic_removed_fraction"] = (
            self._current_stage2_dynamic_removed_fraction.to(
                device=loss.device
            )
        )
        output_batch["metric/stage2_truncated_moving_boxes"] = (
            loss.new_tensor(self._current_stage2_truncated_boxes)
        )
        return output_batch, loss


class HdmapToLidarFactorizedTeacherForcingRFModel(
    HdmapToLidarConcatGeometryTeacherForcingRFModel
):
    """Task-free Stage 2 with factorized prefix, map, and geometry choices."""

    _HDMAP_MODES = ("null", "full", "boxes")
    _GEOMETRY_MODES = (
        "null",
        "rolling_static",
        "frozen_raw",
        "frozen_object",
    )

    def build_model(self) -> torch.nn.ModuleDict:
        model = super().build_model()
        latent_channels = int(self.config.latent_shape[0])
        model["net"].x_embedder.proj[
            1
        ]._checkpoint_zero_init_input_expansion = (
            2 * latent_channels,
            2 * latent_channels + 2,
            3 * latent_channels + 3,
        )
        return model

    @staticmethod
    def _validated_distribution(
        names,
        probabilities,
        *,
        allowed,
        label: str,
    ) -> tuple[tuple, Tensor]:
        names = tuple(names)
        probabilities = tuple(float(value) for value in probabilities)
        if (
            not names
            or len(names) != len(probabilities)
            or len(set(names)) != len(names)
            or any(name not in allowed for name in names)
            or any(value < 0.0 for value in probabilities)
            or sum(probabilities) <= 0.0
        ):
            raise ValueError(
                f"invalid {label} distribution: names={names}, "
                f"probabilities={probabilities}"
            )
        probs = torch.tensor(probabilities, dtype=torch.float32)
        return names, probs / probs.sum()

    def __init__(self, config):
        super().__init__(config)
        prefix_frames = tuple(
            int(value) for value in config.stage2_factorized_prefix_frames
        )
        prefix_probs = tuple(
            float(value) for value in config.stage2_factorized_prefix_probs
        )
        if (
            not prefix_frames
            or len(prefix_frames) != len(prefix_probs)
            or len(set(prefix_frames)) != len(prefix_frames)
            or any(value < 0 for value in prefix_frames)
            or any(value < 0.0 for value in prefix_probs)
            or sum(prefix_probs) <= 0.0
        ):
            raise ValueError(
                "invalid factorized Stage-2 prefix distribution"
            )
        self.stage2_factorized_prefix_frames = prefix_frames
        self.stage2_factorized_prefix_probs = torch.tensor(
            prefix_probs, dtype=torch.float32
        )
        self.stage2_factorized_prefix_probs /= (
            self.stage2_factorized_prefix_probs.sum()
        )
        (
            self.stage2_factorized_hdmap_modes,
            self.stage2_factorized_hdmap_probs,
        ) = self._validated_distribution(
            config.stage2_factorized_hdmap_modes,
            config.stage2_factorized_hdmap_probs,
            allowed=self._HDMAP_MODES,
            label="HDMap",
        )
        (
            self.stage2_factorized_geometry_modes_k0,
            self.stage2_factorized_geometry_probs_k0,
        ) = self._validated_distribution(
            config.stage2_factorized_geometry_modes_k0,
            config.stage2_factorized_geometry_probs_k0,
            allowed=self._GEOMETRY_MODES,
            label="K=0 geometry",
        )
        (
            self.stage2_factorized_geometry_modes_known,
            self.stage2_factorized_geometry_probs_known,
        ) = self._validated_distribution(
            config.stage2_factorized_geometry_modes_known,
            config.stage2_factorized_geometry_probs_known,
            allowed=self._GEOMETRY_MODES,
            label="K>0 geometry",
        )
        self.factorized_decouple_geometry_observations = bool(
            config.factorized_decouple_geometry_observations
        )
        geometry_observation_frames = tuple(
            int(value)
            for value in config.factorized_geometry_observation_frames
        )
        geometry_observation_probs = tuple(
            float(value)
            for value in config.factorized_geometry_observation_probs
        )
        if self.factorized_decouple_geometry_observations:
            if (
                not geometry_observation_frames
                or len(geometry_observation_frames)
                != len(geometry_observation_probs)
                or len(set(geometry_observation_frames))
                != len(geometry_observation_frames)
                or any(value <= 0 for value in geometry_observation_frames)
                or any(value < 0.0 for value in geometry_observation_probs)
                or sum(geometry_observation_probs) <= 0.0
            ):
                raise ValueError(
                    "invalid factorized geometry-observation distribution"
                )
            if (
                self.stage2_factorized_geometry_modes_k0
                != self.stage2_factorized_geometry_modes_known
                or not torch.allclose(
                    self.stage2_factorized_geometry_probs_k0,
                    self.stage2_factorized_geometry_probs_known,
                )
            ):
                raise ValueError(
                    "decoupled geometry requires one K-independent geometry "
                    "mode distribution"
                )
        self.factorized_geometry_observation_frames = (
            geometry_observation_frames
        )
        self.factorized_geometry_observation_probs = torch.tensor(
            geometry_observation_probs, dtype=torch.float32
        )
        if geometry_observation_probs:
            self.factorized_geometry_observation_probs /= (
                self.factorized_geometry_observation_probs.sum()
            )
        invalid_k0 = {"frozen_raw", "frozen_object"}.intersection(
            self.stage2_factorized_geometry_modes_k0
        )
        if invalid_k0 and not self.factorized_decouple_geometry_observations:
            raise ValueError(
                "K=0 geometry cannot use frozen observations: "
                f"{sorted(invalid_k0)}"
            )
        self.stage2_factorized_hdmap_bbox_channel_start = int(
            config.stage2_factorized_hdmap_bbox_channel_start
        )
        self.factorized_geometry_block_size = int(
            config.factorized_geometry_block_size
        )
        self.factorized_geometry_prefix_frames = int(
            config.factorized_geometry_prefix_frames
        )
        self.factorized_geometry_rolling_shutter = bool(
            config.factorized_geometry_rolling_shutter
        )
        self.factorized_geometry_actor_box_time_mode = str(
            config.factorized_geometry_actor_box_time_mode
        )
        if self.factorized_geometry_actor_box_time_mode not in (
            "frame",
            "current_to_next",
            "previous_to_current",
        ):
            raise ValueError(
                "unsupported factorized_geometry_actor_box_time_mode "
                f"{self.factorized_geometry_actor_box_time_mode!r}"
            )
        self.stage2_factorized_force_prefix_k = int(
            config.stage2_factorized_force_prefix_k
        )
        self.stage2_factorized_force_hdmap_mode = str(
            config.stage2_factorized_force_hdmap_mode
        )
        self.stage2_factorized_force_geometry_mode = str(
            config.stage2_factorized_force_geometry_mode
        )
        self.factorized_force_geometry_observations = int(
            config.factorized_force_geometry_observations
        )
        self.factorized_force_geometry_source_indices: Optional[
            tuple[int, ...]
        ] = None
        if (
            self.stage2_factorized_force_prefix_k >= 0
            and self.stage2_factorized_force_prefix_k
            not in self.stage2_factorized_prefix_frames
        ):
            raise ValueError(
                "forced prefix K must be one of the configured prefix lengths"
            )
        if (
            self.stage2_factorized_force_hdmap_mode
            and self.stage2_factorized_force_hdmap_mode
            not in self._HDMAP_MODES
        ):
            raise ValueError("invalid forced HDMap mode")
        if (
            self.stage2_factorized_force_geometry_mode
            and self.stage2_factorized_force_geometry_mode
            not in self._GEOMETRY_MODES
        ):
            raise ValueError("invalid forced geometry mode")
        if (
            self.factorized_force_geometry_observations >= 0
            and (
                not self.factorized_decouple_geometry_observations
                or self.factorized_force_geometry_observations
                not in self.factorized_geometry_observation_frames
            )
        ):
            raise ValueError(
                "forced geometry observations require the decoupled mode and "
                "must use a configured G"
            )
        if not config.stage2_known_mask_concat:
            raise ValueError(
                "factorized Stage 2 requires the known-frame mask channel"
            )
        if self.hdmap_dropout_prob != 0.0:
            raise ValueError(
                "factorized Stage 2 samples HDMap form explicitly; set "
                "hdmap_dropout_prob=0"
            )
        if self.stage2_factorized_hdmap_bbox_channel_start < 0:
            raise ValueError("HDMap bbox channel start must be non-negative")
        if self.factorized_geometry_block_size <= 0:
            raise ValueError(
                "factorized geometry block size must be positive"
            )
        if self.factorized_geometry_prefix_frames < 0:
            raise ValueError(
                "factorized geometry prefix frames must be non-negative"
            )
        net = self.net
        while hasattr(net, "module"):
            net = net.module
        if net.num_frame_type_embeddings != 0:
            raise ValueError(
                "factorized Stage 2 must not use task/frame-type embeddings"
            )

        self._current_stage2_known_frame_mask: Optional[Tensor] = None
        self._current_stage2_task_frame_type_ids: Optional[Tensor] = None
        self._current_stage2_data_batch = None
        self._current_stage2_prefix_k = 0
        self._current_stage2_geometry_prefix_g = 0
        self._current_stage2_geometry_source_mask: Optional[Tensor] = None
        self._current_stage2_hdmap_mode = "full"
        self._current_stage2_geometry_mode = "rolling_static"
        self._current_stage2_geometry_diagnostics: dict[str, Tensor] = {}
        self._current_stage2_truncated_boxes = 0.0
        self.factorized_rolling_geometry_known_prefix_only = False
        self.factorized_static_only_geometry = False
        self._sensor_spatial_map_geometry: dict[
            tuple[int, int, int, str],
            tuple[Tensor, Tensor, Tensor, Tensor],
        ] = {}

    def _sample_factorized_modes(
        self,
        *,
        num_frames: int,
        device: torch.device,
    ) -> tuple[int, str, str, Tensor, int, Tensor]:
        if self.stage2_factorized_force_prefix_k >= 0:
            prefix_k = self.stage2_factorized_force_prefix_k
        else:
            prefix_index = _sample_world_synchronized_categorical(
                self.stage2_factorized_prefix_probs.to(device=device)
            )
            prefix_k = self.stage2_factorized_prefix_frames[prefix_index]
        if prefix_k >= num_frames:
            raise ValueError(
                "factorized Stage-2 prefix must leave a prediction target, "
                f"got K={prefix_k} for T={num_frames}"
            )

        if self.stage2_factorized_force_hdmap_mode:
            hdmap_mode = self.stage2_factorized_force_hdmap_mode
        else:
            hdmap_index = _sample_world_synchronized_categorical(
                self.stage2_factorized_hdmap_probs.to(device=device)
            )
            hdmap_mode = self.stage2_factorized_hdmap_modes[hdmap_index]

        if self.factorized_decouple_geometry_observations:
            geometry_modes = self.stage2_factorized_geometry_modes_known
            geometry_probs = self.stage2_factorized_geometry_probs_known
        else:
            geometry_modes = (
                self.stage2_factorized_geometry_modes_k0
                if prefix_k == 0
                else self.stage2_factorized_geometry_modes_known
            )
            geometry_probs = (
                self.stage2_factorized_geometry_probs_k0
                if prefix_k == 0
                else self.stage2_factorized_geometry_probs_known
            )
        if self.stage2_factorized_force_geometry_mode:
            geometry_mode = self.stage2_factorized_force_geometry_mode
            if geometry_mode not in geometry_modes:
                raise ValueError(
                    f"forced geometry mode {geometry_mode!r} is invalid for "
                    f"K={prefix_k}; allowed={geometry_modes}"
                )
        else:
            geometry_index = _sample_world_synchronized_categorical(
                geometry_probs.to(device=device)
            )
            geometry_mode = geometry_modes[geometry_index]

        known_mask = torch.zeros(
            num_frames, device=device, dtype=torch.float32
        )
        known_mask[:prefix_k] = 1.0
        geometry_prefix_g = (
            prefix_k
            if geometry_mode in ("frozen_raw", "frozen_object")
            else 0
        )
        geometry_source_mask = known_mask.clone()
        if self.factorized_decouple_geometry_observations:
            geometry_source_mask.zero_()
            if geometry_mode in ("frozen_raw", "frozen_object"):
                if self.factorized_force_geometry_observations >= 0:
                    geometry_prefix_g = (
                        self.factorized_force_geometry_observations
                    )
                else:
                    geometry_index = _sample_world_synchronized_categorical(
                        self.factorized_geometry_observation_probs.to(
                            device=device
                        )
                    )
                    geometry_prefix_g = (
                        self.factorized_geometry_observation_frames[
                            geometry_index
                        ]
                    )
                if geometry_prefix_g > num_frames:
                    raise ValueError(
                        "factorized geometry observations exceed the clip: "
                        f"G={geometry_prefix_g}, T={num_frames}"
                    )
                geometry_source_mask[:geometry_prefix_g] = 1.0
            elif self.factorized_force_geometry_observations >= 0:
                raise ValueError(
                    "forced geometry observations require frozen geometry"
                )
            else:
                geometry_prefix_g = 0
        forced_source_indices = (
            self.factorized_force_geometry_source_indices
        )
        if forced_source_indices is not None:
            if (
                not self.factorized_decouple_geometry_observations
                or geometry_mode not in ("frozen_raw", "frozen_object")
            ):
                raise ValueError(
                    "explicit geometry source indices require decoupled "
                    "frozen geometry"
                )
            if (
                len(forced_source_indices) != geometry_prefix_g
                or not forced_source_indices
                or forced_source_indices[0] < 0
                or forced_source_indices[-1] >= num_frames
            ):
                raise ValueError(
                    "explicit geometry source indices must contain exactly "
                    f"G={geometry_prefix_g} valid frame indices"
                )
            geometry_source_mask.zero_()
            geometry_source_mask[list(forced_source_indices)] = 1.0
        return (
            prefix_k,
            hdmap_mode,
            geometry_mode,
            known_mask,
            geometry_prefix_g,
            geometry_source_mask,
        )

    def _sample_imaginaire_conditioned_prefix(
        self,
        *,
        num_frames: int,
        device: torch.device,
    ) -> tuple[int, Tensor]:
        known = self._current_stage2_known_frame_mask
        if known is None or known.shape != (num_frames,):
            raise RuntimeError(
                "factorized Stage-2 prefix was not prepared before RF training"
            )
        known = known.to(device=device, dtype=torch.float32)
        noisy = 1.0 - known
        return self._current_stage2_prefix_k, noisy

    def _reduce_teacher_forcing_per_frame_loss(
        self,
        per_frame_loss: Tensor,
        noisy_mask_T: Tensor,
    ) -> Tensor:
        return per_frame_loss.sum(dim=1) / noisy_mask_T.sum().clamp_min(1.0)

    def _stage2_spatial_map_geometry(
        self, raw_state: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        data_batch = self._current_stage2_data_batch
        sensor_rays = (
            None
            if data_batch is None
            else data_batch.get("sensor_ray_directions")
        )
        if sensor_rays is not None:
            rays = torch.as_tensor(
                sensor_rays,
                device=raw_state.device,
                dtype=torch.float32,
            )
            if rays.ndim != 4 or rays.shape[-1] != 3:
                raise ValueError(
                    "sensor_ray_directions must be (B,H,W,3), got "
                    f"{tuple(rays.shape)}"
                )
            if tuple(rays.shape[1:3]) != tuple(raw_state.shape[-2:]):
                raise ValueError(
                    "sensor rays must match the native range grid: "
                    f"rays={tuple(rays.shape[1:3])}, "
                    f"video={tuple(raw_state.shape[-2:])}"
                )
            dataset_ids = data_batch.get("sensor_dataset_id")
            if dataset_ids is None:
                raise KeyError(
                    "sensor rays require sensor_dataset_id metadata"
                )
            dataset_ids = torch.as_tensor(
                dataset_ids, device=raw_state.device, dtype=torch.long
            ).reshape(-1)
            if dataset_ids.shape[0] != rays.shape[0] or not bool(
                (dataset_ids == dataset_ids[0]).all()
            ):
                raise ValueError(
                    "native-grid batches must contain one homogeneous sensor"
                )
            dataset_id = int(dataset_ids[0].item())
            key = (
                dataset_id,
                int(rays.shape[1]),
                int(rays.shape[2]),
                str(raw_state.device),
            )
            geometry = self._sensor_spatial_map_geometry.get(key)
            if geometry is None:
                ray_grid = rays[0].contiguous()
                elev, azim, row_off = (
                    _sensor_tables_from_ray_directions(
                        ray_grid, allow_approximate=True
                    )
                )
                geometry = (ray_grid, elev, azim, row_off)
                self._sensor_spatial_map_geometry[key] = geometry
            return geometry

        if self._blockwise_spatial_map_geometry is None:
            self._blockwise_spatial_map_geometry = load_waymo_lidar_geometry(
                root_path=self._blockwise_spatial_map_geometry_root_path,
                filter_list_path=(
                    self._blockwise_spatial_map_geometry_filter_list_path
                ),
                resample_width=(
                    self._blockwise_spatial_map_geometry_resample_width
                ),
            )
        if not isinstance(
            self._blockwise_spatial_map_geometry[0], torch.Tensor
        ) or self._blockwise_spatial_map_geometry[0].device != raw_state.device:
            self._blockwise_spatial_map_geometry = tuple(
                torch.as_tensor(
                    value,
                    device=raw_state.device,
                    dtype=torch.float32,
                )
                for value in self._blockwise_spatial_map_geometry
            )
        return self._blockwise_spatial_map_geometry

    @torch.no_grad()
    def _store_blockwise_spatial_map_history(
        self,
        warped: Tensor,
        geometry_source: Optional[Tensor],
        geometry_confidence: Optional[Tensor],
        latent_state: Tensor,
    ) -> None:
        super()._store_blockwise_spatial_map_history(
            warped,
            geometry_source,
            geometry_confidence,
            latent_state,
        )
        if self._current_stage2_geometry_mode == "null":
            self._current_blockwise_warped_history = torch.zeros_like(
                self._current_blockwise_warped_history
            )

    @torch.no_grad()
    def _build_torch_spatial_map_history(
        self,
        raw_state: Tensor,
        lidar_to_world: Tensor,
    ) -> tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        data_batch = self._current_stage2_data_batch
        if data_batch is None:
            raise RuntimeError("factorized Stage-2 batch is unavailable")
        geometry_source_mask = self._current_stage2_geometry_source_mask
        if geometry_source_mask is None:
            geometry_source_mask = self._current_stage2_known_frame_mask
        if geometry_source_mask is None:
            raise RuntimeError(
                "factorized Stage-2 geometry source mask is unavailable"
            )
        ray_dirs, elev, azim, row_off = self._stage2_spatial_map_geometry(
            raw_state
        )
        required = ()
        if self._current_stage2_geometry_mode in (
            "rolling_static",
            "frozen_object",
        ):
            required = (
                "moving_box_object_to_world",
                "moving_box_lwh",
                "moving_box_valid",
                "moving_box_track_id",
            )
        if self.factorized_geometry_rolling_shutter:
            required = (*required, "pose_sensor_start_end")
        missing = [name for name in required if data_batch.get(name) is None]
        if missing:
            raise KeyError(
                f"factorized Stage-2 geometry is missing batch fields {missing}"
            )
        warped = []
        diagnostics: dict[str, list[Tensor]] = {}
        motion_batch = (
            data_batch.get("pose_sensor_start_end")
            if self.factorized_geometry_rolling_shutter
            else None
        )
        geometry_source_pose_batch = data_batch.get(
            "factorized_geometry_source_lidar_to_world"
        )
        geometry_source_motion_batch = (
            data_batch.get(
                "factorized_geometry_source_pose_sensor_start_end"
            )
            if self.factorized_geometry_rolling_shutter
            else None
        )
        depth_bounds_batch = data_batch.get("sensor_depth_bounds")
        if depth_bounds_batch is not None:
            depth_bounds_batch = torch.as_tensor(
                depth_bounds_batch,
                device=raw_state.device,
                dtype=torch.float32,
            )
            if depth_bounds_batch.shape != (raw_state.shape[0], 2):
                raise ValueError(
                    "sensor_depth_bounds must be (B,2), got "
                    f"{tuple(depth_bounds_batch.shape)}"
                )
        for batch_idx in range(raw_state.shape[0]):
            geometry_state = raw_state[batch_idx]
            if (
                self.factorized_rolling_geometry_known_prefix_only
                and self._current_stage2_geometry_mode == "rolling_static"
            ):
                known = self._current_stage2_known_frame_mask
                if known is None:
                    raise RuntimeError(
                        "known-prefix-only rolling geometry requires a "
                        "known-frame mask"
                    )
                geometry_state = geometry_state.clone()
                geometry_state[
                    :,
                    ~known.to(
                        device=geometry_state.device,
                        dtype=torch.bool,
                    ),
                ] = float(self._blockwise_spatial_map_min_value)
            box_object_to_world = data_batch.get(
                "moving_box_object_to_world"
            )
            box_lwh = data_batch.get("moving_box_lwh")
            box_valid = data_batch.get("moving_box_valid")
            box_track_id = data_batch.get("moving_box_track_id")
            min_range = self._blockwise_spatial_map_min_range
            max_range = self._blockwise_spatial_map_max_range
            radius_m = self._blockwise_spatial_map_radius_m
            if depth_bounds_batch is not None:
                min_range = float(depth_bounds_batch[batch_idx, 0].item())
                max_range = float(depth_bounds_batch[batch_idx, 1].item())
                radius_m = max_range
            geometry_i, _, diagnostic_i = (
                build_stage2_factorized_geometry(
                    geometry_state,
                    lidar_to_world[batch_idx],
                    geometry_source_mask,
                    self._current_stage2_geometry_mode,
                    pose_sensor_start_end=(
                        motion_batch[batch_idx]
                        if motion_batch is not None
                        else None
                    ),
                    geometry_source_lidar_to_world=(
                        geometry_source_pose_batch[batch_idx]
                        if geometry_source_pose_batch is not None
                        else None
                    ),
                    geometry_source_pose_sensor_start_end=(
                        geometry_source_motion_batch[batch_idx]
                        if geometry_source_motion_batch is not None
                        else None
                    ),
                    actor_box_time_mode=(
                        self.factorized_geometry_actor_box_time_mode
                    ),
                    static_only_geometry=(
                        self.factorized_static_only_geometry
                    ),
                    box_object_to_world=(
                        None
                        if box_object_to_world is None
                        else box_object_to_world[batch_idx]
                    ),
                    box_lwh=(
                        None if box_lwh is None else box_lwh[batch_idx]
                    ),
                    box_valid=(
                        None if box_valid is None else box_valid[batch_idx]
                    ),
                    box_track_id=(
                        None
                        if box_track_id is None
                        else box_track_id[batch_idx]
                    ),
                    block_size=self.factorized_geometry_block_size,
                    prefix_frames=self.factorized_geometry_prefix_frames,
                    input_has_prepend=self.input_video_has_prepend,
                    ray_dirs=ray_dirs,
                    row_elevations_rad=elev,
                    column_azimuths_rad=azim,
                    row_azimuth_offsets_rad=row_off,
                    min_range=min_range,
                    max_range=max_range,
                    min_value=self._blockwise_spatial_map_min_value,
                    inverse_depth=self._blockwise_spatial_map_inverse_depth,
                    radius_m=radius_m,
                    voxel_size_m=self._blockwise_spatial_map_voxel_m,
                    voxel_policy=self._blockwise_spatial_map_voxel_policy,
                    observation_unit=(
                        self._blockwise_spatial_map_observation_unit
                    ),
                    static_min_observations=(
                        self._blockwise_spatial_map_static_min_observations
                    ),
                    candidate_ttl_updates=(
                        self._blockwise_spatial_map_candidate_ttl_updates
                    ),
                    include_latest=self._blockwise_spatial_map_include_latest,
                    free_space_clear=(
                        self._blockwise_spatial_map_free_space_clear
                    ),
                    free_space_margin_m=(
                        self._blockwise_spatial_map_free_space_margin_m
                    ),
                    free_space_misses_to_remove=(
                        self._blockwise_spatial_map_free_space_misses_to_remove
                    ),
                )
            )
            warped.append(geometry_i)
            for name, value in diagnostic_i.items():
                diagnostics.setdefault(name, []).append(value)
        self._current_stage2_geometry_diagnostics = {
            name: torch.stack(values).mean()
            for name, values in diagnostics.items()
        }
        return torch.stack(warped), None, None

    def get_data_and_condition(
        self, data_batch: dict[str, Tensor]
    ):
        video = data_batch.get(self.input_data_key)
        if video is None or video.ndim != 5:
            raise ValueError(
                "factorized Stage 2 requires video input (B,C,T,H,W)"
            )
        (
            prefix_k,
            hdmap_mode,
            geometry_mode,
            known_mask,
            geometry_prefix_g,
            geometry_source_mask,
        ) = self._sample_factorized_modes(
            num_frames=int(video.shape[2]),
            device=video.device,
        )
        self._current_stage2_prefix_k = prefix_k
        self._current_stage2_geometry_prefix_g = geometry_prefix_g
        self._current_stage2_geometry_source_mask = geometry_source_mask
        self._current_stage2_hdmap_mode = hdmap_mode
        self._current_stage2_geometry_mode = geometry_mode
        self._current_stage2_known_frame_mask = known_mask
        self._current_stage2_task_frame_type_ids = None
        self._current_stage2_data_batch = data_batch

        if hdmap_mode == "boxes":
            semantic = data_batch.get("lidar_hdmap_semantic")
            if semantic is None or semantic.ndim != 5:
                raise KeyError(
                    "boxes-only HDMap requires semantic HDMap (B,K,T,H,W)"
                )
            channel_start = self.stage2_factorized_hdmap_bbox_channel_start
            if channel_start >= semantic.shape[1]:
                raise ValueError(
                    "bbox HDMap channel start is outside the semantic tensor"
                )
            data_batch["lidar_hdmap"] = compose_normalized_range_channels(
                semantic[:, channel_start:],
                channel_dim=1,
                min_value=self._blockwise_spatial_map_min_value,
                inverse_depth=self._blockwise_spatial_map_inverse_depth,
            )

        previous_force_zero = getattr(
            self, "_hdmap_force_zero_latent", False
        )
        self._hdmap_force_zero_latent = hdmap_mode == "null"
        try:
            result = super().get_data_and_condition(data_batch)
            truncated = data_batch.get("moving_box_truncated")
            if truncated is not None:
                if isinstance(truncated, Tensor):
                    self._current_stage2_truncated_boxes = float(
                        truncated.float().sum().item()
                    )
                else:
                    self._current_stage2_truncated_boxes = float(
                        sum(truncated)
                    )
            return result
        finally:
            self._hdmap_force_zero_latent = previous_force_zero
            self._current_stage2_data_batch = None

    def _append_factorized_metrics(
        self,
        output_batch: dict[str, Tensor],
        loss: Tensor,
    ) -> tuple[dict[str, Tensor], Tensor]:
        namespace = getattr(
            self, "_factorized_metric_namespace", "stage2"
        )
        output_batch[f"metric/{namespace}_prefix_k"] = loss.new_tensor(
            float(self._current_stage2_prefix_k)
        )
        output_batch[
            f"metric/{namespace}_geometry_prefix_g"
        ] = loss.new_tensor(float(self._current_stage2_geometry_prefix_g))
        output_batch[
            f"metric/{namespace}_geometry_observations_decoupled"
        ] = loss.new_tensor(
            float(self.factorized_decouple_geometry_observations)
        )
        output_batch[
            f"metric/{namespace}_geometry_rolling_shutter"
        ] = loss.new_tensor(
            float(self.factorized_geometry_rolling_shutter)
        )
        for mode in self._HDMAP_MODES:
            output_batch[
                f"metric/{namespace}_hdmap_{mode}"
            ] = loss.new_tensor(
                float(mode == self._current_stage2_hdmap_mode)
            )
        for mode in self._GEOMETRY_MODES:
            output_batch[
                f"metric/{namespace}_geometry_{mode}"
            ] = loss.new_tensor(
                float(mode == self._current_stage2_geometry_mode)
            )
        for name, value in self._current_stage2_geometry_diagnostics.items():
            output_batch[f"metric/{namespace}_{name}"] = value.to(
                device=loss.device
            )
        output_batch[f"metric/{namespace}_truncated_moving_boxes"] = (
            loss.new_tensor(self._current_stage2_truncated_boxes)
        )
        return output_batch, loss

    def _rectified_flow_forward(
        self, data_batch: dict[str, Tensor]
    ) -> tuple[dict[str, Tensor], Tensor]:
        output_batch, loss = super()._rectified_flow_forward(data_batch)
        return self._append_factorized_metrics(output_batch, loss)


class HdmapToLidarFactorizedBidirectionalRFModel(
    HdmapToLidarFactorizedTeacherForcingRFModel
):
    """Bidirectional RF foundation with the factorized Stage-2 contract."""

    def __init__(self, config):
        super().__init__(config)
        self._factorized_metric_namespace = "stage1"
        if not config.factorized_bidirectional_stage1:
            raise ValueError(
                "factorized bidirectional Stage 1 requires its explicit opt-in"
            )
        full_attention_frames = int(config.latent_shape[1])
        net = self.net
        while hasattr(net, "module"):
            net = net.module
        if (
            self.df_block_size != full_attention_frames
            or net.temporal_block_size != 0
            or net.temporal_prefix_frames != 0
        ):
            raise ValueError(
                "factorized bidirectional Stage 1 requires native unmasked "
                "attention over the complete latent sequence: "
                f"model_block={self.df_block_size}, "
                f"net={net.temporal_prefix_frames}+N*"
                f"{net.temporal_block_size}, T={full_attention_frames}"
            )

    def _rectified_flow_forward(
        self, data_batch: dict[str, Tensor]
    ) -> tuple[dict[str, Tensor], Tensor]:
        _, x0, condition = self.get_data_and_condition(data_batch)
        batch_size, _, num_frames = x0.shape[:3]
        if num_frames != self.df_block_size:
            raise ValueError(
                "factorized bidirectional Stage 1 requires the complete "
                f"training clip, got T={num_frames}, "
                f"expected={self.df_block_size}"
            )
        conditioned_prefix_frames, noisy_mask_T = (
            self._sample_imaginaire_conditioned_prefix(
                num_frames=num_frames,
                device=x0.device,
            )
        )

        epsilon = torch.randn(x0.shape, **self.tensor_kwargs_fp32)
        source, source_weight, source_dropout_fraction = (
            self._sample_teacher_forcing_source(x0, epsilon)
        )
        sampled_time = self.rectified_flow.sample_train_time(batch_size)
        timesteps_B = self.rectified_flow.get_discrete_timestamp(
            sampled_time, self.tensor_kwargs_fp32
        ).reshape(batch_size)
        sigmas_B = self.rectified_flow.get_sigmas(
            timesteps_B, self.tensor_kwargs_fp32
        ).reshape(batch_size)
        time_weights = self.rectified_flow.train_time_weight(
            timesteps_B[:, None], self.tensor_kwargs_fp32
        ).reshape(batch_size, -1).mean(dim=1)

        x0_float = x0.float()
        sigma_B_T = sigmas_B[:, None] * noisy_mask_T[None]
        sigma_5d = sigma_B_T[:, None, :, None, None]
        x_t = sigma_5d * source + (1.0 - sigma_5d) * x0_float
        target = source - x0_float
        active_timesteps = timesteps_B[:, None] * noisy_mask_T[None]
        prediction = self._rf_network_forward(
            x_t.to(**self.tensor_kwargs),
            active_timesteps,
            condition,
        ).float()

        per_frame_loss = (
            prediction - target
        ).square().mean(dim=(1, 3, 4)) * noisy_mask_T[None]
        per_instance_loss = self._reduce_teacher_forcing_per_frame_loss(
            per_frame_loss, noisy_mask_T
        )
        raw_loss = (time_weights * per_instance_loss).mean()
        loss = self.teacher_forcing_loss_scale * raw_loss
        output_batch = {
            "x0": x0,
            "xt": x_t,
            "sigma": sigmas_B,
            "model_pred": prediction,
            "edm_loss": loss,
            "teacher_forcing_loss": raw_loss,
            "timesteps": timesteps_B,
            "per_frame_loss": per_frame_loss,
            "per_instance_loss": per_instance_loss,
            "condition_mask": 1.0 - noisy_mask_T,
            "metric/conditioned_prefix_frames": loss.new_tensor(
                float(conditioned_prefix_frames)
            ),
            "metric/stage1_bidirectional_full_attention": loss.new_tensor(
                1.0
            ),
        }
        if source_weight is not None:
            output_batch.update(
                {
                    "metric/geometry_source_confidence_mean": (
                        self._current_blockwise_geometry_confidence.mean()
                    ),
                    "metric/geometry_source_weight_mean": source_weight.mean(),
                    "metric/geometry_source_coverage": (
                        source_weight > 0.0
                    ).float().mean(),
                    "metric/geometry_source_dropout_fraction": (
                        source_dropout_fraction
                    ),
                }
            )
        return self._append_factorized_metrics(output_batch, loss)


class HdmapToLidarFactorizedBidirectionalIntensityRFModel(
    HdmapToLidarFactorizedBidirectionalRFModel
):
    """Bidirectional factorized model with separate range/intensity VAEs."""

    def build_model(self) -> torch.nn.ModuleDict:
        model = super().build_model()
        range_channels = int(self.vae.latent_ch)
        intensity_vae = getattr(self, "intensity_vae", None)
        if intensity_vae is None:
            raise ValueError("intensity model requires config.intensity_vae")
        intensity_channels = int(intensity_vae.latent_ch)
        if range_channels <= 0 or intensity_channels <= 0:
            raise ValueError(
                "range and intensity tokenizer widths must both be positive"
            )
        combined_channels = range_channels + intensity_channels
        old_external_channels = 3 * range_channels + 1
        new_external_channels = 3 * range_channels + 2 * intensity_channels + 1
        internal_channels = 2
        # Old external input: [range target, HDMap, range geometry, known mask].
        # New external input: [range target, intensity target, HDMap,
        #                      range geometry, intensity geometry, known mask].
        # LVG then appends its condition mask and MinimalV4 appends its padding
        # mask; preserve both pretrained internal channels at the end.
        input_projection = model["net"].x_embedder.proj[1]
        input_projection._checkpoint_channel_remap = (
            "input",
            old_external_channels + internal_channels,
            new_external_channels + internal_channels,
            (
                (0, range_channels, 0, range_channels),
                (
                    range_channels,
                    2 * range_channels,
                    combined_channels,
                    combined_channels + range_channels,
                ),
                (
                    2 * range_channels,
                    3 * range_channels,
                    combined_channels + range_channels,
                    combined_channels + 2 * range_channels,
                ),
                (
                    3 * range_channels,
                    old_external_channels,
                    new_external_channels - 1,
                    new_external_channels,
                ),
                (
                    old_external_channels,
                    old_external_channels + internal_channels,
                    new_external_channels,
                    new_external_channels + internal_channels,
                ),
            ),
        )
        # Preserve the pretrained range head exactly. The new intensity head
        # starts at zero and is learned during fine-tuning.
        model["net"].final_layer.linear._checkpoint_channel_remap = (
            "output",
            range_channels,
            combined_channels,
            ((0, range_channels, 0, range_channels),),
        )
        return model


@diffusion_fsdp_class_decorator
class FSDPHdmapToLidarRectifiedFlowModel(HdmapToLidarRectifiedFlowModel):
    inference_model_cls = HdmapToLidarRectifiedFlowModel


@diffusion_fsdp_class_decorator
class FSDPHdmapToLidarAutoregressiveRFModel(
    HdmapToLidarAutoregressiveRFModel
):
    inference_model_cls = HdmapToLidarAutoregressiveRFModel


@diffusion_fsdp_class_decorator
class FSDPHdmapToLidarLaGenSDERFModel(
    HdmapToLidarLaGenSDERFModel
):
    inference_model_cls = HdmapToLidarLaGenSDERFModel


@diffusion_fsdp_class_decorator
class FSDPHdmapToLidarDiffusionForcingRFModel(
    HdmapToLidarDiffusionForcingRFModel
):
    pass


@diffusion_fsdp_class_decorator
class FSDPHdmapToLidarTeacherForcingRFModel(
    HdmapToLidarTeacherForcingRFModel
):
    pass


@diffusion_fsdp_class_decorator
class FSDPHdmapToLidarImaginaireTeacherForcingRFModel(
    HdmapToLidarImaginaireTeacherForcingRFModel
):
    pass


@diffusion_fsdp_class_decorator
class FSDPHdmapToLidarImaginaireOnePassTeacherForcingRFModel(
    HdmapToLidarImaginaireOnePassTeacherForcingRFModel
):
    pass


@diffusion_fsdp_class_decorator
class FSDPHdmapToLidarConcatGeometryTeacherForcingRFModel(
    HdmapToLidarConcatGeometryTeacherForcingRFModel
):
    pass


@diffusion_fsdp_class_decorator
class FSDPHdmapToLidarMultiTaskTeacherForcingRFModel(
    HdmapToLidarMultiTaskTeacherForcingRFModel
):
    pass


@diffusion_fsdp_class_decorator
class FSDPHdmapToLidarFactorizedTeacherForcingRFModel(
    HdmapToLidarFactorizedTeacherForcingRFModel
):
    pass


@diffusion_fsdp_class_decorator
class FSDPHdmapToLidarFactorizedBidirectionalRFModel(
    HdmapToLidarFactorizedBidirectionalRFModel
):
    pass


@diffusion_fsdp_class_decorator
class FSDPHdmapToLidarFactorizedBidirectionalIntensityRFModel(
    HdmapToLidarFactorizedBidirectionalIntensityRFModel
):
    pass
