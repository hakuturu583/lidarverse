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

from typing import List, Optional, Tuple

import torch
from torch import nn

from diffusion.conditioner import DataType
from diffusion.training.networks.minimal_v4_dit import MiniTrainDIT


class MinimalV1LVGDiT(MiniTrainDIT):
    def __init__(self, *args, timestep_scale: float = 1.0, rgb_drop_pad: bool = False, **kwargs):
        assert "in_channels" in kwargs, "in_channels must be provided"
        # Data (latent) channel count BEFORE the internal condition-mask channel
        # is appended below; the learnable drop-pad lives in this latent space
        # (substituted pre-mask-cat in ``forward``).
        data_channels = kwargs["in_channels"]
        kwargs["in_channels"] += 1  # Add 1 for the condition mask
        self.timestep_scale = timestep_scale
        super().__init__(*args, **kwargs)
        # Learnable placeholder for DROPPED RGB condition frames. The diffusion
        # operates in *normalized* latent space, so plain zeros == the per-
        # channel latent mean (an in-distribution value, a poor "absent"
        # signal). A ZERO-INIT learnable token starts bit-identical to the
        # model's zeroing of dropped frames and drifts to a distinct "this
        # frame is intentionally absent" code. Substituted in ``forward`` at the
        # frames flagged by ``frame_drop_mask``. Default off => no param built.
        self.use_rgb_drop_pad = bool(rgb_drop_pad)
        if self.use_rgb_drop_pad:
            self.rgb_drop_pad = nn.Parameter(torch.zeros(1, data_channels, 1, 1, 1))

    def forward(
        self,
        x_B_C_T_H_W: torch.Tensor,
        timesteps_B_T: torch.Tensor,
        crossattn_emb: torch.Tensor,
        condition_video_input_mask_B_C_T_H_W: Optional[torch.Tensor] = None,
        side_condition_B_C_T_H_W: Optional[torch.Tensor] = None,
        fps: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        data_type: Optional[DataType] = DataType.VIDEO,
        intermediate_feature_ids: Optional[List[int]] = None,
        img_context_emb: Optional[torch.Tensor] = None,
        frame_is_rgb: Optional[torch.Tensor] = None,
        temporal_positions: Optional[torch.Tensor] = None,
        frame_type_ids: Optional[torch.Tensor] = None,
        frame_drop_mask: Optional[torch.Tensor] = None,
        img_context_pos: Optional[torch.Tensor] = None,
        ego_pose_B_T_P: Optional[torch.Tensor] = None,
        ego_pose_null_mask_B: Optional[torch.Tensor] = None,
        dataset_ids_B: Optional[torch.Tensor] = None,
        self_attn_kv_cache: Optional[
            List[Optional[dict[str, torch.Tensor]]]
        ] = None,
        store_self_attn_kv: bool = False,
        self_attn_kv_max_frames: Optional[int] = None,
        self_attn_kv_sink_frames: int = 0,
        detach_self_attn_kv: bool = True,
        return_self_attn_kv: bool = False,
        teacher_forcing_clean_frames: Optional[int] = None,
        **kwargs,
    ) -> torch.Tensor | List[torch.Tensor] | Tuple[torch.Tensor, List[torch.Tensor]]:
        del kwargs

        # Replace DROPPED RGB condition frames' latent content with the learnable
        # placeholder (in normalized latent space, BEFORE the mask channel is
        # appended). ``frame_drop_mask`` is (B, T) bool, True on dropped frames.
        # Zero-init => no-op at iter 0 (matches the model's frame zeroing).
        if (
            frame_drop_mask is not None
            and getattr(self, "use_rgb_drop_pad", False)
        ):
            B, _, T = x_B_C_T_H_W.shape[:3]
            assert frame_drop_mask.shape == (B, T), (
                f"frame_drop_mask {tuple(frame_drop_mask.shape)} must be (B={B}, T={T})"
            )
            dm = frame_drop_mask.to(device=x_B_C_T_H_W.device, dtype=torch.bool).view(B, 1, T, 1, 1)
            x_B_C_T_H_W = torch.where(dm, self.rgb_drop_pad.type_as(x_B_C_T_H_W), x_B_C_T_H_W)

        if data_type == DataType.VIDEO:
            x_B_C_T_H_W = torch.cat([x_B_C_T_H_W, condition_video_input_mask_B_C_T_H_W.type_as(x_B_C_T_H_W)], dim=1)
        else:
            B, _, T, H, W = x_B_C_T_H_W.shape
            x_B_C_T_H_W = torch.cat(
                [x_B_C_T_H_W, torch.zeros((B, 1, T, H, W), dtype=x_B_C_T_H_W.dtype, device=x_B_C_T_H_W.device)], dim=1
            )
        return super().forward(
            x_B_C_T_H_W=x_B_C_T_H_W,
            timesteps_B_T=timesteps_B_T * self.timestep_scale,
            crossattn_emb=crossattn_emb,
            side_condition_B_C_T_H_W=side_condition_B_C_T_H_W,
            fps=fps,
            padding_mask=padding_mask,
            data_type=data_type,
            intermediate_feature_ids=intermediate_feature_ids,
            img_context_emb=img_context_emb,
            frame_is_rgb=frame_is_rgb,
            temporal_positions=temporal_positions,
            frame_type_ids=frame_type_ids,
            img_context_pos=img_context_pos,
            ego_pose_B_T_P=ego_pose_B_T_P,
            ego_pose_null_mask_B=ego_pose_null_mask_B,
            dataset_ids_B=dataset_ids_B,
            self_attn_kv_cache=self_attn_kv_cache,
            store_self_attn_kv=store_self_attn_kv,
            self_attn_kv_max_frames=self_attn_kv_max_frames,
            self_attn_kv_sink_frames=self_attn_kv_sink_frames,
            detach_self_attn_kv=detach_self_attn_kv,
            return_self_attn_kv=return_self_attn_kv,
            teacher_forcing_clean_frames=teacher_forcing_clean_frames,
        )
