# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LVGDiT wrapper for hdmap-conditioned lidar generation.

Adapts MinimalV1LVGDiT's forward signature to the interface used by
HdmapToLidarRectifiedFlowModel._rf_network_forward:

    net(x=..., timesteps=..., **condition.to_dict())

Also exposes ``fsdp_wrap_block_cls`` for FSDP auto-wrap.
"""

from typing import List, Optional, Tuple, Union

import torch

from diffusion.conditioner import DataType
from diffusion.training.networks.minimal_v1_lvg_dit import MinimalV1LVGDiT
from diffusion.training.networks.minimal_v4_dit import Block


class LVGDiTForLidar(MinimalV1LVGDiT):
    """MinimalV1LVGDiT with a GeneralDIT-compatible forward interface.

    The hdmap channel-concat is handled externally by ``_rf_network_forward``
    *before* calling the network, so ``x`` already contains (latent + hdmap)
    channels.  MinimalV1LVGDiT adds +1 to ``in_channels`` for an internal
    condition-mask channel; we satisfy that by always passing
    ``data_type=IMAGE`` which concatenates a zero-channel.

    Image-conditioned (img2lidar-style) usage instead passes an explicit
    ``condition_video_input_mask`` (1 on RGB condition frames, 0 on the
    generated lidar frame); when present we route through the native
    ``data_type=VIDEO`` path so ``MinimalV1LVGDiT`` concatenates *that* mask
    (not a zero) as the +1 channel and the DiT attends jointly over the
    RGB + lidar frames.
    """

    fsdp_wrap_block_cls = Block

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        crossattn_emb: torch.Tensor,
        crossattn_mask: Optional[torch.Tensor] = None,
        data_type: Optional[DataType] = None,
        padding_mask: Optional[torch.Tensor] = None,
        fps: Optional[torch.Tensor] = None,
        condition_video_input_mask: Optional[torch.Tensor] = None,
        side_condition: Optional[torch.Tensor] = None,
        frame_is_rgb: Optional[torch.Tensor] = None,
        temporal_positions: Optional[torch.Tensor] = None,
        frame_type_ids: Optional[torch.Tensor] = None,
        frame_drop_mask: Optional[torch.Tensor] = None,
        img_context_emb: Optional[torch.Tensor] = None,
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
    ) -> Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, List[torch.Tensor]]]:
        if condition_video_input_mask is not None:
            return super().forward(
                x_B_C_T_H_W=x,
                timesteps_B_T=timesteps,
                crossattn_emb=crossattn_emb,
                condition_video_input_mask_B_C_T_H_W=condition_video_input_mask,
                side_condition_B_C_T_H_W=side_condition,
                fps=fps,
                padding_mask=padding_mask,
                data_type=DataType.VIDEO,
                frame_is_rgb=frame_is_rgb,
                temporal_positions=temporal_positions,
                frame_type_ids=frame_type_ids,
                frame_drop_mask=frame_drop_mask,
                img_context_emb=img_context_emb,
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
        return super().forward(
            x_B_C_T_H_W=x,
            timesteps_B_T=timesteps,
            crossattn_emb=crossattn_emb,
            condition_video_input_mask_B_C_T_H_W=None,
            side_condition_B_C_T_H_W=side_condition,
            fps=fps,
            padding_mask=padding_mask,
            data_type=DataType.IMAGE,
            temporal_positions=temporal_positions,
            frame_type_ids=frame_type_ids,
            dataset_ids_B=dataset_ids_B,
            self_attn_kv_cache=self_attn_kv_cache,
            store_self_attn_kv=store_self_attn_kv,
            self_attn_kv_max_frames=self_attn_kv_max_frames,
            self_attn_kv_sink_frames=self_attn_kv_sink_frames,
            detach_self_attn_kv=detach_self_attn_kv,
            return_self_attn_kv=return_self_attn_kv,
            teacher_forcing_clean_frames=teacher_forcing_clean_frames,
        )
