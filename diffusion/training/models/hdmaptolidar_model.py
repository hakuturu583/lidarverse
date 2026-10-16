# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hdmap-to-Lidar diffusion model.

Conditions on an HD-map via channel concatenation: the hdmap is resized to
the latent spatial resolution and concatenated with the noisy latent along
the channel dimension before the network.

Three hdmap-conditioning modes (mutually exclusive, evaluated in order):

* ``encode_hdmap_with_vae=True`` — run the hdmap through the same frozen VAE
  encoder as the lidar. The encoded latent has the same shape as the noisy
  lidar latent, so no resize / 1x1 conv is required; set
  ``additional_concat_ch`` to the VAE's ``latent_ch`` and
  ``net.in_channels = latent_ch * 2``.
* ``skip_hdmap_proj=True`` — bypass the projection and directly concat the
  raw (spatially resized) hdmap; avoids randomly-initialised weights
  corrupting the conditioning signal early in training.
* Default — a learnable 1x1 conv projects ``hdmap_channels`` → 
  ``additional_concat_ch`` before concat.

Uses the standard ``GeneralDIT`` network (no multiview extensions).
"""

from typing import Tuple, Union

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn

from diffusion.functional.batch_ops import batch_mul
from diffusion.training.conditioner import BaseVideoCondition
from diffusion.training.models.model import DiffusionModel, _broadcast, broadcast_condition
from diffusion.training.models.model_image import (
    CosmosCondition,
    DenoisePrediction,
    diffusion_fsdp_class_decorator,
)
from diffusion.conditioner import DataType
from utils import log, misc


# Fixed class -> RGB palette for the "colored" BEV hdmap render fed to the
# frozen Wan2.2 VAE (must match scripts/hdmap_bev_wan2p2_roundtrip.py so the
# training render == the render validated by the roundtrip test).
# Order = waymo_hdmap.CLASS_NAMES: lane, road_line, road_edge, crosswalk,
# vehicle, pedestrian, cyclist.
_BEV_CLASS_COLORS = (
    (0.20, 0.20, 0.20),  # lane        - dark gray
    (1.00, 1.00, 0.00),  # road_line   - yellow
    (1.00, 1.00, 1.00),  # road_edge   - white
    (0.00, 1.00, 1.00),  # crosswalk   - cyan
    (1.00, 0.00, 0.00),  # vehicle     - red
    (0.00, 1.00, 0.00),  # pedestrian  - green
    (0.00, 0.00, 1.00),  # cyclist     - blue
)


class HdmapToLidarModel(DiffusionModel):
    """Rectified-flow diffusion conditioned on HD-map via channel concat."""

    def __init__(self, config):
        super().__init__(config)

        self.diffusion_dtype = torch.bfloat16

        hdmap_ch = getattr(config, "hdmap_channels", 3)
        additional_concat_ch = getattr(config, "additional_concat_ch", 3)
        self.skip_hdmap_proj = getattr(config, "skip_hdmap_proj", False)
        self.encode_hdmap_with_vae = bool(
            getattr(config, "encode_hdmap_with_vae", False)
        )
        self.bev_hdmap_pool = None
        self.bev_hdmap_wan = None
        if getattr(config, "bev_hdmap_wan_encode", False) or getattr(
            config, "bev_hdmap_pool", False
        ):
            raise ValueError(
                "The public LidarVerse release supports range-rendered HDMap "
                "conditioning only"
            )

        if self.encode_hdmap_with_vae:
            # Hdmap is routed through the same frozen VAE as the lidar.
            # The encoded latent already lives in (B, latent_ch, T_z, H_z, W_z)
            # — no spatial/temporal resize and no learnable projection are
            # required. ``additional_concat_ch`` is bookkeeping here and
            # should equal the VAE's ``latent_ch`` so that the downstream
            # ``net.in_channels = lidar_latent_ch + additional_concat_ch``
            # matches reality. Mutually exclusive with ``skip_hdmap_proj``.
            assert not self.skip_hdmap_proj, (
                "encode_hdmap_with_vae and skip_hdmap_proj are mutually "
                "exclusive."
            )
            self.hdmap_proj = None
            self._hdmap_concat_ch = additional_concat_ch
            log.info(
                f"HdmapToLidarModel: encode_hdmap_with_vae=True, "
                f"concat {additional_concat_ch} VAE-encoded hdmap channels"
            )
        elif self.skip_hdmap_proj:
            self.hdmap_proj = None
            self._hdmap_concat_ch = hdmap_ch
            log.info(
                f"HdmapToLidarModel: raw hdmap concat (no proj), {hdmap_ch} ch"
            )
        elif additional_concat_ch > 0:
            self.hdmap_proj = nn.Conv2d(hdmap_ch, additional_concat_ch, kernel_size=1)
            self._hdmap_concat_ch = additional_concat_ch
            log.info(
                f"HdmapToLidarModel: hdmap_proj {hdmap_ch} -> {additional_concat_ch}"
            )
        else:
            self.hdmap_proj = None
            self._hdmap_concat_ch = 0

        # Per-sample dropout of the hdmap condition during training, enabling
        # classifier-free guidance at inference. Defaults to 0.0 (no dropout)
        # so existing configs remain unchanged.
        self.hdmap_dropout_prob = float(getattr(config, "hdmap_dropout_prob", 0.0))
        # Mixture of dropout regimes; see LidarModelConfig.hdmap_dropout_modes.
        # When dropout fires for a sample we draw one of these uniformly.
        # Legacy default is ("full",) which reproduces the original CFG-only
        # behavior bit-for-bit.
        raw_modes = list(getattr(config, "hdmap_dropout_modes", ["full"]))
        _ALLOWED_MODES = ("full", "frame_bernoulli", "keep_first_k")
        for m in raw_modes:
            if m not in _ALLOWED_MODES:
                raise ValueError(
                    f"hdmap_dropout_modes contains unknown mode {m!r}; "
                    f"allowed: {_ALLOWED_MODES}"
                )
        if len(raw_modes) == 0:
            raise ValueError("hdmap_dropout_modes must not be empty")
        self.hdmap_dropout_modes = raw_modes
        self.hdmap_frame_dropout_rate = float(
            getattr(config, "hdmap_frame_dropout_rate", 0.5)
        )
        self.hdmap_keep_first_k = int(getattr(config, "hdmap_keep_first_k", 1))
        self.hdmap_artifixer_dropout = bool(
            getattr(config, "hdmap_artifixer_dropout", False)
        )
        if self.hdmap_dropout_prob > 0.0:
            log.info(
                f"HdmapToLidarModel: hdmap_dropout_prob={self.hdmap_dropout_prob}, "
                f"hdmap_dropout_modes={self.hdmap_dropout_modes} "
                f"(frame_rate={self.hdmap_frame_dropout_rate}, "
                f"keep_first_k={self.hdmap_keep_first_k}, "
                f"artifixer={self.hdmap_artifixer_dropout})"
            )

        self._current_hdmap = None
        self._num_dataset_embeddings = int(
            getattr(config.net, "num_dataset_embeddings", 0)
        )
        self._current_sensor_dataset_ids: Tensor | None = None
        # Transient flag used by `generate_samples_from_batch` to zero the
        # hdmap for the unconditional CFG pass. Kept off outside inference.
        self._hdmap_force_null = False
        # Inference-only: when set to int K, zero all hdmap latent frames except
        # the first K in get_data_and_condition (see usage there). None = off.
        self._hdmap_keep_first_k_infer = None
        # Inference-only: when set to a list of ints, keep exactly those hdmap
        # latent-frame indices (negatives index from the end). Takes precedence
        # over _hdmap_keep_first_k_infer. None = off.
        self._hdmap_keep_frames_infer = None

    # ------------------------------------------------------------------
    # HD-map helpers
    # ------------------------------------------------------------------

    def _prepare_hdmap(
        self, hdmap: torch.Tensor, latent: torch.Tensor
    ) -> torch.Tensor:
        """Resize hdmap to latent spatial dims and optionally project channels.

        When ``skip_hdmap_proj`` is True the raw resized hdmap is returned
        without any learnable transformation.

        Works for both 4-D ``(B, C, H, W)`` and 5-D ``(B, C, T, H, W)`` inputs.
        """
        is_5d = hdmap.ndim == 5
        if is_5d:
            B, C, T, H, W = hdmap.shape
            lat_T, lat_H, lat_W = latent.shape[2], latent.shape[3], latent.shape[4]
            hdmap = F.interpolate(
                hdmap.reshape(B * T, C, H, W),
                size=(lat_H, lat_W),
                mode="nearest-exact",
            )
            if self.hdmap_proj is not None:
                hdmap = self.hdmap_proj(hdmap)
            _, C_out, _, _ = hdmap.shape
            hdmap = hdmap.reshape(B, C_out, T, lat_H, lat_W)
            if T != lat_T:
                hdmap = F.interpolate(
                    hdmap,
                    size=(lat_T, lat_H, lat_W),
                    mode="nearest-exact",
                )
        else:
            lat_H, lat_W = latent.shape[2], latent.shape[3]
            hdmap = F.interpolate(
                hdmap, size=(lat_H, lat_W), mode="nearest-exact"
            )
            if self.hdmap_proj is not None:
                hdmap = self.hdmap_proj(hdmap)
        return hdmap

    # ------------------------------------------------------------------
    # HD-map training-time dropout (3-mode mixture)
    # ------------------------------------------------------------------

    def _apply_hdmap_dropout(self, hdmap_z: torch.Tensor) -> torch.Tensor:
        """Per-sample dropout with a uniform mixture over enabled modes.

        With probability ``self.hdmap_dropout_prob`` each sample is "fired"
        for dropout. Conditional on firing, a mode is drawn uniformly from
        ``self.hdmap_dropout_modes``:

          * ``"full"``            -- zero the entire hdmap latent
          * ``"frame_bernoulli"`` -- zero each frame independently with rate
                                     ``self.hdmap_frame_dropout_rate``
          * ``"keep_first_k"``    -- zero all frames except the first
                                     ``self.hdmap_keep_first_k`` frames

        Frame-wise modes require a temporal axis at ``hdmap_z.shape[2]``;
        image-mode inputs (4-D) silently collapse to the legacy "full" path
        regardless of the configured mixture.
        """
        B = hdmap_z.shape[0]
        device = hdmap_z.device
        if self.hdmap_artifixer_dropout:
            # ArtiFixer samples one conditioning decision for the whole batch.
            # Its RGB condition is masked before VAE encoding and accompanied by
            # a separate opacity mask. LiDAR HDMap conditioning has no opacity
            # channel, so the behaviorally equivalent safe representation is a
            # zero latent suffix (the existing unconditional HDMap convention).
            if bool(
                torch.rand((), device=device) < self.hdmap_dropout_prob
            ):
                return torch.zeros_like(hdmap_z)
            if hdmap_z.ndim != 5 or hdmap_z.shape[2] == 1:
                return hdmap_z
            num_frames = hdmap_z.shape[2]
            keep_frames = int(
                torch.randint(
                    1, num_frames + 1, (1,), device=device
                ).item()
            )
            frame_keep = torch.zeros(
                (1, 1, num_frames, 1, 1),
                device=device,
                dtype=hdmap_z.dtype,
            )
            frame_keep[:, :, :keep_frames] = 1.0
            return hdmap_z * frame_keep

        fire = torch.rand(B, device=device) < self.hdmap_dropout_prob  # (B,)
        if not fire.any():
            return hdmap_z

        is_video = hdmap_z.ndim == 5
        T_lat = hdmap_z.shape[2] if is_video else 1
        # Frame-wise modes are meaningless for 4-D / T=1 hdmaps; fall back
        # to the legacy full-dropout path in that case.
        if (not is_video) or T_lat == 1:
            keep_sample = (~fire).to(hdmap_z.dtype)
            keep_sample = keep_sample.view(B, *([1] * (hdmap_z.ndim - 1)))
            return hdmap_z * keep_sample

        n_modes = len(self.hdmap_dropout_modes)
        mode_ix = torch.randint(0, n_modes, (B,), device=device)  # (B,)

        # frame_keep[b, t] == 1 means "keep frame t of sample b".
        frame_keep = torch.ones(
            (B, T_lat), device=device, dtype=hdmap_z.dtype
        )

        for ix, mode in enumerate(self.hdmap_dropout_modes):
            sel = fire & (mode_ix == ix)  # (B,)
            if not sel.any():
                continue
            if mode == "full":
                # Zero every frame for this sample.
                drop_mask = torch.zeros(
                    (1, T_lat), device=device, dtype=hdmap_z.dtype
                )
                frame_keep = torch.where(sel.view(B, 1), drop_mask, frame_keep)
            elif mode == "frame_bernoulli":
                per_frame = (
                    torch.rand((B, T_lat), device=device)
                    >= self.hdmap_frame_dropout_rate
                ).to(hdmap_z.dtype)
                frame_keep = torch.where(sel.view(B, 1), per_frame, frame_keep)
            elif mode == "keep_first_k":
                K = max(0, min(int(self.hdmap_keep_first_k), T_lat))
                keep_first = torch.zeros(
                    (1, T_lat), device=device, dtype=hdmap_z.dtype
                )
                if K > 0:
                    keep_first[:, :K] = 1.0
                frame_keep = torch.where(sel.view(B, 1), keep_first, frame_keep)
            else:
                # Caught at __init__; defensive.
                raise ValueError(f"Unknown hdmap dropout mode: {mode!r}")

        # Broadcast to (B, 1, T, 1, 1) and apply.
        frame_keep = frame_keep.view(B, 1, T_lat, 1, 1)
        return hdmap_z * frame_keep

    # ------------------------------------------------------------------
    # Override: encode/decode for video support
    # ------------------------------------------------------------------

    def _encode_with_vae(
        self, vae, state: torch.Tensor
    ) -> torch.Tensor:
        if getattr(vae, "is_image", True):
            return vae.encode(state.squeeze(2)).unsqueeze(2) * self.sigma_data
        return vae.encode(state) * self.sigma_data

    def _decode_with_vae(
        self,
        vae,
        latent: torch.Tensor,
        t: float | torch.Tensor | None = None,
    ) -> torch.Tensor:
        is_ltx = getattr(vae, "is_ltx", False)
        if t is not None and not is_ltx:
            raise ValueError(
                f"Got t={t!r} but {type(vae).__name__} is not LTX-style"
            )
        if getattr(vae, "is_image", True):
            return vae.decode(latent.squeeze(2) / self.sigma_data).unsqueeze(2)
        if is_ltx:
            return vae.decode(latent / self.sigma_data, t=t)
        return vae.decode(latent / self.sigma_data)

    @torch.no_grad()
    def encode(self, state: torch.Tensor) -> torch.Tensor:
        # Disable any outer autocast so the VAE wrapper exclusively controls
        # encoder precision. Existing configs remain FP32; selected configs
        # may opt into a measured BF16 encoder.
        state = state.float()
        with torch.autocast("cuda", enabled=False):
            if self.intensity_vae is None:
                return self._encode_with_vae(self.vae, state)
            if state.shape[1] == 1:
                return self._encode_with_vae(self.vae, state)
            if state.shape[1] != 2:
                raise ValueError(
                    "paired range/intensity encoding expects C=1 or C=2, "
                    f"got {tuple(state.shape)}"
                )
            range_z = self._encode_with_vae(self.vae, state[:, :1])
            intensity_z = self._encode_with_vae(
                self.intensity_vae, state[:, 1:2]
            )
            if range_z.shape[2:] != intensity_z.shape[2:]:
                raise ValueError(
                    "range and intensity tokenizer grids do not align: "
                    f"{tuple(range_z.shape)} vs {tuple(intensity_z.shape)}"
                )
            return torch.cat([range_z, intensity_z], dim=1)

    @torch.no_grad()
    def _encode_stream_with_vae(
        self,
        vae,
        state_new: torch.Tensor,
        temporal_kv_cache=None,
        max_cache_frames: int | None = None,
    ):
        if getattr(vae, "is_image", True):
            raise NotImplementedError(
                "streaming encode requires a causal video VAE"
            )
        if not hasattr(vae, "encode_stream"):
            raise NotImplementedError(
                f"{type(vae).__name__} does not implement encode_stream"
            )
        latent, new_cache = vae.encode_stream(
            state_new,
            temporal_kv_cache=temporal_kv_cache,
            max_cache_frames=max_cache_frames,
        )
        return latent * self.sigma_data, new_cache

    @torch.no_grad()
    def encode_stream(
        self,
        state_new: torch.Tensor,
        temporal_kv_cache=None,
        max_cache_frames: int | None = None,
    ):
        """Encode newly appended frames at the VAE's configured precision."""
        state_new = state_new.float()
        with torch.autocast("cuda", enabled=False):
            if self.intensity_vae is None or state_new.shape[1] == 1:
                return self._encode_stream_with_vae(
                    self.vae,
                    state_new,
                    temporal_kv_cache=temporal_kv_cache,
                    max_cache_frames=max_cache_frames,
                )
            if state_new.shape[1] != 2:
                raise ValueError(
                    "paired range/intensity streaming encode expects C=1 "
                    f"or C=2, got {tuple(state_new.shape)}"
                )
            if temporal_kv_cache is None:
                range_cache = None
                intensity_cache = None
            elif isinstance(temporal_kv_cache, dict) and set(
                temporal_kv_cache
            ) == {"range", "intensity"}:
                range_cache = temporal_kv_cache["range"]
                intensity_cache = temporal_kv_cache["intensity"]
            else:
                raise ValueError(
                    "paired range/intensity streaming cache must contain "
                    "exactly 'range' and 'intensity'"
                )
            range_z, range_cache = self._encode_stream_with_vae(
                self.vae,
                state_new[:, :1],
                temporal_kv_cache=range_cache,
                max_cache_frames=max_cache_frames,
            )
            intensity_z, intensity_cache = self._encode_stream_with_vae(
                self.intensity_vae,
                state_new[:, 1:2],
                temporal_kv_cache=intensity_cache,
                max_cache_frames=max_cache_frames,
            )
            if range_z.shape[2:] != intensity_z.shape[2:]:
                raise ValueError(
                    "range and intensity streaming latent grids do not align: "
                    f"{tuple(range_z.shape)} vs {tuple(intensity_z.shape)}"
                )
            return torch.cat([range_z, intensity_z], dim=1), {
                "range": range_cache,
                "intensity": intensity_cache,
            }

    @torch.no_grad()
    def decode(
        self,
        latent: torch.Tensor,
        t: float | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Decode a diffusion-space latent back to pixels.

        ``t`` (rectified-flow timestep in [0, 1]) is forwarded to the VAE
        wrapper only when ``self.vae.is_ltx`` is True (i.e. the wrapped
        VAE net supports LTX-style timestep-conditioned decoding via
        ``decode(z, t=t)``). For non-LTX wrappers -- which is every
        wrapper today other than ``ConfigBasedVideoVAE`` around a
        ``TransformerVAELTX`` with ``denoising_t_max > 0`` -- passing
        ``t`` raises so we don't silently drop it. Note: T=1 single-
        frame inputs go through the video path unchanged; ``is_ltx`` is
        about the wrapper's decode signature, not the input tensor's
        temporal length.
        """
        if self.intensity_vae is not None:
            latent = latent.float()
            range_channels = int(self.vae.latent_ch)
            intensity_channels = int(self.intensity_vae.latent_ch)
            with torch.autocast("cuda", enabled=False):
                if latent.shape[1] == range_channels:
                    return self._decode_with_vae(self.vae, latent, t=t)
                if latent.shape[1] != range_channels + intensity_channels:
                    raise ValueError(
                        "paired range/intensity decoding expects "
                        f"{range_channels} or "
                        f"{range_channels + intensity_channels} channels, "
                        f"got {tuple(latent.shape)}"
                    )
                if t is not None:
                    raise ValueError(
                        "paired range/intensity decoding does not support "
                        "timestep-conditioned VAE decoding"
                    )
                range_recon = self._decode_with_vae(
                    self.vae, latent[:, :range_channels]
                )
                intensity_recon = self._decode_with_vae(
                    self.intensity_vae, latent[:, range_channels:]
                )
                if range_recon.shape[2:] != intensity_recon.shape[2:]:
                    raise ValueError(
                        "range and intensity decoder grids do not align: "
                        f"{tuple(range_recon.shape)} vs "
                        f"{tuple(intensity_recon.shape)}"
                    )
                return torch.cat([range_recon, intensity_recon], dim=1)
        is_ltx = getattr(self.vae, "is_ltx", False)
        if t is not None and not is_ltx:
            raise ValueError(
                f"Got t={t!r} but the wrapped VAE "
                f"({type(self.vae).__name__}) is not LTX-style "
                f"(is_ltx=False). Either drop --decoder_t1 or swap in a "
                f"TransformerVAELTX checkpoint with denoising_t_max > 0."
            )
        latent = latent.float()
        with torch.autocast("cuda", enabled=False):
            if getattr(self.vae, "is_image", True):
                # 4D image-tokenizer wrapper path (no T dim). LTX-style
                # decoders are never image VAEs in this codebase, so we
                # ignore ``t`` here (the gate above already rejected
                # non-None t for any non-LTX wrapper).
                return super().decode(latent)
            if is_ltx:
                return self.vae.decode(latent / self.sigma_data, t=t)
            return self.vae.decode(latent / self.sigma_data)

    @torch.no_grad()
    def decode_stream(
        self,
        latent_new: torch.Tensor,
        temporal_kv_cache=None,
        max_cache_frames: int | None = None,
    ):
        """Decode only newly appended diffusion-space latent frames."""
        if getattr(self.vae, "is_image", True):
            raise NotImplementedError(
                "streaming decode requires a causal video VAE"
            )
        if not hasattr(self.vae, "decode_stream"):
            raise NotImplementedError(
                f"{type(self.vae).__name__} does not implement decode_stream"
            )
        latent_new = latent_new.float()
        with torch.autocast("cuda", enabled=False):
            decode_kwargs = {"temporal_kv_cache": temporal_kv_cache}
            if max_cache_frames is not None:
                decode_kwargs["max_cache_frames"] = max_cache_frames
            return self.vae.decode_stream(
                latent_new / self.sigma_data,
                **decode_kwargs,
            )

    # ------------------------------------------------------------------
    # BEV hdmap -> Wan2.2 latent
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _bev_hdmap_to_wan_latent(self, bev: torch.Tensor) -> torch.Tensor:
        """``(B, C_cls, T, Y, X)`` {0,1} BEV raster -> ``(B, 48, T, Y/16, X/16)``.

        Renders the per-class occupancy grid into a 3-channel image (colored
        palette or grayscale occupancy), maps to [-1, 1], and encodes with the
        frozen Wan2.2 RGB VAE. Matches the render used in the roundtrip test.
        """
        assert bev.ndim == 5, f"bev_hdmap must be 5-D (B,C,T,Y,X), got {bev.shape}"
        B, C, T, Y, X = bev.shape
        bev = (bev.reshape(B * T, C, Y, X) > 0.5).float()
        if self.bev_hdmap_render_mode == "occupancy":
            occ = bev.amax(dim=1, keepdim=True)  # (BT, 1, Y, X)
            img01 = occ.repeat(1, 3, 1, 1)
        else:  # colored (later classes paint over earlier ones)
            pal = self._bev_palette.to(device=bev.device, dtype=bev.dtype)  # (C, 3)
            img01 = bev.new_zeros(B * T, 3, Y, X)
            n = min(C, pal.shape[0])
            for ci in range(n):
                m = bev[:, ci : ci + 1] > 0.5  # (BT, 1, Y, X)
                img01 = torch.where(m, pal[ci].view(1, 3, 1, 1), img01)
        img = (img01 * 2.0 - 1.0).reshape(B, 3, T, Y, X)  # Wan wants (B,3,T,H,W)
        return self.bev_hdmap_wan.encode(img)

    # ------------------------------------------------------------------
    # Override: data preparation
    # ------------------------------------------------------------------

    def _store_hdmap_latent(self, hdmap_z: Tensor | None) -> None:
        """Apply configured HDMap masking and store the current latent."""
        if hdmap_z is None:
            self._current_hdmap = None
            return

        if self.training and self.hdmap_dropout_prob > 0.0:
            hdmap_z = self._apply_hdmap_dropout(hdmap_z)

        frames_inf = getattr(self, "_hdmap_keep_frames_infer", None)
        k_inf = getattr(self, "_hdmap_keep_first_k_infer", None)
        if (frames_inf is not None or k_inf is not None) and hdmap_z.ndim == 5:
            T_lat = hdmap_z.shape[2]
            frame_keep = torch.zeros(
                (1, 1, T_lat, 1, 1),
                device=hdmap_z.device,
                dtype=hdmap_z.dtype,
            )
            if frames_inf is not None:
                kept = sorted({int(i) % T_lat for i in frames_inf})
                for i in kept:
                    frame_keep[:, :, i] = 1.0
                log.info(
                    f"[infer] hdmap keep_frames={kept}/{T_lat}: zeroed "
                    f"{T_lat - len(kept)} other hdmap latent frames"
                )
            else:
                K = max(0, min(int(k_inf), T_lat))
                if K > 0:
                    frame_keep[:, :, :K] = 1.0
                log.info(
                    f"[infer] hdmap keep_first_k={K}/{T_lat}: zeroed "
                    f"{T_lat - K} trailing hdmap latent frames"
                )
            hdmap_z = hdmap_z * frame_keep
        if getattr(self, "_hdmap_force_zero_latent", False):
            hdmap_z = torch.zeros_like(hdmap_z)
        self._current_hdmap = hdmap_z

    def get_data_and_condition(
        self, data_batch: dict[str, Tensor]
    ) -> Tuple[Tensor, Tensor, BaseVideoCondition]:
        self._normalize_video_databatch_inplace(data_batch)
        self._augment_image_dim_inplace(data_batch)
        input_key = self.input_data_key
        is_image_batch = self.is_image_batch(data_batch)
        is_video_batch = not is_image_batch

        for key in sorted(data_batch.keys()):
            data_batch[key] = _broadcast(
                data_batch[key], to_tp=True, to_cp=is_video_batch
            )

        if is_image_batch:
            input_key = self.input_image_key

        # Encode lidar
        raw_state = data_batch[input_key]
        self._current_sensor_dataset_ids = None
        if self._num_dataset_embeddings > 0:
            dataset_ids = data_batch.get("sensor_dataset_id")
            if dataset_ids is None:
                raise KeyError(
                    "dataset-conditioned DiT requires sensor_dataset_id"
                )
            dataset_ids = torch.as_tensor(
                dataset_ids, device=raw_state.device, dtype=torch.long
            ).reshape(-1)
            if dataset_ids.shape != (raw_state.shape[0],):
                raise ValueError(
                    "sensor_dataset_id must have one value per sample, got "
                    f"{tuple(dataset_ids.shape)} for B={raw_state.shape[0]}"
                )
            if bool(
                ((dataset_ids < 0) | (
                    dataset_ids >= self._num_dataset_embeddings
                )).any()
            ):
                raise ValueError(
                    "sensor_dataset_id contains a value outside [0, "
                    f"{self._num_dataset_embeddings})"
                )
            self._current_sensor_dataset_ids = dataset_ids
        latent_state = self.encode(raw_state).contiguous()
        latent_state = latent_state.to(self.diffusion_dtype)

        # Prepare HD-map conditioning
        hdmap_z = None
        if self._hdmap_concat_ch > 0:
            if self.bev_hdmap_wan is not None:
                # BEV-native path: render the multi-channel BEV hdmap raster
                # (B, C_cls, T, Y, X) to a 3-ch image and encode it with the
                # frozen Wan2.2 VAE -> (B, 48, T, Y/16, X/16).
                bev = data_batch.get("bev_hdmap")
                if bev is not None:
                    hdmap_z = self._bev_hdmap_to_wan_latent(bev.float()).to(
                        self.diffusion_dtype
                    )
            elif self.bev_hdmap_pool is not None:
                # BEV-native path: pool the multi-channel BEV hdmap raster
                # (B, C_cls, T, Y, X) down to the latent grid (B, out, T, Hz, Wz).
                bev = data_batch.get("bev_hdmap")
                if bev is not None:
                    hdmap_z = self.bev_hdmap_pool(bev.float()).to(self.diffusion_dtype)
            else:
                hdmap = data_batch.get("lidar_hdmap")
                if hdmap is not None:
                    if self.encode_hdmap_with_vae:
                        if getattr(self, "_hdmap_force_zero_latent", False):
                            hdmap_z = latent_state.new_zeros(
                                latent_state.shape[0],
                                self._hdmap_concat_ch,
                                *latent_state.shape[2:],
                            )
                        else:
                            # Hdmap shares the lidar's pixel format (C, T, H, W);
                            # reuse the lidar VAE so the conditioning is already in
                            # the same latent space as the noisy target.
                            # ``self.encode`` already applies the ``sigma_data``
                            # scaling, matching ``latent_state``. For a mono
                            # ``mask_as_input`` VAE the wrapper auto-derives +
                            # appends the (threshold) validity-mask channel from the
                            # 1-channel depth -- the SAME encode path as the lidar.
                            hdmap_z = self.encode(hdmap.float()).contiguous()
                    else:
                        hdmap_z = self._prepare_hdmap(hdmap.float(), latent_state)
                    hdmap_z = hdmap_z.to(self.diffusion_dtype)
        self._store_hdmap_latent(hdmap_z)

        # Condition
        condition = self.conditioner(data_batch)
        if is_image_batch:
            condition.data_type = DataType.IMAGE
        else:
            condition.data_type = DataType.VIDEO

        latent_state = _broadcast(latent_state, to_tp=True, to_cp=is_video_batch)
        condition = broadcast_condition(condition, to_tp=True, to_cp=is_video_batch)

        return raw_state, latent_state, condition

    # ------------------------------------------------------------------
    # Override: denoise — concat hdmap with network input
    # ------------------------------------------------------------------

    def denoise(
        self, xt: torch.Tensor, sigma: torch.Tensor, condition: CosmosCondition
    ) -> DenoisePrediction:
        if getattr(self.config, "use_dummy_temporal_dim", False):
            xt = xt.unsqueeze(2)

        xt = xt.to(**self.tensor_kwargs)
        sigma = sigma.to(**self.tensor_kwargs)

        c_skip, c_out, c_in, c_noise = self.scaling(sigma=sigma)

        net_input = batch_mul(c_in, xt)

        # Channel-concat HD-map
        if self._current_hdmap is not None:
            hdmap = self._current_hdmap.to(net_input.dtype)
            # Handle CFG batch doubling
            if net_input.shape[0] != hdmap.shape[0]:
                repeats = net_input.shape[0] // hdmap.shape[0]
                hdmap = hdmap.repeat(repeats, *([1] * (hdmap.ndim - 1)))
            # Unconditional CFG pass: replace hdmap with zeros (the "null"
            # condition the model saw during training with hdmap dropout).
            if self._hdmap_force_null:
                hdmap = torch.zeros_like(hdmap)
            net_input = torch.cat([net_input, hdmap], dim=1)

        net_output = self.net(
            x=net_input,
            timesteps=c_noise,
            **condition.to_dict(),
        )

        logvar = self.model.logvar(c_noise)
        x0_pred = batch_mul(c_skip, xt) + batch_mul(c_out, net_output)
        eps_pred = batch_mul(xt - x0_pred, 1.0 / sigma)

        if getattr(self.config, "use_dummy_temporal_dim", False):
            x0_pred = x0_pred.squeeze(2)
            eps_pred = eps_pred.squeeze(2)

        return DenoisePrediction(x0_pred, eps_pred, logvar)

    # ------------------------------------------------------------------
    # Override: validation — provide defaults for missing keys
    # ------------------------------------------------------------------

    @torch.no_grad()
    def validation_step(
        self, data: dict[str, Tensor], iteration: int
    ) -> tuple[dict[str, Tensor], Tensor]:
        raw_data, x0, condition = self.get_data_and_condition(data)

        guidance = data.get("guidance", 1.5)
        data = misc.to(data, **self.tensor_kwargs)

        sample = self.generate_samples_from_batch(
            data,
            guidance=guidance,
            state_shape=x0.shape[1:],
            n_sample=x0.shape[0],
        )
        sample = self.decode(sample)
        reconstruction = self.decode(x0)

        hdmap = data.get("lidar_hdmap")
        if hdmap is not None:
            hdmap = hdmap.float().cpu()

        return {
            "gt": raw_data,
            "result": sample,
            "reconstruction": reconstruction,
            "lidar_hdmap": hdmap,
        }, torch.tensor([0]).to(**self.tensor_kwargs)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def on_train_start(
        self, memory_format: torch.memory_format = torch.preserve_format
    ) -> None:
        super().on_train_start(memory_format)
        if self.hdmap_proj is not None:
            self.hdmap_proj = self.hdmap_proj.to(
                device="cuda", dtype=torch.float32
            )
        # The frozen Wan2.2 hdmap encoder lives on ``self`` (not in
        # ``self.model``), so the trainer's ``self.model.to(cuda)`` does not
        # move it. Place it on CUDA explicitly, mirroring ``hdmap_proj`` (kept
        # fp32 -- it was built with is_bf16=False and encodes in fp32).
        if self.bev_hdmap_wan is not None:
            self.bev_hdmap_wan = self.bev_hdmap_wan.to(device="cuda")
            self._bev_palette = self._bev_palette.to(device="cuda")


@diffusion_fsdp_class_decorator
class FSDPHdmapToLidarDiffusionModel(HdmapToLidarModel):
    pass
