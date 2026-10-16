# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F

from utils.distributed import rank0_first
from utils.misc import load_from_s3_with_cache


class BaseVAE(torch.nn.Module, ABC):
    """
    Abstract base class for a Variational Autoencoder (VAE).

    All subclasses should implement the methods to define the behavior for encoding
    and decoding, along with specifying the latent channel size.
    """

    def __init__(self, channel: int = 3, name: str = "vae"):
        super().__init__()
        self.channel = channel
        self.name = name

    @property
    def latent_ch(self) -> int:
        """
        Returns the number of latent channels in the VAE.
        """
        return self.channel

    @abstractmethod
    def encode(self, state: torch.Tensor) -> torch.Tensor:
        """
        Encodes the input tensor into a latent representation.

        Args:
        - state (torch.Tensor): The input tensor to encode.

        Returns:
        - torch.Tensor: The encoded latent tensor.
        """
        pass

    @abstractmethod
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decodes the latent representation back to the original space.

        Args:
        - latent (torch.Tensor): The latent tensor to decode.

        Returns:
        - torch.Tensor: The decoded tensor.
        """
        pass

    @property
    def spatial_compression_factor(self) -> int:
        """
        Returns the spatial reduction factor for the VAE.
        """
        raise NotImplementedError("The spatial_compression_factor property must be implemented in the derived class.")


class BasePretrainedImageVAE(BaseVAE):
    """
    A base class for pretrained Variational Autoencoder (VAE) that loads mean and standard deviation values
    from a remote store, handles data type conversions, and normalization
    using provided mean and standard deviation values for latent space representation.
    Derived classes should load pre-trained encoder and decoder components from a remote store

    Attributes:
        latent_mean (Tensor): The mean used for normalizing the latent representation.
        latent_std (Tensor): The standard deviation used for normalizing the latent representation.
        dtype (dtype): Data type for model tensors, determined by whether bf16 is enabled.

    Args:
        mean_std_fp (str): File path to the pickle file containing mean and std of the latent space.
        latent_ch (int, optional): Number of latent channels (default is 16).
        is_image (bool, optional): Flag to indicate whether the output is an image (default is True).
        is_bf16 (bool, optional): Flag to use Brain Floating Point 16-bit data type (default is True).
    """

    def __init__(
        self,
        name: str,
        mean_std_fp: str,
        latent_ch: int = 16,
        is_image: bool = True,
        is_bf16: bool = True,
    ) -> None:
        super().__init__(latent_ch, name)
        dtype = torch.bfloat16 if is_bf16 else torch.float32
        self.dtype = dtype
        self.is_image = is_image
        self.mean_std_fp = mean_std_fp
        self.name = name

        self.backend_args = None

        self.register_mean_std(mean_std_fp)

    def register_mean_std(self, mean_std_fp: str) -> None:
        latent_mean, latent_std = torch.load(mean_std_fp, map_location="cuda", weights_only=True)
        target_shape = [1, self.latent_ch, 1, 1] if self.is_image else [1, self.latent_ch, 1, 1, 1]
        self.register_buffer(
            "latent_mean",
            latent_mean.to(self.dtype).reshape(*target_shape),
            persistent=False,
        )
        self.register_buffer(
            "latent_std",
            latent_std.to(self.dtype).reshape(*target_shape),
            persistent=False,
        )

    @torch.no_grad()
    def encode(self, state: torch.Tensor) -> torch.Tensor:
        """
        Encode the input state to latent space; also handle the dtype conversion, mean and std scaling
        """
        in_dtype = state.dtype
        latent_mean = self.latent_mean.to(in_dtype)
        latent_std = self.latent_std.to(in_dtype)
        encoded_state = self.encoder(state.to(self.dtype))
        if isinstance(encoded_state, torch.Tensor):
            pass
        elif isinstance(encoded_state, tuple):
            assert isinstance(encoded_state[0], torch.Tensor)
            encoded_state = encoded_state[0]
        else:
            raise ValueError("Invalid type of encoded state")
        return (encoded_state.to(in_dtype) - latent_mean) / latent_std

    @torch.no_grad()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decode the input latent to state; also handle the dtype conversion, mean and std scaling
        """
        in_dtype = latent.dtype
        latent = latent * self.latent_std.to(in_dtype) + self.latent_mean.to(in_dtype)
        return self.decoder(latent.to(self.dtype)).to(in_dtype)

    def reset_dtype(self, *args, **kwargs):
        """
        Resets the data type of the encoder and decoder to the model's default data type.

        Args:
            *args, **kwargs: Unused, present to allow flexibility in method calls.
        """
        del args, kwargs
        self.decoder.to(self.dtype)
        self.encoder.to(self.dtype)


class JITVAE(BasePretrainedImageVAE):
    """
    A JIT compiled Variational Autoencoder (VAE) that loads pre-trained encoder
    and decoder components from a remote store, handles data type conversions, and normalization
    using provided mean and standard deviation values for latent space representation.

    Attributes:
        encoder (Module): The JIT compiled encoder loaded from storage.
        decoder (Module): The JIT compiled decoder loaded from storage.
        latent_mean (Tensor): The mean used for normalizing the latent representation.
        latent_std (Tensor): The standard deviation used for normalizing the latent representation.
        dtype (dtype): Data type for model tensors, determined by whether bf16 is enabled.

    Args:
        enc_fp (str): File path to the encoder's JIT file on the remote store.
        dec_fp (str): File path to the decoder's JIT file on the remote store.
        name (str): Name of the model, used for differentiating cache file paths.
        mean_std_fp (str): File path to the pickle file containing mean and std of the latent space.
        latent_ch (int, optional): Number of latent channels (default is 16).
        is_image (bool, optional): Flag to indicate whether the output is an image (default is True).
        is_bf16 (bool, optional): Flag to use Brain Floating Point 16-bit data type (default is True).
    """

    def __init__(
        self,
        enc_fp: str,
        dec_fp: str,
        name: str,
        mean_std_fp: str,
        latent_ch: int = 16,
        is_image: bool = True,
        is_bf16: bool = True,
    ):
        super().__init__(name, mean_std_fp, latent_ch, is_image, is_bf16)
        self.load_encoder(enc_fp)
        self.load_decoder(dec_fp)

    def load_encoder(self, enc_fp: str) -> None:
        """
        Load the encoder from the remote store.

        Args:
        - enc_fp (str): File path to the encoder's JIT file on the remote store.
        """
        self.encoder = torch.jit.load(enc_fp, map_location="cuda")
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.to(self.dtype)

    def load_decoder(self, dec_fp: str) -> None:
        """
        Load the decoder from the remote store.

        Args:
        - dec_fp (str): File path to the decoder's JIT file on the remote store.
        """
        self.decoder = torch.jit.load(dec_fp, map_location="cuda")
        self.decoder.eval()
        for param in self.decoder.parameters():
            param.requires_grad = False
        self.decoder.to(self.dtype)


class StateDictVAE(BasePretrainedImageVAE):
    """
    A Variational Autoencoder (VAE) that loads pre-trained weights into
    provided encoder and decoder components from a remote store, handles data type conversions,
    and normalization using provided mean and standard deviation values for latent space representation.

    Attributes:
        encoder (Module): The encoder with weights loaded from storage.
        decoder (Module): The decoder with weights loaded from storage.
        latent_mean (Tensor): The mean used for normalizing the latent representation.
        latent_std (Tensor): The standard deviation used for normalizing the latent representation.
        dtype (dtype): Data type for model tensors, determined by whether bf16 is enabled.

    Args:
        enc_fp (str): File path to the encoder's JIT file on the remote store.
        dec_fp (str): File path to the decoder's JIT file on the remote store.
        vae (Module): Instance of VAE with not loaded weights
        name (str): Name of the model, used for differentiating cache file paths.
        mean_std_fp (str): File path to the pickle file containing mean and std of the latent space.
        latent_ch (int, optional): Number of latent channels (default is 16).
        is_image (bool, optional): Flag to indicate whether the output is an image (default is True).
        is_bf16 (bool, optional): Flag to use Brain Floating Point 16-bit data type (default is True).
    """

    def __init__(
        self,
        enc_fp: str,
        dec_fp: str,
        vae: torch.nn.Module,
        name: str,
        mean_std_fp: str,
        latent_ch: int = 16,
        is_image: bool = True,
        is_bf16: bool = True,
    ):
        super().__init__(name, mean_std_fp, latent_ch, is_image, is_bf16)

        self.load_encoder_and_decoder(enc_fp, dec_fp, vae)

    def load_encoder_and_decoder(self, enc_fp: str, dec_fp: str, vae: torch.nn.Module) -> None:
        """
        Load the encoder from the remote store.

        Args:
        - vae_fp (str): File path to the vae's state dict file on the remote store.
        - vae (str): VAE module into which weights will be loaded.
        """
        state_dict_enc = load_from_s3_with_cache(
            enc_fp,
            f"vae/{self.name}_enc.jit",
            easy_io_kwargs={"map_location": torch.device(torch.cuda.current_device())},
            backend_args=self.backend_args,
        )

        state_dict_dec = load_from_s3_with_cache(
            dec_fp,
            f"vae/{self.name}_dec.jit",
            easy_io_kwargs={"map_location": torch.device(torch.cuda.current_device())},
            backend_args=self.backend_args,
        )

        jit_weights_state_dict = state_dict_enc.state_dict() | state_dict_dec.state_dict()
        jit_weights_state_dict = {
            k: v
            for k, v in jit_weights_state_dict.items()
            # Global variables captured by JIT
            if k
            not in (
                "encoder.patcher.wavelets",
                "encoder.patcher._arange",
                "decoder.unpatcher.wavelets",
                "decoder.unpatcher._arange",
            )
        }

        vae.load_state_dict(jit_weights_state_dict)
        vae.eval()
        for param in vae.parameters():
            param.requires_grad = False
        vae.to(self.dtype)

        self.vae = vae
        self.encoder = self.vae.encode
        self.decoder = self.vae.decode

    def reset_dtype(self, *args, **kwargs):
        """
        Resets the data type of the encoder and decoder to the model's default data type.

        Args:
            *args, **kwargs: Unused, present to allow flexibility in method calls.
        """
        del args, kwargs
        self.vae.to(self.dtype)


class ConfigBasedVideoVAE(BaseVAE):
    """VAE loaded from a tokenizer experiment config + training checkpoint.

    Works with any tokenizer that exposes `encode(x) -> (sample, (mu, logvar))`
    and `decode(z) -> x`, including WanVAE2P1 and TransformerVAE.

    Latent statistics can be supplied either via ``mean_std_fp`` (a saved
    ``(mean, std)`` tuple / ``{"mean", "std"}`` dict) or via inline
    ``latent_mean`` / ``latent_std`` lists. Inline values take precedence
    so callers that don't have a ``latent_stats.pt`` next to their
    checkpoint (e.g. the public ``Wan2.1_VAE.pth`` release) can pass the
    per-channel statistics directly from a constant.
    """

    def __init__(
        self,
        experiment_name: str,
        checkpoint_path: str,
        mean_std_fp: str | None = None,
        latent_ch: int = 64,
        is_bf16: bool = False,
        spatial_compression: int = 8,
        latent_mean: list[float] | tuple[float, ...] | None = None,
        latent_std: list[float] | tuple[float, ...] | None = None,
        apply_validity_mask: bool = False,
        intensity_pad_value: float = 0.0,
        network_overrides: dict | None = None,
        use_ema: bool = False,
        per_frame: bool = False,
        encoder_is_bf16: bool | None = None,
        decoder_is_bf16: bool | None = None,
        strict_checkpoint: bool = False,
    ):
        super().__init__(channel=latent_ch, name=f"config_vae_{experiment_name}")
        from diffusion.training.module.vae_utils import (
            build_vae_from_tokenizer_experiment,
        )

        self.is_image = False
        self._spatial_compression = spatial_compression
        # per_frame: the wrapped net is a 2D (single-frame) tokenizer (cosmos CI,
        # LiDM) whose encode/decode take (B, C, H, W). encode/decode fold the clip's
        # T axis into the batch, run the net per frame, and reshape the latent back
        # to (B, Cz, T, Hz, Wz) so the rest of the video pipeline (5D latent-stat
        # buffers, DiT) is unchanged. The first-frame prepend still happens in the
        # dataset (is_video_tokenizer=True) -> T frames each encoded independently.
        self.per_frame = bool(per_frame)
        encoder_is_bf16 = (
            is_bf16 if encoder_is_bf16 is None else encoder_is_bf16
        )
        decoder_is_bf16 = (
            is_bf16 if decoder_is_bf16 is None else decoder_is_bf16
        )
        self.encoder_dtype = (
            torch.bfloat16 if encoder_is_bf16 else torch.float32
        )
        self.decoder_dtype = (
            torch.bfloat16 if decoder_is_bf16 else torch.float32
        )
        # Keep the legacy public dtype tied to decode. Encode paths use the
        # explicit encoder dtype below.
        self.dtype = self.decoder_dtype

        # ``network_overrides`` are applied AFTER the experiment's own network
        # overrides (e.g. ``utonia_align=False`` to skip building the training-only
        # VF/PTv3 teacher -- encode/decode is bit-identical and the leftover
        # ``utonia.*`` ckpt keys drop harmlessly under the strict=False load below).
        self.vae_net = build_vae_from_tokenizer_experiment(
            experiment_name, **(network_overrides or {})
        )

        # Validity-head support: predict_validity VAEs return a per-pixel raydrop
        # logit from decode(); when ``apply_validity_mask`` is set we fill the
        # predicted-invalid pixels with the depth sentinel so the decoded range
        # map carries raydrop directly (matches the trained 2-channel output).
        self.predict_validity = bool(getattr(self.vae_net, "predict_validity", False))
        # decoder_validity_head: a SEPARATE additive head taps the decoder recon
        # and emits a raydrop logit via decode(z, return_validity=True) (does NOT
        # widen the main conv). Mutually exclusive with predict_validity. Handled
        # on the same masking path below.
        self.decoder_validity_head = bool(getattr(self.vae_net, "decoder_validity_head", False))
        self.apply_validity_mask = bool(apply_validity_mask)
        self.invalid_fill = float(getattr(self.vae_net, "mask_input_min_value", -1.0))
        # Fill value for DATA channels missing from the input (e.g. the intensity
        # channel when encoding a mono-depth hdmap through a [depth, intensity]
        # VAE). Intensity is normalized to [-1, 1]; a flat plane is OOD, so this
        # is exposed for A/B tuning. Default 0.0 preserves prior behavior.
        self.intensity_pad_value = float(intensity_pad_value)

        # ``is_ltx`` flags whether the wrapped VAE supports LTX-style
        # timestep-conditioned decoding (``decode(z, t=...)`` with a
        # rectified-flow t threading through AdaLN). Detected from the
        # underlying net's ``denoising_t_max`` attr -- the LTX
        # TransformerVAELTX sets it to > 0 when AdaLN is built.
        # Callers should gate any ``t`` they want to pass to
        # ``decode`` on this flag rather than guessing from the model
        # class or ``is_image``.
        self.is_ltx = bool(getattr(self.vae_net, "denoising_t_max", 0.0) > 0)

        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = ckpt.get("model", ckpt)
        vae_state = {}
        for k, v in state.items():
            for prefix in ("network.", "vae.", "model.network.", "model.vae."):
                if k.startswith(prefix):
                    k = k[len(prefix):]
                    break
            vae_state[k] = v
        from tokenizer.networks.transformers.transformer_vae_cross_dataset import (
            CrossDatasetTransformerVAE,
        )

        if isinstance(self.vae_net, CrossDatasetTransformerVAE):
            # Cross-dataset encode/decode regenerate this nonlearned grid at
            # the input resolution; its saved shape is not a model constraint.
            saved_coords = vae_state.pop("coords", None)
            if saved_coords is not None:
                print(
                    "[ConfigBasedVideoVAE] ignoring derived cross-dataset "
                    f"coords buffer {tuple(saved_coords.shape)}; "
                    "coordinates are regenerated for each input grid"
                )
        incompatible = self.vae_net.load_state_dict(vae_state, strict=False)
        if strict_checkpoint:
            allowed_missing = (
                {"coords"}
                if isinstance(self.vae_net, CrossDatasetTransformerVAE)
                else set()
            )
            missing = sorted(set(incompatible.missing_keys) - allowed_missing)
            unexpected = incompatible.unexpected_keys
            if missing or unexpected:
                raise RuntimeError(
                    "Tokenizer checkpoint/config mismatch: "
                    f"missing={missing[:12]}, unexpected={unexpected[:12]}. "
                    "Refusing inference with partially loaded weights."
                )
            print("[ConfigBasedVideoVAE] checkpoint/config match: all weights loaded")

        # Optional: overlay the EMA shadow weights bundled in the same tokenizer
        # ckpt (``ema.network-<name>`` where <name> is the param name with "."
        # replaced by "-"; see utils/ema.py). Only trainable params are EMA'd, so
        # buffers stay at their reg values -- hence overlay onto the reg-loaded net
        # with strict=False rather than a from-scratch load. NOTE: the diffusion
        # model was trained on reg-encoded latents, so this only swaps the DECODER
        # weights at inference -- an EMA-vs-reg decode A/B, not a latent-space change.
        if use_ema:
            ema_state = {
                k[len("ema.network-"):].replace("-", "."): v
                for k, v in state.items()
                if k.startswith("ema.network-")
            }
            if not ema_state:
                raise RuntimeError(
                    f"use_ema=True but no 'ema.network-*' keys in {checkpoint_path} "
                    "(EMA was not enabled for this tokenizer run)."
                )
            _, unexpected = self.vae_net.load_state_dict(ema_state, strict=False)
            if unexpected:
                raise RuntimeError(
                    f"EMA overlay had {len(unexpected)} unexpected keys, e.g. "
                    f"{unexpected[:5]} -- ema->net name remap is wrong for this arch."
                )
            print(f"[ConfigBasedVideoVAE] use_ema=True: overlaid {len(ema_state)} "
                  "EMA params onto the decoder/vae_net")

        self.vae_net.eval()
        for p in self.vae_net.parameters():
            p.requires_grad = False
        self._reset_module_dtypes()

        if latent_mean is not None and latent_std is not None:
            assert len(latent_mean) == latent_ch and len(latent_std) == latent_ch, (
                f"Inline latent_mean/latent_std must have length latent_ch={latent_ch}, "
                f"got {len(latent_mean)}/{len(latent_std)}"
            )
            latent_mean_t = torch.tensor(list(latent_mean), dtype=torch.float32)
            latent_std_t = torch.tensor(list(latent_std), dtype=torch.float32)
        else:
            assert mean_std_fp is not None, (
                "ConfigBasedVideoVAE requires either mean_std_fp or "
                "(latent_mean, latent_std)."
            )
            stats = torch.load(mean_std_fp, map_location="cpu", weights_only=False)
            if isinstance(stats, (tuple, list)):
                latent_mean_t, latent_std_t = stats
            else:
                latent_mean_t = stats["mean"]
                latent_std_t = stats["std"]
        shape = [1, latent_ch, 1, 1, 1]
        self.register_buffer(
            "latent_mean",
            latent_mean_t.to(self.dtype).reshape(*shape),
            persistent=False,
        )
        self.register_buffer(
            "latent_std",
            latent_std_t.to(self.dtype).reshape(*shape),
            persistent=False,
        )

    def _reset_module_dtypes(self) -> None:
        self.vae_net.to(self.decoder_dtype)
        if self.encoder_dtype == self.decoder_dtype:
            return
        if not all(
            hasattr(self.vae_net, name) for name in ("encoder", "quant_conv")
        ):
            raise ValueError(
                "mixed VAE precision requires separate encoder and "
                "quant_conv modules"
            )
        self.vae_net.encoder.to(self.encoder_dtype)
        self.vae_net.quant_conv.to(self.encoder_dtype)

    def _prepare_encoder_input(self, state: torch.Tensor) -> torch.Tensor:
        x = state.to(self.encoder_dtype)
        # mask_as_input VAEs expect a [data..., mask] encoder input, but
        # vae_net.encode() does NOT build the mask (forward() does). The net
        # reconstructs ``out_channels`` DATA channels (1 = [depth] mono, 2 =
        # [depth, intensity]); build the full encoder input to match training:
        #   * pad missing data channels with zeros (e.g. hdmap is mono depth even
        #     for an intensity VAE -> [depth, 0(intensity)] before the mask),
        #   * derive the validity mask from depth (ch0, same threshold as the
        #     loss/forward) and concat it.
        # No-op for non-mask VAEs and for inputs that already carry the mask
        # channel (x.shape[1] == data_ch + 1). Bit-identical to the old 1ch path
        # for mono mask VAEs (out_channels == 1).
        if getattr(self.vae_net, "mask_as_input", False):
            data_ch = int(getattr(self.vae_net, "out_channels", 1))
            if x.shape[1] <= data_ch:
                if x.shape[1] < data_ch:  # pad missing data channels (intensity)
                    pad = x.new_full(
                        (x.shape[0], data_ch - x.shape[1]) + tuple(x.shape[2:]),
                        self.intensity_pad_value)
                    x = torch.cat([x, pad], dim=1)
                thr = self.vae_net.mask_input_min_value + self.vae_net.mask_input_eps
                valid = (x[:, :1] > thr).to(x.dtype)
                x = torch.cat([x, valid], dim=1)
        return x

    @torch.no_grad()
    def encode(
        self,
        state: torch.Tensor,
        sensor_dataset_id: torch.Tensor | int | None = None,
    ) -> torch.Tensor:
        if sensor_dataset_id is not None:
            setter = getattr(self.vae_net, "set_dataset_id", None)
            if setter is not None:
                setter(sensor_dataset_id)
            elif getattr(self.vae_net, "uses_dataset_id", False):
                raise TypeError(
                    f"{type(self.vae_net).__name__} requires a dataset ID but "
                    "does not expose set_dataset_id()"
                )
        x = self._prepare_encoder_input(state)
        if self.per_frame:
            # 2D net: (B, C, T, H, W) -> encode each frame as (B*T, C, H, W).
            B, C, T, H, W = x.shape
            frames = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
            enc = self.vae_net.encode(frames)
            z = enc[0] if isinstance(enc, tuple) else enc
            Cz, Hz, Wz = z.shape[1], z.shape[2], z.shape[3]
            latent = z.reshape(B, T, Cz, Hz, Wz).permute(0, 2, 1, 3, 4)
        else:
            encoded = self.vae_net.encode(x)
            latent = encoded[0] if isinstance(encoded, tuple) else encoded
        latent = latent.to(self.encoder_dtype)
        latent_mean = self.latent_mean.to(self.encoder_dtype)
        latent_std = self.latent_std.to(self.encoder_dtype)
        return (latent - latent_mean) / latent_std

    @torch.no_grad()
    def encode_stream(
        self,
        state_new: torch.Tensor,
        temporal_kv_cache=None,
        max_cache_frames: int | None = None,
    ):
        """Encode newly appended frames with the wrapped causal VAE cache."""
        if self.per_frame or not getattr(
            self.vae_net, "supports_encoder_temporal_kv_cache", False
        ):
            raise NotImplementedError(
                f"{type(self.vae_net).__name__} does not support temporal "
                "KV-cache encoding"
            )
        x_new = self._prepare_encoder_input(state_new)
        encoded, new_cache = self.vae_net.encode_stream(
            x_new,
            temporal_kv_cache=temporal_kv_cache,
            max_cache_frames=max_cache_frames,
        )
        latent = encoded[0] if isinstance(encoded, tuple) else encoded
        latent = latent.to(self.encoder_dtype)
        latent_mean = self.latent_mean.to(self.encoder_dtype)
        latent_std = self.latent_std.to(self.encoder_dtype)
        latent = (latent - latent_mean) / latent_std
        return latent, new_cache

    @torch.no_grad()
    def decode(
        self,
        latent: torch.Tensor,
        t: float | torch.Tensor | None = None,
        return_validity: bool = False,
        sensor_dataset_id: torch.Tensor | int | None = None,
    ) -> torch.Tensor:
        """Decode a normalized latent back to pixels.

        Args:
            latent: ``(B, latent_ch, T, H, W)`` in normalized-latent space
                (i.e. already divided by ``self.sigma_data`` upstream).
            t: optional rectified-flow timestep in ``[0, 1]``. Only
                accepted when ``self.is_ltx`` is True (the wrapped VAE
                net supports ``decode(z, t=t)``). For non-LTX wrappers
                the standard ``decode(z)`` signature is used and
                passing ``t`` raises -- prevents silently dropping the
                timestep on an unrelated VAE.
            return_validity: when True (and the wrapped net has a validity
                head), also return the per-pixel validity probability
                ``(B, 1, T, H, W)`` in ``[0, 1]``; ``None`` otherwise.

        When the wrapped net has ``predict_validity`` and this wrapper was
        built with ``apply_validity_mask=True``, predicted-invalid pixels
        (prob < 0.5) are filled with the depth sentinel so the returned range
        map carries raydrop directly.
        """
        if sensor_dataset_id is not None:
            setter = getattr(self.vae_net, "set_dataset_id", None)
            if setter is not None:
                setter(sensor_dataset_id)
            elif getattr(self.vae_net, "uses_dataset_id", False):
                raise TypeError(
                    f"{type(self.vae_net).__name__} requires a dataset ID but "
                    "does not expose set_dataset_id()"
                )
        if t is not None and not self.is_ltx:
            raise ValueError(
                f"Got t={t!r} but the wrapped VAE net "
                f"({type(self.vae_net).__name__}) is not LTX-style "
                f"(is_ltx=False). Use a TransformerVAELTX checkpoint "
                f"with denoising_t_max > 0 to enable timestep-"
                f"conditioned decoding."
            )
        latent = latent.to(self.decoder_dtype)
        latent_std = self.latent_std.to(self.decoder_dtype)
        latent_mean = self.latent_mean.to(self.decoder_dtype)
        latent = latent * latent_std + latent_mean

        if self.per_frame:
            # 2D net: (B, Cz, T, Hz, Wz) -> decode each frame as (B*T, Cz, Hz, Wz).
            # These nets have no validity head, so return_validity yields None.
            B, Cz, T, Hz, Wz = latent.shape
            z = latent.permute(0, 2, 1, 3, 4).reshape(B * T, Cz, Hz, Wz)
            rec = self.vae_net.decode(z)
            if isinstance(rec, tuple):
                rec = rec[0]
            if rec.ndim == 5:  # some nets return (B, C, 1, H, W)
                rec = rec[:, :, 0]
            C, H, W = rec.shape[1], rec.shape[2], rec.shape[3]
            out = rec.reshape(B, T, C, H, W).permute(0, 2, 1, 3, 4)
            if return_validity:
                return out, None
            return out

        validity = None
        if self.is_ltx:
            out = self.vae_net.decode(latent, t=t)
        elif self.predict_validity or self.decoder_validity_head:
            out, logits = self.vae_net.decode(latent, return_validity=True)
            if logits is not None:
                validity = torch.sigmoid(logits.to(out.dtype))
                if self.apply_validity_mask:
                    out = out.masked_fill(validity < 0.5, self.invalid_fill)
        else:
            out = self.vae_net.decode(latent)

        if return_validity:
            return out, validity
        return out

    @torch.no_grad()
    def decode_stream(
        self,
        latent_new: torch.Tensor,
        temporal_kv_cache=None,
        return_validity: bool = False,
        max_cache_frames: int | None = None,
    ):
        """Decode newly appended frames with the wrapped causal VAE cache."""
        if self.per_frame or self.is_ltx or self.decoder_validity_head:
            raise NotImplementedError(
                "streaming decode currently supports the attention-based "
                "TransformerVAE video decoder with its integrated validity head"
            )
        if not getattr(
            self.vae_net, "supports_temporal_kv_cache", False
        ):
            raise NotImplementedError(
                f"{type(self.vae_net).__name__} does not support temporal "
                "KV-cache decoding"
            )

        latent_new = latent_new.to(self.decoder_dtype)
        latent_std = self.latent_std.to(self.decoder_dtype)
        latent_mean = self.latent_mean.to(self.decoder_dtype)
        latent_new = latent_new * latent_std + latent_mean
        decode_kwargs = {
            "temporal_kv_cache": temporal_kv_cache,
            "return_validity": self.predict_validity,
        }
        if max_cache_frames is not None:
            decode_kwargs["max_cache_frames"] = max_cache_frames
        decoded, new_cache = self.vae_net.decode_stream(
            latent_new, **decode_kwargs
        )
        validity = None
        if self.predict_validity:
            out, logits = decoded
            validity = torch.sigmoid(logits.to(out.dtype))
            if self.apply_validity_mask:
                out = out.masked_fill(validity < 0.5, self.invalid_fill)
        else:
            out = decoded

        if return_validity:
            return out, validity, new_cache
        return out, new_cache

    def reset_dtype(self, *args, **kwargs):
        del args, kwargs
        self._reset_module_dtypes()

    @property
    def spatial_compression_factor(self) -> int:
        return self._spatial_compression


class SDVAE(BaseVAE):
    def __init__(self, batch_size=16, count_std: bool = False, is_downsample: bool = True) -> None:
        super().__init__(channel=4, name="sd_vae")
        self.dtype = torch.bfloat16
        self.register_buffer(
            "scale",
            torch.tensor([4.17, 4.62, 3.71, 3.28], dtype=self.dtype).reciprocal().reshape(1, -1, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "bias",
            -1.0 * torch.tensor([5.81, 3.25, 0.12, -2.15], dtype=self.dtype).reshape(1, -1, 1, 1) * self.scale,
            persistent=False,
        )
        self.batch_size = batch_size
        self.count_std = count_std
        self.is_downsample = is_downsample
        self.load_vae()
        self.reset_dtype()

    def reset_dtype(self, *args, **kwargs):
        del args, kwargs
        self.vae.to(self.dtype)

    @rank0_first
    def load_vae(self) -> None:
        os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        import diffusers

        vae_name = "stabilityai/sd-vae-ft-mse"
        try:
            vae = diffusers.models.AutoencoderKL.from_pretrained(vae_name, local_files_only=True)
        except:  # noqa: E722
            # Could not load the model from cache; try without local_files_only.
            vae = diffusers.models.AutoencoderKL.from_pretrained(vae_name)
        self.vae = vae.eval().requires_grad_(False)

    @torch.no_grad()
    def encode(self, state: torch.Tensor) -> torch.Tensor:
        """
        state : pixel range [-1, 1]
        """
        if self.is_downsample:
            _h, _w = state.shape[-2:]
            state = F.interpolate(state, size=(_h // 2, _w // 2), mode="bilinear", align_corners=False)
        in_dtype = state.dtype
        state = state.to(self.dtype)
        state = (state + 1.0) / 2.0
        latent_dist = self.vae.encode(state)["latent_dist"]
        mean, std = latent_dist.mean, latent_dist.std
        if self.count_std:
            latent = mean + torch.randn_like(mean) * std
        else:
            latent = mean
        latent = latent * self.scale
        latent = latent + self.bias
        return latent.to(in_dtype)

    @torch.no_grad()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        in_dtype = latent.dtype
        latent = latent.to(self.dtype)
        latent = latent - self.bias
        latent = latent / self.scale
        latent = torch.cat([self.vae.decode(batch)["sample"] for batch in latent.split(self.batch_size)])
        if self.is_downsample:
            _h, _w = latent.shape[-2:]
            latent = F.interpolate(latent, size=(_h * 2, _w * 2), mode="bilinear", align_corners=False)
        return latent.to(in_dtype) * 2 - 1.0

    @property
    def spatial_compression_factor(self) -> int:
        return 8
