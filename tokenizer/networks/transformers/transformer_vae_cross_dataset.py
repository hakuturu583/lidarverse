# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dynamic-shape TransformerVAE with sensor-agnostic image-grid coordinates."""

import numpy as np
import torch
from torch import nn

from tokenizer.networks.transformers.transformer_vae import (
    LearnablePositionalEmbedding,
    TransformerVAE,
)
from tokenizer.networks.transformers.encoding import generate_polar_coords


class CrossDatasetTransformerVAE(TransformerVAE):
    """Cross-dataset VAE whose network sees only normalized image-grid RoPE."""

    uses_sensor_metadata = False

    @staticmethod
    def _coords_from_grid(
        height: int, width: int, device: torch.device
    ) -> torch.Tensor:
        return generate_polar_coords(height, width, device=device)

    def _network_coords(
        self, height: int, width: int, device: torch.device
    ) -> torch.Tensor:
        return self._coords_from_grid(height, width, device)

    def encode(
        self, x: torch.Tensor, spatial_coords: torch.Tensor | None = None
    ):
        if spatial_coords is None:
            spatial_coords = self._network_coords(
                x.shape[-2], x.shape[-1], x.device
            )
        return super().encode(x, spatial_coords=spatial_coords)

    def encode_stream(
        self,
        x_new: torch.Tensor,
        temporal_kv_cache: dict[
            str, tuple[torch.Tensor, torch.Tensor]
        ] | None = None,
        max_cache_frames: int | None = None,
        spatial_coords: torch.Tensor | None = None,
    ):
        if spatial_coords is None:
            spatial_coords = self._network_coords(
                x_new.shape[-2], x_new.shape[-1], x_new.device
            )
        return super().encode_stream(
            x_new,
            temporal_kv_cache=temporal_kv_cache,
            max_cache_frames=max_cache_frames,
            spatial_coords=spatial_coords,
        )

    def decode(
        self,
        z: torch.Tensor,
        hdmap_z: torch.Tensor | None = None,
        hdmap_raw: torch.Tensor | None = None,
        return_validity: bool = False,
        spatial_coords: torch.Tensor | None = None,
    ):
        if spatial_coords is None:
            height, width = getattr(
                self, "_input_spatial_shape", tuple(self.resolution)
            )
            spatial_coords = self._network_coords(
                height, width, z.device
            )
        return super().decode(
            z,
            hdmap_z=hdmap_z,
            hdmap_raw=hdmap_raw,
            return_validity=return_validity,
            spatial_coords=spatial_coords,
        )

    def decode_stream(
        self,
        z_new: torch.Tensor,
        temporal_kv_cache: dict[
            str, tuple[torch.Tensor, torch.Tensor]
        ] | None = None,
        hdmap_z: torch.Tensor | None = None,
        hdmap_raw: torch.Tensor | None = None,
        return_validity: bool = False,
        max_cache_frames: int | None = None,
        spatial_coords: torch.Tensor | None = None,
    ):
        if spatial_coords is None:
            height, width = getattr(
                self, "_input_spatial_shape", tuple(self.resolution)
            )
            spatial_coords = self._network_coords(
                height, width, z_new.device
            )
        return super().decode_stream(
            z_new,
            temporal_kv_cache=temporal_kv_cache,
            hdmap_z=hdmap_z,
            hdmap_raw=hdmap_raw,
            return_validity=return_validity,
            max_cache_frames=max_cache_frames,
            spatial_coords=spatial_coords,
        )

    def forward(
        self,
        x: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        **kwargs,
    ):
        coords = self._network_coords(
            x.shape[-2], x.shape[-1], x.device
        )
        return super().forward(
            x,
            valid_mask=valid_mask,
            spatial_coords=coords,
            **kwargs,
        )


class DatasetEmbeddingCrossDatasetTransformerVAE(
    CrossDatasetTransformerVAE
):
    """Image-grid RoPE VAE with an additive input map per dataset."""

    uses_dataset_id = True

    def __init__(
        self,
        *args,
        dataset_embedding_shapes: tuple[tuple[int, int], ...],
        dataset_embedding_channels: int,
        spatial_rope: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.dataset_embedding_shapes = tuple(
            tuple(shape) for shape in dataset_embedding_shapes
        )
        self.dataset_embedding_channels = int(dataset_embedding_channels)
        self.spatial_rope = bool(spatial_rope)
        self.dataset_embeddings = nn.ParameterList(
            [
                nn.Parameter(
                    torch.zeros(
                        1,
                        self.dataset_embedding_channels,
                        height,
                        width,
                    )
                )
                for height, width in self.dataset_embedding_shapes
            ]
        )

    def _network_coords(
        self, height: int, width: int, device: torch.device
    ) -> torch.Tensor:
        if self.spatial_rope:
            return super()._network_coords(height, width, device)
        # Every AxialRoPE angle is zero, making all Q/K rotations identity.
        return torch.zeros(1, 2, height, width, device=device)

    @staticmethod
    def _homogeneous_dataset_id(
        sensor_dataset_id: torch.Tensor | int,
    ) -> int:
        if isinstance(sensor_dataset_id, int):
            return sensor_dataset_id
        unique_ids = torch.unique(sensor_dataset_id)
        if unique_ids.numel() != 1:
            raise ValueError(
                "Dataset-specific embeddings require a homogeneous batch; "
                f"got dataset IDs {unique_ids.tolist()}"
            )
        return int(unique_ids.item())

    def _add_dataset_embedding(
        self,
        x: torch.Tensor,
        sensor_dataset_id: torch.Tensor | int,
    ) -> torch.Tensor:
        dataset_id = self._homogeneous_dataset_id(sensor_dataset_id)
        if not 0 <= dataset_id < len(self.dataset_embeddings):
            raise ValueError(
                f"Dataset ID {dataset_id} is outside "
                f"[0, {len(self.dataset_embeddings)})"
            )
        embedding = self.dataset_embeddings[dataset_id]
        if x.shape[1] != self.dataset_embedding_channels:
            raise ValueError(
                f"Dataset embedding has {self.dataset_embedding_channels} "
                f"channels but input has {x.shape[1]}"
            )
        if tuple(x.shape[-2:]) != tuple(embedding.shape[-2:]):
            raise ValueError(
                f"Dataset {dataset_id} expects input grid "
                f"{tuple(embedding.shape[-2:])}, got {tuple(x.shape[-2:])}"
            )
        if x.ndim == 5:
            embedding = embedding.unsqueeze(2)
        elif x.ndim != 4:
            raise ValueError(f"Expected 4D or 5D input, got shape {x.shape}")
        return x + embedding.to(dtype=x.dtype)

    def forward(
        self,
        x: torch.Tensor,
        sensor_dataset_id: torch.Tensor | int,
        valid_mask: torch.Tensor | None = None,
        **kwargs,
    ):
        x = self._add_dataset_embedding(x, sensor_dataset_id)
        return super().forward(
            x,
            valid_mask=valid_mask,
            **kwargs,
        )


class DatasetSpecificLearnablePositionalEmbedding(nn.Module):
    """Per-dataset PE that preserves one checkpoint-compatible embedding key."""

    def __init__(
        self,
        pretrained_embedding: LearnablePositionalEmbedding,
        resolutions: tuple[tuple[int, int], ...],
        pretrained_dataset_id: int,
    ):
        super().__init__()
        if not 0 <= pretrained_dataset_id < len(resolutions):
            raise ValueError(
                f"pretrained_dataset_id={pretrained_dataset_id} is outside "
                f"[0, {len(resolutions)})"
            )
        pretrained_resolution = tuple(
            pretrained_embedding.embedding.shape[1:3]
        )
        if pretrained_resolution != tuple(resolutions[pretrained_dataset_id]):
            raise ValueError(
                f"Pretrained PE grid {pretrained_resolution} does not match "
                f"dataset {pretrained_dataset_id} grid "
                f"{tuple(resolutions[pretrained_dataset_id])}"
            )

        # Keep this exact name so an ordinary TransformerVAE checkpoint loads
        # encoder/decoder `spatial_pe.embedding` directly into the source
        # dataset's positional map.
        self.embedding = pretrained_embedding.embedding
        self.pretrained_dataset_id = int(pretrained_dataset_id)
        self.dataset_ids = tuple(
            dataset_id
            for dataset_id in range(len(resolutions))
            if dataset_id != self.pretrained_dataset_id
        )
        out_dim = int(self.embedding.shape[-1])
        self.other_embeddings = nn.ParameterList(
            [
                nn.Parameter(torch.empty(1, height, width, out_dim))
                for dataset_id, (height, width) in enumerate(resolutions)
                if dataset_id != self.pretrained_dataset_id
            ]
        )
        for embedding in self.other_embeddings:
            nn.init.trunc_normal_(embedding, std=0.02)
        self.active_dataset_id = self.pretrained_dataset_id

    def set_dataset_id(self, dataset_id: int) -> None:
        self.active_dataset_id = int(dataset_id)

    def forward(self, coords=None):
        if self.active_dataset_id == self.pretrained_dataset_id:
            selected = self.embedding
        else:
            try:
                index = self.dataset_ids.index(self.active_dataset_id)
            except ValueError as error:
                raise ValueError(
                    f"Dataset ID {self.active_dataset_id} is not configured"
                ) from error
            selected = self.other_embeddings[index]

        # Homogeneous batches rotate datasets between iterations. Keep every
        # PE parameter in every autograd graph so DDP static_graph sees a fixed
        # used-parameter set; inactive maps receive exact-zero gradients.
        dependency = self.embedding.sum() * 0.0
        for embedding in self.other_embeddings:
            dependency = dependency + embedding.sum() * 0.0
        return selected + dependency


class DatasetPositionalEmbeddingCrossDatasetTransformerVAE(
    CrossDatasetTransformerVAE
):
    """VAE with per-dataset learned PE and optional image-grid RoPE."""

    uses_dataset_id = True

    def __init__(
        self,
        *args,
        dataset_input_shapes: tuple[tuple[int, int], ...],
        pretrained_dataset_id: int,
        spatial_rope: bool = False,
        validity_prior_paths: tuple[str, ...] | None = None,
        validity_residual_gate_init: float = 1.0,
        dataset_validity_logit_bias: bool = False,
        **kwargs,
    ):
        if kwargs.get("positional_embedding", "learnable_embedding") != (
            "learnable_embedding"
        ):
            raise ValueError(
                "Dataset positional embeddings require "
                "positional_embedding='learnable_embedding'"
            )
        super().__init__(*args, **kwargs)
        self.spatial_rope = bool(spatial_rope)
        self.pretrained_dataset_id = int(pretrained_dataset_id)
        self.active_dataset_id = self.pretrained_dataset_id

        patch_h, patch_w = self.patch_size
        encoder_resolutions = tuple(
            (height // patch_h, width // patch_w)
            for height, width in dataset_input_shapes
        )
        compression_h, compression_w = self.spatial_compression
        decoder_resolutions = tuple(
            (height // compression_h, width // compression_w)
            for height, width in dataset_input_shapes
        )
        self.encoder.spatial_pe = DatasetSpecificLearnablePositionalEmbedding(
            self.encoder.spatial_pe,
            encoder_resolutions,
            self.pretrained_dataset_id,
        )
        self.decoder.spatial_pe = DatasetSpecificLearnablePositionalEmbedding(
            self.decoder.spatial_pe,
            decoder_resolutions,
            self.pretrained_dataset_id,
        )

        if validity_prior_paths is not None and dataset_validity_logit_bias:
            raise ValueError(
                "Fixed validity priors and learned dataset validity-logit "
                "biases are mutually exclusive"
            )
        self.validity_logit_biases = nn.ParameterList()
        if dataset_validity_logit_bias:
            if not self.predict_validity:
                raise ValueError(
                    "Dataset validity-logit biases require predict_validity=True"
                )
            self.validity_logit_biases.extend(
                nn.Parameter(torch.zeros(1, 1, height, width))
                for height, width in dataset_input_shapes
            )

        self.validity_prior_names = ()
        if validity_prior_paths is not None:
            if not self.predict_validity:
                raise ValueError(
                    "Validity priors require predict_validity=True"
                )
            if len(validity_prior_paths) != len(dataset_input_shapes):
                raise ValueError(
                    "validity_prior_paths must have one path per dataset; "
                    f"got {len(validity_prior_paths)} paths for "
                    f"{len(dataset_input_shapes)} datasets"
                )
            prior_names = []
            for dataset_id, (path, expected_shape) in enumerate(
                zip(validity_prior_paths, dataset_input_shapes)
            ):
                with np.load(path, allow_pickle=False) as data:
                    if "validity_logit" not in data:
                        raise KeyError(
                            f"{path} does not contain 'validity_logit'"
                        )
                    prior = np.asarray(
                        data["validity_logit"], dtype=np.float32
                    )
                if prior.shape != tuple(expected_shape):
                    raise ValueError(
                        f"Dataset {dataset_id} validity prior has shape "
                        f"{prior.shape}, expected {tuple(expected_shape)}"
                    )
                if not np.isfinite(prior).all():
                    raise ValueError(
                        f"Dataset {dataset_id} validity prior contains "
                        "non-finite values"
                    )
                name = f"validity_prior_{dataset_id}"
                self.register_buffer(
                    name,
                    torch.from_numpy(prior)[None, None],
                    persistent=True,
                )
                prior_names.append(name)
            self.validity_prior_names = tuple(prior_names)
            self.validity_residual_gate = nn.Parameter(
                torch.tensor(float(validity_residual_gate_init))
            )

    def set_dataset_id(self, sensor_dataset_id: torch.Tensor | int) -> None:
        """Select the dataset-specific parameters for direct encode/decode."""
        dataset_id = (
            DatasetEmbeddingCrossDatasetTransformerVAE._homogeneous_dataset_id(
                sensor_dataset_id
            )
        )
        configured_ids = {
            self.encoder.spatial_pe.pretrained_dataset_id,
            *self.encoder.spatial_pe.dataset_ids,
        }
        if dataset_id not in configured_ids:
            raise ValueError(f"Invalid sensor_dataset_id={dataset_id}")
        self.active_dataset_id = dataset_id
        self.encoder.spatial_pe.set_dataset_id(dataset_id)
        self.decoder.spatial_pe.set_dataset_id(dataset_id)

    def _network_coords(
        self, height: int, width: int, device: torch.device
    ) -> torch.Tensor:
        if self.spatial_rope:
            return super()._network_coords(height, width, device)
        return torch.zeros(1, 2, height, width, device=device)

    def decode(
        self,
        z: torch.Tensor,
        hdmap_z: torch.Tensor | None = None,
        hdmap_raw: torch.Tensor | None = None,
        return_validity: bool = False,
        spatial_coords: torch.Tensor | None = None,
    ):
        decoded = super().decode(
            z,
            hdmap_z=hdmap_z,
            hdmap_raw=hdmap_raw,
            return_validity=return_validity,
            spatial_coords=spatial_coords,
        )
        if not return_validity:
            return decoded

        recon, validity_logits = decoded
        if self.validity_prior_names:
            validity_logits = self._combine_validity_logits(validity_logits)
        if self.validity_logit_biases:
            validity_logits = self._add_validity_logit_bias(validity_logits)
        return recon, validity_logits

    def _add_validity_logit_bias(
        self, validity_logits: torch.Tensor
    ) -> torch.Tensor:
        bias = self.validity_logit_biases[self.active_dataset_id]
        if validity_logits.ndim == 5:
            bias = bias.unsqueeze(2)
        if tuple(validity_logits.shape[-2:]) != tuple(bias.shape[-2:]):
            raise ValueError(
                f"Dataset {self.active_dataset_id} validity-logit bias grid "
                f"{tuple(bias.shape[-2:])} does not match decoder output "
                f"{tuple(validity_logits.shape[-2:])}"
            )
        # Homogeneous batches rotate datasets between iterations. Keep every
        # bias map in every graph so DDP static_graph sees a fixed parameter set.
        dependency = validity_logits.new_zeros(())
        for item in self.validity_logit_biases:
            dependency = dependency + item.sum().to(validity_logits.dtype) * 0.0
        return validity_logits + bias.to(validity_logits.dtype) + dependency

    def _combine_validity_logits(
        self, residual_logits: torch.Tensor
    ) -> torch.Tensor:
        prior = getattr(
            self, self.validity_prior_names[self.active_dataset_id]
        )
        if residual_logits.ndim == 5:
            prior = prior.unsqueeze(2)
        if tuple(residual_logits.shape[-2:]) != tuple(prior.shape[-2:]):
            raise ValueError(
                f"Dataset {self.active_dataset_id} validity prior grid "
                f"{tuple(prior.shape[-2:])} does not match decoder output "
                f"{tuple(residual_logits.shape[-2:])}"
            )
        logits = prior.to(
            device=residual_logits.device, dtype=residual_logits.dtype
        ) + (
            self.validity_residual_gate.to(residual_logits.dtype)
            * residual_logits
        )
        return logits

    def forward(
        self,
        x: torch.Tensor,
        sensor_dataset_id: torch.Tensor | int,
        valid_mask: torch.Tensor | None = None,
        **kwargs,
    ):
        self.set_dataset_id(sensor_dataset_id)
        return super().forward(
            x,
            valid_mask=valid_mask,
            **kwargs,
        )
