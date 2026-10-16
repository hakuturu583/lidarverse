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

"""Implements the forward op for training, validation, and inference."""

from typing import Any

import torch
from einops import rearrange
from tokenizer.training.datasets.utils import IMAGE_KEY, INPUT_KEY, MASK_KEY, RECON_KEY, RECON_TARGET_KEY, VIDEO_KEY
from tokenizer.training.losses.continuous import RECON_CONSISTENCY_KEY, VIDEO_CONSISTENCY_LOSS, POINT_LOSS
from tokenizer.training.geo_converter import GeoConverter
from utils import ema, log
from utils.lazy_config import LazyDict, instantiate
from utils.model import Model

PREDICTION = "prediction"
EMA_PREDICTION = "ema_prediction"


class TokenizerModel(Model):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.network = instantiate(config.network)
        self.disc = instantiate(config.disc) if config.disc else config.disc
        self.loss = instantiate(config.loss)
        self.metric = instantiate(config.metric) 
        self.precision = getattr(torch, config.precision)
        if self.config.ema.enabled:
            self.ema = ema.EMAModelTracker(
                self,
                beta=self.config.ema.beta,
                torch_compile_buffer_renaming=self.config.ema.torch_compile_buffer_renaming,
            )
        self.init_input_keys()

        assert hasattr(self.config, "data_config"), "data_config is required for geo_converter"
        self.geo_converter = GeoConverter(self.config.data_config)

    def init_input_keys(self):
        self.image_key = IMAGE_KEY
        self.video_key = VIDEO_KEY

    def get_input_key(self, data_batch: dict[str, torch.Tensor]) -> str:
        if self.image_key in data_batch:
            return self.image_key
        elif self.video_key in data_batch:
            return self.video_key
        else:
            raise ValueError("Input key not found in data_batch.")

    def init_optimizer_scheduler(
        self, optimizer_config: LazyDict, scheduler_config: LazyDict
    ) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
        """Creates the optimizer and scheduler for the network.

        Args:
            optimizer_config: The optimizer config for the net.
            scheduler_config: The scheduler config for the net.

        Returns:
            optimizer (torch.optim.Optimizer): The net optimizer.
            scheduler (torch.optim.lr_scheduler.LRScheduler): The net optimization scheduler.
        """
        # Only pass trainable parameters to the optimizer. This makes
        # decoder-only fine-tuning work cleanly: frozen encoder params
        # don't waste optimizer state and don't trip DDP gradient checks.
        optimizer_config.params = [p for p in self.network.parameters() if p.requires_grad]
        optimizer = instantiate(optimizer_config)
        scheduler_config.optimizer = optimizer
        scheduler = instantiate(scheduler_config)

        return optimizer, scheduler

    def on_train_start(self, memory_format: torch.memory_format = torch.preserve_format) -> None:
        if self.config.ema.enabled:
            self.ema.to(dtype=torch.float32)
        self.network = self.network.to(dtype=self.precision, memory_format=memory_format)
        self.loss = self.loss.to(dtype=self.precision, memory_format=memory_format)

    def on_before_zero_grad(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        iteration: int,
    ) -> None:
        if self.config.ema.enabled:
            self.ema.update_average(self, iteration=iteration)

    def state_dict(
        self, destination: dict[str, Any] = None, prefix: str = "", keep_vars: bool = False
    ) -> dict[str, Any]:
        original_state_dict = super(TokenizerModel, self).state_dict(destination, prefix, keep_vars)

        # Filter out '.loss' and 'ema.loss-' keys from the state dict.
        filtered_state_dict = {k: v for k, v in original_state_dict.items() if not k.startswith("loss.")}
        filtered_state_dict = {k: v for k, v in filtered_state_dict.items() if not k.startswith("ema.loss-")}
        filtered_state_dict = {
            k: v for k, v in filtered_state_dict.items() if not k.startswith("network.encoder.patcher")
        }
        filtered_state_dict = {
            k: v for k, v in filtered_state_dict.items() if not k.startswith("network.decoder.unpatcher")
        }

        return filtered_state_dict

    def load_state_dict(self, state_dict: Any, strict: bool = True) -> None:
        if state_dict and all(key.startswith("network.") for key in state_dict):
            # Release exports contain the materialized EMA network, not a trainer EMA.
            result = self.network.load_state_dict(
                {key[len("network."):]: value for key, value in state_dict.items()}, strict=strict
            )
            if self.config.ema.enabled:
                beta = self.ema.beta
                try:
                    self.ema.beta = 0.0
                    self.ema.update_average(self)
                finally:
                    self.ema.beta = beta
            log.info(f"Loaded tokenizer release network and initialized training EMA: {result}")
            return
        own_state = self.state_dict()
        filtered_state_dict = {k: v for k, v in state_dict.items() if k in own_state}

        # Auto-prefix bare keys with "network." when loading pretrained VAE checkpoints
        if not filtered_state_dict:
            prefixed = {f"network.{k}": v for k, v in state_dict.items()}
            filtered_state_dict = {k: v for k, v in prefixed.items() if k in own_state}

        adapted_keys = []
        for key, value in tuple(filtered_state_dict.items()):
            if not hasattr(value, "shape") or own_state[key].shape == value.shape:
                continue
            adapted = self.adapt_warmstart_tensor(key, value, own_state[key])
            if adapted is not None:
                if adapted.shape != own_state[key].shape:
                    raise ValueError(
                        f"Warm-start adapter returned {adapted.shape} for {key}; "
                        f"expected {own_state[key].shape}"
                    )
                filtered_state_dict[key] = adapted
                adapted_keys.append(key)
        if adapted_keys:
            log.info(
                f"Channel-surgery adapted {len(adapted_keys)} checkpoint tensor(s): "
                f"{sorted(adapted_keys)}"
            )

        # Drop shape-mismatched tensors: torch's ``load_state_dict(strict=False)``
        # tolerates missing/unexpected keys but STILL hard-errors on size
        # mismatch. For a warm-start (``strict_resume=False``) we want the
        # compatible backbone to load and the incompatible layers to re-init
        # (e.g. AE->VAE head doubling z_factor, or a channel-count change), so we
        # skip mismatched keys here; they fall into ``missing_keys`` below and
        # raise only when ``strict``.
        shape_mismatch = [
            k for k, v in filtered_state_dict.items()
            if hasattr(v, "shape") and own_state[k].shape != v.shape
        ]
        if shape_mismatch:
            log.warning(
                f"Skipping {len(shape_mismatch)} shape-mismatched key(s) when loading "
                f"state_dict (will re-init): {sorted(shape_mismatch)}"
            )
            filtered_state_dict = {k: v for k, v in filtered_state_dict.items() if k not in shape_mismatch}

        # Load only filtered state dict.
        super(TokenizerModel, self).load_state_dict(filtered_state_dict, strict=False)

        # If strict is True, ensure all parameters are loaded (except the excluded ones)
        missing_keys = set(own_state.keys()) - set(filtered_state_dict.keys())
        if missing_keys and strict:
            raise KeyError(f"Missing keys in state_dict: {missing_keys}")

    def adapt_warmstart_tensor(
        self,
        name: str,
        source: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor | None:
        """Delegate opt-in checkpoint surgery to the tokenizer network."""
        if name.startswith("ema.network-"):
            network_name = name[len("ema.") :].replace("-", ".")
        elif name.startswith("network."):
            network_name = name
        else:
            return None
        adapter = getattr(self.network, "adapt_warmstart_tensor", None)
        if adapter is None:
            return None
        return adapter(network_name, source, target)

    def _on_before_network_forward(self, data_batch: dict[str, torch.Tensor]) -> None:
        consistency_loss = self.loss.loss_modules[VIDEO_CONSISTENCY_LOSS]
        if hasattr(consistency_loss, "enabled") and consistency_loss.enabled:
            _input_key = self.get_input_key(data_batch)
            if _input_key is self.video_key:
                data_batch[_input_key] = consistency_loss.shuffle(data_batch[_input_key])
        return

    def _on_after_network_forward(
        self, data_batch: dict[str, torch.Tensor], output_batch: dict[str, torch.Tensor]
    ) -> None:
        consistency_loss = self.loss.loss_modules[VIDEO_CONSISTENCY_LOSS]
        if hasattr(consistency_loss, "enabled") and consistency_loss.enabled:
            _input_key = self.get_input_key(data_batch)
            if _input_key is self.video_key:
                data_batch[_input_key] = consistency_loss.unshuffle(data_batch[_input_key])
                output_batch[RECON_CONSISTENCY_KEY] = torch.ones_like(output_batch[RECON_KEY]) * output_batch[RECON_KEY]
                output_batch[RECON_KEY] = consistency_loss.unshuffle(output_batch[RECON_KEY])
        
        # Compute the de-repeated metric point maps when point_loss OR normal
        # loss is enabled (NormalLoss finite-differences these into normals).
        point_loss = self.loss.loss_modules[POINT_LOSS]
        normal_loss = self.loss.loss_modules["normal"] if "normal" in self.loss.loss_modules else None
        chamfer_loss = self.loss.loss_modules["local_chamfer"] if "local_chamfer" in self.loss.loss_modules else None
        needs_points = (
            (hasattr(point_loss, "enabled") and point_loss.enabled)
            or (normal_loss is not None and getattr(normal_loss, "enabled", False))
            or (chamfer_loss is not None and getattr(chamfer_loss, "enabled", False))
        )
        if needs_points:
            geometry_kwargs = {}
            if data_batch.get("sensor_ray_directions") is not None:
                geometry_kwargs = {
                    "ray_directions": data_batch["sensor_ray_directions"],
                    "depth_bounds": data_batch["sensor_depth_bounds"],
                }
            output_batch[f"{RECON_KEY}_points"] = self.geo_converter(
                output_batch[RECON_KEY], **geometry_kwargs
            )
            output_batch[f"{RECON_TARGET_KEY}_points"] = self.geo_converter(
                data_batch[RECON_TARGET_KEY], **geometry_kwargs
            )
        return

    def _network_forward(self, data_batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        # A callback proxy to modify the input before the forward pass.
        if self.get_input_key(data_batch) == "images":
            if data_batch["images"].ndim == 5:  # input should be [B, C, H, W]
                data_batch["images"] = rearrange(data_batch["images"], "B C T H W -> (B T) C H W")
                if "recon_target" in data_batch:
                    data_batch["recon_target"] = rearrange(data_batch["recon_target"], "B C T H W -> (B T) C H W")
                # Keep the explicit valid mask in lockstep with recon_target so
                # MASK_KEY and the mask_as_input concat align in image mode.
                if data_batch.get("loss_mask") is not None and data_batch["loss_mask"].ndim == 5:
                    data_batch["loss_mask"] = rearrange(data_batch["loss_mask"], "B C T H W -> (B T) C H W")
        
        elif self.get_input_key(data_batch) == "video":  # the model takes as input (B, C, T, H, W)
            if data_batch["video"].size(2) == 3:
                data_batch["video"] = data_batch["video"].permute(0, 2, 1, 3, 4)

        self._on_before_network_forward(data_batch)

        # Do the forward pass.
        tensor_batch = data_batch[self.get_input_key(data_batch)]
        # Pass the hdmap conditioning tensor through as a kwarg ONLY when
        # both: (a) the dataloader provides one (datasets with
        # custom_fields=["lidar", "hdmap"] populate sample["lidar_hdmap"];
        # see rangemap_sampler.py), and (b) the network advertises that
        # it consumes hdmap (``uses_hdmap=True`` on the unwrapped module,
        # set by the hdmap-injection variants of TransformerVAE in their
        # __init__). Other networks (TransformerVAELTX, plain
        # TransformerVAE without the injection flags) keep their original
        # ``forward(x)`` signature and don't get the unexpected kwarg.
        forward_kwargs = {}
        unwrapped = getattr(self.network, "module", self.network)
        if (
            getattr(unwrapped, "uses_hdmap", False)
            and "lidar_hdmap" in data_batch
            and data_batch["lidar_hdmap"] is not None
        ):
            forward_kwargs["hdmap"] = data_batch["lidar_hdmap"]
        # Explicit per-pixel validity mask (computed by the dataloader BEFORE
        # per-row normalization). Threaded only when the network advertises it
        # consumes one (``uses_valid_mask`` -> mask_as_input nets) and the
        # dataloader populated ``loss_mask``. Used for the mask_as_input concat
        # in place of the (per-row-broken) depth>thr derivation.
        if (
            getattr(unwrapped, "uses_valid_mask", False)
            and "loss_mask" in data_batch
            and data_batch["loss_mask"] is not None
        ):
            forward_kwargs["valid_mask"] = data_batch["loss_mask"]
        if getattr(unwrapped, "uses_sensor_metadata", False):
            if data_batch.get("sensor_ray_directions") is None:
                raise KeyError(
                    "Cross-dataset tokenizer requires sensor_ray_directions"
                )
            forward_kwargs["sensor_ray_directions"] = data_batch[
                "sensor_ray_directions"
            ]
        if getattr(unwrapped, "uses_dataset_id", False):
            if data_batch.get("sensor_dataset_id") is None:
                raise KeyError(
                    "Dataset-embedding tokenizer requires sensor_dataset_id"
                )
            forward_kwargs["sensor_dataset_id"] = data_batch[
                "sensor_dataset_id"
            ]
        # Ego poses (lidar->world per frame) for the static/dynamic
        # decomposition network's ego warp. Threaded only when the network
        # advertises it consumes poses (``uses_pose`` -> decompose=True) and
        # the dataloader populated ``lidar_to_world`` (rangemap_sampler).
        if (
            getattr(unwrapped, "uses_pose", False)
            and "lidar_to_world" in data_batch
            and data_batch["lidar_to_world"] is not None
        ):
            forward_kwargs["pose"] = data_batch["lidar_to_world"]
        output_batch = self.network(tensor_batch, **forward_kwargs)
        output_batch = output_batch if self.network.training else output_batch._asdict()

        # A callback proxy to modify the output after the forward pass.
        self._on_after_network_forward(data_batch, output_batch)
        return output_batch

    def training_step(
        self,
        data_batch: dict[str, torch.Tensor],
        iteration: int,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        _input_key = self.get_input_key(data_batch)
        output_dict = self._network_forward(data_batch)
        input_images, recon_images, recon_target = (
            data_batch[_input_key], 
            output_dict[RECON_KEY], 
            data_batch["recon_target"]
        )

        # pass loss_mask and recon_target (always present)
        inputs = {
            INPUT_KEY: input_images,
            MASK_KEY: data_batch.get("loss_mask", torch.ones_like(recon_target)),
            RECON_TARGET_KEY: recon_target,
        }
        if data_batch.get("sensor_dataset_id") is not None:
            inputs["sensor_dataset_id"] = data_batch["sensor_dataset_id"]

        loss_dict, loss_value = self.loss(inputs, output_dict, iteration)
        return dict({PREDICTION: recon_images, **loss_dict}), loss_value

    @torch.no_grad()
    def validation_step(
        self,
        data_batch: dict[str, torch.Tensor],
        iteration: int,
        ema_model: bool = False,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        _input_key = self.get_input_key(data_batch)
        output_dict = self._network_forward(data_batch)
        input_images, recon_images, recon_target = (
            data_batch[_input_key], 
            output_dict[RECON_KEY], 
            data_batch["recon_target"]
        )

        # pass loss_mask and recon_target (always present)
        inputs = {
            INPUT_KEY: input_images,
            MASK_KEY: data_batch.get("loss_mask", torch.ones_like(recon_target)),
            RECON_TARGET_KEY: recon_target,
        }
        if data_batch.get("sensor_dataset_id") is not None:
            inputs["sensor_dataset_id"] = data_batch["sensor_dataset_id"]

        loss_dict, loss_value = self.loss(inputs, output_dict, iteration)
        metric_dict = self.metric(recon_target, output_dict, iteration)
        loss_dict.update(metric_dict)
        prediction_key = EMA_PREDICTION if ema_model else PREDICTION
        result = dict({prediction_key: recon_images, **loss_dict})
        # Surface predicted validity logits (predict_validity networks) so the
        # callback can visualize the mask + mask-filtered point cloud.
        if output_dict.get("validity_logits") is not None:
            result["validity_logits"] = output_dict["validity_logits"]
        # Surface multi-lidar (blind-spot) reconstruction for the BEV viz panel.
        for _k in (
            "extra_recon_points",
            "extra_gt_points",
            "extra_valid_points",
            "extra_recon_opacity",
        ):
            if output_dict.get(_k) is not None:
                result[_k] = output_dict[_k]
        return result, loss_value

    @torch.inference_mode()
    def forward(self, data_batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        _input_key = self.get_input_key(data_batch)
        output_dict = self._network_forward(data_batch)
        return dict({PREDICTION: output_dict[RECON_KEY]})
