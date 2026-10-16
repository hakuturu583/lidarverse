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

from __future__ import annotations

import os
import threading
from typing import List, NamedTuple, Tuple

import torch

from utils import distributed, log, misc
from utils.checkpointer import Checkpointer as BaseCheckpointer
from utils.model import Model

TORCH_VERSION: Tuple[int, ...] = tuple(int(x) for x in torch.__version__.split(".")[:2])
if TORCH_VERSION >= (1, 11):
    from torch.ao import quantization
    from torch.ao.quantization import FakeQuantizeBase, ObserverBase
elif (
    TORCH_VERSION >= (1, 8)
    and hasattr(torch.quantization, "FakeQuantizeBase")
    and hasattr(torch.quantization, "ObserverBase")
):
    from torch import quantization
    from torch.quantization import FakeQuantizeBase, ObserverBase


class _IncompatibleKeys(
    NamedTuple(
        "IncompatibleKeys",
        [
            ("missing_keys", List[str]),
            ("unexpected_keys", List[str]),
            ("incorrect_shapes", List[Tuple[str, Tuple[int], Tuple[int]]]),
        ],
    )
):
    pass


def _get_module_for_key(
    model: torch.nn.Module,
    key: str,
) -> torch.nn.Module:
    """Resolve the module owning one state-dict parameter or buffer."""
    current = model
    for key_part in key.split(".")[:-1]:
        current = getattr(current, key_part)
    return current


def _adapt_zero_init_input_expansion(
    target_module: torch.nn.Module,
    checkpoint_tensor: torch.Tensor,
    model_tensor: torch.Tensor,
) -> torch.Tensor | None:
    """Preserve prefix/suffix inputs around zero-initialized inserted slots."""
    layout = getattr(
        target_module,
        "_checkpoint_zero_init_input_expansion",
        None,
    )
    if layout is None:
        return None
    if checkpoint_tensor.ndim != 2 or model_tensor.ndim != 2:
        return None
    prefix_channels, old_channels, new_channels = map(int, layout)
    if (
        prefix_channels <= 0
        or old_channels < prefix_channels
        or new_channels <= old_channels
        or checkpoint_tensor.shape[0] != model_tensor.shape[0]
        or checkpoint_tensor.shape[1] % old_channels != 0
    ):
        return None
    patch_volume = checkpoint_tensor.shape[1] // old_channels
    if model_tensor.shape[1] != new_channels * patch_volume:
        return None
    prefix_features = prefix_channels * patch_volume
    suffix_features = (old_channels - prefix_channels) * patch_volume
    expanded = checkpoint_tensor.new_zeros(model_tensor.shape)
    expanded[:, :prefix_features] = checkpoint_tensor[
        :, :prefix_features
    ]
    if suffix_features > 0:
        expanded[:, -suffix_features:] = checkpoint_tensor[
            :, -suffix_features:
        ]
    return expanded


def _adapt_channel_remap(
    target_module: torch.nn.Module,
    checkpoint_tensor: torch.Tensor,
    model_tensor: torch.Tensor,
) -> torch.Tensor | None:
    """Remap selected channel spans while zero-initializing new channels."""
    layout = getattr(target_module, "_checkpoint_channel_remap", None)
    if layout is None or checkpoint_tensor.ndim != 2 or model_tensor.ndim != 2:
        return None
    kind, old_channels, new_channels, spans = layout
    old_channels = int(old_channels)
    new_channels = int(new_channels)
    spans = tuple(tuple(map(int, span)) for span in spans)
    if old_channels <= 0 or new_channels <= 0:
        return None
    expanded = checkpoint_tensor.new_zeros(model_tensor.shape)
    if kind == "input":
        if (
            checkpoint_tensor.shape[0] != model_tensor.shape[0]
            or checkpoint_tensor.shape[1] % old_channels != 0
        ):
            return None
        patch_volume = checkpoint_tensor.shape[1] // old_channels
        if model_tensor.shape[1] != new_channels * patch_volume:
            return None
        source = checkpoint_tensor.reshape(
            checkpoint_tensor.shape[0], old_channels, patch_volume
        )
        target = expanded.reshape(
            model_tensor.shape[0], new_channels, patch_volume
        )
        for old_start, old_stop, new_start, new_stop in spans:
            if old_stop - old_start != new_stop - new_start:
                return None
            target[:, new_start:new_stop] = source[:, old_start:old_stop]
        return expanded
    if kind == "output":
        if (
            checkpoint_tensor.shape[1] != model_tensor.shape[1]
            or checkpoint_tensor.shape[0] % old_channels != 0
        ):
            return None
        patch_volume = checkpoint_tensor.shape[0] // old_channels
        if model_tensor.shape[0] != new_channels * patch_volume:
            return None
        source = checkpoint_tensor.reshape(
            patch_volume, old_channels, checkpoint_tensor.shape[1]
        )
        target = expanded.reshape(
            patch_volume, new_channels, model_tensor.shape[1]
        )
        for old_start, old_stop, new_start, new_stop in spans:
            if old_stop - old_start != new_stop - new_start:
                return None
            target[:, new_start:new_stop] = source[:, old_start:old_stop]
        return expanded
    return None


class MultiRankCheckpointer(BaseCheckpointer):
    def save(
        self,
        model: Model,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        grad_scaler: torch.amp.GradScaler,
        iteration: int,
    ) -> None:
        """Save network weights, optimizer parameters, scheduler parameters to a checkpoint.

        Args:
            model (Model): The PyTorch model.
            optimizer (torch.optim.Optimizer): The model optimizer.
            scheduler (torch.optim.lr_scheduler.LRScheduler): The optimization scheduler.
            grad_scaler (torch.amp.GradScaler): The gradient scaler (for mixed precision training).
            iteration (int): Current iteration number.
        """
        postfix, _, total_ema_num = model.get_ckpt_postfix()
        checkpoint_file = f"ckpt_{postfix}.pt"
        save_ranks = list(range(total_ema_num))
        for _rank in save_ranks:
            if distributed.get_rank() == _rank:
                state_dict = dict(
                    model=model.state_dict(),
                    optimizer=optimizer.state_dict(),
                    scheduler=scheduler.state_dict(),
                    grad_scaler=grad_scaler.state_dict(),
                    iteration=iteration,
                )
                state_dict = misc.to(state_dict, device="cpu")
                self.callbacks.on_save_checkpoint(model, state_dict=state_dict)
                # Wait for previous saver thread to end.
                if self.save_thread:
                    self.save_thread.join()
                # Run the checkpoint saver in a separate thread.
                self.save_thread = threading.Thread(
                    target=self._save_worker_local,
                    daemon=False,
                    args=(state_dict, checkpoint_file, distributed.get_rank()),
                )
                self.save_thread.start()

    @misc.timer("checkpoint loading")
    def load(
        self,
        model: Model,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        grad_scaler: torch.amp.GradScaler | None = None,
    ) -> int:
        """Load network weights and optimizer states from a checkpoint in a single process.

        The priority of the checkpoint loading logic is:
        1. Attempt to resume training if possible by looking for latest_checkpoint.txt under the same name.
        2. If no latest checkpoint were found, it loads the model weights specified by config_checkpoint.path.
           - This is typically used for inference mode.
           - If config_checkpoint.load_optimizer_state is True, then also load the optimizer and scheduler states.
        3. If none of the above, randomly initialize the model parameters and train from scratch.

        Args:
            model (Model): The PyTorch model.
            optimizer (torch.optim.Optimizer | None): The model optimizer (default: None).
            scheduler (torch.optim.lr_scheduler.LRScheduler | None): The optimization scheduler (default: None).
            grad_scaler (torch.amp.GradScaler | None): The gradient scaler (for mixed precision training).

        Returns:
            iteration (int): the iteration number to start/resume from.
        """
        latest_checkpoint_file = self._read_latest_checkpoint_file()
        if latest_checkpoint_file is not None:
            # different from base checkpointer, this support multi-EMA
            postfix, _, total_ema_num = model.get_ckpt_postfix()
            latest_checkpoint_file = latest_checkpoint_file.replace(".pt", f"{postfix}.pt")
            # 1. Resume training from latest_checkpoint.txt under the same name.
            checkpoint_dir = self.checkpoint_dir_local
            checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint_file)
            resume = True
        else:
            if self.load_path:
                # 2. Load the module weights specified by config_checkpoint.path.
                checkpoint_path = self.load_path
                # different from base checkpointer, this support multi-EMA
                postfix, _, total_ema_num = model.get_ckpt_postfix()
                checkpoint_path = checkpoint_path.replace(".pt", f"{postfix}.pt")
                resume = self.load_training_state
            else:
                # 3. Randomly initialize the model parameters and train from scratch.
                checkpoint_path = None
                resume = False
        # Load checkpoint.
        if checkpoint_path is not None:
            self._check_checkpoint_exists(checkpoint_path)
            log.info(f"Loading checkpoint (local): {checkpoint_path}")
            state_dict = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
            log.success(f"Complete loading checkpoint (local): {checkpoint_path}")
            self.callbacks.on_load_checkpoint(model, state_dict=state_dict)
            # Load the state dicts.
            log.info("- Loading the model...")
            log.critical(model.load_state_dict(state_dict["model"], strict=self.strict_resume))
            if resume:
                iteration = state_dict["iteration"]
                assert optimizer and scheduler
                log.info("- Loading the optimizer...")
                optimizer.load_state_dict(state_dict["optimizer"])
                log.info("- Loading the scheduler...")
                scheduler.load_state_dict(state_dict["scheduler"])
                scheduler.last_epoch = iteration
                log.info("- Loading the gradient scaler...")
                grad_scaler.load_state_dict(state_dict["grad_scaler"])
                log.success(f"Done with loading the checkpoint (iteration {iteration}).")
            else:
                iteration = 0
                log.success("Done with loading the checkpoint.")
        else:
            # Checkpoint not found and not specified. We will train everything from scratch.
            iteration = 0
            log.info("Training from scratch.")
        torch.cuda.empty_cache()
        return iteration


# https://github.com/facebookresearch/fvcore/blob/9d683aae73fb899dd35d6cf6720e5ef567761c57/fvcore/common/checkpoint.py
def non_strict_load_model(model: torch.nn.Module, checkpoint_state_dict: dict) -> _IncompatibleKeys:
    # workaround https://github.com/pytorch/pytorch/issues/24139
    model_state_dict = model.state_dict()
    incorrect_shapes = []
    for k in list(checkpoint_state_dict.keys()):
        if k in model_state_dict:
            if "_extra_state" in k:  # Key introduced by TransformerEngine for FP8
                log.warning(f"Skipping key {k} introduced by TransformerEngine for FP8 in the checkpoint.")
                continue
            model_param = model_state_dict[k]
            # Allow mismatch for uninitialized parameters
            if TORCH_VERSION >= (1, 8) and isinstance(model_param, torch.nn.parameter.UninitializedParameter):
                continue
            if not isinstance(model_param, torch.Tensor):
                raise ValueError(
                    f"Find non-tensor parameter {k} in the model. type: {type(model_param)} {type(checkpoint_state_dict[k])}, please check if this key is safe to skip or not."
                )

            shape_model = tuple(model_param.shape)
            shape_checkpoint = tuple(checkpoint_state_dict[k].shape)
            if shape_model != shape_checkpoint:
                original_shape_checkpoint = shape_checkpoint
                target_module = _get_module_for_key(model, k)
                adapted = _adapt_channel_remap(
                    target_module,
                    checkpoint_state_dict[k],
                    model_param,
                )
                if adapted is None:
                    adapted = _adapt_zero_init_input_expansion(
                        target_module,
                        checkpoint_state_dict[k],
                        model_param,
                    )
                if adapted is not None:
                    checkpoint_state_dict[k] = adapted
                    shape_checkpoint = tuple(adapted.shape)
                    log.info(
                        "Adapted checkpoint channel layout "
                        f"{k}: {original_shape_checkpoint} -> "
                        f"{shape_checkpoint}"
                    )
            if shape_model != shape_checkpoint:
                has_observer_base_classes = (
                    TORCH_VERSION >= (1, 8)
                    and hasattr(quantization, "ObserverBase")
                    and hasattr(quantization, "FakeQuantizeBase")
                )
                if has_observer_base_classes:
                    # Handle the special case of quantization per channel observers,
                    # where buffer shape mismatches are expected.
                    cls_to_skip = (
                        ObserverBase,
                        FakeQuantizeBase,
                    )
                    target_module = _get_module_for_key(model, k)
                    if isinstance(target_module, cls_to_skip):
                        # Do not remove modules with expected shape mismatches
                        # them from the state_dict loading. They have special logic
                        # in _load_from_state_dict to handle the mismatches.
                        continue

                incorrect_shapes.append((k, shape_checkpoint, shape_model))
                checkpoint_state_dict.pop(k)
    incompatible = model.load_state_dict(checkpoint_state_dict, strict=False)
    # Remove keys with "_extra_state" suffix, which are non-parameter items introduced by TransformerEngine for FP8 handling
    missing_keys = [k for k in incompatible.missing_keys if "_extra_state" not in k]
    unexpected_keys = [k for k in incompatible.unexpected_keys if "_extra_state" not in k]
    return _IncompatibleKeys(
        missing_keys=missing_keys,
        unexpected_keys=unexpected_keys,
        incorrect_shapes=incorrect_shapes,
    )
