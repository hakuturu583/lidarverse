# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cycle-based trainer and checkpointing for LiDAR Self-Forcing DMD."""

from __future__ import annotations

import gc
import os
import re
import time
from contextlib import nullcontext
from typing import Any

import torch
from torch.distributed.fsdp import FullStateDictConfig, StateDictType
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from diffusion.checkpointers.ema_fsdp_checkpointer import (
    FSDPCheckpointer as EMAFSDPCheckpointer,
)
from diffusion.training.trainer import Trainer
from diffusion.training.utils.optim_instantiate import get_base_scheduler
from diffusion.training.utils.sf_dmd import normalized_frame_weights
from utils import distributed, log, misc
from utils.atomic_save import atomic_torch_save
from utils.fsdp_checkpointer import FSDPCheckpointer as BaseFSDPCheckpointer
from utils.fsdp_optim_fix import scatter_full_optim_state_dict
from utils.lazy_config import instantiate as lazy_instantiate


class SFDMDCheckpointer(EMAFSDPCheckpointer):
    """Add fake-score model/optimizer state beside the normal student files."""

    _GENERATION_MARKER_RE = re.compile(r"^iter_(\d{9})\.pt$")
    _GENERATION_FILE_RE = re.compile(r"^iter_(\d{9}).*\.pt$")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._resolved_external_checkpoint: str | None = None
        self.keep_last_checkpoints = int(
            getattr(self.config_checkpoint, "keep_last_checkpoints", 0)
        )
        self.save_at_iters = frozenset(
            int(iteration)
            for iteration in getattr(
                self.config_checkpoint, "save_at_iters", []
            )
        )
        if self.keep_last_checkpoints < 0:
            raise ValueError(
                "checkpoint.keep_last_checkpoints must be non-negative, "
                f"got {self.keep_last_checkpoints}"
            )
        if any(iteration <= 0 for iteration in self.save_at_iters):
            raise ValueError(
                "checkpoint.save_at_iters must contain positive iterations, "
                f"got {sorted(self.save_at_iters)}"
            )

    def should_save(self, cycle: int) -> bool:
        save_iter = int(self.config_checkpoint.save_iter)
        if save_iter <= 0:
            raise ValueError(
                f"checkpoint.save_iter must be positive, got {save_iter}"
            )
        return cycle % save_iter == 0 or cycle in self.save_at_iters

    def _load_ckpt_file_during_init(self):
        checkpoint_path, resume = super()._load_ckpt_file_during_init()
        if checkpoint_path is None or not os.path.isdir(checkpoint_path):
            return checkpoint_path, resume
        if self._resolved_external_checkpoint is None:
            latest_path = os.path.join(
                checkpoint_path, "latest_checkpoint.txt"
            )
            self._check_checkpoint_exists(latest_path)
            with open(latest_path) as file:
                latest = file.read().strip()
            if not latest:
                raise RuntimeError(
                    f"empty external latest checkpoint marker: {latest_path}"
                )
            resolved = os.path.join(checkpoint_path, latest)
            self._check_checkpoint_exists(resolved)
            self._resolved_external_checkpoint = resolved
            log.critical(
                "[Checkpoint] Resolved external SF-DMD directory "
                f"{checkpoint_path} to {resolved}"
            )
        return self._resolved_external_checkpoint, resume

    @staticmethod
    def _extra_path(checkpoint_path: str, suffix: str) -> str:
        return checkpoint_path.replace(".pt", suffix)

    def _prune_old_generations(self, current_cycle: int) -> None:
        """Delete complete SF-DMD generations older than the configured limit."""
        if self.keep_last_checkpoints == 0:
            return

        filenames = os.listdir(self.checkpoint_dir_local)
        completed_cycles = []
        for filename in filenames:
            match = self._GENERATION_MARKER_RE.fullmatch(filename)
            if match is not None:
                cycle = int(match.group(1))
                if cycle <= current_cycle:
                    completed_cycles.append(cycle)
        completed_cycles.sort()
        ordinary_cycles = [
            cycle
            for cycle in completed_cycles
            if cycle not in self.save_at_iters
        ]
        stale_cycles = set(
            ordinary_cycles[: -self.keep_last_checkpoints]
        )
        if not stale_cycles:
            return

        removed_files = 0
        for filename in filenames:
            match = self._GENERATION_FILE_RE.fullmatch(filename)
            if match is None or int(match.group(1)) not in stale_cycles:
                continue
            os.remove(os.path.join(self.checkpoint_dir_local, filename))
            removed_files += 1
        log.info(
            "Pruned SF-DMD checkpoint generations "
            f"{sorted(stale_cycles)} ({removed_files} files); retained the "
            f"latest {self.keep_last_checkpoints} non-target generations and "
            f"target iterations {sorted(self.save_at_iters)}"
        )

    def get_fake_score_load_path(self) -> str | None:
        latest = self._read_latest_checkpoint_file()
        if latest is not None:
            checkpoint_path = os.path.join(
                self.checkpoint_dir_local, latest
            )
        else:
            checkpoint_path, resume = self._load_ckpt_file_during_init()
            if checkpoint_path is None or not resume:
                return None
        fake_path = self._extra_path(checkpoint_path, "_fake_score_model.pt")
        self._check_checkpoint_exists(fake_path)
        return fake_path

    def load_model_during_init(
        self, model, is_ema: bool = False, ema_id: int = 0
    ):
        # EMA->regular applies only to the external Stage-2 warm-start. A local
        # SF-DMD resume must restore its independently evolved regular student.
        checkpoint_path, resume = self._load_ckpt_file_during_init()
        if resume:
            return BaseFSDPCheckpointer.load_model_during_init(
                self, model, is_ema=is_ema, ema_id=ema_id
            )
        if (
            is_ema
            and not self.load_ema_to_reg
            and checkpoint_path is not None
        ):
            ema_path = checkpoint_path.replace(".pt", "_ema_model.pt")
            regular_path = checkpoint_path.replace(".pt", "_reg_model.pt")
            if not os.path.exists(ema_path) and os.path.exists(regular_path):
                log.critical(
                    "External warm-start has no EMA payload; initializing "
                    f"Stage-3 EMA from regular student {regular_path}."
                )
                return BaseFSDPCheckpointer.load_model_during_init(
                    self, model, is_ema=False, ema_id=0
                )
        return super().load_model_during_init(
            model, is_ema=is_ema, ema_id=ema_id
        )

    @staticmethod
    def _full_model_state(model: torch.nn.Module) -> dict[str, Any]:
        if isinstance(model, FSDP):
            with FSDP.state_dict_type(
                model,
                StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
            ):
                return model.state_dict()
        return {
            key: value.detach().cpu()
            for key, value in model.state_dict().items()
        }

    @staticmethod
    def _expose_master_weights(optimizer: torch.optim.Optimizer) -> None:
        groups_master = getattr(optimizer, "param_groups_master", None)
        if groups_master is None:
            return
        for group, master_group in zip(
            optimizer.param_groups, groups_master
        ):
            for parameter, master in zip(
                group["params"], master_group["params"]
            ):
                if master is not None:
                    optimizer.state[parameter][
                        "_sfdmd_master_param"
                    ] = master

    @staticmethod
    def _restore_master_weights(optimizer: torch.optim.Optimizer) -> None:
        if not getattr(optimizer, "master_weights", False):
            return
        groups_master = []
        for group in optimizer.param_groups:
            masters = []
            for parameter in group["params"]:
                master = optimizer.state[parameter].get(
                    "_sfdmd_master_param"
                )
                if master is None:
                    # External warm-starts have no optimizer state.
                    optimizer.param_groups_master = None
                    return
                masters.append(
                    master.to(
                        device=parameter.device,
                        dtype=torch.float32,
                    )
                )
            groups_master.append({"params": masters})
        optimizer.param_groups_master = groups_master

    def get_optim_scheduler_state(self, optim, fsdp_model, scheduler):
        self._expose_master_weights(optim)
        return super().get_optim_scheduler_state(
            optim, fsdp_model, scheduler
        )

    def _load_optimizer_file(
        self,
        path: str,
        fsdp_model: FSDP,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
    ) -> None:
        error = None
        if distributed.get_rank() == 0:
            try:
                state = torch.load(
                    path,
                    map_location=lambda storage, loc: storage,
                    weights_only=False,
                )
                full_optim_state = state["optimizer"]
                scheduler_state = state["scheduler"]
            except Exception as exc:  # noqa: BLE001
                full_optim_state = None
                scheduler_state = None
                error = repr(exc)
        else:
            full_optim_state = None
            scheduler_state = None
        errors = [error]
        torch.distributed.broadcast_object_list(errors, src=0)
        if errors[0] is not None:
            raise RuntimeError(f"optimizer load failed: {errors[0]}")
        scatter_error = None
        try:
            sharded = scatter_full_optim_state_dict(
                full_optim_state,
                fsdp_model,
            )
        except Exception as exc:  # noqa: BLE001
            sharded = None
            scatter_error = f"rank {distributed.get_rank()}: {exc!r}"
        scatter_errors = [None] * distributed.get_world_size()
        torch.distributed.all_gather_object(
            scatter_errors, scatter_error
        )
        failures = [item for item in scatter_errors if item is not None]
        if failures:
            raise RuntimeError(
                f"optimizer-state scatter failed: {failures}"
            )

        local_error = None
        try:
            optimizer.load_state_dict(sharded)
        except Exception as exc:  # noqa: BLE001
            local_error = f"rank {distributed.get_rank()}: {exc!r}"
        load_errors = [None] * distributed.get_world_size()
        torch.distributed.all_gather_object(load_errors, local_error)
        failures = [item for item in load_errors if item is not None]
        if failures:
            raise RuntimeError(
                f"optimizer.load_state_dict failed: {failures}"
            )

        objects = [scheduler_state]
        torch.distributed.broadcast_object_list(objects, src=0)
        local_error = None
        try:
            scheduler.load_state_dict(objects[0])
            self._restore_master_weights(optimizer)
        except Exception as exc:  # noqa: BLE001
            local_error = f"rank {distributed.get_rank()}: {exc!r}"
        load_errors = [None] * distributed.get_world_size()
        torch.distributed.all_gather_object(load_errors, local_error)
        failures = [item for item in load_errors if item is not None]
        if failures:
            raise RuntimeError(
                f"scheduler/master restore failed: {failures}"
            )

    def load_optim_scheduler_during_init(
        self,
        fsdp_model,
        optimizer,
        scheduler,
    ):
        checkpoint_path, resume = self._load_ckpt_file_during_init()
        if not resume or checkpoint_path is None:
            return
        path = checkpoint_path.replace(".pt", "_optim.pt")
        self._check_checkpoint_exists(path)
        self._load_optimizer_file(
            path, fsdp_model, optimizer, scheduler
        )
        log.info(f"Loaded SF-DMD student optimizer from {path}")

    def load_fake_optimizer(
        self,
        fake_model: FSDP,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
    ) -> None:
        checkpoint_path, resume = self._load_ckpt_file_during_init()
        if not resume or checkpoint_path is None:
            return
        path = self._extra_path(checkpoint_path, "_fake_optim.pt")
        self._check_checkpoint_exists(path)
        self._load_optimizer_file(
            path, fake_model, optimizer, scheduler
        )
        log.info(f"Loaded SF-DMD fake-score optimizer from {path}")

    def save_sfdmd(
        self,
        model,
        student_optimizer: torch.optim.Optimizer,
        student_scheduler: torch.optim.lr_scheduler.LRScheduler,
        fake_optimizer: torch.optim.Optimizer,
        fake_scheduler: torch.optim.lr_scheduler.LRScheduler,
        grad_scaler: torch.amp.GradScaler,
        cycle: int,
    ) -> None:
        """Write a versioned multi-file generation, then publish one marker."""
        del grad_scaler
        self.callbacks.on_save_checkpoint_start(model, cycle)
        collection_error = None
        try:
            student_state = model.state_dict_model()
            student_optim_state = self.get_optim_scheduler_state(
                student_optimizer,
                model.model,
                student_scheduler,
            )
            fake_state = self._full_model_state(model.fake_score)
            fake_optim_state = self.get_optim_scheduler_state(
                fake_optimizer,
                model.fake_score,
                fake_scheduler,
            )
        except Exception as exc:  # noqa: BLE001
            student_state = None
            student_optim_state = None
            fake_state = None
            fake_optim_state = None
            collection_error = (
                f"rank {distributed.get_rank()}: {exc!r}"
            )
        collection_errors = [None] * distributed.get_world_size()
        torch.distributed.all_gather_object(
            collection_errors, collection_error
        )
        failures = [
            item for item in collection_errors if item is not None
        ]
        if failures:
            raise RuntimeError(
                f"SF-DMD state collection failed: {failures}"
            )
        callback_state = {"iteration": cycle}
        self.callbacks.on_save_checkpoint(
            model, state_dict=callback_state
        )
        postfix, replicate_idx, shard_idx, total_ema_num = (
            model.get_ckpt_postfix()
        )
        checkpoint_file = f"iter_{cycle:09d}{postfix}.pt"
        checkpoint_path = os.path.join(
            self.checkpoint_dir_local, checkpoint_file
        )
        local_error = None
        try:
            os.makedirs(self.checkpoint_dir_local, exist_ok=True)
            if replicate_idx == 0 and shard_idx == 0:
                atomic_torch_save(
                    student_state["model"],
                    self._extra_path(checkpoint_path, "_reg_model.pt"),
                )
                atomic_torch_save(
                    student_state["ema"],
                    self._extra_path(checkpoint_path, "_ema_model.pt"),
                )
                atomic_torch_save(
                    student_optim_state,
                    self._extra_path(checkpoint_path, "_optim.pt"),
                )
                atomic_torch_save(fake_state, self._extra_path(
                    checkpoint_path, "_fake_score_model.pt"
                ))
                atomic_torch_save(fake_optim_state, self._extra_path(
                    checkpoint_path, "_fake_optim.pt"
                ))
                atomic_torch_save(callback_state, checkpoint_path)
            elif (
                replicate_idx < total_ema_num
                and shard_idx == 0
                and student_state["ema"] is not None
            ):
                atomic_torch_save(
                    student_state["ema"],
                    self._extra_path(checkpoint_path, "_ema_model.pt"),
                )
        except Exception as exc:  # noqa: BLE001
            local_error = f"rank {distributed.get_rank()}: {exc!r}"

        errors = [None] * distributed.get_world_size()
        torch.distributed.all_gather_object(errors, local_error)
        failures = [error for error in errors if error is not None]
        if failures:
            raise RuntimeError(
                "SF-DMD checkpoint generation failed; latest pointer was not "
                f"updated: {failures}"
            )
        marker_error = None
        if distributed.get_rank() == 0:
            try:
                self._write_latest_checkpoint_file(
                    f"iter_{cycle:09d}.pt"
                )
            except Exception as exc:  # noqa: BLE001
                marker_error = repr(exc)
        marker_errors = [marker_error]
        torch.distributed.broadcast_object_list(marker_errors, src=0)
        if marker_errors[0] is not None:
            raise RuntimeError(
                "SF-DMD checkpoint files are complete but latest marker "
                f"publication failed: {marker_errors[0]}"
            )
        if distributed.get_rank() == 0:
            try:
                # Prune only after the new generation is complete and its
                # atomic latest pointer has been published. A crash during
                # cleanup can therefore leave extras, never break resume.
                self._prune_old_generations(cycle)
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "SF-DMD checkpoint retention cleanup failed; keeping "
                    f"older generations: {exc!r}"
                )
        self.callbacks.on_save_checkpoint_success(iteration=cycle)
        self.callbacks.on_save_checkpoint_end(model=model, iteration=cycle)


class SFDMDTrainer(Trainer):
    """Train legacy SF-DMD or the ArtiFixer-aligned update schedule."""

    def __init__(self, config):
        super().__init__(config)
        self.checkpointer = SFDMDCheckpointer(
            config.checkpoint,
            config.job,
            callbacks=self.callbacks,
        )

    def _init_optimizers(self, model):
        student_optimizer, student_scheduler = (
            model.init_optimizer_scheduler(
                self.config.optimizer,
                self.config.scheduler,
            )
        )
        fake_optimizer = lazy_instantiate(
            self.config.optimizer_fake,
            model=model.fake_score,
        )
        fake_scheduler = get_base_scheduler(
            fake_optimizer,
            model,
            self.config.scheduler_fake,
        )
        self.checkpointer.load_fake_optimizer(
            model.fake_score,
            fake_optimizer,
            fake_scheduler,
        )
        return (
            student_optimizer,
            student_scheduler,
            fake_optimizer,
            fake_scheduler,
        )

    @staticmethod
    def _mean_metrics(
        metric_rows: list[dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        if not metric_rows:
            return {}
        output = {}
        for key in metric_rows[0]:
            values = [row[key].detach().float() for row in metric_rows]
            output[key] = torch.stack(values).mean()
        return output

    def _fake_updates(
        self,
        model,
        rollout,
        optimizer,
        scheduler,
        grad_scaler,
        cycle: int,
    ) -> list[dict[str, torch.Tensor]]:
        rows = []
        model.fake_score.train()
        if (
            model.sfdmd_artifixer_matching
            or model.sfdmd_imaginaire_matching
        ):
            optimizer.zero_grad(set_to_none=True)
            num_records = (
                1
                if model.sfdmd_imaginaire_matching
                else len(rollout.records)
            )
            record_weights = (
                [1.0]
                if model.sfdmd_imaginaire_matching
                else normalized_frame_weights(
                    [
                        record.score_active_end
                        - record.score_active_start
                        for record in rollout.records
                    ]
                )
            )
            score_batch = model.new_score_batch(rollout)
            for record_idx in range(num_records):
                sync_context = (
                    nullcontext()
                    if record_idx == num_records - 1
                    else model.fake_score.no_sync()
                )
                with sync_context:
                    loss, metrics = model.fake_score_loss(
                        rollout,
                        record_idx,
                        score_batch=score_batch,
                    )
                    weighted_loss = loss * record_weights[record_idx]
                    self.callbacks.on_before_backward(
                        model, weighted_loss, iteration=cycle
                    )
                    grad_scaler.scale(weighted_loss).backward()
                    model.on_after_backward(iteration=cycle)
                    self.callbacks.on_after_backward(
                        model, iteration=cycle
                    )
                rows.append(metrics)
            grads = [
                parameter.grad
                for parameter in model.fake_score.parameters()
                if parameter.grad is not None
            ]
            for grad in grads:
                torch.nan_to_num_(
                    grad, nan=0.0, posinf=0.0, neginf=0.0
                )
            if isinstance(model.fake_score, FSDP):
                model.fake_score.clip_grad_norm_(
                    model.sfdmd_max_grad_norm
                )
            else:
                torch.nn.utils.clip_grad_norm_(
                    model.fake_score.parameters(),
                    model.sfdmd_max_grad_norm,
                )
            grad_scaler.step(optimizer)
            grad_scaler.update()
            scheduler.step()
            # The student phase follows immediately and does not use fake-score
            # gradients. Release their FSDP flat buffers before student replay.
            optimizer.zero_grad(set_to_none=True)
            return rows

        for update_idx in range(model.sfdmd_fake_updates):
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = model.fake_score_loss(
                rollout,
                cycle * model.sfdmd_fake_updates + update_idx,
            )
            self.callbacks.on_before_backward(
                model, loss, iteration=cycle
            )
            grad_scaler.scale(loss).backward()
            model.on_after_backward(iteration=cycle)
            self.callbacks.on_after_backward(model, iteration=cycle)
            # The standard grad-clip callback targets model.model (student).
            # Apply the same finite+norm policy explicitly to fake_score.
            grads = [
                parameter.grad
                for parameter in model.fake_score.parameters()
                if parameter.grad is not None
            ]
            for grad in grads:
                torch.nan_to_num_(
                    grad, nan=0.0, posinf=0.0, neginf=0.0
                )
            if isinstance(model.fake_score, FSDP):
                model.fake_score.clip_grad_norm_(
                    model.sfdmd_max_grad_norm
                )
            else:
                torch.nn.utils.clip_grad_norm_(
                    model.fake_score.parameters(),
                    model.sfdmd_max_grad_norm,
                )
            grad_scaler.step(optimizer)
            grad_scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            rows.append(metrics)
        return rows

    def _student_update(
        self,
        model,
        rollout,
        optimizer,
        scheduler,
        grad_scaler,
        cycle: int,
        target_map=None,
    ) -> list[dict[str, torch.Tensor]]:
        rows = []
        optimizer.zero_grad(set_to_none=True)
        num_records = len(rollout.records)
        record_weights = (
            None
            if model.sfdmd_imaginaire_matching
            else normalized_frame_weights(
                [
                    record.score_active_end - record.score_active_start
                    for record in rollout.records
                ]
            )
        )
        kv_cache = model.new_student_kv_cache()
        if rollout.conditioned_prefix_frames > 0:
            model.seed_student_conditioned_prefix(rollout, kv_cache)
        tbptt_blocks = model.sfdmd_tbptt_blocks
        # FSDP no_sync retains full gradients, defeating memory-bounded TBPTT.
        sync_each_segment = model.sfdmd_allow_truncated_bptt
        for segment_start in range(0, num_records, tbptt_blocks):
            segment_end = min(segment_start + tbptt_blocks, num_records)
            sync_context = (
                nullcontext()
                if sync_each_segment or segment_end == num_records
                else model.model.no_sync()
            )
            with sync_context:
                segment_loss = None
                for record_idx in range(segment_start, segment_end):
                    record = rollout.records[record_idx]
                    if cycle == 1 and distributed.get_rank() == 0:
                        gib = 1024**3
                        log.info(
                            "Student replay memory before block "
                            f"{record.block_index}: allocated="
                            f"{torch.cuda.memory_allocated() / gib:.2f} GiB, "
                            f"reserved={torch.cuda.memory_reserved() / gib:.2f} GiB"
                        )
                    loss, metrics = model.student_replay_loss(
                        record,
                        kv_cache,
                        None
                        if target_map is None
                        else target_map[record.block_index],
                    )
                    scaled_loss = (
                        loss
                        if model.sfdmd_imaginaire_matching
                        else loss * record_weights[record_idx]
                    )
                    segment_loss = (
                        scaled_loss
                        if segment_loss is None
                        else segment_loss + scaled_loss
                    )
                    rows.append(metrics)
                assert segment_loss is not None
                self.callbacks.on_before_backward(
                    model, segment_loss, iteration=cycle
                )
                grad_scaler.scale(segment_loss).backward()
                model.on_after_backward(iteration=cycle)
                self.callbacks.on_after_backward(model, iteration=cycle)
            # Preserve sink/recent contents while cutting the graph only at the
            # configured replay-segment boundary. Canonical ArtiFixer uses one
            # full-rollout segment; explicit memory-bounded handoffs detach at
            # their configured truncated-BPTT boundary.
            model.detach_student_kv_cache(kv_cache)
        self.callbacks.on_before_optimizer_step(
            model,
            optimizer,
            scheduler,
            grad_scaler,
            iteration=cycle,
        )
        grad_scaler.step(optimizer)
        grad_scaler.update()
        scheduler.step()
        self.callbacks.on_before_zero_grad(
            model,
            optimizer,
            scheduler,
            iteration=cycle,
        )
        if model.sfdmd_artifixer_matching and model.config.ema.enabled:
            beta = (
                0.0
                if cycle < model.sfdmd_ema_start_cycle
                else float(model.config.ema.beta)
            )
            ema_worker = getattr(model, "model_ema_worker", None)
            ema_model = getattr(model, "model_ema", None)
            if ema_worker is None or ema_model is None:
                raise RuntimeError(
                    "ArtiFixer-aligned SF-DMD requires an initialized EMA "
                    "worker and model"
                )
            ema_worker.update_average(
                model.model,
                ema_model,
                beta=beta,
            )
        else:
            ema_iteration = (
                cycle // model.sfdmd_student_update_freq
                if model.sfdmd_imaginaire_matching
                else cycle
            )
            model.on_before_zero_grad(
                optimizer,
                scheduler,
                iteration=ema_iteration,
            )
        optimizer.zero_grad(set_to_none=True)
        return rows

    def train(
        self,
        model,
        dataloader_train: torch.utils.data.DataLoader,
        dataloader_val: torch.utils.data.DataLoader,
    ) -> None:
        del dataloader_val
        model = model.to(
            "cuda", memory_format=self.config.trainer.memory_format
        )
        model.on_train_start(self.config.trainer.memory_format)
        self._log_param_count(model)

        self.callbacks.on_optimizer_init_start()
        (
            student_optimizer,
            student_scheduler,
            fake_optimizer,
            fake_scheduler,
        ) = self._init_optimizers(model)
        grad_scaler = torch.amp.GradScaler(
            "cuda", **self.config.trainer.grad_scaler_args
        )
        self.callbacks.on_optimizer_init_end()
        cycle = self.checkpointer.load(
            model,
            student_optimizer,
            student_scheduler,
            grad_scaler,
        )
        self.callbacks.on_train_start(model, iteration=cycle)
        dataloader_iter = iter(dataloader_train)
        run_kv_gate = bool(model.sfdmd_run_kv_equivalence_gate)
        kv_gate_metrics = None if cycle == 0 and run_kv_gate else {}
        if not run_kv_gate:
            log.info(
                "Skipping the DiT KV-cache equivalence gate because "
                "model.sfdmd_run_kv_equivalence_gate=False."
            )
        elif cycle > 0:
            log.info(
                "Skipping the DiT KV-cache equivalence gate when resuming "
                f"from cycle {cycle}; the gate runs only before cycle 0."
            )

        while cycle < self.config.trainer.max_iter:
            self.callbacks.on_before_dataloading(cycle)
            try:
                data_batch = next(dataloader_iter)
            except StopIteration:
                dataloader_iter = iter(dataloader_train)
                data_batch = next(dataloader_iter)
            self.callbacks.on_after_dataloading(cycle)
            data_batch = misc.to(data_batch, device="cuda")
            self.callbacks.on_training_step_start(
                model, data_batch, iteration=cycle
            )
            model.train()
            if kv_gate_metrics is None:
                kv_gate_metrics = model.verify_student_kv_equivalence(
                    data_batch
                )

            cycle_start = time.perf_counter()
            rollout_start = cycle_start
            rollout = model.rollout_sfdmd(data_batch, cycle)
            rollout_seconds = time.perf_counter() - rollout_start

            if model.sfdmd_imaginaire_matching:
                student_due = (
                    cycle % model.sfdmd_student_update_freq == 0
                )
            else:
                student_due = (
                    not model.sfdmd_artifixer_matching
                    or (
                        (cycle - 1)
                        % model.sfdmd_student_update_freq
                        == 0
                    )
                )
            target_start = time.perf_counter()
            student_targets = (
                model.artifixer_student_targets(rollout)
                if (
                    (
                        model.sfdmd_artifixer_matching
                        or model.sfdmd_imaginaire_matching
                    )
                    and student_due
                )
                else None
            )
            target_seconds = time.perf_counter() - target_start

            fake_start = time.perf_counter()
            fake_rows = (
                []
                if model.sfdmd_imaginaire_matching and student_due
                else self._fake_updates(
                    model,
                    rollout,
                    fake_optimizer,
                    fake_scheduler,
                    grad_scaler,
                    cycle,
                )
            )
            fake_seconds = time.perf_counter() - fake_start

            student_start = time.perf_counter()
            student_rows = (
                self._student_update(
                    model,
                    rollout,
                    student_optimizer,
                    student_scheduler,
                    grad_scaler,
                    cycle,
                    target_map=student_targets,
                )
                if student_due
                else []
            )
            student_seconds = time.perf_counter() - student_start

            cycle += 1
            device = rollout.generated_real_latents.device
            output_batch = {
                **kv_gate_metrics,
                **self._mean_metrics(fake_rows),
                **self._mean_metrics(student_rows),
                "sfdmd/cycle": torch.tensor(
                    float(cycle), device=device
                ),
                "sfdmd/num_blocks": torch.tensor(
                    float(len(rollout.records)), device=device
                ),
                "sfdmd/rollout_seconds": torch.tensor(
                    rollout_seconds, device=device
                ),
                "sfdmd/target_seconds": torch.tensor(
                    target_seconds, device=device
                ),
                "sfdmd/fake_seconds": torch.tensor(
                    fake_seconds, device=device
                ),
                "sfdmd/student_seconds": torch.tensor(
                    student_seconds, device=device
                ),
                "sfdmd/cycle_seconds": torch.tensor(
                    time.perf_counter() - cycle_start, device=device
                ),
                "sfdmd/generated_abs": (
                    rollout.generated_real_latents.abs().mean()
                ),
                "sfdmd/exit_step": torch.tensor(
                    float(rollout.exit_step), device=device
                ),
                "sfdmd/student_updated": torch.tensor(
                    float(student_due), device=device
                ),
            }
            if model.sfdmd_factorized_conditioning:
                output_batch["sfdmd/factorized_prefix_k"] = torch.tensor(
                    float(rollout.conditioned_prefix_frames), device=device
                )
                output_batch["sfdmd/factorized_geometry_g"] = torch.tensor(
                    float(rollout.factorized_geometry_observations),
                    device=device,
                )
                for mode in model.sfdmd_factorized_hdmap_modes:
                    output_batch[
                        f"sfdmd/factorized_hdmap_{mode}"
                    ] = torch.tensor(
                        float(mode == rollout.factorized_hdmap_mode),
                        device=device,
                    )
                for mode in model.sfdmd_factorized_geometry_modes:
                    output_batch[
                        f"sfdmd/factorized_geometry_{mode}"
                    ] = torch.tensor(
                        float(mode == rollout.factorized_geometry_mode),
                        device=device,
                    )
            if fake_rows:
                output_batch["loss/sfdmd_fake_score"] = output_batch[
                    "fake_score_loss"
                ]
            if student_rows:
                output_batch["loss/sfdmd_student"] = output_batch[
                    "student_vsd_loss"
                ]
            for row in fake_rows:
                block = int(row["critic_block"].item())
                output_batch[
                    f"loss/sfdmd_fake_score_block_{block:02d}"
                ] = row["fake_score_loss"]
            for row in student_rows:
                block = int(row["student_block"].item())
                output_batch[
                    f"loss/sfdmd_student_block_{block:02d}"
                ] = row["student_vsd_loss"]
                output_batch[
                    f"metric/sfdmd_teacher_fake_x0_l1_block_{block:02d}"
                ] = row["teacher_fake_x0_l1"]
                output_batch[
                    f"metric/sfdmd_teacher_gen_x0_l1_block_{block:02d}"
                ] = row["teacher_gen_x0_l1"]
            loss_for_callbacks = output_batch[
                "student_vsd_loss"
                if student_rows
                else "fake_score_loss"
            ]
            self.callbacks.on_training_step_end(
                model,
                data_batch,
                output_batch,
                loss_for_callbacks,
                iteration=cycle,
            )
            del rollout, data_batch
            gc.collect()
            torch.cuda.empty_cache()

            if self.checkpointer.should_save(cycle):
                self.checkpointer.save_sfdmd(
                    model,
                    student_optimizer,
                    student_scheduler,
                    fake_optimizer,
                    fake_scheduler,
                    grad_scaler,
                    cycle,
                )

        if not self.checkpointer.should_save(cycle):
            self.checkpointer.save_sfdmd(
                model,
                student_optimizer,
                student_scheduler,
                fake_optimizer,
                fake_scheduler,
                grad_scaler,
                cycle,
            )
        self.callbacks.on_train_end(model, iteration=cycle)
        self.checkpointer.finalize()
        distributed.barrier()
        self.callbacks.on_app_end()
        log.success(f"Done with SF-DMD training at cycle {cycle}.")
