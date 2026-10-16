# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Short runs of the real training loop with fail-fast numerical checks.

Use the training environment and processed PAI examples in docs/finetuning.md.
Release weights initialize new optimizers; --resume checks the saved trainer state.
"""

import argparse
import importlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist

from utils.callback import Callback
from utils.config_helper import override
from utils.lazy_config import LazyDict

EXPERIMENTS = {
    "tokenizer": "lidarverse_tokenizer_pai_finetune",
    "stage1": "lidarverse_pai_stage1_finetune",
    "stage2": "lidarverse_pai_stage2_finetune",
    "stage3": "lidarverse_pai_stage3_finetune",
}


def _any_rank(value):
    flag = torch.tensor(int(value), device="cuda")
    dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    return bool(flag.item())


class BackwardCheck(Callback):
    """Inspect gradients before clipping/sanitization and track real updates."""

    def __init__(self, resume=False):
        super().__init__()
        self.resume = resume
        self.start_iteration = 0
        self.before = {}
        self.changed = set()
        self.gradients = set()
        self.losses = []
        self.batches = []

    def on_train_start(self, model, iteration=0):
        self.start_iteration = iteration
        if (self.resume and iteration <= 0) or (not self.resume and iteration != 0):
            raise RuntimeError(f"Unexpected initial iteration {iteration}; resume={self.resume}")
        model = getattr(model, "module", model)
        if hasattr(model, "teacher_score") and any(p.requires_grad for p in model.teacher_score.parameters()):
            raise RuntimeError("The real-data score must remain frozen")

    @staticmethod
    def _groups(model):
        model = getattr(model, "module", model)
        if hasattr(model, "network"):
            return {"tokenizer": model.network}
        groups = {"student": model.model}
        if hasattr(model, "fake_score"):
            groups["fake_score"] = model.fake_score
        return groups

    def on_training_step_start(self, model, data, iteration=0):
        for key in ("video", "recon_target", "lidar_hdmap", "sensor_ray_directions"):
            value = data.get(key)
            if torch.is_tensor(value) and _any_rank(not bool(torch.isfinite(value).all())):
                raise FloatingPointError(f"Non-finite training input: {key}")
        video = data.get("video")
        sensor_id = data.get("sensor_dataset_id")
        batch = {
            "iteration": iteration,
            "video_shape": list(video.shape) if video is not None else None,
            "sensor_id": sensor_id.detach().cpu().tolist()
                         if torch.is_tensor(sensor_id) else sensor_id,
        }
        self.batches.append(batch)
        if dist.get_rank() == 0:
            print(f"[training-check] batch={batch}", flush=True)
        # Clone only local trainable shards, outside forward unsharding.
        self.before = {
            name: [p.detach().clone() for p in group.parameters() if p.requires_grad]
            for name, group in self._groups(model).items()
        }

    def on_before_backward(self, model_ddp, loss, iteration=0):
        if _any_rank(not bool(torch.isfinite(loss.detach()).all())):
            raise FloatingPointError(f"Non-finite loss at iteration {iteration}")
        if not loss.requires_grad:
            raise RuntimeError("Loss is detached from the training graph")
        value = float(loss.detach())
        self.losses.append(value)
        if dist.get_rank() == 0:
            print(f"[training-check] iteration={iteration} loss={value:.8g}", flush=True)

    def on_after_backward(self, model_ddp, iteration=0):
        for name, group in self._groups(model_ddp).items():
            gradients = [p.grad for p in group.parameters() if p.grad is not None]
            bad = any(not bool(torch.isfinite(g).all()) for g in gradients)
            if _any_rank(bad):
                raise FloatingPointError(f"{name}: non-finite gradient at {iteration}")
            nonzero = any(bool(torch.count_nonzero(g)) for g in gradients)
            if _any_rank(nonzero):
                self.gradients.add(name)

    def on_training_step_end(self, model, data, output_batch, loss, iteration=0):
        for name, group in self._groups(model).items():
            params = [p for p in group.parameters() if p.requires_grad]
            old = self.before[name]
            if len(params) != len(old):
                raise RuntimeError(f"{name}: trainable parameter layout changed")
            if _any_rank(any(not bool(torch.isfinite(p).all()) for p in params)):
                raise FloatingPointError(f"{name}: optimizer produced non-finite weights")
            changed = any(not torch.equal(p.detach(), saved) for p, saved in zip(params, old))
            if _any_rank(changed):
                self.changed.add(name)
        self.before.clear()

    def on_train_end(self, model, iteration=0):
        expected = set(self._groups(model))
        missing_grad = expected - self.gradients
        missing_update = expected - self.changed
        if missing_grad or missing_update:
            raise RuntimeError(
                f"Missing nonzero gradients: {sorted(missing_grad)}; "
                f"missing parameter updates: {sorted(missing_update)}"
            )
        if dist.get_rank() == 0:
            report = {
                "status": "PASS",
                "iterations": iteration,
                "start_iteration": self.start_iteration,
                "resumed_training_state": self.resume,
                "backward_calls_rank0": len(self.losses),
                "losses_rank0": self.losses,
                "batches_rank0": self.batches,
                "nonzero_gradients": sorted(self.gradients),
                "updated_networks": sorted(self.changed),
                "scope": "training smoke check; not a convergence or accuracy evaluation",
            }
            path = Path(self.config.job.path_local) / ("training_check_resume.json" if self.resume else "training_check.json")
            path.write_text(json.dumps(report, indent=2) + "\n")
            print(f"[training-check] PASS: {path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=EXPERIMENTS, required=True)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--resume", action="store_true", help="Resume full state from this check's output directory")
    parser.add_argument("--case", choices=("random", "world", "static", "frozen"), default="random")
    parser.add_argument("overrides", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.steps < 2:
        parser.error("Use at least two steps to exercise the SF-DMD student update")
    root = args.output_root.resolve()
    os.environ["OUTPUT_ROOT"] = str(root)
    os.environ["WANDB_MODE"] = "disabled"
    is_tokenizer = args.stage == "tokenizer"
    module_name = (
        "tokenizer.training.configs.config" if is_tokenizer
        else "diffusion.training.config.config_lidar"
    )
    config = importlib.import_module(module_name).make_config()
    extra = args.overrides[1:] if args.overrides[:1] == ["--"] else args.overrides
    condition = []
    if not is_tokenizer and args.case != "random":
        prefix, hdmap, geometry, observations = {
            "world": (0, "full", "null", -1),
            "static": (1, "full", "rolling_static", -1),
            "frozen": (4, "null", "frozen_object", 4),
        }[args.case]
        # Hydra otherwise interprets the mode name null as Python None.
        condition = [f"model.stage2_factorized_force_prefix_k={prefix}",
                     f"model.stage2_factorized_force_hdmap_mode={json.dumps(hdmap)}",
                     f"model.stage2_factorized_force_geometry_mode={json.dumps(geometry)}",
                     f"model.factorized_force_geometry_observations={observations}"]
    config = override(config, [
        "--",
        f"experiment={EXPERIMENTS[args.stage]}", *extra, *condition,
        "job.project=training_checks", "job.group=preflight",
        f"job.name={args.stage}", "job.wandb_mode=disabled",
        f"trainer.max_iter={args.steps}", "trainer.logging_iter=1",
        "trainer.run_validation=False", "trainer.grad_accum_iter=1",
        f"checkpoint.load_training_state={args.resume}", "checkpoint.strict_resume=True",
        f"checkpoint.save_iter={args.steps}",
    ])
    # Every rank checks before trainer initialization, so failures cannot strand peers.
    exists = Path(config.job.path_local).exists()
    if exists and not args.resume:
        raise FileExistsError(f"Choose a fresh --output-root: {config.job.path_local}")
    if args.resume and not exists:
        raise FileNotFoundError(f"No training run to resume: {config.job.path_local}")
    if args.stage == "stage3":
        student = config.model.net
        score = config.model.sfdmd_teacher_net
        keys = ("patch_spatial", "in_channels", "out_channels", "num_dataset_embeddings")
        mismatches = [f"{key}: student={student.get(key)}, score={score.get(key)}"
                      for key in keys if student.get(key) != score.get(key)]
        if mismatches:
            raise ValueError("SF-DMD architecture mismatch: " + "; ".join(mismatches))
    callbacks = config.trainer.callbacks or {}
    config.trainer.callbacks = LazyDict({
        "training_check": LazyDict({"_target_": "scripts.check_training.BackwardCheck", "resume": args.resume}),
        **{k: v for k, v in callbacks.items() if k != "wandbLidar"},
    })
    config.validate()
    train_module = importlib.import_module(
        "tokenizer.training.train" if is_tokenizer else "diffusion.training.train"
    )
    try:
        train_module.launch(config, SimpleNamespace(mp0_only_dl=False))
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
