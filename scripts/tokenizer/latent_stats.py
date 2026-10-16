#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compute per-channel tokenizer latent statistics on native-grid datasets."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.distributed as dist
from hydra.utils import instantiate
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from diffusion.training.module.vae_utils import (
    build_vae_from_tokenizer_experiment,
)
from tokenizer.training.configs.config import make_config
from utils.config_helper import override


EXPERIMENT = (
    "transformer_vae_8x8x1_z64_cross_dataset3_pandar_pandaset_"
    "nuscenes32_gridrope_mask_pts_scratch_200k"
)


class EqualizedDatasets(Dataset):
    def __init__(self, children, limit: int):
        self.children = list(children)
        if not self.children:
            raise ValueError("Tokenizer experiment has no training datasets")
        unique_counts = [
            int(getattr(child, "n_data_per_node", len(child)))
            for child in self.children
        ]
        self.per_dataset = min(unique_counts)
        if limit > 0:
            self.per_dataset = min(self.per_dataset, limit)

    def __len__(self) -> int:
        return self.per_dataset * len(self.children)

    def __getitem__(self, index: int):
        dataset_index = index % len(self.children)
        local_index = index // len(self.children)
        return self.children[dataset_index][local_index]


def _first_sample(samples):
    return samples[0]


def _distributed() -> tuple[int, int, int]:
    if "RANK" not in os.environ:
        torch.cuda.set_device(0)
        return 0, 1, 0
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def _load_weights(model, checkpoint: Path, use_ema: bool) -> None:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("model", payload)
    regular = {}
    for name, value in state.items():
        if name.startswith("ema."):
            continue
        normalized = name
        for prefix in ("model.network.", "network.", "model.vae.", "vae."):
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):]
                break
        regular[normalized] = value
    model.load_state_dict(regular, strict=False)

    if not use_ema:
        return
    ema = {
        name[len("ema.network-"):].replace("-", "."): value
        for name, value in state.items()
        if name.startswith("ema.network-")
    }
    if not ema:
        raise ValueError(f"No EMA tokenizer weights found in {checkpoint}")
    model.load_state_dict(ema, strict=False)


def _dataset(limit: int) -> EqualizedDatasets:
    config = override(make_config(), ["--", f"experiment={EXPERIMENT}"])
    builder = instantiate(config.dataloader_train.dataset)
    builder.shared_overrides.update(
        {
            "frame_sampling_method": "sequential_from_zero",
            "random_n_frames": False,
            "sample_n_frames": 8,
        }
    )
    multi_dataset = builder.build_dataset()
    return EqualizedDatasets(multi_dataset.children, limit)


def _video(sample) -> torch.Tensor:
    video = sample.get("video", sample.get("images"))
    if video is None:
        raise KeyError("Tokenizer sample contains neither video nor images")
    if video.ndim == 4:
        video = video.unsqueeze(0)
    return video


def _latent_mean(model, video: torch.Tensor) -> torch.Tensor:
    if getattr(model, "mask_as_input", False) and video.shape[1] == getattr(
        model, "out_channels", video.shape[1]
    ):
        threshold = getattr(model, "mask_input_min_value", -1.0) + getattr(
            model, "mask_input_eps", 0.02
        )
        valid = (video[:, :1] > threshold).to(video.dtype)
        video = torch.cat([video, valid], dim=1)
    encoded = model.encode(video)
    if not isinstance(encoded, tuple):
        return encoded
    sample, distribution = encoded[0], encoded[1]
    mean = (
        distribution[0]
        if isinstance(distribution, (tuple, list))
        else getattr(distribution, "mean", sample)
    )
    return mean if torch.is_tensor(mean) and mean.ndim == sample.ndim else sample


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--sequences-per-dataset",
        type=int,
        default=-1,
        help="cap unique sequences per dataset; negative uses every sequence",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--use-ema", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required to encode tokenizer latents")
    rank, world_size, local_rank = _distributed()
    model = build_vae_from_tokenizer_experiment(EXPERIMENT)
    _load_weights(model, args.checkpoint, args.use_ema)
    model = model.cuda(local_rank).eval()

    dataset = _dataset(args.sequences_per_dataset)
    sampler = (
        DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False,
        )
        if world_size > 1
        else None
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=_first_sample,
    )

    total = torch.zeros(64, dtype=torch.float64, device="cuda")
    total_sq = torch.zeros_like(total)
    count = torch.zeros(1, dtype=torch.float64, device="cuda")
    for sample in loader:
        latent = _latent_mean(model, _video(sample).cuda(non_blocking=True))
        latent = latent.double()
        total += latent.sum(dim=(0, 2, 3, 4))
        total_sq += latent.square().sum(dim=(0, 2, 3, 4))
        count += latent.shape[0] * latent.shape[2] * latent.shape[3] * latent.shape[4]

    if world_size > 1:
        for tensor in (total, total_sq, count):
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    mean = total / count
    std = (total_sq / count - mean.square()).clamp_min(0).sqrt()
    if rank == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"mean": mean.cpu().float(), "std": std.cpu().float()}, args.output)
        print(f"Wrote {args.output} from {int(count.item())} latent positions")

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
