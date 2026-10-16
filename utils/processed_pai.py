# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small NPZ training adapter using the same PAI contract as the quickstarts."""

import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler, default_collate

from utils.pai_batch import make_pai_batch


class ProcessedPAI(Dataset):
    def __init__(self, root, frames, tokenizer=False):
        self.paths = sorted(Path(root).rglob("*.npz"))
        if not self.paths:
            raise FileNotFoundError(f"No processed samples under {root}; run prepare_pai_bidir.py first")
        self.frames = frames
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        with np.load(path, allow_pickle=False) as data:
            sample = {key: data[key] for key in data.files}
        ranges = sample["range_m"]
        if ranges.shape[1:] != (128, 1024) or len(ranges) < self.frames:
            raise ValueError(f"{path}: need {self.frames} real 128x1024 sweeps")
        if (str(sample["dataset"]) != "pandar" or int(sample["sensor_dataset_id"]) != 0
                or float(sample["min_range_m"]) != 2.5 or float(sample["max_range_m"]) != 100
                or float(sample["fps"]) != 10 or bool(sample["has_prepend"])):
            raise ValueError(f"{path}: sample does not match the PAI release contract")
        start = int(torch.randint(len(ranges) - self.frames + 1, ()).item())
        temporal = ("range_m", "valid_mask", "hdmap", "pose_sensor_start_end",
                    "box_valid", "box_object_to_world", "box_lwh", "box_track_id", "box_is_moving")
        for key in temporal:
            if key in sample:
                sample[key] = sample[key][start:start + self.frames]
        if self.tokenizer:
            ranges = sample["range_m"]
            valid = sample["valid_mask"] & np.isfinite(ranges) & (ranges >= 2.5) & (ranges <= 100)
            video = np.where(valid, 2 * (ranges - 2.5) / 97.5 - 1, -1).astype(np.float32)
            # Match tokenizer training's repeated first sweep, not the DiT layout.
            video = torch.from_numpy(np.concatenate((video[:1], video)))[None]
            mask = torch.from_numpy(np.concatenate((valid[:1], valid)))[None].float()
            batch = dict(video=video, recon_target=video.clone(), loss_mask=mask,
                         loss_mask_is_validity=True, sensor_dataset_id=torch.tensor(0),
                         sensor_ray_directions=torch.from_numpy(sample["ray_directions"]).float(),
                         sensor_depth_bounds=torch.tensor([2.5, 100.]))
        else:
            batch = make_pai_batch(sample, self.frames)
            batch = {key: value[0] if torch.is_tensor(value) else value for key, value in batch.items()}
        batch.update(clip_name=path.stem, data_type="lidar", __key__=path.stem)
        return batch


def _collate(samples):
    batch = default_collate(samples)
    for key in ("is_preprocessed", "loss_mask_is_validity"):
        if key in batch:
            batch[key] = True
    return batch


def build_loader(root, frames=21, tokenizer=False, validation=False, num_workers=2, batch_size=1):
    dataset = ProcessedPAI(root, frames, tokenizer)
    sampler = DistributedSampler(dataset, shuffle=not validation)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers,
                      pin_memory=True, drop_last=False, collate_fn=_collate)


def loader_config(*, tokenizer=False, validation=False):
    from utils.lazy_config import LazyCall as L

    root = os.environ.get("LIDARVERSE_PAI_VAL_DATA" if validation else "LIDARVERSE_PAI_TRAIN_DATA",
                          "data/bidir_quick_start/nvidia_pai")
    return L(build_loader)(root=root, frames=8 if tokenizer else 21, tokenizer=tokenizer,
                           validation=validation, num_workers=2, batch_size=1)
