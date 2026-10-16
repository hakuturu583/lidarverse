#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Download and prepare one official NVIDIA PAI example for the 40k tokenizer."""

import argparse
from io import BytesIO
import json
from pathlib import Path
import tarfile

import numpy as np
import torch
import torch.nn.functional as F
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_REPO = "nvidia/Cosmos-Transfer-LidarGen-Example"
CLIP = "061dbb50-f392-11ee-8e3c-00044baf7cbc_1712352201101085_1712352221101085"


def load_ranges(path, frame_ids):
    clip = path.stem
    frames = []
    with tarfile.open(path, "r") as archive:
        members = {Path(member.name).name: member for member in archive.getmembers() if member.isfile()}
        for frame in frame_ids:
            arrays = []
            for field in ("row", "col", "range"):
                name = f"{clip}.{frame}.lidar_{field}.npz"
                with archive.extractfile(members[name]) as handle:
                    with np.load(BytesIO(handle.read()), allow_pickle=False) as data:
                        arrays.append(data["arr_0"])
            row, col, values = arrays
            dense = np.zeros((128, 3600), dtype=np.float32)
            dense[row.astype(np.int64), col.astype(np.int64)] = values
            # Training: scatter-min over triplets, then nearest-exact 1200 -> 1024.
            groups = dense.reshape(128, 1200, 3)
            reduced = np.where(groups == 0, 1e3, groups).min(axis=-1)
            reduced[reduced == 1e3] = 0
            valid = np.isfinite(reduced) & (reduced >= 2.5) & (reduced <= 100)
            reduced = np.where(valid, reduced, 0).astype(np.float32)
            resized = F.interpolate(
                torch.from_numpy(reduced)[None, None], size=(128, 1024), mode="nearest-exact"
            )[0, 0].numpy()
            frames.append(resized)
    return np.stack(frames)


def sensor_rays():
    with (REPO_ROOT / "assets/lidar/lidar_model.yaml").open() as handle:
        config = yaml.safe_load(handle)
    elevations = config["lidar_models"]["spinning"]["HESAI-Pandar128"]["inclinations_deg"]
    phi, theta = np.meshgrid(
        np.radians(elevations), np.linspace(np.pi, -np.pi, 1024, endpoint=False), indexing="ij"
    )
    return np.stack([
        np.cos(phi) * np.cos(theta), np.cos(phi) * np.sin(theta), np.sin(phi)
    ], axis=-1).astype(np.float32)


def save_sample(path, payload, provenance):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(".npz.tmp").open("wb") as handle:
        np.savez_compressed(handle, **payload)
    path.with_suffix(".npz.tmp").replace(path)
    path.with_suffix(".json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Prepared {path} ({len(payload['range_m'])} frames)", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, help="Existing lidar_dataset_release directory; skips downloads.")
    parser.add_argument("--hdmap-root", type=Path, help="Optional projected HDMap TAR directory.")
    parser.add_argument("--revision", default="main", help="Dataset revision (default: main).")
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.source_root is None:
        from huggingface_hub import hf_hub_download
        download_root = args.output_root / "source"
        for filename in (
            "LICENSE", "README.md",
            f"lidar_dataset_release/lidar/{CLIP}.tar",
            f"lidar_dataset_release/metadata/{CLIP}.npz",
        ):
            hf_hub_download(DATA_REPO, filename, repo_type="dataset", revision=args.revision, local_dir=download_root)
        source = download_root / "lidar_dataset_release"
    else:
        source = args.source_root

    with np.load(source / "metadata" / f"{CLIP}.npz", allow_pickle=False) as metadata:
        frame_ids = metadata["frame_indices"].astype(str)
        timestamps = metadata["timestamps_list"].copy()
        poses = metadata["pose_list"].copy()
    if timestamps.shape != (len(frame_ids), 2) or poses.shape != (len(frame_ids), 2, 4, 4):
        raise ValueError("PAI frame IDs, timestamps, and start/end poses must align")
    if np.any(np.diff(timestamps[:, 0]) <= 0):
        raise ValueError("PAI timestamps must be strictly increasing")
    ranges = load_ranges(source / "lidar" / f"{CLIP}.tar", frame_ids)
    payload = dict(
        range_m=ranges, valid_mask=ranges > 0, ray_directions=sensor_rays(),
        dataset=np.asarray("pandar"), sensor_name=np.asarray("NVIDIA PAI"),
        sensor_dataset_id=np.int64(0), fps=np.float32(10),
        timestamp_us=timestamps[:, 0] - timestamps[0, 0], frame_indices=frame_ids,
        clip_name=np.asarray(CLIP), has_prepend=np.bool_(False),
        min_range_m=np.float32(2.5), max_range_m=np.float32(100),
    )
    provenance = dict(
        source_repo=DATA_REPO, clip=CLIP, visualization_yaw_deg=0,
        processing="scatter_min: 128x3600 -> 128x1200; nearest-exact -> 128x1024; valid range [2.5,100] m",
        rays="Calibrated Pandar128 elevations and uniform final-grid azimuths, matching training",
        motion_compensation=False, source_revision=args.revision,
    )
    save_sample(args.output_root / "examples/nvidia_pai/sample_000.npz", payload, provenance)
    if args.hdmap_root is not None:
        hdmap_path = args.hdmap_root / f"{CLIP}.tar"
        hdmap = load_ranges(hdmap_path, frame_ids)
        condition = np.where(hdmap > 0, 2 * (hdmap - 2.5) / 97.5 - 1, -1).astype(np.float32)
        save_sample(
            args.output_root / "hdmap_examples/nvidia_pai/sample_000.npz",
            {**payload, "hdmap": condition, "pose_sensor_start_end": poses},
            {**provenance, "hdmap_source": str(hdmap_path.resolve())},
        )


if __name__ == "__main__":
    main()
