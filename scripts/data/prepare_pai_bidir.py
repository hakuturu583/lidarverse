#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare official NVIDIA clips, projected conditions, and vector-map previews."""

import argparse
import json
from pathlib import Path
import tarfile
import sys

import numpy as np
import torch

from prepare_pai_example import DATA_REPO, load_ranges, save_sample, sensor_rays

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils.visualize.hdmap import MAP_KINDS, load_vectors, object_class

CLIPS = (
    "061dbb50-f392-11ee-8e3c-00044baf7cbc_1712352201101085_1712352221101085",
    "0679e7b4-7d50-11ec-817d-00044bf65f0e_1643056597567842_1643056617567842",
    "0967af22-fbeb-11ed-9d8b-00044bf65aca_1685123441659000_1685123461659000",
    "14d2ed36-1afa-11ed-88ea-00044bf65d5c_1660389768178294_1660389788178294",
    "16d80b32-3b43-11ef-acfb-00044baf7c0d_1720236057471397_1720236077471397",
)


def load_boxes(path, frame_ids):
    with tarfile.open(path, "r") as archive:
        members = {Path(m.name).name: m for m in archive.getmembers() if m.isfile()}
        frames = []
        for frame in frame_ids:
            name = f"{path.stem}.{frame}.all_object_info.json"
            with archive.extractfile(members[name]) as handle:
                frames.append(json.load(handle))
    names = sorted({name for frame in frames for name in frame},
                   key=lambda name: (0, int(name)) if name.isdigit() else (1, name))
    ids = {name: index for index, name in enumerate(names)}
    count = max(1, max(map(len, frames)))
    poses = np.zeros((len(frames), count, 4, 4), dtype=np.float32)
    sizes = np.zeros((len(frames), count, 3), dtype=np.float32)
    valid = np.zeros((len(frames), count), dtype=bool)
    moving = np.zeros_like(valid)
    tracks = np.full(valid.shape, -1, dtype=np.int64)
    classes = np.full(valid.shape, 4, dtype=np.int64)
    for t, frame in enumerate(frames):
        for index, (name, actor) in enumerate(frame.items()):
            poses[t, index] = actor["object_to_world"]
            sizes[t, index] = actor["object_lwh"]
            moving[t, index] = actor["object_is_moving"]
            tracks[t, index] = ids[name]
            classes[t, index] = object_class(actor.get("object_type", "Others"))
            valid[t, index] = True
    return dict(box_object_to_world=poses, box_lwh=sizes, box_valid=valid,
                box_is_moving=moving, box_track_id=tracks, box_class=classes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path,
                        help="Directory containing official-compatible lidar/ and metadata/ archives.")
    parser.add_argument("--hdmap-root", type=Path)
    parser.add_argument("--object-root", type=Path)
    parser.add_argument("--raw-map-root", type=Path)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, choices=range(1, 6), default=1)
    parser.add_argument("--sample-index", type=int, choices=range(len(CLIPS)),
                        help="Prepare one clip by its zero-based index, retaining its filename.")
    args = parser.parse_args()
    if args.sample_index is not None and args.num_samples != 1:
        parser.error("--sample-index requires --num-samples 1")
    selected = ([(args.sample_index, CLIPS[args.sample_index])] if args.sample_index is not None
                else list(enumerate(CLIPS[:args.num_samples])))
    torch.set_num_threads(4)
    if args.source_root is None:
        from huggingface_hub import hf_hub_download
        root = args.output_root.parent / "source"
        files = ["LICENSE", "README.md", "lidar_dataset_release/HDMAP_README.md",
                 "lidar_dataset_release/hdmap_manifest.json", "lidar_dataset_release/calibration/pandar128.json"]
        for _, clip in selected:
            files.extend(f"lidar_dataset_release/{kind}/{clip}.{suffix}"
                         for kind, suffix in (("lidar", "tar"), ("metadata", "npz"),
                                              ("hdmap", "tar"), ("all_object_info", "tar")))
            files.extend(f"lidar_dataset_release/hdmap_raw/3d_{kind}/{clip}.tar" for kind in MAP_KINDS)
        for name in files:
            hf_hub_download(DATA_REPO, name, repo_type="dataset", revision=args.revision, local_dir=root)
        args.source_root = root / "lidar_dataset_release"
    args.hdmap_root = args.hdmap_root or args.source_root / "hdmap"
    args.object_root = args.object_root or args.source_root / "all_object_info"
    args.raw_map_root = args.raw_map_root or args.source_root / "hdmap_raw"
    for _, clip in selected:
        for path in (args.source_root / "lidar" / f"{clip}.tar",
                     args.source_root / "metadata" / f"{clip}.npz",
                     args.hdmap_root / f"{clip}.tar", args.object_root / f"{clip}.tar"):
            if not path.is_file():
                raise FileNotFoundError(path)
    for index, clip in selected:
        with np.load(args.source_root / "metadata" / f"{clip}.npz", allow_pickle=False) as meta:
            frame_ids = meta["frame_indices"][:21].astype(str)
            poses = meta["pose_list"][:21].copy()
            times = meta["timestamps_list"][:21, 0]
        if len(frame_ids) != 21 or poses.shape != (21, 2, 4, 4):
            raise ValueError(f"{clip}: need 21 real sweeps and start/end poses")
        ranges = load_ranges(args.source_root / "lidar" / f"{clip}.tar", frame_ids)
        hdmap = load_ranges(args.hdmap_root / f"{clip}.tar", frame_ids)
        boxes = load_boxes(args.object_root / f"{clip}.tar", frame_ids)
        vectors = load_vectors(args.raw_map_root, clip)
        save_sample(args.output_root / f"sample_{index:03d}.npz", dict(
            range_m=ranges, valid_mask=ranges > 0,
            hdmap=np.where(hdmap > 0, 2 * (hdmap - 2.5) / 97.5 - 1, -1).astype(np.float32),
            ray_directions=sensor_rays(), pose_sensor_start_end=poses,
            frame_indices=frame_ids, timestamp_us=times - times[0],
            dataset=np.asarray("pandar"), sensor_dataset_id=np.int64(0),
            fps=np.float32(10), min_range_m=np.float32(2.5), max_range_m=np.float32(100),
            has_prepend=np.bool_(False), clip_name=np.asarray(clip), **boxes, **vectors,
        ), dict(source_repo=DATA_REPO, clip=clip, visualization_yaw_deg=0,
                source_revision=args.revision, visualization="glowing_edge_hdmap",
                processing="PAI 128x3600 -> scatter-min 128x1200 -> nearest-exact 128x1024",
                input_has_prepend=False, min_range_m=2.5, max_range_m=100))
    print(f"Prepared {args.num_samples} clips; conditioning and visualization share the released frame IDs.")


if __name__ == "__main__":
    main()
