# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Release artifact checks, called before and after the GPU validation suite."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils.visualize.hdmap import render_hdmap


def check_ranges(path, expected=None):
    with np.load(path, allow_pickle=False) as data:
        ranges, valid = data["range_m"], data["valid_mask"]
        if ranges.shape != valid.shape or ranges.shape[1:] != (128, 1024):
            raise ValueError(f"{path}: invalid range/mask shape")
        if expected is not None and len(ranges) != expected:
            raise ValueError(f"{path}: expected {expected} frames, got {len(ranges)}")
        if not np.isfinite(ranges).all() or not valid.reshape(len(valid), -1).any(axis=1).all():
            raise ValueError(f"{path}: nonfinite or empty sweep")
        return len(ranges)


def check_video(path, frames):
    info = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,nb_read_frames,width,height,pix_fmt,avg_frame_rate",
        "-of", "json", str(path)], text=True))["streams"][0]
    if (info["codec_name"] != "h264" or int(info["nb_read_frames"]) != frames
            or info["pix_fmt"] != "yuv420p" or info["avg_frame_rate"] != "10/1"):
        raise ValueError(f"{path}: invalid video {info}")
    capture = cv2.VideoCapture(str(path))
    count = 0
    try:
        while True:
            ok, image = capture.read()
            if not ok:
                break
            # Exclude the text strip; require content in each panel, not just labels.
            left, right = np.split(image[65:], 2, axis=1)
            if any(np.mean(np.any(panel < 235, axis=2)) < .001 for panel in (left, right)):
                raise ValueError(f"{path}: blank panel at frame {count}")
            count += 1
    finally:
        capture.release()
    if count != frames:
        raise ValueError(f"{path}: only decoded {count}/{frames} frames")
    return info


def inputs(args):
    manifest = json.loads(Path("checkpoints/pai_manifest.json").read_text())
    required_packages = set(args.stages) | {"tokenizer"}
    if "stage3" in required_packages:
        required_packages.add("stage2")
    packages = {key: manifest["packages"][key] for key in sorted(required_packages)}
    for package in packages.values():
        for name, metadata in package["files"].items():
            path = Path("checkpoints") / package["directory"] / name
            print(f"Checking SHA-256: {path}", flush=True)
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != metadata["sha256"]:
                raise ValueError(f"Release checksum mismatch: {path}")
    paths = sorted(args.sample_path.rglob("*.npz"))
    if not paths:
        raise FileNotFoundError(args.sample_path)
    for path in paths:
        if check_ranges(path) < 21:
            raise ValueError(f"{path}: need at least 21 sweeps")
        with np.load(path, allow_pickle=False) as data:
            sample = {key: data[key] for key in data.files}
        required = {"map_vertices", "map_offsets", "map_kinds", "box_class", "box_valid"}
        missing = sorted(required - sample.keys())
        if missing:
            raise ValueError(
                f"{path}: missing release-preview fields: {', '.join(missing)}. "
                "Rebuild the examples with scripts/data/prepare_pai_bidir.py "
                "using the raw-map and actor annotations before rerunning release checks."
            )
        if len(sample["map_kinds"]) == 0 or not sample["box_valid"].any():
            raise ValueError(f"{path}: map/actor visualization would be vacuous")
        image = render_hdmap(sample, 0)
        if np.mean(np.any(image < 235, axis=2)) < .001:
            raise ValueError(f"{path}: raw-map preview is blank")
        preview = args.output_root / "previews" / f"{path.stem}_hdmap.png"
        preview.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(preview), image):
            raise OSError(preview)
    return {"packages": packages, "prepared_samples": len(paths)}


def outputs(args):
    reports = []
    for stage in args.stages:
        cases = ("random",) if stage == "tokenizer" else ("world", "static", "frozen")
        for condition in cases:
            root = args.output_root / "training" / condition / "training_checks/preflight" / stage
            report = json.loads((root / "training_check.json").read_text())
            if report["status"] != "PASS" or report["start_iteration"] != 0 or report["iterations"] != 3:
                raise ValueError(f"Invalid training report: {root}")
            reports.append(report)
            if condition == "world" or stage == "tokenizer":
                resumed = json.loads((root / "training_check_resume.json").read_text())
                if resumed["status"] != "PASS" or resumed["start_iteration"] != 3 or resumed["iterations"] != 7:
                    raise ValueError(f"Invalid resumed training report: {root}")
                reports.append(resumed)
    videos = {}
    if "tokenizer" in args.stages:
        root = args.output_root / "inference/tokenizer"
        frames = check_ranges(root / "reconstruction.npz")
        videos[str(root)] = check_video(root / "reconstruction.mp4", frames)
    for stage_name in args.stages:
        if stage_name == "tokenizer":
            continue
        stage = int(stage_name.removeprefix("stage"))
        for mode in ("world", "forecast", "resim"):
            root = args.output_root / f"inference/stage{stage}" / mode
            paths = list(root.rglob("generation.npz"))
            if len(paths) != 1:
                raise ValueError(f"{root}: expected exactly one generated sample")
            expected = 21 if mode == "forecast" or (mode == "world" and stage != 1) else 20
            check_ranges(paths[0], expected)
            videos[str(paths[0].parent)] = check_video(paths[0].with_suffix(".mp4"), expected)
    return {"training_reports": reports, "videos": videos,
            "scope": "Functional checks only. Inspect previews and evaluate quality separately."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--phase", choices=("inputs", "outputs"), required=True)
    parser.add_argument("--stages", nargs="+", choices=("tokenizer", "stage1", "stage2", "stage3"),
                        default=["tokenizer", "stage1", "stage2", "stage3"])
    args = parser.parse_args()
    result = inputs(args) if args.phase == "inputs" else outputs(args)
    result["status"] = "PASS"
    result["stages"] = args.stages
    path = args.output_root / f"release_{args.phase}.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"PASS: {path}")


if __name__ == "__main__":
    main()
