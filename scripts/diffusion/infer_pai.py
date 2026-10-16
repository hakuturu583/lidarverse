#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate, forecast, and resimulate with a released NVIDIA PAI EMA."""

import argparse
from contextlib import redirect_stderr, redirect_stdout
import math
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--sample-path", type=Path,
                        help="Prepared NPZ or directory (defaults to sample_001 for a single sample)")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--tokenizer-checkpoint", type=Path, default=Path("checkpoints/lidarverse-tokenizer-pai/ckpt.pt"))
    parser.add_argument("--latent-stats", type=Path, default=Path("checkpoints/lidarverse-tokenizer-pai/latent_stats.pt"))
    parser.add_argument("--output-folder", type=Path, required=True)
    parser.add_argument("--mode", choices=("all", "world", "forecast", "resim"), default="all")
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--num-steps", type=int)
    parser.add_argument("--kv-cache-frames", type=int)
    parser.add_argument("--sampler", choices=("sde", "unipc"), default="sde")
    parser.add_argument("--sde-eta", type=float, default=1.0)
    parser.add_argument("--guidance", type=float, help="CFG scale (default: 3 for Stage 1, 1 for Stages 2/3)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--vis-pcd", type=int, choices=(0, 1), default=1)
    parser.add_argument("--raw-hdmap-condition-rasterize", action=argparse.BooleanOptionalAction, default=True,
                        help="World only: replace cached HDMap ranges with raw map/box rasterization")
    parser.add_argument("--raw-hdmap-condition-bbox-dilation", type=int, default=3,
                        help="Positive odd bbox range-image kernel; 3 widens boxes to 3x3 pixels")
    parser.add_argument("--raw-hdmap-seg-interval", type=float, default=0.1,
                        help="Raw map/box edge sampling spacing in meters")
    args = parser.parse_args(argv)
    if args.raw_hdmap_condition_bbox_dilation < 1 or args.raw_hdmap_condition_bbox_dilation % 2 != 1:
        parser.error("raw HDMap bbox dilation must be a positive odd integer")
    if not math.isfinite(args.raw_hdmap_seg_interval) or args.raw_hdmap_seg_interval <= 0:
        parser.error("raw HDMap segment spacing must be positive and finite")
    if args.mode not in ("world", "all"):
        args.raw_hdmap_condition_rasterize = False
    if args.sample_path is None:
        args.sample_path = Path("data/bidir_quick_start/nvidia_pai")
        if args.num_samples == 1:
            args.sample_path /= "sample_001.npz"
    package = {1: "bidir", 2: "causal", 3: "sfdmd"}[args.stage]
    if args.checkpoint is None:
        args.checkpoint = Path(f"checkpoints/lidarverse-{package}-pai/ema_model.pt")
    if args.num_steps is None:
        args.num_steps = 4 if args.stage == 3 else 35
    if args.guidance is None:
        args.guidance = 3.0 if args.stage == 1 else 1.0
    if args.kv_cache_frames is None and args.stage != 1:
        args.kv_cache_frames = 9 if args.stage == 3 else 21
    if args.stage == 1 and args.kv_cache_frames is not None:
        parser.error("Stage 1 is bidirectional and does not use a rolling K/V cache")
    if args.stage != 1:
        if args.sampler != "sde" or args.guidance != 1:
            parser.error("Causal quickstarts currently support SDE with guidance=1 only")
        if args.kv_cache_frames < 5 or (args.kv_cache_frames - 1) % 4:
            parser.error("The K/V budget must be 1+4N frames (including the sink), at least 5")
    if args.num_samples < 1 or args.num_steps < 1:
        parser.error("samples and steps must be positive")
    if not 0 <= args.sde_eta <= 1 or not math.isfinite(args.guidance) or args.guidance < 0:
        parser.error("eta must be in [0,1] and guidance finite and nonnegative")
    if args.mode != "world" and args.sampler != "sde":
        parser.error("forecast/resim use clean-prefix replacement and require SDE")
    for path in (args.checkpoint, args.tokenizer_checkpoint, args.latent_stats):
        if not path.is_file():
            raise FileNotFoundError(path)
    args.output_folder = args.output_folder.resolve()
    args.output_folder.mkdir(parents=True, exist_ok=True)
    console = sys.stdout
    log_path = args.output_folder / "inference.log"
    print(f"PAI Stage {args.stage} ({package}) | {args.mode} | {args.num_steps} {args.sampler.upper()} steps | CFG {args.guidance:g}", flush=True)
    print(f"Output: {args.output_folder}\nDetails: {log_path}\nLoading runtime and model...", flush=True)
    with log_path.open("w", buffering=1) as details:
        def say(message):
            print(message, file=console, flush=True)
            print(message, file=details, flush=True)
        try:
            with redirect_stdout(details), redirect_stderr(details):
                from diffusion.inference.pai import run
                run(args, console, say)
        except BaseException:
            traceback.print_exc(file=details)
            say(f"Stopped. Details: {log_path}")
            raise


if __name__ == "__main__":
    main()
