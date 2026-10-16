#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate native-grid LiDAR from HDMap NPZs with the three-sensor P1x1 model."""

from __future__ import annotations

import argparse
import math
import sys
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout
from datetime import timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-path", "--sample_path", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="EMA .pt file, or training run whose latest complete EMA is pinned.")
    parser.add_argument("--tokenizer-checkpoint", "--tokenizer_checkpoint", type=Path,
                        default=Path("checkpoints/lidarverse-tokenizer/ckpt.pt"))
    parser.add_argument("--latent-stats", "--latent_stats", type=Path,
                        default=Path("checkpoints/lidarverse-tokenizer/latent_stats.pt"))
    parser.add_argument("--output-folder", "--output_folder", type=Path, required=True)
    parser.add_argument("--sample-n-frames", "--sample_n_frames", type=int, default=20)
    parser.add_argument("--num-samples", "--num_samples", type=int, default=5, help="Limit per sensor.")
    parser.add_argument("--num-steps", "--num_steps", type=int, default=35)
    parser.add_argument("--sampler", choices=("sde", "unipc"), default="sde")
    parser.add_argument("--sde-eta", "--sde_eta", type=float, default=1.0)
    parser.add_argument("--guidance", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--vis-pcd", "--vis_pcd", type=int, choices=(0, 1), default=1)
    args = parser.parse_args()
    if not 1 <= args.sample_n_frames <= 32 or args.num_samples < 1 or args.num_steps < 1:
        parser.error("frames must be in [1,32]; samples and steps must be positive")
    if not 0 <= args.sde_eta <= 1 or not math.isfinite(args.guidance) or args.guidance < 0:
        parser.error("sde_eta must be in [0,1] and guidance must be finite and nonnegative")
    for required in (args.checkpoint, args.tokenizer_checkpoint, args.latent_stats):
        if not required.exists():
            raise FileNotFoundError(required)
    args.output_folder = args.output_folder.resolve()
    args.output_folder.mkdir(parents=True, exist_ok=True)
    console, started = sys.stdout, time.perf_counter()
    log_path = args.output_folder / "inference.log"
    print(f"HDMap generation | {args.num_steps} {args.sampler.upper()} steps | CFG {args.guidance:g}", file=console, flush=True)
    print(f"Output: {args.output_folder}\nDetails: {log_path}\nLoading runtime and model...", file=console, flush=True)
    with log_path.open("w", buffering=1) as details:
        def say(message):
            print(message, file=console, flush=True)
            print(message, file=details, flush=True)

        try:
            # Capture import-time warnings too. Only concise status/progress is
            # sent to the original console; exceptions remain visible there.
            with redirect_stdout(details), redirect_stderr(details):
                from diffusion.inference.hdmap import run

                run(args, console, say, started)
        except BaseException:
            traceback.print_exc(file=details)
            say(f"Stopped after {timedelta(seconds=round(time.perf_counter() - started))}. Details: {log_path}")
            raise


if __name__ == "__main__":
    main()
