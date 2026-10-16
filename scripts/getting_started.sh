#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/lidarverse-matplotlib}"
export HF_XET_CACHE="${HF_XET_CACHE:-/tmp/lidarverse-hf-xet}"
DOWNLOAD_ROOT="${DOWNLOAD_ROOT:-$ROOT_DIR/data/tokenizer_quick_start}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT_DIR/outputs/tokenizer_quick_start_pai40k}"
CHECKPOINT="${TOKENIZER_CHECKPOINT:-$ROOT_DIR/checkpoints/lidarverse-tokenizer-pai/ckpt.pt}"
LATENT_STATS="${TOKENIZER_LATENT_STATS:-$ROOT_DIR/checkpoints/lidarverse-tokenizer-pai/latent_stats.pt}"
if [[ ! -s "$CHECKPOINT" || ! -s "$LATENT_STATS" ]]; then
  printf 'Set TOKENIZER_CHECKPOINT and TOKENIZER_LATENT_STATS to the PAI 40k release files.\n' >&2
  exit 1
fi
if ! command -v ffmpeg >/dev/null 2>&1; then
  printf 'Install system ffmpeg with libx264 support before running the quickstart.\n' >&2
  exit 1
fi
prepare_args=(--output-root "$DOWNLOAD_ROOT" --revision "${DATA_REVISION:-main}")
if [[ -n "${PAI_SOURCE_ROOT:-}" ]]; then
  prepare_args+=(--source-root "$PAI_SOURCE_ROOT")
fi
if [[ -n "${HDMAP_ROOT:-}" ]]; then
  prepare_args+=(--hdmap-root "$HDMAP_ROOT")
fi
python -u scripts/data/prepare_pai_example.py "${prepare_args[@]}"

python -u scripts/tokenizer/infer.py \
  --sample_path "$DOWNLOAD_ROOT/examples/nvidia_pai/sample_000.npz" \
  --checkpoint "$CHECKPOINT" --latent_stats "$LATENT_STATS" \
  --tokenizer-name transformer_lidar_video_tokenizer_8x8x1_z64_pandar_2p5to100_1024_ema40k \
  --encoder-dtype float32 --decoder-dtype float32 \
  --output_folder "$OUTPUT_ROOT" --chunk-frames "${CHUNK_FRAMES:-21}" --vis_pcd 1
printf 'Video: %s/reconstruction.mp4\n' "$OUTPUT_ROOT"
