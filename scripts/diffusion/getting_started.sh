#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

STAGE="${STAGE:-1}"
case "$STAGE" in
  1) PACKAGE=bidir; DEFAULT_STEPS=35; DEFAULT_GUIDANCE=3 ;;
  2) PACKAGE=causal; DEFAULT_STEPS=35; DEFAULT_GUIDANCE=1 ;;
  3) PACKAGE=sfdmd; DEFAULT_STEPS=4; DEFAULT_GUIDANCE=1 ;;
  *) printf 'STAGE must be 1, 2, or 3\n' >&2; exit 1 ;;
esac
CHECKPOINT="${CHECKPOINT:-$PWD/checkpoints/lidarverse-$PACKAGE-pai/ema_model.pt}"
TOKENIZER_CHECKPOINT="${TOKENIZER_CHECKPOINT:-$PWD/checkpoints/lidarverse-tokenizer-pai/ckpt.pt}"
LATENT_STATS="${LATENT_STATS:-$PWD/checkpoints/lidarverse-tokenizer-pai/latent_stats.pt}"
prepare_selection=()
if [[ -z "${SAMPLE_PATH:-}" && "${NUM_SAMPLES:-1}" == 1 ]]; then
  PREPARE_ROOT="$PWD/data/bidir_quick_start/nvidia_pai"
  SAMPLE_PATH="$PREPARE_ROOT/sample_001.npz"
  prepare_selection=(--sample-index 1)
else
  SAMPLE_PATH="${SAMPLE_PATH:-$PWD/data/bidir_quick_start/nvidia_pai}"
  PREPARE_ROOT="$SAMPLE_PATH"
fi
OUTPUT_ROOT="${OUTPUT_ROOT:-$PWD/outputs/pai_${PACKAGE}_$(date +%Y%m%d_%H%M%S)}"
if ! command -v ffmpeg >/dev/null 2>&1; then
  printf 'Install ffmpeg with libx264 support.\n' >&2; exit 1
fi

for path in "$CHECKPOINT" "$(dirname "$CHECKPOINT")/config.json" "$TOKENIZER_CHECKPOINT" "$LATENT_STATS"; do
  if [[ ! -s "$path" ]]; then
    printf 'Missing release weight file: %s\n' "$path" >&2
    exit 1
  fi
done
if [[ "${PREPARE_DATA:-1}" == 1 ]]; then
  prepare_args=(--output-root "$PREPARE_ROOT" --num-samples "${NUM_SAMPLES:-1}" --revision "${DATA_REVISION:-main}" "${prepare_selection[@]}")
  if [[ -n "${PAI_SOURCE_ROOT:-}" ]]; then
    prepare_args+=(--source-root "$PAI_SOURCE_ROOT")
  fi
  python -u scripts/data/prepare_pai_bidir.py "${prepare_args[@]}"
fi
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/lidarverse-matplotlib}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
extra_args=()
if [[ "${RAW_HDMAP_CONDITION_RASTERIZE:-1}" == 1 ]]; then
  extra_args+=(--raw-hdmap-condition-rasterize
    --raw-hdmap-condition-bbox-dilation "${RAW_HDMAP_CONDITION_BBOX_DILATION:-3}"
    --raw-hdmap-seg-interval "${RAW_HDMAP_SEG_INTERVAL:-0.1}")
else
  extra_args+=(--no-raw-hdmap-condition-rasterize)
fi
if [[ -n "${KV_CACHE_FRAMES:-}" ]]; then
  extra_args+=(--kv-cache-frames "$KV_CACHE_FRAMES")
fi
python -u scripts/diffusion/infer_pai.py --stage "$STAGE" \
  --sample-path "$SAMPLE_PATH" --checkpoint "$CHECKPOINT" \
  --tokenizer-checkpoint "$TOKENIZER_CHECKPOINT" --latent-stats "$LATENT_STATS" \
  --output-folder "$OUTPUT_ROOT" --mode "${MODE:-all}" \
  --num-samples "${NUM_SAMPLES:-1}" --num-steps "${NUM_STEPS:-$DEFAULT_STEPS}" \
  --sampler "${SAMPLER:-sde}" --sde-eta "${SDE_ETA:-1}" \
  --guidance "${GUIDANCE:-$DEFAULT_GUIDANCE}" --seed "${SEED:-0}" --vis-pcd "${VIS_PCD:-1}" "${extra_args[@]}"
