#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Run inside the training container; this script never submits cluster jobs.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
N_GPUS="${N_GPUS:-8}"
# Colon-separated for compatibility with submit_job's comma-separated --setenv.
IFS=: read -r -a stages <<< "${STAGES:-tokenizer:stage1:stage2:stage3}"
for stage in "${stages[@]}"; do
  case "$stage" in
    tokenizer|stage1|stage2|stage3) ;;
    *) printf 'Invalid release-check stage: %s\n' "$stage" >&2; exit 1 ;;
  esac
done
OUTPUT_ROOT="${OUTPUT_ROOT:-$PWD/outputs/release_checks_$(date +%Y%m%d_%H%M%S)}"
export LIDARVERSE_PAI_TRAIN_DATA="${LIDARVERSE_PAI_TRAIN_DATA:-$PWD/data/bidir_quick_start/nvidia_pai}"
export LIDARVERSE_PAI_VAL_DATA="${LIDARVERSE_PAI_VAL_DATA:-$LIDARVERSE_PAI_TRAIN_DATA}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS=1 WANDB_MODE=disabled MPLCONFIGDIR=/tmp/lidarverse-matplotlib
if [[ "$N_GPUS" != 8 ]]; then
  printf 'The release FSDP check expects one node with 8 GPUs.\n' >&2; exit 1
fi
if [[ -e "$OUTPUT_ROOT" ]]; then
  printf 'Choose a new OUTPUT_ROOT: %s\n' "$OUTPUT_ROOT" >&2; exit 1
fi
mkdir -p "$OUTPUT_ROOT/logs"
python scripts/validation/check_release.py --sample-path "$LIDARVERSE_PAI_TRAIN_DATA" \
  --output-root "$OUTPUT_ROOT" --phase inputs --stages "${stages[@]}"
for stage in "${stages[@]}"; do
  cases=(world static frozen)
  if [[ "$stage" == tokenizer ]]; then cases=(random); fi
  for condition in "${cases[@]}"; do
    root="$OUTPUT_ROOT/training/$condition"
    torchrun --standalone --nproc_per_node="$N_GPUS" -m scripts.check_training \
      --stage "$stage" --case "$condition" --steps 3 --output-root "$root" \
      2>&1 | tee "$OUTPUT_ROOT/logs/${stage}_${condition}.log"
    if [[ "$condition" == world || "$stage" == tokenizer ]]; then
      torchrun --standalone --nproc_per_node="$N_GPUS" -m scripts.check_training \
        --stage "$stage" --case "$condition" --steps 7 --resume --output-root "$root" \
        2>&1 | tee "$OUTPUT_ROOT/logs/${stage}_${condition}_resume.log"
    fi
  done
done
for stage in "${stages[@]}"; do
if [[ "$stage" == tokenizer ]]; then
sample="$(find "$LIDARVERSE_PAI_TRAIN_DATA" -name '*.npz' -type f | sort | head -n 1)"
CUDA_VISIBLE_DEVICES=0 python scripts/tokenizer/infer.py --sample_path "$sample" \
  --checkpoint checkpoints/lidarverse-tokenizer-pai/ckpt.pt \
  --latent_stats checkpoints/lidarverse-tokenizer-pai/latent_stats.pt \
  --tokenizer-name transformer_lidar_video_tokenizer_8x8x1_z64_pandar_2p5to100_1024_ema40k \
  --encoder-dtype float32 --decoder-dtype float32 --chunk-frames 21 --vis_pcd 1 \
  --output_folder "$OUTPUT_ROOT/inference/tokenizer" 2>&1 | tee "$OUTPUT_ROOT/logs/tokenizer_inference.log"
else
  CUDA_VISIBLE_DEVICES=0 PREPARE_DATA=0 STAGE="${stage#stage}" MODE=all NUM_SAMPLES=1 \
  SAMPLE_PATH="$LIDARVERSE_PAI_TRAIN_DATA" OUTPUT_ROOT="$OUTPUT_ROOT/inference/$stage" \
    bash scripts/diffusion/getting_started.sh 2>&1 | tee "$OUTPUT_ROOT/logs/${stage}_inference.log"
fi
done
python scripts/validation/check_release.py --sample-path "$LIDARVERSE_PAI_TRAIN_DATA" \
  --output-root "$OUTPUT_ROOT" --phase outputs --stages "${stages[@]}"
printf 'Release checks completed: %s\nInspect previews before publication.\n' "$OUTPUT_ROOT"
