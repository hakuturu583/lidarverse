#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${EXPERIMENT_NAME:?Set EXPERIMENT_NAME to a public LidarVerse stage}"
N_GPUS="${N_GPUS:-8}"
MASTER_PORT="${MASTER_PORT:-29502}"
RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
JOB_NAME="${JOB_NAME:-${EXPERIMENT_NAME}_$RUN_TS}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT_DIR/checkpoints}"

export OUTPUT_ROOT OMP_NUM_THREADS=1
cd "$ROOT_DIR"
torchrun --standalone --nproc_per_node="$N_GPUS" --master_port="$MASTER_PORT" \
  -m diffusion.training.train \
  --config=diffusion/training/config/config_lidar.py -- \
  "experiment=$EXPERIMENT_NAME" \
  "job.name=$JOB_NAME" \
  "$@"
