#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-lidarverse_pai_stage2_finetune}"
exec "$ROOT_DIR/scripts/diffusion/train.sh" "$@"
