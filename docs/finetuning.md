<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Fine-Tuning

These PAI recipes strictly load the released tokenizer, bidirectional,
causal, and SF-DMD EMA weights. Run from the repository root
in the [training environment](setup.md#runtime-dependencies).

## Data

Prepare the [quickstart examples](setup.md#bidirectional-examples) for a plumbing
check. For meaningful fine-tuning, supply your own processed PAI sequences:

```bash
export LIDARVERSE_PAI_TRAIN_DATA=/path/to/processed/train
export LIDARVERSE_PAI_VAL_DATA=/path/to/processed/val
```

The NPZ contract is defined by `scripts/data/prepare_pai_bidir.py`: metric range,
validity, calibrated rays, start/end sensor poses, projected HDMap, and tracked
actor boxes. Native inputs are 128x1024, 2.5--100 m, 10 Hz. The adapter samples
contiguous windows; the small public examples are not a training benchmark.

## Training

Each command starts a new fine-tuning run with fresh optimizer state:

```bash
N_GPUS=8 bash scripts/tokenizer/train.sh
N_GPUS=8 bash scripts/diffusion/train_stage1.sh
N_GPUS=8 bash scripts/diffusion/train_stage2.sh
N_GPUS=8 bash scripts/diffusion/train_stage3.sh
```

Tokenizer training uses eight real frames plus a repeated first frame, FP32,
and initializes its training EMA from the exported network. The DiT recipes
use 21 real frames, P1x2 patches, three dataset-embedding rows (PAI ID 0), and
BF16. SF-DMD uses a 13-frame scoring window and KV9 including the sink.
Its tokenizer encoder/decoder use FP32 and its spatial cache uses 0.10 m voxels.
Training uses the supplied cached HDMap conditions; raw-vector rasterization
and box thickening are inference-only quickstart defaults.
Stage 2 uses FlexAttention with full-block activation checkpointing.
Stage 3 initializes its frozen real score and new trainable fake score from
the released Stage-2 EMA. This is **not** an exact continuation of the original
SF-DMD run, whose optimizer and fake-score weights are not released.

Override settings at the end of a command, for example
`optimizer.lr=1e-5 trainer.max_iter=2000`. To initialize Stage 2 from your new
Stage 1, set `LIDARVERSE_STAGE2_CHECKPOINT=/path/to/stage1_ema_model.pt`.
For new distillation, set both `LIDARVERSE_STAGE3_CHECKPOINT` and
`LIDARVERSE_SCORE_CHECKPOINT` to your Stage-2 EMA.

Keep the tokenizer and its latent statistics paired. Changing the tokenizer
requires new statistics and diffusion adaptation. Full-state resume requires
a training checkpoint and `checkpoint.load_training_state=True`, not a release
EMA file. Reuse the same `JOB_NAME` and `OUTPUT_ROOT` to resume a saved run.

## Release Check

After downloading the checkpoint packages and preparing the examples, on one eight-GPU node:

```bash
N_GPUS=8 OUTPUT_ROOT="$PWD/outputs/release_checks_$(date +%Y%m%d_%H%M%S)" \
bash scripts/validation/release_checks.sh
```

This checks package hashes, raw-map previews, strict checkpoint loading,
finite losses/gradients, actual weight updates, and save/resume for all four
models. Diffusion checks force null, rolling-static, and frozen-object geometry.
It then runs tokenizer reconstruction and all three inference modes per stage,
checking every video frame. Allow several hundred GB for training checkpoints.
To check only selected stages, set `STAGES=stage2:stage3` or `STAGES=stage3`.
The default checks all four models. `release_outputs.json` is written only
after all selected checks pass and records the stages covered. Checks use a
fresh output directory; the save/resume check runs within that directory.
