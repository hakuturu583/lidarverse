<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Environment and Data

## Runtime Dependencies

`uv sync` (see the [README](../README.md#dependencies)) installs everything
training and diffusion inference import, Megatron Core included, from wheels.
Three CUDA extensions are optional:

| extension | used for | without it (`utils/compat.py`) |
|---|---|---|
| Transformer Engine | RMSNorm and RoPE in the DiT | PyTorch RMSNorm (FP32 statistics) and RoPE; checkpoints written with TE load as is |
| NATTEN | the tokenizer's neighborhood attention | the same neighborhoods (clamped windows, dilation) gathered in PyTorch |
| Apex | `FusedAdam`, multi-tensor gradient clipping | `torch.optim.AdamW(fused=True)`, `torch` norms |

The released checkpoints use the `torch-flex` (causal) and `minimal_a2a`
(bidirectional) attention backends, which need none of them. Set
`LIDARVERSE_DISABLE_EXTENSIONS=te,natten,apex` (or `all`) to force the PyTorch
paths, e.g. with an extension build that has no kernels for the GPU.

### Blackwell (RTX PRO 6000, RTX 50 series)

These are sm_120 GPUs: they need PyTorch built for CUDA 12.8 or newer (the
locked `torch==2.8.0` is) and have about 100 KB of shared memory per block,
less than the Hopper tiles PyTorch's FlexAttention picks for every GPU of
compute capability 9.0 or higher. `utils.compat.flex_attention_kernel_options`
selects 64x64 tiles on such GPUs (sm_86, sm_89, sm_120); A100, H100 and B200 keep
PyTorch's defaults. Cached inference uses PyTorch's FlashAttention, which
supports sm_120. With 96 GB the RTX PRO 6000 runs every quickstart stage.

### Optional extensions

Transformer Engine builds against the installed PyTorch (CUDA toolkit with
`nvcc` and cuDNN headers on the machine):

```bash
uv sync --extra te
```

NATTEN publishes wheels per PyTorch/CUDA pair at
[whl.natten.org](https://whl.natten.org); Apex is built from source
(`APEX_CPP_EXT=1 APEX_CUDA_EXT=1 uv pip install --no-build-isolation
'apex @ git+https://github.com/NVIDIA/apex.git'`). Install them into `.venv`
with `uv pip`; a later `uv sync` removes packages that are not in the lock
(use `uv sync --inexact` to keep them).

### Original environment

The training container used CUDA 12.6, PyTorch 2.7 and these pins
(`requirements.txt` plus):

```bash
python -m pip install \
  setuptools==76.0.0 wheel==0.45.1 packaging==24.2 \
  cmake==3.30.5 ninja==1.11.1.1 pybind11==2.13.6 \
  pydantic==2.10.6 importlib-metadata==8.6.1 psutil==6.0.0 \
  tensorstore==0.1.45

NVTE_FRAMEWORK=pytorch APEX_CPP_EXT=1 APEX_CUDA_EXT=1 \
MAX_JOBS=4 NVTE_BUILD_MAX_JOBS=4 \
python -m pip install --no-build-isolation --no-deps \
  megatron-core==0.10.0 flash-attn==2.4.2 \
  'transformer-engine @ git+https://github.com/NVIDIA/TransformerEngine.git@4df84889cb5743113b0fa59839b941486df16ace' \
  'apex @ git+https://github.com/NVIDIA/apex.git@8a6508aaad6e75a2b939e33f308cd63d745d97f1'
```

Transformer Engine is pinned to the revision recorded in the training container.
Apex uses an explicit upstream build pin: the container recorded `apex==0.1`
without a source revision, so this is not an exact reproduction of that commit.
Install NVIDIA Apex from the pinned Git source, not the unrelated PyPI `apex` package.
CUDA 12.6 has no Blackwell (sm_120) kernels; use the `uv sync` environment there.
System `ffmpeg` and `ffprobe` are also required; a Python package is not a substitute.

## Legacy Dataset Adapters

The release fine-tuning recipes use [processed PAI samples](finetuning.md#data).
The adapters below are retained for development; their older cross-dataset
recipes do not match the released PAI DiT without architecture changes.

Place prepared training data at these default paths. Split lists contain one
sequence name per line; no datasets or splits are distributed with the code.

| Dataset | Root | Train / validation lists |
|---|---|---|
| NVIDIA PAI | `data/nvidia_pai/` | `data/splits/nvidia_pai_{train,val}.lst` |
| PandaSet | `data/pandaset_rangemap_64x1024/` | `data/splits/pandaset_{train,val}.lst` |
| nuScenes | `data/nuscenes_ncore/` | `data/splits/nuscenes_{train,val}.lst` |

PAI uses `lidar/` and `metadata/`, plus `hdmap/` and `all_object_info/` for
diffusion training. nuScenes HDMap caches go in `data/nuscenes_hdmap/`.
PandaSet is used only by the tokenizer recipe and requires calibrated v6 caches
(`pandaset_pandar64_64x1024_v6`). Existing v6 caches can be reused; older caches
must be rebuilt from paired raw and processed releases using the tool below.

Customize paths with the `LIDARVERSE_*` variables defined in the
[tokenizer](../tokenizer/training/datasets/lidar_datasets/configs.py) and
[diffusion](../diffusion/training/datasets/lidar_dataset/configs.py) dataset configs.
The cross-dataset grids are PAI 128x1200, PandaSet 64x1024, and nuScenes 32x1024;
the PAI-only quickstart instead uses 128x1024.

Optional preparation tools:

- [Split lists](../scripts/data/build_split_lists.py)
- [PandaSet range-map cache](../scripts/data/cache_pandaset_rangemaps.py)
- [nuScenes HDMaps](../scripts/data/dump_nuscenes_hdmap.py)
- [Waymo HDMaps](../scripts/data/dump_waymo_hdmap.py) and the [NCore conversion guide](https://nvidia.github.io/ncore/conversions/waymo/waymo.html)

Use each script's `--help` for arguments. NCore loading needs `nvidia-ncore`
(training snapshot: 19.5.0). Preprocessing needs `pandas` for PandaSet,
`nuscenes-devkit` and `pyquaternion` for nuScenes, or a compatible
TensorFlow/Waymo Open Dataset pair for Waymo. These are not quickstart dependencies.

## Bidirectional Examples

Download and prepare the dynamic clip used by the Stage-1 quickstart:

```bash
python scripts/data/prepare_pai_bidir.py \
  --output-root data/bidir_quick_start/nvidia_pai \
  --num-samples 1 --sample-index 1
```

HDMaps, actor boxes, and raw vector maps are available on the default `main`
revision. Accept the dataset terms and authenticate with Hugging Face before downloading.
For a local download, add `--source-root /path/to/lidar_dataset_release`.
That root contains `lidar/`, `metadata/`, `hdmap/`, `all_object_info/`, and
`hdmap_raw/3d_<kind>/`. Raw vector maps and boxes are joined by frame ID,
not by timestamps with different origins. The preparer retains the source
archives and writes numeric NPZs without pickle. Sample licenses still apply.

These same 21-sweep examples work with all three diffusion packages.
This writes `sample_001.npz` without renumbering it. Remove `--sample-index 1`
and set `--num-samples 5` to prepare all five clips instead.
Raw maps provide thin map lines and glowing actor-box edges; the DiT still receives the
original projected range condition. Input PLYs contain that projected condition,
not the decorative vector-map edges.

| Mode | Conditioning | Input / generated frames |
|---|---|---|
| `world` | HDMap; causal stages also accumulate generated geometry | 0 / 20 (Stage 1), 0 / 21 (Stage 2/3) |
| `forecast` | Observed LiDAR; frozen static geometry (Stage 1), rolling cache (Stage 2/3) | 10 / 11 |
| `resim` | Observed LiDAR, frozen canonical-object geometry, target actor poses | 15 / 5 |

Causal forecasting prefills the cache with the ten input sweeps, then updates it
with generated sweeps only. Resimulation keeps the observed geometry frozen;
future actor poses are provided controls, not future LiDAR observations. The
20-frame resimulation output is sampled as 21 frames to complete the last
four-frame block, then cropped. Only poses/annotations for the padded frame are
used, never its GT LiDAR. These short demos replay the supplied sensor path.

Run the [diffusion quickstart](../README.md#stage-1-bidirectional-generation)
after preparing these files. `STAGE=1`, `2`, or `3` selects the package. The old
`scripts/diffusion/infer_bidir.py` entry point remains a Stage-1 alias.
