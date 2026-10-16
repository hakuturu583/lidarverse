# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strict EMA warm-start of the released PAI tokenizer."""

from copy import deepcopy
from collections.abc import Mapping
import os
from hydra.core.config_store import ConfigStore

from tokenizer.training.configs.experiments.transformer_vae_8x8x1_z64_cross_dataset3_pandar_pandaset_nuscenes32_gridrope_mask_pts_scratch_200k import LIDARVERSE_TOKENIZER_EXPERIMENT
from utils.processed_pai import loader_config

config = deepcopy(LIDARVERSE_TOKENIZER_EXPERIMENT)
config.defaults = [item for item in config.defaults
                   if not isinstance(item, Mapping) or not any(k in item for k in ("override /data_train", "override /data_val"))]
# Override the registered groups so no legacy loader arguments survive a merge.
config.defaults.insert(0, {"override /data_train": "pai_release"})
config.defaults.insert(1, {"override /data_val": "pai_release"})
del config["dataloader_train"]
del config["dataloader_val"]
config.job.name = "lidarverse_tokenizer_pai_finetune"
config.checkpoint.update(load_path=os.environ.get("LIDARVERSE_TOKENIZER_CHECKPOINT",
    "checkpoints/lidarverse-tokenizer-pai/ckpt.pt"), load_training_state=False, strict_resume=True, save_iter=200)
config.trainer.update(max_iter=2000, run_validation=False)
config["scheduler"] = dict(warmup=0)
cs = ConfigStore.instance()
for validation in (False, True):
    cs.store(group="data_val" if validation else "data_train", package="dataloader_val" if validation else "dataloader_train",
             name="pai_release", node=loader_config(tokenizer=True, validation=validation))
cs.store(group="experiment", package="_global_", name="lidarverse_tokenizer_pai_finetune", node=config)
