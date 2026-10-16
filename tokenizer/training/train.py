# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import importlib
import os
from copy import deepcopy

from loguru import logger as logging
from omegaconf import OmegaConf

from utils.config import Config, pretty_print_overrides
from utils.config_helper import get_config_module, override
from utils.lazy_config import instantiate
from utils.lazy_config.lazy import LazyConfig


@logging.catch(reraise=True)
def launch(config: Config, args: argparse.Namespace) -> None:
    # Check that the config is valid
    config.validate()
    # Save a copy of train/val data config into config.model for reproducibility (e.g. logging, checkpointing).
    try:
        from utils.processed_pai import build_loader
        if config.dataloader_train.get("_target_") in (build_loader, "utils.processed_pai.build_loader"):
            train_data_config = OmegaConf.create(dict(uses_sensor_metadata=True,
                shared_overrides=dict(process_input="mono", process_output="mono", load_intensity=False,
                                      min_range=2.5, max_range=100., repeat_row=1, repeat_col=1)))
        else:
            train_data_config = OmegaConf.create(deepcopy(config.dataloader_train.dataset))
        OmegaConf.update(config.model.config, "data_config", train_data_config, merge=False, force_add=True)
    except Exception as e:
        logging.warning(f"Could not copy train/val data config into model config: {e}")
    # Freeze the config so developers don't change it during training.
    config.freeze()  # type: ignore
    trainer = config.trainer.type(config)
    # Create the model
    model = instantiate(config.model)
    model.on_model_init_end()
    dataloader_train = instantiate(config.dataloader_train)
    dataloader_val = instantiate(config.dataloader_val)
    # Start training
    trainer.train(
        model,
        dataloader_train,
        dataloader_val,
    )


if __name__ == "__main__":
    # Usage: torchrun --nproc_per_node=1 -m scripts.train --config=projects/tutorials/mnist/config.py

    # Get the config file from the input arguments.
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--config", help="Path to the config file", required=True)
    parser.add_argument(
        "opts",
        help="""
Modify config options at the end of the command. For Yacs configs, use
space-separated "PATH.KEY VALUE" pairs.
For python-based LazyConfig, use "path.key=value".
        """.strip(),
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--dryrun",
        action="store_true",
        help="Do a dry run without training. Useful for debugging the config.",
    )
    args = parser.parse_args()
    config_module = get_config_module(args.config)
    config = importlib.import_module(config_module).make_config()
    config = override(config, args.opts)
    if args.dryrun:
        logging.info(
            "Config:\n" + config.pretty_print(use_color=True) + "\n" + pretty_print_overrides(args.opts, use_color=True)
        )
        os.makedirs(config.job.path_local, exist_ok=True)
        LazyConfig.save_yaml(config, f"{config.job.path_local}/config.yaml")
        print(f"{config.job.path_local}/config.yaml")
    else:
        # Launch the training job.
        launch(config, args)
