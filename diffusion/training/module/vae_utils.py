# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build the public tokenizer architecture from its experiment config."""

import importlib
from copy import deepcopy

from omegaconf import DictConfig

from tokenizer.training.configs.base.nets.lidarverse import (
    LIDARVERSE_TOKENIZER_CONFIG,
    PAI_TOKENIZER_NAME,
)
from utils.lazy_config import instantiate


def build_vae_from_tokenizer_experiment(experiment_name: str, **overrides):
    """Instantiate a tokenizer network without loading checkpoint weights."""
    if experiment_name == PAI_TOKENIZER_NAME:
        # The PAI expert has the same GridRoPE architecture, different weights.
        network = deepcopy(LIDARVERSE_TOKENIZER_CONFIG)
        network.activation_checkpointing = False
        network.update(overrides)
        return instantiate(network)

    module = importlib.import_module(
        f"tokenizer.training.configs.experiments.{experiment_name}"
    )
    experiment = getattr(module, "config", None)
    if not isinstance(experiment, DictConfig):
        experiment = next(
            (
                value
                for value in vars(module).values()
                if isinstance(value, DictConfig) and "defaults" in value
            ),
            None,
        )
    if experiment is None:
        raise ValueError(
            f"Tokenizer experiment {experiment_name!r} has no config"
        )

    network_name = None
    for entry in experiment["defaults"]:
        try:
            network_name = entry["override /network"]
        except (KeyError, TypeError):
            continue
        break
    if network_name != "lidarverse_tokenizer":
        raise ValueError(
            f"Unsupported public tokenizer network: {network_name!r}"
        )

    network = deepcopy(LIDARVERSE_TOKENIZER_CONFIG)
    try:
        experiment_overrides = experiment["model"]["config"]["network"]
    except (KeyError, TypeError):
        experiment_overrides = None
    if isinstance(experiment_overrides, DictConfig):
        for key in experiment_overrides:
            network[key] = experiment_overrides[key]
    for key, value in overrides.items():
        network[key] = value
    return instantiate(network)
