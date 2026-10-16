# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from enum import Enum
from tokenizer.networks.common.distributions import GaussianDistribution, IdentityDistribution
from tokenizer.networks.common.quantizers import FSQuantizer, LFQuantizer, ResidualFSQuantizer, VectorQuantizer, InvQuantizerJit
from tokenizer.networks.common.patching import Patcher, UnPatcher, Patcher3D, UnPatcher3D

class ContinuousFormulation(Enum):
    VAE = GaussianDistribution
    AE = IdentityDistribution


class DiscreteQuantizer(Enum):
    VQ = VectorQuantizer
    LFQ = LFQuantizer
    FSQ = FSQuantizer
    RESFSQ = ResidualFSQuantizer

