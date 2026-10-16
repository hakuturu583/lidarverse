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

import warnings
from io import BytesIO
from typing import Any

from utils.easy_io.handlers.base import BaseFileHandler

_LOAD_WARNING = (
    "Loading data via pickle is deprecated due to security concerns "
    "(arbitrary code execution). Consider converting .pkl files to "
    ".yaml, .json, or .pt format."
)

_DUMP_WARNING = (
    "Writing data via pickle is deprecated. Use .yaml, .json, or .pt "
    "format instead. This will become an error in a future release."
)


def _import_pickle():
    import pickle as _pickle
    return _pickle


class PickleHandler(BaseFileHandler):
    str_like = False

    def load_from_fileobj(self, file: BytesIO, **kwargs):
        warnings.warn(_LOAD_WARNING, DeprecationWarning, stacklevel=3)
        return _import_pickle().load(file, **kwargs)

    def load_from_path(self, filepath, **kwargs):
        return super().load_from_path(filepath, mode="rb", **kwargs)

    def dump_to_str(self, obj, **kwargs):
        warnings.warn(_DUMP_WARNING, DeprecationWarning, stacklevel=3)
        kwargs.setdefault("protocol", 2)
        return _import_pickle().dumps(obj, **kwargs)

    def dump_to_fileobj(self, obj: Any, file: BytesIO, **kwargs):
        warnings.warn(_DUMP_WARNING, DeprecationWarning, stacklevel=3)
        kwargs.setdefault("protocol", 2)
        _import_pickle().dump(obj, file, **kwargs)

    def dump_to_path(self, obj, filepath, **kwargs):
        warnings.warn(_DUMP_WARNING, DeprecationWarning, stacklevel=3)
        with open(filepath, "wb") as f:
            _import_pickle().dump(obj, f, **kwargs)
