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

import gzip
import warnings
from io import BytesIO
from typing import Any

from utils.easy_io.handlers.pickle_handler import PickleHandler, _import_pickle

_GZ_LOAD_WARNING = (
    "Loading gzip-compressed pickle is deprecated due to security concerns. "
    "Consider converting .gz files to a safer format."
)

_GZ_DUMP_WARNING = (
    "Writing gzip-compressed pickle is deprecated. "
    "This will become an error in a future release."
)


class GzipHandler(PickleHandler):
    str_like = False

    def load_from_fileobj(self, file: BytesIO, **kwargs):
        warnings.warn(_GZ_LOAD_WARNING, DeprecationWarning, stacklevel=3)
        with gzip.GzipFile(fileobj=file, mode="rb") as f:
            return _import_pickle().load(f)

    def dump_to_fileobj(self, obj: Any, file: BytesIO, **kwargs):
        warnings.warn(_GZ_DUMP_WARNING, DeprecationWarning, stacklevel=3)
        with gzip.GzipFile(fileobj=file, mode="wb") as f:
            _import_pickle().dump(obj, f)
