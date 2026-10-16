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

# pandas is only needed by the two CSV methods below, but this handler is
# instantiated eagerly in the easy_io handler registry (imported by utils.misc,
# i.e. almost the entire codebase). Importing pandas at module load crashes the
# whole utils stack on numpy-ABI-mismatched containers ("numpy.dtype size
# changed"). Guard it so unrelated code (viz, inference) keeps working; the CSV
# methods raise a clear error only if actually used.
try:
    import pandas as pd
except Exception:  # noqa: BLE001
    pd = None

from utils.easy_io.handlers.base import BaseFileHandler  # isort:skip


class PandasHandler(BaseFileHandler):
    str_like = False

    def load_from_fileobj(self, file, **kwargs):
        if pd is None:
            raise ImportError("pandas failed to import (see pandas_handler.py); cannot read CSV")
        return pd.read_csv(file, **kwargs)

    def dump_to_fileobj(self, obj, file, **kwargs):
        obj.to_csv(file, **kwargs)

    def dump_to_str(self, obj, **kwargs):
        raise NotImplementedError("PandasHandler does not support dumping to str")
