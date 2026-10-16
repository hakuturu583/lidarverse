#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Create deterministic train/validation lists from sequence directories."""

from __future__ import annotations

import argparse
import random
from pathlib import Path


def _sequence_names(root: Path) -> list[str]:
    names = sorted(path.name for path in root.iterdir() if path.is_dir())
    if not names:
        raise ValueError(f"No sequence directories found under {root}")
    return names


def _write(path: Path, names: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(sorted(names)) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--train-output", type=Path, required=True)
    parser.add_argument("--val-output", type=Path, required=True)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not 0.0 < args.val_fraction < 1.0:
        raise ValueError("--val-fraction must be between zero and one")
    names = _sequence_names(args.root)
    random.Random(args.seed).shuffle(names)
    val_count = max(1, round(len(names) * args.val_fraction))
    val_names = names[:val_count]
    train_names = names[val_count:]
    if not train_names:
        raise ValueError("The requested split leaves no training sequences")

    _write(args.train_output, train_names)
    _write(args.val_output, val_names)
    print(
        f"Wrote {len(train_names)} training and {len(val_names)} validation "
        f"sequences from {args.root}"
    )


if __name__ == "__main__":
    main()
