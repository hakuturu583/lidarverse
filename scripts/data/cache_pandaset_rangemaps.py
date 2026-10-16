#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build versioned 64x1024 PandaSet Pandar64 range-image caches."""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tokenizer.training.datasets.lidar_datasets.pandaset import (  # noqa: E402
    PANDASET_MAX_RANGE,
    PANDASET_MIN_RANGE,
    build_pandaset_sequence_cache,
    validate_pandaset_cache,
)


def _sequence_names(raw_root: Path, filter_list: str) -> list[str]:
    if filter_list:
        filter_path = Path(filter_list)
        if not filter_path.is_absolute():
            filter_path = REPO_ROOT / filter_path
        with filter_path.open("r") as handle:
            names = [line.strip() for line in handle if line.strip()]
    else:
        names = [
            path.name
            for path in sorted(raw_root.iterdir())
            if path.is_dir() and (path / "lidar" / "poses.json").is_file()
        ]
    missing = [name for name in names if not (raw_root / name).is_dir()]
    if missing:
        raise FileNotFoundError(
            f"PandaSet sequences absent under {raw_root}: {missing[:8]}"
        )
    return names


def _existing_cache_is_valid(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as payload:
            validate_pandaset_cache(payload, path=path)
        return True
    except (OSError, ValueError):
        return False


def _cache_one(
    raw_root: str,
    packet_root: str,
    output_root: str,
    sequence: str,
    min_range: float,
    max_range: float,
    overwrite: bool,
) -> tuple[str, str]:
    output_path = Path(output_root) / f"{sequence}.npz"
    if not overwrite and _existing_cache_is_valid(output_path):
        return sequence, "skipped"
    payload = build_pandaset_sequence_cache(
        Path(raw_root) / sequence,
        packet_sequence_dir=Path(packet_root) / sequence,
        min_range=min_range,
        max_range=max_range,
    )
    temporary_path = output_path.with_name(
        f".{output_path.name}.tmp-{os.getpid()}.npz"
    )
    try:
        np.savez_compressed(temporary_path, **payload)
        with np.load(temporary_path, allow_pickle=False) as cached:
            validate_pandaset_cache(cached, path=temporary_path)
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return sequence, "written"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw_root",
        required=True,
        help="Processed PandaSet root supplying poses, timestamps, and calibration XYZ.",
    )
    parser.add_argument(
        "--packet_root",
        required=True,
        help=(
            "Official raw PandaSet root containing laser_id, column_id, elevation, "
            "azimuth_col_corrected, distance, and intensity."
        ),
    )
    parser.add_argument(
        "--output_root",
        default=str(REPO_ROOT / "data/pandaset_rangemap_64x1024"),
    )
    parser.add_argument(
        "--filter_list",
        default="",
        help="Optional repo-relative list of three-digit sequence IDs.",
    )
    parser.add_argument("--min_range", type=float, default=PANDASET_MIN_RANGE)
    parser.add_argument("--max_range", type=float, default=PANDASET_MAX_RANGE)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    packet_root = Path(args.packet_root)
    output_root = Path(args.output_root)
    if not np.isclose(args.min_range, PANDASET_MIN_RANGE) or not np.isclose(
        args.max_range, PANDASET_MAX_RANGE
    ):
        raise ValueError(
            "The PandaSet cache contract is fixed at "
            f"[{PANDASET_MIN_RANGE}, {PANDASET_MAX_RANGE}] m; got "
            f"[{args.min_range}, {args.max_range}]."
        )
    if not raw_root.is_dir():
        raise FileNotFoundError(f"PandaSet raw root not found: {raw_root}")
    if not packet_root.is_dir():
        raise FileNotFoundError(f"PandaSet packet root not found: {packet_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    sequences = _sequence_names(raw_root, args.filter_list)
    if not sequences:
        raise RuntimeError("No PandaSet sequences selected.")

    counts = {"written": 0, "skipped": 0}
    if args.num_workers <= 1:
        for index, sequence in enumerate(sequences, start=1):
            name, status = _cache_one(
                str(raw_root),
                str(packet_root),
                str(output_root),
                sequence,
                args.min_range,
                args.max_range,
                args.overwrite,
            )
            counts[status] += 1
            print(f"[{index}/{len(sequences)}] {name}: {status}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.num_workers) as pool:
            futures = {
                pool.submit(
                    _cache_one,
                    str(raw_root),
                    str(packet_root),
                    str(output_root),
                    sequence,
                    args.min_range,
                    args.max_range,
                    args.overwrite,
                ): sequence
                for sequence in sequences
            }
            completed = 0
            for future in as_completed(futures):
                name, status = future.result()
                completed += 1
                counts[status] += 1
                print(
                    f"[{completed}/{len(sequences)}] {name}: {status}",
                    flush=True,
                )
    print(
        f"Done: {counts['written']} written, {counts['skipped']} valid caches skipped "
        f"under {output_root}."
    )


if __name__ == "__main__":
    main()
