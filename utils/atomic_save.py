# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Atomic, crash-safe helpers for writing checkpoints.

The default ``torch.save(obj, path)`` writes incrementally to ``path``. If the
process is killed (SIGTERM/SIGKILL/OOM/wall-time) while the write is in
progress, the file at ``path`` is left torn — typically with a header but no
zip central directory at the tail — and the next ``torch.load(path)`` blows
up with ``PytorchStreamReader failed reading zip archive: failed finding
central directory``. The previous good copy at ``path`` is also lost because
``torch.save`` overwrote it in place.

The functions in this module fix that pattern by:

* Writing first to a temporary sibling file ``<path>.tmp.<pid>``,
* fsync()-ing that file so its bytes are durable on disk,
* Atomically renaming it onto ``path`` with ``os.replace`` (POSIX guarantees
  the rename is atomic on the same filesystem, so a concurrent reader either
  sees the old complete file or the new complete file — never a torn one),
* Optionally fsync()-ing the parent directory so the new directory entry is
  durable across kernel-level crashes (cheap; off by default since our
  failure mode is process-level, not power-loss).

After every successful write we also lightly **validate** the file (check
that ``torch.save`` actually produced a well-formed zip), turning silent
corruption into a loud, immediate failure that the caller can retry.
"""

from __future__ import annotations

import os
import zipfile
from typing import Any

import torch


class CorruptCheckpointError(RuntimeError):
    """Raised when a freshly-written .pt file fails post-save validation."""


def _tmp_path(path: str) -> str:
    """Sibling path used while a write is in flight."""
    return f"{path}.tmp.{os.getpid()}"


def _fsync_file(path: str) -> None:
    """Force ``path``'s bytes to disk. No-op if the file no longer exists."""
    try:
        fd = os.open(path, os.O_RDONLY)
    except FileNotFoundError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_dir(path: str) -> None:
    """Force the *directory entry* of ``path``'s parent to disk.

    Needed for full crash-safety of the rename: without it, an OS crash
    immediately after ``os.replace`` may roll the directory back to a state
    that no longer references the new inode.
    """
    parent = os.path.dirname(os.path.abspath(path)) or "."
    try:
        fd = os.open(parent, os.O_RDONLY)
    except (FileNotFoundError, NotADirectoryError):
        return
    try:
        os.fsync(fd)
    except OSError:
        # Some filesystems (e.g. some FUSE/NFS configs) don't support
        # fsync() on directories. The atomic rename is still the strongest
        # safety we can provide there; swallow rather than fail the save.
        pass
    finally:
        os.close(fd)


def validate_pt_file(path: str) -> None:
    """Verify ``path`` is a structurally valid PyTorch .pt zip archive.

    ``torch.save`` writes a zip; if the zip's central directory at the tail
    is missing or malformed, ``torch.load`` will fail with ``PytorchStream
    Reader failed reading zip archive: failed finding central directory``.
    ``zipfile.is_zipfile`` does the cheap end-of-archive check that catches
    exactly that failure mode (and is O(1) regardless of file size).

    Raises:
        CorruptCheckpointError: if the file is missing or not a valid zip.
    """
    if not os.path.exists(path):
        raise CorruptCheckpointError(f"validation: file does not exist: {path}")
    if os.path.getsize(path) == 0:
        raise CorruptCheckpointError(f"validation: file is empty: {path}")
    if not zipfile.is_zipfile(path):
        raise CorruptCheckpointError(
            f"validation: not a valid zip archive (likely a torn write): {path}"
        )


def atomic_torch_save(
    obj: Any,
    path: str,
    *,
    validate: bool = True,
    fsync_dir: bool = False,
    **torch_save_kwargs: Any,
) -> None:
    """Save ``obj`` to ``path`` atomically using ``torch.save`` + rename.

    A SIGTERM/SIGKILL during the write leaves only the throwaway
    ``<path>.tmp.<pid>`` partially written; ``path`` itself either keeps the
    previous-iteration's complete file or doesn't exist yet — never a torn
    zip. By the time this function returns successfully, ``path`` is the
    fully-written, validated checkpoint.

    Args:
        obj: object to pickle (any ``torch.save``-compatible value).
        path: final destination path (typically ``...iter_NNNNNN.pt``).
        validate: if True (default), verify the written file is a valid
            zip before renaming it into place; raise on failure.
        fsync_dir: if True, also fsync the parent directory after rename
            so the directory entry survives a kernel-level crash. Off by
            default (the rename alone is sufficient for process-level
            kills like SIGTERM/SIGKILL/OOM).
        **torch_save_kwargs: forwarded to ``torch.save``.
    """
    tmp = _tmp_path(path)
    # Make sure no stale tmp from a previous, killed save is in the way.
    try:
        os.remove(tmp)
    except FileNotFoundError:
        pass

    torch.save(obj, tmp, **torch_save_kwargs)
    _fsync_file(tmp)

    if validate:
        try:
            validate_pt_file(tmp)
        except CorruptCheckpointError:
            try:
                os.remove(tmp)
            except FileNotFoundError:
                pass
            raise

    os.replace(tmp, path)
    if fsync_dir:
        _fsync_dir(path)


def atomic_torch_jit_save(
    jit_module: torch.jit.ScriptModule,
    path: str,
    *,
    validate: bool = True,
    fsync_dir: bool = False,
) -> None:
    """``torch.jit.save`` analog of :func:`atomic_torch_save`.

    TorchScript files are also zip archives, so the same tmp+rename and
    zip-validity check apply.
    """
    tmp = _tmp_path(path)
    try:
        os.remove(tmp)
    except FileNotFoundError:
        pass

    torch.jit.save(jit_module, tmp)
    _fsync_file(tmp)

    if validate:
        try:
            validate_pt_file(tmp)
        except CorruptCheckpointError:
            try:
                os.remove(tmp)
            except FileNotFoundError:
                pass
            raise

    os.replace(tmp, path)
    if fsync_dir:
        _fsync_dir(path)


def atomic_write_text(path: str, content: str, *, fsync_dir: bool = False) -> None:
    """Write ``content`` to ``path`` atomically.

    Used for tiny pointer files like ``latest_checkpoint.txt`` where a torn
    write (e.g. a 0-byte file because the process died between
    ``open(..., "w")`` and ``write``) silently breaks resume.
    """
    tmp = _tmp_path(path)
    try:
        os.remove(tmp)
    except FileNotFoundError:
        pass

    with open(tmp, "w") as f:
        f.write(content)
        f.flush()
        os.fsync(f.fileno())

    os.replace(tmp, path)
    if fsync_dir:
        _fsync_dir(path)
