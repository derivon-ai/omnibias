# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Serialize generated Lean obligations across threads and processes."""

from __future__ import annotations

import sys
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

_registry_guard = threading.Lock()
_project_locks: dict[str, threading.Lock] = {}


@contextmanager
def lean_project_lock(root: Path) -> Iterator[None]:
    """Lock a Lean project across cooperating threads and processes.

    OS advisory locks coordinate cooperating bridge calls in separate processes.
    A manual editor or direct external ``lake build`` does not take this lock.
    """
    key = str(root.resolve())
    with _registry_guard:
        mutex = _project_locks.setdefault(key, threading.Lock())
    lock_directory = root / ".lake"
    lock_directory.mkdir(parents=True, exist_ok=True)
    lockfile = lock_directory / "omnibias-bridge.lock"
    with mutex, lockfile.open("a+b") as handle:
        if sys.platform == "win32":  # pragma: no cover - platform specific
            import msvcrt

            handle.seek(0)
            handle.write(b"0")
            handle.flush()
            while True:
                try:
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    time.sleep(0.05)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if sys.platform == "win32":  # pragma: no cover - platform specific
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def generated_lean_obligation(root: Path, generated: Path, source: str) -> Iterator[None]:
    """Lock through write/build/restore, restoring even after a failed build."""
    with lean_project_lock(root):
        original = generated.read_bytes() if generated.exists() else None
        try:
            generated.write_text(source, encoding="utf-8")
            yield
        finally:
            if original is None:
                generated.unlink(missing_ok=True)
            else:
                generated.write_bytes(original)


__all__ = ["generated_lean_obligation", "lean_project_lock"]
