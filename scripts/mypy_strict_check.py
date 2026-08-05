#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Strict-typecheck the authored-strict allowlist for the curated beta packages.

The curated beta packages (``omnibias-fields`` / ``omnibias-geometry``) are not
on the blanket ``mypy --strict`` gate -- their bulk-copied torch/jax field-op
modules carry systematic ``no-any-return`` / ``type-arg`` findings. This script
gates the foundational, authored-strict ``_core`` substrate listed in
``scripts/mypy_strict_allowlist.txt`` with ``--follow-imports=silent`` so the
(ungated) backend-op modules cannot leak errors into the gate. It is an
incremental, growing strict surface: add a module to the allowlist once it is
clean under this exact invocation.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_ALLOWLIST = _ROOT / "scripts" / "mypy_strict_allowlist.txt"

# Source roots that must resolve for the substrate to type-check coherently.
_MYPYPATH = (
    "packages/omnibias-core/src",
    "packages/omnibias-torch/src",
    "packages/omnibias-jax/src",
    "packages/omnibias-fields/src",
    "packages/omnibias-geometry/src",
    "packages/omnibias-pinn/src",
)


def _allowlist_files() -> list[str]:
    files: list[str] = []
    for line in _ALLOWLIST.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            files.append(stripped)
    return files


def main() -> int:
    files = _allowlist_files()
    if not files:
        print("mypy_strict_check: allowlist is empty", file=sys.stderr)
        return 1

    env = dict(os.environ)
    env["MYPYPATH"] = os.pathsep.join(str(_ROOT / p) for p in _MYPYPATH)

    cmd = [
        sys.executable,
        "-m",
        "mypy",
        "--strict",
        "--namespace-packages",
        "--explicit-package-bases",
        "--follow-imports=silent",
        *files,
    ]
    print(f"mypy_strict_check: {len(files)} authored-strict modules")
    return subprocess.call(cmd, cwd=str(_ROOT), env=env)


if __name__ == "__main__":
    raise SystemExit(main())
