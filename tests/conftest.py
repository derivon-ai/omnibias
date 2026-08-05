# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Pytest configuration for the workspace-level cross-backend tests.

Adds every package's ``src/`` directory to ``sys.path`` so the suite can
import from each ``omnibias.*`` namespace without a separate editable
install per package. This mirrors what ``uv sync`` does for development
checkouts; CI in clean venvs uses ``pip install -e packages/<name>``
instead and does not rely on this file.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_PACKAGE_DIRS = [
    _REPO_ROOT / "packages" / "omnibias-core" / "src",
    _REPO_ROOT / "packages" / "omnibias-torch" / "src",
    _REPO_ROOT / "packages" / "omnibias-jax" / "src",
    _REPO_ROOT / "packages" / "omnibias-ferminet" / "src",
]

for _path in _PACKAGE_DIRS:
    spath = str(_path)
    if spath not in sys.path:
        sys.path.insert(0, spath)
