# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Shared fixtures for the omnibias-convex test suite.

Interior-point convergence and the verified enclosures need float64, so we enable
JAX x64 before any jax import (mirrors the omnibias-jax test setup)."""

from __future__ import annotations

import os

os.environ.setdefault("JAX_ENABLE_X64", "true")

try:  # pragma: no cover - only runs when jax is installed
    import jax

    jax.config.update("jax_enable_x64", True)
except ModuleNotFoundError:  # pragma: no cover
    pass
