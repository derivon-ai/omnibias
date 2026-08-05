# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Shared fixtures for the omnibias-submodular test suite.

The differentiable relaxation twins need float64, so we enable JAX x64 before any jax
import (mirrors the omnibias-qubo / omnibias-convex setup)."""

from __future__ import annotations

import os

os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

try:  # pragma: no cover - only runs when jax is installed
    import jax

    jax.config.update("jax_enable_x64", True)
except ModuleNotFoundError:  # pragma: no cover
    pass
