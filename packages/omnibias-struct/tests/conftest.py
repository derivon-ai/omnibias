# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Shared pytest config: enable float64 JAX so the torch <-> jax twins are bit-identical."""

from __future__ import annotations

try:  # float64 parity for the torch <-> jax soft-DP twins
    import jax

    jax.config.update("jax_enable_x64", True)
except ImportError:  # pragma: no cover - jax optional
    pass
