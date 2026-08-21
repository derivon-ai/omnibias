# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""JAX training drivers for omnibias.pinn.train."""

from __future__ import annotations

from omnibias.pinn.train.jax.depth_residual import (
    depth_residual_sweep,
    field_tower,
    poisson_1d_residual,
)
from omnibias.pinn.train.jax.march import MarchResult, WindowResult, march_solve

__all__ = [
    "MarchResult",
    "WindowResult",
    "depth_residual_sweep",
    "field_tower",
    "march_solve",
    "poisson_1d_residual",
]
