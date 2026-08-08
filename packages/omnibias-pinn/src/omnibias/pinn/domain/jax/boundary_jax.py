# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""JAX boundary-factor jets mirroring :mod:`omnibias.pinn.domain._core.boundary`."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from jax import Array
from omnibias.pinn.domain._core.boundary import BCMode, boundary_factor_jet
from omnibias.pinn.domain._core.sdf import SDF


def boundary_factor_jet_at(
    sdf: SDF,
    x0: Array,
    *,
    order: int,
    mode: BCMode = "dirichlet",
    normalize: bool = True,
    robin_alpha: float = 1.0,
    robin_beta: float = 0.0,
    h: float = 1e-6,
) -> Array:
    """Exact multivariate jet of the BC distance factor at ``x0``."""
    x0 = jnp.asarray(x0)
    if x0.ndim != 1:
        raise ValueError(f"x0 must be 1-D, got shape {tuple(x0.shape)}")
    jet = boundary_factor_jet(
        sdf,
        np.asarray(x0, dtype=float),
        order=order,
        mode=mode,
        normalize=normalize,
        robin_alpha=robin_alpha,
        robin_beta=robin_beta,
        h=h,
    )
    return jnp.asarray(jet, dtype=x0.dtype)


__all__ = ["boundary_factor_jet_at"]
