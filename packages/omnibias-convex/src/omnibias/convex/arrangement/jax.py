# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""JAX twin of arrangement-LP soft membership (theory 03-02).

``soft_membership`` matches the numpy / torch twins at float64
(``jax_enable_x64``). Soft mode is temperature collapse (``beta -> inf``,
feasibility), not the founding bias collapse (``delta -> 0``). Do not conflate
the two.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array
from omnibias.convex.arrangement._core import (
    DiffMode,
    LearnedPolytope,
    LPOutput,
    named_pentagon,
    soft_cell_gap_bound,
    solve_arrangement_lp,
    sound_lower_bound,
)


def soft_membership(normals: Array, offsets: Array, x: Array, *, beta: float) -> Array:
    """``prod_i sigma(beta (b_i - a_i · x))``."""
    a = jnp.asarray(normals, dtype=jnp.float64)
    b = jnp.asarray(offsets, dtype=jnp.float64).reshape(-1)
    xv = jnp.asarray(x, dtype=jnp.float64)
    if xv.ndim == 1:
        slack = b - a @ xv
        return jnp.prod(jax_sigmoid(float(beta) * slack))
    slack = b[None, :] - xv @ a.T
    return jnp.prod(jax_sigmoid(float(beta) * slack), axis=-1)


def jax_sigmoid(z: Array) -> Array:
    return 1.0 / (1.0 + jnp.exp(-z))


class LPLayer:
    """Learned-polytope LP. ``mode`` is returned so the gradient is interpretable."""

    def __init__(self, mode: DiffMode, *, beta: float | None = None) -> None:
        self.mode = DiffMode(mode)
        self.beta = 8.0 if beta is None else float(beta)
        if self.mode is DiffMode.SOFT and self.beta <= 0.0:
            raise ValueError("SOFT mode needs beta > 0")

    def forward(self, polytope: LearnedPolytope, c: Array) -> LPOutput:
        cv = jnp.asarray(c, dtype=jnp.float64)
        return solve_arrangement_lp(polytope, cv, mode=self.mode, beta=self.beta)


__all__ = [
    "DiffMode",
    "LPLayer",
    "LPOutput",
    "LearnedPolytope",
    "named_pentagon",
    "soft_cell_gap_bound",
    "soft_membership",
    "solve_arrangement_lp",
    "sound_lower_bound",
]
