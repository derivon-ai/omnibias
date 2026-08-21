# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""JAX twin of CSP softmax / energy (theory 03-03).

``softmax_rows`` matches numpy / torch at float64 (``jax_enable_x64``).
Simplex and clause ``beta -> inf`` are temperature collapse
(feasibility), not the founding bias collapse (``delta -> 0``). Do not conflate the two.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array
from omnibias.discrete.csp._core import (
    CSP,
    CSPResult,
    csp_solve,
    triangle_colouring,
)


def softmax_rows(logits: Array, *, beta: float) -> Array:
    z = float(beta) * jnp.asarray(logits, dtype=jnp.float64)
    z = z - jnp.max(z, axis=-1, keepdims=True)
    w = jnp.exp(z)
    return w / jnp.sum(w, axis=-1, keepdims=True)


def energy(csp: CSP, x: Array) -> Array:
    xv = jnp.asarray(x, dtype=jnp.float64)
    return jnp.asarray(csp.energy(xv), dtype=jnp.float64)


__all__ = ["CSP", "CSPResult", "csp_solve", "energy", "softmax_rows", "triangle_colouring"]
