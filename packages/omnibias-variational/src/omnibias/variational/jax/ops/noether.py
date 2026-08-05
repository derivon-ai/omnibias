# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Noether charges of a continuous symmetry (jax). Bit-identical twin."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
from jax import Array
from omnibias.variational.jax.ops.euler_lagrange import trajectory
from omnibias.variational.jax.ops.hamiltonian import conjugate_momentum

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable

    from omnibias.fields._core.state import FieldState
    from omnibias.variational._core.lagrangian import Lagrangian


def noether_charge(
    state: FieldState,
    lagrangian: Lagrangian,
    generator: Array | Callable[[Array, Array, Array], Array],
) -> Array:
    r"""Conserved charge ``Q = p . X`` of the symmetry ``generator`` ``X``, ``(B,)``."""
    q, qdot, _qddot, t = trajectory(state, lagrangian)
    p = conjugate_momentum(state, lagrangian)
    x = generator(q, qdot, t) if callable(generator) else generator
    return jnp.einsum("bi,bi->b", p, x)


__all__ = ["noether_charge"]
