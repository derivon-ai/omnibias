# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Hamiltonian / energy quantities of a Lagrangian trajectory (jax).

Bit-identical twin of the torch module.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
from jax import Array, jacrev, vmap
from omnibias.variational.jax.ops.euler_lagrange import (
    euler_lagrange_residual,
    trajectory,
)

if TYPE_CHECKING:  # pragma: no cover
    from omnibias.fields._core.state import FieldState
    from omnibias.variational._core.lagrangian import Lagrangian


def conjugate_momentum(state: FieldState, lagrangian: Lagrangian) -> Array:
    r"""Conjugate momentum ``p_i = dL/dqdot_i``, shape ``(B, n_dof)``."""
    q, qdot, _qddot, t = trajectory(state, lagrangian)
    return vmap(jacrev(lagrangian.fn, argnums=1))(q, qdot, t)


def hamiltonian(state: FieldState, lagrangian: Lagrangian) -> Array:
    r"""Energy function ``H = sum_i p_i qdot_i - L``, shape ``(B,)``."""
    q, qdot, _qddot, t = trajectory(state, lagrangian)
    p = vmap(jacrev(lagrangian.fn, argnums=1))(q, qdot, t)
    lval = lagrangian.fn(q, qdot, t)
    return jnp.einsum("bi,bi->b", p, qdot) - lval


def energy(state: FieldState, lagrangian: Lagrangian) -> Array:
    r"""Total mechanical energy along the trajectory (alias of :func:`hamiltonian`)."""
    return hamiltonian(state, lagrangian)


def hamiltons_equations_residual(state: FieldState, lagrangian: Lagrangian) -> Array:
    r"""Energy / Hamilton residual ``sum_i qdot_i EL_i = dH/dt + dL/dt``, ``(B,)``."""
    _q, qdot, _qddot, _t = trajectory(state, lagrangian)
    el = euler_lagrange_residual(state, lagrangian)
    return jnp.einsum("bi,bi->b", qdot, el)


__all__ = [
    "conjugate_momentum",
    "energy",
    "hamiltonian",
    "hamiltons_equations_residual",
]
