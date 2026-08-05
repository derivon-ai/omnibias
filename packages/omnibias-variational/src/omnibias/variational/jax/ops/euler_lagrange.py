# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""The Euler-Lagrange operator (jax). Bit-identical twin of the torch module."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
from jax import Array, jacrev, vmap
from omnibias.fields.jax.ops.basic import stack_components, vector_derivative

if TYPE_CHECKING:  # pragma: no cover
    from omnibias.fields._core.state import FieldState
    from omnibias.variational._core.lagrangian import Lagrangian


def trajectory(state: FieldState, lagrangian: Lagrangian) -> tuple[Array, Array, Array, Array]:
    r"""Closed-form ``(q, qdot, qddot, t)`` along the trajectory."""
    dof = lagrangian.dof
    ax = lagrangian.time_axis
    q = stack_components(state, dof)
    qdot = vector_derivative(state, dof, axis=ax, order=1)
    qddot = vector_derivative(state, dof, axis=ax, order=2)
    idx = state.coordinate_spec.axis_index(ax)
    t = state.coords[:, idx][:, None]
    return q, qdot, qddot, t


def lagrangian_partials(
    lagrangian: Lagrangian, q: Array, qdot: Array, t: Array,
) -> tuple[Array, Array]:
    r"""Autodiff partials ``(dL/dq, dL/dqdot)``, each shape ``(B, n_dof)``."""
    fn = lagrangian.fn
    g_q = vmap(jacrev(fn, argnums=0))(q, qdot, t)
    g_v = vmap(jacrev(fn, argnums=1))(q, qdot, t)
    return g_q, g_v


def euler_lagrange_residual(state: FieldState, lagrangian: Lagrangian) -> Array:
    r"""Euler-Lagrange residual ``d/dt(dL/dqdot) - dL/dq``, shape ``(B, n_dof)``.

    For ``order >= 2`` this is the Euler-Poisson residual ``-functional_derivative``.
    """
    if lagrangian.order != 1:
        from omnibias.variational.jax.ops.functional import functional_derivative

        return -functional_derivative(state, lagrangian)
    q, qdot, qddot, t = trajectory(state, lagrangian)
    fn = lagrangian.fn
    g_q = vmap(jacrev(fn, argnums=0))(q, qdot, t)
    dl_dv = jacrev(fn, argnums=1)
    h_vq = vmap(jacrev(dl_dv, argnums=0))(q, qdot, t)
    h_vv = vmap(jacrev(dl_dv, argnums=1))(q, qdot, t)
    h_vt = vmap(jacrev(dl_dv, argnums=2))(q, qdot, t)
    dp_dt = (
        jnp.einsum("bij,bj->bi", h_vq, qdot)
        + jnp.einsum("bij,bj->bi", h_vv, qddot)
        + h_vt[..., 0]
    )
    return dp_dt - g_q


__all__ = ["euler_lagrange_residual", "lagrangian_partials", "trajectory"]
