# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Geodesics as least action -- the metric Lagrangian bridge (jax).

Bit-identical twin of the torch module.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
from jax import Array, vmap
from omnibias.fields.jax.ops.basic import stack_components, vector_derivative
from omnibias.variational._core.lagrangian import Lagrangian
from omnibias.variational.jax.ops.action import integrate_values

if TYPE_CHECKING:  # pragma: no cover
    from omnibias.fields._core.quadrature import QuadratureSpec
    from omnibias.fields._core.state import FieldState
    from omnibias.geometry._core.manifold import ManifoldSpec


def _metric_quadratic(g_point, q: Array, qdot: Array) -> Array:  # type: ignore[no-untyped-def]
    """``g_ij(q) qdot^i qdot^j`` for a single sample ``(n,)`` or a batch ``(..., n)``."""
    if q.ndim == 1:
        g = g_point(q)
        return jnp.einsum("i,ij,j->", qdot, g, qdot)
    lead = q.shape[:-1]
    n = q.shape[-1]
    gf = vmap(g_point)(q.reshape(-1, n))
    qf = qdot.reshape(-1, n)
    out = jnp.einsum("mi,mij,mj->m", qf, gf, qf)
    return out.reshape(lead)


def metric_lagrangian(
    manifold: ManifoldSpec, *, dof: tuple[str, ...] | None = None, time_axis: str = "t",
) -> Lagrangian:
    r"""Kinetic Lagrangian ``L = 1/2 g_ij(q) qdot^i qdot^j`` of a manifold."""
    g_point = manifold.metric.g_point
    names = dof if dof is not None else tuple(f"q{i}" for i in range(manifold.dim))

    def fn(q: Array, qdot: Array, t: Array) -> Array:
        return 0.5 * _metric_quadratic(g_point, q, qdot)

    return Lagrangian(fn, dof=names, time_axis=time_axis)


def geodesic_action(
    state: FieldState,
    manifold: ManifoldSpec,
    *,
    rule: QuadratureSpec,
    dof: tuple[str, ...] | None = None,
    time_axis: str = "t",
) -> Array:
    r"""Arc length :math:`\int \sqrt{g_{ij}\dot q^i\dot q^j}\,dt`, a scalar array."""
    names = dof if dof is not None else tuple(f"q{i}" for i in range(manifold.dim))
    q = stack_components(state, names)
    qdot = vector_derivative(state, names, axis=time_axis, order=1)
    speed_sq = _metric_quadratic(manifold.metric.g_point, q, qdot)
    return integrate_values(jnp.sqrt(speed_sq), rule=rule)


__all__ = ["geodesic_action", "metric_lagrangian"]
