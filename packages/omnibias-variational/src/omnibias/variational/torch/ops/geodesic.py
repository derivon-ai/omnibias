# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Geodesics as least action -- the metric Lagrangian bridge (torch).

A geodesic extremises the energy functional of the kinetic Lagrangian

.. math::

    L = \tfrac12\,g_{ij}(q)\,\dot q^i\dot q^j,

whose Euler-Lagrange equation is the (index-lowered) geodesic equation
``g_{kj}(qddot^j + Gamma^j_{lm} qdot^l qdot^m) = 0``. This module turns an
``omnibias-geometry`` ``ManifoldSpec`` into that :class:`Lagrangian`, so the
generic :func:`euler_lagrange_residual` reproduces ``omnibias.geometry``'s
``geodesic_rhs``. It also exposes the arc-length functional.

Only the *metric callable* on the manifold is used; ``omnibias-geometry`` is a
soft dependency (nothing here imports it -- pass in the ``ManifoldSpec``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from omnibias.fields.torch.ops.basic import stack_components, vector_derivative
from omnibias.variational._core.lagrangian import Lagrangian
from omnibias.variational.torch.ops.action import integrate_values
from torch import Tensor
from torch.func import vmap

if TYPE_CHECKING:  # pragma: no cover
    from omnibias.fields._core.quadrature import QuadratureSpec
    from omnibias.fields._core.state import FieldState
    from omnibias.geometry._core.manifold import ManifoldSpec


def _metric_quadratic(g_point, q: Tensor, qdot: Tensor) -> Tensor:  # type: ignore[no-untyped-def]
    """``g_ij(q) qdot^i qdot^j`` for a single sample ``(n,)`` or a batch ``(..., n)``."""
    if q.ndim == 1:
        g = g_point(q)
        return torch.einsum("i,ij,j->", qdot, g, qdot)
    lead = q.shape[:-1]
    n = q.shape[-1]
    gf = vmap(g_point)(q.reshape(-1, n))            # (M, n, n)
    qf = qdot.reshape(-1, n)
    out = torch.einsum("mi,mij,mj->m", qf, gf, qf)
    return out.reshape(lead)


def metric_lagrangian(
    manifold: ManifoldSpec, *, dof: tuple[str, ...] | None = None, time_axis: str = "t",
) -> Lagrangian:
    r"""Kinetic Lagrangian ``L = 1/2 g_ij(q) qdot^i qdot^j`` of a manifold.

    Parameters
    ----------
    manifold
        An ``omnibias-geometry`` ``ManifoldSpec`` supplying the per-point metric.
    dof
        Component names of the trajectory's generalized coordinates (default
        ``("q0", ..., "q{d-1}")``); must match the field's components.
    time_axis
        Name of the curve parameter axis (default ``"t"``).
    """
    g_point = manifold.metric.g_point
    names = dof if dof is not None else tuple(f"q{i}" for i in range(manifold.dim))

    def fn(q: Tensor, qdot: Tensor, t: Tensor) -> Tensor:
        return 0.5 * _metric_quadratic(g_point, q, qdot)

    return Lagrangian(fn, dof=names, time_axis=time_axis)


def geodesic_action(
    state: FieldState,
    manifold: ManifoldSpec,
    *,
    rule: QuadratureSpec,
    dof: tuple[str, ...] | None = None,
    time_axis: str = "t",
) -> Tensor:
    r"""Arc length :math:`\int \sqrt{g_{ij}\dot q^i\dot q^j}\,dt`, a scalar tensor."""
    names = dof if dof is not None else tuple(f"q{i}" for i in range(manifold.dim))
    q = stack_components(state, names)
    qdot = vector_derivative(state, names, axis=time_axis, order=1)
    speed_sq = _metric_quadratic(manifold.metric.g_point, q, qdot)
    return integrate_values(torch.sqrt(speed_sq), rule=rule)


__all__ = ["geodesic_action", "metric_lagrangian"]
