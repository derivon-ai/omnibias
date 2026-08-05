# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Definite integration of a field component over a box domain (torch).

The :func:`integrate` op contracts the component values *already evaluated at a
quadrature rule's nodes* with that rule's weights. Evaluate the field at
:func:`quadrature_nodes` first::

    rule = gauss_legendre([(0.0, 1.0)], 16)
    state = field(quadrature_nodes(rule, like=coords))
    total = integrate(state, "u", rule=rule)        # scalar tensor

This is a quadrature sum :math:`\\sum_q w_q\\,u(x_q)`; it is exact for
polynomials up to the rule's degree and converges otherwise. The nodes and
weights come from the shared numpy ``QuadratureSpec``, so torch and jax produce
bit-identical results.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from omnibias.fields._core.quadrature import QuadratureSpec
from omnibias.fields.torch.ops.basic import gradient, value
from torch import Tensor
from torch.func import jacfwd, vmap

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable

    from omnibias.fields._core.state import FieldState


def quadrature_nodes(rule: QuadratureSpec, *, like: Tensor) -> Tensor:
    """Return the rule's nodes as a ``(n_nodes, dim)`` tensor matching ``like``.

    Parameters
    ----------
    rule
        The quadrature rule whose nodes to materialise.
    like
        A tensor whose dtype and device the nodes should match.
    """
    return torch.as_tensor(rule.nodes, dtype=like.dtype, device=like.device)


def _weights(rule: QuadratureSpec, ref: Tensor) -> Tensor:
    return torch.as_tensor(rule.weights, dtype=ref.dtype, device=ref.device)


def integrate(state: FieldState, name: str, *, rule: QuadratureSpec) -> Tensor:
    r"""Definite integral :math:`\int_\Omega u\,dx` via the quadrature ``rule``.

    ``state`` must have been evaluated at ``rule``'s nodes (so that
    ``state.coords.shape[0] == rule.n_nodes``).

    Parameters
    ----------
    state
        Field state evaluated at the quadrature nodes.
    name
        Scalar component name to integrate.
    rule
        The quadrature rule supplying the weights.

    Returns
    -------
    Tensor
        A scalar tensor (shape ``()``) holding the integral.
    """
    vals = value(state, name)
    if vals.shape[0] != rule.n_nodes:
        raise ValueError(
            f"integrate: state has {vals.shape[0]} points but rule has "
            f"{rule.n_nodes} nodes; evaluate the field at quadrature_nodes(rule)"
        )
    w = _weights(rule, vals)
    return torch.tensordot(w, vals, dims=([0], [0]))


def _tangent(curve: Callable[[Tensor], Tensor], x: Tensor) -> Tensor:
    """Batched curve tangent ``r'(t)`` of shape ``(Q, n)`` at param nodes ``x`` (Q, 1)."""
    jac: Tensor = vmap(jacfwd(curve))(x)  # (Q, n, 1)
    return jac[..., 0]


def line_integral(
    state: FieldState,
    name: str,
    curve: Callable[[Tensor], Tensor],
    *,
    rule: QuadratureSpec,
) -> Tensor:
    r"""Gradient-theorem line integral :math:`\int_C \nabla u \cdot d\mathbf r`.

    Evaluates :math:`\int_{t_0}^{t_1} \nabla u(r(t))\cdot r'(t)\,dt` for the scalar
    potential ``name`` along the curve ``r``. By the multivariate Fundamental
    Theorem of Calculus (the gradient theorem) this equals ``u(r(t_1)) -
    u(r(t_0))`` for any path.

    ``state`` must hold ``name`` evaluated at the curve image points
    ``curve(quadrature_nodes(rule))`` (the same pre-evaluation convention as the
    geometry surface integrals); ``curve`` is a bare callable mapping a ``(1,)``
    parameter to an ambient point ``(n,)`` -- fields never import geometry, so the
    curve is not a ``ChartSpec``. The gradient :math:`\nabla u` is the closed-form
    field op and the tangent ``r'(t)`` is exact forward-mode autodiff; the integral
    itself is Gauss-Legendre quadrature over the 1-D parameter box.

    Returns a scalar tensor.
    """
    g = gradient(state, name)  # (Q, dim)
    if g.shape[0] != rule.n_nodes:
        raise ValueError(
            f"line_integral: state has {g.shape[0]} points but rule has "
            f"{rule.n_nodes} nodes; evaluate the field at "
            "curve(quadrature_nodes(rule))"
        )
    x = quadrature_nodes(rule, like=g)  # (Q, param_dim) parameter nodes
    if x.shape[-1] != 1:
        raise ValueError(
            "line_integral requires a 1-D parameter rule (a curve); got a "
            f"{x.shape[-1]}-D rule"
        )
    tangent = _tangent(curve, x)  # (Q, dim)
    if tangent.shape[-1] != g.shape[-1]:
        raise ValueError(
            f"line_integral: curve maps to {tangent.shape[-1]} ambient dims but "
            f"the field gradient spans {g.shape[-1]}; the curve must land in the "
            "field's gradient axes"
        )
    integrand = (g * tangent).sum(dim=-1)  # (Q,)
    w = _weights(rule, integrand)
    return torch.tensordot(w, integrand, dims=([0], [0]))


__all__ = ["integrate", "line_integral", "quadrature_nodes"]
