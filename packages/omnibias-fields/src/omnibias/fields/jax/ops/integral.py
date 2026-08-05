# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Definite integration of a field component over a box domain (jax).

Bit-identical twin of :mod:`omnibias.fields.torch.ops.integral`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax import Array
from omnibias.fields._core.quadrature import QuadratureSpec
from omnibias.fields.jax.ops.basic import gradient, value

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable

    from omnibias.fields._core.state import FieldState


def quadrature_nodes(rule: QuadratureSpec, *, like: Array) -> Array:
    """Return the rule's nodes as a ``(n_nodes, dim)`` array matching ``like``."""
    return jnp.asarray(rule.nodes, dtype=like.dtype)


def _weights(rule: QuadratureSpec, ref: Array) -> Array:
    return jnp.asarray(rule.weights, dtype=ref.dtype)


def integrate(state: FieldState, name: str, *, rule: QuadratureSpec) -> Array:
    r"""Definite integral :math:`\int_\Omega u\,dx` via the quadrature ``rule``.

    ``state`` must have been evaluated at ``rule``'s nodes.
    """
    vals = value(state, name)
    if vals.shape[0] != rule.n_nodes:
        raise ValueError(
            f"integrate: state has {vals.shape[0]} points but rule has "
            f"{rule.n_nodes} nodes; evaluate the field at quadrature_nodes(rule)"
        )
    w = _weights(rule, vals)
    return jnp.tensordot(w, vals, axes=([0], [0]))


def _tangent(curve: Callable[[Array], Array], x: Array) -> Array:
    """Batched curve tangent ``r'(t)`` of shape ``(Q, n)`` at param nodes ``x`` (Q, 1)."""
    jac = jax.vmap(jax.jacfwd(curve))(x)  # (Q, n, 1)
    return jac[..., 0]


def line_integral(
    state: FieldState,
    name: str,
    curve: Callable[[Array], Array],
    *,
    rule: QuadratureSpec,
) -> Array:
    r"""Gradient-theorem line integral :math:`\int_C \nabla u \cdot d\mathbf r`.

    Bit-identical twin of :func:`omnibias.fields.torch.ops.integral.line_integral`;
    see that docstring for the full contract. Evaluates
    :math:`\int_{t_0}^{t_1} \nabla u(r(t))\cdot r'(t)\,dt`, which by the gradient
    theorem equals ``u(r(t_1)) - u(r(t_0))``. ``state`` must hold ``name``
    evaluated at ``curve(quadrature_nodes(rule))``; the gradient is closed form,
    the tangent is exact autodiff, and the integral is Gauss-Legendre quadrature.
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
    integrand = (g * tangent).sum(axis=-1)  # (Q,)
    w = _weights(rule, integrand)
    return jnp.tensordot(w, integrand, axes=([0], [0]))


__all__ = ["integrate", "line_integral", "quadrature_nodes"]
