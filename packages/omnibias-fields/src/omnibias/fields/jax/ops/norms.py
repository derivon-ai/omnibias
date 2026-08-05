# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Inner products and L2 / Sobolev norms as standalone ops (jax).

Bit-identical twin of :mod:`omnibias.fields.torch.ops.norms`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
from jax import Array
from omnibias.fields._core.quadrature import QuadratureSpec
from omnibias.fields.jax.ops.basic import gradient, value
from omnibias.fields.jax.ops.high_order import spatial_hessian

if TYPE_CHECKING:  # pragma: no cover
    from omnibias.fields._core.state import FieldState


def _weights(rule: QuadratureSpec, ref: Array) -> Array:
    return jnp.asarray(rule.weights, dtype=ref.dtype)


def _integrate_values(vals: Array, rule: QuadratureSpec) -> Array:
    if vals.shape[0] != rule.n_nodes:
        raise ValueError(
            f"state has {vals.shape[0]} points but rule has {rule.n_nodes} nodes"
        )
    w = _weights(rule, vals)
    return jnp.tensordot(w, vals, axes=([0], [0]))


def inner_product(
    state: FieldState,
    name_a: str,
    name_b: str,
    *,
    rule: QuadratureSpec,
    weight: str | None = None,
) -> Array:
    r"""Weighted real inner product :math:`\langle a, b\rangle_w
    = \int_\Omega w\,a\,b\,dx`."""
    a = value(state, name_a)
    b = value(state, name_b)
    integrand = a * b
    if weight is not None:
        integrand = integrand * value(state, weight)
    return _integrate_values(integrand, rule)


def l2_norm(state: FieldState, name: str, *, rule: QuadratureSpec) -> Array:
    r""":math:`\lVert u\rVert_{L^2} = \sqrt{\int_\Omega u^2\,dx}`."""
    sq = _integrate_values(value(state, name) ** 2, rule)
    return jnp.sqrt(sq)


def sobolev_norm(
    state: FieldState,
    name: str,
    *,
    rule: QuadratureSpec,
    k: int = 1,
    weights: tuple[float, ...] | None = None,
) -> Array:
    r"""Sobolev norm :math:`\lVert u\rVert_{H^k}` for ``k`` in ``{0, 1, 2}``.

    See :func:`omnibias.fields.torch.ops.norms.sobolev_norm` for the definition.
    """
    if k < 0:
        raise ValueError(f"k must be >= 0, got {k}")
    if k > 2:
        raise NotImplementedError(
            "sobolev_norm currently supports k in {0, 1, 2}; "
            f"got k={k}. Higher orders need the full multi-index expansion."
        )
    coeffs = (1.0,) * (k + 1) if weights is None else weights
    if len(coeffs) != k + 1:
        raise ValueError(f"weights must have length k+1 = {k + 1}, got {len(coeffs)}")

    total = coeffs[0] * _integrate_values(value(state, name) ** 2, rule)
    if k >= 1:
        g = gradient(state, name)
        total = total + coeffs[1] * _integrate_values((g ** 2).sum(axis=-1), rule)
    if k >= 2:
        h = spatial_hessian(state, name)
        total = total + coeffs[2] * _integrate_values((h ** 2).sum(axis=(-2, -1)), rule)
    return jnp.sqrt(total)


__all__ = ["inner_product", "l2_norm", "sobolev_norm"]
