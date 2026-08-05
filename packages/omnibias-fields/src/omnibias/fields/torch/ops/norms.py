# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Inner products and L2 / Sobolev norms as standalone ops (torch).

All quantities are quadrature integrals over the box domain implied by the rule;
``state`` must be evaluated at the rule's nodes (see
:func:`omnibias.fields.torch.ops.integral.integrate`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from omnibias.fields._core.quadrature import QuadratureSpec
from omnibias.fields.torch.ops.basic import gradient, value
from omnibias.fields.torch.ops.high_order import spatial_hessian
from torch import Tensor

if TYPE_CHECKING:  # pragma: no cover
    from omnibias.fields._core.state import FieldState


def _weights(rule: QuadratureSpec, ref: Tensor) -> Tensor:
    return torch.as_tensor(rule.weights, dtype=ref.dtype, device=ref.device)


def _integrate_values(vals: Tensor, rule: QuadratureSpec) -> Tensor:
    if vals.shape[0] != rule.n_nodes:
        raise ValueError(
            f"state has {vals.shape[0]} points but rule has {rule.n_nodes} nodes"
        )
    w = _weights(rule, vals)
    return torch.tensordot(w, vals, dims=([0], [0]))


def inner_product(
    state: FieldState,
    name_a: str,
    name_b: str,
    *,
    rule: QuadratureSpec,
    weight: str | None = None,
) -> Tensor:
    r"""Weighted real inner product :math:`\langle a, b\rangle_w
    = \int_\Omega w\,a\,b\,dx`.

    Parameters
    ----------
    state
        Field state evaluated at the quadrature nodes.
    name_a, name_b
        Component names of the two factors.
    rule
        Quadrature rule.
    weight
        Optional component name of a positive weight field ``w`` (default
        ``None`` means ``w = 1``).
    """
    a = value(state, name_a)
    b = value(state, name_b)
    integrand = a * b
    if weight is not None:
        integrand = integrand * value(state, weight)
    return _integrate_values(integrand, rule)


def l2_norm(state: FieldState, name: str, *, rule: QuadratureSpec) -> Tensor:
    r""":math:`\lVert u\rVert_{L^2} = \sqrt{\int_\Omega u^2\,dx}`."""
    sq = _integrate_values(value(state, name) ** 2, rule)
    return torch.sqrt(sq)


def sobolev_norm(
    state: FieldState,
    name: str,
    *,
    rule: QuadratureSpec,
    k: int = 1,
    weights: tuple[float, ...] | None = None,
) -> Tensor:
    r"""Sobolev norm :math:`\lVert u\rVert_{H^k}`.

    .. math::

        \lVert u\rVert_{H^k}^2 = \sum_{m=0}^{k} c_m
            \sum_{|\alpha|=m} \int_\Omega (\partial^\alpha u)^2\,dx,

    where order ``m=0`` is :math:`\int u^2`, ``m=1`` is the squared spatial
    gradient, and ``m=2`` is the squared spatial Hessian (Frobenius). Every
    order is taken over the *spatial* axes only (the time axis, if any, is
    excluded), so on a spacetime field this is a consistent spatial Sobolev
    seminorm sum rather than a mixed space-time norm. ``k`` must be in
    ``{0, 1, 2}`` (higher orders raise :class:`NotImplementedError`). ``weights``
    supplies the per-order coefficients :math:`c_m` (default all ones).
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
        g = gradient(state, name)                       # (B, n_spatial)
        total = total + coeffs[1] * _integrate_values((g ** 2).sum(dim=-1), rule)
    if k >= 2:
        h = spatial_hessian(state, name)                 # (B, n_spatial, n_spatial)
        total = total + coeffs[2] * _integrate_values((h ** 2).sum(dim=(-2, -1)), rule)
    return torch.sqrt(total)


__all__ = ["inner_product", "l2_norm", "sobolev_norm"]
