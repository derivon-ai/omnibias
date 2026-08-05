# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Exact multivariate (multi-index) jets via Faà di Bruno (torch).

Bit-identical twin of :mod:`omnibias.jax.jet_mv`; see that module for the full
mathematical exposition. A multivariate jet truncated at total order ``N`` over
``D`` variables stores every Taylor coefficient ``c_alpha = D^alpha f(x0)/alpha!``
densely along the leading axis (canonical
:func:`~omnibias.core.multi_index.multi_indices` order), so a single forward pass
yields all mixed partial derivatives up to order ``N``. Affine maps act per
coefficient; activations use the shifted-power identity with *multivariate*
truncated polynomial multiplication driven by
:func:`omnibias.core.multi_index.multiply_table`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from omnibias.core.multi_index import (
    index_position,
    multi_index_factorial,
    multi_indices,
    multiply_table,
    num_multi_indices,
)
from omnibias.torch.activations.registry import get_activation
from omnibias.torch.jet import _sigma_tower, affine_jet

import torch
from torch import Tensor

if TYPE_CHECKING:  # pragma: no cover
    from omnibias.core.spec import ActivationSpec


def compose_jet_mv(
    u_jet: Tensor, sigma_tower: Tensor, dim: int, order: int
) -> Tensor:
    """Compose an elementwise activation onto a multivariate jet: ``b = sigma(u)``.

    ``u_jet`` has shape ``(M, ...)`` with ``M = num_multi_indices(dim, order)``;
    ``sigma_tower[k] = sigma^(k)(u_jet[0])`` for ``k = 0..order``. Returns the
    jet of ``sigma(u)``.
    """
    u_jet = torch.as_tensor(u_jet)
    sigma_tower = torch.as_tensor(sigma_tower)
    m = num_multi_indices(dim, order)
    if u_jet.shape[0] != m:
        raise ValueError(
            f"u_jet has {u_jet.shape[0]} rows but dim={dim}, order={order} "
            f"requires {m}"
        )
    if sigma_tower.shape[0] != order + 1:
        raise ValueError(
            f"sigma_tower order {sigma_tower.shape[0] - 1} must equal jet order "
            f"{order}"
        )

    table = multiply_table(dim, order)
    zero = torch.zeros_like(u_jet[0])
    w = [zero] + [u_jet[g] for g in range(1, m)]
    p = [torch.ones_like(u_jet[0])] + [zero for _ in range(m - 1)]
    result = [sigma_tower[0] * p[0]] + [zero for _ in range(m - 1)]
    fact = 1.0
    for k in range(1, order + 1):
        fact *= k
        acc = [zero for _ in range(m)]
        for g, a, b in table:
            acc[g] = acc[g] + p[a] * w[b]
        p = acc
        dk = sigma_tower[k] / fact
        for g in range(m):
            result[g] = result[g] + dk * p[g]
    return torch.stack(result, dim=0)


def jet_multiply(a: Tensor, b: Tensor, dim: int, order: int) -> Tensor:
    r"""Truncated product of two multivariate jets: ``jet(a) * jet(b)``.

    Given the jets of two analytic functions about the *same* base point, returns
    the jet of their pointwise product via the multivariate Cauchy product

    .. math::

        (a\,b)_\gamma = \sum_{\alpha + \beta = \gamma} a_\alpha\, b_\beta,

    truncated at total order ``order`` (the same
    :func:`~omnibias.core.multi_index.multiply_table` that drives
    :func:`compose_jet_mv`). Both ``a`` and ``b`` carry
    ``M = num_multi_indices(dim, order)`` leading rows; any trailing shape
    broadcasts, so a scalar mask ``(M,)`` multiplies a vector field ``(M, C)``
    component-wise. This is the jet-level Leibniz rule and is what keeps a
    product ansatz such as ``u = g + b * net`` closed-form to arbitrary order.
    """
    a = torch.as_tensor(a)
    b = torch.as_tensor(b)
    m = num_multi_indices(dim, order)
    if a.shape[0] != m:
        raise ValueError(
            f"a has {a.shape[0]} rows but dim={dim}, order={order} requires {m}"
        )
    if b.shape[0] != m:
        raise ValueError(
            f"b has {b.shape[0]} rows but dim={dim}, order={order} requires {m}"
        )
    table = multiply_table(dim, order)
    sample = a[0] * b[0]
    out = [torch.zeros_like(sample) for _ in range(m)]
    for g, al, be in table:
        out[g] = out[g] + a[al] * b[be]
    return torch.stack(out, dim=0)


def affine_jet_mv(z_jet: Tensor, W: Tensor, b: Tensor | None = None) -> Tensor:
    """Push a multivariate jet through an affine map ``u = W z + b``.

    Identical per-coefficient action to :func:`omnibias.torch.affine_jet`: the
    map is degree-1 in the perturbation, so it acts row-wise with the bias on the
    constant coefficient (row 0) only.
    """
    return affine_jet(z_jet, W, b)


def layer_jet_mv(
    z_jet: Tensor,
    W: Tensor,
    b: Tensor | None,
    spec: ActivationSpec[Tensor] | str,
    dim: int,
    order: int,
) -> Tensor:
    """Push a multivariate jet through one ``sigma(W z + b)`` layer."""
    z_jet = torch.as_tensor(z_jet)
    resolved = get_activation(spec)
    u_jet = affine_jet(z_jet, W, b)
    sigma_tower = _sigma_tower(resolved, u_jet[0], order)
    return compose_jet_mv(u_jet, sigma_tower, dim, order)


def identity_jet(x0: Tensor, order: int) -> Tensor:
    """Multivariate jet of the identity map ``x(delta) = x0 + delta``.

    Returns shape ``(M, D)``: row 0 is ``x0`` and each unit-multi-index row
    ``e_i`` is the standard basis vector ``e_i``; all other rows are zero.
    """
    x0 = torch.as_tensor(x0)
    if x0.ndim != 1:
        raise ValueError(f"x0 must be a 1-D vector, got shape {tuple(x0.shape)}")
    dim = int(x0.shape[0])
    if order < 0:
        raise ValueError(f"order must be >= 0, got {order}")
    pos = index_position(dim, order)
    m = num_multi_indices(dim, order)
    rows = [torch.zeros((dim,), dtype=x0.dtype, device=x0.device) for _ in range(m)]
    rows[pos[(0,) * dim]] = x0
    if order >= 1:
        eye = torch.eye(dim, dtype=x0.dtype, device=x0.device)
        for i in range(dim):
            e = tuple(1 if j == i else 0 for j in range(dim))
            rows[pos[e]] = eye[i]
    return torch.stack(rows, dim=0)


def mlp_jet_mv(
    x0: Tensor,
    layers: Sequence[tuple[Tensor, Tensor | None, ActivationSpec[Tensor] | str | None]],
    order: int,
) -> Tensor:
    """Exact multivariate Taylor jet of a deep MLP around ``x0``.

    ``layers`` is a sequence of ``(W, b, spec)``; ``spec=None`` is a pure affine
    readout. Returns a jet of shape ``(M, C)`` whose row ``i`` is
    ``D^alpha f(x0) / alpha!`` for ``alpha = multi_indices(D, order)[i]``. Use
    :func:`jet_partials`, :func:`jet_gradient` or :func:`jet_hessian` to read
    out raw derivatives.
    """
    x0 = torch.as_tensor(x0)
    dim = int(x0.shape[-1])
    jet = identity_jet(x0, order)
    for W, b, spec in layers:
        if spec is None:
            jet = affine_jet(jet, W, b)
        else:
            jet = layer_jet_mv(jet, W, b, spec, dim, order)
    return jet


def jet_partials(jet: Tensor, dim: int, order: int) -> dict[tuple[int, ...], Tensor]:
    """Convert a jet to raw partial derivatives ``{alpha: D^alpha f(x0)}``.

    Each coefficient row is scaled by ``alpha!`` (``D^alpha f = alpha! c_alpha``).
    """
    jet = torch.as_tensor(jet)
    idx = multi_indices(dim, order)
    if jet.shape[0] != len(idx):
        raise ValueError(
            f"jet has {jet.shape[0]} rows but dim={dim}, order={order} "
            f"requires {len(idx)}"
        )
    return {alpha: jet[i] * multi_index_factorial(alpha) for i, alpha in enumerate(idx)}


def jet_gradient(jet: Tensor, dim: int, order: int) -> Tensor:
    """Gradient ``d f / d x_i`` from a jet, shape ``(D, ...)``. Needs ``order >= 1``."""
    if order < 1:
        raise ValueError(f"gradient needs order >= 1, got {order}")
    jet = torch.as_tensor(jet)
    pos = index_position(dim, order)
    rows = []
    for i in range(dim):
        e = tuple(1 if j == i else 0 for j in range(dim))
        rows.append(jet[pos[e]])  # alpha! = 1 for a unit multi-index
    return torch.stack(rows, dim=0)


def jet_hessian(jet: Tensor, dim: int, order: int) -> Tensor:
    """Hessian ``d^2 f / d x_i d x_j`` from a jet, shape ``(D, D, ...)``.

    Needs ``order >= 2``. ``H[i, j]`` is the coefficient of ``alpha = e_i + e_j``
    rescaled by ``alpha!`` (2 on the diagonal, 1 off-diagonal).
    """
    if order < 2:
        raise ValueError(f"hessian needs order >= 2, got {order}")
    jet = torch.as_tensor(jet)
    pos = index_position(dim, order)
    rows = []
    for i in range(dim):
        row = []
        for j in range(dim):
            alpha = tuple(
                (1 if k == i else 0) + (1 if k == j else 0) for k in range(dim)
            )
            row.append(jet[pos[alpha]] * multi_index_factorial(alpha))
        rows.append(torch.stack(row, dim=0))
    return torch.stack(rows, dim=0)


__all__ = [
    "affine_jet_mv",
    "compose_jet_mv",
    "identity_jet",
    "jet_gradient",
    "jet_hessian",
    "jet_multiply",
    "jet_partials",
    "layer_jet_mv",
    "mlp_jet_mv",
]
