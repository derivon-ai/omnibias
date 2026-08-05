# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Soft shape / occupancy fields and their closed-form center derivatives (torch).

A 1-D soft interval indicator is a difference of two sigmoids; a soft axis-aligned
box is the separable product across axes. Center derivatives use the shared Riccati
sigmoid tower (polynomials in ``s = sigmoid(u)``), so ``soft_box_grad`` /
``soft_box_hessian`` are closed form -- no autodiff.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from omnibias.core.polynomials import sigmoid_polynomial_coeffs
from torch import Tensor

__all__ = [
    "soft_box",
    "soft_box_grad",
    "soft_box_hessian",
    "soft_disk",
    "soft_interval",
    "soft_polytope",
]


def _sigmoid_deriv(s: Tensor, order: int) -> Tensor:
    r"""Evaluate ``sigma^(order)`` as the Riccati polynomial ``P_order(s)`` at ``s = sigmoid(u)``."""
    coeffs = sigmoid_polynomial_coeffs(order)
    out = torch.full_like(s, float(coeffs[-1]))
    for k in range(len(coeffs) - 2, -1, -1):
        out = out * s + float(coeffs[k])
    return out


def _side_d(side: float | Tensor, d: int) -> float | Tensor:
    if isinstance(side, Tensor) and side.ndim > 0:
        return side[d]
    return side


def _interval_sigmoids(
    t: Tensor, center: Tensor, side: float | Tensor, beta: float | Tensor
) -> tuple[Tensor, Tensor]:
    half = 0.5 * side
    s_lo = torch.sigmoid(beta * (t - center + half))
    s_hi = torch.sigmoid(beta * (t - center - half))
    return s_lo, s_hi


def soft_interval(
    t: Tensor, center: float | Tensor, side: float | Tensor, beta: float | Tensor
) -> Tensor:
    r"""1-D soft interval indicator ``sigmoid(beta (t - c + A/2)) - sigmoid(beta (t - c - A/2))``.

    Lies in ``(0, 1)`` and converges pointwise to the hard indicator of
    ``[c - A/2, c + A/2]`` as ``beta -> inf``. Broadcasts over ``t`` and ``center``.
    """
    center_t = torch.as_tensor(center, dtype=t.dtype, device=t.device)
    s_lo, s_hi = _interval_sigmoids(t, center_t, side, beta)
    return s_lo - s_hi


def _axis_factors(
    axis: Tensor, centers_axis: Tensor, side: float | Tensor, beta: float | Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    r"""Per-axis box factor and its first/second center derivatives, each shape ``(K, n)``."""
    t = axis.reshape(1, -1)
    c = centers_axis.reshape(-1, 1)
    s_lo, s_hi = _interval_sigmoids(t, c, side, beta)
    b = s_lo - s_hi
    d1 = -beta * (_sigmoid_deriv(s_lo, 1) - _sigmoid_deriv(s_hi, 1))
    d2 = (beta * beta) * (_sigmoid_deriv(s_lo, 2) - _sigmoid_deriv(s_hi, 2))
    return b, d1, d2


def _grid_product(per_axis: Sequence[Tensor]) -> Tensor:
    r"""Outer product of per-axis ``(K, n_d)`` factors into ``(K, n_0, ..., n_{D-1})``."""
    k = per_axis[0].shape[0]
    d = len(per_axis)
    out: Tensor | None = None
    for axis_idx, fac in enumerate(per_axis):
        shape = [k] + [1] * d
        shape[axis_idx + 1] = fac.shape[1]
        f = fac.reshape(shape)
        out = f if out is None else out * f
    assert out is not None
    return out


def _all_axis_factors(
    axes: Sequence[Tensor], centers: Tensor, side: float | Tensor, beta: float | Tensor
) -> list[tuple[Tensor, Tensor, Tensor]]:
    return [
        _axis_factors(axis, centers[:, d], _side_d(side, d), beta) for d, axis in enumerate(axes)
    ]


def soft_box(
    axes: Sequence[Tensor], centers: Tensor, side: float | Tensor, beta: float | Tensor
) -> Tensor:
    r"""Separable soft box / hyper-rectangle occupancy, shape ``(K, n_0, ..., n_{D-1})``.

    ``axes`` is a tuple of ``D`` 1-D coordinate vectors; ``centers`` is ``(K, D)``;
    ``side`` is a scalar or ``(D,)``. Axis-aligned square when ``D == 2`` and ``side``
    is scalar.
    """
    parts = _all_axis_factors(axes, centers, side, beta)
    return _grid_product([p[0] for p in parts])


def soft_box_grad(
    axes: Sequence[Tensor], centers: Tensor, side: float | Tensor, beta: float | Tensor
) -> Tensor:
    r"""Closed-form gradient of the box occupancy, shape ``(K, D, n_0, ..., n_{D-1})``.

    ``out[k, d] = d m[k] / d centers[k, d]``.
    """
    parts = _all_axis_factors(axes, centers, side, beta)
    d = len(axes)
    grads = []
    for grad_axis in range(d):
        per_axis = [parts[a][1] if a == grad_axis else parts[a][0] for a in range(d)]
        grads.append(_grid_product(per_axis))
    return torch.stack(grads, dim=1)


def soft_box_hessian(
    axes: Sequence[Tensor], centers: Tensor, side: float | Tensor, beta: float | Tensor
) -> Tensor:
    r"""Closed-form per-shape occupancy Hessian, shape ``(K, D, D, n_0, ..., n_{D-1})``.

    ``out[k, a, b] = d^2 m[k] / d centers[k, a] d centers[k, b]``. Diagonal ``a == b``
    carries the order-2 axis derivative; off-diagonal is a product of two order-1
    axis derivatives (separability).
    """
    parts = _all_axis_factors(axes, centers, side, beta)
    d = len(axes)
    rows = []
    for a in range(d):
        row = []
        for b in range(d):
            if a == b:
                per_axis = [parts[x][2] if x == a else parts[x][0] for x in range(d)]
            else:
                per_axis = [parts[x][1] if x in (a, b) else parts[x][0] for x in range(d)]
            row.append(_grid_product(per_axis))
        rows.append(torch.stack(row, dim=1))
    return torch.stack(rows, dim=1)


def _mesh(axes: Sequence[Tensor]) -> list[Tensor]:
    grids = torch.meshgrid(*axes, indexing="ij")
    return list(grids)


def soft_disk(
    axes: Sequence[Tensor], centers: Tensor, radius: float | Tensor, beta: float | Tensor
) -> Tensor:
    r"""Soft ball occupancy ``sigmoid(beta (radius^2 - ||x - center||^2))``, shape ``(K, ...)``."""
    mesh = _mesh(axes)
    k = centers.shape[0]
    dist2 = torch.zeros((k, *mesh[0].shape), dtype=centers.dtype, device=centers.device)
    for d, g in enumerate(mesh):
        diff = g.unsqueeze(0) - centers[:, d].reshape((-1,) + (1,) * g.ndim)
        dist2 = dist2 + diff * diff
    r2 = radius * radius
    return torch.sigmoid(beta * (r2 - dist2))


def soft_polytope(
    axes: Sequence[Tensor], normals: Tensor, offsets: Tensor, beta: float | Tensor
) -> Tensor:
    r"""Soft convex polytope occupancy for constraints ``normals[i] . x <= offsets[i]``.

    Smooth AND via product of per-constraint sigmoids ``sigmoid(beta (offset - n . x))``;
    returns a single occupancy field of shape ``(n_0, ..., n_{D-1})``. As ``beta -> inf``
    it converges to the indicator of the polytope.
    """
    mesh = _mesh(axes)
    occ = torch.ones(mesh[0].shape, dtype=offsets.dtype, device=offsets.device)
    for i in range(normals.shape[0]):
        lin = torch.zeros(mesh[0].shape, dtype=offsets.dtype, device=offsets.device)
        for d, g in enumerate(mesh):
            lin = lin + normals[i, d] * g
        occ = occ * torch.sigmoid(beta * (offsets[i] - lin))
    return occ
