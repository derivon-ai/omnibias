# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Soft shape / occupancy fields and their closed-form center derivatives (jax).

Bit-identical algorithm to :mod:`omnibias.shape.torch.ops.occupancy`; coefficients come
from the same pure-Python Riccati tower in ``omnibias.core``.
"""

from __future__ import annotations

from collections.abc import Sequence

import jax.numpy as jnp
from jax import Array, nn
from omnibias.core.polynomials import sigmoid_polynomial_coeffs

__all__ = [
    "soft_box",
    "soft_box_grad",
    "soft_box_hessian",
    "soft_disk",
    "soft_interval",
    "soft_polytope",
]


def _sigmoid_deriv(s: Array, order: int) -> Array:
    coeffs = sigmoid_polynomial_coeffs(order)
    out = jnp.full_like(s, float(coeffs[-1]))
    for k in range(len(coeffs) - 2, -1, -1):
        out = out * s + float(coeffs[k])
    return out


def _side_d(side: float | Array, d: int) -> float | Array:
    if isinstance(side, Array) and side.ndim > 0:
        return side[d]
    return side


def _interval_sigmoids(
    t: Array, center: Array, side: float | Array, beta: float | Array
) -> tuple[Array, Array]:
    half = 0.5 * side
    s_lo = nn.sigmoid(beta * (t - center + half))
    s_hi = nn.sigmoid(beta * (t - center - half))
    return s_lo, s_hi


def soft_interval(
    t: Array, center: float | Array, side: float | Array, beta: float | Array
) -> Array:
    r"""1-D soft interval indicator (a difference of two sigmoids), in ``(0, 1)``."""
    center_a = jnp.asarray(center, dtype=t.dtype)
    s_lo, s_hi = _interval_sigmoids(t, center_a, side, beta)
    return s_lo - s_hi


def _axis_factors(
    axis: Array, centers_axis: Array, side: float | Array, beta: float | Array
) -> tuple[Array, Array, Array]:
    t = axis.reshape(1, -1)
    c = centers_axis.reshape(-1, 1)
    s_lo, s_hi = _interval_sigmoids(t, c, side, beta)
    b = s_lo - s_hi
    d1 = -beta * (_sigmoid_deriv(s_lo, 1) - _sigmoid_deriv(s_hi, 1))
    d2 = (beta * beta) * (_sigmoid_deriv(s_lo, 2) - _sigmoid_deriv(s_hi, 2))
    return b, d1, d2


def _grid_product(per_axis: Sequence[Array]) -> Array:
    k = per_axis[0].shape[0]
    d = len(per_axis)
    out: Array | None = None
    for axis_idx, fac in enumerate(per_axis):
        shape = [k] + [1] * d
        shape[axis_idx + 1] = fac.shape[1]
        f = fac.reshape(shape)
        out = f if out is None else out * f
    assert out is not None
    return out


def _all_axis_factors(
    axes: Sequence[Array], centers: Array, side: float | Array, beta: float | Array
) -> list[tuple[Array, Array, Array]]:
    return [
        _axis_factors(axis, centers[:, d], _side_d(side, d), beta) for d, axis in enumerate(axes)
    ]


def soft_box(
    axes: Sequence[Array], centers: Array, side: float | Array, beta: float | Array
) -> Array:
    r"""Separable soft box occupancy, shape ``(K, n_0, ..., n_{D-1})``."""
    parts = _all_axis_factors(axes, centers, side, beta)
    return _grid_product([p[0] for p in parts])


def soft_box_grad(
    axes: Sequence[Array], centers: Array, side: float | Array, beta: float | Array
) -> Array:
    r"""Closed-form gradient of the box occupancy, shape ``(K, D, n_0, ..., n_{D-1})``."""
    parts = _all_axis_factors(axes, centers, side, beta)
    d = len(axes)
    grads = []
    for grad_axis in range(d):
        per_axis = [parts[a][1] if a == grad_axis else parts[a][0] for a in range(d)]
        grads.append(_grid_product(per_axis))
    return jnp.stack(grads, axis=1)


def soft_box_hessian(
    axes: Sequence[Array], centers: Array, side: float | Array, beta: float | Array
) -> Array:
    r"""Closed-form per-shape occupancy Hessian, shape ``(K, D, D, n_0, ..., n_{D-1})``."""
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
        rows.append(jnp.stack(row, axis=1))
    return jnp.stack(rows, axis=1)


def _mesh(axes: Sequence[Array]) -> list[Array]:
    return list(jnp.meshgrid(*axes, indexing="ij"))


def soft_disk(
    axes: Sequence[Array], centers: Array, radius: float | Array, beta: float | Array
) -> Array:
    r"""Soft ball occupancy ``sigmoid(beta (radius^2 - ||x - center||^2))``, shape ``(K, ...)``."""
    mesh = _mesh(axes)
    k = centers.shape[0]
    dist2 = jnp.zeros((k, *mesh[0].shape), dtype=centers.dtype)
    for d, g in enumerate(mesh):
        diff = g.reshape((1, *g.shape)) - centers[:, d].reshape((-1,) + (1,) * g.ndim)
        dist2 = dist2 + diff * diff
    r2 = radius * radius
    return nn.sigmoid(beta * (r2 - dist2))


def soft_polytope(
    axes: Sequence[Array], normals: Array, offsets: Array, beta: float | Array
) -> Array:
    r"""Soft convex polytope occupancy for constraints ``normals[i] . x <= offsets[i]``."""
    mesh = _mesh(axes)
    occ = jnp.ones(mesh[0].shape, dtype=offsets.dtype)
    for i in range(normals.shape[0]):
        lin = jnp.zeros(mesh[0].shape, dtype=offsets.dtype)
        for d, g in enumerate(mesh):
            lin = lin + normals[i, d] * g
        occ = occ * nn.sigmoid(beta * (offsets[i] - lin))
    return occ
