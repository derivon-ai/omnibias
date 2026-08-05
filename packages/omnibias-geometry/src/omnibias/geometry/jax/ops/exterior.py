# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Exterior calculus on differential forms (jax).

Bit-identical twin of :mod:`omnibias.geometry.torch.ops.exterior`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
from jax import Array
from omnibias.fields.jax.ops.basic import gradient, mixed_partial
from omnibias.fields.jax.ops.high_order import hessian
from omnibias.geometry._core.exterior_core import (
    codifferential_components,
    lie_derivative_components,
)
from omnibias.geometry._core.forms import (
    DifferentialForm,
    interior_product,
    permutation_sign,
    sorted_index_sets,
    wedge,
)
from omnibias.geometry.jax.ops.connection import (
    christoffel,
    inverse_metric,
    metric_density_divergence,
    sqrt_det_metric,
)

if TYPE_CHECKING:  # pragma: no cover
    from omnibias.fields._core.state import FieldState
    from omnibias.geometry._core.manifold import ManifoldSpec

__all__ = [
    "codifferential",
    "codifferential_exact_scalar",
    "d_squared_scalar",
    "exterior_derivative",
    "hodge_laplacian_scalar",
    "hodge_star",
    "interior_product",
    "lie_derivative",
    "wedge",
]


def lie_derivative(
    state: FieldState, vector_names: tuple[str, ...], form: DifferentialForm,
) -> dict[tuple[int, ...], Array]:
    r"""Lie derivative :math:`\mathcal L_X\omega` (see :mod:`...._core.exterior_core`)."""
    return lie_derivative_components(state, vector_names, form)


def codifferential(
    state: FieldState, form: DifferentialForm, manifold: ManifoldSpec,
) -> dict[tuple[int, ...], Array]:
    r"""Codifferential :math:`\delta\omega` of a general ``k``-form (``k >= 1``)."""
    ginv = inverse_metric(state.coords, manifold)
    gamma = christoffel(state.coords, manifold)
    return codifferential_components(state, form, ginv, gamma)


def _from_field(state: FieldState, name: str, axis: int) -> Array:
    from omnibias.fields.jax.ops.basic import derivative

    return derivative(state, name, axis=axis, order=1)


def exterior_derivative(
    state: FieldState, form: DifferentialForm,
) -> dict[tuple[int, ...], Array]:
    r"""Exterior derivative ``d`` of a name-form -> evaluated ``(k+1)``-form."""
    d = form.dim
    k = form.degree
    out: dict[tuple[int, ...], Array] = {}
    for j in sorted_index_sets(d, k + 1):
        acc: Array | None = None
        for p, ip in enumerate(j):
            rest = j[:p] + j[p + 1:]
            name = form.comps.get(rest)
            if name is None:
                continue
            term = _from_field(state, name, ip)
            contrib = term if p % 2 == 0 else -term
            acc = contrib if acc is None else acc + contrib
        if acc is not None:
            out[j] = acc
    return out


def d_squared_scalar(state: FieldState, name: str) -> dict[tuple[int, ...], Array]:
    r"""The 2-form ``d(df)`` for a 0-form; identically zero (``d^2 = 0``)."""
    d = state.coordinate_spec.ndim
    out: dict[tuple[int, ...], Array] = {}
    for i, jx in sorted_index_sets(d, 2):
        dij = mixed_partial(state, name, (i, jx), (1, 1))
        dji = mixed_partial(state, name, (jx, i), (1, 1))
        out[(i, jx)] = dij - dji
    return out


def hodge_star(
    values: dict[tuple[int, ...], Any],
    degree: int,
    coords: Array,
    manifold: ManifoldSpec,
) -> dict[tuple[int, ...], Array]:
    r"""Hodge star ``*`` of an evaluated ``k``-form -> ``(d-k)``-form."""
    d = manifold.dim
    k = degree
    ginv = inverse_metric(coords, manifold)
    sdet = sqrt_det_metric(coords, manifold)
    iset = sorted_index_sets(d, k)
    out: dict[tuple[int, ...], Array] = {}
    for j in sorted_index_sets(d, d - k):
        i_comp = tuple(sorted(set(range(d)) - set(j)))
        eps = permutation_sign(i_comp + j)
        acc: Array | None = None
        for m in iset:
            am = values.get(m)
            if am is None:
                continue
            if k == 0:
                detsub = jnp.ones_like(sdet)
            else:
                sub = ginv[:, i_comp, :][:, :, m]
                detsub = jnp.linalg.det(sub)
            term = detsub * am
            acc = term if acc is None else acc + term
        if acc is not None:
            out[j] = (eps * sdet) * acc
    return out


def hodge_laplacian_scalar(
    state: FieldState, name: str, manifold: ManifoldSpec,
) -> Array:
    r"""Hodge Laplacian on a 0-form: :math:`\Delta f = \delta d f = -\Delta_g f`."""
    axes = tuple(state.coordinate_spec.axes)
    coords = state.coords
    grad = gradient(state, name, axes=axes)
    hess = hessian(state, name, axes=axes)
    ginv = inverse_metric(coords, manifold)
    a_vec = metric_density_divergence(coords, manifold)
    lap_g = (
        jnp.einsum("bj,bj->b", a_vec, grad)
        + jnp.einsum("bij,bij->b", ginv, hess)
    )
    return -lap_g


def codifferential_exact_scalar(
    state: FieldState, name: str, manifold: ManifoldSpec,
) -> Array:
    r"""Codifferential of the exact 1-form ``df``: :math:`\delta(df)`."""
    return hodge_laplacian_scalar(state, name, manifold)
