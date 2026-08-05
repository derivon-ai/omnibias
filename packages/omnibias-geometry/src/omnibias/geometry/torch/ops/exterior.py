# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Exterior calculus on differential forms (torch).

Operators: the exterior derivative ``d``, the wedge product, the Hodge star, and
the codifferential / Hodge-Laplacian on 0-forms. Forms are addressed by sorted
index tuples; component values come from closed-form field derivatives, metric
factors from the manifold. See ``GEOMETRY_DERIVATIONS.md`` and the de Rham
identity tests.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from omnibias.fields.torch.ops.basic import gradient, mixed_partial
from omnibias.fields.torch.ops.high_order import hessian
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
from omnibias.geometry.torch.ops.connection import (
    christoffel,
    inverse_metric,
    metric_density_divergence,
    sqrt_det_metric,
)
from torch import Tensor

if TYPE_CHECKING:  # pragma: no cover
    from omnibias.fields._core.state import FieldState
    from omnibias.geometry._core.manifold import ManifoldSpec

# Re-export the backend-agnostic wedge / interior_product for convenience.
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
) -> dict[tuple[int, ...], Tensor]:
    r"""Lie derivative :math:`\mathcal L_X\omega` (see :mod:`...._core.exterior_core`)."""
    return lie_derivative_components(state, vector_names, form)


def codifferential(
    state: FieldState, form: DifferentialForm, manifold: ManifoldSpec,
) -> dict[tuple[int, ...], Tensor]:
    r"""Codifferential :math:`\delta\omega` of a general ``k``-form (``k >= 1``)."""
    ginv = inverse_metric(state.coords, manifold)
    gamma = christoffel(state.coords, manifold)
    return codifferential_components(state, form, ginv, gamma)


def _from_field(state: FieldState, name: str, axis: int) -> Tensor:
    from omnibias.fields.torch.ops.basic import derivative

    return derivative(state, name, axis=axis, order=1)


def exterior_derivative(
    state: FieldState, form: DifferentialForm,
) -> dict[tuple[int, ...], Tensor]:
    r"""Exterior derivative ``d`` of a name-form -> evaluated ``(k+1)``-form.

    .. math::

        (d\alpha)_{i_0\dots i_k} = \sum_{p=0}^{k} (-1)^p\,
            \partial_{i_p}\,\alpha_{i_0\dots\widehat{i_p}\dots i_k}.

    The component partials are closed-form field first derivatives.
    """
    d = form.dim
    k = form.degree
    out: dict[tuple[int, ...], Tensor] = {}
    for j in sorted_index_sets(d, k + 1):
        acc: Tensor | None = None
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


def d_squared_scalar(state: FieldState, name: str) -> dict[tuple[int, ...], Tensor]:
    r"""The 2-form ``d(df)`` for a 0-form; identically zero (``d^2 = 0``).

    Computed from closed-form second mixed partials, so the test exercises the
    symmetry :math:`\partial_i\partial_j f = \partial_j\partial_i f`.
    """
    d = state.coordinate_spec.ndim
    out: dict[tuple[int, ...], Tensor] = {}
    for i, jx in sorted_index_sets(d, 2):
        dij = mixed_partial(state, name, (i, jx), (1, 1))
        dji = mixed_partial(state, name, (jx, i), (1, 1))
        out[(i, jx)] = dij - dji
    return out


def hodge_star(
    values: dict[tuple[int, ...], Any],
    degree: int,
    coords: Tensor,
    manifold: ManifoldSpec,
) -> dict[tuple[int, ...], Tensor]:
    r"""Hodge star ``*`` of an evaluated ``k``-form -> ``(d-k)``-form.

    Uses :math:`(*\alpha)_J = \sqrt{|g|}\,\varepsilon(I, J)\sum_M
    \det\big(g^{-1}_{I,M}\big)\,\alpha_M`, with ``I`` the complement of ``J``.
    """
    d = manifold.dim
    k = degree
    ginv = inverse_metric(coords, manifold)
    sdet = sqrt_det_metric(coords, manifold)
    iset = sorted_index_sets(d, k)
    out: dict[tuple[int, ...], Tensor] = {}
    for j in sorted_index_sets(d, d - k):
        i_comp = tuple(sorted(set(range(d)) - set(j)))
        eps = permutation_sign(i_comp + j)
        acc: Tensor | None = None
        for m in iset:
            am = values.get(m)
            if am is None:
                continue
            if k == 0:
                detsub = torch.ones_like(sdet)
            else:
                sub = ginv[:, i_comp, :][:, :, m]  # (B, k, k)
                detsub = torch.linalg.det(sub)
            term = detsub * am
            acc = term if acc is None else acc + term
        if acc is not None:
            out[j] = (eps * sdet) * acc
    return out


def hodge_laplacian_scalar(
    state: FieldState, name: str, manifold: ManifoldSpec,
) -> Tensor:
    r"""Hodge Laplacian on a 0-form: :math:`\Delta f = \delta d f = -\Delta_g f`.

    Computed independently of the Christoffel-based ``laplace_beltrami`` via the
    metric-density divergence, so the two agree only if the connection is
    consistent (a de Rham cross-check).
    """
    axes = tuple(state.coordinate_spec.axes)
    coords = state.coords
    grad = gradient(state, name, axes=axes)
    hess = hessian(state, name, axes=axes)
    ginv = inverse_metric(coords, manifold)
    a_vec = metric_density_divergence(coords, manifold)
    lap_g = (
        torch.einsum("bj,bj->b", a_vec, grad)
        + torch.einsum("bij,bij->b", ginv, hess)
    )
    return -lap_g


def codifferential_exact_scalar(
    state: FieldState, name: str, manifold: ManifoldSpec,
) -> Tensor:
    r"""Codifferential of the exact 1-form ``df``: :math:`\delta(df)`.

    Alias of :func:`hodge_laplacian_scalar` (on 0-forms ``d\delta f = 0`` so the
    Hodge Laplacian is just ``\delta d``).
    """
    return hodge_laplacian_scalar(state, name, manifold)
