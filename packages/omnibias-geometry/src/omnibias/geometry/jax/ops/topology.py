# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""de Rham / characteristic-number topology on manifolds (jax).

Bit-identical twin of :mod:`omnibias.geometry.torch.ops.topology`; see that
module for the full operator and honesty documentation.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import jax.numpy as jnp
from jax import Array
from omnibias.fields._core.quadrature import QuadratureSpec
from omnibias.fields.jax.ops.basic import derivative, value
from omnibias.geometry._core.forms import DifferentialForm
from omnibias.geometry.jax.ops.connection import (
    christoffel,
    scalar_curvature,
    sqrt_det_metric,
)
from omnibias.geometry.jax.ops.exterior import hodge_laplacian_scalar

if TYPE_CHECKING:  # pragma: no cover
    from omnibias.fields._core.state import FieldState
    from omnibias.geometry._core.manifold import ManifoldSpec

_TWO_PI = 2.0 * math.pi
_FOUR_PI = 4.0 * math.pi

#: Christoffel magnitude below which the metric is treated as constant (flat
#: chart) for the ``k >= 1`` Hodge Laplacian.
CONSTANT_METRIC_TOL = 1e-9


def _weights(rule: QuadratureSpec, ref: Array) -> Array:
    return jnp.asarray(rule.weights, dtype=ref.dtype)


def _require_nodes(coords: Array, rule: QuadratureSpec, op: str) -> None:
    if coords.shape[0] != rule.n_nodes:
        raise ValueError(
            f"{op}: state/coords has {coords.shape[0]} points but rule has "
            f"{rule.n_nodes} nodes; evaluate at quadrature_nodes(rule) first"
        )


def _is_constant_metric(coords: Array, manifold: ManifoldSpec) -> bool:
    """Whether the Christoffel symbols vanish (constant / Cartesian-flat chart)."""
    gamma = christoffel(coords, manifold)
    return bool(jnp.max(jnp.abs(gamma)) <= CONSTANT_METRIC_TOL)


def hodge_laplacian(
    state: FieldState, form: DifferentialForm, manifold: ManifoldSpec,
) -> dict[tuple[int, ...], Array]:
    r"""Hodge-de Rham Laplacian :math:`\Delta\omega=(d\delta+\delta d)\omega`."""
    if form.degree == 0:
        name = form.comps.get(())
        if name is None:
            return {}
        return {(): hodge_laplacian_scalar(state, name, manifold)}
    if not _is_constant_metric(state.coords, manifold):
        raise NotImplementedError(
            "hodge_laplacian on a k>=1 form is implemented only for a constant "
            "(flat / Cartesian) metric, where it is the componentwise Laplacian. "
            "A curved-metric k-form Laplacian needs the Weitzenboeck curvature "
            "term; use the scalar (k=0) op on curved manifolds. See "
            "GEOMETRY_DERIVATIONS.md (de Rham / Hodge Laplacian)."
        )
    out: dict[tuple[int, ...], Array] = {}
    for idx, name in form.comps.items():
        out[idx] = hodge_laplacian_scalar(state, name, manifold)
    return out


def hodge_laplacian_matrix(
    state: FieldState,
    basis: list[DifferentialForm],
    manifold: ManifoldSpec,
    *,
    rule: QuadratureSpec,
) -> Array:
    r"""Gram matrix :math:`M_{ij}=\langle b_i,\Delta b_j\rangle_{L^2}` in a form basis."""
    _require_nodes(state.coords, rule, "hodge_laplacian_matrix")
    w = _weights(rule, state.coords)
    lap = [hodge_laplacian(state, b, manifold) for b in basis]
    vals = [{idx: value(state, nm) for idx, nm in b.comps.items()} for b in basis]
    n = len(basis)
    rows: list[Array] = []
    for i in range(n):
        row: list[Array] = []
        for j in range(n):
            acc = jnp.zeros((), dtype=w.dtype)
            for idx, vi in vals[i].items():
                lj = lap[j].get(idx)
                if lj is None:
                    continue
                acc = acc + jnp.tensordot(w, vi * lj, axes=([0], [0]))
            row.append(acc)
        rows.append(jnp.stack(row))
    return jnp.stack(rows)


def betti_number(laplacian_matrix: Array, *, tol: float = 1e-8) -> int:
    r"""Betti number = dim of the harmonic space = nullity of the Hodge Laplacian."""
    m = 0.5 * (laplacian_matrix + jnp.swapaxes(laplacian_matrix, -1, -2))
    s = jnp.linalg.svd(m, compute_uv=False)
    if s.size == 0:
        return 0
    smax = float(s[0])
    thresh = tol * max(smax, 1.0)
    rank = int(jnp.sum(s > thresh))
    return int(laplacian_matrix.shape[0]) - rank


def harmonic_projection(
    laplacian_matrix: Array, coeffs: Array, *, tol: float = 1e-8,
) -> Array:
    r"""Project a basis-coefficient vector onto the harmonic (kernel) subspace."""
    m = 0.5 * (laplacian_matrix + jnp.swapaxes(laplacian_matrix, -1, -2))
    evals, evecs = jnp.linalg.eigh(m)
    wmax = float(jnp.max(jnp.abs(evals))) if evals.size else 0.0
    thresh = tol * max(wmax, 1.0)
    mask = jnp.abs(evals) <= thresh
    q = evecs[:, mask]
    return q @ (jnp.swapaxes(q, -1, -2) @ coeffs)


def winding_number(
    state: FieldState, name: str, *, axis: int, rule: QuadratureSpec,
) -> Array:
    r"""Winding number (degree) of a circle map :math:`S^1\to S^1`."""
    _require_nodes(state.coords, rule, "winding_number")
    dphi = derivative(state, name, axis=axis, order=1)
    w = _weights(rule, dphi)
    total = jnp.tensordot(w, dphi, axes=([0], [0]))
    return total / _TWO_PI


def map_degree(
    state: FieldState, names: tuple[str, str, str], *, rule: QuadratureSpec,
) -> Array:
    r"""Topological degree of a map :math:`M^2\to S^2`."""
    _require_nodes(state.coords, rule, "map_degree")
    n = jnp.stack([value(state, nm) for nm in names], axis=-1)
    d0 = jnp.stack([derivative(state, nm, axis=0, order=1) for nm in names], axis=-1)
    d1 = jnp.stack([derivative(state, nm, axis=1, order=1) for nm in names], axis=-1)
    integrand = jnp.sum(n * jnp.cross(d0, d1), axis=-1)
    w = _weights(rule, integrand)
    total = jnp.tensordot(w, integrand, axes=([0], [0]))
    return total / _FOUR_PI


def gauss_bonnet_euler(
    coords: Array, manifold: ManifoldSpec, *, rule: QuadratureSpec,
) -> Array:
    r"""Euler characteristic of a closed surface via Gauss-Bonnet."""
    if manifold.dim != 2:
        raise ValueError(
            f"gauss_bonnet_euler is a surface (dim=2) integral, got dim={manifold.dim}"
        )
    _require_nodes(coords, rule, "gauss_bonnet_euler")
    gauss_k = 0.5 * scalar_curvature(coords, manifold)
    area_element = sqrt_det_metric(coords, manifold)
    integrand = gauss_k * area_element
    w = _weights(rule, integrand)
    total = jnp.tensordot(w, integrand, axes=([0], [0]))
    return total / _TWO_PI


__all__ = [
    "CONSTANT_METRIC_TOL",
    "betti_number",
    "gauss_bonnet_euler",
    "harmonic_projection",
    "hodge_laplacian",
    "hodge_laplacian_matrix",
    "map_degree",
    "winding_number",
]
