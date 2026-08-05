# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""de Rham / characteristic-number topology on manifolds (torch).

This module realises the *differential-form* (de Rham) slice of algebraic
topology on the closed-form substrate; it is the twin of
:mod:`omnibias.geometry.jax.ops.topology`.

Operators
---------
- :func:`hodge_laplacian` -- the Hodge-de Rham Laplacian
  :math:`\Delta = d\delta + \delta d` on a ``k``-form. On 0-forms it is the
  curved-manifold scalar Laplacian (delegated to
  :func:`...exterior.hodge_laplacian_scalar`); on ``k >= 1`` forms it is the
  *constant-metric* (Levi-Civita connection with vanishing Christoffel symbols)
  componentwise Laplacian ``(\Delta\omega)_I = -g^{mn}\partial_m\partial_n
  \omega_I``. Genuinely curved ``k``-form Laplacians (which pick up the
  Weitzenboeck curvature term) raise :class:`NotImplementedError` -- an honest
  scope boundary, since composing ``d`` and ``\delta`` requires the intermediate
  form to be re-named.
- :func:`hodge_laplacian_matrix` / :func:`betti_number` /
  :func:`harmonic_projection` -- assemble the Hodge Laplacian in a finite basis
  of forms and read off the harmonic (kernel) dimension = Betti number.
- :func:`winding_number` -- degree of a circle map ``S^1 -> S^1``.
- :func:`map_degree` -- degree of a map ``M^2 -> S^2`` (the normalised
  pullback of the target area form).
- :func:`gauss_bonnet_euler` -- Euler characteristic of a closed surface via
  Gauss-Bonnet ``\chi = (1/2\pi)\int K\,dA``.

Honesty
-------
Field-component derivatives are closed-form (sigma-tower); metric quantities
(curvature, ``\sqrt{|g|}``) are autodiff-exact for analytic metrics; the Betti
number and characteristic-number integrals are *numerical* (quadrature + a
rank/nullity count), certifiable by an :class:`omnibias.core.verified.Interval`
enclosure of the integral. Combinatorial topology (homotopy groups
:math:`\pi_n`, persistent homology, simplicial :math:`\mathbb Z`-homology,
Smith normal form) is **out of thesis** -- see ``docs/scope-and-guarantees.md``.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from omnibias.fields._core.quadrature import QuadratureSpec
from omnibias.fields.torch.ops.basic import derivative, value
from omnibias.geometry._core.forms import DifferentialForm
from omnibias.geometry.torch.ops.connection import (
    christoffel,
    scalar_curvature,
    sqrt_det_metric,
)
from omnibias.geometry.torch.ops.exterior import hodge_laplacian_scalar
from torch import Tensor

if TYPE_CHECKING:  # pragma: no cover
    from omnibias.fields._core.state import FieldState
    from omnibias.geometry._core.manifold import ManifoldSpec

_TWO_PI = 2.0 * math.pi
_FOUR_PI = 4.0 * math.pi

#: Christoffel magnitude below which the metric is treated as constant (flat
#: chart) for the ``k >= 1`` Hodge Laplacian.
CONSTANT_METRIC_TOL = 1e-9


def _weights(rule: QuadratureSpec, ref: Tensor) -> Tensor:
    return torch.as_tensor(rule.weights, dtype=ref.dtype, device=ref.device)


def _require_nodes(coords: Tensor, rule: QuadratureSpec, op: str) -> None:
    if coords.shape[0] != rule.n_nodes:
        raise ValueError(
            f"{op}: state/coords has {coords.shape[0]} points but rule has "
            f"{rule.n_nodes} nodes; evaluate at quadrature_nodes(rule) first"
        )


def _is_constant_metric(coords: Tensor, manifold: ManifoldSpec) -> bool:
    """Whether the Christoffel symbols vanish (constant / Cartesian-flat chart)."""
    gamma = christoffel(coords, manifold)
    return bool(torch.max(torch.abs(gamma)).item() <= CONSTANT_METRIC_TOL)


def hodge_laplacian(
    state: FieldState, form: DifferentialForm, manifold: ManifoldSpec,
) -> dict[tuple[int, ...], Tensor]:
    r"""Hodge-de Rham Laplacian :math:`\Delta\omega=(d\delta+\delta d)\omega`.

    On 0-forms this is the full curved-manifold scalar Laplacian. On ``k >= 1``
    forms it is the constant-metric componentwise Laplacian
    :math:`(\Delta\omega)_I=-g^{mn}\partial_m\partial_n\omega_I`, which is exact
    whenever the Christoffel symbols vanish (a flat / Cartesian chart, e.g. the
    flat torus used for Betti numbers); a genuinely curved chart raises
    :class:`NotImplementedError`.
    """
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
    out: dict[tuple[int, ...], Tensor] = {}
    for idx, name in form.comps.items():
        out[idx] = hodge_laplacian_scalar(state, name, manifold)
    return out


def hodge_laplacian_matrix(
    state: FieldState,
    basis: list[DifferentialForm],
    manifold: ManifoldSpec,
    *,
    rule: QuadratureSpec,
) -> Tensor:
    r"""Gram matrix :math:`M_{ij}=\langle b_i,\Delta b_j\rangle_{L^2}` in a form basis.

    ``basis`` is a list of ``k``-forms whose component names are all resolvable
    against ``state`` (evaluated at ``rule``'s nodes). The inner product is the
    Euclidean :math:`L^2` product :math:`\sum_I\int (b_i)_I (\Delta b_j)_I\,dx`
    over the quadrature domain (the constant-metric case, e.g. the flat torus).
    The nullity of this symmetric positive-semidefinite matrix is the Betti
    number (see :func:`betti_number`).
    """
    _require_nodes(state.coords, rule, "hodge_laplacian_matrix")
    w = _weights(rule, state.coords)
    lap = [hodge_laplacian(state, b, manifold) for b in basis]
    vals = [{idx: value(state, nm) for idx, nm in b.comps.items()} for b in basis]
    n = len(basis)
    rows: list[Tensor] = []
    for i in range(n):
        row: list[Tensor] = []
        for j in range(n):
            acc = torch.zeros((), dtype=w.dtype, device=w.device)
            for idx, vi in vals[i].items():
                lj = lap[j].get(idx)
                if lj is None:
                    continue
                acc = acc + torch.tensordot(w, vi * lj, dims=([0], [0]))
            row.append(acc)
        rows.append(torch.stack(row))
    return torch.stack(rows)


def betti_number(laplacian_matrix: Tensor, *, tol: float = 1e-8) -> int:
    r"""Betti number = dim of the harmonic space = nullity of the Hodge Laplacian.

    ``laplacian_matrix`` is the symmetric PSD matrix from
    :func:`hodge_laplacian_matrix`; the count of singular values below
    ``tol * max(1, sigma_max)`` is the kernel dimension (harmonic forms), which
    by the Hodge theorem equals the de Rham Betti number ``b_k``.
    """
    s = torch.linalg.svdvals(0.5 * (laplacian_matrix + laplacian_matrix.transpose(-1, -2)))
    if s.numel() == 0:
        return 0
    smax = float(s[0].item())
    thresh = tol * max(smax, 1.0)
    rank = int(torch.sum(s > thresh).item())
    return int(laplacian_matrix.shape[0]) - rank


def harmonic_projection(
    laplacian_matrix: Tensor, coeffs: Tensor, *, tol: float = 1e-8,
) -> Tensor:
    r"""Project a basis-coefficient vector onto the harmonic (kernel) subspace.

    Uses the symmetric eigendecomposition of the Hodge Laplacian matrix and
    keeps the eigenvectors with eigenvalue ``|lambda| <= tol * max(1, |lambda|_max)``
    (the harmonic representatives).
    """
    m = 0.5 * (laplacian_matrix + laplacian_matrix.transpose(-1, -2))
    evals, evecs = torch.linalg.eigh(m)
    wmax = float(torch.max(torch.abs(evals)).item()) if evals.numel() else 0.0
    thresh = tol * max(wmax, 1.0)
    mask = torch.abs(evals) <= thresh
    q = evecs[:, mask]
    return q @ (q.transpose(-1, -2) @ coeffs)


def winding_number(
    state: FieldState, name: str, *, axis: int, rule: QuadratureSpec,
) -> Tensor:
    r"""Winding number (degree) of a circle map :math:`S^1\to S^1`.

    For an angular map :math:`\varphi(x)` (the field component ``name``) around a
    loop of period :math:`2\pi` parametrised by axis ``axis``,

    .. math::

        \deg = \frac{1}{2\pi}\oint \partial_{axis}\varphi\,dx.

    ``state`` must be evaluated at ``rule``'s nodes (a 1-D Gauss-Legendre rule on
    the loop). The winding of :math:`\varphi = q\,x` is exactly ``q``.
    """
    _require_nodes(state.coords, rule, "winding_number")
    dphi = derivative(state, name, axis=axis, order=1)
    w = _weights(rule, dphi)
    total = torch.tensordot(w, dphi, dims=([0], [0]))
    return total / _TWO_PI


def map_degree(
    state: FieldState, names: tuple[str, str, str], *, rule: QuadratureSpec,
) -> Tensor:
    r"""Topological degree of a map :math:`M^2\to S^2`.

    ``names`` are the three components of the unit target vector
    :math:`n(x)\in S^2` as functions of the two chart coordinates (axes 0, 1).
    The degree is the normalised pullback of the target area form,

    .. math::

        \deg = \frac{1}{4\pi}\int_{M} n\cdot(\partial_0 n\times\partial_1 n)\,dx,

    computed from closed-form first derivatives + quadrature. The identity map of
    :math:`S^2` (``n = (\sin\theta\cos\phi,\sin\theta\sin\phi,\cos\theta)``)
    integrates to degree 1.
    """
    _require_nodes(state.coords, rule, "map_degree")
    n = torch.stack([value(state, nm) for nm in names], dim=-1)
    d0 = torch.stack([derivative(state, nm, axis=0, order=1) for nm in names], dim=-1)
    d1 = torch.stack([derivative(state, nm, axis=1, order=1) for nm in names], dim=-1)
    integrand = torch.sum(n * torch.linalg.cross(d0, d1), dim=-1)
    w = _weights(rule, integrand)
    total = torch.tensordot(w, integrand, dims=([0], [0]))
    return total / _FOUR_PI


def gauss_bonnet_euler(
    coords: Tensor, manifold: ManifoldSpec, *, rule: QuadratureSpec,
) -> Tensor:
    r"""Euler characteristic of a closed surface via Gauss-Bonnet.

    .. math::

        \chi = \frac{1}{2\pi}\int_M K\,dA,\qquad K=\tfrac12 R,\; dA=\sqrt{|g|}\,dx,

    with ``K`` the Gaussian curvature (half the scalar curvature in 2-D) and the
    integral taken over the chart at ``rule``'s nodes (which ``coords`` must be).
    Reuses :func:`...connection.scalar_curvature` and
    :func:`...connection.sqrt_det_metric` -- so it is an independent tie-back to
    the curvature stack. The round 2-sphere integrates to :math:`\chi = 2`.
    """
    if manifold.dim != 2:
        raise ValueError(
            f"gauss_bonnet_euler is a surface (dim=2) integral, got dim={manifold.dim}"
        )
    _require_nodes(coords, rule, "gauss_bonnet_euler")
    gauss_k = 0.5 * scalar_curvature(coords, manifold)
    area_element = sqrt_det_metric(coords, manifold)
    integrand = gauss_k * area_element
    w = _weights(rule, integrand)
    total = torch.tensordot(w, integrand, dims=([0], [0]))
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
