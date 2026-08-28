# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Certified **invariant-subspace** (Davis-Kahan ``sin Theta``) enclosures.

:mod:`omnibias.core.verified.eig_operator` and :mod:`omnibias.core.verified.eig`
certify *scalar* facts about a symmetric operator's spectrum: a lower bound on
one eigenvalue, an exact eigenvalue *count* (with multiplicity) below a
threshold, or a spectral gap between two named eigenvalues.  None of them
certify that a whole **cluster** of eigenvalues (a genuinely degenerate ground
state, or a near-degenerate representation multiplet) has been resolved as a
*subspace* -- i.e. that a candidate ``k``-dimensional ``span(V)`` is close to
some true ``k``-dimensional invariant subspace of ``A``, without requiring the
``k`` eigenvalues inside the cluster to be individually separated from *each
other*.  This module closes that gap with a single certificate,
:func:`certified_invariant_subspace`, implementing the classical **Davis-Kahan
sin Theta theorem** via its residual/Sylvester-equation proof, entirely in
outward-rounded interval arithmetic and entirely on top of the *existing*
eigenvalue machinery (no new eigenvalue solver):

1. **Ritz-value cluster bracket.**  With ``H = V^T A V`` and Gram ``G = V^T V``
   (``V``'s columns need not be orthonormal), the generalized eigenvalues of
   the pencil ``(H, G)`` are exactly the Ritz values of the candidate subspace.
   :func:`~omnibias.core.verified.eig_operator.generalized_eigenvalue_enclosure`
   brackets the smallest (index 1) and largest (index ``k``) of them, giving a
   certified cluster interval ``[ritz_lower, ritz_upper]`` -- unchanged whether
   the ``k`` Ritz values are distinct or exactly repeated.
2. **Separation gap of the true operator.**  Rather than construct an explicit
   ``(n-k)``-dimensional orthogonal-complement basis (which would need its own
   soundness argument, e.g. a Gershgorin bound on a deflated ``A - V H V^T``),
   this module reuses
   :func:`~omnibias.core.verified.eig_operator.count_eigenvalues_below` and
   :func:`~omnibias.core.verified.eig_operator.generalized_eigenvalue_enclosure`
   directly on the pencil ``(A, I)`` -- i.e. on ``A`` itself.  Let ``p`` be the
   certified number of eigenvalues of ``A`` strictly below ``ritz_lower``
   (inertia bisection, exact for a genuinely repeated eigenvalue).  Assign the
   cluster to eigenvalue indices ``p+1 .. p+k`` and certify the gap to
   *everything else*: a lower bound on ``ritz_lower - lambda_p`` (if ``p>=1``)
   and on ``lambda_{p+k+1} - ritz_upper`` (if ``p+k<n``); their minimum is
   ``gap``.  This is *sound for any choice of the p+1..p+k block* (see the
   Sylvester-equation argument below) and directly bounds the true spectral
   gap of ``A`` -- tighter, and requiring strictly less new machinery, than a
   Gershgorin estimate on a deflated matrix.  It is also exact for an
   exactly-degenerate cluster: the inertia bisection that produces ``p`` never
   confuses a repeated eigenvalue with two nearby distinct ones.
3. **Residual.**  ``V`` is first turned into a basis ``Vtilde`` that is
   *exactly* orthonormal (to interval precision) but spans the same subspace,
   via the certified interval ``LDL^T`` factor ``G = L D L^T``
   (``Vtilde = V L^{-T} D^{-1/2}``, using
   :func:`~omnibias.core.verified.eig_operator.interval_ldlt_factor` and
   :func:`~omnibias.core.verified.linalg.interval_triangular_solve` -- no new
   factorization).  The residual ``R = A Vtilde - Vtilde Htilde``
   (``Htilde = Vtilde^T A Vtilde``) is then formed in interval arithmetic and
   its **Frobenius norm** ``||R||_F`` bounded (``.hi`` of the enclosed
   ``sum R_ij^2``, then ``sqrt``).  Frobenius is the norm convention used
   throughout this module: it is the simplest to bound rigorously without
   further (operator-norm) machinery, and it is exactly the norm for which the
   classical proof below is cleanest.

**The theorem** (finite-dimensional Davis-Kahan via the Sylvester equation):
let ``S`` be the true invariant subspace of ``A`` spanned by the eigenvectors
with eigenvalues ``lambda_{p+1..p+k}`` and let ``S_perp`` be its orthogonal
complement (spanned by every *other* eigenvector, i.e.
``lambda_1..lambda_p, lambda_{p+k+1..n}``).  Writing ``Y = P_{S_perp} Vtilde``,
the residual identity ``A Vtilde = Vtilde Htilde + R`` gives the Sylvester
equation ``A_2 Y - Y Htilde = P_{S_perp} R`` on ``S_perp`` (where
``A_2 = A|_{S_perp}``, spectrum ``spec(A) \ {lambda_{p+1..p+k}}``).  Because
``spec(Htilde)`` lies inside ``[ritz_lower, ritz_upper]`` and every point of
``spec(A_2)`` is at distance ``>= gap`` from that interval (step 2), the
Sylvester operator is invertible with ``||.^{-1}|| <= 1/gap``, hence
``||Y||_F <= ||R||_F / gap``.  ``Y`` is exactly the matrix whose singular
values are the sines of the principal angles between ``span(V)`` and ``S``, so

.. math:: \sin\Theta(\mathrm{span}(V),\, S) \;\le\; \frac{\|R\|_F}{\mathrm{gap}}.

The certificate reports ``min(||R||_F / gap, sqrt(k))``.  The ``sqrt(k)``
clamp is itself always sound: ``Y``'s ``k`` columns each have norm ``<= 1``
(each is the projection ``P_{S_perp}`` -- a contraction -- applied to a unit
column of the exactly orthonormal ``Vtilde``), so ``||Y||_F <= sqrt(k)``
regardless of the residual/gap estimate (for ``k = 1`` this is the familiar
``sin Theta <= 1``).  When ``G`` cannot
be certified positive definite (``V``'s columns are not certified
independent), or the separation gap cannot be certified positive, the
certificate honestly reports ``certified=False`` with the unresolved fields
left ``None`` -- matching the
:func:`~omnibias.core.verified.eig_operator.count_eigenvalues_below`
returns-``None``-on-failure idiom used throughout this file -- rather than
fabricating a bound.

This module lives apart from :mod:`eig_operator` (rather than being folded
into it) because it is a genuinely different *object*: a subspace/projector
statement built by composing the scalar eigenvalue machinery, not another
scalar eigenvalue bound in the same Temple/Lehmann-Maehly-Goerisch ladder.
Every statement here is a **fixed-operator** result; nothing makes a
continuum or limit claim.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import inf, sqrt

from omnibias.core.verified.eig_operator import (
    count_eigenvalues_below,
    generalized_eigenvalue_enclosure,
    interval_ldlt_factor,
    is_positive_definite,
)
from omnibias.core.verified.interval import Interval, IntervalLike, sum_intervals
from omnibias.core.verified.linalg import identity_matrix, interval_triangular_solve

Matrix = Sequence[Sequence[IntervalLike]]
Vector = Sequence[IntervalLike]


def _to_sym_matrix(matrix: Matrix) -> list[list[Interval]]:
    rows = [[Interval.from_value(x) for x in row] for row in matrix]
    n = len(rows)
    if n == 0:
        raise ValueError("matrix must be non-empty")
    for row in rows:
        if len(row) != n:
            raise ValueError("matrix must be square")
    return rows


def _dot(u: Sequence[Interval], v: Sequence[Interval]) -> Interval:
    return sum_intervals([ui * vi for ui, vi in zip(u, v, strict=True)])


def _matvec_sym(a: list[list[Interval]], v: Sequence[Interval]) -> list[Interval]:
    return [_dot(row, v) for row in a]


def _orthonormal_basis(
    vs: list[list[Interval]], gram: list[list[Interval]]
) -> list[list[Interval]] | None:
    r"""Rigorous orthonormal basis spanning ``span(vs)`` (or ``None``).

    ``gram`` must be the certified positive-definite Gram matrix of ``vs``
    (``gram = vs^T vs``).  Uses the certified interval ``LDL^T`` factor
    ``gram = L D L^T`` (every pivot certified ``> 0``) to form
    ``Vtilde = vs @ L^{-T} @ D^{-1/2}``: ``Vtilde^T Vtilde = I`` to interval
    precision and ``span(Vtilde) = span(vs)`` exactly, since ``L^{-T} D^{-1/2}``
    is an invertible ``k x k`` change of basis.  ``L^{-1}`` is obtained by
    ``k`` unit-lower-triangular solves
    (:func:`~omnibias.core.verified.linalg.interval_triangular_solve`); no new
    factorization or eigensolver is introduced.
    """
    k = len(vs)
    n = len(vs[0]) if vs else 0
    factor = interval_ldlt_factor(gram)
    if factor is None:
        return None
    lower, diagonal = factor
    lower_rows = [list(row) for row in lower]
    # l_inv_cols[i] solves L x = e_i, i.e. it is column i of L^{-1}.
    l_inv_cols: list[list[Interval]] = []
    for i in range(k):
        e_i = [Interval.point(1.0 if j == i else 0.0) for j in range(k)]
        l_inv_cols.append(
            interval_triangular_solve(lower_rows, e_i, lower=True, unit_diagonal=True)
        )
    d_inv_sqrt = [d.sqrt().reciprocal() for d in diagonal]
    vtilde: list[list[Interval]] = []
    for col in range(k):
        # (L^{-T})[j, col] = (L^{-1})[col, j] = l_inv_cols[j][col].
        w = [Interval.point(0.0) for _ in range(n)]
        for j in range(k):
            coeff = l_inv_cols[j][col]
            for row in range(n):
                w[row] = w[row] + coeff * vs[j][row]
        scale = d_inv_sqrt[col]
        vtilde.append([x * scale for x in w])
    return vtilde


def _ritz_block(
    a: list[list[Interval]], vs: list[list[Interval]]
) -> list[list[Interval]]:
    """Symmetrized Ritz block ``H = V^T A V`` for column vectors ``vs``."""
    k = len(vs)
    av = [_matvec_sym(a, v) for v in vs]
    half = Interval.point(0.5)
    return [
        [(_dot(vs[i], av[j]) + _dot(vs[j], av[i])) * half for j in range(k)]
        for i in range(k)
    ]


def _residual_frobenius_upper(
    a: list[list[Interval]], vtilde: list[list[Interval]]
) -> float:
    r"""Certified upper bound on ``||A Vtilde - Vtilde Htilde||_F``."""
    k = len(vtilde)
    n = len(vtilde[0])
    av = [_matvec_sym(a, v) for v in vtilde]
    htilde = _ritz_block(a, vtilde)
    total = Interval.point(0.0)
    for j in range(k):
        for row in range(n):
            entry = av[j][row]
            for i in range(k):
                entry = entry - vtilde[i][row] * htilde[i][j]
            total = total + entry.pow_int(2)
    clamped = Interval(max(total.lo, 0.0), max(total.hi, 0.0))
    return clamped.sqrt().hi


@dataclass(frozen=True)
class InvariantSubspaceCertificate:
    """Certified Davis-Kahan ``sin Theta`` bound for a candidate invariant subspace.

    ``certified`` is ``True`` only when a positive separation ``gap`` was
    actually established; otherwise every downstream field
    (``residual_norm_upper``, ``gap_lower``, ``sin_theta_upper``) is ``None``
    -- an honest refusal, never a fabricated bound.
    """

    dimension: int
    cluster_size: int
    gram_positive_definite: bool
    ritz_lower: float | None
    ritz_upper: float | None
    cluster_start_index: int | None
    gap_lower: float | None
    residual_norm_upper: float | None
    sin_theta_upper: float | None
    certified: bool


def certified_invariant_subspace(
    matrix: Matrix, basis: Sequence[Vector]
) -> InvariantSubspaceCertificate:
    r"""Certify a Davis-Kahan ``sin Theta`` bound for ``span(basis)``.

    Parameters
    ----------
    matrix:
        The ``n x n`` (numerically) symmetric operator ``A``, as an interval
        matrix (entries may themselves be non-degenerate intervals for an
        uncertain ``A``).
    basis:
        The ``k`` candidate basis vectors (the columns of ``V``), each of
        length ``n``.  Columns need not be orthonormal or even orthogonal --
        they are rescaled by a per-column float norm (mirroring
        :func:`omnibias.core.verified.eig._partner_eigenvalue_lower`'s
        diagonal-congruence trick, for Gershgorin-free conditioning only) and
        then rigorously re-orthonormalized inside interval arithmetic (see
        :func:`_orthonormal_basis`); soundness never depends on ``basis``
        being well-conditioned, only on ``k`` linearly independent columns
        (a certified positive-definite Gram matrix).

    Returns
    -------
    InvariantSubspaceCertificate
        Works correctly for an **exactly degenerate** cluster (the ``k``
        Ritz values need not be separated from *each other*, only the
        cluster as a whole from the rest of ``A``'s spectrum) and for any
        ``k >= 1``.
    """
    a = _to_sym_matrix(matrix)
    n = len(a)
    k = len(basis)
    if k < 1:
        raise ValueError("basis must contain at least one vector")
    if k > n:
        raise ValueError("basis dimension k must not exceed the matrix dimension n")
    raw = [[Interval.from_value(x) for x in v] for v in basis]
    for v in raw:
        if len(v) != n:
            raise ValueError("basis vector length must match matrix dimension")

    # Per-column float rescaling: conditions the Gram/Ritz blocks (mirrors
    # eig._partner_eigenvalue_lower) but never affects soundness, since every
    # certified quantity below is recomputed in interval arithmetic from the
    # rescaled vectors.
    vs: list[list[Interval]] = []
    for v in raw:
        norm = sqrt(sum((0.5 * (x.lo + x.hi)) ** 2 for x in v))
        scale = Interval.point(1.0 / norm) if norm > 0.0 else Interval.point(1.0)
        vs.append([x * scale for x in v])

    gram = [[_dot(vs[i], vs[j]) for j in range(k)] for i in range(k)]
    if not is_positive_definite(gram):
        return InvariantSubspaceCertificate(
            dimension=n,
            cluster_size=k,
            gram_positive_definite=False,
            ritz_lower=None,
            ritz_upper=None,
            cluster_start_index=None,
            gap_lower=None,
            residual_norm_upper=None,
            sin_theta_upper=None,
            certified=False,
        )

    h = _ritz_block(a, vs)
    ritz_lo = generalized_eigenvalue_enclosure(h, gram, 1).lo
    ritz_hi = generalized_eigenvalue_enclosure(h, gram, k).hi

    ident = identity_matrix(n)
    p = count_eigenvalues_below(a, ident, ritz_lo)
    if p is None or p + k > n:
        return InvariantSubspaceCertificate(
            dimension=n,
            cluster_size=k,
            gram_positive_definite=True,
            ritz_lower=ritz_lo,
            ritz_upper=ritz_hi,
            cluster_start_index=None,
            gap_lower=None,
            residual_norm_upper=None,
            sin_theta_upper=None,
            certified=False,
        )

    ritz_lo_iv = Interval.point(ritz_lo)
    ritz_hi_iv = Interval.point(ritz_hi)
    if p >= 1:
        left = generalized_eigenvalue_enclosure(a, ident, p)
        left_gap = (ritz_lo_iv - Interval.point(left.hi)).lo
    else:
        left_gap = inf
    if p + k < n:
        right = generalized_eigenvalue_enclosure(a, ident, p + k + 1)
        right_gap = (Interval.point(right.lo) - ritz_hi_iv).lo
    else:
        right_gap = inf
    gap = min(left_gap, right_gap)
    if not gap > 0.0:
        return InvariantSubspaceCertificate(
            dimension=n,
            cluster_size=k,
            gram_positive_definite=True,
            ritz_lower=ritz_lo,
            ritz_upper=ritz_hi,
            cluster_start_index=p + 1,
            gap_lower=(gap if gap != inf else None),
            residual_norm_upper=None,
            sin_theta_upper=None,
            certified=False,
        )

    vtilde = _orthonormal_basis(vs, gram)
    assert vtilde is not None  # gram already certified PD above
    residual_norm = _residual_frobenius_upper(a, vtilde)
    sin_theta = min(residual_norm / gap, sqrt(k))
    return InvariantSubspaceCertificate(
        dimension=n,
        cluster_size=k,
        gram_positive_definite=True,
        ritz_lower=ritz_lo,
        ritz_upper=ritz_hi,
        cluster_start_index=p + 1,
        gap_lower=gap,
        residual_norm_upper=residual_norm,
        sin_theta_upper=sin_theta,
        certified=True,
    )


__all__ = [
    "InvariantSubspaceCertificate",
    "certified_invariant_subspace",
]
