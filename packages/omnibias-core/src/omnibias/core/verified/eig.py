# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Certified eigenvalue and spectral-gap enclosures -- the verified ``gap`` operator.

Three rigorous primitives that turn a (lattice transfer / Galerkin) matrix into a
theorem-grade *spectral-gap* statement, all in outward-rounded interval
arithmetic:

* :func:`rayleigh_quotient` -- for a real symmetric ``A`` and test vector ``v``,
  the quotient ``R(v) = vᵀAv / vᵀv`` obeys ``lambda_min <= R(v) <= lambda_max``,
  so ``R(v).lo`` is a certified lower bound on the largest eigenvalue.
* :func:`symmetric_eigenvalue_residual_enclosure` -- the symmetric residual
  (Bauer-Fike / Krylov-Bogoliubov) bound: with ``theta = R(v)`` and residual
  ``r = A v - theta v`` there is an eigenvalue of ``A`` within
  ``rho = ||r||_2 / ||v||_2`` of ``theta``; the returned interval encloses an
  actual eigenvalue.
* :func:`certified_perron_spectral_gap` -- for an **entrywise-positive** matrix the
  Birkhoff-Hopf theorem bounds the subdominant eigenvalue ratio by the Hilbert
  projective contraction ``tau = (sqrt(kappa) - 1) / (sqrt(kappa) + 1)`` with
  ``kappa = max_{i,j,k,l} (a_ik a_jl) / (a_jk a_il)``.  Hence
  ``|lambda_1| <= tau * lambda_0`` and the (lattice-unit) mass gap obeys
  ``m a = -ln(lambda_1 / lambda_0) >= -ln(tau) > 0``.  The bound is *tight* for
  ``2x2`` positive matrices and conservative (an over-estimate of the ratio,
  hence a valid under-estimate of the gap) otherwise.

Every result is a **fixed-matrix** (fixed lattice spacing / finite volume)
statement, never a continuum or asymptotic claim.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import inf, sqrt

from omnibias.core.verified.interval import Interval, IntervalLike, sum_intervals
from omnibias.core.verified.transcend import ln_iv

Matrix = Sequence[Sequence[IntervalLike]]
Vector = Sequence[IntervalLike]


def _to_matrix(matrix: Matrix) -> list[list[Interval]]:
    rows = [[Interval.from_value(x) for x in row] for row in matrix]
    n = len(rows)
    if n == 0:
        raise ValueError("matrix must be non-empty")
    for row in rows:
        if len(row) != n:
            raise ValueError("matrix must be square")
    return rows


def _square(x: Interval) -> Interval:
    """Rigorous enclosure of ``x**2`` (non-negative)."""
    ax = x.abs()
    return ax * ax


def _matvec(a: list[list[Interval]], v: list[Interval]) -> list[Interval]:
    n = len(v)
    return [sum_intervals([a[i][j] * v[j] for j in range(n)]) for i in range(len(a))]


def _dot(u: list[Interval], v: list[Interval]) -> Interval:
    return sum_intervals([u[i] * v[i] for i in range(len(v))])


def _norm_sq(v: list[Interval]) -> Interval:
    return sum_intervals([_square(x) for x in v])


def rayleigh_quotient(matrix: Matrix, vector: Vector) -> Interval:
    r"""Enclosure of the Rayleigh quotient ``R(v) = vᵀAv / vᵀv``.

    For real symmetric ``A`` this satisfies ``lambda_min <= R(v) <= lambda_max``,
    so ``R(v).lo`` is a certified lower bound on the largest eigenvalue and
    ``R(v).hi`` an upper bound on the smallest.
    """
    a = _to_matrix(matrix)
    v = [Interval.from_value(x) for x in vector]
    if len(v) != len(a):
        raise ValueError("vector length must match matrix dimension")
    denom = _norm_sq(v)
    if denom.lo <= 0.0:
        raise ValueError("vector must be nonzero (vᵀv must exclude 0)")
    num = _dot(v, _matvec(a, v))
    return num * denom.reciprocal()


def symmetric_eigenvalue_residual_enclosure(
    matrix: Matrix, vector: Vector, *, value: IntervalLike | None = None
) -> Interval:
    r"""Interval guaranteed to contain an eigenvalue of the symmetric matrix ``A``.

    With ``theta`` the Rayleigh quotient (or the supplied ``value``) and residual
    ``r = A v - theta v``, the symmetric residual theorem gives an eigenvalue of
    ``A`` within ``rho = ||r||_2 / ||v||_2`` of ``theta``.  The returned interval
    is ``theta + [-rho, rho]`` (outward rounded).  ``A`` is assumed symmetric; the
    enclosure is only meaningful in that case.
    """
    a = _to_matrix(matrix)
    v = [Interval.from_value(x) for x in vector]
    if len(v) != len(a):
        raise ValueError("vector length must match matrix dimension")
    theta = Interval.from_value(value) if value is not None else rayleigh_quotient(a, v)
    av = _matvec(a, v)
    resid = [av[i] - theta * v[i] for i in range(len(v))]
    v2 = _norm_sq(v)
    if v2.lo <= 0.0:
        raise ValueError("vector must be nonzero")
    # The true radicand ||r||^2 / ||v||^2 is >= 0; outward rounding can push the
    # lower endpoint a denormal below 0, so clamp before the (lo >= 0) sqrt.
    radicand = _norm_sq(resid) * v2.reciprocal()
    rho_hi = Interval(max(radicand.lo, 0.0), max(radicand.hi, 0.0)).sqrt().hi
    return theta + Interval(-rho_hi, rho_hi)


def _positive_matrix(matrix: Matrix) -> list[list[Interval]]:
    a = _to_matrix(matrix)
    for row in a:
        for x in row:
            if x.lo <= 0.0:
                raise ValueError(
                    "Birkhoff-Hopf requires an entrywise-positive matrix "
                    "(every entry's lower bound must exceed 0)"
                )
    return a


def birkhoff_projective_diameter(matrix: Matrix) -> Interval:
    r"""Enclosure of ``kappa = max_{i,j,k,l} (a_ik a_jl) / (a_jk a_il)`` (``>= 1``).

    Uses the column factorization ``kappa = max_{k,l} m(k,l) m(l,k)`` with
    ``m(k,l) = max_i a_ik / a_il`` (``O(n^3)``).  Each ratio is enclosed; the
    maxima are taken endpoint-wise so the returned interval encloses the true
    ``kappa``.
    """
    a = _positive_matrix(matrix)
    n = len(a)

    def m(k: int, ell: int) -> Interval:
        ratios = [a[i][k] * a[i][ell].reciprocal() for i in range(n)]
        return Interval(max(r.lo for r in ratios), max(r.hi for r in ratios))

    products = [m(k, ell) * m(ell, k) for k in range(n) for ell in range(n)]
    return Interval(max(p.lo for p in products), max(p.hi for p in products))


def birkhoff_contraction_ratio(matrix: Matrix) -> Interval:
    r"""Enclosure of ``tau = (sqrt(kappa) - 1) / (sqrt(kappa) + 1)`` (in ``[0, 1)``).

    ``tau`` bounds the subdominant eigenvalue ratio: ``|lambda_1| <= tau lambda_0``
    for an entrywise-positive matrix.  ``tau.hi`` is the certified upper bound.
    """
    kappa = birkhoff_projective_diameter(matrix)
    root = kappa.sqrt()
    return (root - Interval.point(1.0)) * (root + Interval.point(1.0)).reciprocal()


@dataclass(frozen=True)
class PerronGapCertificate:
    """Rigorous Birkhoff-Hopf spectral-gap data for an entrywise-positive matrix."""

    dimension: int
    min_entry: float
    kappa_upper: float
    subdominant_ratio_upper: float
    spectral_gap_lower: float
    spectral_gap_lower_per_unit: float


def certified_perron_spectral_gap(
    matrix: Matrix, *, lattice_spacing: float = 1.0
) -> PerronGapCertificate:
    r"""Certified subdominant-ratio and mass-gap lower bound for a positive matrix.

    Returns ``tau`` with ``|lambda_1| <= tau lambda_0`` and the lattice-unit gap
    lower bound ``m a >= -ln(tau)`` (plus ``-ln(tau) / a`` for spacing ``a``).  The
    matrix must be entrywise positive.  A rank-one positive matrix has ``tau = 0``
    and the gap is reported as ``inf``.
    """
    a = _positive_matrix(matrix)
    if lattice_spacing <= 0.0:
        raise ValueError(f"lattice_spacing must be > 0, got {lattice_spacing!r}")
    n = len(a)
    min_entry = min(x.lo for row in a for x in row)
    kappa = birkhoff_projective_diameter(a)
    root = kappa.sqrt()
    tau = (root - Interval.point(1.0)) * (root + Interval.point(1.0)).reciprocal()
    tau_upper = max(tau.hi, 0.0)
    if tau_upper <= 0.0:
        gap_lower = inf
    else:
        # m a >= -ln(tau): -ln is decreasing, so a guaranteed lower bound on the
        # gap is -(upper bound of ln(tau_upper)).
        gap_lower = max(-ln_iv(Interval.point(tau_upper)).hi, 0.0)
    gap_per_unit = inf if gap_lower == inf else gap_lower / lattice_spacing
    return PerronGapCertificate(
        dimension=n,
        min_entry=float(min_entry),
        kappa_upper=float(kappa.hi),
        subdominant_ratio_upper=float(tau_upper),
        spectral_gap_lower=float(gap_lower),
        spectral_gap_lower_per_unit=float(gap_per_unit),
    )


def _frobenius_norm_sq(a: list[list[Interval]]) -> Interval:
    r"""Enclosure of ``sum_ij a_ij**2`` (= ``trace(A**2)`` when ``A`` is symmetric).

    For a symmetric matrix ``trace(A**2) = sum_i lambda_i**2``, so the returned
    upper bound is an enclosure of the second power-sum of the spectrum.
    """
    return sum_intervals(
        [_square(a[i][j]) for i in range(len(a)) for j in range(len(a[i]))]
    )


def collatz_wielandt_perron_bounds(matrix: Matrix, vector: Vector) -> Interval:
    r"""Collatz-Wielandt enclosure of the Perron root of a positive matrix.

    For an entrywise-positive ``A`` and a strictly positive test vector ``x`` the
    Perron-Frobenius / Collatz-Wielandt inequalities give
    ``min_i (A x)_i / x_i <= lambda_0 <= max_i (A x)_i / x_i``.  The returned
    interval ``[min_i ratio, max_i ratio]`` brackets the dominant eigenvalue and
    is tight as ``x`` approaches the Perron vector.  ``lo`` is therefore a
    certified lower bound on ``lambda_0``.
    """
    a = _positive_matrix(matrix)
    v = [Interval.from_value(x) for x in vector]
    if len(v) != len(a):
        raise ValueError("vector length must match matrix dimension")
    for x in v:
        if x.lo <= 0.0:
            raise ValueError("Collatz-Wielandt requires a strictly positive vector")
    av = _matvec(a, v)
    ratios = [av[i] * v[i].reciprocal() for i in range(len(v))]
    return Interval(min(r.lo for r in ratios), max(r.hi for r in ratios))


def _gershgorin_eig_lower(m: list[list[Interval]]) -> float:
    """Rigorous lower bound on the smallest eigenvalue of a symmetric interval matrix.

    Every eigenvalue of a real symmetric matrix lies in the union of Gershgorin
    disks, so ``lambda_min >= min_i (m_ii - sum_{j!=i} |m_ij|)``; the interval
    endpoints (``m_ii.lo``, ``|m_ij|.hi``) keep it valid for the whole box.
    """
    n = len(m)
    return min(
        m[i][i].lo
        - sum_intervals([m[i][j].abs() for j in range(n) if j != i]).hi
        if n > 1
        else m[i][i].lo
        for i in range(n)
    )


def _gershgorin_eig_upper(m: list[list[Interval]]) -> float:
    """Rigorous upper bound on the largest eigenvalue of a symmetric interval matrix."""
    n = len(m)
    return max(
        m[i][i].hi
        + sum_intervals([m[i][j].abs() for j in range(n) if j != i]).hi
        if n > 1
        else m[i][i].hi
        for i in range(n)
    )


def _partner_eigenvalue_lower(
    a: list[list[Interval]], frame: Sequence[Vector]
) -> float:
    r"""Certified lower bound on the ``k``-th largest eigenvalue for a ``k``-frame.

    For *any* ``k``-dimensional subspace ``S = span(frame)`` the Courant-Fischer
    max-min principle gives ``lambda_k >= min_{0 != x in S} R(x) = mu_min(H, G)``,
    the smallest generalized eigenvalue of the pencil ``H = V^T A V``,
    ``G = V^T V``.  With ``H`` symmetric and ``G`` positive definite,
    ``mu_min(H, G) >= mu_min(H) / mu_max(G)``, and Gershgorin disks bound
    ``mu_min(H)`` below and ``mu_max(G)`` above -- so the returned value is a
    rigorous ``lambda_k`` lower bound for **arbitrary** frame vectors (no
    orthonormality assumed).  With the top-``k`` approximate eigenvectors the
    pencil is nearly ``A``-diagonal, so the bound is tight (``~lambda_k``).

    To deflate the degenerate partner ``lambda_3`` of a doubly-degenerate
    subdominant ``lambda_2``, pass the ``k = 3`` frame ``[perron, x_2, x_3]``:
    including the *dominant* direction is what forces ``mu_min <= lambda_3``
    (a 2-frame ``span(x_2, x_3)`` only gives ``mu_min <= lambda_2``, which is
    **not** a valid ``lambda_3`` bound once the inputs are inexact).
    """
    raw = [[Interval.from_value(x) for x in v] for v in frame]
    k = len(raw)
    for v in raw:
        if len(v) != len(a):
            raise ValueError("deflation vector length must match matrix dimension")
    # Normalise each frame vector by a positive float scale.  Diagonal congruence
    # leaves the generalized spectrum mu(H, G) invariant but drives G's diagonal to
    # ~1, which makes the Gershgorin mu_min(H)/mu_max(G) bound tight for any
    # orthogonal basis (a no-op for already-orthonormal eigenvectors).
    vs: list[list[Interval]] = []
    for v in raw:
        norm = sqrt(sum((0.5 * (x.lo + x.hi)) ** 2 for x in v))
        scale = Interval.point(1.0 / norm) if norm > 0.0 else Interval.point(1.0)
        vs.append([x * scale for x in v])
    av = [_matvec(a, v) for v in vs]
    half = Interval.point(0.5)
    # H = V^T A V (off-diagonals symmetrised: A is symmetric so v_i^T A v_j ==
    # v_j^T A v_i, and averaging two enclosures stays valid) and Gram G = V^T V.
    h = [
        [(_dot(vs[i], av[j]) + _dot(vs[j], av[i])) * half for j in range(k)]
        for i in range(k)
    ]
    g = [[_dot(vs[i], vs[j]) for j in range(k)] for i in range(k)]
    h_min = _gershgorin_eig_lower(h)
    g_min = _gershgorin_eig_lower(g)
    g_max = _gershgorin_eig_upper(g)
    # ``G`` must be certified positive definite (``g_min > 0``) so the frame is
    # genuinely full rank: only a full-rank ``(k)``-frame makes Courant-Fischer give
    # ``mu_min(H, G) <= lambda_{k-1}`` -- a rank-deficient frame would bound a
    # *larger* eigenvalue and break rigour.  With ``h_min >= 0`` and ``G`` SPD,
    # ``mu_min(H, G) = min_y (yᵀHy)/(yᵀGy) >= h_min / g_max``.
    if h_min <= 0.0 or g_min <= 0.0 or g_max <= 0.0:
        return 0.0
    return max(h_min / g_max, 0.0)


@dataclass(frozen=True)
class SymmetricGapCertificate:
    """Rigorous spectral-gap data for a real *symmetric* matrix via power sums.

    Tighter than :class:`PerronGapCertificate` whenever the subdominant
    eigenvalue dominates the remaining spectral tail (e.g. heat-kernel transfer
    matrices, whose eigenvalues decay like ``e^{-t C2}``).
    """

    dimension: int
    perron_lower: float
    subdominant_upper: float
    subdominant_ratio_upper: float
    spectral_gap_lower: float
    spectral_gap_lower_per_unit: float
    partner_lower: float = 0.0
    # number of partner eigenvalues whose certified lower bound was deflated
    partners_deflated: int = 0
    perron_upper: float = inf
    subdominant_lower: float = 0.0
    spectral_gap_upper: float = inf
    spectral_gap_upper_per_unit: float = inf


def certified_symmetric_spectral_gap(
    matrix: Matrix,
    perron_vector: Vector,
    *,
    subdominant_vectors: Sequence[Vector] | None = None,
    lattice_spacing: float = 1.0,
) -> SymmetricGapCertificate:
    r"""Certified subdominant-ratio / mass-gap for a real **symmetric** matrix.

    ``A`` must be (numerically) symmetric so that its eigenvalues are real.  Given
    an approximate dominant eigenvector ``perron_vector`` the certificate combines
    two rigorous facts:

    * a lower bound ``lambda_0 >= max(R(x), min_i (A x)_i / x_i)`` from the
      Rayleigh quotient and -- when ``x > 0`` and ``A`` is positive --
      Collatz-Wielandt;
    * the power-sum / Schur inequality
      ``lambda_1**2 <= trace(A**2) - lambda_0**2 - sum_{i>=2} lambda_i**2`` for the
      second-largest eigenvalue, since ``sum_i lambda_i**2 = trace(A**2) =
      sum_ij a_ij**2``.

    Hence ``|lambda_1| / lambda_0 <= ratio`` and the lattice-unit gap obeys
    ``m a >= -ln(ratio)``.  Unlike Birkhoff-Hopf this is *tight* whenever the
    subdominant eigenvalue dominates the remaining tail, recovering nearly the
    full gap for rapidly decaying spectra.  It is still a **fixed-matrix**
    statement (fixed spacing / finite dimension), never a continuum claim.

    ``subdominant_vectors`` optionally supplies approximate eigenvectors for the
    partners ``v_1 ~ lambda_1, v_2 ~ lambda_2, ...`` below the dominant mode.  When
    ``m >= 2`` are given, a *chain* of nested frames ``[perron, v_1, ..., v_k]``
    (``k = 2 .. m``) each yields a Courant-Fischer lower bound
    ``ell_k <= lambda_k`` (:func:`_partner_eigenvalue_lower`), and because the
    ``lambda_k`` are distinct power-sum terms ``sum_k ell_k**2 <= sum_{i>=2}
    lambda_i**2``.  Subtracting the whole chain,
    ``lambda_1**2 <= trace(A**2) - lambda_0**2 - sum_k ell_k**2``, drives the bound
    toward the exact ``lambda_1`` as more partners are supplied.  Deflating a single
    partner removes the ``sqrt(multiplicity)`` inflation of a doubly-degenerate
    subdominant (the ``+/- n`` U(1) modes, the ``(p, q) <-> (q, p)`` SU(3) pairs);
    deflating the chain additionally tames a **slowly-decaying tail** (e.g. the
    Wilson transfer matrix, whose Bessel eigenvalues ``I_n`` decay far slower than a
    heat kernel's ``e^{-t C2}``).  Including the dominant direction in every frame is
    what makes ``ell_k`` a valid ``lambda_k`` (not ``lambda_{k-1}``) bound; the chain
    is rigorous for *any* test vectors (each ``ell_k >= 0``, and ``0`` whenever
    positivity or frame full-rank cannot be certified) and never weakens the
    certificate.  ``partners_deflated`` reports how many chain terms were certified
    positive; ``partner_lower`` is ``ell_2`` (the first partner).

    When at least one ``subdominant_vectors`` entry is given the certificate also
    reports a rigorous **upper** bound on the gap, bracketing ``m a`` in
    ``[spectral_gap_lower, spectral_gap_upper]``: ``lambda_0`` is bounded above by
    ``min(sqrt(trace(A**2)), max-Gershgorin-row)`` and the subdominant ``lambda_1``
    below by the smallest Ritz value of the ``2``-frame ``[perron, v_2]``
    (Cauchy interlacing: ``mu_min(V^T A V, V^T V) <= lambda_1``), so
    ``ratio >= lambda_1_lower / lambda_0_upper`` and ``m a <= -ln(that)``.  The
    bracket collapsing onto a point certifies the gap essentially exactly (a
    fixed-matrix statement -- it does *not* address the continuum / uniform limit).

    A non-positive certified Perron lower bound raises ``ValueError``; a ratio
    ``>= 1`` (no separation certifiable) yields ``spectral_gap_lower == 0.0``.
    """
    a = _to_matrix(matrix)
    if lattice_spacing <= 0.0:
        raise ValueError(f"lattice_spacing must be > 0, got {lattice_spacing!r}")
    n = len(a)
    perron_lo = rayleigh_quotient(a, perron_vector).lo
    try:
        perron_lo = max(perron_lo, collatz_wielandt_perron_bounds(a, perron_vector).lo)
    except ValueError:
        # Non-positive matrix or non-positive test vector: keep the Rayleigh bound.
        pass
    if perron_lo <= 0.0:
        raise ValueError("could not certify a positive Perron lower bound")
    perron_lo_iv = Interval.point(perron_lo)
    fro = _frobenius_norm_sq(a)
    # Deflate a *chain* of certified lower bounds on the subdominant's partners.
    # Each nested frame [perron, v_1, ..., v_k] (k = 2 .. m) yields a Courant-Fischer
    # lower bound ell_k <= lambda_k (0-indexed), and since the lambda_k are distinct
    # power-sum terms, sum_k ell_k^2 <= sum_{i>=2} lambda_i^2.  Subtracting the whole
    # chain tightens lambda_1^2 <= trace(A^2) - lambda_0^2 - sum_k ell_k^2 toward the
    # exact value -- crucial for slowly-decaying spectra (e.g. the Wilson transfer
    # matrix, whose Bessel eigenvalues I_n decay far slower than the heat kernel's
    # e^{-t C2}, so a single partner leaves a large polluting tail).  Rigorous for
    # arbitrary inputs: ell_k >= 0, and 0 whenever positivity / frame full-rank
    # cannot be certified, so the tail is never over-deflated.
    partner_lo = 0.0
    partners_deflated = 0
    partner_sq = Interval.point(0.0)
    if subdominant_vectors is not None and len(subdominant_vectors) >= 2:
        subs = [list(v) for v in subdominant_vectors]
        pv = list(perron_vector)
        for k in range(2, len(subs) + 1):
            ell = _partner_eigenvalue_lower(a, [pv, *subs[:k]])
            partner_sq = partner_sq + _square(Interval.point(ell))
            if ell > 0.0:
                partners_deflated += 1
                if k == 2:
                    partner_lo = ell
    # lambda_1^2 <= trace(A^2) - lambda_0^2 - sum_k ell_k^2: upper-bound trace(A^2),
    # lower-bound lambda_0^2 and each ell_k^2; interval subtraction keeps tail.hi an
    # upper bound (subtracting the rounded-down squares only enlarges it).
    tail = fro - _square(perron_lo_iv) - partner_sq
    sub_hi = max(tail.hi, 0.0)
    sub_upper = Interval(sub_hi, sub_hi).sqrt().hi
    ratio = Interval.point(sub_upper) * perron_lo_iv.reciprocal()
    ratio_upper = max(ratio.hi, 0.0)  # honest: > 1 means no certifiable separation
    if 0.0 < ratio_upper < 1.0:
        gap_lower = max(-ln_iv(Interval.point(ratio_upper)).hi, 0.0)
    else:
        # ratio_upper >= 1: no certifiable separation.  ratio_upper == 0: the tail
        # collapsed to numerical zero, indistinguishable from rounding.  Either way
        # the conservative (never-over-claim) gap lower bound is 0.
        gap_lower = 0.0
    # Two-sided bracket: upper bound on lambda_0 (lambda_0^2 <= tr(A^2); Gershgorin
    # row bound) and -- given a subdominant direction -- a lower bound on lambda_1
    # (smallest Ritz value of the 2-frame [perron, v_2]) give an upper gap bound.
    fro_hi = max(fro.hi, 0.0)
    perron_up = min(Interval(fro_hi, fro_hi).sqrt().hi, _gershgorin_eig_upper(a))
    sub_lo = 0.0
    if subdominant_vectors is not None and len(subdominant_vectors) >= 1:
        sub_lo = _partner_eigenvalue_lower(
            a, [list(perron_vector), list(subdominant_vectors[0])]
        )
    gap_upper = inf
    if sub_lo > 0.0 and perron_up > 0.0:
        ratio_lo = (Interval.point(sub_lo) * Interval.point(perron_up).reciprocal()).lo
        if 0.0 < ratio_lo < 1.0:
            gap_upper = max(-ln_iv(Interval.point(ratio_lo)).lo, gap_lower)
    gap_upper_per_unit = gap_upper / lattice_spacing if gap_upper != inf else inf
    return SymmetricGapCertificate(
        dimension=n,
        perron_lower=float(perron_lo),
        subdominant_upper=float(sub_upper),
        subdominant_ratio_upper=float(ratio_upper),
        spectral_gap_lower=float(gap_lower),
        spectral_gap_lower_per_unit=float(gap_lower / lattice_spacing),
        partner_lower=float(partner_lo),
        partners_deflated=int(partners_deflated),
        perron_upper=float(perron_up),
        subdominant_lower=float(sub_lo),
        spectral_gap_upper=float(gap_upper),
        spectral_gap_upper_per_unit=float(gap_upper_per_unit),
    )


@dataclass(frozen=True)
class BlockOperatorGapCertificate:
    r"""Finite-section + tail-defect coercivity bound for a self-adjoint operator.

    For a self-adjoint operator ``S`` split by an orthogonal projection ``P``
    (finite ``N``-dimensional range) and ``Q = I - P`` into blocks

    .. math::

        S = \begin{pmatrix} A & B \\ B^{\!*} & D \end{pmatrix},\qquad
        A = P S P,\ \ D = Q S Q,\ \ B = P S Q,

    the Rayleigh quotient of any ``v = (p, q)`` obeys
    ``<Sv,v> >= a\|p\|^2 - 2b\|p\|\|q\| + d\|q\|^2`` with ``a = \lambda_min(A)``,
    ``b >= \|B\|``, ``d = \lambda_min(D)``; minimising the right-hand quadratic form
    over the unit circle gives the **certified** lower bound

    .. math::

        \lambda_{\min}(S)\ \ge\
        \tfrac12\Big[(a+d) - \sqrt{(a-d)^2 + 4b^2}\Big].

    The finite gap ``a`` is computed here (Gershgorin) from the supplied finite
    block; the coupling bound ``b`` is supplied by the caller; the **tail gap
    ``d`` is an explicit hypothesis** -- it is *not* certified by this primitive
    (bounding the infinite tail is the hard analytic step).  Coercivity
    (``gap_lower > 0``) holds iff ``a > 0`` and ``d > threshold_tail_gap = b^2/a``;
    that single scalar inequality is the isolated open obligation of a conditional
    coercivity / spectral-gap program (e.g. the linearised rescaled SQG operator in
    a weighted norm).  This is a **fixed-finite-block** statement; it never claims
    the tail bound and never makes a continuum / blow-up / global-regularity claim.
    """

    finite_dim: int
    finite_gap_lower: float
    coupling_norm_upper: float
    tail_gap_lower: float | None
    tail_is_hypothesis: bool
    gap_lower: float | None
    coercive: bool
    threshold_tail_gap: float


def certified_block_operator_gap(
    finite_block: Matrix,
    *,
    coupling_norm_upper: float,
    tail_gap_lower: float | None = None,
) -> BlockOperatorGapCertificate:
    r"""Conditional coercivity gap of a self-adjoint block operator (finite + tail).

    Computes a rigorous lower bound on the finite block's smallest eigenvalue
    ``a = \lambda_min(A)`` (Gershgorin) and combines it with a coupling bound
    ``b >= \|B\|`` (``coupling_norm_upper``) and an **assumed** tail gap
    ``d <= \lambda_min(D)`` (``tail_gap_lower``) via

    .. math::
        \lambda_{\min}(S) \ge \tfrac12[(a+d) - \sqrt{(a-d)^2 + 4b^2}].

    All endpoints are outward rounded so the returned ``gap_lower`` is a genuine
    lower bound *given* the hypothesis ``d``.  When ``tail_gap_lower`` is ``None``
    only ``threshold_tail_gap = b^2/a`` is reported: the minimal tail gap for which
    coercivity would follow -- i.e. the single inequality a conditional program must
    still close.  ``tail_is_hypothesis`` is always ``True``: this primitive does not
    prove the tail bound.

    Parameters
    ----------
    finite_block:
        The ``N x N`` (numerically) symmetric finite section ``A = P S P``.
    coupling_norm_upper:
        A certified upper bound ``b >= \|B\|`` on the finite-to-tail coupling.
    tail_gap_lower:
        An assumed lower bound ``d <= \lambda_min(D)`` on the tail block.  Left
        ``None`` to report only the threshold the tail must clear.
    """
    a_mat = _to_matrix(finite_block)
    b = float(coupling_norm_upper)
    if b < 0.0:
        raise ValueError("coupling_norm_upper must be >= 0")
    a = _gershgorin_eig_lower(a_mat)
    a_iv = Interval.point(a)
    b_iv = Interval.point(b)
    if a > 0.0:
        threshold = (b_iv * b_iv * a_iv.reciprocal()).hi
    else:
        threshold = inf
    gap_lower: float | None = None
    coercive = False
    if tail_gap_lower is not None:
        d = float(tail_gap_lower)
        d_iv = Interval.point(d)
        # radicand (a-d)^2 + 4 b^2; clamp the lower endpoint against rounding.
        radicand = (a_iv - d_iv).pow_int(2) + Interval.point(4.0) * b_iv * b_iv
        radicand = Interval(max(radicand.lo, 0.0), max(radicand.hi, 0.0))
        root = radicand.sqrt()
        # lower bound on gamma uses (a+d) low and root.hi (the worst case).
        gamma = (a_iv + d_iv - root) * Interval.point(0.5)
        gap_lower = float(gamma.lo)
        coercive = bool(gamma.lo > 0.0)
    return BlockOperatorGapCertificate(
        finite_dim=len(a_mat),
        finite_gap_lower=float(a),
        coupling_norm_upper=b,
        tail_gap_lower=(float(tail_gap_lower) if tail_gap_lower is not None else None),
        tail_is_hypothesis=True,
        gap_lower=gap_lower,
        coercive=coercive,
        threshold_tail_gap=float(threshold),
    )


__all__ = [
    "BlockOperatorGapCertificate",
    "PerronGapCertificate",
    "SymmetricGapCertificate",
    "birkhoff_contraction_ratio",
    "birkhoff_projective_diameter",
    "certified_block_operator_gap",
    "certified_perron_spectral_gap",
    "certified_symmetric_spectral_gap",
    "collatz_wielandt_perron_bounds",
    "rayleigh_quotient",
    "symmetric_eigenvalue_residual_enclosure",
]
