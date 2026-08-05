# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Certified eigenvalue **lower** bounds at the bottom of the spectrum.

:mod:`omnibias.core.verified.eig` certifies the *top* of the spectrum (Rayleigh
quotients, Perron / power-sum gaps for transfer matrices).  This module does the
genuinely hard direction -- rigorous **lower** bounds on the lowest eigenvalues of
a self-adjoint operator bounded below -- via the classical
**Temple -> Lehmann-Maehly -> Goerisch** ladder, plus the finite-section
machinery that turns a truncated (Galerkin) problem into a statement about the
true operator.

The cast (all in outward-rounded interval arithmetic):

* :func:`temple_lower_bound` -- Temple's inequality, the one-vector base case:
  with Rayleigh quotient ``theta = <Au,u>/<u,u>``, second moment
  ``eta = ||Au||^2/<u,u>`` and an a-priori ``rho <= lambda_2``,

  .. math:: \lambda_1 \ge \theta - \frac{\eta - \theta^2}{\rho - \theta}.

* :func:`interval_ldlt_inertia` -- certified inertia of a symmetric interval
  matrix by an interval ``LDL^T`` factorisation (every pivot sign-definite =>
  Sylvester's law fixes the inertia of the whole matrix box).
* :func:`count_eigenvalues_below` -- the number of generalized eigenvalues of the
  pencil ``(a, M)`` below ``t`` equals the negative inertia of ``a - t M``
  (``M`` SPD); the rigorous spectral counter that drives the bisection.
* :func:`generalized_eigenvalue_enclosure` -- enclose the ``i``-th smallest
  generalized eigenvalue of a symmetric-definite pencil by inertia bisection.
* :func:`lehmann_maehly_lower_bounds` -- the Lehmann-Maehly-Goerisch theorem:
  with ``a = A_1 - rho A_0`` and ``M = A_2 - 2 rho A_1 + rho^2 A_0`` (the Gram
  matrices ``A_0=<w_i,w_j>``, ``A_1=<Aw_i,w_j>``, ``A_2=<Aw_i,Aw_j>``), the
  negative generalized eigenvalues ``tau_1<=...<=tau_p<0`` give
  ``lambda_{N-i+1} >= rho + 1/tau_i`` where ``N`` is the (a-priori) number of
  eigenvalues at or below ``rho``.  Goerisch's choice ``A_2=<Aw_i,Aw_j>``
  computed with the *exact* operator captures the discarded-subspace tail, so the
  bound is valid for the true (possibly infinite-dimensional) operator, not just
  the finite section.
* :func:`operator_comparison_bounds` -- Weyl monotonicity a-priori brackets for
  ``H = H_0 + V`` from a known base spectrum and ``min V <= V <= max V``; this is
  how one obtains the a-priori ``rho`` (a certified lower bound on
  ``lambda_{N+1}``) that the Lehmann labelling requires.
* :func:`certified_spectral_gap` -- combine a Rayleigh-Ritz **upper** bound on
  ``lambda_1`` with the Lehmann **lower** bound on ``lambda_2`` into a certified
  positive gap ``lambda_2 - lambda_1``.

Every statement is a **fixed-operator** result; nothing here makes a continuum or
limit claim.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import inf

from omnibias.core.verified.interval import Interval, IntervalLike
from omnibias.core.verified.linalg import (
    inf_norm_matrix,
    neumann_inverse_norm_bound,
    to_interval_matrix,
)

Matrix = Sequence[Sequence[IntervalLike]]
Vector = Sequence[IntervalLike]


# --------------------------------------------------------------------------- #
# Symmetric-matrix helpers.
# --------------------------------------------------------------------------- #
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
    acc = Interval.point(0.0)
    for ui, vi in zip(u, v, strict=True):
        acc = acc + ui * vi
    return acc


def _matvec_sym(a: list[list[Interval]], v: Sequence[Interval]) -> list[Interval]:
    return [_dot(row, v) for row in a]


def _float_inverse(matrix: Sequence[Sequence[float]]) -> list[list[float]] | None:
    """Plain Gauss-Jordan inverse with partial pivoting (``None`` if singular).

    Only used to build the *approximate* inverse ``B`` fed to the rigorous
    :func:`neumann_inverse_norm_bound`; soundness never depends on its accuracy.
    """
    n = len(matrix)
    aug = [[float(matrix[i][j]) for j in range(n)] + [1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < 1e-300:
            return None
        aug[col], aug[pivot] = aug[pivot], aug[col]
        piv = aug[col][col]
        aug[col] = [x / piv for x in aug[col]]
        for r in range(n):
            if r != col and aug[r][col] != 0.0:
                factor = aug[r][col]
                aug[r] = [aug[r][k] - factor * aug[col][k] for k in range(2 * n)]
    return [[aug[i][n + j] for j in range(n)] for i in range(n)]


@dataclass(frozen=True)
class Inertia:
    """Certified signature of a symmetric interval matrix (counts must sum to ``n``)."""

    negative: int
    positive: int
    pivots: tuple[float, ...]


def _ldlt_pivots(s: list[list[Interval]]) -> list[Interval] | None:
    r"""The interval ``LDL^T`` pivots ``D_jj`` of a symmetric interval matrix.

    Returns ``None`` as soon as a pivot interval straddles ``0`` (its sign -- hence
    the inertia -- cannot be certified for the whole matrix box).  Otherwise the
    returned list holds each certified pivot interval ``D_jj`` in order; each
    encloses the corresponding exact pivot of *every* point matrix in the box
    (inclusion-isotone arithmetic), and ``S = L D L^T`` is a congruence, so by
    Sylvester's law of inertia the whole box shares the pivots' sign pattern.
    """
    n = len(s)
    lmat = [[Interval.point(0.0) for _ in range(n)] for _ in range(n)]
    d = [Interval.point(0.0) for _ in range(n)]
    for j in range(n):
        dj = s[j][j]
        for k in range(j):
            dj = dj - lmat[j][k] * lmat[j][k] * d[k]
        if dj.lo <= 0.0 <= dj.hi:
            return None
        d[j] = dj
        inv = dj.reciprocal()
        for i in range(j + 1, n):
            lij = s[i][j]
            for k in range(j):
                lij = lij - lmat[i][k] * lmat[j][k] * d[k]
            lmat[i][j] = lij * inv
    return d


def interval_ldlt_inertia(matrix: Matrix) -> Inertia | None:
    r"""Certified inertia of a symmetric interval matrix via interval ``LDL^T``.

    Returns ``None`` when a pivot interval straddles ``0`` (the sign -- hence the
    inertia -- cannot be certified for the whole matrix box; ``t`` is too close to
    an eigenvalue).  Otherwise every point matrix in the box shares the returned
    inertia, because each interval pivot contains the corresponding exact pivot
    (inclusion-isotone arithmetic) and ``S = L D L^T`` is a congruence
    (Sylvester's law of inertia).
    """
    d = _ldlt_pivots(_to_sym_matrix(matrix))
    if d is None:
        return None
    neg = sum(1 for dj in d if dj.hi < 0.0)
    return Inertia(negative=neg, positive=len(d) - neg, pivots=tuple(x.mid for x in d))


def interval_ldlt_pivots(matrix: Matrix) -> tuple[Interval, ...] | None:
    r"""Certified interval ``LDL^T`` pivots of a symmetric interval matrix (or ``None``).

    The finer-grained companion of :func:`interval_ldlt_inertia`: instead of only
    the *counts* of positive / negative pivots it returns each pivot **interval**
    ``D_jj`` in order.  A matrix box is certified positive definite exactly when the
    result is not ``None`` and every returned pivot has ``.lo > 0`` -- the full
    inertia vector that the Lean kernel's ``allPivotsPos`` obligation re-checks
    (lifting the positive-definiteness claim above its single scalar ``eig_min``
    shadow).  Returns ``None`` when a pivot straddles ``0``.
    """
    d = _ldlt_pivots(_to_sym_matrix(matrix))
    return None if d is None else tuple(d)


def is_positive_definite(matrix: Matrix) -> bool:
    """``True`` iff a symmetric interval matrix is *certified* positive definite."""
    inertia = interval_ldlt_inertia(matrix)
    return inertia is not None and inertia.negative == 0


def _pencil_shift(a: list[list[Interval]], m: list[list[Interval]], t: float) -> list[list[Interval]]:
    t_iv = Interval.point(t)
    n = len(a)
    return [[a[i][j] - t_iv * m[i][j] for j in range(n)] for i in range(n)]


def count_eigenvalues_below(a: Matrix, m: Matrix, t: float) -> int | None:
    r"""Number of generalized eigenvalues of ``(a, m)`` strictly below ``t``.

    For symmetric ``a`` and SPD ``m`` the matrix ``a - t m`` is congruent to
    ``diag(tau_i - t)``, so its negative inertia counts ``#{tau_i < t}``.  Returns
    ``None`` when the count cannot be certified (a pivot straddles ``0``, i.e.
    ``t`` is essentially on the spectrum).
    """
    ai = _to_sym_matrix(a)
    mi = _to_sym_matrix(m)
    inertia = interval_ldlt_inertia(_pencil_shift(ai, mi, t))
    return None if inertia is None else inertia.negative


def _spectral_bracket(a: list[list[Interval]], m: list[list[Interval]]) -> float:
    r"""A finite ``B`` with every generalized eigenvalue of ``(a, m)`` in ``[-B, B]``.

    Uses ``|tau| <= ||m^{-1}||_inf ||a||_inf`` with a rigorous Neumann bound on
    ``||m^{-1}||``; falls back to a crude diagonal estimate if the float inverse
    is unavailable.
    """
    n = len(a)
    m_mid = [[m[i][j].mid for j in range(n)] for i in range(n)]
    a_norm = inf_norm_matrix(a)
    binv = _float_inverse(m_mid)
    if binv is not None:
        nb = neumann_inverse_norm_bound(m_mid, binv)
        if nb["certified"]:
            bound = float(nb["inverse_norm_bound"]) * a_norm
            if bound > 0.0:
                return bound * 2.0 + 1.0
    return (a_norm + inf_norm_matrix(m) + 1.0) * float(n) * 1e3


def generalized_eigenvalue_enclosure(
    a: Matrix, m: Matrix, index: int, *, bracket: tuple[float, float] | None = None, iters: int = 200
) -> Interval:
    r"""Enclose the ``index``-th smallest (1-based) generalized eigenvalue of ``(a, m)``.

    ``m`` must be certified SPD.  Two inertia bisections (one for each endpoint)
    sandwich ``tau_index``: the lower endpoint is the greatest certified ``t`` with
    ``count_below(t) < index`` and the upper endpoint the least certified ``t``
    with ``count_below(t) >= index``.  Rigorous for any symmetric-definite pencil.
    """
    ai = _to_sym_matrix(a)
    mi = _to_sym_matrix(m)
    n = len(ai)
    if not 1 <= index <= n:
        raise ValueError(f"index must be in 1..{n}, got {index}")
    if not is_positive_definite(mi):
        raise ValueError("right matrix m must be certified positive definite")
    if bracket is None:
        b = _spectral_bracket(ai, mi)
        lo0, hi0 = -b, b
    else:
        lo0, hi0 = bracket
    return Interval(
        _bisect_endpoint(ai, mi, index, lo0, hi0, iters, want_upper=False),
        _bisect_endpoint(ai, mi, index, lo0, hi0, iters, want_upper=True),
    )


def _bisect_endpoint(
    a: list[list[Interval]],
    m: list[list[Interval]],
    index: int,
    lo0: float,
    hi0: float,
    iters: int,
    *,
    want_upper: bool,
) -> float:
    """Greatest ``t`` with ``count<index`` (lower) / least ``t`` with ``count>=index`` (upper)."""
    lo, hi = lo0, hi0
    for _ in range(iters):
        if hi - lo <= 1e-15 * max(1.0, abs(lo), abs(hi)):
            break
        mid = 0.5 * (lo + hi)
        c = count_eigenvalues_below(a, m, mid)
        if c is None:
            break
        if c >= index:
            hi = mid
        else:
            lo = mid
    return hi if want_upper else lo


def temple_lower_bound(
    rayleigh: IntervalLike, second_moment: IntervalLike, rho: float
) -> Interval:
    r"""Temple's lower bound on ``lambda_1`` (its ``.lo`` is the certified bound).

    ``rayleigh`` is ``theta = <Au,u>/<u,u>`` (so ``theta >= lambda_1``),
    ``second_moment`` is ``eta = ||Au||^2/<u,u>`` (so ``eta >= theta^2``), and
    ``rho`` must satisfy ``theta < rho <= lambda_2``.  Returns the enclosure of
    ``theta - (eta - theta^2)/(rho - theta)``; its lower endpoint underestimates
    the true Temple value, hence ``lambda_1``.
    """
    theta = Interval.from_value(rayleigh)
    eta = Interval.from_value(second_moment)
    rho_f = float(rho)
    if theta.hi >= rho_f:
        raise ValueError("Temple requires the Rayleigh quotient strictly below rho")
    denom = Interval.point(rho_f) - theta  # > 0
    variance = eta - theta * theta  # >= 0
    return theta - variance * denom.reciprocal()


def temple_lower_bound_vector(matrix: Matrix, vector: Vector, rho: float) -> Interval:
    """Temple's bound from an explicit matrix ``A`` and (approx ground-state) vector."""
    a = _to_sym_matrix(matrix)
    v = [Interval.from_value(x) for x in vector]
    if len(v) != len(a):
        raise ValueError("vector length must match matrix dimension")
    vv = _dot(v, v)
    if vv.lo <= 0.0:
        raise ValueError("vector must be nonzero")
    inv = vv.reciprocal()
    av = _matvec_sym(a, v)
    theta = _dot(v, av) * inv
    eta = _dot(av, av) * inv
    return temple_lower_bound(theta, eta, rho)


@dataclass(frozen=True)
class EigenvalueLowerBound:
    """One certified ``lambda_index >= lower_bound`` from the Lehmann pencil."""

    index: int
    lower_bound: float
    tau_upper: float
    rho: float


@dataclass(frozen=True)
class LehmannCertificate:
    """Lehmann-Maehly-Goerisch certified lower bounds for the lowest eigenvalues."""

    rho: float
    n_below: int
    negatives: int
    m_positive_definite: bool
    inertia_certified: bool
    bounds: tuple[EigenvalueLowerBound, ...]


def _lehmann_matrices(
    a0: list[list[Interval]],
    a1: list[list[Interval]],
    a2: list[list[Interval]],
    rho: float,
) -> tuple[list[list[Interval]], list[list[Interval]]]:
    n = len(a0)
    rho_iv = Interval.point(rho)
    two_rho = Interval.point(2.0) * rho_iv
    rho_sq = rho_iv * rho_iv
    a = [[a1[i][j] - rho_iv * a0[i][j] for j in range(n)] for i in range(n)]
    m = [[a2[i][j] - two_rho * a1[i][j] + rho_sq * a0[i][j] for j in range(n)] for i in range(n)]
    return a, m


def lehmann_maehly_lower_bounds(
    a0: Matrix, a1: Matrix, a2: Matrix, rho: float, *, n_below: int
) -> LehmannCertificate:
    r"""Certified lower bounds on the lowest eigenvalues (Lehmann-Maehly-Goerisch).

    Parameters
    ----------
    a0, a1, a2:
        The Gram matrices ``A_0=<w_i,w_j>``, ``A_1=<Aw_i,w_j>``,
        ``A_2=<Aw_i,Aw_j>`` over the trial vectors ``w_i`` (``A_2`` evaluated with
        the *exact* operator -- Goerisch -- so the discarded-subspace tail is
        included).
    rho:
        An a-priori value with ``lambda_{n_below} <= rho <= lambda_{n_below+1}``
        (typically from :func:`operator_comparison_bounds`).  The *labelling* of
        the bounds relies on this hypothesis; the inequalities themselves are
        certified.
    n_below:
        ``N``: the number of eigenvalues at or below ``rho``.

    Returns a :class:`LehmannCertificate`; ``bounds[k]`` carries
    ``lambda_{N-k+1} >= rho + 1/tau_{k+1}``.  Empty ``bounds`` (with the flags
    showing why) when ``M`` is not certified SPD or the pencil's inertia cannot be
    certified.
    """
    a0i, a1i, a2i = (to_interval_matrix(x) for x in (a0, a1, a2))
    rho_f = float(rho)
    a, m = _lehmann_matrices(a0i, a1i, a2i, rho_f)
    pd = is_positive_definite(m)
    inertia = interval_ldlt_inertia(a)
    if not pd or inertia is None:
        return LehmannCertificate(rho_f, int(n_below), 0, pd, inertia is not None, ())
    p = inertia.negative
    rho_iv = Interval.point(rho_f)
    bracket = (-_spectral_bracket(a, m), 0.0)
    bounds: list[EigenvalueLowerBound] = []
    for i in range(1, p + 1):
        tau_hi = _bisect_endpoint(a, m, i, bracket[0], bracket[1], 200, want_upper=True)
        if tau_hi >= 0.0:
            continue
        sigma = rho_iv + Interval.point(tau_hi).reciprocal()
        idx = n_below - i + 1
        if idx < 1:
            continue
        bounds.append(
            EigenvalueLowerBound(index=idx, lower_bound=sigma.lo, tau_upper=tau_hi, rho=rho_f)
        )
    bounds.sort(key=lambda b: b.index)
    return LehmannCertificate(rho_f, int(n_below), p, True, True, tuple(bounds))


def operator_comparison_bounds(
    base_eigenvalues: Sequence[float],
    perturbation_lower: float,
    perturbation_upper: float,
) -> list[Interval]:
    r"""Weyl monotonicity brackets for ``H = H_0 + V``, ``min V <= V <= max V``.

    For self-adjoint ``H_0`` with eigenvalues ``mu_k`` and a bounded symmetric
    perturbation ``V`` with ``min V <= V <= max V`` (as forms), Weyl's inequality
    gives ``mu_k + min V <= lambda_k(H) <= mu_k + max V``.  Each returned interval
    rigorously brackets ``lambda_k``; used to place the Lehmann ``rho`` in a
    certified spectral gap.
    """
    lo = Interval.point(float(perturbation_lower))
    hi = Interval.point(float(perturbation_upper))
    out: list[Interval] = []
    for mu in base_eigenvalues:
        mu_iv = Interval.point(float(mu))
        out.append(Interval((mu_iv + lo).lo, (mu_iv + hi).hi))
    return out


@dataclass(frozen=True)
class SpectralGapCertificate:
    """Certified ``lambda_2 - lambda_1`` for the bottom of a self-adjoint spectrum."""

    lambda1_upper: float
    lambda2_lower: float
    gap_lower: float
    rho: float
    certified: bool


def certified_spectral_gap(
    a0: Matrix,
    a1: Matrix,
    a2: Matrix,
    rho: float,
    lambda1_upper: float,
) -> SpectralGapCertificate:
    r"""Certified positive gap ``lambda_2 - lambda_1`` from a Ritz upper + Lehmann lower bound.

    ``lambda1_upper`` is any rigorous upper bound on ``lambda_1`` (e.g. a
    Rayleigh-Ritz value); the Lehmann lower bound on ``lambda_2`` is computed from
    the pencil with ``n_below = 2``.  ``gap_lower = lambda2_lower - lambda1_upper``
    is a certified lower bound on the true gap (``<= lambda_2 - lambda_1``);
    ``certified`` is ``True`` only when it is positive.
    """
    cert = lehmann_maehly_lower_bounds(a0, a1, a2, rho, n_below=2)
    lam2 = next((b.lower_bound for b in cert.bounds if b.index == 2), None)
    lam1_up = float(lambda1_upper)
    if lam2 is None:
        return SpectralGapCertificate(lam1_up, -inf, -inf, float(rho), False)
    gap = (Interval.point(lam2) - Interval.point(lam1_up)).lo
    return SpectralGapCertificate(lam1_up, lam2, gap, float(rho), gap > 0.0)


def ritz_upper_bound(matrix: Matrix, vector: Vector) -> Interval:
    r"""Rayleigh-quotient **upper** bound on ``lambda_1`` (``.hi`` is the bound).

    Any Rayleigh quotient ``R(v) = v^T A v / v^T v`` satisfies ``R(v) >= lambda_1``,
    so the upper endpoint of its enclosure is a certified upper bound on the
    smallest eigenvalue.
    """
    a = _to_sym_matrix(matrix)
    v = [Interval.from_value(x) for x in vector]
    if len(v) != len(a):
        raise ValueError("vector length must match matrix dimension")
    vv = _dot(v, v)
    if vv.lo <= 0.0:
        raise ValueError("vector must be nonzero")
    return _dot(v, _matvec_sym(a, v)) * vv.reciprocal()


__all__ = [
    "EigenvalueLowerBound",
    "Inertia",
    "LehmannCertificate",
    "SpectralGapCertificate",
    "certified_spectral_gap",
    "count_eigenvalues_below",
    "generalized_eigenvalue_enclosure",
    "interval_ldlt_inertia",
    "interval_ldlt_pivots",
    "is_positive_definite",
    "lehmann_maehly_lower_bounds",
    "operator_comparison_bounds",
    "ritz_upper_bound",
    "temple_lower_bound",
    "temple_lower_bound_vector",
]
