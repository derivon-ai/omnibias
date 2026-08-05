# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Pade approximants + Thiele rational interpolation (exact rational, certified remainder).

The rational-function counterpart of the Taylor tower: from a power series you get a
Pade approximant ``[m/n] = P(x)/Q(x)`` whose Taylor expansion matches the series
through order ``m + n`` -- a far better extrapolant than the raw truncation, and often
convergent where the series diverges. Two honesty registers:

* **closed-form / exact** -- :func:`pade_approximant` (an exact-rational linear solve),
  :func:`thiele_interpolation` (reciprocal-difference continued fraction), and the
  evaluators are all exact :class:`~fractions.Fraction` arithmetic; the Pade equations
  hold *exactly*.
* **numerical (certified)** -- :func:`pade_certified_remainder` bounds
  ``|f(x) - P(x)/Q(x)|`` over ``|x| <= r`` from certified Taylor-coefficient
  :class:`Interval`\ s of ``f`` plus a geometric tail majorant, via the exact residual
  ``W = Q f - P`` (which vanishes through order ``m + n``) divided by a rigorous lower
  bound on ``|Q|`` -- no fudge factor.
"""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction

from omnibias.core.verified.interval import Interval

Rational = Fraction | int


def _frac(v: Rational) -> Fraction:
    return v if isinstance(v, Fraction) else Fraction(v)


def _solve_rational(matrix: list[list[Fraction]], rhs: list[Fraction]) -> list[Fraction]:
    """Exact Gauss-Jordan solve of a square rational system (raises if singular)."""
    n = len(matrix)
    aug = [[*row, rhs[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = next((r for r in range(col, n) if aug[r][col] != 0), None)
        if pivot is None:
            raise ValueError("singular Pade system (denominator order too high for the data)")
        aug[col], aug[pivot] = aug[pivot], aug[col]
        inv = Fraction(1) / aug[col][col]
        aug[col] = [x * inv for x in aug[col]]
        for r in range(n):
            if r != col and aug[r][col] != 0:
                factor = aug[r][col]
                aug[r] = [a - factor * b for a, b in zip(aug[r], aug[col], strict=True)]
    return [aug[i][n] for i in range(n)]


def pade_approximant(
    coeffs: Sequence[Rational], m: int, n: int
) -> tuple[tuple[Fraction, ...], tuple[Fraction, ...]]:
    r"""Exact Pade approximant ``[m/n]``: numerator ``P`` (deg ``m``), denominator ``Q`` (deg ``n``).

    Given Taylor coefficients ``c_0, ..., c_{m+n}`` of ``f``, returns ``(P, Q)`` (both
    ascending-order tuples, ``Q_0 = 1``) with ``P/Q = f + O(x^{m+n+1})``. Solves the
    exact-rational linear system ``sum_{j=1}^{n} Q_j c_{m+i-j} = -c_{m+i}`` (``i = 1..n``)
    for the denominator, then reads off ``P_i = sum_{j=0}^{i} Q_j c_{i-j}``. Requires
    ``len(coeffs) >= m + n + 1``.
    """
    if m < 0 or n < 0:
        raise ValueError(f"m and n must be >= 0, got m={m}, n={n}")
    c = [_frac(v) for v in coeffs]
    if len(c) < m + n + 1:
        raise ValueError(f"need >= {m + n + 1} coefficients for a [{m}/{n}] Pade, got {len(c)}")

    def cc(i: int) -> Fraction:
        return c[i] if 0 <= i < len(c) else Fraction(0)

    if n > 0:
        amat = [[cc(m + i - j) for j in range(1, n + 1)] for i in range(1, n + 1)]
        bvec = [-cc(m + i) for i in range(1, n + 1)]
        qtail = _solve_rational(amat, bvec)
    else:
        qtail = []
    q = [Fraction(1), *qtail]
    p = [sum((q[j] * cc(i - j) for j in range(0, min(i, n) + 1)), Fraction(0)) for i in range(m + 1)]
    return tuple(p), tuple(q)


def _poly_eval(coeffs: Sequence[Rational], x: Rational) -> Fraction:
    xf = _frac(x)
    return sum((_frac(c) * xf**i for i, c in enumerate(coeffs)), Fraction(0))


def pade_evaluate(
    numer: Sequence[Rational], denom: Sequence[Rational], x: Rational
) -> Fraction:
    r"""Evaluate the Pade rational ``P(x)/Q(x)`` at a rational ``x`` (exact)."""
    den = _poly_eval(denom, x)
    if den == 0:
        raise ZeroDivisionError(f"Pade denominator vanishes at x={x}")
    return _poly_eval(numer, x) / den


def _poly_eval_interval(coeffs: Sequence[Rational], x: Interval) -> Interval:
    acc = Interval.point(0.0)
    xp = Interval.point(1.0)
    for c in coeffs:
        acc = acc + Interval.from_rational(_frac(c)) * xp
        xp = xp * x
    return acc


def pade_evaluate_interval(
    numer: Sequence[Rational], denom: Sequence[Rational], x: Interval
) -> Interval:
    r"""Evaluate ``P(x)/Q(x)`` over an :class:`Interval` ``x`` (outward-rounded)."""
    return _poly_eval_interval(numer, x) * _poly_eval_interval(denom, x).reciprocal()


def rational_series(
    numer: Sequence[Rational], denom: Sequence[Rational], order: int
) -> tuple[Fraction, ...]:
    r"""Maclaurin coefficients ``[x^0..x^order]`` of ``numer/denom`` (exact long division)."""
    if order < 0:
        raise ValueError(f"order must be >= 0, got {order}")
    p = [_frac(v) for v in numer]
    q = [_frac(v) for v in denom]
    if not q or q[0] == 0:
        raise ValueError("denominator must have a non-zero constant term")
    out: list[Fraction] = []
    for i in range(order + 1):
        acc = p[i] if i < len(p) else Fraction(0)
        for j in range(1, min(i, len(q) - 1) + 1):
            acc -= q[j] * out[i - j]
        out.append(acc / q[0])
    return tuple(out)


def pade_certified_remainder(
    numer: Sequence[Rational],
    denom: Sequence[Rational],
    taylor_ivs: Sequence[Interval],
    radius: float,
    *,
    tail_bound: float,
    tail_ratio: float,
) -> Interval:
    r"""Rigorous bound on ``|f(x) - P(x)/Q(x)|`` for ``|x| <= radius``.

    ``taylor_ivs`` are certified enclosures ``a_0..a_M`` of ``f``'s Taylor coefficients
    (``M >= m + n``), and ``|a_k| <= tail_bound * tail_ratio^k`` for every ``k > M``.
    The residual ``W(x) = Q(x) f(x) - P(x)`` vanishes through order ``m + n`` by
    construction, so ``W_k = sum_{j<=n} Q_j a_{k-j}`` for ``k > m + n``; the finite part
    (``m+n < k <= M``) is summed as intervals and the ``k > M`` tail is geometrically
    majorised (needs ``radius * tail_ratio < 1``). Dividing by the rigorous lower bound
    ``|Q(x)| >= |Q_0| - sum_{j>=1} |Q_j| radius^j > 0`` gives the certified remainder --
    outward-rounded, no fudge factor.
    """
    if radius <= 0.0:
        raise ValueError(f"radius must be > 0, got {radius}")
    if not 0.0 <= tail_ratio:
        raise ValueError(f"tail_ratio must be >= 0, got {tail_ratio}")
    if radius * tail_ratio >= 1.0:
        raise ValueError("need radius * tail_ratio < 1 for a convergent geometric tail")
    q = [_frac(v) for v in denom]
    p = [_frac(v) for v in numer]
    m = len(p) - 1
    n = len(q) - 1
    big_m = len(taylor_ivs) - 1
    if big_m < m + n:
        raise ValueError(f"need >= {m + n + 1} Taylor enclosures, got {big_m + 1}")

    r_iv = Interval.from_value(radius)

    # Finite residual W_k = sum_{j=0}^{n} Q_j a_{k-j} for m+n < k <= M.
    w_finite = Interval.point(0.0)
    for k in range(m + n + 1, big_m + 1):
        wk = Interval.point(0.0)
        for j in range(0, n + 1):
            if 0 <= k - j <= big_m:
                wk = wk + Interval.from_rational(q[j]) * taylor_ivs[k - j]
        w_finite = w_finite + Interval(-wk.mag, wk.mag) * r_iv.pow_int(k)

    # Tail k > M: |W_k| <= tail_bound * (sum_j |Q_j| tail_ratio^{-j}) * tail_ratio^k.
    qsum = sum(
        (abs(float(q[j])) * tail_ratio ** (-j) if tail_ratio > 0 else abs(float(q[j])))
        for j in range(0, n + 1)
    )
    sr = radius * tail_ratio
    tail_geo = tail_bound * qsum * (sr ** (big_m + 1)) / (1.0 - sr) if sr > 0 else 0.0
    w_tail = Interval(-tail_geo, tail_geo)
    numerator_bound = (w_finite + w_tail).mag

    # Rigorous lower bound on |Q(x)| over |x| <= radius.
    q_lo = abs(float(q[0])) - sum(abs(float(q[j])) * radius**j for j in range(1, n + 1))
    if q_lo <= 0.0:
        raise ValueError("cannot certify |Q| > 0 over the disc; shrink radius")
    return Interval(-numerator_bound / q_lo, numerator_bound / q_lo)


def thiele_interpolation(
    xs: Sequence[Rational], ys: Sequence[Rational]
) -> tuple[Fraction, ...]:
    r"""Thiele continued-fraction coefficients from samples (exact rational).

    Returns ``a_0, ..., a_{N-1}`` for the interpolating continued fraction ``f(x) = a_0 +
    (x - x_0)/(a_1 + (x - x_1)/(a_2 + ...))``, built from the reciprocal-difference table
    (``a_k = rho_k(x_0) - rho_{k-2}(x_0)``). Raises :class:`ZeroDivisionError` at an
    *unattainable point* (a reciprocal difference with a zero denominator -- e.g. sampling
    a low-degree rational at too many nodes); :func:`thiele_evaluate` inverts it.
    """
    xf = [_frac(v) for v in xs]
    yf = [_frac(v) for v in ys]
    if len(xf) != len(yf):
        raise ValueError("xs and ys must have equal length")
    if len(set(xf)) != len(xf):
        raise ValueError("interpolation nodes must be distinct")
    size = len(xf)
    rho = [[Fraction(0)] * size for _ in range(size)]
    for i in range(size):
        rho[i][0] = yf[i]
    for j in range(1, size):
        for i in range(size - j):
            denom = rho[i][j - 1] - rho[i + 1][j - 1]
            if denom == 0:
                raise ZeroDivisionError("unattainable point in the Thiele continued fraction")
            correction = rho[i + 1][j - 2] if j >= 2 else Fraction(0)
            rho[i][j] = correction + (xf[i] - xf[i + j]) / denom
    return tuple(rho[0][k] - (rho[0][k - 2] if k >= 2 else Fraction(0)) for k in range(size))


def thiele_evaluate(
    xs: Sequence[Rational], coeffs: Sequence[Rational], x: Rational
) -> Fraction:
    r"""Evaluate a Thiele continued fraction (from :func:`thiele_interpolation`) at ``x``."""
    xf = [_frac(v) for v in xs]
    a = [_frac(v) for v in coeffs]
    if not a:
        raise ValueError("coeffs must be non-empty")
    xq = _frac(x)
    val = a[-1]
    for j in range(len(a) - 2, -1, -1):
        if val == 0:
            raise ZeroDivisionError(f"Thiele continued fraction has a pole near x={x}")
        val = a[j] + (xq - xf[j]) / val
    return val


__all__ = [
    "pade_approximant",
    "pade_certified_remainder",
    "pade_evaluate",
    "pade_evaluate_interval",
    "rational_series",
    "thiele_evaluate",
    "thiele_interpolation",
]
