# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Guess a P-recursive / D-finite / algebraic annihilator from exact samples.

The *guessing* step is heuristic -- it fits the minimal exact homogeneous relation to a finite
prefix (an exact rational null space, so the fitted relation is satisfied by every supplied
sample by construction). Whether it continues to hold for *all* ``n`` is a separate claim,
discharged by :mod:`omnibias.holonomic._core.certify` (verified on a range; all-``n`` is the
holonomic continuation / Zeilberger obligation). Three guessers share the discipline:

* :func:`guess_recurrence` -- the minimal P-recurrence ``sum_j p_j(n) a_{n-j} = 0`` (via
  :func:`omnibias.symbolic.discover_recurrence`), returned as a forward shift operator;
* :func:`guess_dfinite` -- the minimal **differential** annihilator ``sum_i c_i(x) D^i f = 0``
  of a power series (null space on the ``apply_series`` map);
* :func:`guess_algebraic` -- the minimal **algebraic** equation ``P(x, y) = 0`` (``y = f(x)``)
  satisfied by a power series (null space over the ``x^i y^j`` monomials).

Each is **guessed** (which order / degrees) but **verified exactly** on the whole supplied
prefix -- unlike a floating-point or monic least-squares fit, the recovered coefficients are
exact rationals and the residual is identically zero, not merely small.
"""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction

from omnibias.holonomic._core.linalg import null_space
from omnibias.holonomic._core.ore import OrePolynomial, diff_algebra, shift_algebra
from omnibias.holonomic._core.poly2 import Poly2
from omnibias.holonomic._core.rational_poly import Poly, pshift, to_poly


def recurrence_to_operator(rel: object) -> OrePolynomial:
    """Convert a symbolic ``RecurrenceRelation`` (lag form) to a forward shift operator.

    The symbolic relation is ``sum_{j=0}^{r} p_j(n) a_{n-j} = 0``; substituting ``n = m + r``
    and ``i = r - j`` gives the forward operator ``sum_{i=0}^{r} c_i(m) a_{m+i} = 0`` with
    ``c_i(m) = p_{r-i}(m + r)``.
    """
    order: int = rel.order  # type: ignore[attr-defined]
    coeffs: Sequence[Sequence[Fraction]] = rel.coefficients  # type: ignore[attr-defined]
    algebra = shift_algebra()
    forward: list[Poly] = []
    for i in range(order + 1):
        j = order - i
        forward.append(pshift(to_poly(coeffs[j]), order))
    return algebra.operator(forward)


def guess_recurrence(
    samples: Sequence[Fraction | int],
    *,
    max_order: int = 4,
    max_index_degree: int = 3,
) -> OrePolynomial | None:
    """Guess the minimal P-recursive annihilator of ``samples`` as a forward shift operator.

    Returns ``None`` when no polynomial recurrence up to the search bounds fits (a genuine
    finding -- the sequence is not P-recursive within those bounds).
    """
    from omnibias.symbolic import discover_recurrence

    rel = discover_recurrence(
        [Fraction(s) for s in samples],
        max_order=max_order,
        max_index_degree=max_index_degree,
    )
    if rel is None:
        return None
    return recurrence_to_operator(rel)


def _falling_value(m: int, i: int) -> Fraction:
    r"""The falling factorial ``m^{\underline i} = m(m-1)...(m-i+1)`` (exact)."""
    out = Fraction(1)
    for t in range(i):
        out *= m - t
    return out


def guess_dfinite(
    series: Sequence[Fraction | int],
    *,
    max_order: int = 4,
    max_degree: int = 4,
) -> OrePolynomial | None:
    r"""Guess the minimal differential annihilator ``sum_i c_i(x) D^i f = 0`` of a series.

    ``series`` are the Taylor coefficients ``a_0, a_1, ...``. Fits ``c_i`` (polynomials in
    ``x`` of degree ``<= max_degree``) so the coefficient of every checked ``x^s`` in
    ``L[f]`` vanishes exactly (rational null space); the smallest order / degree that fits and
    verifies on the whole prefix is returned. ``None`` is a genuine finding -- the series is
    not D-finite within the bounds.
    """
    a = [Fraction(v) for v in series]
    M = len(a)
    min_check = 3  # held-out equations required to reject an over-fit
    for r in range(1, max_order + 1):
        for D in range(max_degree + 1):
            unknowns = [(i, j) for i in range(r + 1) for j in range(D + 1)]
            ncol = len(unknowns)
            max_s = M - 1 - r
            fit_count = ncol - 1
            if max_s + 1 < fit_count + min_check:
                continue  # not enough held-out data to guess reliably
            rows: list[list[Fraction]] = []
            for s in range(fit_count):
                row: list[Fraction] = []
                for i, j in unknowns:
                    m = s + i - j
                    row.append(_falling_value(m, i) * a[m] if 0 <= m < M else Fraction(0))
                rows.append(row)
            for sol in null_space(rows):
                coeffs = [[Fraction(0)] * (D + 1) for _ in range(r + 1)]
                for idx, (i, j) in enumerate(unknowns):
                    coeffs[i][j] = sol[idx]
                op = diff_algebra().operator(coeffs)
                if op.order != r:
                    continue
                if all(op.apply_series(a, s) == 0 for s in range(max_s + 1)):
                    return op
    return None


def _series_mul(u: Sequence[Fraction], v: Sequence[Fraction], m: int) -> list[Fraction]:
    out = [Fraction(0)] * m
    for s in range(m):
        if u[s] == 0:
            continue
        for t in range(m - s):
            if v[t]:
                out[s + t] += u[s] * v[t]
    return out


def guess_algebraic(
    series: Sequence[Fraction | int],
    *,
    max_x_degree: int = 4,
    max_y_degree: int = 4,
) -> Poly2 | None:
    r"""Guess the minimal algebraic equation ``P(x, y) = 0`` (``y = f(x)``) for a series.

    ``series`` are the Taylor coefficients of ``f``. Returns the minimal-``y``-degree bivariate
    polynomial (a :data:`~.poly2.Poly2` keyed by ``(x-power, y-power)``, normalised so the
    lexicographically-least non-zero coefficient is ``1``) whose composition ``P(x, f(x))``
    vanishes as a power series on the whole prefix. ``None`` is a genuine finding -- ``f`` is
    not algebraic within the bounds.
    """
    a = [Fraction(v) for v in series]
    M = len(a)
    if M == 0:
        return None
    powers: list[list[Fraction]] = [[Fraction(0)] * M for _ in range(max_y_degree + 1)]
    powers[0][0] = Fraction(1)
    for j in range(1, max_y_degree + 1):
        powers[j] = _series_mul(powers[j - 1], a, M)
    min_check = 3  # held-out equations required to reject an over-fit
    for dy in range(1, max_y_degree + 1):
        for dx in range(max_x_degree + 1):
            unknowns = [(i, j) for j in range(dy + 1) for i in range(dx + 1)]
            ncol = len(unknowns)
            fit_count = ncol - 1
            if M < fit_count + min_check:
                continue  # not enough held-out data to guess reliably
            rows: list[list[Fraction]] = []
            for s in range(fit_count):
                row = [powers[j][s - i] if 0 <= s - i < M else Fraction(0) for i, j in unknowns]
                rows.append(row)
            for sol in null_space(rows):
                terms = {(i, j): sol[idx] for idx, (i, j) in enumerate(unknowns) if sol[idx] != 0}
                if not any(j >= 1 for (_i, j) in terms):
                    continue  # must genuinely involve y
                lead = terms[min(terms, key=lambda key: (key[1], key[0]))]
                poly: Poly2 = {key: val / lead for key, val in terms.items()}
                residual = [Fraction(0)] * M
                for (i, j), c in poly.items():
                    for s in range(M):
                        if 0 <= s - i < M and powers[j][s - i]:
                            residual[s] += c * powers[j][s - i]
                if all(v == 0 for v in residual):
                    return poly
    return None


__all__ = [
    "guess_algebraic",
    "guess_dfinite",
    "guess_recurrence",
    "recurrence_to_operator",
]
