# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Exact rational recurrence fitting on finite sequence samples.

A fitted relation is verified on supplied samples; continuation beyond that
prefix requires an independent mathematical argument.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction

from omnibias.core.proof.lift import integer_null_space

__all__ = ["RecurrenceRelation", "discover_recurrence", "try_recurrence"]

Number = int | Fraction

def _as_fractions(samples: Sequence[Number]) -> list[Fraction]:
    return [v if isinstance(v, Fraction) else Fraction(v) for v in samples]


@dataclass(frozen=True)
class RecurrenceRelation:
    r"""A homogeneous P-recursive relation ``sum_j p_j(n) a_{n-j} = 0`` (exact).

    ``coefficients[j][d]`` is the ``n^d`` coefficient of the polynomial ``p_j`` that
    multiplies the lag-``j`` term ``a_{n-j}``. ``order`` is the recurrence order (max
    lag) and ``index_degree`` the max polynomial degree in ``n``.
    """

    order: int
    index_degree: int
    coefficients: tuple[tuple[Fraction, ...], ...]

    def coefficient_poly(self, lag: int, n: int) -> Fraction:
        """Evaluate the polynomial ``p_lag(n)`` at integer ``n``."""
        acc = Fraction(0)
        for degree, coef in enumerate(self.coefficients[lag]):
            acc += coef * Fraction(n) ** degree
        return acc

    def evaluate_residuals(self, samples: Sequence[Number]) -> list[Fraction]:
        r"""``sum_j p_j(n) a_{n-j}`` for every valid ``n`` (all zero iff satisfied)."""
        vals = _as_fractions(samples)
        residuals: list[Fraction] = []
        for n in range(self.order, len(vals)):
            acc = Fraction(0)
            for lag in range(self.order + 1):
                acc += self.coefficient_poly(lag, n) * vals[n - lag]
            residuals.append(acc)
        return residuals

    def max_abs_residual(self, samples: Sequence[Number]) -> Fraction:
        residuals = self.evaluate_residuals(samples)
        return max((abs(r) for r in residuals), default=Fraction(0))

    def is_satisfied_by(self, samples: Sequence[Number]) -> bool:
        return all(r == 0 for r in self.evaluate_residuals(samples))

    def pretty(self, symbol: str = "a") -> str:
        """Human-readable relation, e.g. ``(n + 1) a[n] + (-4 n + 2) a[n-1] = 0``."""
        pieces: list[str] = []
        for lag in range(self.order + 1):
            poly = _format_poly(self.coefficients[lag])
            if poly is None:
                continue
            term = f"{symbol}[n]" if lag == 0 else f"{symbol}[n-{lag}]"
            pieces.append(f"({poly}) {term}")
        body = " + ".join(pieces) if pieces else "0"
        return f"{body} = 0"


def _format_poly(coeffs: Sequence[Fraction]) -> str | None:
    """Render a polynomial in ``n``; return ``None`` if identically zero."""
    terms: list[str] = []
    for degree, coef in enumerate(coeffs):
        if coef == 0:
            continue
        c = _format_fraction(coef)
        if degree == 0:
            terms.append(c)
        elif degree == 1:
            terms.append("n" if coef == 1 else ("-n" if coef == -1 else f"{c} n"))
        else:
            terms.append(f"n^{degree}" if coef == 1 else f"{c} n^{degree}")
    if not terms:
        return None
    out = terms[0]
    for term in terms[1:]:
        out += f" - {term[1:]}" if term.startswith("-") else f" + {term}"
    return out


def _format_fraction(value: Fraction) -> str:
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


def _clear_denominators(rows: list[list[Fraction]]) -> list[list[int]]:
    """Scale each row by the lcm of its denominators -> integer rows (null space kept)."""
    int_rows: list[list[int]] = []
    for row in rows:
        lcm = 1
        for value in row:
            den = value.denominator
            lcm = lcm * den // _gcd(lcm, den)
        int_rows.append([int(value * lcm) for value in row])
    return int_rows


def _gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return abs(a)


def _vector_to_coefficients(
    vector: Sequence[int], order: int, index_degree: int
) -> tuple[tuple[Fraction, ...], ...]:
    width = index_degree + 1
    return tuple(
        tuple(Fraction(vector[lag * width + d]) for d in range(width))
        for lag in range(order + 1)
    )


def _poly_is_zero(coeffs: Sequence[Fraction]) -> bool:
    return all(c == 0 for c in coeffs)


def try_recurrence(
    samples: Sequence[Number],
    order: int,
    index_degree: int,
    *,
    min_equation_margin: int = 2,
) -> RecurrenceRelation | None:
    """Exact nullity-one P-recurrence at one ``(order, index_degree)`` cell."""

    if order < 1:
        raise ValueError(f"order must be >= 1, got {order}")
    if index_degree < 0:
        raise ValueError(f"index_degree must be >= 0, got {index_degree}")
    vals = _as_fractions(samples)
    width = index_degree + 1
    unknowns = (order + 1) * width
    row_indices = list(range(order, len(vals)))
    if len(row_indices) < unknowns + min_equation_margin:
        return None
    design: list[list[Fraction]] = []
    for n in row_indices:
        row: list[Fraction] = []
        for lag in range(order + 1):
            a_shift = vals[n - lag]
            for d in range(width):
                row.append(Fraction(n) ** d * a_shift)
        design.append(row)
    null = integer_null_space(_clear_denominators(design))
    if len(null) != 1:
        return None
    coeffs = _vector_to_coefficients(null[0], order, index_degree)
    if _poly_is_zero(coeffs[0]) or _poly_is_zero(coeffs[order]):
        return None
    relation = RecurrenceRelation(order=order, index_degree=index_degree, coefficients=coeffs)
    if relation.is_satisfied_by(vals):
        return relation
    return None


def discover_recurrence(
    samples: Sequence[Number],
    *,
    max_order: int = 4,
    max_index_degree: int = 3,
    min_equation_margin: int = 2,
) -> RecurrenceRelation | None:
    r"""Recover the minimal exact P-recursive relation ``sum_j p_j(n) a_{n-j} = 0``.

    Searches ``(order, index_degree)`` in increasing order (order-major, so the
    lowest-order / lowest-degree relation wins) and, for each, solves for the exact
    homogeneous null space of the design whose columns are ``n^d a_{n-j}``
    (``j = 0 .. order``, ``d = 0 .. index_degree``). A **unique** (nullity-one) null
    vector whose leading (``a_n``) and trailing (``a_{n-order}``) coefficient
    polynomials are both non-trivial is returned as a :class:`RecurrenceRelation`;
    otherwise the search continues.

    Unlike a monic least-squares fit, the leading coefficient ``p_0(n)``
    is free to be a non-constant polynomial, so genuinely P-recursive sequences whose
    natural relation is *not* monic -- e.g. Catalan ``(n+1)C_n - (4n-2)C_{n-1} = 0`` --
    are recovered exactly. Returns ``None`` when no unique relation fits the supplied prefix within
    the search bounds. Neither a fit nor ``None`` proves a global sequence law.
    """
    if max_order < 1:
        raise ValueError(f"max_order must be >= 1, got {max_order}")
    if max_index_degree < 0:
        raise ValueError(f"max_index_degree must be >= 0, got {max_index_degree}")
    for order in range(1, max_order + 1):
        for index_degree in range(max_index_degree + 1):
            relation = try_recurrence(
                samples,
                order,
                index_degree,
                min_equation_margin=min_equation_margin,
            )
            if relation is not None:
                return relation
    return None
