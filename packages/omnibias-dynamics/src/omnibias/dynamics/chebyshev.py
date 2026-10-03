# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Extended-Chebyshev / Wronskian zero-count certificates over exact Q."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Sequence

from omnibias.core.realization.algebraic import root_count, sturm_sequence
from omnibias.holonomic._core.poly_n import PolyN

__all__ = [
    "BudanFourierCertificate",
    "ECTCertificate",
    "budan_fourier_negative_roots",
    "certify_budan_fourier",
    "certify_ect",
    "wronskian_determinant",
]


def wronskian_determinant(functions: Sequence[PolyN], order: int) -> PolyN:
    """Exact-Q Wronskian determinant of the first ``order`` functions."""
    if order < 1:
        raise ValueError("order must be positive")
    if len(functions) < order:
        raise ValueError("need at least order many basis functions")
    rows: list[list[PolyN]] = []
    for j in range(order):
        row = []
        for func in functions[:order]:
            deriv = func
            for _ in range(j):
                deriv = deriv.partial(0)
            row.append(deriv)
        rows.append(row)
    return _determinant(rows)


def _determinant(matrix: list[list[PolyN]]) -> PolyN:
    n = len(matrix)
    if n == 1:
        return matrix[0][0]
    if n == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    total = PolyN.zero(matrix[0][0].nvars)
    for col in range(n):
        minor = [[matrix[row][other] for other in range(n) if other != col] for row in range(1, n)]
        sign = Fraction(1) if col % 2 == 0 else Fraction(-1)
        total = total + sign * matrix[0][col] * _determinant(minor)
    return total


@dataclass(frozen=True)
class ECTCertificate:
    """A Wronskian nonvanishing certificate on a declared basis."""

    wronskian: PolyN
    order: int
    proved_ect_on_interval: bool

    def to_payload(self) -> dict[str, object]:
        return {
            "order": self.order,
            "wronskian_degree": max((sum(mon) for mon in self.wronskian.terms), default=0),
            "proved_ect_on_interval": self.proved_ect_on_interval,
        }


def certify_ect(functions: Sequence[PolyN], *, order: int) -> ECTCertificate:
    wronskian = wronskian_determinant(functions, order)
    proved = not wronskian.is_zero()
    return ECTCertificate(wronskian, order, proved)


@dataclass(frozen=True)
class BudanFourierCertificate:
    polynomial: PolyN
    sign_changes_at_zero: int
    sign_changes_at_positive: int
    negative_root_upper_bound: int

    def to_payload(self) -> dict[str, object]:
        return {
            "sign_changes_at_zero": self.sign_changes_at_zero,
            "sign_changes_at_positive": self.sign_changes_at_positive,
            "negative_root_upper_bound": self.negative_root_upper_bound,
        }


def _sign_changes(coeffs: Sequence[Fraction]) -> int:
    signs = [1 if c > 0 else -1 if c < 0 else 0 for c in coeffs]
    filtered = [s for s in signs if s != 0]
    if len(filtered) < 2:
        return 0
    return sum(1 for a, b in zip(filtered, filtered[1:], strict=False) if a != b)


def _binom(n: int, k: int) -> Fraction:
    if k < 0 or k > n:
        return Fraction(0)
    num = Fraction(1)
    for i in range(k):
        num *= Fraction(n - i, i + 1)
    return num


def budan_fourier_negative_roots(poly: PolyN) -> int:
    """Descartes/Budan upper bound on negative real roots (univariate)."""
    if poly.nvars != 1:
        raise ValueError("budan_fourier_negative_roots requires a univariate polynomial")
    degree = max((mon[0] for mon in poly.terms), default=0)
    coeffs = [poly.terms.get((deg,), Fraction(0)) for deg in range(degree + 1)]
    transformed = [
        sum(coeffs[j] * Fraction(-1) ** (j - i) * _binom(j, i) for j in range(i, degree + 1))
        for i in range(degree + 1)
    ]
    return _sign_changes(list(reversed(transformed)))


def certify_budan_fourier(poly: PolyN) -> BudanFourierCertificate:
    if poly.nvars != 1:
        raise ValueError("certify_budan_fourier requires a univariate polynomial")
    degree = max((mon[0] for mon in poly.terms), default=0)
    coeffs = [poly.terms.get((deg,), Fraction(0)) for deg in range(degree + 1)]
    at_zero = _sign_changes(coeffs)
    at_pos = budan_fourier_negative_roots(poly)
    upper = max(0, at_zero - at_pos)
    return BudanFourierCertificate(poly, at_zero, at_pos, upper)


def _poly_to_upoly(poly: PolyN) -> tuple[Fraction, ...]:
    if poly.nvars != 1:
        raise ValueError("_poly_to_upoly requires a univariate polynomial")
    degree = max((mon[0] for mon in poly.terms), default=0)
    return tuple(poly.terms.get((deg,), Fraction(0)) for deg in range(degree + 1))


def sturm_root_count(poly: PolyN, lo: Fraction, hi: Fraction) -> int:
    """Exact Sturm count of distinct real roots in ``(lo, hi]``."""
    return root_count(_poly_to_upoly(poly), lo, hi)
