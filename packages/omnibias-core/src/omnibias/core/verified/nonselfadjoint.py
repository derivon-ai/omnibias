# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Finite-dimensional certified eigenvalue counts for non-self-adjoint matrices.

For a finite real matrix ``A``, the argument principle gives the algebraic
number of eigenvalues inside a contour ``Γ`` as the winding number of
``det(z I - A)`` along ``Γ``.  This module certifies both required facts:

* interval evaluation of the characteristic polynomial is passed to the
  existing contour-winding enclosure; and
* ``interval_solve`` certifies that the real block representation of
  ``z I - A`` is invertible on every contour segment, and bounds its
  infinity-norm resolvent.

The result is only an eigenvalue count for the supplied finite-dimensional
matrix (or connected interval matrix box).  It makes no operator-limit,
continuum-spectrum, or pseudospectral claim.  A failed contraction or a
non-isolated winding is returned as an explicitly inconclusive result.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import permutations
from math import isfinite

from omnibias.core.collapse.winding import (
    contour_parameter_segments,
    integers_in,
    winding_enclosure,
)
from omnibias.core.verified.complex_interval import ComplexInterval
from omnibias.core.verified.interval import Interval, IntervalLike
from omnibias.core.verified.linalg import interval_solve

Matrix = Sequence[Sequence[IntervalLike]]


@dataclass(frozen=True)
class EigenvalueCountCertificate:
    """Finite-dimensional contour count and the segment resolvent certificates.

    ``count`` is non-``None`` only when every segment was certified resolvent
    regular and the determinant winding enclosed one integer.  Each entry of
    ``resolvent_inf_norm_bounds`` is an upper bound on the infinity norm of the
    real ``2n × 2n`` block representation of ``(z I - A)^-1`` over its segment.
    """

    count: int | None
    winding: Interval | None
    resolvent_inf_norm_bounds: tuple[float, ...]
    segments: int
    certified: bool
    detail: str


def _interval_matrix(matrix: Matrix) -> list[list[Interval]]:
    rows = [[Interval.from_value(value) for value in row] for row in matrix]
    n = len(rows)
    if n == 0 or any(len(row) != n for row in rows):
        raise ValueError("matrix must be non-empty and square")
    if any(not isfinite(value) for row in rows for entry in row for value in (entry.lo, entry.hi)):
        raise ValueError("matrix endpoints must be finite")
    return rows


def _poly_mul(
    left: Sequence[ComplexInterval], right: Sequence[ComplexInterval]
) -> list[ComplexInterval]:
    result = [ComplexInterval.zero() for _ in range(len(left) + len(right) - 1)]
    for i, lhs in enumerate(left):
        for j, rhs in enumerate(right):
            result[i + j] = result[i + j] + lhs * rhs
    return result


def _permutation_sign(permutation: Sequence[int]) -> int:
    inversions = sum(
        permutation[i] > permutation[j]
        for i in range(len(permutation))
        for j in range(i + 1, len(permutation))
    )
    return -1 if inversions % 2 else 1


def characteristic_polynomial_enclosure(matrix: Matrix) -> tuple[ComplexInterval, ...]:
    """Outward-rounded coefficients of ``det(z I - A)``, low degree first.

    The Leibniz expansion is intentionally limited to small finite matrices:
    it is a transparent interval enclosure, not a scalable characteristic
    polynomial algorithm.  The returned boxes contain the coefficients for
    every matrix in the input interval box.
    """
    a = _interval_matrix(matrix)
    n = len(a)
    coeffs = [ComplexInterval.zero() for _ in range(n + 1)]
    for permutation in permutations(range(n)):
        term = [ComplexInterval.one()]
        for i, j in enumerate(permutation):
            factor = (
                [ComplexInterval(-a[i][i], Interval.point(0.0)), ComplexInterval.one()]
                if i == j
                else [ComplexInterval(-a[i][j], Interval.point(0.0))]
            )
            term = _poly_mul(term, factor)
        sign = _permutation_sign(permutation)
        for degree, coefficient in enumerate(term):
            coeffs[degree] = coeffs[degree] + sign * coefficient
    return tuple(coeffs)


def _real_block_resolvent_matrix(
    matrix: Sequence[Sequence[Interval]], z: ComplexInterval
) -> list[list[Interval]]:
    n = len(matrix)
    zero = Interval.point(0.0)
    result = [[zero for _ in range(2 * n)] for _ in range(2 * n)]
    for i in range(n):
        for j in range(n):
            real = (z.re if i == j else zero) - matrix[i][j]
            imag = z.im if i == j else zero
            result[i][j] = real
            result[i][n + j] = -imag
            result[n + i][j] = imag
            result[n + i][n + j] = real
    return result


def _resolvent_inf_norm_bound(
    matrix: Sequence[Sequence[Interval]], z: ComplexInterval
) -> float | None:
    """Certify a sup-norm resolvent bound over one complex rectangle."""
    block = _real_block_resolvent_matrix(matrix, z)
    dimension = len(block)
    columns: list[list[Interval]] = []
    for column in range(dimension):
        rhs = [0.0] * dimension
        rhs[column] = 1.0
        try:
            columns.append(interval_solve(block, rhs, max_iter=12))
        except ValueError:
            return None
    return max(
        sum(columns[column][row].abs().hi for column in range(dimension))
        for row in range(dimension)
    )


def count_eigenvalues_in_contour(
    matrix: Matrix,
    center: complex = 0j,
    radius: float = 1.0,
    *,
    segments: int = 32,
    max_segments: int = 256,
    contour: str = "circle",
    half_width: float | None = None,
    half_height: float | None = None,
) -> EigenvalueCountCertificate:
    """Certify the algebraic eigenvalue count inside a circle or rectangle.

    The contour is counter-clockwise.  Refinement doubles its segment count
    until the resolvent is certified on every segment and the determinant
    winding isolates one integer, or until ``max_segments`` is exhausted.
    """
    if max_segments < segments:
        raise ValueError("max_segments must be >= segments")
    a = _interval_matrix(matrix)
    coefficients = characteristic_polynomial_enclosure(a)
    current = segments
    last_winding: Interval | None = None
    while current <= max_segments:
        domains = contour_parameter_segments(
            center,
            radius,
            segments=current,
            contour=contour,
            half_width=half_width,
            half_height=half_height,
        )
        bounds: list[float] = []
        for domain in domains:
            bound = _resolvent_inf_norm_bound(a, domain)
            if bound is None:
                return EigenvalueCountCertificate(
                    None,
                    last_winding,
                    tuple(bounds),
                    current,
                    False,
                    "resolvent regularity is inconclusive on a contour segment",
                )
            bounds.append(bound)
        last_winding = winding_enclosure(
            coefficients,
            center,
            radius,
            segments=current,
            contour=contour,
            half_width=half_width,
            half_height=half_height,
        )
        if last_winding is not None:
            hits = integers_in(last_winding)
            if len(hits) == 1:
                return EigenvalueCountCertificate(
                    hits[0],
                    last_winding,
                    tuple(bounds),
                    current,
                    True,
                    "finite-dimensional algebraic eigenvalue count certified",
                )
        current *= 2
    return EigenvalueCountCertificate(
        None,
        last_winding,
        (),
        max_segments,
        False,
        "determinant winding did not isolate a unique integer",
    )


__all__ = [
    "EigenvalueCountCertificate",
    "characteristic_polynomial_enclosure",
    "count_eigenvalues_in_contour",
]
