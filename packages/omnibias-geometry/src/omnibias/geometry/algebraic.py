# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Exact rational certificates for smooth projective plane curves.

Three polynomial Bezout identities exclude complex projective singularities.
Opposite strict signs on rectangular annulus boundaries force distinct ovals.
When these ovals (and the obligatory odd-degree pseudoline) attain Harnack's
bound, their checked nesting gives the complete real scheme, up to ambient
isotopy. It does not determine rigid isotopy or complex orientations.

The annuli lie in one declared projective affine chart. Failure to find such
annuli or bounded-degree Bezout identities is inconclusive, not exclusion.
All coefficient arithmetic is rational. No caller assertion of smoothness,
regularity, or topology is accepted as a premise.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from math import comb

from omnibias.core.realization.polynomial import (
    AlgebraBudgetExceeded,
    Rational,
    SparsePolynomial,
    rational,
)


@dataclass(frozen=True)
class HomogeneousPlaneCurve:
    """A nonzero homogeneous rational polynomial in the order X, Y, Z."""

    polynomial: SparsePolynomial

    def __post_init__(self) -> None:
        p = self.polynomial
        degrees = {sum(index) for index, _ in p.terms}
        if p.nvars != 3 or len(degrees) != 1 or next(iter(degrees), 0) < 1:
            raise ValueError("a nonzero homogeneous polynomial of positive degree is required")

    @property
    def degree(self) -> int:
        return sum(self.polynomial.terms[0][0])

    @property
    def digest(self) -> str:
        raw = json.dumps(self.polynomial.to_payload(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode()).hexdigest()


def affine_chart(polynomial: SparsePolynomial, fixed_axis: int) -> SparsePolynomial:
    """Set coordinate ``fixed_axis`` to one, retaining the other axes in order."""
    if polynomial.nvars != 3 or type(fixed_axis) is not int or fixed_axis not in (0, 1, 2):
        raise ValueError("a ternary polynomial and a projective chart 0, 1, or 2 are required")
    terms: dict[tuple[int, ...], Fraction] = {}
    for index, coefficient in polynomial.terms:
        key = index[:fixed_axis] + index[fixed_axis + 1:]
        terms[key] = terms.get(key, Fraction(0)) + coefficient
    return SparsePolynomial(2, terms, budget=polynomial.budget)


@dataclass(frozen=True)
class ChartBezoutIdentity:
    """Multipliers for the three homogeneous partials after dehomogenization.

    Their sum of products must be the constant one in Q[u,v]. This rules out
    every common zero over C, including nonreal zeros.
    """

    fixed_axis: int
    multipliers: tuple[SparsePolynomial, SparsePolynomial, SparsePolynomial]

    def __post_init__(self) -> None:
        if type(self.fixed_axis) is not int or self.fixed_axis not in (0, 1, 2):
            raise ValueError("invalid projective chart")
        if len(self.multipliers) != 3 or any(p.nvars != 2 for p in self.multipliers):
            raise ValueError("three bivariate multipliers are required")

    def verifies(self, curve: HomogeneousPlaneCurve) -> bool:
        result = SparsePolynomial.constant(2, 0, budget=curve.polynomial.budget)
        for axis, multiplier in enumerate(self.multipliers):
            result = result + multiplier * affine_chart(curve.polynomial.derivative(axis), self.fixed_axis)
        return result.terms == (((0, 0), Fraction(1)),)


@dataclass(frozen=True)
class ProjectiveSmoothnessWitness:
    """Source-bound identities covering all three charts of CP2."""

    curve_digest: str
    charts: tuple[ChartBezoutIdentity, ChartBezoutIdentity, ChartBezoutIdentity]

    def verifies(self, curve: HomogeneousPlaneCurve) -> bool:
        return (
            self.curve_digest == curve.digest
            and len(self.charts) == 3
            and tuple(c.fixed_axis for c in self.charts) == (0, 1, 2)
            and all(c.verifies(curve) for c in self.charts)
        )


def _linear_solve(rows: list[list[Fraction]], ncols: int) -> list[Fraction] | None:
    """Exact elimination; free variables are set to zero."""
    pivot_rows: list[tuple[int, int]] = []
    next_row = 0
    for column in range(ncols):
        pivot = next((r for r in range(next_row, len(rows)) if rows[r][column]), None)
        if pivot is None:
            continue
        rows[next_row], rows[pivot] = rows[pivot], rows[next_row]
        factor = rows[next_row][column]
        rows[next_row] = [v / factor for v in rows[next_row]]
        for r in range(len(rows)):
            if r != next_row and rows[r][column]:
                factor = rows[r][column]
                rows[r] = [a - factor * b for a, b in zip(rows[r], rows[next_row], strict=True)]
        pivot_rows.append((column, next_row))
        next_row += 1
        if next_row == len(rows):
            break
    if any(not any(row[:ncols]) and row[-1] for row in rows):
        return None
    solution = [Fraction(0)] * ncols
    for column, row in pivot_rows:
        solution[column] = rows[row][-1]
    return solution


def find_smoothness_witness(
    curve: HomogeneousPlaneCurve,
    *,
    max_multiplier_degree: int = 4,
    max_unknowns: int = 512,
) -> ProjectiveSmoothnessWitness | None:
    """Search a bounded rational linear system for Nullstellensatz identities.

    ``None`` means no witness at the requested degree. Even for a nonsingular
    curve a higher degree may be needed. The explicit unknown budget prevents
    accidentally starting an unbounded symbolic search.
    """
    if type(max_multiplier_degree) is not int or max_multiplier_degree < 0:
        raise ValueError("the multiplier degree must be a nonnegative integer")
    if type(max_unknowns) is not int or max_unknowns < 1:
        raise ValueError("the unknown budget must be positive")
    count = (max_multiplier_degree + 1) * (max_multiplier_degree + 2) // 2
    if 3 * count > max_unknowns:
        raise AlgebraBudgetExceeded("smoothness identity unknown budget exceeded")
    indices = [(i, total - i) for total in range(max_multiplier_degree + 1) for i in range(total + 1)]
    witnesses: list[ChartBezoutIdentity] = []
    for chart in range(3):
        generators = [affine_chart(curve.polynomial.derivative(axis), chart) for axis in range(3)]
        products = [
            SparsePolynomial(2, {index: 1}, budget=curve.polynomial.budget) * generator
            for generator in generators for index in indices
        ]
        all_indices = sorted({(0, 0)} | {index for p in products for index, _ in p.terms})
        columns = [dict(p.terms) for p in products]
        rows = [
            [column.get(index, Fraction(0)) for column in columns]
            + [Fraction(int(index == (0, 0)))]
            for index in all_indices
        ]
        solution = _linear_solve(rows, len(products))
        if solution is None:
            return None
        multipliers = [
            SparsePolynomial(2, dict(zip(indices, solution[k * count:(k + 1) * count], strict=True)),
                             budget=curve.polynomial.budget)
            for k in range(3)
        ]
        witnesses.append(ChartBezoutIdentity(chart, (multipliers[0], multipliers[1], multipliers[2])))
    witness = ProjectiveSmoothnessWitness(curve.digest, (witnesses[0], witnesses[1], witnesses[2]))
    if not witness.verifies(curve):
        raise ArithmeticError("constructed smoothness identity failed exact replay")
    return witness


@dataclass(frozen=True)
class RationalRectangle:
    """A closed rectangle with strictly positive widths in an affine chart."""

    xmin: Rational
    xmax: Rational
    ymin: Rational
    ymax: Rational

    def __post_init__(self) -> None:
        for name in ("xmin", "xmax", "ymin", "ymax"):
            object.__setattr__(self, name, rational(getattr(self, name)))
        if self.xmin >= self.xmax or self.ymin >= self.ymax:
            raise ValueError("rectangle bounds must be strictly ordered")

    def strictly_contains(self, other: RationalRectangle) -> bool:
        return (self.xmin < other.xmin < other.xmax < self.xmax
                and self.ymin < other.ymin < other.ymax < self.ymax)

    def disjoint(self, other: RationalRectangle) -> bool:
        return (self.xmax < other.xmin or other.xmax < self.xmin
                or self.ymax < other.ymin or other.ymax < self.ymin)

    @property
    def vertices(self) -> tuple[tuple[Rational, Rational], ...]:
        return ((self.xmin, self.ymin), (self.xmax, self.ymin),
                (self.xmax, self.ymax), (self.xmin, self.ymax))


@dataclass(frozen=True)
class RectangularAnnulus:
    """The closed region between a rectangle and a strictly enclosed one."""

    inner: RationalRectangle
    outer: RationalRectangle

    def __post_init__(self) -> None:
        if not self.outer.strictly_contains(self.inner):
            raise ValueError("the outer rectangle must strictly contain the inner rectangle")


def segment_bernstein_coefficients(
    polynomial: SparsePolynomial,
    start: Sequence[Rational],
    end: Sequence[Rational],
) -> tuple[Fraction, ...]:
    """Exact Bernstein coefficients on the entire affine line segment."""
    if len(start) != polynomial.nvars or len(end) != polynomial.nvars:
        raise ValueError("segment dimension does not match the polynomial")
    a, b = tuple(map(rational, start)), tuple(map(rational, end))
    degree = max((sum(index) for index, _ in polynomial.terms), default=0)
    power = [Fraction(0)] * (degree + 1)
    for index, coefficient in polynomial.terms:
        term = [coefficient]
        for axis, exponent in enumerate(index):
            factor = [Fraction(comb(exponent, k)) * a[axis]**(exponent - k)
                      * (b[axis] - a[axis])**k for k in range(exponent + 1)]
            product = [Fraction(0)] * (len(term) + exponent)
            for i, x in enumerate(term):
                for j, y in enumerate(factor):
                    product[i + j] += x * y
            term = product
        for i, value in enumerate(term):
            power[i] += value
    return tuple(sum((power[k] * Fraction(comb(j, k), comb(degree, k))
                      for k in range(j + 1)), Fraction(0)) for j in range(degree + 1))


def _edge_margin(
    polynomial: SparsePolynomial,
    start: tuple[Rational, Rational],
    end: tuple[Rational, Rational],
    sign: int,
    depth: int,
) -> Fraction:
    margin = min(sign * c for c in segment_bernstein_coefficients(polynomial, start, end))
    if margin > 0:
        return margin
    if depth == 0:
        raise ValueError("whole-edge strict sign is not certified at this subdivision depth")
    midpoint = (Fraction(start[0] + end[0], 2), Fraction(start[1] + end[1], 2))
    return min(_edge_margin(polynomial, start, midpoint, sign, depth - 1),
               _edge_margin(polynomial, midpoint, end, sign, depth - 1))


def _boundary_sign(
    polynomial: SparsePolynomial, rectangle: RationalRectangle, depth: int,
) -> tuple[int, Fraction]:
    vertices = rectangle.vertices
    value = polynomial.evaluate(vertices[0])
    if value == 0:
        raise ValueError("a boundary vertex lies on the curve")
    sign = 1 if value > 0 else -1
    margin = min(_edge_margin(polynomial, vertices[i], vertices[(i + 1) % 4], sign, depth)
                 for i in range(4))
    return sign, margin


def _nesting(annuli: tuple[RectangularAnnulus, ...]) -> tuple[int, ...]:
    for i, a in enumerate(annuli):
        for b in annuli[i + 1:]:
            if not (a.outer.disjoint(b.outer) or a.inner.strictly_contains(b.outer)
                    or b.inner.strictly_contains(a.outer)):
                raise ValueError("annuli must have separated or strictly nested enclosing rectangles")
    parents = []
    for i, annulus in enumerate(annuli):
        containers = [j for j, candidate in enumerate(annuli)
                      if j != i and candidate.inner.strictly_contains(annulus.outer)]
        immediate = [j for j in containers if not any(
            k != j and annuli[j].inner.strictly_contains(annuli[k].outer) for k in containers)]
        parents.append(immediate[0] if immediate else -1)
    return tuple(parents)


@dataclass(frozen=True)
class ProjectiveCurveCertificate:
    """Replayable inputs and derived conclusions, bound to actual coefficients.

    ``barrier_parents`` indexes the input annuli, with -1 denoting the exterior.
    It is the entire oval nesting forest only if ``complete_real_scheme`` is
    true. In the other case additional curve components have not been excluded.
    No formal theorem-prover verification is claimed by this Python replay.
    """

    curve_digest: str
    smoothness: ProjectiveSmoothnessWitness
    annuli: tuple[RectangularAnnulus, ...]
    fixed_axis: int
    subdivision_depth: int
    boundary_signs: tuple[tuple[int, int], ...]
    boundary_margins: tuple[tuple[Fraction, Fraction], ...]
    barrier_parents: tuple[int, ...]
    component_lower_bound: int
    harnack_upper_bound: int
    pseudoline_count: int
    complete_real_scheme: bool


def certify_curve(
    curve: HomogeneousPlaneCurve,
    smoothness: ProjectiveSmoothnessWitness,
    annuli: Sequence[RectangularAnnulus],
    *,
    fixed_axis: int = 2,
    subdivision_depth: int = 8,
) -> ProjectiveCurveCertificate:
    """Certify distinct ovals; conclude the full scheme only at maximality.

    Boundary components have opposite strict signs, so each disjoint annulus
    contains an essential oval. The complex-smooth homogeneous curve is
    irreducible, hence Harnack applies. In odd degree its unique pseudoline is
    distinct from the bounded annular ovals. All geometry is checked exactly.
    """
    if not smoothness.verifies(curve):
        raise ValueError("complex projective smoothness identities failed or have a different source")
    if type(subdivision_depth) is not int or not 0 <= subdivision_depth <= 16:
        raise ValueError("the subdivision depth must be an integer from zero to sixteen")
    polynomial = affine_chart(curve.polynomial, fixed_axis)
    barriers = tuple(annuli)
    parents = _nesting(barriers)
    signs: list[tuple[int, int]] = []
    margins: list[tuple[Fraction, Fraction]] = []
    for annulus in barriers:
        inner_sign, inner_margin = _boundary_sign(polynomial, annulus.inner, subdivision_depth)
        outer_sign, outer_margin = _boundary_sign(polynomial, annulus.outer, subdivision_depth)
        if inner_sign == outer_sign:
            raise ValueError("annulus boundary signs must be opposite")
        signs.append((inner_sign, outer_sign))
        margins.append((inner_margin, outer_margin))
    pseudoline = curve.degree % 2
    lower = len(barriers) + pseudoline
    upper = (curve.degree - 1) * (curve.degree - 2) // 2 + 1
    if lower > upper:
        raise ValueError("the supplied barriers contradict Harnack's bound")
    return ProjectiveCurveCertificate(curve.digest, smoothness, barriers, fixed_axis,
                                      subdivision_depth, tuple(signs), tuple(margins), parents,
                                      lower, upper, pseudoline, lower == upper)


def replay_curve_certificate(
    curve: HomogeneousPlaneCurve, certificate: ProjectiveCurveCertificate,
) -> bool:
    """Recompute identities, geometry, signs, and every reported conclusion."""
    if curve.digest != certificate.curve_digest:
        return False
    try:
        expected = certify_curve(curve, certificate.smoothness, certificate.annuli,
                                 fixed_axis=certificate.fixed_axis,
                                 subdivision_depth=certificate.subdivision_depth)
    except (ValueError, TypeError, ArithmeticError, AlgebraBudgetExceeded):
        return False
    return expected == certificate


__all__ = [
    "ChartBezoutIdentity",
    "HomogeneousPlaneCurve",
    "ProjectiveCurveCertificate",
    "ProjectiveSmoothnessWitness",
    "RationalRectangle",
    "RectangularAnnulus",
    "affine_chart",
    "certify_curve",
    "find_smoothness_witness",
    "replay_curve_certificate",
    "segment_bernstein_coefficients",
]
