# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Exact rational certificates for smooth projective plane curves.

Three polynomial Bezout identities exclude complex projective singularities.
Opposite strict signs on rational polygonal-annulus boundaries force distinct ovals.
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
from math import comb, lcm
from typing import Any

from omnibias.core.proof.certificate import Cert, make_certificate, verify_certificate_digest
from omnibias.core.proof.lean_check import LeanCheckResult, check_certificate
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


RationalPoint = tuple[Fraction, Fraction]


def _orientation_q(a: RationalPoint, b: RationalPoint, c: RationalPoint) -> int:
    value = (
        (b[0] - a[0]) * (c[1] - a[1])
        - (b[1] - a[1]) * (c[0] - a[0])
    )
    return (value > 0) - (value < 0)


def _point_on_segment_q(a: RationalPoint, b: RationalPoint, p: RationalPoint) -> bool:
    return (
        _orientation_q(a, b, p) == 0
        and min(a[0], b[0]) <= p[0] <= max(a[0], b[0])
        and min(a[1], b[1]) <= p[1] <= max(a[1], b[1])
    )


def _segments_intersect_q(
    a: RationalPoint,
    b: RationalPoint,
    c: RationalPoint,
    d: RationalPoint,
) -> bool:
    o1, o2 = _orientation_q(a, b, c), _orientation_q(a, b, d)
    o3, o4 = _orientation_q(c, d, a), _orientation_q(c, d, b)
    if o1 * o2 < 0 and o3 * o4 < 0:
        return True
    return (
        (o1 == 0 and _point_on_segment_q(a, b, c))
        or (o2 == 0 and _point_on_segment_q(a, b, d))
        or (o3 == 0 and _point_on_segment_q(c, d, a))
        or (o4 == 0 and _point_on_segment_q(c, d, b))
    )


@dataclass(frozen=True)
class RationalPolygon:
    """A simple closed polygon with exact rational vertices."""

    vertices: tuple[RationalPoint, ...]

    def __post_init__(self) -> None:
        vertices = tuple(
            (rational(vertex[0]), rational(vertex[1]))
            for vertex in self.vertices
        )
        if len(vertices) < 3 or len(set(vertices)) != len(vertices):
            raise ValueError("a polygon needs at least three distinct vertices")
        area_twice = sum(
            vertices[i][0] * vertices[(i + 1) % len(vertices)][1]
            - vertices[i][1] * vertices[(i + 1) % len(vertices)][0]
            for i in range(len(vertices))
        )
        if area_twice == 0:
            raise ValueError("polygon area must be nonzero")
        for i in range(len(vertices)):
            a, b = vertices[i], vertices[(i + 1) % len(vertices)]
            if a == b:
                raise ValueError("polygon edges must be nondegenerate")
            for j in range(i + 1, len(vertices)):
                if j in (i, (i + 1) % len(vertices)) or i == (j + 1) % len(vertices):
                    continue
                c, d = vertices[j], vertices[(j + 1) % len(vertices)]
                if _segments_intersect_q(a, b, c, d):
                    raise ValueError("polygon boundary must be simple")
        if area_twice < 0:
            vertices = (vertices[0], *reversed(vertices[1:]))
        object.__setattr__(self, "vertices", vertices)

    @property
    def edges(self) -> tuple[tuple[RationalPoint, RationalPoint], ...]:
        return tuple(
            (self.vertices[i], self.vertices[(i + 1) % len(self.vertices)])
            for i in range(len(self.vertices))
        )

    def contains_strictly(self, point: Sequence[Rational]) -> bool:
        """Exact odd-crossing containment, excluding the boundary."""
        p = (rational(point[0]), rational(point[1]))
        if any(_point_on_segment_q(a, b, p) for a, b in self.edges):
            return False
        inside = False
        for a, b in self.edges:
            if (a[1] > p[1]) == (b[1] > p[1]):
                continue
            crossing_x = a[0] + (p[1] - a[1]) * (b[0] - a[0]) / (b[1] - a[1])
            if crossing_x > p[0]:
                inside = not inside
        return inside

    def strictly_contains(self, other: RationalPolygon) -> bool:
        return (
            not _polygon_boundaries_intersect(self, other)
            and all(self.contains_strictly(vertex) for vertex in other.vertices)
        )

    def disjoint(self, other: RationalPolygon) -> bool:
        return (
            not _polygon_boundaries_intersect(self, other)
            and not self.contains_strictly(other.vertices[0])
            and not other.contains_strictly(self.vertices[0])
        )


def _polygon_boundaries_intersect(a: RationalPolygon, b: RationalPolygon) -> bool:
    return any(
        _segments_intersect_q(a0, a1, b0, b1)
        for a0, a1 in a.edges
        for b0, b1 in b.edges
    )


@dataclass(frozen=True)
class PolygonalAnnulus:
    """The closed region between two strictly nested rational polygons."""

    inner: RationalPolygon
    outer: RationalPolygon

    def __post_init__(self) -> None:
        if not self.outer.strictly_contains(self.inner):
            raise ValueError("the outer polygon must strictly contain the inner polygon")


BarrierAnnulus = RectangularAnnulus | PolygonalAnnulus


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


def _edge_positive_coefficients(
    polynomial: SparsePolynomial,
    start: RationalPoint,
    end: RationalPoint,
    sign: int,
    depth: int,
) -> tuple[Fraction, ...]:
    signed = tuple(
        sign * coefficient
        for coefficient in segment_bernstein_coefficients(polynomial, start, end)
    )
    if min(signed) > 0:
        return signed
    if depth == 0:
        raise ValueError("whole-edge strict sign is not certified at this subdivision depth")
    midpoint = (Fraction(start[0] + end[0], 2), Fraction(start[1] + end[1], 2))
    return (
        *_edge_positive_coefficients(polynomial, start, midpoint, sign, depth - 1),
        *_edge_positive_coefficients(polynomial, midpoint, end, sign, depth - 1),
    )


def _boundary_vertices(
    boundary: RationalRectangle | RationalPolygon,
) -> tuple[RationalPoint, ...]:
    return tuple(
        (rational(vertex[0]), rational(vertex[1]))
        for vertex in boundary.vertices
    )


def _boundary_sign(
    polynomial: SparsePolynomial,
    boundary: RationalRectangle | RationalPolygon,
    depth: int,
) -> tuple[int, Fraction]:
    vertices = _boundary_vertices(boundary)
    value = polynomial.evaluate(vertices[0])
    if value == 0:
        raise ValueError("a boundary vertex lies on the curve")
    sign = 1 if value > 0 else -1
    margin = min(_edge_margin(
        polynomial,
        vertices[i],
        vertices[(i + 1) % len(vertices)],
        sign,
        depth,
    )
                 for i in range(len(vertices)))
    return sign, margin


def _boundary_positive_coefficients(
    polynomial: SparsePolynomial,
    boundary: RationalRectangle | RationalPolygon,
    sign: int,
    depth: int,
) -> tuple[Fraction, ...]:
    vertices = _boundary_vertices(boundary)
    return tuple(
        coefficient
        for i in range(len(vertices))
        for coefficient in _edge_positive_coefficients(
            polynomial,
            vertices[i],
            vertices[(i + 1) % len(vertices)],
            sign,
            depth,
        )
    )


def _as_polygon(shape: RationalRectangle | RationalPolygon) -> RationalPolygon:
    return shape if isinstance(shape, RationalPolygon) else RationalPolygon(shape.vertices)


def _shape_contains(
    outer: RationalRectangle | RationalPolygon,
    inner: RationalRectangle | RationalPolygon,
) -> bool:
    return _as_polygon(outer).strictly_contains(_as_polygon(inner))


def _shape_disjoint(
    a: RationalRectangle | RationalPolygon,
    b: RationalRectangle | RationalPolygon,
) -> bool:
    return _as_polygon(a).disjoint(_as_polygon(b))


def _nesting(annuli: tuple[BarrierAnnulus, ...]) -> tuple[int, ...]:
    for i, a in enumerate(annuli):
        for b in annuli[i + 1:]:
            if not (
                _shape_disjoint(a.outer, b.outer)
                or _shape_contains(a.inner, b.outer)
                or _shape_contains(b.inner, a.outer)
            ):
                raise ValueError("annuli must have separated or strictly nested boundaries")
    parents = []
    for i, annulus in enumerate(annuli):
        containers = [j for j, candidate in enumerate(annuli)
                      if j != i and _shape_contains(candidate.inner, annulus.outer)]
        immediate = [j for j in containers if not any(
            k != j and _shape_contains(annuli[j].inner, annuli[k].outer)
            for k in containers
        )]
        parents.append(immediate[0] if immediate else -1)
    return tuple(parents)


@dataclass(frozen=True)
class ProjectiveCurveCertificate:
    """Replayable inputs and derived conclusions, bound to actual coefficients.

    ``barrier_parents`` indexes the input annuli, with -1 denoting the exterior.
    It is the entire oval nesting forest only if ``complete_real_scheme`` is
    true. In the other case additional curve components have not been excluded.
    ``formal_seal`` carries the finite Bezout/sign obligation; its Lean result
    is reported separately and does not formalize the topological implication.
    """

    curve_digest: str
    smoothness: ProjectiveSmoothnessWitness
    annuli: tuple[BarrierAnnulus, ...]
    fixed_axis: int
    subdivision_depth: int
    boundary_signs: tuple[tuple[int, int], ...]
    boundary_margins: tuple[tuple[Fraction, Fraction], ...]
    barrier_parents: tuple[int, ...]
    component_lower_bound: int
    harnack_upper_bound: int
    pseudoline_count: int
    complete_real_scheme: bool
    formal_seal: Cert


@dataclass(frozen=True)
class ProjectiveCurveFormalVerification:
    """Lean verdict for the finite identities/signs, not the topology theorem."""

    finite_obligation: LeanCheckResult
    theorem_prover_verified: bool


def _bezout_coefficient_equations(
    curve: HomogeneousPlaneCurve,
    smoothness: ProjectiveSmoothnessWitness,
) -> list[dict[str, Any]]:
    equations: list[dict[str, Any]] = []
    for chart in smoothness.charts:
        contributions: dict[
            tuple[int, int],
            list[tuple[Fraction, Fraction]],
        ] = {}
        for axis, multiplier in enumerate(chart.multipliers):
            derivative = affine_chart(curve.polynomial.derivative(axis), chart.fixed_axis)
            for left_index, left in multiplier.terms:
                for right_index, right in derivative.terms:
                    index = (
                        left_index[0] + right_index[0],
                        left_index[1] + right_index[1],
                    )
                    contributions.setdefault(index, []).append((left, right))
        for index in sorted(contributions):
            products = contributions[index]
            denominator = 1
            for left, right in products:
                denominator = lcm(
                    denominator,
                    left.denominator * right.denominator,
                )
            terms = [
                [
                    left.numerator * right.numerator,
                    denominator // (left.denominator * right.denominator),
                ]
                for left, right in products
            ]
            equations.append({
                "chart": chart.fixed_axis,
                "monomial": list(index),
                "terms": terms,
                "rhs": denominator if index == (0, 0) else 0,
            })
    return equations


def _curve_formal_seal(
    curve: HomogeneousPlaneCurve,
    smoothness: ProjectiveSmoothnessWitness,
    polynomial: SparsePolynomial,
    barriers: tuple[BarrierAnnulus, ...],
    signs: tuple[tuple[int, int], ...],
    subdivision_depth: int,
) -> Cert:
    positive = [
        coefficient
        for annulus, (inner_sign, outer_sign) in zip(barriers, signs, strict=True)
        for boundary, sign in (
            (annulus.inner, inner_sign),
            (annulus.outer, outer_sign),
        )
        for coefficient in _boundary_positive_coefficients(
            polynomial,
            boundary,
            sign,
            subdivision_depth,
        )
    ]
    payload = {
        "type": "polynomial_identity_q",
        "source_digest": curve.digest,
        "equations": _bezout_coefficient_equations(curve, smoothness),
        "positive": [[value.numerator, value.denominator] for value in positive],
        "formal_scope": (
            "finite Bezout coefficient identities and signed Bernstein margins; "
            "not separation, Harnack, isotopy, or algebraic realization"
        ),
    }
    return make_certificate(
        claim="finite exact-Q smoothness identities and whole-boundary signs",
        payload=payload,
        honesty={
            "finite_curve_obligations_exact_q": True,
            "topological_implication_formally_verified": False,
            "full_hilbert16_solved": False,
        },
        meta={"transcend_backend": "not_used"},
    )


def certify_curve(
    curve: HomogeneousPlaneCurve,
    smoothness: ProjectiveSmoothnessWitness,
    annuli: Sequence[BarrierAnnulus],
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
    formal_seal = _curve_formal_seal(
        curve,
        smoothness,
        polynomial,
        barriers,
        tuple(signs),
        subdivision_depth,
    )
    return ProjectiveCurveCertificate(
        curve.digest,
        smoothness,
        barriers,
        fixed_axis,
        subdivision_depth,
        tuple(signs),
        tuple(margins),
        parents,
        lower,
        upper,
        pseudoline,
        lower == upper,
        formal_seal,
    )


def replay_curve_certificate(
    curve: HomogeneousPlaneCurve, certificate: ProjectiveCurveCertificate,
) -> bool:
    """Recompute identities, geometry, signs, and every reported conclusion."""
    if (
        curve.digest != certificate.curve_digest
        or not verify_certificate_digest(certificate.formal_seal)
    ):
        return False
    try:
        expected = certify_curve(curve, certificate.smoothness, certificate.annuli,
                                 fixed_axis=certificate.fixed_axis,
                                 subdivision_depth=certificate.subdivision_depth)
    except (ValueError, TypeError, ArithmeticError, AlgebraBudgetExceeded):
        return False
    return expected == certificate


def verify_curve_certificate_formally(
    certificate: ProjectiveCurveCertificate,
) -> ProjectiveCurveFormalVerification:
    """Run the Mathlib-free kernel on the finite curve obligation."""
    result = check_certificate(certificate.formal_seal)
    return ProjectiveCurveFormalVerification(result, result.verified)


__all__ = [
    "ChartBezoutIdentity",
    "HomogeneousPlaneCurve",
    "PolygonalAnnulus",
    "ProjectiveCurveCertificate",
    "ProjectiveCurveFormalVerification",
    "ProjectiveSmoothnessWitness",
    "RationalPoint",
    "RationalPolygon",
    "RationalRectangle",
    "RectangularAnnulus",
    "affine_chart",
    "certify_curve",
    "find_smoothness_witness",
    "replay_curve_certificate",
    "segment_bernstein_coefficients",
    "verify_curve_certificate_formally",
]
