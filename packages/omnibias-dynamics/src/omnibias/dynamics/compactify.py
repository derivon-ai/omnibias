# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Exact-Q Poincare compactification of planar polynomial vector fields.

The two charts at infinity use rational polynomial index remaps, not numerical
substitution.  Their desingularized normal component has an explicit factor of
the inverse radial coordinate, so the equator is invariant.  The certificate
checks those coefficient identities and isolates every *simple* real
singularity on the two affine equator charts.

This module certifies the compactified source field. It does not prove that a
declared graphic is present, that a return itinerary is complete, or that a
singular Dulac passage has finite cyclicity.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import ceil, isqrt, lcm
from typing import Literal

from omnibias.core.proof.certificate import Cert, make_certificate, verify_certificate_digest
from omnibias.core.proof.lean_check import LeanCheckResult, check_certificate
from omnibias.core.proof.realization_replay import source_digest
from omnibias.core.realization.polynomial import SparsePolynomial
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.rootfind import interval_newton
from omnibias.dynamics.cyclicity import certify_polynomial_cyclicity

ChartName = Literal["U1", "U2", "U3"]


def _poly_payload(polynomial: SparsePolynomial) -> dict[str, object]:
    return polynomial.to_payload()


@dataclass(frozen=True)
class PlanarPolynomialField:
    """Two exact rational polynomials with one declared projective degree."""

    p: SparsePolynomial
    q: SparsePolynomial
    degree: int = 2

    def __post_init__(self) -> None:
        if self.p.nvars != 2 or self.q.nvars != 2:
            raise ValueError("a planar field requires two bivariate polynomials")
        if type(self.degree) is not int or self.degree < 1:
            raise ValueError("the declared field degree must be a positive integer")
        actual = max(
            (sum(index) for polynomial in (self.p, self.q) for index, _ in polynomial.terms),
            default=0,
        )
        if actual > self.degree:
            raise ValueError("a field term exceeds the declared degree")

    @property
    def digest(self) -> str:
        return source_digest(self.to_payload())

    def to_payload(self) -> dict[str, object]:
        return {
            "degree": self.degree,
            "p": _poly_payload(self.p),
            "q": _poly_payload(self.q),
        }


@dataclass(frozen=True)
class PoincareChart:
    """One exact desingularized chart, ordered as ``(u_dot, v_dot)``."""

    name: ChartName
    tangential: SparsePolynomial
    normal: SparsePolynomial

    def __post_init__(self) -> None:
        if self.tangential.nvars != 2 or self.normal.nvars != 2:
            raise ValueError("compactification charts must be bivariate")

    @property
    def equator_polynomial(self) -> SparsePolynomial:
        """Tangential field on ``v=0`` as a univariate polynomial in ``u``."""
        terms: dict[tuple[int], Fraction] = {}
        for (u_power, v_power), coefficient in self.tangential.terms:
            if v_power == 0:
                terms[(u_power,)] = terms.get((u_power,), Fraction(0)) + coefficient
        return SparsePolynomial(1, terms, budget=self.tangential.budget)

    @property
    def equator_invariant(self) -> bool:
        return all(v_power >= 1 for (_, v_power), _ in self.normal.terms)

    def to_payload(self) -> dict[str, object]:
        return {
            "name": self.name,
            "tangential": _poly_payload(self.tangential),
            "normal": _poly_payload(self.normal),
        }


def _add(
    terms: dict[tuple[int, int], Fraction],
    index: tuple[int, int],
    coefficient: Fraction,
) -> None:
    terms[index] = terms.get(index, Fraction(0)) + coefficient


def poincare_chart(field: PlanarPolynomialField, chart: ChartName) -> PoincareChart:
    """Construct one exact affine chart of the Poincare compactification.

    ``U1`` uses ``u=y/x, v=1/x`` and ``U2`` uses ``u=x/y, v=1/y``.
    The common positive time desingularization multiplies the original vector
    field by ``v**(degree-1)``. ``U3`` is the original affine plane.
    """
    if chart == "U3":
        return PoincareChart(chart, field.p, field.q)
    if chart not in ("U1", "U2"):
        raise ValueError("chart must be U1, U2, or U3")

    tangent: dict[tuple[int, int], Fraction] = {}
    normal: dict[tuple[int, int], Fraction] = {}
    n = field.degree
    if chart == "U1":
        for (i, j), coefficient in field.q.terms:
            _add(tangent, (j, n - i - j), coefficient)
        for (i, j), coefficient in field.p.terms:
            _add(tangent, (j + 1, n - i - j), -coefficient)
            _add(normal, (j, n - i - j + 1), -coefficient)
    else:
        for (i, j), coefficient in field.p.terms:
            _add(tangent, (i, n - i - j), coefficient)
        for (i, j), coefficient in field.q.terms:
            _add(tangent, (i + 1, n - i - j), -coefficient)
            _add(normal, (i, n - i - j + 1), -coefficient)
    return PoincareChart(
        chart,
        SparsePolynomial(2, tangent, budget=field.p.budget),
        SparsePolynomial(2, normal, budget=field.p.budget),
    )


def _integer_coefficients(polynomial: SparsePolynomial) -> list[int]:
    if polynomial.nvars != 1:
        raise ValueError("a univariate polynomial is required")
    degree = max((index[0] for index, _ in polynomial.terms), default=0)
    denominators = [coefficient.denominator for _, coefficient in polynomial.terms]
    scale = lcm(*denominators) if denominators else 1
    coefficients = [0] * (degree + 1)
    for (power,), coefficient in polynomial.terms:
        coefficients[power] = coefficient.numerator * (scale // coefficient.denominator)
    while len(coefficients) > 1 and coefficients[-1] == 0:
        coefficients.pop()
    return coefficients


def _divisors(value: int) -> tuple[int, ...]:
    value = abs(value)
    if value == 0:
        return (0,)
    output: set[int] = set()
    for candidate in range(1, isqrt(value) + 1):
        if value % candidate == 0:
            output.update((candidate, value // candidate))
    return tuple(sorted(output))


def _rational_roots(polynomial: SparsePolynomial) -> tuple[Fraction, ...]:
    coefficients = _integer_coefficients(polynomial)
    if not any(coefficients):
        raise ValueError("the equator field vanishes identically")
    roots: set[Fraction] = set()
    first_nonzero = next(i for i, coefficient in enumerate(coefficients) if coefficient)
    if first_nonzero:
        roots.add(Fraction(0))
        coefficients = coefficients[first_nonzero:]
    constant, leading = coefficients[0], coefficients[-1]
    for numerator in _divisors(constant):
        for denominator in _divisors(leading):
            if denominator == 0:
                continue
            for sign in (-1, 1):
                candidate = Fraction(sign * numerator, denominator)
                if polynomial.evaluate((candidate,)) == 0:
                    roots.add(candidate)
    return tuple(sorted(roots))


def _interval_polynomial(polynomial: SparsePolynomial, value: Interval) -> Interval:
    output = Interval.point(0.0)
    for (power,), coefficient in polynomial.terms:
        output = output + Interval.from_rational(coefficient) * value**power
    return output


@dataclass(frozen=True)
class InfiniteSingularity:
    """One real zero of the compactified tangential field."""

    chart: Literal["U1", "U2"]
    enclosure: Interval
    exact_coordinate: Fraction | None
    multiplicity: int

    @property
    def rational(self) -> bool:
        return self.exact_coordinate is not None


def infinite_singular_points(
    chart: PoincareChart,
    *,
    subdivisions: int = 2048,
) -> tuple[InfiniteSingularity, ...]:
    """Isolate every distinct real equator singularity in one chart.

    Rational roots, including multiple ones, are found exactly. Remaining roots
    are admitted only after interval Newton proves uniqueness. A multiple
    irrational root is unresolved rather than silently omitted.
    """
    if chart.name not in ("U1", "U2"):
        raise ValueError("infinite singularities live on chart U1 or U2")
    if type(subdivisions) is not int or not 32 <= subdivisions <= 65_536:
        raise ValueError("subdivisions must be an integer from 32 to 65536")
    polynomial = chart.equator_polynomial
    rational_roots = _rational_roots(polynomial)
    coefficients = _integer_coefficients(polynomial)
    leading = abs(coefficients[-1])
    bound = max(
        1,
        ceil(
            1 + max(
                (Fraction(abs(coefficient), leading) for coefficient in coefficients[:-1]),
                default=Fraction(0),
            )
        ),
    )
    exact_count = certify_polynomial_cyclicity(polynomial, ((-bound, bound),)).exact_count
    if exact_count is None:
        raise ArithmeticError("failed to count equator singularities exactly")
    roots: list[InfiniteSingularity] = []
    for root in rational_roots:
        multiplicity = 0
        derivative_at_root = polynomial
        while derivative_at_root.evaluate((root,)) == 0:
            multiplicity += 1
            derivative_at_root = derivative_at_root.derivative(0)
            if not derivative_at_root.terms:
                break
        roots.append(
            InfiniteSingularity(
                chart.name,
                Interval.from_rational(root),
                root,
                multiplicity,
            )
        )
    derivative = polynomial.derivative(0)
    width = Fraction(2 * bound, subdivisions)
    for index in range(subdivisions):
        lo = Fraction(-bound) + index * width
        hi = lo + width
        if any(lo <= root <= hi for root in rational_roots):
            continue
        flo, fhi = polynomial.evaluate((lo,)), polynomial.evaluate((hi,))
        if flo == 0 or fhi == 0 or (flo < 0 < fhi) or (fhi < 0 < flo):
            result = interval_newton(
                lambda box: _interval_polynomial(polynomial, box),
                lambda box: _interval_polynomial(derivative, box),
                (
                    Interval.from_rational(lo).lo,
                    Interval.from_rational(hi).hi,
                ),
            )
            if result["status"] == "unique_root" and bool(result["unique"]):
                enclosure = Interval(*result["enclosure"])
                if not any(
                    max(enclosure.lo, root.enclosure.lo)
                    <= min(enclosure.hi, root.enclosure.hi)
                    for root in roots
                ):
                    roots.append(InfiniteSingularity(chart.name, enclosure, None, 1))
    roots.sort(key=lambda root: root.enclosure.mid)
    if len(roots) != exact_count:
        raise ArithmeticError(
            "not every equator singularity was isolated as a simple real root"
        )
    return tuple(roots)


def _coefficient_rows(
    field: PlanarPolynomialField,
    chart: PoincareChart,
) -> list[dict[str, object]]:
    contributions: dict[tuple[str, int, int], list[Fraction]] = {}

    def add(kind: str, index: tuple[int, int], value: Fraction) -> None:
        contributions.setdefault((kind, *index), []).append(value)

    n = field.degree
    if chart.name == "U1":
        for (i, j), coefficient in field.q.terms:
            add("t", (j, n - i - j), coefficient)
        for (i, j), coefficient in field.p.terms:
            add("t", (j + 1, n - i - j), -coefficient)
            add("n", (j, n - i - j + 1), -coefficient)
    elif chart.name == "U2":
        for (i, j), coefficient in field.p.terms:
            add("t", (i, n - i - j), coefficient)
        for (i, j), coefficient in field.q.terms:
            add("t", (i + 1, n - i - j), -coefficient)
            add("n", (i, n - i - j + 1), -coefficient)
    else:
        raise ValueError("coefficient rows apply only at infinity")

    output = {"t": dict(chart.tangential.terms), "n": dict(chart.normal.terms)}
    rows: list[dict[str, object]] = []
    for (kind, i, j), values in sorted(contributions.items()):
        denominator = lcm(
            *(value.denominator for value in values),
            output[kind].get((i, j), Fraction(0)).denominator,
        )
        terms = [
            [value.numerator, denominator // value.denominator] for value in values
        ]
        target = output[kind].get((i, j), Fraction(0))
        rows.append(
            {
                "terms": terms,
                "rhs": target.numerator * (denominator // target.denominator),
            }
        )
    return rows


@dataclass(frozen=True)
class PoincareCompactificationCertificate:
    """Replayable exact-source compactification and simple equator roots."""

    field: PlanarPolynomialField
    charts: tuple[PoincareChart, PoincareChart, PoincareChart]
    singularities: tuple[InfiniteSingularity, ...]
    root_subdivisions: int
    source_digest: str
    formal_seal: Cert


@dataclass(frozen=True)
class PoincareCompactificationFormalVerification:
    result: LeanCheckResult
    theorem_prover_verified: bool


def certify_poincare_compactification(
    field: PlanarPolynomialField,
    *,
    root_subdivisions: int = 2048,
) -> PoincareCompactificationCertificate:
    """Certify exact chart coefficients, equator invariance, and simple roots."""
    u1 = poincare_chart(field, "U1")
    u2 = poincare_chart(field, "U2")
    u3 = poincare_chart(field, "U3")
    if not u1.equator_invariant or not u2.equator_invariant:
        raise ArithmeticError("the compactified equator is not invariant")
    singularities = (
        *infinite_singular_points(u1, subdivisions=root_subdivisions),
        *infinite_singular_points(u2, subdivisions=root_subdivisions),
    )
    rows = [*_coefficient_rows(field, u1), *_coefficient_rows(field, u2)]
    if not rows:
        rows = [{"terms": [[0, 1]], "rhs": 0}]
    seal = make_certificate(
        claim="Exact-Q Poincare chart identities and invariant equator.",
        payload={
            "type": "polynomial_identity_q",
            "source_digest": field.digest,
            "equations": rows,
            "positive": [],
            "formal_scope": (
                "finite chart coefficient identities; not graphic presence, "
                "singular passage, itinerary capture, or cyclicity"
            ),
        },
        honesty={
            "poincare_compactification_exact_q": True,
            "equator_invariance_exact_q": True,
            "distinct_real_equator_roots_isolated": True,
            "multiple_irrational_equator_roots_supported": False,
            "graphic_finite_cyclicity_proved": False,
            "full_hilbert16_solved": False,
        },
        meta={"transcend_backend": "not_used"},
    )
    return PoincareCompactificationCertificate(
        field,
        (u1, u2, u3),
        singularities,
        root_subdivisions,
        field.digest,
        seal,
    )


def verify_poincare_compactification(
    certificate: PoincareCompactificationCertificate,
) -> bool:
    """Replay all exact chart and root claims from the source field."""
    if (
        certificate.source_digest != certificate.field.digest
        or not verify_certificate_digest(certificate.formal_seal)
    ):
        return False
    try:
        expected = certify_poincare_compactification(
            certificate.field,
            root_subdivisions=certificate.root_subdivisions,
        )
    except (ArithmeticError, TypeError, ValueError):
        return False
    return expected == certificate


def verify_poincare_compactification_formally(
    certificate: PoincareCompactificationCertificate,
) -> PoincareCompactificationFormalVerification:
    """Run the Mathlib-free kernel on the finite coefficient identities."""
    result = check_certificate(certificate.formal_seal)
    return PoincareCompactificationFormalVerification(result, result.verified)


__all__ = [
    "ChartName",
    "InfiniteSingularity",
    "PlanarPolynomialField",
    "PoincareChart",
    "PoincareCompactificationCertificate",
    "PoincareCompactificationFormalVerification",
    "certify_poincare_compactification",
    "infinite_singular_points",
    "poincare_chart",
    "verify_poincare_compactification",
    "verify_poincare_compactification_formally",
]
