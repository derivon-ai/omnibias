# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Source-bound zero counts for exact polynomial displacement families.

The first variable is height; all remaining variables are parameters.  A
certificate counts distinct *isolated* real zeros on a closed height interval.
An identity fiber has zero isolated zeros.  Polynomial identity fibers are
decided exactly, never by a small numerical coefficient or a truncated jet.

These are displacement certificates.  A physical cycle interpretation requires
an actual return-map identity and capture theorem; neither is inferred here.
The checker is rational Python arithmetic, not a Lean verification claim.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Literal, TypeAlias

from omnibias.core.proof.certificate import Cert, make_certificate, verify_certificate_digest
from omnibias.core.proof.realization_replay import source_digest
from omnibias.core.realization.algebraic import divide, evaluate, root_count, trim
from omnibias.core.realization.polynomial import Rational, SparsePolynomial, rational
from omnibias.core.verified.interval import Interval
from omnibias.dynamics.return_maps import StoppedEventResult, verify_stopped_event

RationalInterval: TypeAlias = tuple[Fraction, Fraction]
RationalBox: TypeAlias = tuple[RationalInterval, ...]
ZeroCountMethod: TypeAlias = Literal["identity", "sturm", "rolle", "polynomial_degree"]


def rational_box(box: Sequence[Sequence[Rational]], dimension: int) -> RationalBox:
    """Validate exact closed bounds; floats and reversed bounds are rejected."""
    if len(box) != dimension or any(len(v) != 2 for v in box):
        raise ValueError("one pair of rational endpoints per variable required")
    result = tuple((rational(v[0]), rational(v[1])) for v in box)
    if any(a > b for a, b in result):
        raise ValueError("box endpoints are reversed")
    return result


def _multiply(a: RationalInterval, b: RationalInterval) -> RationalInterval:
    products = tuple(x * y for x in a for y in b)
    return min(products), max(products)


def _power(a: RationalInterval, n: int) -> RationalInterval:
    if n == 0:
        return Fraction(1), Fraction(1)
    lo, hi = a
    if n % 2:
        return lo**n, hi**n
    upper = max(lo**n, hi**n)
    return (Fraction(0) if lo <= 0 <= hi else min(lo**n, hi**n)), upper


def polynomial_range(
    polynomial: SparsePolynomial, box: Sequence[Sequence[Rational]]
) -> RationalInterval:
    """Exact natural interval extension, with dependency overestimation."""
    domain = rational_box(box, polynomial.nvars)
    lower = upper = Fraction(0)
    for powers, coefficient in polynomial.terms:
        term = (coefficient, coefficient)
        for bounds, power in zip(domain, powers, strict=True):
            term = _multiply(term, _power(bounds, power))
        lower += term[0]
        upper += term[1]
    return lower, upper


def identity_coefficients(polynomial: SparsePolynomial) -> tuple[SparsePolynomial, ...]:
    """Exact center equations: every returned parameter polynomial must vanish.

    This characterizes the identity of the supplied polynomial in height. It
    does not identify a center of a field without a displacement identity.
    """
    if polynomial.nvars < 1:
        raise ValueError("a height variable is required")
    degree = max((powers[0] for powers, _ in polynomial.terms), default=0)
    rows: list[dict[tuple[int, ...], Fraction]] = [{} for _ in range(degree + 1)]
    for powers, coefficient in polynomial.terms:
        rows[powers[0]][powers[1:]] = coefficient
    return tuple(
        SparsePolynomial(polynomial.nvars - 1, row, budget=polynomial.budget)
        for row in rows
    )


def _closed_root_count(coefficients: tuple[Fraction, ...], lo: Fraction, hi: Fraction) -> int:
    """Remove endpoint factors exactly before invoking the shared Sturm engine."""
    p = trim(coefficients)
    endpoints = 0
    for endpoint in (lo, hi):
        if p and evaluate(p, endpoint) == 0:
            endpoints += 1
            while len(p) > 1 and evaluate(p, endpoint) == 0:
                p, remainder = divide(p, (-endpoint, Fraction(1)))
                if remainder:  # pragma: no cover - exact division invariant
                    raise ArithmeticError("endpoint factor failed exact division")
    return endpoints + (root_count(p, lo, hi) if len(p) > 1 else 0)


def _q(value: Fraction) -> list[int]:
    return [value.numerator, value.denominator]


def _box_payload(box: RationalBox) -> list[list[list[int]]]:
    return [[_q(a), _q(b)] for a, b in box]


@dataclass(frozen=True)
class CyclicityCertificate:
    """A replayable isolated-zero bound, including possible identity fibers."""

    domain: RationalBox
    upper_bound: int
    exact_count: int | None
    identity: bool | None
    method: ZeroCountMethod
    derivative_order: int | None
    derivative_range: RationalInterval | None
    source_digest: str
    seal: Cert


def certify_polynomial_cyclicity(
    polynomial: SparsePolynomial, box: Sequence[Sequence[Rational]]
) -> CyclicityCertificate:
    """Uniform isolated-zero bound for an exact polynomial family on a box.

    Fixed parameters use a distinct-root Sturm count (endpoints included).
    Variable parameters use the first uniformly nonzero height derivative and
    Rolle, or the exact height-degree bound. The latter remains valid when a
    parameter fiber is identically zero because its zeros are not isolated.
    """
    if polynomial.nvars < 1:
        raise ValueError("a height variable is required")
    domain = rational_box(box, polynomial.nvars)
    if domain[0][0] >= domain[0][1]:
        raise ValueError("height domain must be a nondegenerate connected interval")
    equations = identity_coefficients(polynomial)
    degree = len(equations) - 1
    exact: int | None = None
    identity: bool | None = None
    order: int | None = None
    enclosure: RationalInterval | None = None
    method: ZeroCountMethod
    if all(a == b for a, b in domain[1:]):
        point = tuple(a for a, _ in domain[1:])
        coefficients = trim(tuple(p.evaluate(point) for p in equations))
        identity = not coefficients
        exact = 0 if identity else _closed_root_count(coefficients, *domain[0])
        bound = exact
        method = "identity" if identity else "sturm"
    else:
        bound = degree
        method = "polynomial_degree"
        derivative = polynomial
        for k in range(degree + 1):
            current = polynomial_range(derivative, domain)
            if current[0] > 0 or current[1] < 0:
                bound, order, enclosure = k, k, current
                method, identity = "rolle", False
                break
            derivative = derivative.derivative(0)
        if not polynomial.terms:
            exact, identity, method = 0, True, "identity"
    source = {"polynomial": polynomial.to_payload(), "domain": _box_payload(domain)}
    digest = source_digest(source)
    seal = make_certificate(
        claim="Distinct isolated zeros of the specified polynomial family; no physical capture claim.",
        payload={
            "type": "polynomial_cyclicity",
            "source": source,
            "source_digest": digest,
            "upper_bound": bound,
            "exact_count": exact,
            "identity": identity,
            "method": method,
            "derivative_order": order,
            "derivative_range": None if enclosure is None else [_q(v) for v in enclosure],
        },
        meta={"transcend_backend": "not_used"},
    )
    return CyclicityCertificate(domain, bound, exact, identity, method, order, enclosure, digest, seal)


def verify_cyclicity_certificate(
    polynomial: SparsePolynomial,
    certificate: CyclicityCertificate,
    *,
    expected_domain: Sequence[Sequence[Rational]] | None = None,
) -> bool:
    """Recompute the claim from its source, not from its digest or result alone."""
    try:
        if not verify_certificate_digest(certificate.seal):
            return False
        if expected_domain is not None and rational_box(
            expected_domain, polynomial.nvars
        ) != certificate.domain:
            return False
        expected = certify_polynomial_cyclicity(polynomial, certificate.domain)
        return certificate == expected
    except (ValueError, TypeError, ArithmeticError, KeyError):
        return False


@dataclass(frozen=True, init=False)
class ConfluentExponentialPolynomial:
    """Exact finite sum of p_a(kappa)*exp(a*kappa), for real rational a.

    Polynomial factors encode confluent real exponents. Imaginary exponents,
    arbitrary quotients, asymptotic remainders and fitted prefixes are outside
    this class. Equal exponents are merged before any zero count is made.
    """

    terms: tuple[tuple[Fraction, tuple[Fraction, ...]], ...]

    def __init__(self, terms: Sequence[tuple[Rational, Sequence[Rational]]]) -> None:
        merged: dict[Fraction, list[Fraction]] = {}
        for exponent, coefficients in terms:
            a, p = rational(exponent), trim(coefficients)
            old = merged.setdefault(a, [])
            old.extend([Fraction(0)] * max(0, len(p) - len(old)))
            for j, c in enumerate(p):
                old[j] += c
        object.__setattr__(self, "terms", tuple(
            (a, p) for a, c in sorted(merged.items()) if (p := trim(c))
        ))

    @property
    def dimension(self) -> int:
        return sum(len(p) for _, p in self.terms)

    def weighted_derivative(self, exponent: Rational) -> ConfluentExponentialPolynomial:
        """Exact (d/dkappa - exponent) F; positive exponential division is valid."""
        a = rational(exponent)
        return ConfluentExponentialPolynomial([
            (b, tuple(
                (b - a) * c + ((j + 1) * p[j + 1] if j + 1 < len(p) else 0)
                for j, c in enumerate(p)
            ))
            for b, p in self.terms
        ])

    def to_payload(self) -> list[object]:
        return [[_q(a), [_q(c) for c in p]] for a, p in self.terms]


@dataclass(frozen=True)
class ExponentialCyclicityCertificate:
    """Global real isolated-zero upper bound for the supplied exact finite sum."""

    upper_bound: int
    identity: bool
    chain: tuple[ConfluentExponentialPolynomial, ...]
    seal: Cert


def certify_exponential_cyclicity(
    displacement: ConfluentExponentialPolynomial, *, max_dimension: int = 256
) -> ExponentialCyclicityCertificate:
    """Terminate derivation-division in the real confluent exponential class.

    At each step choose the least exponent a. The zeros of F equal those of
    exp(-a*kappa) F; Rolle reduces the count to (D-a)F plus one. Its dimension
    drops exactly one. The terminal constant times exp(a*kappa) has no zeros.
    This proves dimension-1 globally (including multiplicities), and hence for
    distinct isolated zeros. Identity sums have no isolated zeros. Membership
    of an actual return displacement is a separate, required mathematical fact.
    """
    if type(max_dimension) is not int or max_dimension < 1:
        raise ValueError("positive integer dimension budget required")
    if displacement.dimension > max_dimension:
        raise ValueError("exponential displacement dimension budget exceeded")
    chain = [displacement]
    while chain[-1].dimension > 1:
        current = chain[-1]
        derived = current.weighted_derivative(current.terms[0][0])
        if derived.dimension != current.dimension - 1:
            raise ArithmeticError("derivation-division dimension invariant failed")
        chain.append(derived)
    identity = not displacement.terms
    bound = max(0, displacement.dimension - 1)
    seal = make_certificate(
        claim="Real zeros of an exact confluent exponential polynomial; no return-map membership claim.",
        payload={
            "type": "confluent_exponential_cyclicity",
            "source": displacement.to_payload(),
            "source_digest": source_digest(displacement.to_payload()),
            "chain": [p.to_payload() for p in chain],
            "upper_bound": bound,
            "identity": identity,
        },
        meta={"transcend_backend": "not_used"},
    )
    return ExponentialCyclicityCertificate(bound, identity, tuple(chain), seal)


def verify_exponential_cyclicity(
    displacement: ConfluentExponentialPolynomial,
    certificate: ExponentialCyclicityCertificate,
) -> bool:
    try:
        return verify_certificate_digest(certificate.seal) and certificate == certify_exponential_cyclicity(
            displacement, max_dimension=max(1, displacement.dimension)
        )
    except (TypeError, ValueError, ArithmeticError, KeyError):
        return False


@dataclass(frozen=True)
class ReturnCyclicityCertificate:
    """Bound for cycles closing at the certified first eligible section hit.

    A finite-horizon failure or a flat derivative enclosure is unresolved. The
    certificate does not capture cycles using other itineraries or sections.
    """

    event: StoppedEventResult
    height_parameter: int
    section_offset: Fraction
    upper_bound: int | None
    method: str
    displacement_range: Interval
    first_derivative: Interval | None
    second_derivative: Interval | None


def certify_planar_return_cyclicity(
    event: StoppedEventResult,
    *,
    height_parameter: int = 0,
    section_offset: Rational = 0,
) -> ReturnCyclicityCertificate:
    """Actual return bound for x=offset, y=height in an autonomous planar field.

    Field coefficients must be independent of the designated initial-height
    parameter. The initial embedding and target equation are checked as exact
    polynomial identities. A polynomial guard may restrict the target to a
    transversal branch. The generated first/second event jets, not arbitrary
    derivative callbacks, feed Rolle on the connected height interval.
    """
    request = event.request
    flow, m = request.flow, request.flow.parameter_count
    if flow.state_count != 2 or type(height_parameter) is not int or not 0 <= height_parameter < m:
        raise ValueError("a planar flow and a designated initial-height parameter are required")
    if request.parameters[height_parameter].lo >= request.parameters[height_parameter].hi:
        raise ValueError("the height interval must be connected and nondegenerate")
    offset = rational(section_offset)
    x = SparsePolynomial.variable(flow.dimension, 0)
    h = SparsePolynomial.variable(m, height_parameter)
    if (request.target.polynomial.terms != (x - offset).terms
            or request.initial[0].terms != SparsePolynomial.constant(m, offset).terms
            or request.initial[1].terms != h.terms):
        raise ValueError("exact section x=offset and initial embedding (offset,height) required")
    if any(p.derivative(flow.dimension - 1).terms or p.derivative(2 + height_parameter).terms
           for p in flow.components):
        raise ValueError("field must be autonomous and independent of the initial-height parameter")
    if not verify_stopped_event(event) or event.return_box is None:
        raise ValueError("an actual replayed first-hit event certificate is required")
    value = event.return_box[1] - request.parameters[height_parameter]
    first = (None if event.return_jacobian is None
             else event.return_jacobian[1][height_parameter] - 1)
    second = (None if event.return_hessian is None
              else event.return_hessian[1][height_parameter][height_parameter])
    bound: int | None = None
    method = "unresolved_zero_count"
    for order, enclosure in enumerate((value, first, second)):
        if enclosure is not None and not enclosure.contains_zero():
            bound, method = order, ("value_exclusion" if order == 0 else "actual_return_rolle")
            break
    return ReturnCyclicityCertificate(event, height_parameter, offset, bound, method, value, first, second)


def verify_return_cyclicity(
    certificate: ReturnCyclicityCertificate, *, expected_source_fingerprint: str
) -> bool:
    """Bind to the expected physical source and replay the entire flow and count."""
    if certificate.event.request.fingerprint != expected_source_fingerprint:
        return False
    try:
        return certificate == certify_planar_return_cyclicity(
            certificate.event, height_parameter=certificate.height_parameter,
            section_offset=certificate.section_offset,
        )
    except (ValueError, TypeError, ArithmeticError, KeyError):
        return False


__all__ = [
    "ConfluentExponentialPolynomial",
    "CyclicityCertificate",
    "ExponentialCyclicityCertificate",
    "RationalBox",
    "RationalInterval",
    "ReturnCyclicityCertificate",
    "ZeroCountMethod",
    "certify_exponential_cyclicity",
    "certify_planar_return_cyclicity",
    "certify_polynomial_cyclicity",
    "identity_coefficients",
    "polynomial_range",
    "rational_box",
    "verify_cyclicity_certificate",
    "verify_exponential_cyclicity",
    "verify_return_cyclicity",
]
