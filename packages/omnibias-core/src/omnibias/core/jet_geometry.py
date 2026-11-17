# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Exact global contact identities for rational polynomial jet sections.

The coordinates are RAW partial derivatives, not factorial-normalized Taylor
coefficients. A complete section through order k is realizable precisely when
``partial_i p_alpha == p_(alpha+e_i)`` for every ``|alpha| < k``. Polynomial
coefficient equality checks these identities everywhere, not at sample points.
This is a finite exact algebra checker, not a formal proof-kernel certificate.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from math import comb

from omnibias.core.multi_index import multi_indices
from omnibias.core.realization.polynomial import AlgebraBudget, SparsePolynomial, rational

MultiIndex = tuple[int, ...]
Rational = int | Fraction
PolynomialValue = SparsePolynomial | Rational


class JetGeometryBudgetExceeded(RuntimeError):
    """A check was refused for resources; no realizability verdict follows."""


@dataclass(frozen=True)
class PolynomialJetBudget:
    """Bounds for one construction/check, including cumulative polynomial work."""

    max_dimension: int = 8
    max_order: int = 16
    max_coordinates: int = 4096
    max_terms: int = 10_000
    max_total_terms: int = 100_000
    max_degree: int = 64
    max_bits: int = 8192
    max_work: int = 2_000_000

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class LinearJetEquation:
    """Exact identity ``sum(coeff_alpha * p_alpha) == rhs`` on the full domain.

    Coefficients may be rational polynomials in the same independent variables.
    This represents scalar linear differential equations, including variable
    polynomial coefficients; it is not a nonlinear PDE parser.
    """

    terms: Mapping[MultiIndex, PolynomialValue]
    rhs: PolynomialValue = 0
    label: str = "equation"


@dataclass(frozen=True)
class PolynomialJetBoundary:
    """Identity ``p_derivative == rhs`` on the coordinate plane x[axis]=value.

    The identity holds as a polynomial in every remaining coordinate. It does
    not certify arbitrary curved boundaries or floating sampled boundary data.
    """

    derivative: MultiIndex
    axis: int
    value: Rational
    rhs: PolynomialValue = 0
    label: str = "boundary"


@dataclass(frozen=True)
class PolynomialIdentityCheck:
    kind: str
    label: str
    residual: SparsePolynomial

    @property
    def passed(self) -> bool:
        return not self.residual.terms


@dataclass(frozen=True, init=False)
class PolynomialJetCertificate:
    """Finite global polynomial identities and explicit failure witnesses.

    For objects returned by :func:`check_polynomial_jet`, when ``valid`` is
    true the section is the k-jet of its zero coordinate,
    and all supplied PDE/boundary identities hold globally in the stated
    polynomial representation. No existence claim outside that representation,
    approximation error bound, numerical tolerance, or formal proof is added.
    """

    dimension: int
    order: int
    section: tuple[tuple[MultiIndex, SparsePolynomial], ...]
    checks: tuple[PolynomialIdentityCheck, ...]

    def __init__(self) -> None:
        raise TypeError("use check_polynomial_jet")

    @property
    def field(self) -> SparsePolynomial:
        return self.section[0][1]

    @property
    def holonomic(self) -> bool:
        return all(check.passed for check in self.checks if check.kind == "contact")

    @property
    def valid(self) -> bool:
        return all(check.passed for check in self.checks)

    @property
    def failures(self) -> tuple[PolynomialIdentityCheck, ...]:
        return tuple(check for check in self.checks if not check.passed)

    @property
    def fingerprint(self) -> str:
        """Content integrity of the section/checks; not an external proof flag."""
        payload = {
            "dimension": self.dimension, "order": self.order, "raw_derivatives": True,
            "section": [[list(index), polynomial.to_payload()] for index, polynomial in self.section],
            "checks": [[check.kind, check.label, check.residual.to_payload()] for check in self.checks],
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class _Meter:
    def __init__(self, budget: PolynomialJetBudget | None) -> None:
        self.limits = PolynomialJetBudget() if budget is None else budget
        if not isinstance(self.limits, PolynomialJetBudget):
            raise TypeError("budget must be a PolynomialJetBudget")
        self.work = 0
        self.total_terms = 0
        self.algebra = AlgebraBudget(max_terms=self.limits.max_terms, max_degree=self.limits.max_degree,
                                     max_bits=self.limits.max_bits, max_products=self.limits.max_work)

    def charge(self, work: int) -> None:
        self.work += work
        if self.work > self.limits.max_work:
            raise JetGeometryBudgetExceeded("polynomial jet cumulative work exceeds max_work")

    def keep(self, polynomial: SparsePolynomial) -> SparsePolynomial:
        self.total_terms += len(polynomial.terms)
        if self.total_terms > self.limits.max_total_terms:
            raise JetGeometryBudgetExceeded("polynomial jet intermediates exceed max_total_terms")
        return polynomial

    def polynomial(self, value: PolynomialValue, dimension: int) -> SparsePolynomial:
        if isinstance(value, SparsePolynomial):
            if value.nvars != dimension:
                raise ValueError("polynomial dimensions differ")
            if len(value.terms) > self.limits.max_terms:
                raise JetGeometryBudgetExceeded("polynomial jet input exceeds max_terms")
            self.charge(len(value.terms) + 1)
            # Rebind to this check's limits, even if the source was constructed
            # with an arbitrarily permissive algebra budget.
            return self.keep(SparsePolynomial(dimension, dict(value.terms), budget=self.algebra))
        return self.keep(SparsePolynomial.constant(dimension, rational(value), budget=self.algebra))

    def derivative(self, polynomial: SparsePolynomial, axis: int) -> SparsePolynomial:
        self.charge(len(polynomial.terms) + 1)
        return self.keep(polynomial.derivative(axis))

    def add(self, left: SparsePolynomial, right: SparsePolynomial) -> SparsePolynomial:
        self.charge(len(left.terms) + len(right.terms) + 1)
        return self.keep(left + right)

    def subtract(self, left: SparsePolynomial, right: SparsePolynomial) -> SparsePolynomial:
        self.charge(len(left.terms) + len(right.terms) + 1)
        return self.keep(left - right)

    def multiply(self, left: SparsePolynomial, right: SparsePolynomial) -> SparsePolynomial:
        self.charge(len(left.terms) * len(right.terms) + 1)
        return self.keep(left * right)

    def exact(self, value: Rational) -> Fraction:
        result = rational(value)
        self.charge(1)
        if max(result.numerator.bit_length(), result.denominator.bit_length()) > self.limits.max_bits:
            raise JetGeometryBudgetExceeded("polynomial jet rational exceeds max_bits")
        return result


def _index(index: MultiIndex, dimension: int) -> None:
    if (not isinstance(index, tuple) or len(index) != dimension
            or any(type(value) is not int or value < 0 for value in index)):
        raise ValueError("jet indices must be nonnegative integer tuples of the declared dimension")


def _layout(dimension: int, order: int, meter: _Meter) -> tuple[MultiIndex, ...]:
    if type(dimension) is not int or dimension < 1 or type(order) is not int or order < 0:
        raise ValueError("polynomial jets need positive dimension and nonnegative integer order")
    if dimension > meter.limits.max_dimension or order > meter.limits.max_order:
        raise JetGeometryBudgetExceeded("polynomial jet dimension/order exceeds its budget")
    count = comb(dimension + order, dimension)
    if count > meter.limits.max_coordinates:
        raise JetGeometryBudgetExceeded("polynomial jet layout exceeds max_coordinates")
    meter.charge(count * dimension)
    try:
        return tuple(multi_indices(dimension, order))
    except ValueError as exc:
        # The shared combinatorics can impose a stricter process-wide limit.
        raise JetGeometryBudgetExceeded(str(exc)) from exc


def polynomial_jet(
    field: SparsePolynomial, order: int, *, budget: PolynomialJetBudget | None = None,
) -> dict[MultiIndex, SparsePolynomial]:
    """Differentiate one polynomial into a complete raw-derivative section."""
    if not isinstance(field, SparsePolynomial):
        raise TypeError("field must be a SparsePolynomial")
    meter = _Meter(budget)
    indices = _layout(field.nvars, order, meter)
    result = {indices[0]: meter.polynomial(field, field.nvars)}
    for index in indices[1:]:
        axis = next(i for i, power in enumerate(index) if power)
        parent = tuple(power - int(i == axis) for i, power in enumerate(index))
        result[index] = meter.derivative(result[parent], axis)
    return result


def _restrict(polynomial: SparsePolynomial, axis: int, value: Fraction, meter: _Meter) -> SparsePolynomial:
    terms: dict[MultiIndex, Fraction] = {}
    for index, coefficient in polynomial.terms:
        # Checked rational powers avoid allocating an unbounded value before a
        # coefficient check. Degree is already bounded by the algebra budget.
        power = Fraction(1)
        for _ in range(index[axis]):
            power = meter.exact(power * value)
        reduced = tuple(0 if i == axis else exponent for i, exponent in enumerate(index))
        product = meter.exact(coefficient * power)
        terms[reduced] = meter.exact(terms.get(reduced, Fraction(0)) + product)
    return meter.keep(SparsePolynomial(polynomial.nvars, terms, budget=meter.algebra))


def check_polynomial_jet(
    section: Mapping[MultiIndex, SparsePolynomial], *,
    equations: Sequence[LinearJetEquation] = (), boundaries: Sequence[PolynomialJetBoundary] = (),
    budget: PolynomialJetBudget | None = None,
) -> PolynomialJetCertificate:
    """Check complete polynomial contact, PDE and coordinate-boundary identities.

    Invalid or incomplete representations raise. A nonzero identity residual
    yields ``valid=False`` with that exact polynomial witness. Resource refusal
    raises ``JetGeometryBudgetExceeded`` or the reused ``AlgebraBudgetExceeded``;
    it never manufactures either a pass or a mathematical exclusion.
    """
    if not isinstance(section, Mapping) or not section:
        raise ValueError("section must be a nonempty mapping")
    meter = _Meter(budget)
    if len(section) > meter.limits.max_coordinates:
        raise JetGeometryBudgetExceeded("section exceeds max_coordinates")
    first = next(iter(section.values()))
    if not isinstance(first, SparsePolynomial):
        raise TypeError("section coordinates must be SparsePolynomial values")
    dimension = first.nvars
    if dimension > meter.limits.max_dimension:
        raise JetGeometryBudgetExceeded("section exceeds max_dimension")
    meter.charge(len(section) * max(1, dimension))
    for index in section:
        _index(index, dimension)
    order = max(map(sum, section))
    indices = _layout(dimension, order, meter)
    if set(section) != set(indices):
        raise ValueError("section must contain every multi-index through its maximum order")
    coordinates: dict[MultiIndex, SparsePolynomial] = {}
    for index in indices:
        value = section[index]
        if not isinstance(value, SparsePolynomial):
            raise TypeError("section coordinates must be SparsePolynomial values")
        coordinates[index] = meter.polynomial(value, dimension)
    if not isinstance(equations, Sequence) or not isinstance(boundaries, Sequence):
        raise TypeError("equations and boundaries must be finite sequences")
    meter.charge(len(equations) + len(boundaries))
    checks: list[PolynomialIdentityCheck] = []
    for index in indices:
        if sum(index) == order:
            continue
        for axis in range(dimension):
            successor = tuple(value + int(i == axis) for i, value in enumerate(index))
            residual = meter.subtract(meter.derivative(coordinates[index], axis), coordinates[successor])
            checks.append(PolynomialIdentityCheck("contact", f"d{axis} p{index} = p{successor}", residual))
    for equation in equations:
        if not isinstance(equation, LinearJetEquation) or not isinstance(equation.terms, Mapping):
            raise TypeError("equations must be LinearJetEquation objects with term mappings")
        if not isinstance(equation.label, str) or not equation.label:
            raise ValueError("equation labels must be nonempty strings")
        total = meter.polynomial(0, dimension)
        for index, coefficient in equation.terms.items():
            _index(index, dimension)
            if index not in coordinates:
                raise ValueError("equation requests a derivative outside the declared jet")
            total = meter.add(total, meter.multiply(meter.polynomial(coefficient, dimension), coordinates[index]))
        residual = meter.subtract(total, meter.polynomial(equation.rhs, dimension))
        checks.append(PolynomialIdentityCheck("equation", equation.label, residual))
    for boundary in boundaries:
        if not isinstance(boundary, PolynomialJetBoundary):
            raise TypeError("boundaries must be PolynomialJetBoundary objects")
        _index(boundary.derivative, dimension)
        if boundary.derivative not in coordinates:
            raise ValueError("boundary requests a derivative outside the declared jet")
        if type(boundary.axis) is not int or not 0 <= boundary.axis < dimension:
            raise ValueError("boundary axis is outside the independent variables")
        if not isinstance(boundary.label, str) or not boundary.label:
            raise ValueError("boundary labels must be nonempty strings")
        boundary_value = meter.exact(boundary.value)
        rhs = meter.polynomial(boundary.rhs, dimension)
        residual = meter.subtract(_restrict(coordinates[boundary.derivative], boundary.axis, boundary_value, meter),
                                  _restrict(rhs, boundary.axis, boundary_value, meter))
        checks.append(PolynomialIdentityCheck("boundary", boundary.label, residual))
    certificate = object.__new__(PolynomialJetCertificate)
    for attribute_name, attribute_value in (("dimension", dimension), ("order", order),
                                            ("section", tuple(coordinates.items())), ("checks", tuple(checks))):
        object.__setattr__(certificate, attribute_name, attribute_value)
    return certificate


__all__ = ["JetGeometryBudgetExceeded", "LinearJetEquation", "PolynomialIdentityCheck",
           "PolynomialJetBoundary", "PolynomialJetBudget", "PolynomialJetCertificate",
           "check_polynomial_jet", "polynomial_jet"]
