# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Exact rational exponential fields and finite-library differential relations.

Identities hold on the common regular domain of every original denominator.
Simplification never fills a removed pole. Relation discovery matches exact
coefficients; it does not infer physical laws from samples or establish that a
chosen candidate library is complete.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from fractions import Fraction
from itertools import chain

from omnibias.core._rational_linear import solve_affine
from omnibias.core.exp_polynomial import ExponentialBudget, ExponentialTerm, ExpPolynomial
from omnibias.core.ridge_system import RidgeBudget

Rational = int | Fraction
Factor = tuple[ExpPolynomial, int]
_Key = tuple[ExponentialTerm, ...]


def _integer(value: int, name: str, *, signed: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if not signed and value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def _compatible(left: ExpPolynomial, right: object) -> ExpPolynomial:
    if not isinstance(right, ExpPolynomial):
        raise TypeError("rational field data must be ExpPolynomial objects")
    if left.dimension != right.dimension:
        raise ValueError("rational field dimensions must match")
    if left.budget != right.budget:
        raise ValueError("rational field polynomial budgets must match")
    return right


class _Work:
    """Cumulative conservative polynomial-work bound for one complete call."""

    def __init__(self, reference: ExpPolynomial, discovery: RidgeBudget | None = None):
        self.reference = reference
        self.budget = reference.budget
        self.discovery = discovery
        self.used = 0
        self.storage = 0
        self.matrix_entries = 0

    def charge(self, count: int) -> None:
        self.used += count
        self.budget._work(self.used)
        if self.discovery is not None:
            self.discovery._work(self.used)

    def check(self, polynomial: ExpPolynomial) -> ExpPolynomial:
        _compatible(self.reference, polynomial)
        self.charge(len(polynomial.terms) * (2 * polynomial.dimension + 1) + 1)
        if self.discovery is not None:
            # Conservatively retain every encountered snapshot through the
            # complete discovery call, including nested power/product inputs.
            self.reserve(len(polynomial.terms) * (2 * polynomial.dimension + 1) + 1)
            for frequency, _, coefficient in polynomial.terms:
                self.discovery._coefficient(coefficient)
                for entry in frequency:
                    self.discovery._coefficient(entry)
        return polynomial

    def sorting(self, count: int) -> int:
        return count * max(1, count.bit_length()) * (2 * self.reference.dimension + 1)

    def reserve(self, count: int) -> None:
        self.storage += count
        if self.discovery is not None:
            self.discovery._entries(self.storage + 2 * self.matrix_entries)

    def temporary(self, terms: int) -> None:
        if self.discovery is not None:
            self.discovery._entries(self.storage + 2 * self.matrix_entries + terms * (2 * self.reference.dimension + 1))

    def constant(self, value: Rational) -> ExpPolynomial:
        self.temporary(4)
        self.charge(8 * self.reference.dimension + 8)
        return self.check(ExpPolynomial.constant(value, self.reference.dimension, budget=self.budget))

    def add(self, left: ExpPolynomial, right: ExpPolynomial) -> ExpPolynomial:
        n, m, d = len(left.terms), len(right.terms), self.reference.dimension
        self.temporary(3 * (n + m))
        self.charge(n * (2 * d + 1) + m * (2 * d + 3) + self.sorting(n + m))
        return self.check(left + right)

    def multiply(self, left: ExpPolynomial, right: ExpPolynomial) -> ExpPolynomial:
        n, m, d = len(left.terms), len(right.terms), self.reference.dimension
        self.temporary(n + m + 3 * n * m)
        self.charge((n + m) * (2 * d + 1) + n * m * (2 * d + 2) + self.sorting(n * m))
        return self.check(left * right)

    def scale(self, polynomial: ExpPolynomial, value: Rational) -> ExpPolynomial:
        n, d = len(polynomial.terms), self.reference.dimension
        self.temporary(3 * n)
        self.charge(n * (2 * d + 2) + 1 + self.sorting(n))
        return self.check(polynomial.scale(value))

    def derivative(self, polynomial: ExpPolynomial, axis: int) -> ExpPolynomial:
        n, d = len(polynomial.terms), self.reference.dimension
        self.temporary(6 * n)
        self.charge(n * (3 * d + 5) + 1 + self.sorting(2 * n))
        return self.check(polynomial.differentiate(axis))

    def power(self, polynomial: ExpPolynomial, exponent: int) -> ExpPolynomial:
        self.charge(exponent.bit_length() + 1)
        result = self.constant(1)
        base = polynomial
        while exponent:
            if exponent & 1:
                result = self.multiply(result, base)
            exponent >>= 1
            if exponent:
                base = self.multiply(base, base)
        return result

    def product(self, factors: Iterable[Factor]) -> ExpPolynomial:
        result = self.constant(1)
        for polynomial, exponent in factors:
            result = self.multiply(result, self.power(polynomial, exponent))
        return result


def _canonical_factors(factors: Iterable[Factor], work: _Work, *, maximum: bool = False) -> tuple[Factor, ...]:
    combined: dict[_Key, Factor] = {}
    for polynomial, exponent in factors:
        work.check(polynomial)
        _integer(exponent, "denominator factor exponent")
        if exponent < 1:
            raise ValueError("denominator factor exponent must be positive")
        if polynomial.is_zero():
            raise ZeroDivisionError("an identically-zero denominator is undefined")
        previous = combined.get(polynomial.terms)
        value = exponent if previous is None else (max(exponent, previous[1]) if maximum else exponent + previous[1])
        work.budget._coefficient(Fraction(value))
        combined[polynomial.terms] = (polynomial, value)
    work.charge(max(1, len(combined).bit_length()) * sum(len(key) * (2 * work.reference.dimension + 1) + 1 for key in combined))
    return tuple(combined[key] for key in sorted(combined))


def _guards(conditions: Iterable[ExpPolynomial], work: _Work) -> tuple[ExpPolynomial, ...]:
    unique: dict[_Key, ExpPolynomial] = {}
    for polynomial in conditions:
        work.check(polynomial)
        if polynomial.is_zero():
            raise ZeroDivisionError("an identically-zero nonzero condition is impossible")
        if polynomial.terms not in unique:
            work.reserve(len(polynomial.terms) * (2 * polynomial.dimension + 1) + 1)
        unique[polynomial.terms] = polynomial
    work.charge(max(1, len(unique).bit_length()) * sum(len(key) * (2 * work.reference.dimension + 1) + 1 for key in unique))
    return tuple(unique[key] for key in sorted(unique))


@dataclass(frozen=True, init=False)
class RationalField:
    """A numerator divided by factored exponential polynomials, with guards.

    Direct construction takes ``numerator`` and optional ``denominator``.
    All expressions are immutable exact metadata with matching polynomial
    dimensions and budgets. ``nonzero_conditions`` records every original
    denominator, including poles removed by zero simplification or power zero.
    No numerical tolerance, unrestricted symbolic GCD or sampled zero test is
    used. ``is_zero`` means zero on this retained common regular domain.

    Binary arithmetic accepts fields; use :meth:`scale` for rational scalars
    and :meth:`from_polynomial` for polynomial features. Every complete public
    operation has a cumulative polynomial-work budget, including its internal
    multiplications and differentiations.
    """

    numerator: ExpPolynomial
    denominator_factors: tuple[Factor, ...]
    nonzero_conditions: tuple[ExpPolynomial, ...]

    def __init__(self, numerator: ExpPolynomial, denominator: ExpPolynomial | None = None):
        if not isinstance(numerator, ExpPolynomial):
            raise TypeError("numerator must be an ExpPolynomial")
        work = _Work(numerator)
        factors: tuple[Factor, ...] = ()
        guards: tuple[ExpPolynomial, ...] = ()
        if denominator is not None:
            _compatible(numerator, denominator)
            factors = ((denominator, 1),)
            guards = (denominator,)
        result = self._make(numerator, factors, guards, work)
        for name in ("numerator", "denominator_factors", "nonzero_conditions"):
            object.__setattr__(self, name, getattr(result, name))

    @classmethod
    def _make(
        cls, numerator: ExpPolynomial, factors: Sequence[Factor],
        guards: Sequence[ExpPolynomial], work: _Work,
    ) -> RationalField:
        work.check(numerator)
        canonical = _canonical_factors(factors, work)
        conditions = _guards(chain(guards, (factor for factor, _ in canonical)), work)
        result = object.__new__(cls)
        object.__setattr__(result, "numerator", numerator)
        object.__setattr__(result, "denominator_factors", () if numerator.is_zero() else canonical)
        object.__setattr__(result, "nonzero_conditions", conditions)
        return result

    @classmethod
    def from_polynomial(cls, polynomial: ExpPolynomial) -> RationalField:
        """Embed a polynomial without imposing additional nonzero conditions."""
        return cls(polynomial)

    @property
    def dimension(self) -> int:
        return self.numerator.dimension

    @property
    def budget(self) -> ExponentialBudget:
        return self.numerator.budget

    @property
    def denominator(self) -> ExpPolynomial:
        """Expand the arithmetic denominator under the polynomial work budget.

        The separate guards remain necessary even if this polynomial is one.
        """
        return _Work(self.numerator).product(self.denominator_factors)

    def is_zero(self) -> bool:
        return self.numerator.is_zero()

    def _other(self, other: object) -> RationalField:
        if not isinstance(other, RationalField):
            raise TypeError("binary operations require RationalField operands; use scale or from_polynomial")
        _compatible(self.numerator, other.numerator)
        return other

    def scale(self, value: Rational) -> RationalField:
        work = _Work(self.numerator)
        return self._make(work.scale(self.numerator, value), self.denominator_factors, self.nonzero_conditions, work)

    def __neg__(self) -> RationalField:
        return self.scale(-1)

    def _add(self, other: object, sign: int) -> RationalField:
        other = self._other(other)
        work = _Work(self.numerator)
        common = _canonical_factors((*self.denominator_factors, *other.denominator_factors), work, maximum=True)
        numerators = []
        for field in (self, other):
            present = {factor.terms: exponent for factor, exponent in field.denominator_factors}
            missing = tuple((factor, exponent - present.get(factor.terms, 0))
                            for factor, exponent in common if exponent > present.get(factor.terms, 0))
            numerators.append(work.multiply(field.numerator, work.product(missing)))
        numerator = work.add(numerators[0], numerators[1] if sign == 1 else work.scale(numerators[1], -1))
        return self._make(numerator, common, (*self.nonzero_conditions, *other.nonzero_conditions), work)

    def __add__(self, other: RationalField) -> RationalField:
        return self._add(other, 1)

    def __sub__(self, other: RationalField) -> RationalField:
        return self._add(other, -1)

    def __mul__(self, other: RationalField) -> RationalField:
        other = self._other(other)
        work = _Work(self.numerator)
        return self._make(work.multiply(self.numerator, other.numerator),
                          (*self.denominator_factors, *other.denominator_factors),
                          (*self.nonzero_conditions, *other.nonzero_conditions), work)

    def __truediv__(self, other: RationalField) -> RationalField:
        other = self._other(other)
        if other.is_zero():
            raise ZeroDivisionError("division by an identically-zero rational field")
        work = _Work(self.numerator)
        numerator = work.multiply(self.numerator, work.product(other.denominator_factors))
        return self._make(numerator, (*self.denominator_factors, (other.numerator, 1)),
                          (*self.nonzero_conditions, *other.nonzero_conditions, other.numerator), work)

    def __pow__(self, exponent: int) -> RationalField:
        exponent = _integer(exponent, "exponent", signed=True)
        work = _Work(self.numerator)
        work.charge(abs(exponent).bit_length() + 1)
        factors: tuple[Factor, ...]
        if exponent < 0:
            if self.is_zero():
                raise ZeroDivisionError("negative power of an identically-zero rational field")
            numerator = work.power(work.product(self.denominator_factors), -exponent)
            factors = ((self.numerator, -exponent),)
            guards = (*self.nonzero_conditions, self.numerator)
        else:
            numerator = work.power(self.numerator, exponent)
            factors = tuple((factor, power * exponent) for factor, power in self.denominator_factors) if exponent else ()
            guards = self.nonzero_conditions
        return self._make(numerator, factors, guards, work)

    def differentiate(self, axis: int, order: int = 1) -> RationalField:
        """Raw partial derivative retaining poles and factored denominators.

        Each step increments every axis-dependent factor exponent just once;
        it does not square an expanded denominator. All substeps share a work
        meter. No factorial normalization is applied.
        """
        axis, order = _integer(axis, "axis"), _integer(order, "derivative order")
        if axis >= self.dimension:
            raise ValueError("derivative axis must be less than dimension")
        if order > self.budget.max_derivative_order:
            raise ValueError("derivative order exceeds the resource budget")
        if not order:
            return self
        work = _Work(self.numerator)
        work.charge(order)
        result = self
        for _ in range(order):
            dependent = []
            factors = []
            for factor, exponent in result.denominator_factors:
                derivative = work.derivative(factor, axis)
                factors.append((factor, exponent + int(not derivative.is_zero())))
                if not derivative.is_zero():
                    dependent.append((factor, exponent, derivative))
            prefix = [work.constant(1)]
            for factor, _, _ in dependent:
                prefix.append(work.multiply(prefix[-1], factor))
            suffix = [work.constant(1)]
            for factor, _, _ in reversed(dependent):
                suffix.append(work.multiply(factor, suffix[-1]))
            suffix.reverse()
            numerator = work.multiply(work.derivative(result.numerator, axis), prefix[-1])
            for i, (_, exponent, derivative) in enumerate(dependent):
                term = work.multiply(work.multiply(result.numerator, derivative), work.multiply(prefix[i], suffix[i + 1]))
                numerator = work.add(numerator, work.scale(term, -exponent))
            result = self._make(numerator, factors, result.nonzero_conditions, work)
            if result.is_zero():
                break
        return result


@dataclass(frozen=True)
class RationalRelations:
    """Exact nullspace of the supplied finite candidate libraries.

    Each relation holds for every supplied family on its retained regular
    domain. The result makes no completeness or physical-law uniqueness claim
    outside those libraries and families.
    """

    relations: tuple[tuple[Fraction, ...], ...]
    rank: int
    candidate_count: int
    constraint_count: int
    nonzero_conditions: tuple[tuple[ExpPolynomial, ...], ...]


class _LinearBudget:
    def __init__(self, limits: RidgeBudget, polynomial_work: int, storage: int):
        self.limits, self.polynomial_work, self.storage = limits, polynomial_work, storage

    def _entries(self, count: int) -> None:
        self.limits._entries(self.storage + count)

    def _work(self, count: int) -> None:
        self.limits._work(self.polynomial_work + count)

    def _coefficient(self, value: Fraction) -> Fraction:
        return self.limits._coefficient(value)


def discover_rational_relations(
    families: Sequence[Sequence[RationalField]], *, budget: RidgeBudget | None = None,
) -> RationalRelations:
    """Match exact coefficients after clearing shared factored denominators.

    Rows of ``families`` are aligned candidate libraries. Every family must
    have the same nonzero candidate count, dimension and polynomial budget.
    Denominators are cleared using structural factor maxima, without a GCD.
    Zero features retain their original nonzero conditions.

    Polynomial arithmetic is metered cumulatively against both the common
    ``ExponentialBudget`` and ``RidgeBudget.max_elimination_work``. The latter
    also covers matrix construction and a conservative elimination estimate.
    ``max_matrix_entries`` conservatively bounds cumulative polynomial/guard
    snapshots, scratch terms, dense matrices and returned relation storage.
    These are scalar metadata slots, not a byte-accurate memory estimate.
    Refusal never returns a truncated or sampled relation.
    """
    limits = budget if budget is not None else RidgeBudget()
    if not isinstance(limits, RidgeBudget):
        raise TypeError("budget must be a RidgeBudget")
    if not families or not families[0]:
        raise ValueError("relation discovery requires nonempty candidate families")
    count = len(families[0])
    limits._entries(len(families) * count)
    first = families[0][0]
    if not isinstance(first, RationalField):
        raise TypeError("candidate libraries must contain RationalField objects")
    work = _Work(first.numerator, limits)
    matrix: list[list[Fraction]] = []
    all_guards: list[tuple[ExpPolynomial, ...]] = []
    for family in families:
        if len(family) != count:
            raise ValueError("every family must have the same candidate count")
        for field in family:
            first._other(field)
            work.check(field.numerator)
        conditions = _guards((guard for field in family for guard in field.nonzero_conditions), work)
        all_guards.append(conditions)
        common = _canonical_factors((factor for field in family for factor in field.denominator_factors), work, maximum=True)
        cleared = []
        for field in family:
            present = {factor.terms: exponent for factor, exponent in field.denominator_factors}
            missing = tuple((factor, exponent - present.get(factor.terms, 0))
                            for factor, exponent in common if exponent > present.get(factor.terms, 0))
            numerator = work.multiply(field.numerator, work.product(missing))
            work.reserve(len(numerator.terms) * (2 * first.dimension + 1) + 1)
            limits._entries(work.storage + 2 * len(matrix) * count + count * count)
            cleared.append({(frequency, powers): coefficient for frequency, powers, coefficient in numerator.terms})
        # Bound union and sorting before constructing either temporary. Count
        # duplicates conservatively so the preflight itself requires no set.
        key_bound = sum(len(polynomial) for polynomial in cleared)
        work.reserve(2 * key_bound * (2 * first.dimension + 1))
        limits._entries(work.storage + 2 * (len(matrix) + key_bound) * count + count * count)
        work.charge(key_bound * max(1, key_bound.bit_length()) * (2 * first.dimension + 1))
        keys = sorted({key for polynomial in cleared for key in polynomial})
        limits._entries(work.storage + 2 * (len(matrix) + len(keys)) * count + count * count)
        work.charge(len(keys) * count)
        matrix.extend([[polynomial.get(key, Fraction(0)) for polynomial in cleared] for key in keys])
        work.matrix_entries = len(matrix) * count
    constraints = len(matrix)
    if not matrix:
        matrix = [[Fraction(0)] * count]
    # Include both the retained input matrix and the elimination copy while
    # reserving returned basis entries separately inside solve_affine.
    linear_limits = _LinearBudget(limits, work.used, work.storage + 2 * len(matrix) * count)
    solved = solve_affine(matrix, None, linear_limits)
    return RationalRelations(solved.homogeneous_basis, solved.rank, count, constraints, tuple(all_guards))


__all__ = ["RationalField", "RationalRelations", "discover_rational_relations"]
