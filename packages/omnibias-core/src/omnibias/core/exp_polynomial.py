# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Bounded exact sparse algebra for polynomial factors times exponentials.

A term ``(frequency, powers, coefficient)`` denotes
``coefficient * x**powers * exp(frequency.x)``. Distinct frequency/power
pairs form an exact coefficient representation over the rationals. Derivatives
are ordinary raw derivatives, not factorial-normalized Taylor coefficients.

Finite expressions are closed under addition, multiplication and differentiation;
their support may grow under these operations.
It does not represent every smooth function, solve arbitrary PDEs, or perform
numerical evaluation. Construction and every operation have explicit resource
limits; exhaustion raises ``ValueError`` rather than returning a truncated result.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from fractions import Fraction

Rational = int | Fraction
Frequency = tuple[Fraction, ...]
Powers = tuple[int, ...]
ExponentialTerm = tuple[Frequency, Powers, Fraction]
TermInput = tuple[Sequence[Rational], Sequence[int], Rational]
_Key = tuple[Frequency, Powers]
_Table = dict[_Key, Fraction]


def _integer(value: object, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer >= {minimum}")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _exact(value: object) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, (int, Fraction)):
        raise TypeError("exponential-polynomial metadata must be int or Fraction")
    return Fraction(value)


@dataclass(frozen=True)
class ExponentialBudget:
    """Limits for each complete exact construction or algebra operation.

    ``max_terms`` bounds distinct nonzero terms in every intermediate sparse
    accumulator. ``max_work`` bounds a cumulative conservative count of scalar
    arithmetic and metadata/sorting work, including all power or derivative
    substeps. Coefficient and frequency numerators/denominators are bit-limited
    after each exact operation. These are refusal limits, not truncation rules.
    """

    max_terms: int = 4096
    max_work: int = 2_000_000
    max_coefficient_bits: int = 4096
    max_dimension: int = 32
    max_total_degree: int = 128
    max_derivative_order: int = 128

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            minimum = 0 if name in {"max_total_degree", "max_derivative_order"} else 1
            object.__setattr__(self, name, _integer(getattr(self, name), name, minimum))

    def _work(self, count: int) -> None:
        if count > self.max_work:
            raise ValueError("exponential-polynomial arithmetic work exceeds the resource budget")

    def _coefficient(self, value: Fraction) -> Fraction:
        if max(value.numerator.bit_length(), value.denominator.bit_length()) > self.max_coefficient_bits:
            raise ValueError("exponential-polynomial coefficient or frequency exceeds the exact bit budget")
        return value


class _Arithmetic:
    def __init__(self, budget: ExponentialBudget):
        self.budget = budget
        self.used = 0

    def work(self, count: int) -> None:
        self.used += count
        self.budget._work(self.used)

    def coefficient(self, value: Fraction) -> Fraction:
        return self.budget._coefficient(value)

    def accumulate(self, table: _Table, key: _Key, value: Fraction) -> None:
        if not value:
            return
        self.work(1)
        value = self.coefficient(table.get(key, Fraction(0)) + value)
        if not value:
            table.pop(key, None)
        else:
            if key not in table and len(table) >= self.budget.max_terms:
                raise ValueError("exponential-polynomial terms exceed the resource budget")
            table[key] = value


def _limits(budget: ExponentialBudget | None) -> ExponentialBudget:
    result = budget if budget is not None else ExponentialBudget()
    if not isinstance(result, ExponentialBudget):
        raise TypeError("budget must be an ExponentialBudget")
    return result


def _dimension(value: int, budget: ExponentialBudget) -> int:
    result = _integer(value, "dimension", 1)
    if result > budget.max_dimension:
        raise ValueError("exponential-polynomial dimension exceeds the resource budget")
    budget._work(result)
    return result


def _terms(table: _Table, dimension: int, arithmetic: _Arithmetic) -> tuple[ExponentialTerm, ...]:
    # The comparison/copy estimate is reserved before sorting or allocating output.
    arithmetic.work(len(table) * max(1, len(table).bit_length()) * (2 * dimension + 1))
    return tuple((frequency, powers, coefficient)
                 for (frequency, powers), coefficient in sorted(table.items()))


def _canonical(
    dimension: int, terms: Iterable[TermInput], arithmetic: _Arithmetic,
) -> tuple[ExponentialTerm, ...]:
    arithmetic.work(dimension)
    result: _Table = {}
    for term in terms:
        arithmetic.work(2 * dimension + 3)
        if not isinstance(term, Sequence) or len(term) != 3:
            raise ValueError("each exponential-polynomial term must be a frequency, powers, coefficient triple")
        raw_frequency, raw_powers, raw_coefficient = term
        if not isinstance(raw_frequency, Sequence) or len(raw_frequency) != dimension:
            raise ValueError("frequency must have exactly dimension entries")
        if not isinstance(raw_powers, Sequence) or len(raw_powers) != dimension:
            raise ValueError("powers must have exactly dimension entries")
        frequency = tuple(arithmetic.coefficient(_exact(value)) for value in raw_frequency)
        powers = tuple(_integer(value, "power") for value in raw_powers)
        if sum(powers) > arithmetic.budget.max_total_degree:
            raise ValueError("exponential-polynomial total degree exceeds the resource budget")
        coefficient = arithmetic.coefficient(_exact(raw_coefficient))
        arithmetic.accumulate(result, (frequency, powers), coefficient)
    return _terms(result, dimension, arithmetic)


def _table(polynomial: ExpPolynomial, arithmetic: _Arithmetic) -> _Table:
    arithmetic.work(len(polynomial.terms) * (2 * polynomial.dimension + 1))
    return {(frequency, powers): coefficient for frequency, powers, coefficient in polynomial.terms}


def _multiply(left: _Table, right: _Table, dimension: int, arithmetic: _Arithmetic) -> _Table:
    if not left or not right:
        return {}
    # Each pair constructs two dimension-sized keys, multiplies a coefficient,
    # and potentially accumulates it. Reject before allocating these products.
    arithmetic.work(len(left) * len(right) * (2 * dimension + 1))
    degree_left = max(sum(powers) for _, powers in left)
    degree_right = max(sum(powers) for _, powers in right)
    if degree_left + degree_right > arithmetic.budget.max_total_degree:
        raise ValueError("exponential-polynomial total degree exceeds the resource budget")
    result: _Table = {}
    for (left_frequency, left_powers), left_coefficient in left.items():
        for (right_frequency, right_powers), right_coefficient in right.items():
            frequency = tuple(arithmetic.coefficient(a + b)
                              for a, b in zip(left_frequency, right_frequency, strict=True))
            powers = tuple(a + b for a, b in zip(left_powers, right_powers, strict=True))
            coefficient = arithmetic.coefficient(left_coefficient * right_coefficient)
            arithmetic.accumulate(result, (frequency, powers), coefficient)
    return result


@dataclass(frozen=True)
class ExpPolynomial:
    """Immutable canonical sparse exponential polynomial with rational data.

    Direct construction snapshots input containers, merges repeated pairs,
    removes zeros, and sorts lexicographically by frequency then powers.
    Only exact ``int`` / ``Fraction`` metadata is accepted; bools and floats
    are refused. Binary operations require matching dimensions and budgets.
    """

    dimension: int
    terms: tuple[ExponentialTerm, ...]
    budget: ExponentialBudget = ExponentialBudget()

    def __post_init__(self) -> None:
        limits = _limits(self.budget)
        dimension = _dimension(self.dimension, limits)
        canonical = _canonical(dimension, self.terms, _Arithmetic(limits))
        object.__setattr__(self, "dimension", dimension)
        object.__setattr__(self, "terms", canonical)
        object.__setattr__(self, "budget", limits)

    @classmethod
    def from_terms(
        cls, dimension: int, terms: Iterable[TermInput], *, budget: ExponentialBudget | None = None,
    ) -> ExpPolynomial:
        """Construct from exact triples, accepting finite iterables of terms."""
        limits = _limits(budget)
        dimension = _dimension(dimension, limits)
        result = object.__new__(cls)
        object.__setattr__(result, "dimension", dimension)
        object.__setattr__(result, "terms", _canonical(dimension, terms, _Arithmetic(limits)))
        object.__setattr__(result, "budget", limits)
        return result

    @classmethod
    def constant(
        cls, value: Rational, dimension: int, *, budget: ExponentialBudget | None = None,
    ) -> ExpPolynomial:
        """Construct an exact rational constant in the specified dimension."""
        limits = _limits(budget)
        dimension = _dimension(dimension, limits)
        return cls.from_terms(dimension, [((0,) * dimension, (0,) * dimension, value)], budget=limits)

    @classmethod
    def exponential(
        cls, frequency: Sequence[Rational], coefficient: Rational = 1, *,
        budget: ExponentialBudget | None = None,
    ) -> ExpPolynomial:
        """Construct ``coefficient * exp(frequency.x)``."""
        limits = _limits(budget)
        dimension = _dimension(len(frequency), limits)
        return cls.from_terms(dimension, [(frequency, (0,) * dimension, coefficient)], budget=limits)

    @classmethod
    def monomial(
        cls, powers: Sequence[int], coefficient: Rational = 1, *,
        budget: ExponentialBudget | None = None,
    ) -> ExpPolynomial:
        """Construct the ordinary monomial ``coefficient * x**powers``."""
        limits = _limits(budget)
        dimension = _dimension(len(powers), limits)
        return cls.from_terms(dimension, [((0,) * dimension, powers, coefficient)], budget=limits)

    def _finish(self, table: _Table, arithmetic: _Arithmetic) -> ExpPolynomial:
        result = object.__new__(ExpPolynomial)
        object.__setattr__(result, "dimension", self.dimension)
        object.__setattr__(result, "terms", _terms(table, self.dimension, arithmetic))
        object.__setattr__(result, "budget", self.budget)
        return result

    def _compatible(self, other: object) -> ExpPolynomial:
        if not isinstance(other, ExpPolynomial):
            raise TypeError("binary operations require ExpPolynomial operands; use constant or scale for scalars")
        if self.dimension != other.dimension:
            raise ValueError("exponential-polynomial dimensions must match")
        if self.budget != other.budget:
            raise ValueError("exponential-polynomial budgets must match")
        return other

    def is_zero(self) -> bool:
        """Whether every exact coefficient vanishes."""
        return not self.terms

    def known_global_sign(self) -> int | None:
        """Sufficient STRICT global sign on real space, or ``None`` if unknown.

        Returns zero exactly for the zero expression. For nonzero expressions,
        coefficients must share a sign, every polynomial power must be even,
        and a pure exponential term must be present to make the sign strict.
        Failure of this sufficient test does not establish a real zero.
        """
        if not self.terms:
            return 0
        self.budget._work(len(self.terms) * (self.dimension + 2))
        sign = 1 if self.terms[0][2] > 0 else -1
        if any((coefficient > 0) != (sign > 0) or any(power % 2 for power in powers)
               for _, powers, coefficient in self.terms):
            return None
        return sign if any(not any(powers) for _, powers, _ in self.terms) else None

    def scale(self, value: Rational) -> ExpPolynomial:
        """Multiply every term by an exact rational scalar."""
        arithmetic = _Arithmetic(self.budget)
        factor = arithmetic.coefficient(_exact(value))
        arithmetic.work(len(self.terms) * (2 * self.dimension + 2) + 1)
        table = {(frequency, powers): arithmetic.coefficient(coefficient * factor)
                 for frequency, powers, coefficient in self.terms} if factor else {}
        return self._finish(table, arithmetic)

    def __neg__(self) -> ExpPolynomial:
        return self.scale(-1)

    def _add(self, other: object, sign: int) -> ExpPolynomial:
        other = self._compatible(other)
        arithmetic = _Arithmetic(self.budget)
        result = _table(self, arithmetic)
        arithmetic.work(len(other.terms) * (2 * self.dimension + 2))
        for frequency, powers, coefficient in other.terms:
            arithmetic.accumulate(result, (frequency, powers), coefficient if sign == 1 else -coefficient)
        return self._finish(result, arithmetic)

    def __add__(self, other: ExpPolynomial) -> ExpPolynomial:
        return self._add(other, 1)

    def __sub__(self, other: ExpPolynomial) -> ExpPolynomial:
        return self._add(other, -1)

    def __mul__(self, other: ExpPolynomial) -> ExpPolynomial:
        other = self._compatible(other)
        arithmetic = _Arithmetic(self.budget)
        result = _multiply(_table(self, arithmetic), _table(other, arithmetic), self.dimension, arithmetic)
        return self._finish(result, arithmetic)

    def __pow__(self, exponent: int) -> ExpPolynomial:
        """Nonnegative integral power with a shared budget across all substeps."""
        exponent = _integer(exponent, "exponent")
        arithmetic = _Arithmetic(self.budget)
        arithmetic.work(exponent.bit_length() + 2 * self.dimension + 1)
        degree = max((sum(powers) for _, powers, _ in self.terms), default=0)
        if degree * exponent > self.budget.max_total_degree:
            raise ValueError("exponential-polynomial total degree exceeds the resource budget")
        zero_frequency = (Fraction(0),) * self.dimension
        zero_powers = (0,) * self.dimension
        result: _Table = {(zero_frequency, zero_powers): Fraction(1)}
        base = _table(self, arithmetic)
        while exponent:
            if exponent & 1:
                result = _multiply(result, base, self.dimension, arithmetic)
            exponent >>= 1
            if exponent:
                base = _multiply(base, base, self.dimension, arithmetic)
        return self._finish(result, arithmetic)

    def differentiate(self, axis: int, order: int = 1) -> ExpPolynomial:
        """Apply ``D_axis**order``, sharing one budget across every product rule."""
        axis = _integer(axis, "axis")
        if axis >= self.dimension:
            raise ValueError("derivative axis must be less than dimension")
        order = _integer(order, "derivative order")
        if order > self.budget.max_derivative_order:
            raise ValueError("derivative order exceeds the resource budget")
        if not order:
            return self
        arithmetic = _Arithmetic(self.budget)
        arithmetic.work(order)
        table = _table(self, arithmetic)
        for _ in range(order):
            arithmetic.work(len(table) * (self.dimension + 2))
            result: _Table = {}
            for (frequency, powers), coefficient in table.items():
                if frequency[axis]:
                    arithmetic.accumulate(result, (frequency, powers),
                                          arithmetic.coefficient(coefficient * frequency[axis]))
                if powers[axis]:
                    lower = list(powers)
                    lower[axis] -= 1
                    arithmetic.accumulate(result, (frequency, tuple(lower)),
                                          arithmetic.coefficient(coefficient * powers[axis]))
            table = result
            if not table:
                break
        return self._finish(table, arithmetic)


__all__ = ["ExpPolynomial", "ExponentialBudget", "ExponentialTerm", "Frequency", "Powers", "Rational", "TermInput"]
