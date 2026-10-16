# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Interval evaluation of exact exponential polynomials and rational fields.

These enclosures connect exact identity algebra to numerical queries on real
boxes. Every original denominator condition is checked, even after algebraic
cancellation. Failure to exclude zero is inconclusive, not proof of a pole.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from omnibias.core.exp_polynomial import ExponentialBudget, ExpPolynomial
from omnibias.core.rational_field import RationalField
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.transcend import (
    exp_iv,
    require_rigorous_backend,
    set_strict_backend,
)


def _box(box: Sequence[Interval], dimension: int) -> tuple[Interval, ...]:
    if len(box) != dimension:
        raise ValueError("evaluation box dimension must match the exact field")
    result = tuple(box)
    normalized: list[Interval] = []
    for axis in result:
        if not isinstance(axis, Interval):
            raise TypeError("evaluation box entries must be Interval objects")
        endpoints: list[float] = []
        for endpoint in (axis.lo, axis.hi):
            if type(endpoint) not in (int, float):
                raise TypeError("box endpoints must be binary64 floats or exactly representable integers")
            try:
                converted = float(endpoint)
            except OverflowError as exc:
                raise ValueError("box endpoints must be finite and exactly representable") from exc
            if not math.isfinite(converted) or converted != endpoint:
                raise ValueError("box endpoints must be finite and exactly representable")
            endpoints.append(converted)
        normalized.append(Interval(*endpoints))
    return tuple(normalized)


def _finite(value: Interval) -> Interval:
    if not math.isfinite(value.lo) or not math.isfinite(value.hi):
        raise ArithmeticError("nonfinite interval evaluation; subdivide or rescale the expression")
    return value


class _Evaluation:
    """One bounded evaluation scope, preserving existing backend provenance."""

    def __init__(self, box: tuple[Interval, ...], budget: ExponentialBudget) -> None:
        self.box = box
        self.budget = budget
        self.work = 0
        self.previous_strict: bool | None = None
        self.cache: dict[ExpPolynomial, Interval] = {}

    def charge(self, count: int = 1) -> None:
        self.work += count
        if self.work > self.budget.max_work:
            raise ValueError("interval evaluation exceeds the arithmetic work budget")

    def close(self) -> None:
        if self.previous_strict is not None:
            set_strict_backend(self.previous_strict)

    def multiply(self, left: Interval, right: Interval) -> Interval:
        self.charge()
        return _finite(left * right)

    def power(self, value: Interval, order: int) -> Interval:
        # Check every intermediate before any possible 0*infinity operation.
        base = _finite(value.abs()) if order % 2 == 0 else value
        result = Interval.point(1.0)
        while order:
            if order & 1:
                result = self.multiply(result, base)
            order >>= 1
            if order:
                base = self.multiply(base, base)
        return result

    def exponential(self, argument: Interval) -> Interval:
        self.charge()
        if self.previous_strict is None:
            require_rigorous_backend()
            self.previous_strict = set_strict_backend(True)
        return _finite(exp_iv(argument))

    def polynomial(self, polynomial: ExpPolynomial) -> Interval:
        if polynomial in self.cache:
            return self.cache[polynomial]
        self.charge(len(polynomial.terms))
        result = Interval.point(0.0)
        for frequency, powers, coefficient in polynomial.terms:
            term = _finite(Interval.from_rational(coefficient))
            phase = Interval.point(0.0)
            for axis, rate, order in zip(self.box, frequency, powers, strict=True):
                self.charge()
                if rate:
                    product = self.multiply(_finite(Interval.from_rational(rate)), axis)
                    self.charge()
                    phase = _finite(phase + product)
                if order:
                    term = self.multiply(term, self.power(axis, order))
            if any(frequency):
                term = self.multiply(term, self.exponential(phase))
            self.charge()
            result = _finite(result + term)
        self.cache[polynomial] = result
        return result


def evaluate_exp_polynomial(polynomial: ExpPolynomial, box: Sequence[Interval]) -> Interval:
    """Enclose an exact polynomial-exponential field over a finite real box.

    Rational coefficients and frequencies are injected with outward rounding.
    A nonzero exponential frequency requires the rigorous mpmath backend;
    ordinary polynomial evaluation needs no optional dependency. Work is
    bounded cumulatively by ``polynomial.budget.max_work``. Nonfinite
    intermediate bounds raise ``ArithmeticError`` instead of producing a
    misleading finite interval. No rigorous backend provenance is reset.
    """
    if not isinstance(polynomial, ExpPolynomial):
        raise TypeError("polynomial must be an ExpPolynomial")
    evaluation = _Evaluation(_box(box, polynomial.dimension), polynomial.budget)
    try:
        return evaluation.polynomial(polynomial)
    finally:
        evaluation.close()


def evaluate_rational_field(field: RationalField, box: Sequence[Interval]) -> Interval:
    """Enclose a rational field only where all original guards are discharged.

    A guard with a known strict global sign needs no numerical evaluation.
    Every other guard must have an interval excluding zero throughout the
    box. A ``ZeroDivisionError`` means that this evaluator could not prove
    the required nonzero condition; it does not assert that a pole exists.
    Even zero fields and powers zero retain their original domain guards.
    Arithmetic overflow and unresolved denominator bounds are explicit
    refusals. The computation uses one work budget across all guard, numerator
    and denominator evaluations and preserves the caller's backend state.
    """
    if not isinstance(field, RationalField):
        raise TypeError("field must be a RationalField")
    evaluation = _Evaluation(_box(box, field.dimension), field.budget)
    try:
        evaluation.charge(len(field.nonzero_conditions) + len(field.denominator_factors))
        for condition in field.nonzero_conditions:
            evaluation.charge(len(condition.terms) * (field.dimension + 2))
            if condition.known_global_sign() in (-1, 1):
                continue
            if evaluation.polynomial(condition).contains_zero():
                raise ZeroDivisionError("cannot establish an original nonzero condition throughout the box")
        numerator = evaluation.polynomial(field.numerator)
        if field.is_zero() or not field.denominator_factors:
            return numerator
        denominator = Interval.point(1.0)
        for factor, order in field.denominator_factors:
            enclosed = evaluation.polynomial(factor)
            if enclosed.contains_zero():
                raise ZeroDivisionError("cannot exclude zero from a denominator enclosure")
            denominator = evaluation.multiply(denominator, evaluation.power(enclosed, order))
        if denominator.contains_zero():
            raise ZeroDivisionError("cannot exclude zero from the accumulated denominator enclosure")
        evaluation.charge()
        inverse = _finite(denominator.reciprocal())
        return evaluation.multiply(numerator, inverse)
    finally:
        evaluation.close()


__all__ = ["evaluate_exp_polynomial", "evaluate_rational_field"]
