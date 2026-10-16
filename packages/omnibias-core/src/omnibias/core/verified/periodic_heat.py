# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Rigorous prescribed-data periodic heat and Cole--Hopf enclosures.

The exact plan represents ``Q_M(g)=sum(g**j/j!,j=0..M)``. The omitted
exponential series is a GLOBAL Fourier-l1 remainder: it can contain low modes
and must not be treated as a remainder supported beyond a frequency cutoff.
All spatial derivatives get their own remainder bound, including at time zero.

The true heat solution with initial data ``exp(g)`` satisfies
``tau >= exp(-||g||_l1)`` by the heat maximum principle. Consequently the
Cole--Hopf field solves the periodic irrotational Burgers system wherever its
numerical quotient can be enclosed. Failure of numerical bounds is inconclusive,
not evidence of a singularity or failure of the continuum solution.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from itertools import product

from omnibias.core.bell import stirling_second_row
from omnibias.core.multi_index import MultiIndex, multi_indices, num_multi_indices
from omnibias.core.periodic_heat import PeriodicHeatPlan
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.transcend import (
    cos_iv,
    exp_iv,
    require_rigorous_backend,
    set_strict_backend,
    sin_iv,
)


@dataclass(frozen=True)
class HeatEnclosure:
    """Finite Fourier enclosure plus a rigorous uniform derivative error."""

    finite_part: Interval
    remainder_bound: float
    value: Interval


@dataclass(frozen=True)
class BurgersJet:
    """Raw spatial derivative vectors in canonical multi-index order.

    ``heat_lower_bound`` is the strictly positive bound actually used for
    division on the transported input box. It can exceed the uniform
    maximum-principle bound when a local enclosure is sharper.
    """

    multi_indices: tuple[MultiIndex, ...]
    derivatives: tuple[tuple[Interval, ...], ...]
    heat_lower_bound: float

    @property
    def value(self) -> tuple[Interval, ...]:
        return self.derivatives[0]

    def derivative(self, alpha: Sequence[int]) -> tuple[Interval, ...]:
        """Retrieve a raw spatial partial, refusing unavailable orders."""
        index = _alpha(alpha, len(self.value))
        try:
            position = self.multi_indices.index(index)
        except ValueError as exc:
            raise ValueError("requested derivative is not stored in this jet") from exc
        return self.derivatives[position]


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a nonnegative integer, excluding bool")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def _alpha(values: Sequence[int] | None, dimension: int) -> MultiIndex:
    if values is None:
        return (0,) * dimension
    if len(values) != dimension:
        raise ValueError("derivative multi-index dimension must match the plan")
    return tuple(_integer(value, "derivative order") for value in values)


def _interval(value: Interval, label: str) -> Interval:
    if not isinstance(value, Interval):
        raise TypeError(f"{label} must be an Interval")
    endpoints = []
    for endpoint in (value.lo, value.hi):
        if isinstance(endpoint, bool) or not isinstance(endpoint, (int, float)):
            raise TypeError(f"{label} endpoints must be binary64 floats or exactly representable integers")
        try:
            converted = float(endpoint)
        except OverflowError as exc:
            raise ValueError(f"{label} endpoints must be finite and exactly representable") from exc
        if not math.isfinite(converted) or converted != endpoint:
            raise ValueError(f"{label} endpoints must be finite and exactly representable")
        endpoints.append(converted)
    return Interval(*endpoints)


def _inputs(plan: PeriodicHeatPlan, point: Sequence[Interval], time: Interval) -> tuple[tuple[Interval, ...], Interval]:
    if not isinstance(plan, PeriodicHeatPlan):
        raise TypeError("plan must be a PeriodicHeatPlan")
    if len(point) != plan.dimension:
        raise ValueError("point dimension must match the plan")
    box = tuple(_interval(value, "point") for value in point)
    elapsed = _interval(time, "time")
    if elapsed.lo < 0.0:
        raise ValueError("heat evolution requires nonnegative time")
    return box, elapsed


def _finite(value: Interval) -> Interval:
    if not math.isfinite(value.lo) or not math.isfinite(value.hi):
        raise ArithmeticError("nonfinite periodic enclosure; refine the plan, subdivide or rescale")
    return value


class _Evaluation:
    def __init__(self, plan: PeriodicHeatPlan):
        self.plan = plan
        self.used = 0
        self.remainders: dict[int, float] = {}
        self.mode_values: dict[tuple[int, ...], tuple[Interval, Interval, Interval]] = {}
        self.positivity_lower: float | None = None

    def charge(self, count: int = 1) -> None:
        self.used += count
        self.plan.budget._work(self.used)

    def exact(self, value: Fraction) -> Fraction:
        return self.plan.budget._coefficient(value)

    def rational(self, value: Fraction) -> Interval:
        self.charge()
        return _finite(Interval.from_rational(self.exact(value)))

    def add(self, left: Interval, right: Interval) -> Interval:
        self.charge()
        if left.lo == left.hi == 0.0:
            return right
        if right.lo == right.hi == 0.0:
            return left
        return _finite(left + right)

    def multiply(self, left: Interval, right: Interval) -> Interval:
        self.charge()
        if left.lo == left.hi == 0.0 or right.lo == right.hi == 0.0:
            return Interval.point(0.0)
        return _finite(left * right)

    def divide(self, numerator: Interval, denominator: Interval) -> Interval:
        self.charge(2)
        if denominator.lo <= 0.0:
            raise ArithmeticError("cannot establish a positive heat denominator on this box")
        if numerator.lo == numerator.hi == 0.0:
            return Interval.point(0.0)
        inverse = _finite(denominator.reciprocal())
        return self.multiply(numerator, inverse)

    def power(self, value: Fraction, exponent: int) -> Fraction:
        self.charge(exponent + 1)
        result = Fraction(1)
        for _ in range(exponent):
            result = self.exact(result * value)
        return result

    def remainder(self, order: int) -> float:
        if order in self.remainders:
            return self.remainders[order]
        a = self.plan.norm_bound
        if not a:
            self.remainders[order] = 0.0
            return 0.0
        self.charge(2 * (order + 1)**2)
        row = stirling_second_row(order)
        total = Fraction(0)
        for ell, coefficient in enumerate(row):
            coefficient = int(self.exact(Fraction(coefficient)))
            if not coefficient:
                continue
            q = max(self.plan.degree + 1 - ell, 0)
            term = self.power(a, ell + q)
            self.charge(q + 2)
            for divisor in range(1, q + 1):
                term = self.exact(term / divisor)
            total = self.exact(total + self.exact(coefficient * term))
        total = self.exact(total * self.power(Fraction(self.plan.max_frequency), order))
        self.charge()
        exponential = _finite(exp_iv(self.rational(a)))
        bound = self.multiply(self.rational(total), exponential).hi
        self.remainders[order] = max(0.0, bound)
        return self.remainders[order]

    def positivity(self) -> float:
        if self.positivity_lower is None:
            if not self.plan.norm_bound:
                self.positivity_lower = 1.0
            else:
                self.charge()
                self.positivity_lower = max(0.0, _finite(exp_iv(-self.rational(self.plan.norm_bound))).lo)
        return self.positivity_lower

    def mode(
        self, frequency: tuple[int, ...], box: tuple[Interval, ...], time: Interval,
    ) -> tuple[Interval, Interval, Interval]:
        if frequency in self.mode_values:
            return self.mode_values[frequency]
        self.charge(2 * self.plan.dimension + 3)
        phase = Interval.point(0.0)
        norm_squared = Fraction(0)
        for rate, coordinate in zip(frequency, box, strict=True):
            norm_squared = self.exact(norm_squared + self.exact(Fraction(rate) * rate))
            if rate:
                phase = self.add(phase, self.multiply(self.rational(Fraction(rate)), coordinate))
        cosine, sine = _finite(cos_iv(phase)), _finite(sin_iv(phase))
        decay = Interval.point(1.0)
        if norm_squared and time.hi:
            decay_rate = self.exact(self.plan.viscosity * norm_squared)
            decay = _finite(exp_iv(-self.multiply(self.rational(decay_rate), time)))
        result = (cosine, sine, decay)
        self.mode_values[frequency] = result
        return result

    def heat(self, box: tuple[Interval, ...], time: Interval, alpha: MultiIndex) -> HeatEnclosure:
        order = sum(alpha)
        self.charge(len(self.plan.heat_coefficients) * (self.plan.dimension + 2) + 1)
        finite = Interval.point(0.0)
        for frequency, real, imaginary in self.plan.heat_coefficients:
            multiplier = Fraction(1)
            for rate, exponent in zip(frequency, alpha, strict=True):
                multiplier = self.exact(multiplier * self.power(Fraction(rate), exponent))
            if not multiplier:
                continue
            real = self.exact(real * multiplier)
            imaginary = self.exact(imaginary * multiplier)
            if order % 4 == 1:
                real, imaginary = -imaginary, real
            elif order % 4 == 2:
                real, imaginary = -real, -imaginary
            elif order % 4 == 3:
                real, imaginary = imaginary, -real
            cosine, sine, decay = self.mode(frequency, box, time)
            term = self.add(self.multiply(self.rational(real), cosine),
                            -self.multiply(self.rational(imaginary), sine))
            finite = self.add(finite, self.multiply(term, decay))
        remainder = self.remainder(order)
        value = self.add(finite, Interval(-remainder, remainder))
        if not order:
            lower = max(value.lo, self.positivity())
            if lower > value.hi:
                raise ArithmeticError("heat approximation contradicts its positivity bound")
            value = Interval(lower, value.hi)
        return HeatEnclosure(finite, remainder, value)


def enclose_heat_derivative(
    plan: PeriodicHeatPlan, point: Sequence[Interval], time: Interval,
    alpha: Sequence[int] | None = None,
) -> HeatEnclosure:
    """Enclose the PURE heat field's raw spatial partial, without a drift shift.

    The global Fourier-l1 error of derivative order ``r`` is bounded by
    ``L**r exp(A) sum(S(r,l) A**(l+q)/q!)``, where
    ``q=max(M+1-l,0)``, ``A=||g||_l1`` and ``L=max|k|_1`` of the input.
    This follows from differentiating every omitted power ``g**j`` before
    bounding its norm. It is valid for all nonnegative times, including zero.
    Strict transcendental evaluation preserves prior sticky backend history.
    """
    box, elapsed = _inputs(plan, point, time)
    index = _alpha(alpha, plan.dimension)
    if sum(index) > plan.budget.max_derivative_order:
        raise ValueError("spatial derivative order exceeds the resource budget")
    evaluation = _Evaluation(plan)
    require_rigorous_backend()
    previous = set_strict_backend(True)
    try:
        return evaluation.heat(box, elapsed, index)
    finally:
        set_strict_backend(previous)


def enclose_burgers_jet(
    plan: PeriodicHeatPlan, point: Sequence[Interval], time: Interval, order: int = 0,
) -> BurgersJet:
    """Enclose raw spatial derivatives of ``drift-2*nu*grad(log(tau))``.

    Heat derivatives are evaluated at ``point-drift*time``. The multivariate
    Leibniz recurrence for ``tau*q=tau_i`` computes the quotient jet and
    retains each derivative's own global truncation error. One shared budget
    covers every heat derivative, mode, remainder, and quotient operation.
    A positive numerical denominator is required; inability to enclose it
    raises ``ArithmeticError`` without asserting a pole of the true solution.
    """
    box, elapsed = _inputs(plan, point, time)
    order = _integer(order, "jet order")
    if order + 1 > plan.budget.max_derivative_order:
        raise ValueError("required heat derivative order exceeds the resource budget")
    dimension = plan.dimension
    heat_count = num_multi_indices(dimension, order + 1)
    output_count = num_multi_indices(dimension, order)
    if heat_count > plan.budget.max_terms or output_count * dimension > plan.budget.max_terms:
        raise ValueError("periodic jet coefficient count exceeds the resource budget")
    pair_count = math.comb(2 * dimension + order, order)
    evaluation = _Evaluation(plan)
    evaluation.charge(heat_count * (dimension + 1) + output_count * dimension +
                      pair_count * dimension * (dimension + 3))
    indices = tuple(multi_indices(dimension, order))
    heat_indices = tuple(multi_indices(dimension, order + 1))
    require_rigorous_backend()
    previous = set_strict_backend(True)
    try:
        transported = tuple(evaluation.add(coordinate, -evaluation.multiply(evaluation.rational(drift), elapsed))
                            for coordinate, drift in zip(box, plan.drift, strict=True))
        heat = {alpha: evaluation.heat(transported, elapsed, alpha).value for alpha in heat_indices}
        denominator = heat[(0,) * dimension]
        if denominator.lo <= 0.0:
            raise ArithmeticError("cannot establish a positive heat denominator on this box")
        quotient: dict[MultiIndex, tuple[Interval, ...]] = {}
        derivatives: list[tuple[Interval, ...]] = []
        factor = evaluation.rational(evaluation.exact(-2 * plan.viscosity))
        for alpha in indices:
            row: list[Interval] = []
            for component in range(dimension):
                raised = tuple(value + int(axis == component) for axis, value in enumerate(alpha))
                numerator = heat[raised]
                for beta in product(*(range(value + 1) for value in alpha)):
                    if not any(beta):
                        continue
                    remainder = tuple(a - b for a, b in zip(alpha, beta, strict=True))
                    coefficient = Fraction(1)
                    for a, b in zip(alpha, beta, strict=True):
                        coefficient = evaluation.exact(coefficient * math.comb(a, b))
                    term = evaluation.multiply(evaluation.rational(coefficient),
                                               evaluation.multiply(heat[beta], quotient[remainder][component]))
                    numerator = evaluation.add(numerator, -term)
                row.append(evaluation.divide(numerator, denominator))
            quotient[alpha] = tuple(row)
            velocity = tuple(evaluation.multiply(factor, value) for value in row)
            if not any(alpha):
                velocity = tuple(evaluation.add(value, evaluation.rational(drift))
                                 for value, drift in zip(velocity, plan.drift, strict=True))
            derivatives.append(velocity)
        return BurgersJet(indices, tuple(derivatives), denominator.lo)
    finally:
        set_strict_backend(previous)


__all__ = ["BurgersJet", "HeatEnclosure", "enclose_burgers_jet", "enclose_heat_derivative"]
