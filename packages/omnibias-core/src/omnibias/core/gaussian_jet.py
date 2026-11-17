# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Exact forward inverse Taylor coefficients for Gaussian heat decisions.

An exact rational posterior at an exact rational base point defines the
positive Gaussian family without pretending that its log weights are rational.
The coefficients use the implicit equation and one reusable rational LDLT
factorization. These are ordinary Taylor coefficients, not raw derivatives.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING

from omnibias.core._rational_linear import factor_positive_definite
from omnibias.core.ridge_system import RidgeBudget

if TYPE_CHECKING:
    from omnibias.core.verified.gaussian_heat import GaussianHeatMap

Rational = int | Fraction
Vector = tuple[Fraction, ...]
Matrix = tuple[Vector, ...]


class _Arithmetic:
    def __init__(self, limits: RidgeBudget) -> None:
        self.limits = limits
        self.used = 0

    def _entries(self, count: int) -> None:
        self.limits._entries(count)

    def _work(self, count: int) -> None:
        self.used += count
        self.limits._work(self.used)

    def _coefficient(self, value: Fraction) -> Fraction:
        self._work(1)
        return self.limits._coefficient(value)

    def exact(self, value: Rational) -> Fraction:
        if isinstance(value, bool) or not isinstance(value, (int, Fraction)):
            raise TypeError("Gaussian series data must be exact integers or Fractions")
        return self._coefficient(Fraction(value))

    def add(self, left: Fraction, right: Fraction) -> Fraction:
        return self._coefficient(left + right)

    def mul(self, left: Fraction, right: Fraction) -> Fraction:
        return self._coefficient(left * right)

    def sum(self, values: Sequence[Fraction]) -> Fraction:
        result = Fraction(0)
        for value in values:
            result = self.add(result, value)
        return result


@dataclass(frozen=True, init=False)
class GaussianInverseSeries:
    r"""Exact local inverse series and the family/path it belongs to.

    At ``base_point=x0``, ``posterior=p`` is strictly positive and sums to one.
    This defines Gaussian log weights ``log(p_j)+|x0-a_j|²/(4*nu*s)`` up to a
    common constant. The implied target at the base is ``(x0-sum(p*a))/s``.
    Changing the supplied posterior changes that family. Use
    :meth:`from_equal_logits` to bind a supported existing exact model.

    ``target_coefficients`` contains an exact polynomial path, including its
    base value. ``coefficients[n]`` is the inverse's Taylor coefficient, equal
    to its raw derivative divided by n!. The finite inverse polynomial is not
    asserted to be exact away from zero; analytic remainder bounds live in
    ``verified.gaussian_analytic``. Construction refuses q>=1, even though a
    locally invertible branch might exist beyond this sufficient global bound.
    """

    centers: Matrix
    posterior: Vector
    base_point: Vector
    target_coefficients: Matrix
    coefficients: Matrix
    viscosity: Fraction
    time: Fraction
    diameter_squared: Fraction
    contraction_bound: Fraction
    budget: RidgeBudget

    def __init__(self) -> None:
        raise TypeError("use gaussian_inverse_series or GaussianInverseSeries.from_equal_logits")

    @property
    def dimension(self) -> int:
        return len(self.base_point)

    @property
    def mode_count(self) -> int:
        return len(self.centers)

    @property
    def order(self) -> int:
        return len(self.coefficients) - 1

    def evaluate(self, value: Rational) -> Vector:
        """Evaluate the finite polynomial exactly; does not add a remainder."""
        arithmetic = _Arithmetic(self.budget)
        parameter = arithmetic.exact(value)
        result = self.coefficients[-1]
        for row in reversed(self.coefficients[:-1]):
            result = tuple(arithmetic.add(coefficient, arithmetic.mul(parameter, component))
                           for coefficient, component in zip(row, result, strict=True))
        return result

    @classmethod
    def from_equal_logits(
        cls, model: GaussianHeatMap, base_point: Sequence[Rational],
        higher_target_coefficients: Sequence[Sequence[Rational]], *, budget: RidgeBudget | None = None,
    ) -> GaussianInverseSeries:
        """Bind an existing model at an exactly equal-logit rational base point.

        The exact check gives posterior 1/m without evaluating exponentials.
        Unequal logits are refused, not replaced by rounded rational weights.
        Supply rows v_1 through v_N; the correct constant target is computed.
        """
        from omnibias.core.verified.gaussian_heat import GaussianHeatMap

        if not isinstance(model, GaussianHeatMap):
            raise TypeError("model must be a GaussianHeatMap")
        limits = model.budget if budget is None else budget
        if not isinstance(limits, RidgeBudget):
            raise TypeError("budget must be a RidgeBudget")
        arithmetic = _Arithmetic(limits)
        if not isinstance(base_point, Sequence) or not isinstance(higher_target_coefficients, Sequence):
            raise TypeError("base point and target coefficients must be finite sequences")
        if len(higher_target_coefficients) > limits.max_order:
            raise ValueError("Gaussian inverse series order exceeds the resource budget")
        arithmetic._entries(8 * model.mode_count * model.dimension
                            + 6 * (len(higher_target_coefficients)+1) * (model.mode_count+model.dimension)
                            + 4 * model.dimension**2)
        if len(base_point) != model.dimension:
            raise ValueError("base point dimension must match the model")
        point = tuple(arithmetic.exact(value) for value in base_point)
        denominator = arithmetic.mul(arithmetic.mul(Fraction(4), model.viscosity), model.time)
        logits: list[Fraction] = []
        for center, bias in zip(model.centers, model.biases, strict=True):
            squares = []
            for coordinate, value in zip(point, center, strict=True):
                delta = arithmetic.add(coordinate, -value)
                squares.append(arithmetic.mul(delta, delta))
            logits.append(arithmetic.add(bias, -arithmetic._coefficient(arithmetic.sum(squares) / denominator)))
        if any(value != logits[0] for value in logits[1:]):
            raise ValueError("base Gaussian logits must be exactly equal for this factory")
        posterior = (Fraction(1, model.mode_count),) * model.mode_count
        mean = tuple(arithmetic.sum(tuple(arithmetic.mul(weight, center[axis])
                                          for weight, center in zip(posterior, model.centers, strict=True)))
                     for axis in range(model.dimension))
        base_target = tuple(arithmetic._coefficient(arithmetic.add(value, -center) / model.time)
                            for value, center in zip(point, mean, strict=True))
        return _build(model.centers, posterior, point, (base_target, *higher_target_coefficients),
                      model.viscosity, model.time, limits, arithmetic)


def _build(
    centers: Sequence[Sequence[Rational]], posterior: Sequence[Rational], base_point: Sequence[Rational],
    target_coefficients: Sequence[Sequence[Rational]], viscosity: Rational, time: Rational,
    limits: RidgeBudget, arithmetic: _Arithmetic,
) -> GaussianInverseSeries:
    if not all(isinstance(value, Sequence) for value in (centers, posterior, base_point, target_coefficients)):
        raise TypeError("Gaussian series data must be finite sequences")
    if not centers or not base_point or not target_coefficients:
        raise ValueError("Gaussian series needs positive mode count, dimension and coefficient count")
    dimension, count, order = len(base_point), len(centers), len(target_coefficients)-1
    if order > limits.max_order:
        raise ValueError("Gaussian inverse series order exceeds the resource budget")
    arithmetic._entries(8 * count * dimension + 6 * (order+1) * (count+dimension) + 4 * dimension**2)
    arithmetic._work(count*count*dimension + dimension**3 + (count+dimension)*(order+1)**2)
    if len(posterior) != count or any(not isinstance(row, Sequence) or len(row) != dimension for row in centers):
        raise ValueError("centers and posterior must match the base point dimension")
    if any(not isinstance(row, Sequence) or len(row) != dimension for row in target_coefficients):
        raise ValueError("target coefficient rows must match the base point dimension")
    a = tuple(tuple(arithmetic.exact(value) for value in row) for row in centers)
    p = tuple(arithmetic.exact(value) for value in posterior)
    x0 = tuple(arithmetic.exact(value) for value in base_point)
    target = tuple(tuple(arithmetic.exact(value) for value in row) for row in target_coefficients)
    nu, s = arithmetic.exact(viscosity), arithmetic.exact(time)
    if nu <= 0 or s <= 0:
        raise ValueError("viscosity and time must be strictly positive")
    if any(value <= 0 for value in p) or arithmetic.sum(p) != 1:
        raise ValueError("posterior must be strictly positive and sum exactly to one")
    relative = tuple(tuple(arithmetic.add(value, -reference) for value, reference in zip(row, a[0], strict=True)) for row in a)
    mean = tuple(arithmetic.sum(tuple(arithmetic.mul(weight, row[axis]) for weight, row in zip(p, relative, strict=True)))
                 for axis in range(dimension))
    expected_target = tuple(arithmetic._coefficient(arithmetic.add(arithmetic.add(x, -origin), -mu) / s)
                            for x, origin, mu in zip(x0, a[0], mean, strict=True))
    if target[0] != expected_target:
        raise ValueError("constant target does not equal the exact velocity at the declared base point")
    diameter = Fraction(0)
    for i, row in enumerate(relative):
        for other in relative[:i]:
            difference = tuple(arithmetic.add(value, -reference) for value, reference in zip(row, other, strict=True))
            diameter = max(diameter, arithmetic.sum(tuple(arithmetic.mul(value, value) for value in difference)))
    covariance_scale = arithmetic.mul(arithmetic.mul(Fraction(2), nu), s)
    q = arithmetic._coefficient(diameter / arithmetic.mul(Fraction(4), covariance_scale))
    if q >= 1:
        raise ValueError("Gaussian inverse series requires the sufficient contraction bound q<1")
    centered = tuple(tuple(arithmetic.add(value, -center) for value, center in zip(row, mean, strict=True)) for row in relative)
    matrix = tuple(tuple(arithmetic.add(Fraction(int(i == j)), -arithmetic._coefficient(
        arithmetic.sum(tuple(arithmetic.mul(weight, arithmetic.mul(row[i], row[j]))
                             for weight, row in zip(p, centered, strict=True))) / covariance_scale))
                         for j in range(dimension)) for i in range(dimension))
    factor = factor_positive_definite(matrix, arithmetic) if order else None
    rates = tuple(tuple(arithmetic._coefficient(value / covariance_scale) for value in row) for row in relative)
    exponential = [(Fraction(1),) * count]
    logits = [(Fraction(0),) * count]
    denominator = [Fraction(1)]
    means = [mean]
    coefficients = [x0]

    def moments(e: Vector) -> tuple[Fraction, Vector]:
        weights = tuple(arithmetic.mul(weight, value) for weight, value in zip(p, e, strict=True))
        return arithmetic.sum(weights), tuple(arithmetic.sum(tuple(arithmetic.mul(weight, row[axis])
            for weight, row in zip(weights, relative, strict=True))) for axis in range(dimension))

    for n in range(1, order+1):
        proposed_exp = tuple(arithmetic._coefficient(arithmetic.sum(tuple(
            arithmetic.mul(arithmetic.mul(Fraction(k), logits[k][j]), exponential[n-k][j])
            for k in range(1, n))) / n) for j in range(count))
        proposed_denominator, proposed_moment = moments(proposed_exp)
        proposed_mean = tuple(arithmetic.add(arithmetic.add(proposed_moment[axis],
            -arithmetic.sum(tuple(arithmetic.mul(denominator[k], means[n-k][axis]) for k in range(1, n)))),
            -arithmetic.mul(proposed_denominator, mean[axis])) for axis in range(dimension))
        rhs = tuple(arithmetic.add(arithmetic.mul(s, value), component)
                    for value, component in zip(target[n], proposed_mean, strict=True))
        assert factor is not None
        coefficient = factor.solve(rhs, arithmetic)
        coefficients.append(coefficient)
        logit = tuple(arithmetic.sum(tuple(arithmetic.mul(rate, value) for rate, value in zip(row, coefficient, strict=True)))
                      for row in rates)
        logits.append(logit)
        exponential.append(tuple(arithmetic.add(value, known) for value, known in zip(logit, proposed_exp, strict=True)))
        den, moment = moments(exponential[-1])
        denominator.append(den)
        means.append(tuple(arithmetic.add(moment[axis], -arithmetic.sum(tuple(
            arithmetic.mul(denominator[k], means[n-k][axis]) for k in range(1, n+1)))) for axis in range(dimension)))
    result = object.__new__(GaussianInverseSeries)
    for name, value in (("centers", a), ("posterior", p), ("base_point", x0), ("target_coefficients", target),
                        ("coefficients", tuple(coefficients)), ("viscosity", nu), ("time", s),
                        ("diameter_squared", diameter), ("contraction_bound", q), ("budget", limits)):
        object.__setattr__(result, name, value)
    return result


def gaussian_inverse_series(
    centers: Sequence[Sequence[Rational]], posterior: Sequence[Rational], base_point: Sequence[Rational],
    target_coefficients: Sequence[Sequence[Rational]], *, viscosity: Rational = 1, time: Rational = 1,
    budget: RidgeBudget | None = None,
) -> GaussianInverseSeries:
    """Construct exact Taylor coefficients with one shared implicit factorization."""
    limits = RidgeBudget() if budget is None else budget
    if not isinstance(limits, RidgeBudget):
        raise TypeError("budget must be a RidgeBudget")
    return _build(centers, posterior, base_point, target_coefficients, viscosity, time, limits, _Arithmetic(limits))


__all__ = ["GaussianInverseSeries", "gaussian_inverse_series"]
