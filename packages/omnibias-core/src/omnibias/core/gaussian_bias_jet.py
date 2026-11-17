# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Exact inverse Taylor series for simultaneous Gaussian bias/target paths."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING

from omnibias.core._rational_linear import factor_positive_definite
from omnibias.core.gaussian_jet import _Arithmetic
from omnibias.core.ridge_system import RidgeBudget

if TYPE_CHECKING:
    from omnibias.core.verified.gaussian_heat import GaussianHeatMap

Rational = int | Fraction
Vector = tuple[Fraction, ...]
Matrix = tuple[Vector, ...]


def _vector(values: Sequence[Rational], dimension: int, name: str) -> None:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a finite exact sequence")
    if len(values) != dimension:
        raise ValueError(f"{name} has the wrong dimension")
    if any(isinstance(value, bool) or not isinstance(value, (int, Fraction)) for value in values):
        raise TypeError(f"{name} must contain exact int or Fraction values, excluding bool")


@dataclass(frozen=True, init=False)
class GaussianBiasInverseSeries:
    r"""Exact ordinary Taylor coefficients at an equal-original-logit root.

    ``target_coefficients`` and ``bias_coefficients`` describe v(t) and the
    ORIGINAL b(t). Their order-zero values are the exact base velocity and
    the model biases. ``relative_bias_coefficients`` subtracts the first
    mode separately at every order, including order zero. These relative
    order-zero biases need not vanish; the base-point Gaussian contrasts
    cancel them exactly. Higher common bias gauges do not affect the series.

    ``coefficients[n]`` is the physical inverse coefficient x^(n)(0)/n!.
    ``posterior_coefficients[n]`` contains the normalized original-mode
    probability coefficients; their sums are one for n=0 and zero otherwise.
    The supplied model, centers, viscosity and age stay fixed. The base target
    is inferred, not accepted as a rounded root. q<1 is a sufficient global
    inverse condition. This finite polynomial has no remainder certificate
    and is not asserted to be exact away from its base point.
    """

    model: GaussianHeatMap
    base_point: Vector
    target_coefficients: Matrix
    bias_coefficients: Matrix
    relative_bias_coefficients: Matrix
    coefficients: Matrix
    posterior_coefficients: Matrix
    budget: RidgeBudget

    def __init__(self) -> None:
        raise TypeError("use gaussian_bias_inverse_series")

    @property
    def order(self) -> int:
        return len(self.coefficients)-1

    @property
    def dimension(self) -> int:
        return len(self.base_point)

    @property
    def mode_count(self) -> int:
        return self.model.mode_count

    def evaluate(self, value: Rational) -> Vector:
        """Evaluate the finite inverse polynomial exactly, without a remainder."""
        arithmetic = _Arithmetic(self.budget)
        parameter = arithmetic.exact(value)
        arithmetic._entries(4*(self.dimension+1))
        arithmetic._work(4*(self.order+1)*(self.dimension+1))
        result = self.coefficients[-1]
        for row in reversed(self.coefficients[:-1]):
            result = tuple(arithmetic.add(coefficient, arithmetic.mul(parameter, component))
                           for coefficient, component in zip(row, result, strict=True))
        return result


def _build_equal_logits(
    model: GaussianHeatMap, base_point: Sequence[Rational],
    higher_target_coefficients: Sequence[Sequence[Rational]], higher_bias_coefficients: Sequence[Sequence[Rational]],
    limits: RidgeBudget, arithmetic: _Arithmetic,
) -> GaussianBiasInverseSeries:
    """Build with the caller's cumulative meter, including binding and solves."""
    from omnibias.core.verified.gaussian_heat import GaussianHeatMap

    if not isinstance(model, GaussianHeatMap):
        raise TypeError("model must be a GaussianHeatMap")
    if not isinstance(limits, RidgeBudget):
        raise TypeError("budget must be a RidgeBudget")
    if arithmetic.limits != limits:
        raise ValueError("shared arithmetic and series limits must agree")
    dimension, count = model.dimension, model.mode_count
    _vector(base_point, dimension, "base_point")
    for name, rows in (("higher_target_coefficients", higher_target_coefficients), ("higher_bias_coefficients", higher_bias_coefficients)):
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
            raise TypeError(f"{name} must be a finite sequence of exact rows")
    order = len(higher_target_coefficients)
    if order != len(higher_bias_coefficients):
        raise ValueError("higher target and bias arrays must have the same number of rows")
    if order > limits.max_order:
        raise ValueError("Gaussian bias inverse series order exceeds the resource budget")
    for row in higher_target_coefficients:
        _vector(row, dimension, "target coefficient row")
    for row in higher_bias_coefficients:
        _vector(row, count, "bias coefficient row")
    arithmetic._entries(32*count*dimension + 16*(order+1)*(count+dimension) + 8*dimension**2 + 64)
    arithmetic._work(64*count*dimension**2 + 32*dimension**3 + 8*order*dimension**2
                     + 32*(count+dimension)*(order+1)**2 + 64*count*dimension*(order+1) + 64)
    centers = tuple(tuple(arithmetic.exact(value) for value in row) for row in model.centers)
    original_bias = tuple(arithmetic.exact(value) for value in model.biases)
    nu, time = arithmetic.exact(model.viscosity), arithmetic.exact(model.time)
    if arithmetic.exact(model.contraction_bound) >= 1:
        raise ValueError("Gaussian bias inverse series requires the sufficient contraction bound q<1")
    point = tuple(arithmetic.exact(value) for value in base_point)
    higher_target = tuple(tuple(arithmetic.exact(value) for value in row) for row in higher_target_coefficients)
    higher_bias = tuple(tuple(arithmetic.exact(value) for value in row) for row in higher_bias_coefficients)
    bias_coefficients = (original_bias, *higher_bias)
    relative_bias = tuple(tuple(arithmetic.add(value, -row[0]) for value in row) for row in bias_coefficients)
    relative = tuple(tuple(arithmetic.add(value, -reference) for value, reference in zip(row, centers[0], strict=True))
                     for row in centers)
    position = tuple(arithmetic.add(value, -reference) for value, reference in zip(point, centers[0], strict=True))
    denominator = arithmetic.mul(arithmetic.mul(Fraction(4), nu), time)
    for row, bias in zip(relative, relative_bias[0], strict=True):
        norm = arithmetic.sum(tuple(arithmetic.mul(value, value) for value in row))
        dot = arithmetic.sum(tuple(arithmetic.mul(value, coordinate) for value, coordinate in zip(row, position, strict=True)))
        contrast = arithmetic.add(norm, -arithmetic.mul(Fraction(2), dot))
        if arithmetic.add(bias, -arithmetic._coefficient(contrast/denominator)):
            raise ValueError("base original Gaussian logits must be exactly equal")
    weight = arithmetic._coefficient(Fraction(1, count))
    posterior0 = (weight,)*count
    mean0 = tuple(arithmetic.mul(weight, arithmetic.sum(tuple(row[axis] for row in relative))) for axis in range(dimension))
    target0 = tuple(arithmetic._coefficient(arithmetic.add(value, -average)/time)
                    for value, average in zip(position, mean0, strict=True))
    target = (target0, *higher_target)
    centered = tuple(tuple(arithmetic.add(value, -average) for value, average in zip(row, mean0, strict=True)) for row in relative)
    covariance_scale = arithmetic._coefficient(denominator/2)
    matrix = tuple(tuple(arithmetic.add(Fraction(int(i == j)), -arithmetic._coefficient(arithmetic.mul(weight,
        arithmetic.sum(tuple(arithmetic.mul(row[i], row[j]) for row in centered)))/covariance_scale))
                         for j in range(dimension)) for i in range(dimension))
    factor = factor_positive_definite(matrix, arithmetic) if order else None
    rates = tuple(tuple(arithmetic._coefficient(value/covariance_scale) for value in row) for row in relative)
    exponentials = [(Fraction(1),)*count]
    logits = [(Fraction(0),)*count]
    denominators = [Fraction(1)]
    means = [mean0]
    coefficients = [point]
    posteriors = [posterior0]

    def moments(exponential: Vector) -> tuple[Fraction, Vector]:
        weighted = tuple(arithmetic.mul(weight, value) for value in exponential)
        return arithmetic.sum(weighted), tuple(arithmetic.sum(tuple(arithmetic.mul(value, row[axis])
            for value, row in zip(weighted, relative, strict=True))) for axis in range(dimension))

    for n in range(1, order+1):
        proposed = tuple(arithmetic.add(relative_bias[n][j], arithmetic._coefficient(arithmetic.sum(tuple(
            arithmetic.mul(arithmetic.mul(Fraction(k), logits[k][j]), exponentials[n-k][j])
            for k in range(1, n)))/n)) for j in range(count))
        proposed_denominator, proposed_moment = moments(proposed)
        proposed_mean = tuple(arithmetic.add(arithmetic.add(proposed_moment[axis], -arithmetic.sum(tuple(
            arithmetic.mul(denominators[k], means[n-k][axis]) for k in range(1, n)))),
            -arithmetic.mul(proposed_denominator, mean0[axis])) for axis in range(dimension))
        rhs = tuple(arithmetic.add(arithmetic.mul(time, value), component)
                    for value, component in zip(target[n], proposed_mean, strict=True))
        assert factor is not None
        coefficient = factor.solve(rhs, arithmetic)
        coefficients.append(coefficient)
        feedback = tuple(arithmetic.sum(tuple(arithmetic.mul(rate, value) for rate, value in zip(row, coefficient, strict=True)))
                         for row in rates)
        logits.append(tuple(arithmetic.add(gamma, value) for gamma, value in zip(relative_bias[n], feedback, strict=True)))
        # The proposed coefficient already includes gamma_n exactly once.
        exponentials.append(tuple(arithmetic.add(known, value) for known, value in zip(proposed, feedback, strict=True)))
        den, _ = moments(exponentials[-1])
        denominators.append(den)
        posterior = tuple(arithmetic.add(arithmetic.mul(weight, exponentials[n][j]), -arithmetic.sum(tuple(
            arithmetic.mul(denominators[k], posteriors[n-k][j]) for k in range(1, n+1)))) for j in range(count))
        if arithmetic.sum(posterior) != 0:
            raise ArithmeticError("posterior Taylor normalization replay failed")
        posteriors.append(posterior)
        mean = tuple(arithmetic.sum(tuple(arithmetic.mul(value, row[axis]) for value, row in zip(posterior, relative, strict=True)))
                     for axis in range(dimension))
        means.append(mean)
        replay = tuple(arithmetic.add(arithmetic.mul(time, value), component)
                       for value, component in zip(target[n], mean, strict=True))
        if replay != coefficient:
            raise ArithmeticError("Gaussian inverse Taylor coefficient replay failed")
    result = object.__new__(GaussianBiasInverseSeries)
    for name, value in (("model", model), ("base_point", point), ("target_coefficients", target),
                        ("bias_coefficients", bias_coefficients), ("relative_bias_coefficients", relative_bias),
                        ("coefficients", tuple(coefficients)), ("posterior_coefficients", tuple(posteriors)), ("budget", limits)):
        object.__setattr__(result, name, value)
    return result


def gaussian_bias_inverse_series(
    model: GaussianHeatMap, base_point: Sequence[Rational],
    higher_target_coefficients: Sequence[Sequence[Rational]], higher_bias_coefficients: Sequence[Sequence[Rational]],
    *, budget: RidgeBudget | None = None,
) -> GaussianBiasInverseSeries:
    """Build exact ordinary joint-path coefficients with one reused factor.

    Supply N rows for target orders 1..N and N rows for bias orders 1..N,
    including explicit zero rows. Both paths use ordinary Taylor coefficients,
    not raw derivatives. Base target is computed at the supplied exactly
    equal-original-logit point; posterior there is uniform. The model's budget
    is used unless overridden. All intermediate rational arithmetic, retained
    series storage, and O((m+d)*N²+m*d*N+(m+N)*d²+d³) work are bounded cumulatively.
    Invalid metadata, unsupported base logits, q>=1 and resource exhaustion
    raise; no numerical backend, exponential evaluation or fitted root is used.
    """
    from omnibias.core.verified.gaussian_heat import GaussianHeatMap

    if not isinstance(model, GaussianHeatMap):
        raise TypeError("model must be a GaussianHeatMap")
    limits = model.budget if budget is None else budget
    if not isinstance(limits, RidgeBudget):
        raise TypeError("budget must be a RidgeBudget")
    return _build_equal_logits(model, base_point, higher_target_coefficients, higher_bias_coefficients, limits, _Arithmetic(limits))


__all__ = ["GaussianBiasInverseSeries", "gaussian_bias_inverse_series"]
