# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Exact Gaussian heat-mixture data and rigorous Cole-Hopf velocity maps."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from fractions import Fraction

from omnibias.core.ridge_system import RidgeBudget
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.mean_map import MeanMap

Rational = int | Fraction
Vector = tuple[Fraction, ...]
Matrix = tuple[Vector, ...]


def _exact(value: Rational, budget: RidgeBudget) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, (int, Fraction)):
        raise TypeError("Gaussian heat metadata must be exact int or Fraction, excluding bool")
    return budget._coefficient(Fraction(value))


def _float(value: int | float) -> float:
    if type(value) not in (int, float):
        raise TypeError("numerical coordinates must be binary64 floats or exactly representable integers")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError("numerical coordinates must be finite binary64 values") from exc
    if not math.isfinite(result) or result != value:
        raise ValueError("numerical coordinates must be finite and exactly representable as binary64")
    return result


def _finite(value: Interval) -> Interval:
    if not math.isfinite(value.lo) or not math.isfinite(value.hi):
        raise ArithmeticError("nonfinite Gaussian heat enclosure; subdivide or rescale the data")
    return value


def _rational(value: Fraction) -> Interval:
    try:
        return _finite(Interval.from_rational(value))
    except (OverflowError, ValueError) as exc:
        raise ArithmeticError("exact Gaussian heat coefficient exceeds the binary64 range") from exc


def _add(left: Interval, right: Interval) -> Interval:
    return _finite(left + right)


def _multiply(left: Interval, right: Interval) -> Interval:
    return _finite(left * right)


@dataclass(frozen=True, init=False)
class GaussianHeatMap:
    r"""A positive finite Gaussian heat family at an exact positive time s.

    With ``nu=viscosity`` and ``s=time``, the heat potential is
    ``tau=s**(-d/2)*sum(exp(b_j-|x-a_j|**2/(4*nu*s)))`` and the
    Cole-Hopf velocity is ``U(x)=(x-E[a])/s``. ``time`` may include an
    initial smoothing offset; it must be strictly positive. The common
    Gaussian normalization is immaterial to this velocity. Weights are
    positive ``exp(biases[j])``, with exact rational biases, not supplied
    floating amplitudes.

    The global scaled inverse map ``y -> target+E[a]/s``, where ``y=x/s``,
    has Euclidean Lipschitz bound ``contraction_bound=diameter_squared/(8*nu*s)``.
    A bound strictly below one implies contraction; values at or above one
    make no uniqueness claim. In particular, small-time mixtures can have
    multiple velocity inverses even if their centers have full affine rank.

    Exact centering and common-bias subtraction precede binary conversion.
    Interval methods reuse the strict mean-map backend without resetting
    fallback history. Numerical inputs are finite binary64 intervals;
    nonfinite intermediates are refused. Floating proposals require only
    stdlib and are never certificates. Metadata is immutable and bounded by
    the shared exact budget; enclosing solver iteration budgets are separate.
    """

    centers: Matrix
    biases: Vector
    viscosity: Fraction
    time: Fraction
    budget: RidgeBudget
    diameter_squared: Fraction
    contraction_bound: Fraction
    _mean_map: MeanMap = field(repr=False)
    _reference_scaled: Vector = field(repr=False)
    _inverse_time: Fraction = field(repr=False)
    _velocity_scale: Fraction = field(repr=False)

    def __init__(
        self, centers: Sequence[Sequence[Rational]], biases: Sequence[Rational], *,
        viscosity: Rational = 1, time: Rational = 1, budget: RidgeBudget | None = None,
    ) -> None:
        limits = RidgeBudget() if budget is None else budget
        if not isinstance(limits, RidgeBudget):
            raise TypeError("budget must be a RidgeBudget")
        if not isinstance(centers, Sequence) or not isinstance(biases, Sequence):
            raise TypeError("centers and biases must be finite sequences")
        if not centers or not isinstance(centers[0], Sequence) or not centers[0]:
            raise ValueError("Gaussian heat centers need positive count and dimension")
        count, dimension = len(centers), len(centers[0])
        limits._entries(16 * count * dimension + 16 * count + 8 * dimension * dimension + 16 * dimension)
        rank_rows = max(1, count - 1)
        rank_work = 3 * rank_rows * dimension * min(rank_rows, dimension) + rank_rows * dimension
        limits._work(64 * (count + 1) * (dimension + 1) ** 2 + rank_work
                     + 32 * count * count * dimension + 64 * count * dimension + 64 * dimension * dimension)
        if any(not isinstance(row, Sequence) for row in centers):
            raise TypeError("each Gaussian heat center must be a finite sequence")
        if len(biases) != count or any(len(row) != dimension for row in centers):
            raise ValueError("centers must be rectangular and match the bias count")
        nu, smoothing_time = _exact(viscosity, limits), _exact(time, limits)
        if nu <= 0 or smoothing_time <= 0:
            raise ValueError("viscosity and time must be strictly positive")
        frozen_centers = tuple(tuple(_exact(value, limits) for value in row) for row in centers)
        frozen_biases = tuple(_exact(value, limits) for value in biases)
        relative = tuple(tuple(limits._coefficient(value - reference)
                               for value, reference in zip(row, frozen_centers[0], strict=True)) for row in frozen_centers)
        inverse_time = limits._coefficient(1 / smoothing_time)
        twice_nu = limits._coefficient(2 * nu)
        gaussian_denominator = limits._coefficient(limits._coefficient(2 * twice_nu) * smoothing_time)
        rates = tuple(tuple(limits._coefficient(value / twice_nu) for value in row) for row in relative)
        shifted_biases = []
        for row, bias in zip(relative, frozen_biases, strict=True):
            squared = Fraction(0)
            for value in row:
                squared = limits._coefficient(squared + limits._coefficient(value * value))
            centered_bias = limits._coefficient(bias - frozen_biases[0])
            shifted_biases.append(limits._coefficient(centered_bias - limits._coefficient(squared / gaussian_denominator)))
        diameter = Fraction(0)
        for i, center in enumerate(relative):
            for other in relative[:i]:
                distance = Fraction(0)
                for value, reference in zip(center, other, strict=True):
                    difference = limits._coefficient(value - reference)
                    distance = limits._coefficient(distance + limits._coefficient(difference * difference))
                diameter = max(diameter, distance)
        contraction = limits._coefficient(diameter / limits._coefficient(2 * gaussian_denominator))
        reference_scaled = tuple(limits._coefficient(value * inverse_time) for value in frozen_centers[0])
        velocity_scale = limits._coefficient(twice_nu * inverse_time)
        mean_map = MeanMap(rates, shifted_biases, (Fraction(0),) * dimension, budget=limits)
        for name, stored_value in (("centers", frozen_centers), ("biases", frozen_biases), ("viscosity", nu),
                                   ("time", smoothing_time), ("budget", limits), ("diameter_squared", diameter),
                                   ("contraction_bound", contraction), ("_mean_map", mean_map),
                                   ("_reference_scaled", reference_scaled), ("_inverse_time", inverse_time),
                                   ("_velocity_scale", velocity_scale)):
            object.__setattr__(self, name, stored_value)

    @property
    def dimension(self) -> int:
        return len(self.centers[0])

    @property
    def mode_count(self) -> int:
        return len(self.centers)

    def _evaluation_work(self) -> int:
        """Conservative scalar work for one interval or floating evaluation."""
        return self._mean_map._evaluation_work() + 32 * self.dimension * self.dimension + 32 * self.dimension

    def _box(self, box: Sequence[Interval]) -> tuple[Interval, ...]:
        if len(box) != self.dimension:
            raise ValueError("box dimension must match Gaussian heat dimension")
        result = []
        for axis in box:
            if not isinstance(axis, Interval):
                raise TypeError("box entries must be Interval objects")
            result.append(Interval(_float(axis.lo), _float(axis.hi)))
        return tuple(result)

    def _target(self, target: Sequence[Rational]) -> Vector:
        if len(target) != self.dimension:
            raise ValueError("target dimension must match Gaussian heat dimension")
        return tuple(_exact(value, self.budget) for value in target)

    def _centered(self, box: tuple[Interval, ...], *, physical: bool) -> tuple[Interval, ...]:
        if physical:
            return tuple(_multiply(_finite(axis - _rational(reference)), _rational(self._inverse_time))
                         for axis, reference in zip(box, self.centers[0], strict=True))
        return tuple(_finite(axis - _rational(reference))
                     for axis, reference in zip(box, self._reference_scaled, strict=True))

    def _velocity(self, centered: tuple[Interval, ...]) -> tuple[Interval, ...]:
        mean = self._mean_map.residual(centered)
        scale = _rational(self._velocity_scale)
        return tuple(_finite(axis - _multiply(value, scale)) for axis, value in zip(centered, mean, strict=True))

    def velocity(self, box: Sequence[Interval]) -> tuple[Interval, ...]:
        """Enclose U(x) over a physical-coordinate box."""
        self.budget._work(self._evaluation_work())
        return self._velocity(self._centered(self._box(box), physical=True))

    def scaled_velocity(self, box: Sequence[Interval]) -> tuple[Interval, ...]:
        """Enclose U(s*y) over a scaled-coordinate box y=x/s."""
        self.budget._work(self._evaluation_work())
        return self._velocity(self._centered(self._box(box), physical=False))

    def scaled_fixed_point(self, box: Sequence[Interval], target: Sequence[Rational]) -> tuple[Interval, ...]:
        """Enclose target+E[a]/s, the inverse-velocity fixed-point map."""
        self.budget._work(self._evaluation_work())
        frozen = self._box(box)
        goal = self._target(target)
        mean = self._mean_map.residual(self._centered(frozen, physical=False))
        offsets = tuple(self.budget._coefficient(value + reference)
                        for value, reference in zip(goal, self._reference_scaled, strict=True))
        scale = _rational(self._velocity_scale)
        return tuple(_add(_rational(offset), _multiply(value, scale)) for offset, value in zip(offsets, mean, strict=True))

    def scaled_jacobian(self, box: Sequence[Interval]) -> tuple[tuple[Interval, ...], ...]:
        """Enclose d[U(s*y)]/dy = I-Cov(a)/(2*viscosity*time)."""
        self.budget._work(self._evaluation_work())
        covariance = self._mean_map.jacobian(self._centered(self._box(box), physical=False))
        scale = _rational(self._velocity_scale)
        return tuple(tuple(_finite(Interval.point(float(i == j)) - _multiply(value, scale))
                           for j, value in enumerate(row)) for i, row in enumerate(covariance))

    def scaled_jacobian_action(self, box: Sequence[Interval], vector: Sequence[Interval]) -> tuple[Interval, ...]:
        """Enclose ``(I-Cov(a)/(2*nu*s)) @ vector`` without a dense matrix."""
        direction = self._box(vector)
        self.budget._work(64 * (self.mode_count + 1) * (self.dimension + 1) + 32 * self.dimension)
        action = self._mean_map.jacobian_action(self._centered(self._box(box), physical=False), direction)
        scale = _rational(self._velocity_scale)
        return tuple(_finite(component - _multiply(value, scale))
                     for component, value in zip(direction, action, strict=True))

    def float_scaled_fixed_point(self, point: Sequence[int | float], target: Sequence[Rational]) -> tuple[float, ...]:
        """Untrusted stdlib fixed-point proposal, with no rigorous backend needed."""
        self.budget._work(self._evaluation_work())
        if len(point) != self.dimension:
            raise ValueError("point dimension must match Gaussian heat dimension")
        coordinates = tuple(_float(value) for value in point)
        goal = self._target(target)
        try:
            centered = tuple(value - float(reference) for value, reference in zip(coordinates, self._reference_scaled, strict=True))
            if not all(math.isfinite(value) for value in centered):
                raise ArithmeticError("nonfinite centered Gaussian heat proposal")
            mean = self._mean_map.float_evaluate(centered).residual
            scale = float(self._velocity_scale)
            offsets = tuple(self.budget._coefficient(value + reference)
                            for value, reference in zip(goal, self._reference_scaled, strict=True))
            result = tuple(math.fsum((float(offset), scale * value)) for offset, value in zip(offsets, mean, strict=True))
            if not all(math.isfinite(value) for value in result):
                raise ArithmeticError("nonfinite Gaussian heat fixed-point proposal")
            return result
        except (OverflowError, ZeroDivisionError) as exc:
            raise ArithmeticError("Gaussian heat proposal exceeds the numerical range") from exc


__all__ = ["GaussianHeatMap"]
