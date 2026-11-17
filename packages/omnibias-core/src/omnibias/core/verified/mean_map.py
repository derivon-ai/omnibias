# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Exact finite exponential-family data and rigorous mean-map enclosures."""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from fractions import Fraction

from omnibias.core._rational_linear import solve_affine
from omnibias.core.ridge_system import RidgeBudget
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.transcend import (
    exp_iv,
    ln_iv,
    require_rigorous_backend,
    set_strict_backend,
)

Rational = int | Fraction
Vector = tuple[Fraction, ...]
Matrix = tuple[Vector, ...]


def _exact(value: Rational, budget: RidgeBudget) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, (int, Fraction)):
        raise TypeError("mean-map metadata must be exact int or Fraction, excluding bool")
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
        raise ArithmeticError("nonfinite mean-map enclosure; subdivide or rescale the data")
    return value


def _rational(value: Fraction) -> Interval:
    try:
        return _finite(Interval.from_rational(value))
    except (OverflowError, ValueError) as exc:
        raise ArithmeticError("exact mean-map coefficient exceeds the binary64 range") from exc


def _add(left: Interval, right: Interval) -> Interval:
    return _finite(left + right)


def _multiply(left: Interval, right: Interval) -> Interval:
    return _finite(left * right)


def _positive_sum(left: Interval, right: Interval) -> Interval:
    result = _add(left, right)
    return Interval(max(0.0, result.lo), result.hi)


@contextmanager
def _strict() -> Iterator[None]:
    require_rigorous_backend()
    previous = set_strict_backend(True)
    try:
        yield
    finally:
        set_strict_backend(previous)


def _linear(coefficients: Vector, box: tuple[Interval, ...], constant: Fraction = Fraction(0)) -> Interval:
    result = _rational(constant)
    for coefficient, axis in zip(coefficients, box, strict=True):
        if coefficient and not (axis.lo == axis.hi == 0.0):
            result = _add(result, _multiply(_rational(coefficient), axis))
    return result


def _exponential(value: Interval) -> Interval:
    result = _finite(exp_iv(value))
    return Interval(max(0.0, result.lo), result.hi)


def _logsum(values: Sequence[float]) -> Interval:
    """Rigorous log-sum-exp of exact binary64 entries, without overflow."""
    shift = max(values)
    total = Interval.point(0.0)
    for value in values:
        argument = _finite(Interval.point(value) - Interval.point(shift))
        # The selected maximum is a structural upper bound, despite rounding.
        argument = Interval(argument.lo, min(0.0, argument.hi))
        total = _positive_sum(total, _exponential(argument))
    return _add(Interval.point(shift), _finite(ln_iv(total)))


def _probabilities(logits: tuple[Interval, ...]) -> tuple[Interval, ...]:
    shift = max(value.hi for value in logits)
    scaled: list[Interval] = []
    for value in logits:
        argument = _finite(value - Interval.point(shift))
        argument = Interval(argument.lo, min(0.0, argument.hi))
        exponential = _exponential(argument)
        scaled.append(Interval(exponential.lo, min(1.0, exponential.hi)))
    prefix, suffix = [Interval.point(0.0)], [Interval.point(0.0)]
    for value in scaled:
        prefix.append(_positive_sum(prefix[-1], value))
    for value in reversed(scaled):
        suffix.append(_positive_sum(value, suffix[-1]))
    suffix.reverse()
    result: list[Interval] = []
    for index, value in enumerate(scaled):
        other = _positive_sum(prefix[index], suffix[index + 1])
        # q/(q+r) is increasing in q and decreasing in r. This remains
        # meaningful if all lower exponential endpoints underflow to zero.
        lower = 0.0
        if value.lo > 0.0:
            denominator = _positive_sum(Interval.point(value.lo), Interval.point(other.hi))
            lower = _finite(Interval.point(value.lo) / denominator).lo
        upper = 1.0
        if other.lo > 0.0:
            denominator = _positive_sum(Interval.point(value.hi), Interval.point(other.lo))
            upper = _finite(Interval.point(value.hi) / denominator).hi
        result.append(Interval(max(0.0, lower), min(1.0, upper)))
    return tuple(result)


@dataclass(frozen=True)
class MeanEvaluation:
    """Untrusted floating-point quantities for proposing a Newton step.

    ``reduced_objective`` omits the constant ``biases[0]`` and retains small
    descent differences in the presence of a large common bias offset.
    """

    objective: float
    residual: tuple[float, ...]
    covariance: tuple[tuple[float, ...], ...]
    reduced_objective: float


@dataclass(frozen=True, init=False)
class MeanMap:
    r"""A finite exponential family with immutable exact rational metadata.

    ``p_j(x) = exp(modes[j].x + biases[j]) / sum_l exp(...)`` and the
    residual is ``sum_j p_j modes[j] - target``. The Jacobian is its
    covariance matrix. The objective is ``logsumexp(logits) - target.x``.
    Affine rank is computed exactly; duplicate modes are allowed.

    Interval methods require the rigorous mpmath backend, preserve sticky
    fallback history, and restore the prior strict-mode setting. Coordinates
    are finite binary64 intervals. Exact common mode/bias offsets are removed
    before numerical conversion; original objective offsets remain present.
    Remaining nonfinite arithmetic is refused. Wide boxes can give loose
    enclosures. ``float_evaluate`` provides no certificate.

    The shared budget bounds metadata, rank elimination and a conservative
    scalar work/storage estimate for each evaluation. It is not a byte-level
    memory bound or a budget for an enclosing iterative solver.
    """

    modes: Matrix
    biases: Vector
    target: Vector
    budget: RidgeBudget
    _relative_modes: Matrix = field(repr=False)
    _relative_biases: Vector = field(repr=False)
    _reference_residual: Vector = field(repr=False)
    _affine_rank: int = field(repr=False)

    def __init__(
        self, modes: Sequence[Sequence[Rational]], biases: Sequence[Rational],
        target: Sequence[Rational], *, budget: RidgeBudget | None = None,
    ) -> None:
        limits = RidgeBudget() if budget is None else budget
        if not isinstance(limits, RidgeBudget):
            raise TypeError("budget must be a RidgeBudget")
        if not all(isinstance(values, Sequence) for values in (modes, biases, target)):
            raise TypeError("modes, biases and target must be finite sequences")
        if not modes or not target:
            raise ValueError("mean maps need positive mode count and dimension")
        count, dimension = len(modes), len(target)
        limits._entries(8 * count * dimension + 8 * count + 8 * dimension * dimension + 8 * dimension)
        rows = max(1, count - 1)
        rank_work = 3 * rows * dimension * min(rows, dimension) + rows * dimension
        limits._work(64 * (count + 1) * (dimension + 1) ** 2 + rank_work + 8 * count * dimension)
        if any(not isinstance(row, Sequence) for row in modes):
            raise TypeError("each mode must be a finite sequence")
        if len(biases) != count or any(len(row) != dimension for row in modes):
            raise ValueError("modes must be rectangular and match target/bias dimensions")
        frozen_modes = tuple(tuple(_exact(value, limits) for value in row) for row in modes)
        frozen_biases = tuple(_exact(value, limits) for value in biases)
        frozen_target = tuple(_exact(value, limits) for value in target)
        relative_modes = tuple(tuple(limits._coefficient(value - reference)
                                     for value, reference in zip(row, frozen_modes[0], strict=True)) for row in frozen_modes)
        relative_biases = tuple(limits._coefficient(value - frozen_biases[0]) for value in frozen_biases)
        reference_residual = tuple(limits._coefficient(value - goal)
                                   for value, goal in zip(frozen_modes[0], frozen_target, strict=True))
        rank = solve_affine(relative_modes[1:], None, limits).rank if count > 1 else 0
        for name, value in (("modes", frozen_modes), ("biases", frozen_biases), ("target", frozen_target),
                            ("budget", limits), ("_relative_modes", relative_modes), ("_relative_biases", relative_biases),
                            ("_reference_residual", reference_residual), ("_affine_rank", rank)):
            object.__setattr__(self, name, value)

    @property
    def dimension(self) -> int:
        return len(self.target)

    @property
    def mode_count(self) -> int:
        return len(self.modes)

    @property
    def affine_rank(self) -> int:
        return self._affine_rank

    def _evaluation_work(self) -> int:
        """Conservative scalar-operation bound for one numerical method call."""
        return 64 * (self.mode_count + 1) * (self.dimension + 1) ** 2

    def _box(self, box: Sequence[Interval]) -> tuple[Interval, ...]:
        if len(box) != self.dimension:
            raise ValueError("box dimension must match mean-map dimension")
        result = []
        for axis in box:
            if not isinstance(axis, Interval):
                raise TypeError("box entries must be Interval objects")
            result.append(Interval(_float(axis.lo), _float(axis.hi)))
        return tuple(result)

    def _logits(self, box: tuple[Interval, ...]) -> tuple[Interval, ...]:
        return tuple(_linear(row, box, offset) for row, offset in zip(self._relative_modes, self._relative_biases, strict=True))

    def _centered_mean(self, probabilities: tuple[Interval, ...]) -> tuple[Interval, ...]:
        result = []
        for axis in range(self.dimension):
            value = Interval.point(0.0)
            for probability, mode in zip(probabilities, self._relative_modes, strict=True):
                if mode[axis]:
                    value = _add(value, _multiply(probability, _rational(mode[axis])))
            result.append(value)
        return tuple(result)

    def residual(self, box: Sequence[Interval]) -> tuple[Interval, ...]:
        """Enclose mean minus target throughout a finite binary64 box."""
        frozen = self._box(box)
        with _strict():
            mean = self._centered_mean(_probabilities(self._logits(frozen)))
            return tuple(_add(value, _rational(offset)) for value, offset in zip(mean, self._reference_residual, strict=True))

    def jacobian(self, box: Sequence[Interval]) -> tuple[tuple[Interval, ...], ...]:
        """Enclose covariance using exact modes centered at a fixed reference."""
        frozen = self._box(box)
        with _strict():
            probabilities = _probabilities(self._logits(frozen))
            mean = self._centered_mean(probabilities)
            rows: list[list[Interval]] = [[Interval.point(0.0)] * self.dimension for _ in range(self.dimension)]
            for a in range(self.dimension):
                for b in range(a, self.dimension):
                    second = Interval.point(0.0)
                    for probability, mode in zip(probabilities, self._relative_modes, strict=True):
                        coefficient = self.budget._coefficient(mode[a] * mode[b])
                        if coefficient:
                            second = _add(second, _multiply(probability, _rational(coefficient)))
                    value = _finite(second - _multiply(mean[a], mean[b]))
                    if a == b:
                        value = Interval(max(0.0, value.lo), max(0.0, value.hi))
                    rows[a][b] = rows[b][a] = value
            return tuple(tuple(row) for row in rows)

    def jacobian_action(self, box: Sequence[Interval], vector: Sequence[Interval]) -> tuple[Interval, ...]:
        """Enclose covariance times a vector without forming a dense matrix.

        The covariance is evaluated by centered outer products. Storage and
        arithmetic for this action grow with modes times dimension; constructor
        metadata and shared budgets retain their conservative existing bounds.
        Both arguments use finite binary64 interval endpoints.
        """
        frozen, direction = self._box(box), self._box(vector)
        self.budget._work(64 * (self.mode_count + 1) * (self.dimension + 1))
        with _strict():
            probabilities = _probabilities(self._logits(frozen))
            mean = self._centered_mean(probabilities)
            result = [Interval.point(0.0) for _ in range(self.dimension)]
            for probability, mode in zip(probabilities, self._relative_modes, strict=True):
                centered = tuple(_finite(_rational(value) - center)
                                 for value, center in zip(mode, mean, strict=True))
                inner = Interval.point(0.0)
                for value, component in zip(centered, direction, strict=True):
                    inner = _add(inner, _multiply(value, component))
                factor = _multiply(probability, inner)
                for axis, value in enumerate(centered):
                    result[axis] = _add(result[axis], _multiply(factor, value))
            return tuple(result)

    def objective(self, box: Sequence[Interval]) -> Interval:
        """Enclose log-partition minus target dot x, including common offsets."""
        frozen = self._box(box)
        with _strict():
            logits = self._logits(frozen)
            lower = _logsum(tuple(value.lo for value in logits))
            upper = _logsum(tuple(value.hi for value in logits))
            partition = Interval(lower.lo, upper.hi)
            return _add(partition, _linear(self._reference_residual, frozen, self.biases[0]))

    def float_evaluate(self, point: Sequence[int | float]) -> MeanEvaluation:
        """Stable stdlib proposal values; all results remain untrusted.

        The gauge-reduced objective omits ``biases[0]`` for line searches.
        All returned quantities, including the original objective, must be
        finite. A common bias outside binary64 range therefore refuses this
        method even when rigorous residual evaluation remains possible.
        """
        if len(point) != self.dimension:
            raise ValueError("point dimension must match mean-map dimension")
        coordinates = tuple(_float(value) for value in point)
        try:
            modes = tuple(tuple(float(value) for value in row) for row in self._relative_modes)
            logits = tuple(math.fsum([float(offset), *(value * axis for value, axis in zip(row, coordinates, strict=True))])
                           for row, offset in zip(modes, self._relative_biases, strict=True))
            shift = max(logits)
            scaled = tuple(math.exp(value - shift) for value in logits)
            total = math.fsum(scaled)
            probabilities = tuple(value / total for value in scaled)
            mean = tuple(math.fsum(probability * mode[axis] for probability, mode in zip(probabilities, modes, strict=True))
                         for axis in range(self.dimension))
            residual = tuple(math.fsum((value, float(offset)))
                             for value, offset in zip(mean, self._reference_residual, strict=True))
            covariance = tuple(tuple(math.fsum(probability * (mode[a] - mean[a]) * (mode[b] - mean[b])
                                              for probability, mode in zip(probabilities, modes, strict=True))
                                     for b in range(self.dimension)) for a in range(self.dimension))
            reduced_objective = math.fsum([shift, math.log(total),
                                          *(float(value) * axis for value, axis in zip(self._reference_residual, coordinates, strict=True))])
            objective = math.fsum((float(self.biases[0]), reduced_objective))
            values = (*logits, *scaled, *mean, *residual, objective, reduced_objective,
                      *(value for row in covariance for value in row))
            if not all(math.isfinite(value) for value in values):
                raise ArithmeticError("nonfinite floating-point mean-map proposal")
            return MeanEvaluation(objective, residual, covariance, reduced_objective)
        except (OverflowError, ZeroDivisionError, ValueError) as exc:
            raise ArithmeticError("mean-map proposal exceeds the numerical range") from exc

    def _witness(self, weights: Sequence[Rational]) -> tuple[Vector, bool]:
        if len(weights) != self.mode_count:
            raise ValueError("witness must have one exact weight per mode")
        frozen = tuple(_exact(value, self.budget) for value in weights)
        total = Fraction(0)
        centered = [Fraction(0)] * self.dimension
        for weight, mode in zip(frozen, self._relative_modes, strict=True):
            total = self.budget._coefficient(total + weight)
            for axis in range(self.dimension):
                product = self.budget._coefficient(weight * mode[axis])
                centered[axis] = self.budget._coefficient(centered[axis] + product)
        valid = all(weight >= 0 for weight in frozen) and total == 1
        valid = valid and all(value == -offset for value, offset in zip(centered, self._reference_residual, strict=True))
        return frozen, valid

    def target_interior(self, weights: Sequence[Rational]) -> bool:
        """Check a strictly positive exact witness and full affine rank.

        False means the supplied witness does not establish ambient interior;
        it does not assert that no different witness exists.
        """
        frozen, valid = self._witness(weights)
        return valid and self.affine_rank == self.dimension and all(weight > 0 for weight in frozen)

    def dual_bound(self, weights: Sequence[Rational]) -> Interval:
        """Enclose the global lower bound q.bias minus sum(q log q).

        The exact witness must satisfy q>=0, sum(q)=1 and sum(q*mode)=target.
        Zero weights contribute zero entropy. The returned lower endpoint is
        a rigorous objective lower bound for every real coordinate vector.
        """
        frozen, valid = self._witness(weights)
        if not valid:
            raise ValueError("dual weights must be a nonnegative exact target witness summing to one")
        offset = self.biases[0]
        for weight, bias in zip(frozen, self._relative_biases, strict=True):
            product = self.budget._coefficient(weight * bias)
            offset = self.budget._coefficient(offset + product)
        with _strict():
            entropy = Interval.point(0.0)
            for weight in frozen:
                if weight:
                    exponent = weight.numerator.bit_length() - weight.denominator.bit_length()
                    mantissa = weight / (1 << exponent) if exponent >= 0 else weight * (1 << -exponent)
                    mantissa = self.budget._coefficient(mantissa)
                    logarithm = _finite(ln_iv(_rational(mantissa)))
                    if exponent:
                        logarithm = _add(logarithm, _multiply(_rational(Fraction(exponent)), ln_iv(Interval.point(2.0))))
                    term = _multiply(_rational(weight), -logarithm)
                    entropy = _positive_sum(entropy, Interval(max(0.0, term.lo), max(0.0, term.hi)))
            return _add(_rational(offset), entropy)


__all__ = ["MeanEvaluation", "MeanMap"]
