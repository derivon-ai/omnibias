# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Exact finite Fourier plans for heat evolution of exp(trigonometric data).

The plan compiles the Taylor polynomial of the exponential with no frequency
truncation. Approximation of the original exponential is the responsibility of
the verified evaluator and its remainder enclosure; exact coefficient replay
alone is not a certificate for the omitted infinite series.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction

from omnibias.core.exp_polynomial import ExponentialBudget

Rational = int | Fraction
Frequency = tuple[int, ...]
FourierTerm = tuple[Frequency, Fraction, Fraction]
_Complex = tuple[Fraction, Fraction]
_Table = dict[Frequency, _Complex]


def _integer(value: int, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer, excluding bool")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


class _Arithmetic:
    """One cumulative work meter for the complete public construction."""

    def __init__(self, budget: ExponentialBudget | None) -> None:
        self.budget = ExponentialBudget() if budget is None else budget
        if not isinstance(self.budget, ExponentialBudget):
            raise TypeError("budget must be an ExponentialBudget")
        self.used = 0

    def charge(self, count: int) -> None:
        self.used += count
        self.budget._work(self.used)

    def coefficient(self, value: Fraction) -> Fraction:
        return self.budget._coefficient(value)

    def exact(self, value: Rational) -> Fraction:
        if isinstance(value, bool) or not isinstance(value, (int, Fraction)):
            raise TypeError("periodic heat metadata must be exact int or Fraction, excluding bool")
        return self.coefficient(Fraction(value))

    def frequency(self, values: Frequency, dimension: int) -> Frequency:
        if not isinstance(values, tuple) or len(values) != dimension:
            raise ValueError("Fourier frequencies must be integer tuples matching dimension")
        self.charge(dimension + 1)
        result = []
        for value in values:
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError("Fourier frequency entries must be integers, excluding bool")
            self.coefficient(Fraction(value))
            result.append(value)
        return tuple(result)

    def accumulate(self, table: _Table, key: Frequency, value: _Complex) -> None:
        previous = table.get(key, (Fraction(0), Fraction(0)))
        real = self.coefficient(previous[0] + value[0])
        imaginary = self.coefficient(previous[1] + value[1])
        if not real and not imaginary:
            table.pop(key, None)
        else:
            if key not in table and len(table) >= self.budget.max_terms:
                raise ValueError("periodic Fourier support exceeds the resource budget")
            table[key] = (real, imaginary)

    def multiply(self, left: _Table, right: _Table, dimension: int) -> _Table:
        self.charge(len(left) * len(right) * (dimension + 18))
        result: _Table = {}
        for frequency, (real, imaginary) in left.items():
            for other_frequency, (other_real, other_imaginary) in right.items():
                key = tuple(int(self.coefficient(Fraction(a + b))) for a, b in zip(frequency, other_frequency, strict=True))
                product_real = self.coefficient(self.coefficient(real * other_real) - self.coefficient(imaginary * other_imaginary))
                product_imaginary = self.coefficient(self.coefficient(real * other_imaginary) + self.coefficient(imaginary * other_real))
                self.accumulate(result, key, (product_real, product_imaginary))
        return result

    def scale(self, table: _Table, factor: Fraction) -> _Table:
        self.charge(3 * len(table) + 1)
        factor = self.coefficient(factor)
        return {key: (self.coefficient(real * factor), self.coefficient(imaginary * factor))
                for key, (real, imaginary) in table.items()}

    def add_into(self, target: _Table, source: _Table, dimension: int) -> None:
        self.charge(len(source) * (dimension + 4))
        for key, value in source.items():
            self.accumulate(target, key, value)

    def freeze(self, table: _Table, dimension: int) -> tuple[FourierTerm, ...]:
        self.charge(len(table) * max(1, len(table).bit_length()) * (dimension + 2))
        return tuple((key, real, imaginary) for key, (real, imaginary) in sorted(table.items()))


def _header(dimension: int, degree: int, viscosity: Rational, arithmetic: _Arithmetic) -> tuple[int, int, Fraction]:
    dimension = _integer(dimension, "dimension", 1)
    degree = _integer(degree, "Taylor degree")
    if dimension > arithmetic.budget.max_dimension:
        raise ValueError("periodic dimension exceeds the resource budget")
    if degree > arithmetic.budget.max_total_degree:
        raise ValueError("Taylor degree exceeds max_total_degree in the resource budget")
    arithmetic.charge(dimension + degree + 1)
    nu = arithmetic.exact(viscosity)
    if nu <= 0:
        raise ValueError("viscosity must be positive")
    return dimension, degree, nu


def _mapping(values: Mapping[Frequency, Rational] | None, name: str) -> Mapping[Frequency, Rational]:
    if values is None:
        if name == "sine":
            return {}
        raise TypeError(f"{name} coefficients must be a mapping")
    if not isinstance(values, Mapping):
        raise TypeError(f"{name} coefficients must be a mapping")
    return values


def _potential(
    dimension: int, cosine: Mapping[Frequency, Rational], sine: Mapping[Frequency, Rational] | None,
    arithmetic: _Arithmetic,
) -> _Table:
    result: _Table = {}
    for imaginary, terms in ((False, _mapping(cosine, "cosine")), (True, _mapping(sine, "sine"))):
        for raw_frequency, raw_coefficient in terms.items():
            arithmetic.charge(2 * dimension + 16)
            frequency = arithmetic.frequency(raw_frequency, dimension)
            coefficient = arithmetic.exact(raw_coefficient)
            # g's constant is a positive scaling of exp(g), irrelevant to the
            # logarithmic derivative. sin(0) is identically zero as well.
            if not coefficient or not any(frequency):
                continue
            half = arithmetic.coefficient(coefficient / 2)
            reverse = tuple(-value for value in frequency)
            if imaginary:
                arithmetic.accumulate(result, frequency, (Fraction(0), -half))
                arithmetic.accumulate(result, reverse, (Fraction(0), half))
            else:
                arithmetic.accumulate(result, frequency, (half, Fraction(0)))
                arithmetic.accumulate(result, reverse, (half, Fraction(0)))
    return result


@dataclass(frozen=True, init=False)
class PeriodicHeatPlan:
    r"""Exact Fourier compilation of Q_M(g) for a real trigonometric g.

    Use :meth:`from_log_potential` or the one-dimensional :meth:`from_velocity`.
    Coordinates have period 2*pi. Every stored coefficient triple is
    ``(integer_frequency, real_fraction, imaginary_fraction)`` and represents
    ``(real + i*imaginary) * exp(i*frequency.x)``. Canonical tuples are sorted,
    omit zeros, and obey conjugate symmetry. Inputs are copied; floats and
    booleans are refused. The constant Fourier mode of g is removed as a
    velocity-preserving gauge before Q_M(g) is compiled.

    ``heat_coefficients`` represent the initial polynomial
    ``Q_M(g)=sum(g**j/j!, j=0..M)``. Pure heat evolution multiplies mode k by
    ``exp(-viscosity*|k|**2*t)``. The Burgers evaluator additionally applies
    the Galilean coordinate ``x-drift*t``. The intended velocity is
    ``drift - 2*viscosity*grad(log(heat(exp(g))))``. The finite coefficients
    require a rigorous omitted-series bound before claiming this solution.

    ``norm_bound=sum(abs(real)+abs(imaginary))`` is an unweighted Fourier-l1
    upper bound and bounds abs(g); ``max_frequency=max(sum(abs(k)))`` is the original
    potential's largest L1 frequency. There is no harmonic cutoff: every
    convolution frequency is retained, or construction refuses explicitly.
    ``max_terms`` bounds every intermediate support, ``max_total_degree``
    bounds Taylor degree, and work is cumulative across the complete factory
    call. ``max_derivative_order`` is reserved for verified evaluation.
    """

    dimension: int
    potential_coefficients: tuple[FourierTerm, ...]
    heat_coefficients: tuple[FourierTerm, ...]
    viscosity: Fraction
    degree: int
    drift: tuple[Fraction, ...]
    budget: ExponentialBudget
    norm_bound: Fraction
    max_frequency: int

    def __init__(self) -> None:
        raise TypeError("construct a PeriodicHeatPlan with from_log_potential or from_velocity")

    @classmethod
    def _build(
        cls, dimension: int, cosine: Mapping[Frequency, Rational], sine: Mapping[Frequency, Rational] | None,
        viscosity: Fraction, degree: int, drift: Sequence[Rational] | None, arithmetic: _Arithmetic,
    ) -> PeriodicHeatPlan:
        if drift is not None and (not isinstance(drift, Sequence) or len(drift) != dimension):
            raise ValueError("drift must be a sequence matching dimension")
        arithmetic.charge(dimension)
        offsets = (Fraction(0),) * dimension if drift is None else tuple(arithmetic.exact(value) for value in drift)
        potential = _potential(dimension, cosine, sine, arithmetic)
        potential_coefficients = arithmetic.freeze(potential, dimension)
        arithmetic.charge(len(potential_coefficients) * (dimension + 2))
        potential = {frequency: (real, imaginary) for frequency, real, imaginary in potential_coefficients}
        arithmetic.charge(len(potential) * (dimension + 4))
        norm = Fraction(0)
        frequency_bound = 0
        for frequency, (real, imaginary) in potential.items():
            norm = arithmetic.coefficient(arithmetic.coefficient(norm + abs(real)) + abs(imaginary))
            frequency_size = 0
            for value in frequency:
                frequency_size = int(arithmetic.coefficient(Fraction(frequency_size + abs(value))))
            frequency_bound = max(frequency_bound, frequency_size)
        zero = (0,) * dimension
        power: _Table = {zero: (Fraction(1), Fraction(0))}
        polynomial = dict(power)
        for order in range(1, degree + 1):
            if not potential:
                break
            power = arithmetic.scale(arithmetic.multiply(power, potential, dimension), Fraction(1, order))
            arithmetic.add_into(polynomial, power, dimension)
        heat_coefficients = arithmetic.freeze(polynomial, dimension)
        result = object.__new__(cls)
        for name, stored_value in (("dimension", dimension), ("potential_coefficients", potential_coefficients),
                                   ("heat_coefficients", heat_coefficients), ("viscosity", viscosity), ("degree", degree),
                                   ("drift", offsets), ("budget", arithmetic.budget), ("norm_bound", norm),
                                   ("max_frequency", frequency_bound)):
            object.__setattr__(result, name, stored_value)
        return result

    @classmethod
    def from_log_potential(
        cls, dimension: int, cosine: Mapping[Frequency, Rational], sine: Mapping[Frequency, Rational] | None = None,
        *, viscosity: Rational = 1, degree: int = 16, drift: Sequence[Rational] | None = None,
        budget: ExponentialBudget | None = None,
    ) -> PeriodicHeatPlan:
        """Compile an exact real Fourier log potential, dropping its constant."""
        arithmetic = _Arithmetic(budget)
        dimension, degree, nu = _header(dimension, degree, viscosity, arithmetic)
        return cls._build(dimension, cosine, sine, nu, degree, drift, arithmetic)

    @classmethod
    def from_velocity(
        cls, cosine: Mapping[int, Rational], sine: Mapping[int, Rational] | None = None,
        *, mean: Rational = 0, viscosity: Rational = 1, degree: int = 16,
        budget: ExponentialBudget | None = None,
    ) -> PeriodicHeatPlan:
        r"""Compile one-dimensional u0=mean+sum(a_n cos(nx)+b_n sin(nx)).

        Keys n must be positive integers. Integrating
        ``g'=-(u0-mean)/(2*viscosity)`` gives cosine coefficients
        ``b_n/(2*viscosity*n)`` and sine coefficients
        ``-a_n/(2*viscosity*n)``. Integration, convolution and compilation
        share one work budget. The initial-velocity approximation error is
        handled by the verified omitted-series bound, not this constructor.
        """
        arithmetic = _Arithmetic(budget)
        dimension, degree, nu = _header(1, degree, viscosity, arithmetic)
        drift = arithmetic.exact(mean)
        potential_cosine: dict[Frequency, Rational] = {}
        potential_sine: dict[Frequency, Rational] = {}
        for is_cosine, source in ((True, cosine), (False, {} if sine is None else sine)):
            if not isinstance(source, Mapping):
                raise TypeError("velocity coefficients must be mappings")
            for raw_frequency, raw_coefficient in source.items():
                arithmetic.charge(12)
                frequency = _integer(raw_frequency, "velocity frequency", 1)
                arithmetic.coefficient(Fraction(frequency))
                coefficient = arithmetic.exact(raw_coefficient)
                denominator = arithmetic.coefficient(arithmetic.coefficient(2 * nu) * frequency)
                integrated = arithmetic.coefficient(coefficient / denominator)
                if not integrated:
                    continue
                destination = potential_sine if is_cosine else potential_cosine
                key = (frequency,)
                if key not in destination and len(potential_cosine) + len(potential_sine) >= arithmetic.budget.max_terms:
                    raise ValueError("periodic velocity support exceeds the resource budget")
                destination[key] = -integrated if is_cosine else integrated
        return cls._build(dimension, potential_cosine, potential_sine, nu, degree, (drift,), arithmetic)


__all__ = ["FourierTerm", "PeriodicHeatPlan"]
