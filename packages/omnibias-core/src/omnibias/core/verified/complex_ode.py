# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Validated real-time integration of complexified analytic ODEs.

This module is the rectangular-complex counterpart of
:mod:`omnibias.core.verified.ode`.  It encloses

.. math::

   Y'(t) = F(Y(t)), \qquad t \in [t_0,t_1] \subset \mathbb R,

uniformly for every complex initial value in a supplied rectangle.  Taylor
coefficients, the Picard a-priori tube, and the Lagrange remainder all use
outward-rounded :class:`~omnibias.core.verified.complex_interval.ComplexInterval`
arithmetic.

The result is a fixed-real-time complex flow enclosure.  It does **not** by
itself certify a parameter-dependent first-hit time, a holomorphic return map,
or Log-Noetherian differential-polynomial closure.  Those require a complex
implicit-event argument and separate format bounds.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from fractions import Fraction

from omnibias.core.verified.complex_interval import ComplexInterval, ComplexLike
from omnibias.core.verified.interval import Interval

ComplexVectorField = Callable[
    [list["ComplexTaylorSeries"]],
    list["ComplexTaylorSeries"],
]


class ComplexTaylorSeries:
    """A truncated power series with rectangular complex coefficients."""

    __slots__ = ("coeffs",)

    def __init__(self, coeffs: Sequence[ComplexInterval]) -> None:
        if not coeffs:
            raise ValueError("ComplexTaylorSeries needs at least one coefficient")
        self.coeffs: tuple[ComplexInterval, ...] = tuple(coeffs)

    @classmethod
    def constant(
        cls,
        value: ComplexLike | Fraction,
        order: int,
    ) -> ComplexTaylorSeries:
        """Series ``[value, 0, ..., 0]`` of length ``order + 1``."""
        if type(order) is not int or order < 0:
            raise ValueError("order must be a nonnegative integer")
        c0 = ComplexInterval.from_value(value)
        zero = ComplexInterval.zero()
        return cls([c0, *([zero] * order)])

    @property
    def order(self) -> int:
        return len(self.coeffs) - 1

    def __len__(self) -> int:
        return len(self.coeffs)

    def __add__(
        self,
        other: ComplexTaylorSeries | ComplexLike | Fraction,
    ) -> ComplexTaylorSeries:
        if isinstance(other, ComplexTaylorSeries):
            n = min(len(self.coeffs), len(other.coeffs))
            return ComplexTaylorSeries(
                [self.coeffs[index] + other.coeffs[index] for index in range(n)]
            )
        shifted = list(self.coeffs)
        shifted[0] = shifted[0] + ComplexInterval.from_value(other)
        return ComplexTaylorSeries(shifted)

    __radd__ = __add__

    def __neg__(self) -> ComplexTaylorSeries:
        return ComplexTaylorSeries([-coefficient for coefficient in self.coeffs])

    def __sub__(
        self,
        other: ComplexTaylorSeries | ComplexLike | Fraction,
    ) -> ComplexTaylorSeries:
        if isinstance(other, ComplexTaylorSeries):
            n = min(len(self.coeffs), len(other.coeffs))
            return ComplexTaylorSeries(
                [self.coeffs[index] - other.coeffs[index] for index in range(n)]
            )
        shifted = list(self.coeffs)
        shifted[0] = shifted[0] - ComplexInterval.from_value(other)
        return ComplexTaylorSeries(shifted)

    def __rsub__(
        self,
        other: ComplexLike | Fraction,
    ) -> ComplexTaylorSeries:
        return (-self).__add__(other)

    def __mul__(
        self,
        other: ComplexTaylorSeries | ComplexLike | Fraction,
    ) -> ComplexTaylorSeries:
        if isinstance(other, ComplexTaylorSeries):
            n = min(len(self.coeffs), len(other.coeffs))
            out: list[ComplexInterval] = []
            for degree in range(n):
                coefficient = ComplexInterval.zero()
                for left_degree in range(degree + 1):
                    coefficient = (
                        coefficient
                        + self.coeffs[left_degree]
                        * other.coeffs[degree - left_degree]
                    )
                out.append(coefficient)
            return ComplexTaylorSeries(out)
        scalar = ComplexInterval.from_value(other)
        return ComplexTaylorSeries(
            [coefficient * scalar for coefficient in self.coeffs]
        )

    __rmul__ = __mul__

    def __repr__(self) -> str:
        return f"ComplexTaylorSeries({list(self.coeffs)!r})"


def _solution_coeffs(
    field: ComplexVectorField,
    y0: Sequence[ComplexInterval],
    upto: int,
) -> list[list[ComplexInterval]]:
    """Taylor coefficients of the complexified flow through ``upto``."""
    coefficients: list[list[ComplexInterval]] = [[value] for value in y0]
    for degree in range(upto):
        length = degree + 1
        series = [
            ComplexTaylorSeries(
                [coefficients[axis][index] for index in range(length)]
            )
            for axis in range(len(y0))
        ]
        field_series = field(series)
        if len(field_series) != len(y0):
            raise ValueError("complex vector field returned the wrong dimension")
        inverse = Fraction(1, degree + 1)
        for axis, component in enumerate(field_series):
            coefficients[axis].append(
                component.coeffs[degree] * inverse
            )
    return coefficients


def _field_at(
    field: ComplexVectorField,
    box: Sequence[ComplexInterval],
) -> list[ComplexInterval]:
    series = [ComplexTaylorSeries([value]) for value in box]
    result = field(series)
    if len(result) != len(box):
        raise ValueError("complex vector field returned the wrong dimension")
    return [component.coeffs[0] for component in result]


def _inflate_part(interval: Interval, atol: float) -> Interval:
    radius = interval.rad + atol
    return Interval(interval.lo - radius, interval.hi + radius)


def _inflate_rectangle(value: ComplexInterval, atol: float) -> ComplexInterval:
    return ComplexInterval(
        _inflate_part(value.re, atol),
        _inflate_part(value.im, atol),
    )


def _contains_rectangle(
    outer: ComplexInterval,
    inner: ComplexInterval,
) -> bool:
    return bool(
        outer.re.lo <= inner.re.lo
        and inner.re.hi <= outer.re.hi
        and outer.im.lo <= inner.im.lo
        and inner.im.hi <= outer.im.hi
    )


def _apriori_enclosure(
    field: ComplexVectorField,
    y0: Sequence[ComplexInterval],
    step_size: float,
    *,
    max_iter: int = 60,
    atol: float = 1e-30,
) -> list[ComplexInterval]:
    """Complex rectangle containing every trajectory over one real-time step."""
    step = ComplexInterval.from_parts(Interval(0.0, step_size))
    initial_field = _field_at(field, y0)
    box = [
        y0[axis] + step * initial_field[axis]
        for axis in range(len(y0))
    ]
    for _ in range(max_iter):
        inflated = [_inflate_rectangle(value, atol) for value in box]
        field_box = _field_at(field, inflated)
        candidate = [
            y0[axis] + step * field_box[axis]
            for axis in range(len(y0))
        ]
        if all(
            _contains_rectangle(inflated[axis], candidate[axis])
            for axis in range(len(y0))
        ):
            return inflated
        box = candidate
    raise RuntimeError(
        "complex a-priori enclosure did not converge; reduce the step size"
    )


def _step(
    field: ComplexVectorField,
    y0: Sequence[ComplexInterval],
    step_size: float,
    order: int,
) -> list[ComplexInterval]:
    coefficients = _solution_coeffs(field, y0, order)
    tube = _apriori_enclosure(field, y0, step_size)
    remainder = _solution_coeffs(field, tube, order + 1)
    step = ComplexInterval.point(step_size)
    powers = [ComplexInterval.one()]
    for _ in range(order + 1):
        powers.append(powers[-1] * step)
    out: list[ComplexInterval] = []
    for axis in range(len(y0)):
        value = ComplexInterval.zero()
        for degree in range(order + 1):
            value = value + coefficients[axis][degree] * powers[degree]
        value = value + remainder[axis][order + 1] * powers[order + 1]
        out.append(value)
    return out


def integrate_complex_ivp(
    field: ComplexVectorField,
    y0: Sequence[ComplexLike | Fraction],
    t0: float,
    t1: float,
    *,
    order: int = 14,
    n_steps: int = 8,
) -> list[ComplexInterval]:
    """Enclose a complexified ODE flow for every supplied initial value.

    Time remains real.  Complex rectangles in ``y0`` may therefore encode a
    complex neighborhood of state or frozen parameter coordinates.
    """
    if t1 < t0:
        raise ValueError("integrate_complex_ivp requires t1 >= t0")
    if type(n_steps) is not int or n_steps < 1:
        raise ValueError("n_steps must be a positive integer")
    if type(order) is not int or order < 1:
        raise ValueError("order must be a positive integer")
    state = [ComplexInterval.from_value(value) for value in y0]
    if t1 == t0:
        return state
    step_size = (t1 - t0) / n_steps
    for _ in range(n_steps):
        state = _step(field, state, step_size, order)
    return state


__all__ = [
    "ComplexTaylorSeries",
    "ComplexVectorField",
    "integrate_complex_ivp",
]
