# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Validated complex continuation for a scalar D-finite equation.

The core package cannot import the extension-tier Ore algebra, so an operator
is supplied as its serial form: ``coeffs[i]`` is the ascending rational
polynomial multiplying ``D^i``.  The order-``r`` equation is converted to its
companion system and then realified:

.. math::

   M=A+iB \longmapsto \begin{bmatrix}A&-B\\B&A\end{bmatrix}.

On each path segment, an interval matrix encloses the companion matrix at
every point.  The Peano--Baker series of the time-varying linear system is
then enclosed by :func:`interval_matrix_exp`: each ordered product of
possibly different matrices from the box is contained in the corresponding
interval-matrix power.  This avoids the floating QR assumption documented by
the generic Lohner implementation.

A path box that may meet a zero of the leading coefficient is refused.  The
result is a finite-path enclosure, not a global monodromy or zero-count
claim.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction

from omnibias.core.verified.complex_interval import ComplexInterval, ComplexLike
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.linalg import IntervalMatrix, matvec
from omnibias.core.verified.lohner import interval_matrix_exp

Rational = Fraction | int
SerializedOperator = tuple[tuple[Fraction, ...], ...]


def _fraction(value: Rational) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, int | Fraction):
        raise TypeError("D-finite operator coefficients must be exact rationals")
    return Fraction(value)


def normalize_operator(
    coeffs: Sequence[Sequence[Rational]],
) -> SerializedOperator:
    """Validate and trim a serialized differential operator over ``Q[h]``."""

    normalized: list[tuple[Fraction, ...]] = []
    for polynomial in coeffs:
        values = [_fraction(value) for value in polynomial]
        while values and values[-1] == 0:
            values.pop()
        normalized.append(tuple(values))
    while normalized and not normalized[-1]:
        normalized.pop()
    if len(normalized) < 2:
        raise ValueError("a D-finite continuation operator must have order >= 1")
    return tuple(normalized)


def evaluate_complex_polynomial(
    coeffs: Sequence[Rational], z: ComplexInterval
) -> ComplexInterval:
    """Horner enclosure of a rational polynomial over a complex box."""

    result = ComplexInterval.zero()
    for coefficient in reversed(coeffs):
        c = ComplexInterval.from_parts(Interval.from_rational(_fraction(coefficient)))
        result = result * z + c
    return result


def companion_matrix(
    coeffs: Sequence[Sequence[Rational]], z: ComplexInterval
) -> tuple[tuple[ComplexInterval, ...], ...]:
    """Complex interval companion matrix for ``sum_i c_i(z) f^(i)=0``."""

    operator = normalize_operator(coeffs)
    order = len(operator) - 1
    values = tuple(evaluate_complex_polynomial(polynomial, z) for polynomial in operator)
    lead = values[-1]
    if lead.modulus().lo <= 0.0:
        raise ZeroDivisionError(
            "continuation path may meet a zero of the leading coefficient"
        )
    rows: list[tuple[ComplexInterval, ...]] = []
    for index in range(order - 1):
        rows.append(
            tuple(
                ComplexInterval.one() if column == index + 1 else ComplexInterval.zero()
                for column in range(order)
            )
        )
    rows.append(tuple(-values[index] / lead for index in range(order)))
    return tuple(rows)


def realify_complex_matrix(
    matrix: Sequence[Sequence[ComplexInterval]],
) -> IntervalMatrix:
    """Realify ``A+iB`` to the interval block matrix ``[[A,-B],[B,A]]``."""

    size = len(matrix)
    if size == 0 or any(len(row) != size for row in matrix):
        raise ValueError("a nonempty square complex matrix is required")
    result: IntervalMatrix = []
    for row in matrix:
        result.append([value.re for value in row] + [-value.im for value in row])
    for row in matrix:
        result.append([value.im for value in row] + [value.re for value in row])
    return result


def _scale_complex_matrix(
    matrix: Sequence[Sequence[ComplexInterval]], factor: ComplexInterval
) -> tuple[tuple[ComplexInterval, ...], ...]:
    return tuple(tuple(value * factor for value in row) for row in matrix)


def _scale_real_matrix(matrix: IntervalMatrix, factor: Interval) -> IntervalMatrix:
    return [[value * factor for value in row] for row in matrix]


def _realify_state(values: Sequence[ComplexInterval]) -> list[Interval]:
    return [value.re for value in values] + [value.im for value in values]


def _complexify_state(values: Sequence[Interval]) -> tuple[ComplexInterval, ...]:
    if len(values) % 2:
        raise ValueError("a realified complex state must have even dimension")
    size = len(values) // 2
    return tuple(ComplexInterval(values[index], values[index + size]) for index in range(size))


@dataclass(frozen=True)
class DFiniteContinuation:
    """Validated finite-path enclosure of a D-finite solution jet."""

    initial_h: complex
    target: ComplexInterval
    jet: tuple[ComplexInterval, ...]
    path_boxes: tuple[ComplexInterval, ...]
    n_steps: int
    order: int

    @property
    def value(self) -> ComplexInterval:
        return self.jet[0]


def continue_dfinite(
    coeffs: Sequence[Sequence[Rational]],
    initial_h: complex | float | int,
    initial_jet: Sequence[ComplexLike],
    target: ComplexLike,
    *,
    n_steps: int = 64,
    order: int = 18,
) -> DFiniteContinuation:
    """Continue a D-finite solution from a point to every point in ``target``.

    The straight homotopy ``h(s)=initial_h+s*(target-initial_h)`` is enclosed
    for ``0 <= s <= 1``.  A rectangular target therefore represents a family
    of straight paths and the returned jet encloses all their endpoints.
    """

    operator = normalize_operator(coeffs)
    differential_order = len(operator) - 1
    if len(initial_jet) != differential_order:
        raise ValueError(
            f"operator order {differential_order} requires {differential_order} initial derivatives"
        )
    if type(n_steps) is not int or n_steps < 1:
        raise ValueError("n_steps must be a positive integer")
    if type(order) is not int or order < 4:
        raise ValueError("matrix-exponential order must be an integer >= 4")

    start = complex(initial_h)
    start_box = ComplexInterval.point(start)
    target_box = ComplexInterval.from_value(target)
    delta = target_box - start_box
    whole_path = start_box + delta * ComplexInterval.from_value(Interval(0.0, 1.0))
    # Refuse a path family that may cross a singular leading coefficient
    # before attempting any matrix exponential.
    companion_matrix(operator, whole_path)
    state = _realify_state(
        tuple(ComplexInterval.from_value(value) for value in initial_jet)
    )
    step_size = Interval.from_rational(Fraction(1, n_steps))
    path_boxes: list[ComplexInterval] = []

    for index in range(n_steps):
        # Rational injection inflates binary endpoints when needed, keeping
        # the path cover sound.
        s_box = Interval.hull(
            Interval.from_rational(Fraction(index, n_steps)),
            Interval.from_rational(Fraction(index + 1, n_steps)),
        )
        h_box = start_box + delta * ComplexInterval.from_value(s_box)
        path_boxes.append(h_box)
        complex_generator = _scale_complex_matrix(
            companion_matrix(operator, h_box),
            delta,
        )
        real_generator = _scale_real_matrix(
            realify_complex_matrix(complex_generator),
            step_size,
        )
        transition = interval_matrix_exp(real_generator, order=order)
        state = matvec(transition, state)

    return DFiniteContinuation(
        initial_h=start,
        target=target_box,
        jet=_complexify_state(state),
        path_boxes=tuple(path_boxes),
        n_steps=n_steps,
        order=order,
    )


def continue_dfinite_path(
    coeffs: Sequence[Sequence[Rational]],
    initial_h: complex | float | int,
    initial_jet: Sequence[ComplexLike],
    waypoints: Sequence[complex | float | int],
    *,
    steps_per_segment: int = 32,
    order: int = 18,
) -> tuple[DFiniteContinuation, ...]:
    """Continue successively through point waypoints, retaining every enclosure."""

    current_h = complex(initial_h)
    current_jet = tuple(ComplexInterval.from_value(value) for value in initial_jet)
    results: list[DFiniteContinuation] = []
    for waypoint in waypoints:
        result = continue_dfinite(
            coeffs,
            current_h,
            current_jet,
            complex(waypoint),
            n_steps=steps_per_segment,
            order=order,
        )
        results.append(result)
        current_h = complex(waypoint)
        current_jet = result.jet
    return tuple(results)


__all__ = [
    "DFiniteContinuation",
    "SerializedOperator",
    "companion_matrix",
    "continue_dfinite",
    "continue_dfinite_path",
    "evaluate_complex_polynomial",
    "normalize_operator",
    "realify_complex_matrix",
]
