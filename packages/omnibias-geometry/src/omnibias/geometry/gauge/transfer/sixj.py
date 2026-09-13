# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Racah 6j symbols for SU(2) on integer ``two_j`` labels.

The formula is the standard Racah sum (Wikipedia / Edmonds): factorials
are exact :class:`~fractions.Fraction`; the four triangle coefficients
``Δ`` contribute square roots that go through
:meth:`~omnibias.core.verified.interval.Interval.sqrt`.

A 6j is zero when any of the four triads fails the triangle inequalities
or the integer-sum (even ``two_j`` total) rule.  Locked textbook values
live in :data:`TEXTBOOK_SIXJ` and :data:`VANISHING_SIXJ`.

This is a finite recoupling identity.  It is not a continuum gauge claim.
"""

from __future__ import annotations

import math
from fractions import Fraction

from omnibias.core.verified.interval import Interval

#: ``{1/2 1/2 0; 1/2 1/2 0} = -1/2`` and ``{1 1 1; 1 1 1} = 1/6``.
TEXTBOOK_SIXJ: tuple[tuple[tuple[int, int, int, int, int, int], Fraction], ...] = (
    ((1, 1, 0, 1, 1, 0), Fraction(-1, 2)),
    ((2, 2, 2, 2, 2, 2), Fraction(1, 6)),
    ((0, 0, 0, 0, 0, 0), Fraction(1)),
)

#: All-``1/2`` (illegal triad) and one vanishing triangle.
VANISHING_SIXJ: tuple[tuple[int, int, int, int, int, int], ...] = (
    (1, 1, 1, 1, 1, 1),
    (2, 0, 0, 2, 2, 2),
)


def _triangle(two_a: int, two_b: int, two_c: int) -> bool:
    if min(two_a, two_b, two_c) < 0:
        return False
    if (two_a + two_b + two_c) % 2 != 0:
        return False
    return abs(two_a - two_b) <= two_c <= two_a + two_b


def _delta_sq(two_a: int, two_b: int, two_c: int) -> Fraction:
    """``Δ(a,b,c)² = (a+b-c)! (a-b+c)! (-a+b+c)! / (a+b+c+1)!``."""
    return Fraction(
        math.factorial((two_a + two_b - two_c) // 2)
        * math.factorial((two_a - two_b + two_c) // 2)
        * math.factorial((-two_a + two_b + two_c) // 2),
        math.factorial((two_a + two_b + two_c) // 2 + 1),
    )


def _racah_factors(
    two_j1: int,
    two_j2: int,
    two_j3: int,
    two_j4: int,
    two_j5: int,
    two_j6: int,
) -> tuple[Fraction, Fraction]:
    """Exact squared radical and rational Racah sum, shared by both APIs."""
    labels = (two_j1, two_j2, two_j3, two_j4, two_j5, two_j6)
    if any(not isinstance(value, int) or isinstance(value, bool) for value in labels):
        raise ValueError(f"sixj labels must be integers, got {labels!r}")
    if not (
        _triangle(two_j1, two_j2, two_j3)
        and _triangle(two_j1, two_j5, two_j6)
        and _triangle(two_j4, two_j2, two_j6)
        and _triangle(two_j4, two_j5, two_j3)
    ):
        return Fraction(0), Fraction(0)
    delta2 = (
        _delta_sq(two_j1, two_j2, two_j3)
        * _delta_sq(two_j1, two_j5, two_j6)
        * _delta_sq(two_j4, two_j2, two_j6)
        * _delta_sq(two_j4, two_j5, two_j3)
    )
    t_lo = max(
        (two_j1 + two_j2 + two_j3) // 2,
        (two_j1 + two_j5 + two_j6) // 2,
        (two_j4 + two_j2 + two_j6) // 2,
        (two_j4 + two_j5 + two_j3) // 2,
    )
    t_hi = min(
        (two_j1 + two_j2 + two_j4 + two_j5) // 2,
        (two_j2 + two_j3 + two_j5 + two_j6) // 2,
        (two_j3 + two_j1 + two_j6 + two_j4) // 2,
    )
    acc = Fraction(0)
    for t in range(t_lo, t_hi + 1):
        denoms = (
            t - (two_j1 + two_j2 + two_j3) // 2,
            t - (two_j1 + two_j5 + two_j6) // 2,
            t - (two_j4 + two_j2 + two_j6) // 2,
            t - (two_j4 + two_j5 + two_j3) // 2,
            (two_j1 + two_j2 + two_j4 + two_j5) // 2 - t,
            (two_j2 + two_j3 + two_j5 + two_j6) // 2 - t,
            (two_j3 + two_j1 + two_j6 + two_j4) // 2 - t,
        )
        if any(value < 0 for value in denoms):
            continue
        den = 1
        for value in denoms:
            den *= math.factorial(value)
        acc += Fraction(((-1) ** t) * math.factorial(t + 1), den)
    return delta2, acc


def racah_sixj(
    two_j1: int,
    two_j2: int,
    two_j3: int,
    two_j4: int,
    two_j5: int,
    two_j6: int,
) -> Interval:
    """Enclosure of ``{j1 j2 j3; j4 j5 j6}`` from integer ``two_j = 2j`` labels."""
    delta2, acc = _racah_factors(two_j1, two_j2, two_j3, two_j4, two_j5, two_j6)
    if acc == 0:
        return Interval.point(0.0)
    return Interval.from_value(delta2).sqrt() * Interval.from_value(acc)


def racah_sixj_squared(
    two_j1: int,
    two_j2: int,
    two_j3: int,
    two_j4: int,
    two_j5: int,
    two_j6: int,
) -> Fraction:
    """Exact rational square of a real SU(2) Racah symbol; no float squaring."""
    delta2, acc = _racah_factors(two_j1, two_j2, two_j3, two_j4, two_j5, two_j6)
    return delta2 * acc * acc


def magnetic_sixj_amplitude(
    two_j_a: int,
    two_j_s: int,
    two_j_spectator: int,
    two_j_a_prime: int,
    two_j_s_prime: int,
) -> Interval:
    r"""Fundamental character multiplication on the two-vertex theta graph.

    ``√[(2j_a+1)(2j_a'+1)(2j_s+1)(2j_s'+1)] × {j_a j_s j_spec; j_s' j_a' 1/2}²``

    Both trivalent vertices contribute a Racah factor.  In the normalized
    Haar projector basis their phases cancel, and the amplitude is symmetric
    before any matrix symmetrization.  A single Racah factor is incorrect:
    it makes the vacuum-to-fundamental amplitude sqrt(2), instead of 1.
    For zero spectator the character recurrence gives amplitude exactly 1.
    """
    six_squared = racah_sixj_squared(
        two_j_a,
        two_j_s,
        two_j_spectator,
        two_j_s_prime,
        two_j_a_prime,
        1,
    )
    dim = (
        Interval.from_value(two_j_a + 1)
        * Interval.from_value(two_j_a_prime + 1)
        * Interval.from_value(two_j_s + 1)
        * Interval.from_value(two_j_s_prime + 1)
    )
    if six_squared == 0:
        return Interval.point(0.0)
    return dim.sqrt() * Interval.from_value(six_squared)


__all__ = [
    "TEXTBOOK_SIXJ",
    "VANISHING_SIXJ",
    "magnetic_sixj_amplitude",
    "racah_sixj",
    "racah_sixj_squared",
]
