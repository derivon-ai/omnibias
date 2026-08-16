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


def racah_sixj(
    two_j1: int,
    two_j2: int,
    two_j3: int,
    two_j4: int,
    two_j5: int,
    two_j6: int,
) -> Interval:
    """Enclosure of ``{j1 j2 j3; j4 j5 j6}`` from integer ``two_j = 2j`` labels."""
    labels = (two_j1, two_j2, two_j3, two_j4, two_j5, two_j6)
    if any(not isinstance(value, int) or isinstance(value, bool) for value in labels):
        raise ValueError(f"sixj labels must be integers, got {labels!r}")
    if not (
        _triangle(two_j1, two_j2, two_j3)
        and _triangle(two_j1, two_j5, two_j6)
        and _triangle(two_j4, two_j2, two_j6)
        and _triangle(two_j4, two_j5, two_j3)
    ):
        return Interval.point(0.0)
    delta2 = (
        _delta_sq(two_j1, two_j2, two_j3)
        * _delta_sq(two_j1, two_j5, two_j6)
        * _delta_sq(two_j4, two_j2, two_j6)
        * _delta_sq(two_j4, two_j5, two_j3)
    )
    prefactor = Interval.from_value(delta2).sqrt()
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
    return prefactor * Interval.from_value(acc)


def magnetic_sixj_amplitude(
    two_j_a: int,
    two_j_s: int,
    two_j_spectator: int,
    two_j_a_prime: int,
    two_j_s_prime: int,
) -> Interval:
    r"""Locked two-plaquette magnetic recoupling.

    ``phase × √[(2j_a+1)(2j_a'+1)(2j_s+1)(2j_s'+1)] × {j_a j_s j_spec; j_s' j_a' 1/2}``

    with ``phase = (-1)^{j_a + j_spec + j_s' + 1/2}``.  ``2j+1 = two_j + 1``.
    The Hamiltonian symmetrises directed amplitudes so the matrix stays
    Hermitian.
    """
    six = racah_sixj(
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
    phase = Interval.from_value((-1) ** ((two_j_a + two_j_spectator + two_j_s_prime + 1) // 2))
    return phase * dim.sqrt() * six


__all__ = [
    "TEXTBOOK_SIXJ",
    "VANISHING_SIXJ",
    "magnetic_sixj_amplitude",
    "racah_sixj",
]
