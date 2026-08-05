# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""q-exponentials and q-deformed Bernoulli / Euler numbers, incl. q -> 1 reductions."""

from __future__ import annotations

import math
from fractions import Fraction

import pytest
from omnibias.qcalculus import q_bernoulli, q_euler, q_exp, q_exp_big

# Classical references.
_BERNOULLI = {
    0: Fraction(1),
    1: Fraction(-1, 2),
    2: Fraction(1, 6),
    3: Fraction(0),
    4: Fraction(-1, 30),
    5: Fraction(0),
    6: Fraction(1, 42),
    8: Fraction(-1, 30),
}
_EULER = {0: Fraction(1), 2: Fraction(-1), 4: Fraction(5), 6: Fraction(-61), 8: Fraction(1385)}


@pytest.mark.parametrize("n", sorted(_BERNOULLI))
def test_q_bernoulli_reduces_to_classical(n: int) -> None:
    assert q_bernoulli(n, 1) == _BERNOULLI[n]


@pytest.mark.parametrize("n", sorted(_EULER))
def test_q_euler_reduces_to_classical(n: int) -> None:
    assert q_euler(n, 1) == _EULER[n]


def test_q_bernoulli_matches_core_exact() -> None:
    # The q -> 1 q-Bernoulli must equal omnibias.core's exact Bernoulli numbers.
    from omnibias.core.verified.coeffs import bernoulli_number_exact

    for n in range(9):
        assert q_bernoulli(n, 1) == bernoulli_number_exact(n)


def test_q_euler_matches_core_exact() -> None:
    from omnibias.core.verified.coeffs import euler_number_exact

    for n in range(0, 9, 2):
        assert q_euler(n, 1) == euler_number_exact(n)


def test_q_bernoulli_is_rational_at_generic_q() -> None:
    # Away from q = 1 the numbers are still exact rationals (closed-form).
    q = Fraction(1, 2)
    assert q_bernoulli(1, q) == -1 / (1 + q)  # -1/[2]_q
    assert isinstance(q_bernoulli(4, q), Fraction)


def test_q_euler_odd_is_zero() -> None:
    for n in (1, 3, 5, 7):
        assert q_euler(n, Fraction(1, 3)) == 0


def test_q_exponential_reciprocal_identity() -> None:
    # e_q(z) E_q(-z) = 1.
    for q in (0.3, 0.5, 0.7):
        for z in (0.1, 0.25, -0.2):
            assert q_exp(z, q) * q_exp_big(-z, q) == pytest.approx(1.0, rel=1e-9, abs=1e-9)


def test_q_exp_approaches_exp_as_q_to_one() -> None:
    z = 0.4
    errs = [abs(q_exp(z, q) - math.exp(z)) for q in (0.9, 0.99, 0.999)]
    assert errs == sorted(errs, reverse=True)
    assert errs[-1] < 1e-2


def test_errors() -> None:
    with pytest.raises(ValueError):
        q_exp(0.1, 1.5)
    with pytest.raises(ValueError):
        q_exp_big(0.1, 0.0)
    with pytest.raises(ValueError):
        q_bernoulli(-1, Fraction(1, 2))
    with pytest.raises(ValueError):
        q_euler(-1, Fraction(1, 2))
