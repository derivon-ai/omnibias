# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Boolean differential calculus: derivative, set-derivative, and integral."""

from __future__ import annotations

import random

import pytest
from omnibias.boolean._core.derivative import (
    boolean_derivative,
    boolean_derivative_reduced,
    boolean_derivative_set,
    boolean_integral,
    is_independent_of,
    restrict,
)
from omnibias.boolean._core.truth_table import truth_table_from_callable

AND = truth_table_from_callable(lambda a, b: a & b, 2)
XOR = truth_table_from_callable(lambda a, b: a ^ b, 2)


def test_derivative_of_and_is_other_input() -> None:
    # d(x0 & x1)/dx0 = x1.
    assert boolean_derivative_reduced(AND, 0) == (0, 1)
    assert boolean_derivative_reduced(AND, 1) == (0, 1)


def test_derivative_of_xor_is_constant_one() -> None:
    # d(x0 ^ x1)/dx0 = 1 (flipping x0 always flips the output).
    assert boolean_derivative_reduced(XOR, 0) == (1, 1)
    assert not is_independent_of(XOR, 0)


def test_derivative_is_constant_in_its_variable() -> None:
    d = boolean_derivative(AND, 0)
    assert is_independent_of(d, 0)


def test_set_derivative_order_independent() -> None:
    rng = random.Random(3)
    for _ in range(20):
        tt = tuple(rng.randint(0, 1) for _ in range(1 << 3))
        d01 = boolean_derivative(boolean_derivative(tt, 0), 1)
        assert boolean_derivative_set(tt, [0, 1]) == d01
        assert boolean_derivative_set(tt, [1, 0]) == d01


def test_integral_recovers_derivative_for_every_constant() -> None:
    rng = random.Random(4)
    for n in (1, 2, 3):
        for _ in range(20):
            tt = tuple(rng.randint(0, 1) for _ in range(1 << n))
            i = rng.randrange(n)
            g = boolean_derivative(tt, i)
            anti = boolean_integral(g, i)
            # Every choice of the free constant yields an antiderivative of g.
            length = 1 << (n - 1)
            for c_int in range(1 << length):
                c = tuple((c_int >> k) & 1 for k in range(length))
                f = anti.general(c)
                assert boolean_derivative(f, i) == g
            # The original function is the antiderivative for c = f|_{x_i=0}.
            assert anti.general(restrict(tt, i, 0)) == tt


def test_integral_requires_exactness() -> None:
    # A function that depends on x0 has no antiderivative w.r.t. x0.
    with pytest.raises(ValueError):
        boolean_integral(AND, 0)
