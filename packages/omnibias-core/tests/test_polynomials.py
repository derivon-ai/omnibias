# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Pure-Python regression tests for the polynomial coefficient generators.

These run with no torch / jax / numpy dependency; they only validate
the mathematical contracts of the recurrences. The cross-backend ULP
parity tests live in `tests/test_cross_backend_parity.py` (workspace
level), which depends on torch and jax.
"""

from __future__ import annotations

import pytest
from omnibias.core.polynomials import (
    hermite_coeffs,
    sigmoid_polynomial_coeffs,
    tanh_polynomial_coeffs,
)

# ----- sigmoid (Eulerian) -----


def test_sigmoid_p0() -> None:
    # P_0(s) = s
    assert sigmoid_polynomial_coeffs(0) == (0.0, 1.0)


def test_sigmoid_p1() -> None:
    # P_1(s) = s - s^2
    assert sigmoid_polynomial_coeffs(1) == (0.0, 1.0, -1.0)


def test_sigmoid_p2() -> None:
    # P_2(s) = s - 3 s^2 + 2 s^3 (Eulerian numbers row 2 with sign)
    assert sigmoid_polynomial_coeffs(2) == (0.0, 1.0, -3.0, 2.0)


def test_sigmoid_p3() -> None:
    # P_3(s) = s - 7 s^2 + 12 s^3 - 6 s^4
    assert sigmoid_polynomial_coeffs(3) == (0.0, 1.0, -7.0, 12.0, -6.0)


def test_sigmoid_negative_order_raises() -> None:
    with pytest.raises(ValueError, match="order n must be"):
        sigmoid_polynomial_coeffs(-1)


# ----- tanh (Legendre-style) -----


def test_tanh_t0() -> None:
    # T_0(t) = t
    assert tanh_polynomial_coeffs(0) == (0.0, 1.0)


def test_tanh_t1() -> None:
    # T_1(t) = 1 - t^2
    assert tanh_polynomial_coeffs(1) == (1.0, 0.0, -1.0)


def test_tanh_t2() -> None:
    # T_2(t) = -2 t + 2 t^3
    assert tanh_polynomial_coeffs(2) == (0.0, -2.0, 0.0, 2.0)


def test_tanh_t3() -> None:
    # T_3(t) = -2 + 8 t^2 - 6 t^4
    assert tanh_polynomial_coeffs(3) == (-2.0, 0.0, 8.0, 0.0, -6.0)


def test_tanh_negative_order_raises() -> None:
    with pytest.raises(ValueError, match="order n must be"):
        tanh_polynomial_coeffs(-1)


# ----- Hermite (probabilist's) -----


def test_hermite_he0() -> None:
    # He_0(z) = 1
    assert hermite_coeffs(0) == (1.0,)


def test_hermite_he1() -> None:
    # He_1(z) = z
    assert hermite_coeffs(1) == (0.0, 1.0)


def test_hermite_he2() -> None:
    # He_2(z) = z^2 - 1
    assert hermite_coeffs(2) == (-1.0, 0.0, 1.0)


def test_hermite_he3() -> None:
    # He_3(z) = z^3 - 3 z
    assert hermite_coeffs(3) == (0.0, -3.0, 0.0, 1.0)


def test_hermite_he4() -> None:
    # He_4(z) = z^4 - 6 z^2 + 3
    assert hermite_coeffs(4) == (3.0, 0.0, -6.0, 0.0, 1.0)


def test_hermite_negative_order_raises() -> None:
    with pytest.raises(ValueError, match="order n must be"):
        hermite_coeffs(-1)


# ----- recurrence sanity -----


def test_sigmoid_recurrence_consistency() -> None:
    """``P_{n+1}(s) = s (1 - s) P_n'(s)`` derivation up to n = 5."""
    for n in range(5):
        prev = sigmoid_polynomial_coeffs(n)
        curr = sigmoid_polynomial_coeffs(n + 1)
        # Compute expected: s(1-s) * P_n'(s), then compare.
        deriv = tuple(k * prev[k] for k in range(1, len(prev)))
        expected = [0.0] * (len(deriv) + 2)
        for i, c in enumerate(deriv):
            expected[i + 1] += c
            expected[i + 2] -= c
        assert curr == tuple(expected)


def test_tanh_recurrence_consistency() -> None:
    """``T_{n+1}(t) = (1 - t^2) T_n'(t)`` derivation up to n = 5."""
    for n in range(5):
        prev = tanh_polynomial_coeffs(n)
        curr = tanh_polynomial_coeffs(n + 1)
        deriv = tuple(k * prev[k] for k in range(1, len(prev)))
        expected = [0.0] * (len(deriv) + 2)
        for i, c in enumerate(deriv):
            expected[i] += c
            expected[i + 2] -= c
        assert curr == tuple(expected)


def test_hermite_recurrence_consistency() -> None:
    """``He_{n+1}(z) = z He_n(z) - n He_{n-1}(z)`` up to n = 6."""
    for n in range(1, 6):
        prev1 = hermite_coeffs(n)
        prev2 = hermite_coeffs(n - 1)
        curr = hermite_coeffs(n + 1)
        expected = [0.0] * (n + 2)
        for k, c in enumerate(prev1):
            expected[k + 1] += c
        for k, c in enumerate(prev2):
            expected[k] -= n * c
        assert curr == tuple(expected)
