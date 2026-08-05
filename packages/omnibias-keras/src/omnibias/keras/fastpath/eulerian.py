# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Closed-form derivatives of sigmoid (and softplus) via the Eulerian /
sigmoid-polynomial recursion, evaluated with ``keras.ops``.

Let ``s = sigmoid(z)``. Then ``s' = s (1 - s)`` (a Riccati identity) and
every higher derivative is a polynomial in ``s``:

    sigma^(n)(z) = P_n(s),   P_0(s) = s,
                             P_{n+1}(s) = s (1 - s) * P_n'(s).

The coefficients come from the canonical, backend-agnostic
:mod:`omnibias.core.polynomials` so this Keras backend stays
bit-identical to the torch and JAX backends. Only the Horner evaluation
is backend-specific (it touches the tensor).
"""

from __future__ import annotations

from typing import Any

from omnibias.core.polynomials import (
    sigmoid_polynomial_coeffs as _sigmoid_polynomial_coeffs_core,
)

from keras import ops

sigmoid_polynomial_coeffs = _sigmoid_polynomial_coeffs_core


def _horner(coeffs: tuple[float, ...], x: Any) -> Any:
    """Stable Horner evaluation of ``sum_k coeffs[k] * x^k`` on a tensor."""
    deg = len(coeffs) - 1
    result = ops.ones_like(x) * coeffs[deg]
    for k in range(deg - 1, -1, -1):
        result = result * x + coeffs[k]
    return result


def sigmoid_nth_derivative(z: Any, n: int) -> Any:
    """``sigma^(n)(z)`` in closed form via Eulerian polynomials.

    One ``sigmoid`` call regardless of ``n``, then ``O(n)`` multiply-adds.
    """
    if n < 0:
        raise ValueError(f"order n must be >= 0, got {n}")
    s = ops.sigmoid(z)
    if n == 0:
        return s
    return _horner(sigmoid_polynomial_coeffs(n), s)


def softplus_nth_derivative(z: Any, n: int) -> Any:
    """``softplus^(n)(z)``: ``softplus(z)`` for ``n == 0``, else ``sigma^(n-1)(z)``."""
    if n < 0:
        raise ValueError(f"order n must be >= 0, got {n}")
    if n == 0:
        return ops.softplus(z)
    return sigmoid_nth_derivative(z, n - 1)


__all__ = [
    "sigmoid_nth_derivative",
    "sigmoid_polynomial_coeffs",
    "softplus_nth_derivative",
]
