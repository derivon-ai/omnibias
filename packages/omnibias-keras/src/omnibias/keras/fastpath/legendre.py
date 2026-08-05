# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Closed-form derivatives of ``tanh`` via the Legendre-style recursion,
evaluated with ``keras.ops``.

Let ``t = tanh(z)``. Then ``t' = 1 - t^2`` and each higher derivative is
a polynomial in ``t``:

    tanh^(n)(z) = T_n(t),    T_0(t) = t,
                             T_{n+1}(t) = (1 - t^2) * T_n'(t).

Coefficients come from the shared :mod:`omnibias.core.polynomials`.
"""

from __future__ import annotations

from typing import Any

from omnibias.core.polynomials import (
    tanh_polynomial_coeffs as _tanh_polynomial_coeffs_core,
)

from keras import ops

tanh_polynomial_coeffs = _tanh_polynomial_coeffs_core


def _horner(coeffs: tuple[float, ...], x: Any) -> Any:
    deg = len(coeffs) - 1
    result = ops.ones_like(x) * coeffs[deg]
    for k in range(deg - 1, -1, -1):
        result = result * x + coeffs[k]
    return result


def tanh_nth_derivative(z: Any, n: int) -> Any:
    """``tanh^(n)(z)`` evaluated in closed form."""
    if n < 0:
        raise ValueError(f"order n must be >= 0, got {n}")
    t = ops.tanh(z)
    if n == 0:
        return t
    return _horner(tanh_polynomial_coeffs(n), t)


__all__ = [
    "tanh_nth_derivative",
    "tanh_polynomial_coeffs",
]
