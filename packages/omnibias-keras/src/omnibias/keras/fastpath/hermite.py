# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Closed-form derivatives of the Gaussian activation via probabilist's
Hermite polynomials, evaluated with ``keras.ops``.

Let ``g(z) = exp(-z^2/2)``. Then ``g^(n)(z) = (-1)^n He_n(z) g(z)``,
where ``He_n`` obeys ``He_0 = 1, He_1 = z, He_{n+1} = z He_n - n He_{n-1}``.
Coefficients come from the shared :mod:`omnibias.core.polynomials`.
"""

from __future__ import annotations

from typing import Any

from omnibias.core.polynomials import hermite_coeffs as _hermite_coeffs_core

from keras import ops

hermite_coeffs = _hermite_coeffs_core


def _horner(coeffs: tuple[float, ...], x: Any) -> Any:
    deg = len(coeffs) - 1
    result = ops.ones_like(x) * coeffs[deg]
    for k in range(deg - 1, -1, -1):
        result = result * x + coeffs[k]
    return result


def gaussian_forward(z: Any) -> Any:
    """``g(z) = exp(-z^2/2)``."""
    return ops.exp(-0.5 * z * z)


def gaussian_nth_derivative(z: Any, n: int) -> Any:
    """``g^(n)(z)`` via ``g^(n)(z) = (-1)^n He_n(z) g(z)``."""
    if n < 0:
        raise ValueError(f"order n must be >= 0, got {n}")
    g = gaussian_forward(z)
    if n == 0:
        return g
    poly = _horner(hermite_coeffs(n), z)
    sign = -1.0 if (n & 1) else 1.0
    return sign * poly * g


__all__ = [
    "gaussian_forward",
    "gaussian_nth_derivative",
    "hermite_coeffs",
]
