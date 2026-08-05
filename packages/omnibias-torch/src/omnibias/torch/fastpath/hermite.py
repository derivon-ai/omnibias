# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Closed-form derivatives of the Gaussian activation via probabilist's
Hermite polynomials.

Let ``g(z) = e^{-z^2/2}`` be the (unnormalised) Gaussian. Then

    g^(n)(z) = (-1)^n * He_n(z) * g(z),

where ``He_n`` is the probabilist's Hermite polynomial obeying the
three-term recurrence

    He_0(z) = 1, He_1(z) = z, He_{n+1}(z) = z * He_n(z) - n * He_{n-1}(z).

The coefficient generator lives in the canonical
:mod:`omnibias.core.polynomials` module shared by torch and JAX.
The cost per call is one ``exp`` plus ``O(n)`` multiply-adds in Horner
form.
"""

from __future__ import annotations

from omnibias.core.polynomials import hermite_coeffs as _hermite_coeffs_core

import torch
from torch import Tensor

hermite_coeffs = _hermite_coeffs_core


def _horner(coeffs: tuple[float, ...], x: Tensor) -> Tensor:
    deg = len(coeffs) - 1
    result = torch.full_like(x, coeffs[deg])
    for k in range(deg - 1, -1, -1):
        result = result * x + coeffs[k]
    return result


def gaussian_nth_derivative(z: Tensor, n: int) -> Tensor:
    """``g^(n)(z)`` where ``g(z) = exp(-z^2/2)``.

    Uses the identity ``g^(n)(z) = (-1)^n * He_n(z) * g(z)``.
    """
    if n < 0:
        raise ValueError(f"order n must be >= 0, got {n}")
    g = torch.exp(-0.5 * z * z)
    if n == 0:
        return g
    coeffs = hermite_coeffs(n)
    poly = _horner(coeffs, z)
    sign = -1.0 if (n & 1) else 1.0
    return sign * poly * g


def gaussian_forward(z: Tensor) -> Tensor:
    """``g(z) = exp(-z^2/2)``; provided here so callers do not need to
    duplicate the formula."""
    return torch.exp(-0.5 * z * z)


__all__ = [
    "gaussian_forward",
    "gaussian_nth_derivative",
    "hermite_coeffs",
]
