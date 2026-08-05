# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Closed-form derivatives of ``tanh`` via the Legendre-style recursion.

Let ``t = tanh(z)``. Then ``t' = 1 - t^2`` (a Riccati identity), and
each higher derivative of ``tanh`` is a polynomial in ``t``:

    tanh^(n)(z) = T_n(t),    T_0(t) = t,
                             T_{n+1}(t) = (1 - t^2) * T_n'(t).

Each step of the recursion factors out ``(1 - t^2)``, the same weight
appearing in the Legendre orthogonality on ``[-1, 1]`` -- hence the
module name.  The coefficient generator lives in the canonical
:mod:`omnibias.core.polynomials` module shared by torch and JAX.
"""

from __future__ import annotations

from omnibias.core.polynomials import (
    tanh_polynomial_coeffs as _tanh_polynomial_coeffs_core,
)

import torch
from torch import Tensor

tanh_polynomial_coeffs = _tanh_polynomial_coeffs_core


def _horner(coeffs: tuple[float, ...], x: Tensor) -> Tensor:
    deg = len(coeffs) - 1
    result = torch.full_like(x, coeffs[deg])
    for k in range(deg - 1, -1, -1):
        result = result * x + coeffs[k]
    return result


def tanh_nth_derivative(z: Tensor, n: int) -> Tensor:
    """``tanh^(n)(z)`` evaluated in closed form."""
    if n < 0:
        raise ValueError(f"order n must be >= 0, got {n}")
    t = torch.tanh(z)
    if n == 0:
        return t
    coeffs = tanh_polynomial_coeffs(n)
    return _horner(coeffs, t)


__all__ = [
    "tanh_nth_derivative",
    "tanh_polynomial_coeffs",
]
