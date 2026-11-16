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
    """Horner evaluation of ``sum_k coeffs[k] * x^k``: ``result = result * x + coeffs[k]`` from the leading coefficient down.

    The leading step is ``x * c`` rather than a ``full_like`` tensor times ``x`` (the same product, one launch fewer), and the
    add of a zero coefficient is skipped when the next coefficient down is not zero (the ``tanh`` and sigmoid-family
    polynomials have parity, so that is about half of them). The next step multiplies by ``x`` and adds that nonzero
    coefficient, which absorbs the one thing the skipped ``+ 0.0`` could change (the sign of an exact zero), so every bit of
    the result is unchanged. Half precision keeps the original form, which rounds the coefficient to the tensor dtype first.
    """
    deg = len(coeffs) - 1
    if deg == 0 or x.dtype not in (torch.float32, torch.float64):
        result = torch.full_like(x, coeffs[deg])
        for k in range(deg - 1, -1, -1):
            result = result * x + coeffs[k]
        return result
    result = x * coeffs[deg]  # r_{k+1} * x for k = deg - 1
    for k in range(deg - 1, -1, -1):
        if k == 0 or coeffs[k] != 0.0 or coeffs[k - 1] == 0.0:
            result = result + coeffs[k]
        if k:
            result = result * x
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



def tanh_tower_rows(z: Tensor, max_order: int) -> list[Tensor]:
    """The rows of :func:`tanh_tower` as a list, for callers that index them and need no stacked copy."""
    if max_order < 0:
        raise ValueError("max_order must be nonnegative")
    value = torch.tanh(z)
    return [value] + [_horner(tanh_polynomial_coeffs(n), value) for n in range(1, max_order + 1)]


def tanh_tower(z: Tensor, max_order: int) -> Tensor:
    """All derivatives through max_order from one native tanh evaluation."""
    return torch.stack(tanh_tower_rows(z, max_order), dim=0)


__all__ = [
    "tanh_nth_derivative",
    "tanh_polynomial_coeffs",
    "tanh_tower",
    "tanh_tower_rows",
]
