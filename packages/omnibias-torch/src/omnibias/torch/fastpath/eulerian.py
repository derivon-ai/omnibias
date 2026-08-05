# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Closed-form derivatives of sigmoid (and softplus) via the
Eulerian/sigmoid-polynomial recursion.

Let ``s = sigma(z) = 1 / (1 + e^{-z})``. Then ``s' = s (1 - s)`` (a
Riccati identity), and every higher derivative of ``sigma`` is a
polynomial in ``s``:

    sigma^(n)(z) = P_n(s),   where  P_0(s) = s
                                    P_{n+1}(s) = s (1 - s) * P_n'(s).

The coefficients of ``P_n`` are (signed) Eulerian numbers:
``P_n(s) = sum_{k=0}^{n} (-1)^{n-k} <n, k> s^{k+1} (1 - s)^{n-k}``.
The recurrence lives in the **canonical** :mod:`omnibias.core.polynomials`
module so the torch and JAX backends always share a single source of
truth.  We import it here and re-export it under the historical name for
back-compat (callers writing ``from omnibias.torch.fastpath.eulerian
import sigmoid_polynomial_coeffs`` keep working).

For ``softplus(z) = log(1 + e^z)`` the relation
``softplus^(n)(z) = sigma^(n-1)(z)`` for ``n >= 1`` lets us reuse the
same machinery; the only change is an off-by-one in the requested order.

Sigmoid evaluation strategy
---------------------------
The base value ``s = sigma(z)`` is evaluated with the **framework-native**
``torch.sigmoid``. The JAX twin uses ``jax.nn.sigmoid``
(:func:`omnibias.jax._fastpath.jax_sigmoid`). Both are the stable
two-branch form ``where(z >= 0, 1/(1+exp(-z)), exp(z)/(1+exp(z)))``, but
they are *not* guaranteed bit-identical in the extreme tails
(``|z| \\gtrsim 40`` in float64): each vendor's ``exp`` / fused kernel can
differ by a few ULPs once ``s`` is near 0 or 1. Shared polynomial
coefficients still make the *tower* bit-identical **given the same** ``s``;
cross-backend parity tests therefore compare the closed-form derivatives on
moderate ``z``, while ``tests/test_sigmoid_tail_parity.py`` locks the
documented tail contract (finite, monotone, and within a few ULPs of each
other at ``z = \\pm 40, \\pm 80``). Replacing either call with a hand-rolled
common formula would risk breaking existing bit-identity fixtures that pin
the native kernels, so we document rather than unify.
"""

from __future__ import annotations

from omnibias.core.polynomials import (
    sigmoid_polynomial_coeffs as _sigmoid_polynomial_coeffs_core,
)

import torch
from torch import Tensor

# Re-export under the historical name for back-compat.
sigmoid_polynomial_coeffs = _sigmoid_polynomial_coeffs_core


def _horner(coeffs: tuple[float, ...], x: Tensor) -> Tensor:
    """Stable Horner evaluation of ``sum_k coeffs[k] * x^k`` on a tensor."""
    deg = len(coeffs) - 1
    result = torch.full_like(x, coeffs[deg])
    for k in range(deg - 1, -1, -1):
        result = result * x + coeffs[k]
    return result


def sigmoid_nth_derivative(z: Tensor, n: int) -> Tensor:
    """``sigma^(n)(z)`` evaluated in closed form via Eulerian polynomials.

    One framework-native ``torch.sigmoid`` call regardless of ``n``, then
    ``O(n)`` multiply-adds in Horner form. See the module docstring for the
    torch / JAX tail-parity contract.
    """
    if n < 0:
        raise ValueError(f"order n must be >= 0, got {n}")
    # Framework-native stable sigmoid (see module docstring: not bit-identical
    # to jax.nn.sigmoid in extreme tails; polynomials are shared either way).
    s = torch.sigmoid(z)
    if n == 0:
        return s
    coeffs = sigmoid_polynomial_coeffs(n)
    return _horner(coeffs, s)


def softplus_nth_derivative(z: Tensor, n: int) -> Tensor:
    """``softplus^(n)(z)``.

    For ``n = 0`` returns ``softplus(z)`` directly; for ``n >= 1``
    returns ``sigma^(n-1)(z)``.
    """
    if n < 0:
        raise ValueError(f"order n must be >= 0, got {n}")
    if n == 0:
        return torch.nn.functional.softplus(z)
    return sigmoid_nth_derivative(z, n - 1)


__all__ = [
    "sigmoid_nth_derivative",
    "sigmoid_polynomial_coeffs",
    "softplus_nth_derivative",
]
