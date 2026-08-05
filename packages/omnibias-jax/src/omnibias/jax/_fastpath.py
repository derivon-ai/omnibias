# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""JAX closed-form derivative-tower kernels.

These are point-for-point mirrors of the PyTorch fast paths in
:mod:`omnibias.fastpath.eulerian`,
:mod:`omnibias.fastpath.legendre`, and
:mod:`omnibias.fastpath.hermite`, with the polynomial coefficients
shared via :mod:`omnibias.fastpath.polynomials`.

All functions are pure JAX (``jax.numpy`` only). They are
trace-friendly, ``jit``-friendly, ``vmap``-friendly, and produce no
side effects.

Sigmoid evaluation strategy
---------------------------
The base value ``s = sigma(z)`` is evaluated with the **framework-native**
``jax.nn.sigmoid`` (via :func:`jax_sigmoid`). The torch twin uses
``torch.sigmoid`` (:mod:`omnibias.torch.fastpath.eulerian`). Both are the
stable two-branch form
``where(z >= 0, 1/(1+exp(-z)), exp(z)/(1+exp(z)))``, but they are *not*
guaranteed bit-identical in the extreme tails (``|z| \\gtrsim 40`` in
float64): each vendor's ``exp`` / fused kernel can differ by a few ULPs
once ``s`` is near 0 or 1. Shared polynomial coefficients still make the
*tower* bit-identical **given the same** ``s``; cross-backend parity tests
therefore compare the closed-form derivatives on moderate ``z``, while
``tests/test_sigmoid_tail_parity.py`` locks the documented tail contract
(finite, monotone, and within a few ULPs of each other at
``z = \\pm 40, \\pm 80``). Replacing either call with a hand-rolled common
formula would risk breaking existing bit-identity fixtures that pin the
native kernels, so we document rather than unify.
"""

from __future__ import annotations

from omnibias.core.polynomials import (
    hermite_coeffs,
    sigmoid_polynomial_coeffs,
    tanh_polynomial_coeffs,
)

import jax.numpy as jnp
from jax import Array


def _horner(coeffs: tuple[float, ...], x: Array) -> Array:
    """Stable Horner evaluation of ``sum_k coeffs[k] * x^k`` on a JAX array.

    The coefficient tuple is captured at trace time as a Python list of
    floats, so the resulting computation graph is fully static.
    """
    deg = len(coeffs) - 1
    result = jnp.full_like(x, coeffs[deg])
    for k in range(deg - 1, -1, -1):
        result = result * x + coeffs[k]
    return result


# ---------------------------------------------------------------------------
# sigmoid / softplus
# ---------------------------------------------------------------------------


def sigmoid_nth_derivative(z: Array, n: int) -> Array:
    """``sigma^(n)(z)`` via Eulerian polynomials. One ``sigmoid`` call."""
    if n < 0:
        raise ValueError(f"order n must be >= 0, got {n}")
    s = jax_sigmoid(z)
    if n == 0:
        return s
    return _horner(sigmoid_polynomial_coeffs(n), s)


def softplus_nth_derivative(z: Array, n: int) -> Array:
    """``softplus^(n)(z)``. For ``n >= 1`` equals ``sigma^(n-1)(z)``."""
    if n < 0:
        raise ValueError(f"order n must be >= 0, got {n}")
    if n == 0:
        # log1p(exp(z)) in a numerically stable way.
        return jnp.logaddexp(jnp.zeros_like(z), z)
    return sigmoid_nth_derivative(z, n - 1)


def jax_sigmoid(z: Array) -> Array:
    """Framework-native stable sigmoid (``jax.nn.sigmoid``).

    Routes through ``jax.nn.sigmoid``, which uses the two-branch form
    ``where(z >= 0, 1/(1+exp(-z)), exp(z)/(1+exp(z)))``; the naive
    ``1/(1+exp(-z))`` overflows for ``z << -88`` (f32) / ``-709`` (f64)
    and silently produces NaN through the ``(1 + inf) = inf → 0/0`` path.

    This is the intentional twin of ``torch.sigmoid`` in
    :mod:`omnibias.torch.fastpath.eulerian`. See the module docstring for
    the documented extreme-tail parity contract (not bit-identical; within
    a few ULPs).
    """
    import jax.nn as _jnn

    return _jnn.sigmoid(z)


# ---------------------------------------------------------------------------
# tanh
# ---------------------------------------------------------------------------


def tanh_nth_derivative(z: Array, n: int) -> Array:
    """``tanh^(n)(z)`` via the Legendre-style recursion."""
    if n < 0:
        raise ValueError(f"order n must be >= 0, got {n}")
    t = jnp.tanh(z)
    if n == 0:
        return t
    return _horner(tanh_polynomial_coeffs(n), t)


# ---------------------------------------------------------------------------
# Gaussian g(z) = exp(-z^2/2)
# ---------------------------------------------------------------------------


def gaussian_forward(z: Array) -> Array:
    return jnp.exp(-0.5 * z * z)


def gaussian_nth_derivative(z: Array, n: int) -> Array:
    """``g^(n)(z) = (-1)^n He_n(z) g(z)`` for ``g(z) = exp(-z^2/2)``."""
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
    "jax_sigmoid",
    "sigmoid_nth_derivative",
    "softplus_nth_derivative",
    "tanh_nth_derivative",
]
