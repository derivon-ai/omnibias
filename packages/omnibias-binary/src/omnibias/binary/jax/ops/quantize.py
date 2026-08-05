# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""JAX closed-form quantization-gradient kernels."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array
from omnibias.core.polynomials import sigmoid_polynomial_coeffs, tanh_polynomial_coeffs

__all__ = [
    "binarize",
    "binarize01",
    "heaviside",
    "kbit_quantize",
    "riccati_sigmoid_derivative",
    "riccati_tanh_derivative",
    "ternarize",
]


def _horner(coeffs: tuple[float, ...], x: Array) -> Array:
    deg = len(coeffs) - 1
    result = jnp.full_like(x, coeffs[deg])
    for k in range(deg - 1, -1, -1):
        result = result * x + coeffs[k]
    return result


def riccati_tanh_derivative(t: Array, order: int = 1) -> Array:
    """Evaluate ``T_order(t) = tanh^(order)(z)`` as a polynomial in ``t = tanh(z)``."""
    if order < 0:
        raise ValueError(f"order must be >= 0, got {order}")
    if order == 0:
        return t
    coeffs = tanh_polynomial_coeffs(order)
    return _horner(coeffs, t)


def _tanh_beta_prime(z: Array, beta: float) -> Array:
    """``beta * tanh'(beta z)`` via the Riccati polynomial at ``t = tanh(beta z)``."""
    t = jnp.tanh(beta * z)
    return beta * riccati_tanh_derivative(t, order=1)


def riccati_sigmoid_derivative(s: Array, order: int = 1) -> Array:
    """Evaluate ``P_order(s) = sigmoid^(order)(z)`` as a polynomial in ``s = sigmoid(z)``."""
    if order < 0:
        raise ValueError(f"order must be >= 0, got {order}")
    if order == 0:
        return s
    coeffs = sigmoid_polynomial_coeffs(order)
    return _horner(coeffs, s)


def _sigmoid_beta_prime(z: Array, beta: float) -> Array:
    """``beta * sigmoid'(beta z)`` via the Riccati polynomial at ``s = sigmoid(beta z)``."""
    s = jax.nn.sigmoid(beta * z)
    return beta * riccati_sigmoid_derivative(s, order=1)


def _kbit_thresholds(bits: int, lo: float, hi: float) -> tuple[float, int, tuple[float, ...]]:
    n_levels = 2**bits
    level_step = (hi - lo) / (n_levels - 1)
    thresholds = tuple(lo + k * level_step for k in range(1, n_levels))
    return level_step, n_levels, thresholds


def _kbit_forward(z: Array, bits: int, lo: float, hi: float) -> Array:
    level_step, n_levels, _ = _kbit_thresholds(bits, lo, hi)
    idx = jnp.round((z - lo) / level_step)
    idx = jnp.clip(idx, 0, n_levels - 1)
    return lo + idx * level_step


def _kbit_surrogate_grad(z: Array, bits: int, lo: float, hi: float, beta: float) -> Array:
    level_step, _, thresholds = _kbit_thresholds(bits, lo, hi)
    grad = jnp.zeros_like(z)
    for t_k in thresholds:
        grad = grad + level_step * _tanh_beta_prime(z - t_k, beta)
    return grad


def _binarize_fwd(z: Array, beta: float) -> tuple[Array, tuple[Array, float]]:
    y = jnp.where(z >= 0, 1.0, -1.0)
    return y, (z, beta)


def _binarize_bwd(res: tuple[Array, float], grad_out: Array) -> tuple[Array, Array]:
    z, beta = res
    grad_z = grad_out * _tanh_beta_prime(z, beta)
    # Learnable-beta surrogate cotangent: d/dbeta tanh(beta z) = z * (1 - tanh^2).
    t = jnp.tanh(beta * z)
    grad_beta = jnp.sum(grad_out * z * riccati_tanh_derivative(t, order=1))
    return grad_z, grad_beta


@jax.custom_vjp
def binarize(z: Array, beta: float = 10.0) -> Array:
    """Hard ``sign(z)`` in ``{-1, +1}`` (``sign(0)=+1``); Riccati surrogate backward."""
    return jnp.where(z >= 0, 1.0, -1.0)


binarize.defvjp(_binarize_fwd, _binarize_bwd)


def _binarize01_fwd(z: Array, beta: float) -> tuple[Array, tuple[Array, float]]:
    y = jnp.where(z >= 0, 1.0, 0.0)
    return y, (z, beta)


def _binarize01_bwd(res: tuple[Array, float], grad_out: Array) -> tuple[Array, Array]:
    z, beta = res
    grad_z = grad_out * _sigmoid_beta_prime(z, beta)
    # Learnable-beta surrogate cotangent: d/dbeta sigmoid(beta z) = z * s (1 - s).
    s = jax.nn.sigmoid(beta * z)
    grad_beta = jnp.sum(grad_out * z * riccati_sigmoid_derivative(s, order=1))
    return grad_z, grad_beta


@jax.custom_vjp
def binarize01(z: Array, beta: float = 10.0) -> Array:
    """Hard Heaviside step in ``{0, 1}`` (``H(0)=1``); ``sigmoid(beta z)`` surrogate backward.

    The ``{0, 1}`` codomain twin of :func:`binarize`, using the Eulerian
    ``sigmoid_polynomial_coeffs`` Riccati tower. Affinely conjugate to
    :func:`binarize`: equals ``(binarize(z, beta / 2) + 1) / 2`` in forward and
    backward.
    """
    return jnp.where(z >= 0, 1.0, 0.0)


binarize01.defvjp(_binarize01_fwd, _binarize01_bwd)


def heaviside(z: Array, beta: float = 10.0) -> Array:
    """Alias of :func:`binarize01` (the hard ``{0, 1}`` Heaviside step)."""
    return binarize01(z, beta)


def _ternarize_fwd(z: Array, beta: float, delta: float) -> tuple[Array, tuple[Array, float, float]]:
    y = jnp.where(z > delta, 1.0, jnp.where(z < -delta, -1.0, 0.0))
    return y, (z, beta, delta)


def _ternarize_bwd(res: tuple[Array, float, float], grad_out: Array) -> tuple[Array, Array, None]:
    z, beta, delta = res
    half = 0.5
    grad_z = grad_out * half * (
        _tanh_beta_prime(z - delta, beta) + _tanh_beta_prime(z + delta, beta)
    )
    t_lo = jnp.tanh(beta * (z - delta))
    t_hi = jnp.tanh(beta * (z + delta))
    grad_beta = jnp.sum(
        grad_out
        * half
        * (
            (z - delta) * riccati_tanh_derivative(t_lo, order=1)
            + (z + delta) * riccati_tanh_derivative(t_hi, order=1)
        )
    )
    return grad_z, grad_beta, None


@jax.custom_vjp
def ternarize(z: Array, beta: float = 10.0, delta: float = 0.5) -> Array:
    """Hard ternary ``{-1, 0, +1}``; smooth ``tanh`` dead-zone surrogate backward."""
    return jnp.where(z > delta, 1.0, jnp.where(z < -delta, -1.0, 0.0))


ternarize.defvjp(_ternarize_fwd, _ternarize_bwd)


def _kbit_quantize_fwd(
    z: Array, bits: int, lo: float, hi: float, beta: float,
) -> tuple[Array, tuple[Array, int, float, float, float]]:
    y = _kbit_forward(z, bits, lo, hi)
    return y, (z, bits, lo, hi, beta)


def _kbit_quantize_bwd(
    res: tuple[Array, int, float, float, float], grad_out: Array,
) -> tuple[Array, None, None, None, Array]:
    z, bits, lo, hi, beta = res
    grad_z = grad_out * _kbit_surrogate_grad(z, bits, lo, hi, beta)
    level_step, _, thresholds = _kbit_thresholds(bits, lo, hi)
    acc = jnp.zeros_like(z)
    for t_k in thresholds:
        t = jnp.tanh(beta * (z - t_k))
        acc = acc + level_step * (z - t_k) * riccati_tanh_derivative(t, order=1)
    grad_beta = jnp.sum(grad_out * acc)
    return grad_z, None, None, None, grad_beta


@jax.custom_vjp
def kbit_quantize(
    z: Array,
    bits: int = 2,
    lo: float = -1.0,
    hi: float = 1.0,
    beta: float = 10.0,
) -> Array:
    """Uniform k-bit quantize on ``[lo, hi]``; tanh-step surrogate backward."""
    if bits < 1:
        raise ValueError(f"bits must be >= 1, got {bits}")
    return _kbit_forward(z, bits, lo, hi)


kbit_quantize.defvjp(_kbit_quantize_fwd, _kbit_quantize_bwd)
