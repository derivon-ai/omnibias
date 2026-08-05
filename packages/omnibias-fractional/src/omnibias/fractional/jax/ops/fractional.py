# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Grid-based fractional derivatives (jax).

Bit-identical twin of :mod:`omnibias.fractional.torch.ops.fractional`. These are
non-local numerical approximations, not closed-form sigma-tower derivatives. For
the package's *closed-form* operator (on the analytic-function class) see
:mod:`omnibias.fractional.jax.ops.analytic`.
"""

from __future__ import annotations

import math

import jax.numpy as jnp
from jax import Array
from omnibias.fractional._core.kernels import gl_matrix, spectral_multiplier


def _gl_weights_backend(alpha: Array, n: int) -> Array:
    r"""Differentiable Grunwald-Letnikov weights (in-backend ``cumprod`` recurrence).

    JAX twin of :func:`omnibias.fractional.torch.ops.fractional._gl_weights_backend`:
    ``w_0 = 1``, ``w_k = w_{k-1} (1 - (alpha+1)/k)`` -- smooth in ``alpha``.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    ones = jnp.ones(1, dtype=alpha.dtype)
    if n == 1:
        return ones
    k = jnp.arange(1, n, dtype=alpha.dtype)
    factors = 1.0 - (alpha + 1.0) / k
    return jnp.concatenate([ones, jnp.cumprod(factors)])


def _gl_matmul(f: Array, alpha: Array, h: float) -> Array:
    r"""Grunwald-Letnikov operator with a tensor-valued ``alpha`` (differentiable).

    Same lower-triangular Toeplitz operator as
    :func:`omnibias.fractional._core.kernels.gl_matrix`, built in-backend so it is
    differentiable w.r.t. ``alpha`` through the weights and the ``h^{-alpha}`` scale.
    """
    if h <= 0.0:
        raise ValueError(f"grid spacing h must be > 0, got {h}")
    n = f.shape[0]
    a = alpha.astype(f.dtype)
    w = _gl_weights_backend(a, n)
    idx = jnp.arange(n)
    diff = idx[:, None] - idx[None, :]
    mat = jnp.where(diff >= 0, w[jnp.maximum(diff, 0)], 0.0)
    return (mat * (h ** (-a))) @ f


def grunwald_letnikov(f: Array, *, alpha: float | Array, h: float) -> Array:
    r"""Grunwald-Letnikov fractional derivative of order ``alpha`` on a grid.

    A Python ``float`` ``alpha`` uses the fast numpy kernel (unchanged); a JAX
    array (or a tracer under ``jax.grad`` / ``jax.jit``) takes the differentiable
    in-backend path so the order is learnable.
    """
    if isinstance(alpha, int | float):
        n = f.shape[0]
        mat = jnp.asarray(gl_matrix(alpha, n, h), dtype=f.dtype)
        return mat @ f
    return _gl_matmul(f, alpha, h)


def riemann_liouville(f: Array, *, alpha: float | Array, h: float) -> Array:
    r"""Riemann-Liouville fractional derivative (Grunwald-Letnikov discretisation)."""
    return grunwald_letnikov(f, alpha=alpha, h=h)


def caputo(f: Array, *, alpha: float | Array, h: float) -> Array:
    r"""Caputo fractional derivative for ``0 < alpha < 1``.

    ``alpha`` may be a JAX array, in which case the order is learnable; the range
    is validated only for Python-``float`` orders (a tracer cannot be checked).
    """
    if isinstance(alpha, int | float) and not (0.0 < alpha < 1.0):
        raise ValueError(f"caputo here supports 0 < alpha < 1, got {alpha}")
    return grunwald_letnikov(f - f[0], alpha=alpha, h=h)


def _spectral_multiplier_backend(alpha: Array, n: int, length: float) -> Array:
    r"""Differentiable Fourier multiplier ``(i k)^alpha`` via ``exp(alpha log(i k))``.

    JAX twin of the torch backend multiplier; the zero mode is masked before the
    ``log`` so the gradient w.r.t. ``alpha`` is exact and ``nan``-free.
    """
    if length <= 0.0:
        raise ValueError(f"length must be > 0, got {length}")
    kk = jnp.fft.fftfreq(n, d=length / n) * (2.0 * math.pi)
    z = 1j * kk.astype(jnp.complex128)
    z_safe = jnp.where(kk == 0, jnp.ones_like(z), z)
    mult = jnp.exp(alpha.astype(jnp.float64) * jnp.log(z_safe))
    return jnp.where(kk == 0, jnp.zeros_like(mult), mult)


def spectral_fractional(f: Array, *, alpha: float | Array, length: float) -> Array:
    r"""Spectral fractional derivative on a periodic domain via the FFT.

    A tensor ``alpha`` takes the differentiable in-backend path (learnable order).
    """
    n = f.shape[0]
    if isinstance(alpha, int | float):
        mult = jnp.asarray(spectral_multiplier(alpha, n, length), dtype=jnp.complex128)
    else:
        mult = _spectral_multiplier_backend(alpha, n, length)
    fhat = jnp.fft.fft(f.astype(jnp.complex128))
    return jnp.fft.ifft(mult * fhat)


__all__ = ["caputo", "grunwald_letnikov", "riemann_liouville", "spectral_fractional"]
