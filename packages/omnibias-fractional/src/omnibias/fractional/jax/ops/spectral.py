# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Non-periodic spectral fractional operators (jax twin).

Bit-identical twin of :mod:`omnibias.fractional.torch.ops.spectral`:
:func:`spectral_fractional_laplacian` (two-sided ``(-Delta)^{alpha/2}`` via
orthonormal DST-I / DCT-II on a bounded interval) and
:func:`windowed_spectral_fractional` (Tukey-windowed periodic operator). See the
torch twin for the full semantics; differentiable in the order ``alpha``.
"""

from __future__ import annotations

import math

import jax.numpy as jnp
from jax import Array


def _dst1_matrix(n: int, dtype: jnp.dtype) -> Array:
    r"""Orthonormal, symmetric, self-inverse DST-I matrix (see the torch twin)."""
    idx = jnp.arange(1, n + 1, dtype=dtype)
    ang = math.pi / (n + 1) * idx.reshape(n, 1) * idx.reshape(1, n)
    return math.sqrt(2.0 / (n + 1)) * jnp.sin(ang)


def _dct2_matrix(n: int, dtype: jnp.dtype) -> Array:
    r"""Orthonormal DCT-II matrix ``C`` (``C C^T = I``); inverse is ``C^T``."""
    k = jnp.arange(n, dtype=dtype).reshape(n, 1)
    j = jnp.arange(n, dtype=dtype).reshape(1, n)
    c = math.sqrt(2.0 / n) * jnp.cos(math.pi * (2.0 * j + 1.0) * k / (2.0 * n))
    return c.at[0, :].multiply(1.0 / math.sqrt(2.0))


def _symbol(xi: Array, alpha: float | Array, *, mask_zero: bool) -> Array:
    r"""Spectral symbol ``xi^alpha`` (differentiable in ``alpha``; zero mode masked)."""
    a = jnp.asarray(alpha, dtype=xi.dtype)
    safe = jnp.where(xi > 0, xi, jnp.ones_like(xi))
    mult = jnp.exp(a * jnp.log(safe))
    if mask_zero:
        return jnp.where(xi > 0, mult, jnp.zeros_like(mult))
    return mult


def spectral_fractional_laplacian(
    f: Array,
    *,
    alpha: float | Array,
    length: float,
    bc: str = "dirichlet",
) -> Array:
    r"""Two-sided spectral fractional Laplacian ``(-Delta)^{alpha/2} f`` (jax twin)."""
    if length <= 0.0:
        raise ValueError(f"length must be > 0, got {length}")
    if f.ndim != 1:
        raise ValueError(f"f must be 1-D (samples on a grid), got shape {tuple(f.shape)}")
    n = f.shape[0]
    if bc == "dirichlet":
        q = _dst1_matrix(n, f.dtype)
        xi = jnp.arange(1, n + 1, dtype=f.dtype) * (math.pi / length)
        mult = _symbol(xi, alpha, mask_zero=False)
        return q @ (mult * (q @ f))
    if bc == "neumann":
        c = _dct2_matrix(n, f.dtype)
        xi = jnp.arange(n, dtype=f.dtype) * (math.pi / length)
        mult = _symbol(xi, alpha, mask_zero=True)
        return c.T @ (mult * (c @ f))
    raise ValueError(f"bc must be 'dirichlet' or 'neumann', got {bc!r}")


def tukey_window(n: int, taper: float, dtype: jnp.dtype) -> Array:
    r"""Tukey (tapered-cosine) window of length ``n`` (see the torch twin)."""
    if not (0.0 <= taper <= 1.0):
        raise ValueError(f"taper must be in [0, 1], got {taper}")
    if taper == 0.0 or n <= 1:
        return jnp.ones(n, dtype=dtype)
    x = jnp.arange(n, dtype=dtype) / (n - 1)
    edge = taper / 2.0
    lo = x < edge
    hi = x > 1.0 - edge
    w = jnp.ones(n, dtype=dtype)
    w = jnp.where(lo, 0.5 * (1.0 + jnp.cos(math.pi * (2.0 * x / taper - 1.0))), w)
    w = jnp.where(hi, 0.5 * (1.0 + jnp.cos(math.pi * (2.0 * x / taper - 2.0 / taper + 1.0))), w)
    return w


def windowed_spectral_fractional(
    f: Array,
    *,
    alpha: float | Array,
    length: float,
    taper: float = 0.1,
) -> Array:
    r"""Windowed-FFT spectral fractional derivative for a non-periodic ``f`` (jax twin)."""
    from omnibias.fractional.jax.ops.fractional import spectral_fractional

    if f.ndim != 1:
        raise ValueError(f"f must be 1-D (samples on a grid), got shape {tuple(f.shape)}")
    w = tukey_window(f.shape[0], taper, f.dtype)
    out: Array = spectral_fractional(f * w, alpha=alpha, length=length)
    return out


__all__ = [
    "spectral_fractional_laplacian",
    "tukey_window",
    "windowed_spectral_fractional",
]
