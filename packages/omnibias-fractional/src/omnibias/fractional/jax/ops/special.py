# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Differentiable special functions of fractional calculus (jax twin).

Bit-identical twin of :mod:`omnibias.fractional.torch.ops.special`:
:func:`mittag_leffler`, :func:`polylog`, :func:`lerch`,
:func:`lower_incomplete_gamma`. Each is a differentiable truncated series; see the
torch twin for the identities and truncation notes.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array
from jax.scipy.special import gammaln


def _recip_gamma(y: Array) -> Array:
    r"""Reciprocal gamma ``1 / Gamma(y)`` (see the torch twin)."""
    log_mag = -gammaln(y)
    parity = jnp.remainder(jnp.ceil(-y), 2.0)
    sign = jnp.where(y > 0, jnp.ones_like(y), 1.0 - 2.0 * parity)
    return sign * jnp.exp(log_mag)


def _int_powers(z: Array, terms: int, *, start_one: bool) -> Array:
    """Stack integer powers of ``z`` via ``cumprod`` (see the torch twin)."""
    count = terms - 1 if start_one else terms
    zr = jnp.broadcast_to(z[None], (count, *z.shape))
    cp = jnp.cumprod(zr, axis=0) if count > 0 else zr
    if start_one:
        ones = jnp.ones((1, *z.shape), dtype=z.dtype)
        return jnp.concatenate([ones, cp], axis=0)
    return cp


def mittag_leffler(
    z: Array | float,
    alpha: float | Array,
    beta: float | Array = 1.0,
    *,
    terms: int = 64,
) -> Array:
    r"""Mittag-Leffler ``E_{alpha,beta}(z)`` (jax twin; see the torch twin)."""
    if terms < 1:
        raise ValueError("terms must be >= 1")
    z_t = jnp.asarray(z)
    a = jnp.asarray(alpha, dtype=z_t.dtype)
    b = jnp.asarray(beta, dtype=z_t.dtype)
    k = jnp.arange(terms, dtype=z_t.dtype)
    rg = _recip_gamma(a * k + b).reshape((terms,) + (1,) * z_t.ndim)
    zk = _int_powers(z_t, terms, start_one=True)
    out: Array = (zk * rg).sum(axis=0)
    return out


def polylog(s: float | Array, z: Array | float, *, terms: int = 64) -> Array:
    r"""Polylogarithm ``Li_s(z)`` (jax twin; see the torch twin)."""
    if terms < 1:
        raise ValueError("terms must be >= 1")
    z_t = jnp.asarray(z)
    s_t = jnp.asarray(s, dtype=z_t.dtype)
    k = jnp.arange(1, terms + 1, dtype=z_t.dtype)
    ks = jnp.exp(-s_t * jnp.log(k)).reshape((terms,) + (1,) * z_t.ndim)
    zk = _int_powers(z_t, terms, start_one=False)
    out: Array = (zk * ks).sum(axis=0)
    return out


def lerch(
    z: Array | float,
    s: float | Array,
    a: float | Array,
    *,
    terms: int = 64,
) -> Array:
    r"""Lerch transcendent ``Phi(z, s, a)`` (jax twin; see the torch twin)."""
    if terms < 1:
        raise ValueError("terms must be >= 1")
    z_t = jnp.asarray(z)
    s_t = jnp.asarray(s, dtype=z_t.dtype)
    a_t = jnp.asarray(a, dtype=z_t.dtype)
    k = jnp.arange(terms, dtype=z_t.dtype)
    denom = jnp.exp(s_t * jnp.log(k + a_t)).reshape((terms,) + (1,) * z_t.ndim)
    zk = _int_powers(z_t, terms, start_one=True)
    out: Array = (zk / denom).sum(axis=0)
    return out


def lower_incomplete_gamma(
    s: float | Array,
    x: Array | float,
    *,
    terms: int = 64,
) -> Array:
    r"""Lower incomplete gamma ``gamma(s, x)`` (jax twin; see the torch twin)."""
    if terms < 1:
        raise ValueError("terms must be >= 1")
    x_t = jnp.asarray(x)
    s_t = jnp.asarray(s, dtype=x_t.dtype)
    k = jnp.arange(terms, dtype=x_t.dtype)
    kfact = jnp.exp(gammaln(k + 1.0)).reshape((terms,) + (1,) * x_t.ndim)
    denom = kfact * (s_t + k).reshape((terms,) + (1,) * x_t.ndim)
    mxk = _int_powers(-x_t, terms, start_one=True)
    series = (mxk / denom).sum(axis=0)
    out: Array = jnp.exp(s_t * jnp.log(x_t)) * series
    return out


__all__ = [
    "lerch",
    "lower_incomplete_gamma",
    "mittag_leffler",
    "polylog",
]
