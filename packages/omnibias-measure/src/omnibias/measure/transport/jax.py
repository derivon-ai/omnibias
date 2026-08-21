# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""JAX twin of activation-mixture CDF / PDF (theory 03-04).

Mixtures come from the founding bias collapse (``delta -> 0``).
Temperature collapse (``beta -> inf``, feasibility) does not appear.
Do not conflate the two.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array
from omnibias.measure.transport._core import ActivationMixture


def _sigmoid(z: Array) -> Array:
    x = jnp.asarray(z, dtype=jnp.float64)
    return jnp.where(x >= 0.0, 1.0 / (1.0 + jnp.exp(-x)), jnp.exp(x) / (1.0 + jnp.exp(x)))


def cdf(mix: ActivationMixture, x: Array) -> Array:
    xv = jnp.asarray(x, dtype=jnp.float64)
    loc = jnp.asarray(mix.loc(), dtype=jnp.float64)
    alpha = jnp.asarray(mix.alpha, dtype=jnp.float64)
    w = jnp.asarray(mix.weights, dtype=jnp.float64)
    z = alpha * (xv[..., None] - loc)
    return jnp.sum(w * _sigmoid(z), axis=-1)


def pdf(mix: ActivationMixture, x: Array) -> Array:
    xv = jnp.asarray(x, dtype=jnp.float64)
    loc = jnp.asarray(mix.loc(), dtype=jnp.float64)
    alpha = jnp.asarray(mix.alpha, dtype=jnp.float64)
    w = jnp.asarray(mix.weights, dtype=jnp.float64)
    z = alpha * (xv[..., None] - loc)
    s = _sigmoid(z)
    return jnp.sum(w * alpha * s * (1.0 - s), axis=-1)


def quantile(mix: ActivationMixture, q: Array, *, steps: int = 40) -> Array:
    qq = jnp.asarray(q, dtype=jnp.float64)
    mean = jnp.asarray(mix.mean(), dtype=jnp.float64)
    spread = 1.0 / float(mix.alpha.mean())
    x = mean + spread * (2.0 * qq - 1.0)
    for _ in range(int(steps)):
        dens = jnp.maximum(pdf(mix, x), 1e-30)
        x = x - (cdf(mix, x) - qq) / dens
    return x


__all__ = ["ActivationMixture", "cdf", "pdf", "quantile"]
