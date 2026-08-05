# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Modern Hopfield / attention operators (JAX).

Thin wrappers over the shared :mod:`omnibias.struct.jax` ``lse_beta`` path
(:func:`softmax_beta`, :func:`logsumexp_beta`, Jacobian / Hessian). The
public Hopfield names are kept for API stability; the math is not duplicated.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array
from omnibias.struct.jax._logsumexp import (
    logsumexp_beta,
    logsumexp_beta_hessian,
    logsumexp_beta_jacobian,
    softmax_beta,
)


def softmax(a: Array, beta: float = 1.0, *, axis: int = -1) -> Array:
    r"""Numerically stable ``softmax(beta * a)`` along ``axis``."""
    return softmax_beta(a, beta, axis=axis)


def logsumexp_value(a: Array, beta: float = 1.0, *, axis: int = -1) -> Array:
    r"""``lse(beta, a) = (1/beta) log sum_i exp(beta a_i)`` along ``axis`` (stable).

    The ``1/beta`` prefactor makes ``grad lse = softmax(beta a)`` and
    ``hess lse = beta (diag(p) - p p^T)``.
    """
    return logsumexp_beta(a, beta, axis=axis)


def logsumexp_jacobian(a: Array, beta: float = 1.0, *, axis: int = -1) -> Array:
    r"""Gradient of ``lse(beta, a)`` w.r.t. ``a``: ``softmax(beta a)``."""
    return logsumexp_beta_jacobian(a, beta, axis=axis)


def logsumexp_hessian(a: Array, beta: float = 1.0, *, axis: int = -1) -> Array:
    r"""Hessian of ``lse(beta, a)``: ``beta (diag(p) - p p^T)`` with shape ``(..., n, n)``."""
    return logsumexp_beta_hessian(a, beta, axis=axis)


def modern_hopfield_retrieve(xi: Array, X: Array, beta: float = 1.0) -> Array:
    r"""One modern Hopfield retrieval step: ``X^T softmax(beta X xi)``.

    Parameters
    ----------
    xi
        Query / state of shape ``(..., d)``.
    X
        Stored patterns of shape ``(..., n, d)``.
    beta
        Inverse temperature.
    """
    scores = beta * jnp.einsum("...nd,...d->...n", X, xi)
    p = softmax(scores, beta=1.0, axis=-1)
    return jnp.einsum("...n,...nd->...d", p, X)


def hopfield_energy(xi: Array, X: Array, beta: float = 1.0) -> Array:
    r"""Hopfield energy ``E(xi)`` of shape ``(...)``.

    .. math::

        E(\xi) = -\mathrm{lse}(\beta, X\xi) + \tfrac12 \|\xi\|^2 + \tfrac12 M^2,

    where ``\mathrm{lse}(\beta, a) = \beta^{-1}\log\sum_i e^{\beta a_i}``,
    ``M = max_i \|X_i\|``, and the ``\tfrac12 M^2`` constant is included
    so the energy matches the standard Ramsauer et al. form.
    """
    scores = jnp.einsum("...nd,...d->...n", X, xi)
    lse = logsumexp_value(scores, beta=beta, axis=-1)
    quad = 0.5 * jnp.einsum("...d,...d->...", xi, xi)
    pattern_norms = jnp.linalg.norm(X, axis=-1)
    m_sq = 0.5 * jnp.max(pattern_norms, axis=-1) ** 2
    return -lse + quad + m_sq


def attention(
    query: Array,
    keys: Array,
    values: Array,
    beta: float = 1.0,
) -> Array:
    r"""Scaled dot-product attention: ``softmax(beta Q K^T) V``.

    Parameters
    ----------
    query
        Shape ``(..., m, d)``.
    keys, values
        Shape ``(..., n, d)``.
    """
    scores = beta * jnp.einsum("...md,...nd->...mn", query, keys)
    weights = softmax(scores, beta=1.0, axis=-1)
    return jnp.einsum("...mn,...nd->...md", weights, values)
