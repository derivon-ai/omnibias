# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Modern Hopfield / attention operators (torch).

Thin wrappers over the shared :mod:`omnibias.struct.torch` ``lse_beta`` path
(:func:`softmax_beta`, :func:`logsumexp_beta`, Jacobian / Hessian). The
public Hopfield names are kept for API stability; the math is not duplicated.
"""

from __future__ import annotations

import torch
from omnibias.struct.torch._logsumexp import (
    logsumexp_beta,
    logsumexp_beta_hessian,
    logsumexp_beta_jacobian,
    softmax_beta,
)
from torch import Tensor


def softmax(a: Tensor, beta: float = 1.0, *, axis: int = -1) -> Tensor:
    r"""Numerically stable ``softmax(beta * a)`` along ``axis``."""
    return softmax_beta(a, beta, axis=axis)


def logsumexp_value(a: Tensor, beta: float = 1.0, *, axis: int = -1) -> Tensor:
    r"""``lse(beta, a) = (1/beta) log sum_i exp(beta a_i)`` along ``axis`` (stable).

    The ``1/beta`` prefactor makes ``grad lse = softmax(beta a)`` and
    ``hess lse = beta (diag(p) - p p^T)``.
    """
    return logsumexp_beta(a, beta, axis=axis)


def logsumexp_jacobian(a: Tensor, beta: float = 1.0, *, axis: int = -1) -> Tensor:
    r"""Gradient of ``lse(beta, a)`` w.r.t. ``a``: ``softmax(beta a)``."""
    return logsumexp_beta_jacobian(a, beta, axis=axis)


def logsumexp_hessian(a: Tensor, beta: float = 1.0, *, axis: int = -1) -> Tensor:
    r"""Hessian of ``lse(beta, a)``: ``beta (diag(p) - p p^T)`` with shape ``(..., n, n)``."""
    return logsumexp_beta_hessian(a, beta, axis=axis)


def modern_hopfield_retrieve(xi: Tensor, X: Tensor, beta: float = 1.0) -> Tensor:
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
    scores = beta * torch.einsum("...nd,...d->...n", X, xi)
    p = softmax(scores, beta=1.0, axis=-1)
    return torch.einsum("...n,...nd->...d", p, X)


def hopfield_energy(xi: Tensor, X: Tensor, beta: float = 1.0) -> Tensor:
    r"""Hopfield energy ``E(xi)`` of shape ``(...)``.

    .. math::

        E(\xi) = -\mathrm{lse}(\beta, X\xi) + \tfrac12 \|\xi\|^2 + \tfrac12 M^2,

    where ``\mathrm{lse}(\beta, a) = \beta^{-1}\log\sum_i e^{\beta a_i}``,
    ``M = max_i \|X_i\|``, and the ``\tfrac12 M^2`` constant is included
    so the energy matches the standard Ramsauer et al. form.
    """
    scores = torch.einsum("...nd,...d->...n", X, xi)
    lse = logsumexp_value(scores, beta=beta, axis=-1)
    quad = 0.5 * torch.einsum("...d,...d->...", xi, xi)
    pattern_norms = torch.linalg.norm(X, dim=-1)
    m_sq = 0.5 * pattern_norms.amax(dim=-1).square()
    return -lse + quad + m_sq


def attention(
    query: Tensor,
    keys: Tensor,
    values: Tensor,
    beta: float = 1.0,
) -> Tensor:
    r"""Scaled dot-product attention: ``softmax(beta Q K^T) V``.

    Parameters
    ----------
    query
        Shape ``(..., m, d)``.
    keys, values
        Shape ``(..., n, d)``.
    """
    scores = beta * torch.einsum("...md,...nd->...mn", query, keys)
    weights = softmax(scores, beta=1.0, axis=-1)
    return torch.einsum("...mn,...nd->...md", weights, values)
