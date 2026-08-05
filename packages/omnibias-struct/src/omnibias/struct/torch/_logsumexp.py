# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Closed-form log-sum-exp / softmax primitives and jets (torch).

Bit-identical twin of :mod:`omnibias.struct.jax._logsumexp`. The soft-DP combine is
``lse_beta``; its exact derivatives are the closed-form omnibias tower, **not** autodiff:

- ``grad lse_beta = softmax(beta a)`` (:func:`logsumexp_beta_jacobian`);
- ``hess lse_beta = beta (diag(p) - p p^T)`` (:func:`logsumexp_beta_hessian`);
- and, pairwise, ``lse_beta(a, b) = a + beta^-1 softplus(beta (b - a))`` whose whole
  Taylor jet is the beta-tempered ``softplus`` tower ``softplus^(n) = sigma^(n-1)`` from
  :mod:`omnibias.core`, propagated with :func:`omnibias.torch.jet.compose_jet`
  (:func:`pairwise_lse_jet`). This is the ``delta -> 0`` engine that differentiates the
  ``beta -> inf`` relaxation.
"""

from __future__ import annotations

import torch
from omnibias.torch.activations.registry import get_activation
from omnibias.torch.jet import compose_jet
from torch import Tensor


def logsumexp_beta(a: Tensor, beta: float = 1.0, *, axis: int = -1) -> Tensor:
    r"""Stable ``lse_beta(a) = beta^-1 log sum_i exp(beta a_i)`` along ``axis``."""
    scaled = beta * a
    max_a = scaled.amax(dim=axis, keepdim=True)
    log_sum = max_a.squeeze(axis) + torch.log(torch.exp(scaled - max_a).sum(dim=axis))
    return log_sum / beta


def softmax_beta(a: Tensor, beta: float = 1.0, *, axis: int = -1) -> Tensor:
    r"""Stable ``softmax(beta a)`` along ``axis`` -- the gradient of :func:`logsumexp_beta`."""
    scaled = beta * a
    max_a = scaled.amax(dim=axis, keepdim=True)
    exp_a = torch.exp(scaled - max_a)
    return exp_a / exp_a.sum(dim=axis, keepdim=True)


def logsumexp_beta_jacobian(a: Tensor, beta: float = 1.0, *, axis: int = -1) -> Tensor:
    r"""Closed-form gradient of ``lse_beta`` w.r.t. ``a``: ``softmax(beta a)``."""
    return softmax_beta(a, beta, axis=axis)


def logsumexp_beta_hessian(a: Tensor, beta: float = 1.0, *, axis: int = -1) -> Tensor:
    r"""Closed-form Hessian of ``lse_beta``: ``beta (diag(p) - p p^T)`` (``(..., n, n)``)."""
    p = softmax_beta(a, beta, axis=axis)
    outer = p.unsqueeze(-1) * p.unsqueeze(-2)
    diag = torch.diag_embed(p)
    return beta * (diag - outer)


def pairwise_lse(a: Tensor, b: Tensor, beta: float = 1.0) -> Tensor:
    r"""Elementwise soft-max combine ``lse_beta(a, b) = beta^-1 log(e^{beta a} + e^{beta b})``.

    Equal to ``a + beta^-1 softplus(beta (b - a))``; computed by the stable symmetric
    log-sum-exp of the stacked pair.
    """
    stacked = torch.stack([a, b], dim=-1)
    return logsumexp_beta(stacked, beta, axis=-1)


def pairwise_lse_jet(
    a0: Tensor,
    b0: Tensor,
    db: Tensor,
    beta: float = 1.0,
    order: int = 1,
) -> Tensor:
    r"""Taylor jet (order ``order``) of ``t -> lse_beta(a0, b0 + t db)`` from the tower.

    Writing ``lse_beta(a0, b0 + t db) = a0 + beta^-1 softplus(u(t))`` with
    ``u(t) = beta (b0 - a0) + (beta db) t``, the jet is
    :func:`omnibias.torch.jet.compose_jet` of the linear ``u``-jet with the closed-form
    ``softplus`` derivative tower ``softplus^(k)(u_0)`` (``softplus^(k) = sigma^(k-1)``
    from :mod:`omnibias.core`) -- exact, no autodiff or finite differences. Returns
    coefficients ``c_k = f^(k)(0) / k!`` along a new leading axis of length ``order + 1``.
    """
    if order < 0:
        raise ValueError(f"order must be >= 0, got {order}")
    spec = get_activation("softplus")
    fastpath = spec.fastpath
    if fastpath is None:  # defensive: softplus always ships its closed-form tower
        raise NotImplementedError("softplus is missing its closed-form fastpath tower")
    a0 = torch.as_tensor(a0)
    b0 = torch.as_tensor(b0)
    db = torch.as_tensor(db)
    u0 = beta * (b0 - a0)
    zero = torch.zeros_like(u0)
    u_rows = [u0]
    if order >= 1:
        u_rows.append((beta * db) + zero)
    u_rows.extend(zero for _ in range(order - 1))
    u_jet = torch.stack(u_rows[: order + 1], dim=0)
    tower_rows = [spec.forward(u0)]
    for k in range(1, order + 1):
        tower_rows.append(fastpath(u0, k))
    sigma_tower = torch.stack(tower_rows, dim=0)
    sp_jet = compose_jet(u_jet, sigma_tower) / beta
    out0 = (sp_jet[0] + a0).unsqueeze(0)
    return torch.cat([out0, sp_jet[1:]], dim=0)


__all__ = [
    "logsumexp_beta",
    "logsumexp_beta_hessian",
    "logsumexp_beta_jacobian",
    "pairwise_lse",
    "pairwise_lse_jet",
    "softmax_beta",
]
