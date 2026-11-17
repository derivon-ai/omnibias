# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Linear solves with the Jacobian of residual MLP blocks through the Woodbury identity (PyTorch).

For a vector field ``s(x) = W_L a_n + b_L`` the residual block ``x + s(x)`` has the Jacobian ``I_D + W_L A W_1`` with
``A = S_n W_n ... W_2 S_1`` and ``S_l = diag sigma_l'(z_l)``. Woodbury's identity moves the ``D x D`` system to the
narrowest layer,

    (I_D + W_L A W_1)^{-1} g = g - W_L (I_{H_n} + A C)^{-1} A W_1 g,        C = W_1 W_L,

so a solve costs one ``H_n x H_n`` factorization per point instead of a ``D x D`` one. This is the PyTorch counterpart of
:mod:`omnibias.jax.experimental.residual_woodbury`; gradients follow from autograd (the implicit derivative of
``torch.linalg.solve``).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from omnibias.torch.experimental._chain import (
    affine_readout,
    carry_columns,
    hidden_slopes,
    readout_value,
)

import torch
from torch import Tensor

__all__ = ["residual_inverse_newton", "residual_solve", "residual_solve_transpose"]


def _slopes(x: Tensor, layers: Sequence[Any]) -> tuple[list[Tensor], list[Tensor], Tensor, Tensor]:
    hidden, WL, bL = affine_readout(layers, require_hidden=True)
    weights, slopes, a = hidden_slopes(x.T, hidden)
    return weights, slopes, WL, readout_value(WL, bL, a, len(hidden))


def _apply_A(weights: list[Tensor], slopes: list[Tensor], v: Tensor) -> Tensor:
    v = slopes[0] * v
    for W, s in zip(weights[1:], slopes[1:], strict=True):
        v = s * (W @ v)
    return v


def _apply_At(weights: list[Tensor], slopes: list[Tensor], q: Tensor) -> Tensor:
    w = slopes[-1] * q
    for layer in range(len(weights) - 1, 0, -1):
        w = slopes[layer - 1] * (weights[layer].T @ w)
    return w


def _system(weights: list[Tensor], slopes: list[Tensor], WL: Tensor) -> Tensor:
    """``I + A C`` per point, ``(B, H_n, H_n)``, or ``I + J``, ``(B, D, D)``, when ``D < H_n``."""
    m, width = WL.shape
    seeds = weights[0] if m < width else weights[0] @ WL
    T = carry_columns(weights, slopes, seeds)
    block = torch.einsum("ji,ibd->bjd", WL, T) if m < width else T.permute(1, 0, 2)
    return torch.eye(block.shape[-1], dtype=block.dtype, device=block.device) + block


def _solve(
    weights: list[Tensor],
    slopes: list[Tensor],
    WL: Tensor,
    K: Tensor,
    g: Tensor,
    transpose: bool,
) -> Tensor:
    if WL.shape[0] < WL.shape[1]:
        Km = K.transpose(-1, -2) if transpose else K
        return torch.linalg.solve(Km, g[..., None])[..., 0]
    if not transpose:
        v = _apply_A(weights, slopes, weights[0] @ g.T)
        q = torch.linalg.solve(K, v.T[..., None])[..., 0]
        return g - (WL @ q.T).T
    t = WL.T @ g.T
    q = torch.linalg.solve(K.transpose(-1, -2), t.T[..., None])[..., 0]
    w = _apply_At(weights, slopes, q.T)
    return g - (weights[0].T @ w).T


def _check(x: Tensor, WL: Tensor) -> None:
    if WL.shape[0] != x.shape[1]:
        raise ValueError(f"the field has {WL.shape[0]} outputs for {x.shape[1]} inputs")


def residual_solve(
    x: Tensor, layers: Sequence[Any], rhs: Tensor, *, transpose: bool = False
) -> tuple[Tensor, Tensor]:
    r"""``(s(x), y)`` with ``(I_D + J_s(x))^{(T)} y = rhs`` for the residual block ``x + s(x)``.

    ``x`` and ``rhs`` have shape ``(D,)`` or ``(B, D)``; the last layer must be affine and map to ``D`` outputs.
    """
    x = torch.as_tensor(x)
    rhs = torch.as_tensor(rhs)
    if rhs.shape != x.shape:
        raise ValueError("rhs must have the shape of x")
    batched = x.ndim > 1
    a, g = (x, rhs) if batched else (x[None, :], rhs[None, :])
    if a.ndim != 2:
        raise ValueError("x must have shape (D,) or (B, D)")
    weights, slopes, WL, value = _slopes(a, layers)
    _check(a, WL)
    K = _system(weights, slopes, WL)
    y = _solve(weights, slopes, WL, K, g, transpose)
    if not batched:
        return value[:, 0], y[0]
    return value.T, y


def residual_solve_transpose(
    x: Tensor, layers: Sequence[Any], rhs: Tensor
) -> tuple[Tensor, Tensor]:
    """``residual_solve(x, layers, rhs, transpose=True)``."""
    return residual_solve(x, layers, rhs, transpose=True)


def residual_inverse_newton(
    y: Tensor,
    layers: Sequence[Any],
    *,
    tol: float = 1e-12,
    max_iter: int = 30,
    x0: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Solve ``x + s(x) = y`` by Newton's method with one Woodbury solve per step.

    Returns ``(x, steps, residual)`` as the JAX program does. The iteration is not differentiated through.
    """
    y = torch.as_tensor(y)
    batched = y.ndim > 1
    target = y if batched else y[None, :]
    if target.ndim != 2:
        raise ValueError("y must have shape (D,) or (B, D)")
    start = target if x0 is None else torch.as_tensor(x0)
    start = start if batched or x0 is None else start[None, :]
    with torch.no_grad():
        x = start
        weights, slopes, WL, value = _slopes(x, layers)
        _check(x, WL)
        r = x + value.T - target
        res = torch.amax(torch.abs(r), dim=-1)
        it = 0
        while float(torch.amax(res)) > tol and it < max_iter:
            K = _system(weights, slopes, WL)
            delta = _solve(weights, slopes, WL, K, r, False)
            x = torch.where((res > tol)[:, None], x - delta, x)
            weights, slopes, WL, value = _slopes(x, layers)
            r = x + value.T - target
            res = torch.amax(torch.abs(r), dim=-1)
            it += 1
    steps = torch.tensor(it, dtype=torch.int32, device=x.device)
    if not batched:
        return x[0], steps, res[0]
    return x, steps, res
