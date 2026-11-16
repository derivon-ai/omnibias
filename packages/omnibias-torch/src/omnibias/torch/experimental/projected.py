# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""First-layer projections of the Laplacian, divergence and log-determinant for deep MLPs (PyTorch).

The programs of :mod:`omnibias.jax.experimental.projected_laplacian`, ``projected_divergence`` and ``projected_logdet``,
written for tensors. The Laplacian of a deep MLP only needs the first layer through ``W_1 S S^T W_1^T``: when the input
dimension ``D`` exceeds the first width ``H`` the ``D`` tangent columns are replaced by an ``H``-column factor ``F`` with
``F F^T = W_1 S S^T W_1^T``, so the cost no longer grows with ``D``. The divergence and the Sylvester log-determinant
carry the ``H_n`` columns of ``C = W_1 W_L`` through the hidden layers instead of the ``D`` columns of ``W_1``.

Everything is ordinary PyTorch: gradients into the weights, the inputs and higher derivatives follow from autograd.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator, Sequence
from typing import Any

import numpy as np
from omnibias.torch.experimental._chain import (
    affine_readout,
    carry_columns,
    columns,
    distinct_indices,
    fastpath_activation,
    hidden_slopes,
    pairing_index,
    points_2d,
    readout_value,
    require_layers,
)

import torch
from torch import Tensor

__all__ = [
    "capture_safe_branches",
    "input_factor",
    "projected_deep_divergence",
    "projected_deep_laplacian",
    "projected_deep_logdet",
]

_PIVOT_RATIO = 1e-4
_CAPTURE_SAFE = False


@contextlib.contextmanager
def capture_safe_branches() -> Iterator[None]:
    """Inside the block the pivot check of :func:`input_factor` picks its branch on the device instead of on the host.

    The default reads the check back (one host synchronization) and runs only the branch it needs. A CUDA graph cannot
    contain that read, so while one is captured both branches are computed and ``torch.where`` selects: the selected
    values are those the default computes. The block is for values only (the unselected branch's gradient is not
    meaningful), which is all a replayed graph provides.
    """
    global _CAPTURE_SAFE
    previous, _CAPTURE_SAFE = _CAPTURE_SAFE, True
    try:
        yield
    finally:
        _CAPTURE_SAFE = previous


def _first_layer_block(
    W1: Tensor, diffusion: Tensor | None, coordinates: Sequence[int] | np.ndarray | None
) -> Tensor:
    """``W1 S`` with ``S`` restricted to ``coordinates``, without forming a selection matrix."""
    dim = W1.shape[1]
    M = W1
    if coordinates is not None:
        M = columns(W1, distinct_indices(coordinates, dim))
    if diffusion is None:
        return M
    S = torch.as_tensor(diffusion)
    m = M.shape[1]
    if S.ndim == 1 and S.shape[0] == m:
        return M * S
    if S.ndim == 2 and S.shape[0] == m and S.shape[1] >= 1:
        return M @ S
    raise ValueError(f"diffusion must have shape ({m},) or ({m}, r) with r >= 1")


def input_factor(
    W1: Tensor,
    diffusion: Tensor | None = None,
    *,
    coordinates: Sequence[int] | np.ndarray | None = None,
) -> Tensor:
    """Factor ``F`` with ``F F^T = W1 S S^T W1^T`` and ``min(r, H)`` columns.

    For ``r > H`` this is the Cholesky factor of ``G = M M^T``, ``M = W1 S`` (differentiable to every order through
    autograd), or, when ``M`` is rank-deficient or ill-conditioned, ``M Q`` with a Householder basis ``Q`` held
    constant, which keeps values and first derivatives of ``F F^T`` exact.
    """
    W1 = torch.as_tensor(W1)
    if W1.ndim != 2:
        raise ValueError("the first layer weight must have shape (H, D)")
    M = _first_layer_block(W1, diffusion, coordinates)
    H, r = M.shape
    if r <= H:
        return M
    G = M @ M.T
    L, info = torch.linalg.cholesky_ex(G)
    pivots = torch.diagonal(L.detach())
    # one device-to-host read: the three checks are combined on the device instead of read back one by one
    healthy = (info == 0).all() & torch.isfinite(pivots).all() & (torch.min(pivots) > _PIVOT_RATIO * torch.max(pivots))
    if _CAPTURE_SAFE:
        q, _ = torch.linalg.qr(M.detach().T)
        return torch.where(healthy, L, M @ q)
    if bool(healthy):
        return L
    q, _ = torch.linalg.qr(M.detach().T)
    return M @ q


def _point_factor(
    W1: Tensor, s: Tensor, coordinates: Sequence[int] | np.ndarray | None
) -> tuple[Tensor, Tensor]:
    """Per-point ``F_b`` with ``F_b F_b^T = M diag(s_b^2) M^T`` and its diagonal."""
    M = _first_layer_block(W1, None, coordinates)
    H, m = M.shape
    s2 = s * s
    if m <= H:
        return M[None] * s[:, None, :], s2 @ (M * M).T
    G = torch.einsum("bhm,gm->bhg", M[None] * s2[:, None, :], M)
    norms = torch.diagonal(G, dim1=-2, dim2=-1)
    L, info = torch.linalg.cholesky_ex(G)
    pivots = torch.diagonal(L.detach(), dim1=-2, dim2=-1)
    healthy = (
        (info == 0).all()
        & torch.isfinite(pivots).all()
        & (torch.min(pivots, dim=-1).values > _PIVOT_RATIO * torch.max(pivots, dim=-1).values).all()
    )
    if not _CAPTURE_SAFE and bool(healthy):
        return L, norms
    Ms = M[None] * s[:, None, :]
    q, _ = torch.linalg.qr(Ms.detach().transpose(-1, -2))
    return (torch.where(healthy, L, Ms @ q) if _CAPTURE_SAFE else Ms @ q), norms


def projected_deep_laplacian(
    x: Tensor,
    layers: Sequence[Any],
    *,
    diffusion: Tensor | None = None,
    point_diffusion: Tensor | None = None,
    coordinates: Sequence[int] | np.ndarray | None = None,
    directions: Tensor | None = None,
    allow_almost_everywhere: bool = False,
    form: str = "auto",
) -> tuple[Tensor, Tensor, Tensor, Tensor | None]:
    r"""``(u, |S^T grad u|^2, tr(S S^T hess u), v . grad u)`` for a deep MLP (see the JAX program of the same name).

    ``form="auto"`` (also ``"diag"``, ``"gram"``) runs the stacked, factorization-free stream engine
    (:func:`omnibias.torch.experimental.streams.second_order_streams`): one matmul per layer, no Cholesky or QR, and no
    device-to-host synchronization, so the call is capturable in a CUDA graph. ``form="cholesky"`` selects the first-layer
    factorization program (fewer channels at ``r > H``, but it reads a pivot check back to the host).

    ``x`` has shape ``(D,)`` or ``(B, D)``; ``layers`` are ``(W, b, spec)`` with ``W`` of shape ``(out, in)``;
    ``diffusion`` is the factor ``S`` (``(m, r)`` or its diagonal ``(m,)``) of the metric on the selected inputs,
    ``point_diffusion`` a state-dependent diagonal factor ``(B, m)`` (values enter as given), ``coordinates`` the inputs
    the operator acts on and ``directions`` drift directions ``(D, q)`` or ``(B, D, q)``.
    Returns value, squared gradient norm and Laplacian ``(B, C)`` (``(C,)`` for an unbatched ``x``) and the directional
    derivatives ``(B, q, C)`` or ``None``.
    """
    if form != "cholesky":
        from omnibias.torch.experimental.streams import second_order_streams

        r = second_order_streams(
            x, layers, diffusion=diffusion, point_diffusion=point_diffusion, coordinates=coordinates,
            directions=directions, form=form, allow_almost_everywhere=allow_almost_everywhere,
        )
        return r.value, r.grad_sq, r.laplacian, r.directional
    require_layers(layers)
    a, batched = points_2d(x)
    B, dim = a.shape
    W1 = torch.as_tensor(layers[0][0])
    if W1.ndim != 2 or W1.shape[1] != dim:
        raise ValueError(f"the first layer weight must have shape (H, {dim})")
    if point_diffusion is None:
        F = input_factor(W1, diffusion, coordinates=coordinates)
        J = F[None].expand(B, *F.shape)
        norms = torch.sum(F * F, dim=-1)
    else:
        if diffusion is not None:
            raise ValueError("pass either diffusion or point_diffusion, not both")
        m = _first_layer_block(W1, None, coordinates).shape[1]
        s = torch.as_tensor(point_diffusion)
        s = s if batched else s[None]
        if tuple(s.shape) != (B, m):
            shape = f"({B}, {m})" if batched else f"({m},)"
            raise ValueError(f"point_diffusion must have shape {shape}")
        J, norms = _point_factor(W1, s, coordinates)
    L = torch.zeros((B, W1.shape[0]), dtype=a.dtype, device=a.device)
    V: Tensor | None = None
    if directions is not None:
        v = torch.as_tensor(directions)
        if v.ndim == 2 and v.shape[0] == dim:
            V = (W1 @ v)[None].expand(B, W1.shape[0], v.shape[1])
        elif batched and v.ndim == 3 and tuple(v.shape[:2]) == (B, dim):
            V = torch.einsum("hd,bdm->bhm", W1, v)
        else:
            raise ValueError("directions must have shape (D, q) or (B, D, q)")
    for i, (W, b, spec) in enumerate(layers):
        W = torch.as_tensor(W)
        if W.ndim != 2 or W.shape[1] != a.shape[1]:
            raise ValueError(f"layer {i} weight must have shape (out, {a.shape[1]})")
        u = a @ W.T
        if b is not None:
            u = u + torch.as_tensor(b)
        if i:
            J = torch.einsum("oh,bhk->bok", W, J)
            L = L @ W.T
            if V is not None:
                V = torch.einsum("oh,bhm->bom", W, V)
        if spec is None:
            a = u
            if i == 0:
                L = torch.zeros_like(u)
            continue
        resolved = fastpath_activation(spec, allow_almost_everywhere=allow_almost_everywhere)
        s1 = resolved.fastpath(u, 1)
        s2 = resolved.fastpath(u, 2)
        sq = norms if i == 0 else torch.sum(J * J, dim=-1)
        L = s2 * sq + s1 * L
        J = s1[..., None] * J
        if V is not None:
            V = s1[..., None] * V
        a = resolved.forward(u)
    grad_sq = torch.sum(J * J, dim=-1)
    directional = None if V is None else V.transpose(-1, -2)
    if not batched:
        a, grad_sq, L = a[0], grad_sq[0], L[0]
        directional = None if directional is None else directional[0]
    return a, grad_sq, L, directional


def _trace(
    weights: list[Tensor], slopes: list[Tensor], WL: Tensor, index: np.ndarray | None
) -> Tensor:
    """``tr(W_L S_n W_n ... S_1 W_1[:, index])`` per point; ``slopes`` are unit-major ``(H_l, B)``."""
    n = len(weights)
    W1 = columns(weights[0], index)
    if n == 1:
        return torch.sum(W1 * WL.T, dim=1) @ slopes[0]
    if n == 2:
        E = weights[1] * (W1 @ WL).T
        return torch.sum((E @ slopes[0]) * slopes[1], dim=0)
    if (n - 2) * WL.shape[1] < (n - 1) * WL.shape[0]:
        T = carry_columns(weights[:-1], slopes[:-1], lambda: W1 @ WL)
        return torch.einsum("ib,ij,jbi->b", slopes[-1], weights[-1], T)
    T = carry_columns(weights, slopes, W1)
    return torch.einsum("ji,ibj->b", WL, T)


def projected_deep_divergence(
    x: Tensor,
    layers: Sequence[Any],
    *,
    coordinates: Sequence[int] | np.ndarray | None = None,
) -> tuple[Tensor, Tensor]:
    r"""``(s(x), div s(x))`` for a deep MLP vector field; the last layer must be affine.

    ``coordinates`` pairs output ``j`` with input ``coordinates[j]`` (``None``: output ``j`` with input ``j``).
    """
    require_layers(layers)
    a, batched = points_2d(x)
    batch, dim = a.shape
    hidden, WL, bL = affine_readout(layers)
    outputs = WL.shape[0]
    index = pairing_index(coordinates, outputs, dim)
    weights, slopes, act = hidden_slopes(a.T, hidden)
    value = readout_value(WL, bL, act, len(hidden))
    if weights:
        divergence = _trace(weights, slopes, WL, index)
    else:
        total = (
            torch.trace(WL)
            if index is None
            else torch.sum(WL[torch.arange(outputs), torch.as_tensor(index)])
        )
        divergence = total.expand(batch)
    if not batched:
        return value[:, 0], divergence[0]
    return value.T, divergence


def _block(
    weights: list[Tensor], slopes: list[Tensor], WL: Tensor, index: np.ndarray | None
) -> Tensor:
    """Per point, ``J[:, index]`` (``m < H_n``) or ``A C`` (otherwise): ``(B, k, k)`` with equal ``det(I + .)``."""
    W1 = columns(weights[0], index)
    m, width = WL.shape
    seeds = W1 if m < width else W1 @ WL
    T = carry_columns(weights, slopes, seeds)
    if m < width:
        return torch.einsum("ji,ibd->bjd", WL, T)
    return T.permute(1, 0, 2)


def projected_deep_logdet(
    x: Tensor,
    layers: Sequence[Any],
    *,
    coordinates: Sequence[int] | np.ndarray | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    r"""``(s(x), sign, log|det(I + J_s(x))|)`` for a deep MLP vector field by the Sylvester identity."""
    require_layers(layers)
    a, batched = points_2d(x)
    batch, dim = a.shape
    hidden, WL, bL = affine_readout(layers)
    outputs = WL.shape[0]
    index = pairing_index(coordinates, outputs, dim)
    weights, slopes, act = hidden_slopes(a.T, hidden)
    value = readout_value(WL, bL, act, len(hidden))
    if weights:
        block = _block(weights, slopes, WL, index)
    else:
        block = columns(WL, index)[None].expand(batch, outputs, outputs)
    eye = torch.eye(block.shape[-1], dtype=block.dtype, device=block.device)
    sign, logabsdet = torch.linalg.slogdet(eye + block)
    if not batched:
        return value[:, 0], sign[0], logabsdet[0]
    return value.T, sign, logabsdet
