# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Gram matrices of PDE residual gradients for deep MLP fields, without the Jacobian (PyTorch).

The full-gradient stream program of :mod:`omnibias.jax.experimental.residual_gram`. For a pointwise residual
``r_b = rho(u(x_b), grad u(x_b), lap u(x_b), x_b)`` of a deep MLP, every dense layer ``z = W a + b`` maps the ``T``
forward-Laplacian streams of its input at a point (value, first derivatives, Laplacian) by the same ``W``, so the gradient
of ``r_b`` in ``W`` is ``Q_b P_b^T`` with ``P_b`` the ``(in, T)`` input streams and ``Q_b`` the ``(out, T)`` cotangents of the
output streams (one reverse pass per residual component). Then

.. math::

    \langle \partial_W r_b, \partial_W r_c \rangle = \sum_{t, s} (P_b^T P_c)_{ts} (Q_b^T Q_c)_{ts},

which replaces ``in * out`` per pair of rows by ``T (T + 1) / 2 (in + out)``. Hidden layers use it where it is cheaper by
a calibrated margin; the first layer, the readout and narrow layers form per-row gradients from the same cotangents.

The result equals ``J J^T`` up to rounding. Gram matrices and steps are returned as values: they are not differentiable
in the parameters (the stream cotangents are formed with detached weights).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from omnibias.core.demand import Quantities, Residual, block_order
from omnibias.torch.experimental._chain import distinct_indices, fastpath_activation
from omnibias.torch.experimental.conditioning import ConditioningPolicy, damped_solve

import torch
from torch import Tensor

__all__ = [
    "deep_residual_gauss_newton",
    "deep_residual_gram",
    "deep_residual_ntk_diagonal",
    "dual_step",
    "gram_with_streams",
    "lower_full",
]

_STREAMS_COST = 2.2
_DIAG_COST = 1.0
_DIAG_CHUNK = 1 << 25
_FULL_ENGINE = frozenset({"value", "grad", "grad_sq", "lap"})


def lower_full(residual: Residual, order: int, weights: Tensor) -> Callable[..., Tensor]:
    """The ``(u, grad, lap, x)`` callback of the full-gradient engine; ``weights`` is the metric ``s^2`` per input."""
    extra = residual.demand.quantities - _FULL_ENGINE
    if extra:
        raise ValueError(
            f"the full-gradient engine cannot provide {sorted(extra)}; use the first-layer projected engine"
        )
    want_sq = "grad_sq" in residual.demand

    def callback(u: Tensor, grad: Tensor | None, lap: Tensor | None, x: Tensor) -> Tensor:
        grad_sq = None
        if want_sq and grad is not None:
            grad_sq = torch.einsum("bkc,k->bc", grad * grad, weights)
        return residual(Quantities(u=u, x=x, grad=grad, grad_sq=grad_sq, lap=lap))

    return callback


def _activate(S: Tensor, spec: Any, order: int, w: Tensor, allow_ae: bool) -> Tensor:
    if spec is None:
        return S
    act = fastpath_activation(spec, allow_almost_everywhere=allow_ae, smooth=2 if order else 0)
    z = S[:, 0]
    a = act.forward(z)[:, None]
    if order == 0:
        return a
    D = w.shape[0]
    Jz = S[:, 1 : D + 1]
    s1 = act.fastpath(z, 1)
    parts = [a, s1[:, None] * Jz]
    if order == 2:
        L = s1 * S[:, D + 1] + act.fastpath(z, 2) * torch.einsum("bkh,k->bh", Jz * Jz, w)
        parts.append(L[:, None])
    return torch.cat(parts, dim=1)


def _forward(
    layers: Sequence[tuple[Tensor, Tensor | None, Any]],
    x: Tensor,
    order: int,
    w: Tensor,
    eps: list[Tensor],
    allow_ae: bool,
) -> tuple[Tensor, list[Tensor]]:
    """Output streams ``(B, T, C)`` and the input streams of every layer after the first."""
    B, D = x.shape
    inputs: list[Tensor] = []
    A = x
    for i, (W, b, spec) in enumerate(layers):
        if i == 0:
            parts = [(x @ W.T)[:, None]]
            if order >= 1:
                parts.append(W.T[None].expand(B, D, W.shape[0]))
            if order == 2:
                parts.append(torch.zeros((B, 1, W.shape[0]), dtype=x.dtype, device=x.device))
            S = torch.cat(parts, dim=1)
        else:
            inputs.append(A)
            S = A @ W.T
        if b is not None:
            S = torch.cat([S[:, :1] + b[None, None, :], S[:, 1:]], dim=1)
        A = _activate(S + eps[i], spec, order, w, allow_ae)
    return A, inputs


def _streams(
    layers: Sequence[tuple[Tensor, Tensor | None, Any]],
    x: Tensor,
    residual: Callable[..., Tensor],
    order: int,
    w: Tensor,
    allow_ae: bool,
) -> tuple[Tensor, list[Tensor], list[Tensor]]:
    """Residuals ``(B, R)``, layer input streams ``(B, T, in)`` and output-stream cotangents ``(B, R, T, out)``."""
    B = x.shape[0]
    D = w.shape[0]
    T = (1, 1 + D, 2 + D)[order]
    with torch.enable_grad():
        eps = [
            torch.zeros((B, T, W.shape[0]), dtype=x.dtype, device=x.device, requires_grad=True)
            for W, _, _ in layers
        ]
        out, inputs = _forward(layers, x, order, w, eps, allow_ae)
        g = out[:, 1 : D + 1] if order >= 1 else None
        lap = out[:, D + 1] if order == 2 else None
        r = torch.as_tensor(residual(out[:, 0], g, lap, x))
        if r.ndim not in (1, 2) or r.shape[0] != B:
            raise ValueError(f"residual must return shape ({B},) or ({B}, R), got {tuple(r.shape)}")
        r2 = r.reshape(B, 1) if r.ndim == 1 else r
        R = r2.shape[1]
        cols: list[list[Tensor]] = []
        for k in range(R):
            grads = torch.autograd.grad(r2[:, k].sum(), eps, retain_graph=k < R - 1)
            cols.append(list(grads))
    Qs = [torch.stack([cols[k][li] for k in range(R)], dim=1) for li in range(len(layers))]
    return r2.detach(), [p.detach() for p in inputs], Qs


def _first_jacobian(x: Tensor, Q: Tensor, order: int, bias: bool) -> Tensor:
    B, R, _, H = Q.shape
    D = x.shape[1]
    qv = Q[:, :, 0]
    G = qv[..., None] * x[:, None, None, :]
    if order >= 1:
        G = G + Q[:, :, 1 : D + 1].transpose(-1, -2)
    J = G.reshape(B * R, H * D)
    return torch.cat([J, qv.reshape(B * R, H)], dim=1) if bias else J


def _explicit_jacobian(P: Tensor, Q: Tensor, bias: bool) -> Tensor:
    B, R = Q.shape[:2]
    J = torch.einsum("brto,bti->broi", Q, P).reshape(B * R, -1)
    return torch.cat([J, Q[:, :, 0].reshape(B * R, -1)], dim=1) if bias else J


def _streams_gram(Pi: Tensor, Qi: Tensor, Pj: Tensor, Qj: Tensor, same: bool) -> Tensor:
    """``sum_{t,s} (P_t P_s^T) o (Q_t Q_s^T)`` between two blocks, one stream ``t`` against all ``s`` at a time."""
    Bi, Ri, Ti, H = Qi.shape
    Bj, Rj, Tj, _ = Qj.shape
    Pit, Pjt = Pi.transpose(0, 1), Pj.transpose(0, 1)
    Qit = Qi.permute(2, 0, 1, 3).reshape(Ti, Bi * Ri, H)
    Qjt = Qj.permute(2, 0, 1, 3).reshape(Tj, Bj * Rj, H)
    A = torch.zeros((Bi * Ri, Bj * Rj), dtype=Qi.dtype, device=Qi.device)
    for t in range(Ti):
        s0 = t if same else 0
        n = Tj - s0
        gp = (Pit[t] @ Pjt[s0:].reshape(n * Bj, -1).T).reshape(Bi, n, Bj)
        gq = (Qit[t] @ Qjt[s0:].reshape(n * Bj * Rj, H).T).reshape(Bi, Ri, n, Bj, Rj)
        if same:
            gp = torch.cat([gp[:, :1] * 0.5, gp[:, 1:]], dim=1)
        A = A + torch.einsum("bsc,brscq->brcq", gp, gq).reshape(Bi * Ri, Bj * Rj)
    return A + A.T if same else A


def _metric_weights(
    D: int,
    coordinates: Sequence[int] | np.ndarray | None,
    diffusion: Tensor | None,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    idx = distinct_indices(
        coordinates, D, message=f"coordinates must be distinct integers in [0, {D})"
    )
    if diffusion is None:
        s2 = torch.ones(idx.size, dtype=dtype, device=device)
    else:
        s2 = torch.as_tensor(diffusion, dtype=dtype, device=device) ** 2
    if tuple(s2.shape) != (idx.size,):
        raise ValueError(f"diffusion must have shape ({idx.size},)")
    w = torch.zeros(D, dtype=dtype, device=device)
    w[torch.as_tensor(idx, device=device)] = s2
    return w


def _prepare(
    blocks: Sequence[Any],
    layers: Sequence[tuple[Tensor, Tensor | None, Any]],
    coordinates: Sequence[int] | np.ndarray | None,
    diffusion: Tensor | None,
    strategy: str,
) -> tuple[list[tuple[Tensor, Callable[..., Tensor], int]], Tensor, torch.dtype]:
    if not layers:
        raise ValueError("layers must be non-empty")
    if strategy not in ("auto", "streams", "explicit"):
        raise ValueError("strategy must be 'auto', 'streams' or 'explicit'")
    norm: list[tuple[Tensor, Callable[..., Tensor] | Residual, int]] = []
    for blk in blocks:
        x, res, *rest = blk
        given = int(rest[0]) if rest else None
        order = (
            block_order(res, given, "full")
            if isinstance(res, Residual)
            else (2 if given is None else given)
        )
        if order not in (0, 1, 2):
            raise ValueError("order must be 0, 1 or 2")
        x = torch.as_tensor(x)
        if x.ndim != 2:
            raise ValueError("each block's points must have shape (B, D)")
        norm.append((x, res, order))
    if not norm:
        raise ValueError("blocks must be non-empty")
    D = norm[0][0].shape[1]
    if any(x.shape[1] != D for x, _, _ in norm):
        raise ValueError("every block must have the same input dimension")
    W0 = torch.as_tensor(layers[0][0])
    if W0.ndim != 2 or W0.shape[1] != D:
        raise ValueError(f"the first layer weight must have shape (out, {D})")
    dtype = torch.promote_types(norm[0][0].dtype, W0.dtype)
    w = _metric_weights(D, coordinates, diffusion, dtype, W0.device)
    lowered = [
        (x, lower_full(res, order, w) if isinstance(res, Residual) else res, order)
        for x, res, order in norm
    ]
    return lowered, w, dtype


def _detached(
    layers: Sequence[tuple[Tensor, Tensor | None, Any]],
) -> list[tuple[Tensor, Tensor | None, Any]]:
    return [
        (
            torch.as_tensor(W).detach(),
            None if b is None else torch.as_tensor(b).detach(),
            s,
        )
        for W, b, s in layers
    ]


def _assemble(
    norm: list[tuple[Tensor, Callable[..., Tensor], int]],
    data: list[tuple[Tensor, list[Tensor], list[Tensor]]],
    layers: Sequence[tuple[Tensor, Tensor | None, Any]],
    strategy: str,
    first_gram: Callable[[int, int], Tensor],
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor]:
    sizes = [r.numel() for r, _, _ in data]
    device = data[0][0].device
    K = [[torch.zeros((ni, nj), dtype=dtype, device=device) for nj in sizes] for ni in sizes]
    nb = len(data)
    for li, (W, b, _) in enumerate(layers):
        bias = b is not None
        if li == 0:
            for i in range(nb):
                for j in range(i, nb):
                    K[i][j] = K[i][j] + first_gram(i, j)
            continue
        out, inp = W.shape
        cache: dict[int, Tensor] = {}
        for i in range(nb):
            for j in range(i, nb):
                Pi, Qi = data[i][1][li - 1], data[i][2][li]
                Pj, Qj = data[j][1][li - 1], data[j][2][li]
                Ti, Tj = Qi.shape[2], Qj.shape[2]
                pairs = Ti * (Ti + 1) // 2 if i == j else Ti * Tj
                use = strategy == "streams" or (
                    strategy == "auto" and _STREAMS_COST * pairs * (inp + out) < inp * out
                )
                if use:
                    G = _streams_gram(Pi, Qi, Pj, Qj, i == j)
                    if bias:
                        G = G + Qi[:, :, 0].reshape(-1, out) @ Qj[:, :, 0].reshape(-1, out).T
                    K[i][j] = K[i][j] + G
                else:
                    for k, (P, Q) in ((i, (Pi, Qi)), (j, (Pj, Qj))):
                        if k not in cache:
                            cache[k] = _explicit_jacobian(P, Q, bias)
                    K[i][j] = K[i][j] + cache[i] @ cache[j].T
    rows = [
        torch.cat([K[i][j] if j >= i else K[j][i].T for j in range(nb)], dim=1) for i in range(nb)
    ]
    return torch.cat([r.reshape(-1) for r, _, _ in data]), torch.cat(rows, dim=0)


def gram_with_streams(
    blocks: Sequence[Any],
    layers: Sequence[tuple[Tensor, Tensor | None, Any]],
    coordinates: Sequence[int] | np.ndarray | None,
    diffusion: Tensor | None,
    strategy: str,
    allow_ae: bool,
) -> tuple[
    Tensor,
    Tensor,
    list[tuple[Tensor, Callable[..., Tensor], int]],
    list[tuple[Tensor, list[Tensor], list[Tensor]]],
]:
    layers = _detached(layers)
    norm, w, dtype = _prepare(blocks, layers, coordinates, diffusion, strategy)
    norm = [(x.detach(), res, order) for x, res, order in norm]
    data = [_streams(layers, x, res, order, w, allow_ae) for x, res, order in norm]
    bias0 = layers[0][1] is not None
    J0 = [
        _first_jacobian(x, Q[0], order, bias0)
        for (x, _, order), (_, _, Q) in zip(norm, data, strict=True)
    ]
    r, K = _assemble(norm, data, layers, strategy, lambda i, j: J0[i] @ J0[j].T, dtype)
    return r, K, norm, data


def deep_residual_gram(
    blocks: Sequence[Any],
    layers: Sequence[tuple[Tensor, Tensor | None, Any]],
    *,
    coordinates: Sequence[int] | np.ndarray | None = None,
    diffusion: Tensor | None = None,
    strategy: str = "auto",
    allow_almost_everywhere: bool = False,
) -> tuple[Tensor, Tensor]:
    r"""Residuals ``r`` and ``K = J J^T``, ``J = dr / d theta`` over every weight and bias in ``layers``.

    ``blocks`` are ``(x, residual)`` or ``(x, residual, order)`` with ``x`` of shape ``(B, D)``; the residual is a
    :class:`~omnibias.core.demand.Residual` or an old ``residual(u, grad, lap, x)`` callback returning ``(B,)`` or
    ``(B, R)`` and acting pointwise. ``order`` (default 2) is the highest derivative used (``0``: ``grad = lap = None``).
    ``coordinates`` and ``diffusion`` define the Laplacian ``sum_k s_k^2 d^2 u / dx_k^2`` over the listed inputs.
    ``strategy`` chooses per hidden layer between the stream identity and explicit per-row gradients.
    Returns ``r`` of shape ``(N,)`` (rows ordered by block, point, component) and ``K`` of shape ``(N, N)``.
    """
    r, K, _, _ = gram_with_streams(
        blocks, layers, coordinates, diffusion, strategy, allow_almost_everywhere
    )
    return r, K


def deep_residual_gauss_newton(
    blocks: Sequence[Any],
    layers: Sequence[tuple[Tensor, Tensor | None, Any]],
    damping: float | Tensor,
    *,
    coordinates: Sequence[int] | np.ndarray | None = None,
    diffusion: Tensor | None = None,
    strategy: str = "auto",
    allow_almost_everywhere: bool = False,
    conditioning: ConditioningPolicy | None = None,
) -> tuple[Tensor, Tensor, list[tuple[Tensor, Tensor | None]]]:
    r"""Gauss-Newton step in dual form, ``step = J^T (J J^T + damping I)^{-1} r``; the update is ``theta - step``.

    Returns ``r``, ``K`` and ``step``, a list of ``(dW, db)`` with the shapes of the weights and biases of ``layers``
    (``db`` is ``None`` where the bias is ``None``). A ``conditioning`` policy of mode ``"floor"`` raises ``damping``.
    """
    r, K, norm, data = gram_with_streams(
        blocks, layers, coordinates, diffusion, strategy, allow_almost_everywhere
    )
    alpha, _ = damped_solve(K, r, damping, conditioning)
    return r, K, dual_step(alpha, norm, data, layers)


def dual_step(
    alpha: Tensor,
    norm: list[tuple[Tensor, Callable[..., Tensor], int]],
    data: list[tuple[Tensor, list[Tensor], list[Tensor]]],
    layers: Sequence[tuple[Tensor, Tensor | None, Any]],
) -> list[tuple[Tensor, Tensor | None]]:
    """``J^T alpha`` as per-layer ``(dW, db)`` from the stream cotangents of :func:`_gram`."""
    offsets = np.cumsum([0] + [rb.numel() for rb, _, _ in data])
    a = [
        alpha[o0:o1].reshape(rb.shape)
        for o0, o1, (rb, _, _) in zip(offsets[:-1], offsets[1:], data, strict=True)
    ]
    step: list[tuple[Tensor, Tensor | None]] = []
    for li, (W, b, _) in enumerate(layers):
        dW = torch.zeros(W.shape, dtype=alpha.dtype, device=alpha.device)
        db = torch.zeros(W.shape[0], dtype=alpha.dtype, device=alpha.device)
        for ab, (x, _, order), (_, P, Q) in zip(a, norm, data, strict=True):
            Qa = torch.einsum("br,brto->bto", ab, Q[li])
            if li == 0:
                D = x.shape[1]
                dW = dW + Qa[:, 0].T @ x
                if order >= 1:
                    dW = dW + torch.sum(Qa[:, 1 : D + 1], dim=0).T
            else:
                dW = dW + torch.einsum("bto,bti->oi", Qa, P[li - 1])
            db = db + torch.sum(Qa[:, 0], dim=0)
        step.append((dW, None if b is None else db))
    return step


def _row_chunks(fn: Callable[..., Tensor], per_row: int, chunk: int, *arrays: Tensor) -> Tensor:
    B = arrays[0].shape[0]
    c = max(1, chunk // max(per_row, 1))
    if c >= B:
        return fn(*arrays)
    return torch.cat([fn(*[a[i : i + c] for a in arrays]) for i in range(0, B, c)], dim=0)


def _layer_diag(P: Tensor, Q: Tensor, bias: bool, strategy: str, chunk: int) -> Tensor:
    """``|dr_{br}/dW|^2`` (+ bias) of one dense layer from input streams ``(B, T, in)`` and cotangents ``(B, R, T, out)``."""
    B, R, T, out = Q.shape
    inp = P.shape[2]
    use = strategy == "streams" or (
        strategy == "auto" and _DIAG_COST * T * (inp + R * out) < R * inp * out
    )
    if use:
        gp = torch.einsum("bti,bsi->bts", P, P)
        gq = torch.einsum("brto,brso->brts", Q, Q)
        d = torch.einsum("bts,brts->br", gp, gq)
    else:

        def rows(Pc: Tensor, Qc: Tensor) -> Tensor:
            G = torch.einsum("brto,bti->broi", Qc, Pc)
            return torch.sum(G * G, dim=(2, 3))

        d = _row_chunks(rows, R * out * inp, chunk, P, Q)
    return d + torch.sum(Q[:, :, 0] ** 2, dim=-1) if bias else d


def deep_residual_ntk_diagonal(
    blocks: Sequence[Any],
    layers: Sequence[tuple[Tensor, Tensor | None, Any]],
    *,
    coordinates: Sequence[int] | np.ndarray | None = None,
    diffusion: Tensor | None = None,
    strategy: str = "auto",
    allow_almost_everywhere: bool = False,
) -> tuple[Tensor, Tensor]:
    r"""Residuals ``r`` and the diagonal of ``K = J J^T``, ``|dr_b / d theta|^2`` per row, without ``K`` or ``J``."""
    layers = _detached(layers)
    norm, w, _ = _prepare(blocks, layers, coordinates, diffusion, strategy)
    rs, ds = [], []
    for x, res, order in norm:
        x = x.detach()
        r, P, Q = _streams(layers, x, res, order, w, allow_almost_everywhere)
        d: Tensor | float = 0.0
        for li, (_, b, _) in enumerate(layers):
            bias = b is not None
            if li == 0:
                H = Q[0].shape[-1]
                per_row = Q[0].shape[1] * H * (x.shape[1] + 1)

                def first(xc: Tensor, Qc: Tensor, order: int = order, bias: bool = bias) -> Tensor:
                    Bc, Rc = Qc.shape[:2]
                    J = _first_jacobian(xc, Qc, order, bias)
                    return torch.sum(J * J, dim=-1).reshape(Bc, Rc)

                d = d + _row_chunks(first, per_row, _DIAG_CHUNK, x, Q[0])
            else:
                d = d + _layer_diag(P[li - 1], Q[li], bias, strategy, _DIAG_CHUNK)
        rs.append(r.reshape(-1))
        ds.append(torch.as_tensor(d).reshape(-1))
    return torch.cat(rs), torch.cat(ds)


def deep_residual_values(
    blocks: Sequence[Any],
    layers: Sequence[tuple[Tensor, Tensor | None, Any]],
    *,
    coordinates: Sequence[int] | np.ndarray | None = None,
    diffusion: Tensor | None = None,
    allow_almost_everywhere: bool = False,
) -> Tensor:
    r"""The residual vector ``r`` alone: one forward stream pass, no cotangents, no Jacobian, no ``K``.

    This is the cheap evaluation a line search or a Levenberg-Marquardt acceptance test needs after a trial step.
    """
    layers = _detached(layers)
    norm, w, _ = _prepare(blocks, layers, coordinates, diffusion, "auto")
    parts: list[Tensor] = []
    with torch.no_grad():
        for x, res, order in norm:
            x = x.detach()
            B, D = x.shape
            T = (1, 1 + D, 2 + D)[order]
            eps = [
                torch.zeros((B, T, W.shape[0]), dtype=x.dtype, device=x.device)
                for W, _, _ in layers
            ]
            out, _ = _forward(layers, x, order, w, eps, allow_almost_everywhere)
            g = out[:, 1 : D + 1] if order >= 1 else None
            lap = out[:, D + 1] if order == 2 else None
            parts.append(torch.as_tensor(res(out[:, 0], g, lap, x)).reshape(-1))
    return torch.cat(parts)


def deep_residual_gn_diagonal(
    blocks: Sequence[Any],
    layers: Sequence[tuple[Tensor, Tensor | None, Any]],
    *,
    coordinates: Sequence[int] | np.ndarray | None = None,
    diffusion: Tensor | None = None,
    allow_almost_everywhere: bool = False,
) -> tuple[Tensor, list[tuple[Tensor, Tensor | None]]]:
    r"""Residuals ``r`` and the per-*parameter* Gauss-Newton diagonal ``sum_n (d r_n / d theta_i)^2``.

    Returns ``(r, [(dW, db), ...])`` with the shapes of the weights and biases (``db`` is ``None`` where the bias is).
    The per-row gradients come from the stream cotangents in row chunks, so no nested graph and no per-row backward
    pass is built; it is the exact quantity ``J.pow(2).sum(0)`` without storing ``J``.
    """
    layers = _detached(layers)
    norm, w, _ = _prepare(blocks, layers, coordinates, diffusion, "auto")
    rs: list[Tensor] = []
    acc: list[tuple[Tensor, Tensor | None]] = [
        (
            torch.zeros_like(W),
            None if b is None else torch.zeros_like(b),
        )
        for W, b, _ in layers
    ]
    for x, res, order in norm:
        x = x.detach()
        r, P, Q = _streams(layers, x, res, order, w, allow_almost_everywhere)
        rs.append(r.reshape(-1))
        for li, (W, b, _) in enumerate(layers):
            out, inp = W.shape
            bias = b is not None
            R = Q[li].shape[1]
            dW, db = acc[li]
            if li == 0:
                D = x.shape[1]

                def rows(xc: Tensor, Qc: Tensor, order: int = order, out: int = out) -> Tensor:
                    return _first_jacobian(xc, Qc, order, False).reshape(-1, out, xc.shape[1])

                G = _row_chunks_sum(rows, R * out * D, x, Q[0])
            else:

                def rows(Pc: Tensor, Qc: Tensor, out: int = out) -> Tensor:
                    return torch.einsum("brto,bti->broi", Qc, Pc).reshape(-1, out, Pc.shape[-1])

                G = _row_chunks_sum(rows, R * out * inp, P[li - 1], Q[li])
            acc[li] = (dW + G, db)
            if bias:
                qb = torch.sum(Q[li][:, :, 0] ** 2, dim=(0, 1))
                acc[li] = (acc[li][0], acc[li][1] + qb)  # type: ignore[operator]
    return torch.cat(rs), acc


def _row_chunks_sum(fn: Callable[..., Tensor], per_row: int, *arrays: Tensor) -> Tensor:
    """``sum`` over rows of ``fn(...)**2`` for row chunks of a ``(B, ...)`` batch (``fn`` returns ``(rows, out, in)``)."""
    B = arrays[0].shape[0]
    c = max(1, _DIAG_CHUNK // max(per_row, 1))
    total: Tensor | None = None
    for i in range(0, B, c):
        G = fn(*[a[i : i + c] for a in arrays])
        part = torch.sum(G * G, dim=0)
        total = part if total is None else total + part
    assert total is not None
    return total
