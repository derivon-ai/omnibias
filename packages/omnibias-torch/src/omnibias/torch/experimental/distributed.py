# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Data-parallel residual Gram, Gauss-Newton step and NTK diagonal over ``torch.distributed`` ranks.

Every rank passes the blocks of its **own** points (the same number of blocks, the same residuals and orders, and the same
number of points per block on every rank) and the same replicated weights. Each rank forms the forward-Laplacian streams of
its points, all-gathers the compact per-layer factors of the others (input streams ``P`` of shape ``(B, T, in)``, output
cotangents ``Q`` of shape ``(B, R, T, out)`` and the explicit first-layer rows), and fills its slab ``K[local rows, all
rows]`` with the stream identity at ``1/world_size`` of the arithmetic. ``K`` is then gathered and the damped solve runs on
every rank (``N^2`` memory and ``N^3 / 3`` arithmetic per rank, **not** divided by the world size); ``J^T alpha`` is formed
from local rows and summed with an all-reduce. The values equal the single-process program's up to rounding; rows are
ordered by block, then rank, then point, then residual component, so a block that is the concatenation of the ranks' points
in rank order gives the single-process ordering exactly.

Limits as for the JAX module :mod:`omnibias.jax.experimental.distributed`: pointwise residuals, first-layer rows gathered
(``N`` rows of ``H (D + 1)`` numbers), no distributed version of the projected program for ``D > H``. Autograd does not
flow through these programs (they return values and steps, not differentiable losses); use DDP on a loss for that.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from omnibias.torch.experimental import residual_gram as rg
from omnibias.torch.experimental.conditioning import ConditioningPolicy, damped_solve

import torch
import torch.distributed as dist
from torch import Tensor

__all__ = [
    "distributed_residual_gauss_newton",
    "distributed_residual_gram",
    "distributed_residual_ntk_diagonal",
]


def _gather(a: Tensor, group: Any) -> Tensor:
    """All-gather equal-shaped ``a`` along dimension 0 in rank order."""
    world = dist.get_world_size(group)
    out = torch.empty((world * a.shape[0], *a.shape[1:]), dtype=a.dtype, device=a.device)
    dist.all_gather_into_tensor(out, a.contiguous(), group=group)
    return out


def _gather_stack(a: Tensor, group: Any) -> Tensor:
    """All-gather ``a`` into shape ``(world, *a.shape)``."""
    world = dist.get_world_size(group)
    return _gather(a, group).reshape(world, *a.shape)


def _rows_in_block_order(stacked: Tensor, sizes: Sequence[int]) -> Tensor:
    world = stacked.shape[0]
    out, off = [], 0
    for s in sizes:
        out.append(stacked[:, off : off + s].reshape(world * s, *stacked.shape[2:]))
        off += s
    return torch.cat(out, dim=0)


@torch.no_grad()
def _local(
    blocks: Sequence[Any],
    layers: Sequence[Any],
    group: Any,
    coordinates: Any,
    diffusion: Any,
    strategy: str,
    allow_ae: bool,
):
    layers = rg._detached(layers)
    norm, w, dtype = rg._prepare(blocks, layers, coordinates, diffusion, strategy)
    norm = [(x.detach(), res, order) for x, res, order in norm]
    data = [rg._streams(layers, x, res, order, w, allow_ae) for x, res, order in norm]
    nb = len(data)
    world = dist.get_world_size(group)
    sizes = [r.numel() for r, _, _ in data]
    glob = [s * world for s in sizes]
    device = data[0][0].device
    bias0 = layers[0][1] is not None
    J0 = [
        rg._first_jacobian(x, Q[0], order, bias0)
        for (x, _, order), (_, _, Q) in zip(norm, data, strict=True)
    ]
    J0_all = [_gather(j, group) for j in J0]
    slab = [
        [torch.zeros((sizes[i], glob[j]), dtype=dtype, device=device) for j in range(nb)]
        for i in range(nb)
    ]
    for i in range(nb):
        for j in range(nb):
            slab[i][j] = slab[i][j] + J0[i] @ J0_all[j].T
    for li in range(1, len(layers)):
        W, b, _ = layers[li]
        bias = b is not None
        out, inp = W.shape
        P_all = [_gather(data[j][1][li - 1], group) for j in range(nb)]
        Q_all = [_gather(data[j][2][li], group) for j in range(nb)]
        explicit_all: dict[int, Tensor] = {}
        for i in range(nb):
            Pi, Qi = data[i][1][li - 1], data[i][2][li]
            for j in range(nb):
                Pj, Qj = P_all[j], Q_all[j]
                pairs = Qi.shape[2] * Qj.shape[2]
                use = strategy == "streams" or (
                    strategy == "auto" and rg._STREAMS_COST * pairs * (inp + out) < inp * out
                )
                if use:
                    G = rg._streams_gram(Pi, Qi, Pj, Qj, False)
                    if bias:
                        G = G + Qi[:, :, 0].reshape(-1, out) @ Qj[:, :, 0].reshape(-1, out).T
                else:
                    if j not in explicit_all:
                        explicit_all[j] = rg._explicit_jacobian(Pj, Qj, bias)
                    G = rg._explicit_jacobian(Pi, Qi, bias) @ explicit_all[j].T
                slab[i][j] = slab[i][j] + G
    slab_rows = torch.cat([torch.cat(slab[i], dim=1) for i in range(nb)], dim=0)
    return layers, norm, data, sizes, glob, slab_rows


@torch.no_grad()
def distributed_residual_gram(
    blocks: Sequence[Any],
    layers: Sequence[Any],
    *,
    group: Any = None,
    coordinates: Any = None,
    diffusion: Any = None,
    strategy: str = "auto",
    allow_almost_everywhere: bool = False,
) -> tuple[Tensor, Tensor]:
    """``(r, K)`` of :func:`~omnibias.torch.experimental.residual_gram.deep_residual_gram` over the ranks' points.

    ``blocks`` holds this rank's points of each block; ``r`` and ``K`` are the same on every rank.
    """
    _, _, data, sizes, _, slab = _local(
        blocks, layers, group, coordinates, diffusion, strategy, allow_almost_everywhere
    )
    r_local = torch.cat([r.reshape(-1) for r, _, _ in data])
    r = _rows_in_block_order(_gather_stack(r_local, group), sizes)
    K = _rows_in_block_order(_gather_stack(slab, group), sizes)
    return r, K


@torch.no_grad()
def distributed_residual_gauss_newton(
    blocks: Sequence[Any],
    layers: Sequence[Any],
    damping: float | Tensor,
    *,
    group: Any = None,
    coordinates: Any = None,
    diffusion: Any = None,
    strategy: str = "auto",
    allow_almost_everywhere: bool = False,
    conditioning: ConditioningPolicy | None = None,
) -> tuple[Tensor, Tensor, list[tuple[Tensor, Tensor | None]]]:
    """``(r, K, step)`` of the single-process Gauss-Newton step over the ranks' points; the step is the same on every rank."""
    layers_d, norm, data, sizes, glob, slab = _local(
        blocks, layers, group, coordinates, diffusion, strategy, allow_almost_everywhere
    )
    r_local = torch.cat([r.reshape(-1) for r, _, _ in data])
    r = _rows_in_block_order(_gather_stack(r_local, group), sizes)
    K = _rows_in_block_order(_gather_stack(slab, group), sizes)
    alpha, _ = damped_solve(K, r, damping, conditioning)
    me = dist.get_rank(group)
    offsets = np.cumsum([0] + glob)
    mine = [
        alpha[offsets[k] : offsets[k + 1]][me * sizes[k] : (me + 1) * sizes[k]]
        for k in range(len(sizes))
    ]
    step = rg.dual_step(torch.cat(mine), norm, data, layers_d)
    out: list[tuple[Tensor, Tensor | None]] = []
    for dW, db in step:
        dist.all_reduce(dW, group=group)
        if db is not None:
            dist.all_reduce(db, group=group)
        out.append((dW, db))
    return r, K, out


@torch.no_grad()
def distributed_residual_ntk_diagonal(
    blocks: Sequence[Any],
    layers: Sequence[Any],
    *,
    group: Any = None,
    coordinates: Any = None,
    diffusion: Any = None,
    strategy: str = "auto",
    allow_almost_everywhere: bool = False,
) -> tuple[Tensor, Tensor]:
    """``(r, diag K)`` over the ranks' points; each rank computes its rows and the rows are gathered (no other communication)."""
    rs, ds = [], []
    for blk in blocks:
        r, d = rg.deep_residual_ntk_diagonal(
            [blk],
            layers,
            coordinates=coordinates,
            diffusion=diffusion,
            strategy=strategy,
            allow_almost_everywhere=allow_almost_everywhere,
        )
        rs.append(_gather(r.reshape(-1), group))
        ds.append(_gather(d.reshape(-1), group))
    return torch.cat(rs), torch.cat(ds)
