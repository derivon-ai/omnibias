# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Bi- and tri-Laplacian of deep MLPs through the first-layer projection (PyTorch counterpart of the JAX programs).

For ``u(x) = f(W_1 x + b_1)`` and a constant metric ``A = S S^T`` every rotation-invariant constant-coefficient operator
contracts derivatives of ``f`` at ``z = W_1 x + b_1`` with products of ``G = W_1 A W_1^T`` only, so it equals the same
operator of ``g(y) = f(z + F y)`` at ``y = 0`` for any ``F`` with ``F F^T = G``
(:func:`~omnibias.torch.experimental.projected.input_factor` gives one with ``k = min(r, H)`` columns).

:func:`projected_deep_bilaplacian` carries, per hidden unit, the gradient (``k``), the Hessian (``k x k``), the Laplacian, the
gradient of the Laplacian and the bi-Laplacian along ``y`` from closed-form activation derivatives up to order four: every
layer costs ``O(B H^2 k^2)`` instead of ``O(B H^2 D^2)`` on the input. The Hessians are stored densely; the packed
storages of the JAX program are a memory optimization this port does not repeat. Every operation is a plain
differentiable tensor expression, so parameter derivatives of every order are exact and nothing synchronizes with the
host.

:func:`projected_deep_trilaplacian` takes the outer Laplacian as ``k`` second directional derivatives of the
bi-Laplacian program, in forward-over-forward mode along the columns of ``F``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from omnibias.torch.experimental._chain import fastpath_activation, points_2d, require_layers
from omnibias.torch.experimental.projected import input_factor
from omnibias.torch.experimental.streams import activation_tower, has_row_tower

import torch
from torch import Tensor

__all__ = ["projected_deep_bilaplacian", "projected_deep_trilaplacian"]


def _matvec(T: Tensor, g: Tensor, k: int) -> Tensor:
    return torch.einsum("...ij,...j->...i", T.reshape(*T.shape[:-1], k, k), g)


def _outer(g: Tensor) -> Tensor:
    return (g[..., :, None] * g[..., None, :]).reshape(*g.shape[:-1], g.shape[-1] * g.shape[-1])


def projected_deep_bilaplacian(
    x: Tensor,
    layers: Sequence[Any],
    *,
    diffusion: Tensor | None = None,
    coordinates: Sequence[int] | np.ndarray | None = None,
    directions: Tensor | None = None,
    allow_almost_everywhere: bool = False,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor | None]:
    r"""``(u, |S^T grad u|^2, tr(A hess u), tr(A hess)^2 u, v . grad u)`` for a deep MLP, ``A = S S^T``.

    Arguments and results are those of the JAX program of the same name: ``x`` is ``(D,)`` or ``(B, D)``; ``layers`` are
    ``(W, b, spec)`` with ``W`` of shape ``(out, in)`` (``spec=None`` affine); ``diffusion`` the factor ``S``;
    ``coordinates`` the inputs the operator acts on; ``directions`` drift directions ``(D, q)`` or ``(B, D, q)``. Returns
    value, squared gradient norm, Laplacian and bi-Laplacian ``(B, C)`` (``(C,)`` for an unbatched ``x``) and the
    directional derivatives ``(B, q, C)`` or ``None``.
    """
    require_layers(layers)
    a, batched = points_2d(x)
    B, dim = a.shape
    W1 = torch.as_tensor(layers[0][0])
    if W1.ndim != 2 or W1.shape[1] != dim:
        raise ValueError(f"the first layer weight must have shape (H, {dim})")
    F = input_factor(W1, diffusion, coordinates=coordinates)
    k = F.shape[1]
    V: Tensor | None = None
    if directions is not None:
        v = torch.as_tensor(directions)
        if v.ndim == 2 and v.shape[0] == dim:
            V = (W1 @ v)[:, None, :]
        elif batched and v.ndim == 3 and tuple(v.shape[:2]) == (B, dim):
            V = torch.einsum("hd,bdq->hbq", W1, v)
        else:
            raise ValueError("directions must have shape (D, q) or (B, D, q)")
    a = a.T  # unit-major layout (unit, point, ...)
    state: tuple[Tensor, Tensor, Tensor, Tensor, Tensor] | None = None
    for i, (W, b, spec) in enumerate(layers):
        W = torch.as_tensor(W)
        if W.ndim != 2 or W.shape[1] != a.shape[0]:
            raise ValueError(f"layer {i} weight must have shape (out, {a.shape[0]})")
        z = W @ a
        if b is not None:
            z = z + torch.as_tensor(b)[:, None]
        lower: tuple[Tensor, Tensor, Tensor, Tensor] | None
        if state is None:
            gz = F[:, None, :]
            n = torch.sum(F * F, dim=-1)[:, None]
            lower = None
        else:
            J, T, L, P, Q = state
            gz = torch.tensordot(W, J, dims=1)
            n = torch.sum(gz * gz, dim=-1)
            lower = (
                torch.tensordot(W, T, dims=1),
                W @ L,
                torch.tensordot(W, P, dims=1),
                W @ Q,
            )
            if V is not None:
                V = torch.tensordot(W, V, dims=1)
        if spec is None:
            if lower is None:
                zero = torch.zeros_like(z)
                gz = gz.expand(*z.shape, k)
                lower = (
                    torch.zeros((*z.shape, k * k), dtype=z.dtype, device=z.device),
                    zero,
                    torch.zeros_like(gz),
                    zero,
                )
            state = (gz, *lower)
            a = z
            continue
        resolved = fastpath_activation(
            spec, allow_almost_everywhere=allow_almost_everywhere, smooth=4
        )
        value: Tensor | None = None
        if has_row_tower(resolved):  # one base evaluation, not five (the forward value and four derivatives)
            value, s1, s2, s3, s4 = activation_tower(resolved, z, 4)
        else:
            try:
                s1, s2, s3, s4 = (resolved.fastpath(z, order) for order in (1, 2, 3, 4))
            except NotImplementedError as exc:
                raise ValueError(
                    f"activation {resolved.name!r} fastpath does not support order 4, required for the bi-Laplacian"
                ) from exc
        outer = _outer(gz)
        if lower is None:
            T = s2[..., None] * outer
            L = s2 * n
            P = (s3 * n)[..., None] * gz
            Q = s4 * n * n
        else:
            Hz, Lz, Pz, Qz = lower
            Hg = _matvec(Hz, gz, k)
            T = s1[..., None] * Hz + s2[..., None] * outer
            L = s1 * Lz + s2 * n
            P = (
                s1[..., None] * Pz
                + s2[..., None] * (Lz[..., None] * gz + 2.0 * Hg)
                + (s3 * n)[..., None] * gz
            )
            Q = (
                s1 * Qz
                + s2
                * (4.0 * torch.sum(gz * Pz, dim=-1) + Lz * Lz + 2.0 * torch.sum(Hz * Hz, dim=-1))
                + s3 * (2.0 * n * Lz + 4.0 * torch.sum(gz * Hg, dim=-1))
                + s4 * n * n
            )
        state = (s1[..., None] * gz, T, L, P, Q)
        if V is not None:
            V = s1[..., None] * V
        a = resolved.forward(z) if value is None else value
    assert state is not None
    value, L, Q = a.T, state[2].T, state[4].T
    grad_sq = torch.sum(state[0] * state[0], dim=-1).T
    directional = None
    if V is not None:
        directional = V.expand(V.shape[0], B, V.shape[2]).permute(1, 2, 0)
    if not batched:
        return value[0], grad_sq[0], L[0], Q[0], None if directional is None else directional[0]
    return value, grad_sq, L, Q, directional


def projected_deep_trilaplacian(
    x: Tensor,
    layers: Sequence[Any],
    *,
    coordinates: Sequence[int] | np.ndarray | None = None,
    directions: Tensor | None = None,
    outer: str = "projected",
    chunk: int | None = None,
    allow_almost_everywhere: bool = False,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor | None]:
    r"""``(u, |grad u|^2, Delta u, Delta^2 u, Delta^3 u, v . grad u)`` for a deep MLP.

    ``outer="projected"`` takes the outer Laplacian over the ``k = min(r, H)`` columns of the first-layer factor,
    ``"coordinates"`` over the ``r`` selected inputs (the same value at ``r`` second derivatives instead of ``k``).
    ``chunk`` bounds the outer directions evaluated at once (``None``: all). Other arguments as for
    :func:`projected_deep_bilaplacian`.
    """
    require_layers(layers)
    if outer not in ("projected", "coordinates"):
        raise ValueError("outer must be 'projected' or 'coordinates'")
    a, batched = points_2d(x)
    value, grad_sq, lap, bilap, directional = projected_deep_bilaplacian(
        a,
        layers,
        coordinates=coordinates,
        directions=directions,
        allow_almost_everywhere=allow_almost_everywhere,
    )
    W1 = torch.as_tensor(layers[0][0])
    b1, spec1 = layers[0][1], layers[0][2]
    rest = [
        (torch.as_tensor(W), b if b is None else torch.as_tensor(b), spec)
        for W, b, spec in layers[1:]
    ]
    F = input_factor(W1, None, coordinates=coordinates)
    if outer == "coordinates":
        cols = np.arange(W1.shape[1]) if coordinates is None else np.asarray(coordinates)
        seeds = W1[:, cols].T
    else:
        seeds = F.T
    z0 = a @ W1.T
    if b1 is not None:
        z0 = z0 + torch.as_tensor(b1)
    y0 = torch.zeros((F.shape[1],), dtype=z0.dtype, device=z0.device)

    def bilaplacian_of(z: Tensor) -> Tensor:
        local = [(F, z, spec1), *rest]
        return projected_deep_bilaplacian(
            y0, local, allow_almost_everywhere=allow_almost_everywhere
        )[3]

    def second(z: Tensor, c: Tensor) -> Tensor:
        def first(zz: Tensor) -> Tensor:
            return torch.func.jvp(bilaplacian_of, (zz,), (c,))[1]

        return torch.func.jvp(first, (z,), (c,))[1]

    per_direction = torch.func.vmap(
        lambda c: torch.func.vmap(second, in_dims=(0, None))(z0, c),
        chunk_size=chunk,
    )(seeds)
    tri = per_direction.sum(dim=0)
    if not batched:
        return (
            value[0],
            grad_sq[0],
            lap[0],
            bilap[0],
            tri[0],
            None if directional is None else directional[0],
        )
    return value, grad_sq, lap, bilap, tri, directional
