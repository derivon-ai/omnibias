# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Tri-Laplacian of deep MLPs by an outer Laplacian over the projected bi-Laplacian program.

For ``u(x) = f(W_1 x + b_1)`` every rotation-invariant constant-coefficient operator contracts derivatives of ``f`` at
``z = W_1 x + b_1`` with products of ``G = W_1 W_1^T``.  With any ``F`` (``H x k``), ``F F^T = G``, the operator of
``u`` at ``x`` equals the same operator of ``g(y) = f(z + F y)`` at ``y = 0``; in particular

    Delta^3 u (x) = Delta_y [ Delta^2 g ] (y) at y = 0 = sum_i d^2/dt^2 B(z + t F e_i)  at t = 0,

where ``B(z) = Delta_y^2 f(z + F y)|_(y = 0)`` is the bi-Laplacian program of :mod:`projected_bilaplacian` with the
first pre-activation moved to ``z`` and the seed ``F`` held fixed.  The outer Laplacian is a sum of ``k = min(r, H)``
second directional derivatives taken in forward mode (forward over forward) along the columns of ``F``; a Laplacian
over the ``D`` input coordinates (``outer="coordinates"``) would take ``D`` of them.  Both are the same number
because ``sum_j d^2_(W_1 e_j) B = tr(W_1 W_1^T hess_z B) = sum_i d^2_(F e_i) B``.  The carried streams per hidden unit
are those of the bi-Laplacian program with ``k`` columns, so a layer costs ``O(H^2 k^3)`` per point against
``O(H^2 D^3)`` for the same composition seeded on the input.

``F`` is :func:`omnibias.jax.experimental.projected_laplacian.input_factor` (Cholesky factor differentiated to every
order, Householder fallback for rank-deficient or ill-conditioned ``G``), so parameter gradients are exact; they flow
through ``F`` both as the seed and as the outer directions.

Time-dependent fields use ``coordinates`` (the operator acts on those inputs only) and ``directions`` (drift
tangents, one forward pass along ``W_1 v``) exactly as :func:`projected_deep_bilaplacian`; the phase-field-crystal
operator ``u_t - Delta [ (1 + r) u + 2 Delta u + Delta^2 u + u^3 ]`` follows from the returned terms.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from omnibias.jax.experimental._chain import points_2d, require_layers
from omnibias.jax.experimental.projected_bilaplacian import projected_deep_bilaplacian
from omnibias.jax.experimental.projected_laplacian import input_factor

import jax
import jax.numpy as jnp
from jax import Array

__all__ = ["projected_deep_trilaplacian"]

# Elements of the batched per-direction state (about a dozen tensors of the bi-Laplacian program and their forward
# tangents) allowed in flight at once; the outer directions are processed in chunks below this budget.
_STATE_BUDGET = 3.0e8
_STATE_TENSORS = 12


def _chunk_size(batch: int, width: int, k: int, n_dir: int) -> int:
    per_direction = batch * width * (k * k + 3 * k + 3) * _STATE_TENSORS
    return int(max(1, min(n_dir, _STATE_BUDGET // max(per_direction, 1))))


def _second_derivative_sum(f, z: Array, directions: Array, chunk: int) -> Array:  # type: ignore[no-untyped-def]
    """``sum_i d^2/dt^2 f(z + t c_i)`` at ``t = 0`` for the rows ``c_i`` of ``directions``; ``z`` is ``(B, H)``.

    Directions are processed in chunks under ``jax.checkpoint`` so that reverse mode stores the chunk inputs only;
    zero rows pad the last chunk and contribute nothing.
    """
    n = directions.shape[0]
    chunk = max(1, min(chunk, n))
    pad = (-n) % chunk
    if pad:
        directions = jnp.concatenate(
            [directions, jnp.zeros((pad, directions.shape[1]), directions.dtype)]
        )
    blocks = directions.reshape(-1, chunk, directions.shape[1])

    def one(zb: Array, c: Array) -> Array:
        def first(zz: Array) -> Array:
            return jax.jvp(f, (zz,), (c,))[1]  # type: ignore[no-any-return]

        return jax.jvp(first, (zb,), (c,))[1]  # type: ignore[no-any-return]

    def block(c_block: Array) -> Array:
        per_direction = jax.vmap(lambda c: jax.vmap(one, in_axes=(0, None))(z, c))(c_block)
        return jnp.sum(per_direction, axis=0)

    if blocks.shape[0] == 1:
        return block(blocks[0])
    return jnp.sum(jax.lax.map(jax.checkpoint(block), blocks), axis=0)


def projected_deep_trilaplacian(
    x: Array,
    layers: Sequence[tuple[Array, Array | None, object]],
    *,
    coordinates: Sequence[int] | np.ndarray | None = None,
    directions: Array | None = None,
    outer: str = "projected",
    chunk: int | None = None,
    allow_almost_everywhere: bool = False,
) -> tuple[Array, Array, Array, Array, Array, Array | None]:
    r"""``(u, |grad u|^2, Delta u, Delta^2 u, Delta^3 u, v . grad u)`` for a deep MLP.

    Parameters
    ----------
    x : Array of shape ``(D,)`` or ``(B, D)``.
    layers : sequence of ``(W, b, spec)`` as for :func:`projected_deep_bilaplacian` (``spec=None`` is affine).
    coordinates : optional concrete distinct integer indices: the operators act on these inputs only.
    directions : optional drift directions ``(D, q)``, as for :func:`projected_deep_bilaplacian`.
    outer : ``"projected"`` (default) takes the outer Laplacian over the ``k = min(r, H)`` columns of the first-layer
        factor; ``"coordinates"`` takes it over the ``r`` selected inputs (the same value at ``r`` second
        derivatives instead of ``k``, kept as the control of the projection).
    chunk : outer directions processed at once (``None``: the largest number inside a fixed memory budget).
    allow_almost_everywhere : accepted for activations whose derivatives jump, as in the bi-Laplacian.

    Returns
    -------
    value, grad_sq, laplacian, bilaplacian, trilaplacian : Arrays of shape ``(C,)`` or ``(B, C)``.
    directional : ``(q, C)`` or ``(B, q, C)``, or ``None`` without directions.
    """
    require_layers(layers)
    if outer not in ("projected", "coordinates"):
        raise ValueError("outer must be 'projected' or 'coordinates'")
    a, batched = points_2d(x)
    base = projected_deep_bilaplacian(
        a,
        layers,
        coordinates=coordinates,
        directions=directions,
        allow_almost_everywhere=allow_almost_everywhere,  # type: ignore[arg-type]
    )
    value, grad_sq, lap, bilap, directional = base
    W1 = jnp.asarray(layers[0][0])
    b1 = layers[0][1]
    spec1 = layers[0][2]
    rest = [(jnp.asarray(W), b if b is None else jnp.asarray(b), spec) for W, b, spec in layers[1:]]
    F = input_factor(W1, None, coordinates=coordinates)
    if outer == "coordinates":
        cols = np.arange(W1.shape[1]) if coordinates is None else np.asarray(coordinates)
        seeds = W1[:, cols].T
    else:
        seeds = F.T
    z0 = a @ W1.T
    if b1 is not None:
        z0 = z0 + jnp.asarray(b1)
    k = F.shape[1]
    y0 = jnp.zeros((k,), z0.dtype)

    def bilaplacian_of(z: Array) -> Array:
        local = [(F, z, spec1), *rest]
        return projected_deep_bilaplacian(
            y0,
            local,
            allow_almost_everywhere=allow_almost_everywhere,  # type: ignore[arg-type]
        )[3]

    size = chunk if chunk is not None else _chunk_size(a.shape[0], W1.shape[0], k, seeds.shape[0])
    tri = _second_derivative_sum(bilaplacian_of, z0, seeds, size)
    if not batched:
        out = (
            value[0],
            grad_sq[0],
            lap[0],
            bilap[0],
            tri[0],
            None if directional is None else directional[0],
        )
        return out
    return value, grad_sq, lap, bilap, tri, directional
