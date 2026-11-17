# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Second-order operators of deep MLPs through the first-layer projection.

For ``u(x) = f(W_1 x + b_1)`` with ``W_1`` of shape ``(H, D)``, every
second-order quantity of a constant metric ``A = S S^T`` enters through the
first layer only as ``F F^T = W_1 A W_1^T``:

    tr(A hess u),  |S^T grad u|^2.

The forward Laplacian carries the Jacobian of every hidden unit along ``D``
(or ``r`` columns of ``S``) input directions. Any factor ``F`` with
``F F^T = W_1 S S^T W_1^T`` carries the same information, and one with
``k = min(r, H)`` columns exists: ``F = M Q`` for ``M = W_1 S`` and ``Q`` an
orthonormal basis of the row space of ``M``. The hidden recursion then costs
``O(B H^2 k)`` per layer instead of ``O(B H^2 r)``; only products with ``M``
see ``D``.

``Q`` is never formed: with ``G = M M^T = L L^T``, ``Q = M^T L^{-T}`` gives
``F = L``, one ``(H, r) x (r, H)`` product and an ``H x H`` factorization.
``F`` is differentiated as the Cholesky factor of ``G``, so ``F F^T = G``
holds identically in the parameters and derivatives of every order are
exact. When the pivots indicate an ill-conditioned or rank-deficient ``M``,
a Householder QR of ``M^T`` supplies ``Q`` instead and ``Q`` is held
constant: ``Q Q^T`` fixes the rows of ``M``, so values and first
derivatives stay exact, but higher parameter derivatives (Hessian-vector
products) are not. The squared row norms of ``M`` that seed the first-layer
Laplacian are read from ``F F^T = G``, so ``M`` enters the program, and its
gradient, once. Constant or per-point drift directions ``v`` are carried as
separate tangents ``W_1 v`` and return ``v . grad u``. Full input gradients
are not formed; use reverse mode for them.

An operator on a subset of the inputs, such as the space coordinates of a
field that also takes time, is ``S = E`` restricted to those coordinates:
``coordinates`` reads the matching columns of ``W_1`` (a static slice when
they are consecutive) instead of multiplying by an explicit ``(D, m)``
selection matrix, and a one-dimensional ``diffusion`` is a diagonal ``S``,
applied as a column scaling. Neither forms a ``D x D`` or ``D x m`` matrix.

A state-dependent diagonal diffusion ``A(x_b) = diag(s_b)^2`` changes the
first-layer metric per point, ``G_b = M diag(s_b^2) M^T``, so each point gets
its own factor ``F_b`` (a batched ``H x H`` Cholesky) and the hidden
recursion still carries ``min(m, H)`` columns. For batches ``B >= H / 2``
the ``B`` Gram matrices come from one ``(B, m) x (m, H (H + 1) / 2)``
product with the shared rows ``M_im M_jm`` (``i <= j``), half the flops of
the batched ``M diag(s_b^2) M^T``, which smaller batches use.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
from omnibias.jax.experimental._chain import (
    columns,
    distinct_indices,
    fastpath_activation,
    points_2d,
    require_layers,
)

import jax
import jax.numpy as jnp
from jax import Array

if TYPE_CHECKING:  # pragma: no cover
    from omnibias.jax.experimental._chain import LayerSpec

__all__ = ["input_factor", "projected_deep_laplacian"]

_PIVOT_RATIO = 1e-4


@jax.custom_jvp
def _gram(M: Array) -> Array:
    """``M M^T`` whose tangent ``dM M^T + (dM M^T)^T`` is one product."""
    return M @ M.T


@_gram.defjvp
def _gram_jvp(primals: tuple[Array], tangents: tuple[Array]) -> tuple[Array, Array]:
    (M,), (dM,) = primals, tangents
    half = dM @ M.T
    return M @ M.T, half + half.T


@jax.custom_jvp
def _cholesky_of(G: Array, L: Array) -> Array:
    """The supplied Cholesky factor ``L`` of ``G``, differentiated in ``G`` to every order."""
    return L


@_cholesky_of.defjvp
def _cholesky_of_jvp(
    primals: tuple[Array, Array], tangents: tuple[Array, Array]
) -> tuple[Array, Array]:
    G, L = primals
    dG, _ = tangents
    L = _cholesky_of(G, L)
    solve = jax.scipy.linalg.solve_triangular
    X = jnp.swapaxes(solve(L, jnp.swapaxes(solve(L, dG, lower=True), -1, -2), lower=True), -1, -2)
    return L, L @ (jnp.tril(X) - 0.5 * X * jnp.eye(X.shape[-1], dtype=X.dtype))


def input_factor(
    W1: Array,
    diffusion: Array | None = None,
    *,
    coordinates: Sequence[int] | np.ndarray | None = None,
) -> Array:
    """Factor ``F`` with ``F F^T = W1 S S^T W1^T`` and ``min(r, H)`` columns.

    ``diffusion`` is ``S`` of shape ``(m, r)``, or its diagonal of shape
    ``(m,)``, acting on the ``m`` inputs listed in ``coordinates`` (all ``D``
    inputs when ``None``); ``None`` means the identity on them.
    For ``r > H`` this is the Cholesky factor of ``G = M M^T``, ``M = W1 S``,
    differentiated exactly to every order, or, when ``M`` is rank-deficient or
    ill-conditioned, ``M Q`` with a Householder basis ``Q`` held constant,
    which keeps values and first derivatives of ``F F^T`` exact.
    """
    W1 = jnp.asarray(W1)
    if W1.ndim != 2:
        raise ValueError("the first layer weight must have shape (H, D)")
    M = _first_layer_block(W1, diffusion, coordinates)
    H, r = M.shape
    if r <= H:
        return M
    sg = jax.lax.stop_gradient
    G = _gram(M)
    L = jnp.linalg.cholesky(sg(G))
    pivots = jnp.diagonal(L)
    well_conditioned = jnp.all(jnp.isfinite(L)) & (jnp.min(pivots) > _PIVOT_RATIO * jnp.max(pivots))
    # A failed factorization must not reach the unselected branch: under vmap
    # both branches run and NaNs would leak into the gradient.
    eye = jnp.eye(H, dtype=L.dtype)
    G, L = jnp.where(well_conditioned, G, eye), jnp.where(well_conditioned, L, eye)

    def cholesky_basis(operands: tuple[Array, Array, Array, Array | None]) -> Array:
        G, L, _, _ = operands
        out: Array = _cholesky_of(G, L)
        return out

    def householder_basis(operands: tuple[Array, Array, Array, Array | None]) -> Array:
        _, _, W1, diffusion = operands
        M = _first_layer_block(W1, diffusion, coordinates)
        q, _ = jnp.linalg.qr(sg(M).T)
        out: Array = M @ q
        return out

    # The branches see W1 and S rather than M, and the fallback is
    # rematerialized, so that the (H, r) block is neither a materialized
    # operand nor a reverse-mode residual of the conditional.
    S = None if diffusion is None else jnp.asarray(diffusion)
    F: Array = jax.lax.cond(
        well_conditioned, cholesky_basis, jax.checkpoint(householder_basis), (G, L, W1, S)
    )
    return F


def _point_factor(
    W1: Array, s: Array, coordinates: Sequence[int] | np.ndarray | None
) -> tuple[Array, Array]:
    """Per-point ``F_b`` with ``F_b F_b^T = M diag(s_b^2) M^T`` and its diagonal."""
    M = _first_layer_block(W1, None, coordinates)
    H, m = M.shape
    s2 = s * s
    if m <= H:
        return M[None] * s[:, None, :], s2 @ (M * M).T
    if 2 * s.shape[0] >= H:
        iu, ju = np.triu_indices(H)
        index = np.zeros((H, H), dtype=np.int64)
        index[iu, ju] = index[ju, iu] = np.arange(iu.size)
        G = (s2 @ (M[iu] * M[ju]).T)[:, index]
    else:
        G = jnp.einsum("bhm,gm->bhg", M[None] * s2[:, None, :], M)
    norms = jnp.diagonal(G, axis1=-2, axis2=-1)
    sg = jax.lax.stop_gradient
    L = jnp.linalg.cholesky(sg(G))
    pivots = jnp.diagonal(L, axis1=-2, axis2=-1)
    well_conditioned = jnp.all(jnp.isfinite(L)) & jnp.all(
        jnp.min(pivots, axis=-1) > _PIVOT_RATIO * jnp.max(pivots, axis=-1)
    )
    eye = jnp.eye(H, dtype=L.dtype)
    G, L = jnp.where(well_conditioned, G, eye), jnp.where(well_conditioned, L, eye)

    def cholesky_basis(operands: tuple[Array, Array, Array, Array]) -> Array:
        G, L, _, _ = operands
        out: Array = _cholesky_of(G, L)
        return out

    def householder_basis(operands: tuple[Array, Array, Array, Array]) -> Array:
        _, _, W1, s = operands
        Ms = _first_layer_block(W1, None, coordinates)[None] * s[:, None, :]
        q, _ = jnp.linalg.qr(jnp.swapaxes(sg(Ms), -1, -2))
        out: Array = Ms @ q
        return out

    # As in input_factor; here the residuals of an unrematerialized fallback
    # would be the per-point (B, m, H) Householder bases.
    F: Array = jax.lax.cond(
        well_conditioned, cholesky_basis, jax.checkpoint(householder_basis), (G, L, W1, s)
    )
    return F, norms


def _first_layer_block(
    W1: Array, diffusion: Array | None, coordinates: Sequence[int] | np.ndarray | None
) -> Array:
    """``W1 S`` with ``S`` restricted to ``coordinates``, without forming a selection matrix."""
    dim = W1.shape[1]
    M = W1
    if coordinates is not None:
        M = columns(W1, distinct_indices(coordinates, dim))
    if diffusion is None:
        return M
    S = jnp.asarray(diffusion)
    m = M.shape[1]
    if S.ndim == 1 and S.shape[0] == m:
        return M * S
    if S.ndim == 2 and S.shape[0] == m and S.shape[1] >= 1:
        return M @ S
    raise ValueError(f"diffusion must have shape ({m},) or ({m}, r) with r >= 1")


def projected_deep_laplacian(
    x: Array,
    layers: Sequence[LayerSpec],
    *,
    diffusion: Array | None = None,
    point_diffusion: Array | None = None,
    coordinates: Sequence[int] | np.ndarray | None = None,
    directions: Array | None = None,
    allow_almost_everywhere: bool = False,
    form: str = "auto",
) -> tuple[Array, Array, Array, Array | None]:
    r"""``(u, |S^T grad u|^2, tr(S S^T hess u), v . grad u)`` for a deep MLP.

    ``form="auto"`` (also ``"diag"`` and ``"gram"``) runs the stacked, factorization-free stream engine
    (:func:`omnibias.jax.experimental.streams.second_order_streams`): one ``dot_general`` per layer, no conditional, no
    Cholesky or QR, exact to every parameter-derivative order. ``form="cholesky"`` selects the first-layer
    factorization program described in the module docstring (fewer stream channels at ``r > H``; a conditional and a
    Householder fallback; higher parameter derivatives are exact only on its Cholesky branch).

    Parameters
    ----------
    x : Array of shape ``(D,)`` or ``(B, D)``.
    layers : sequence of ``(W, b, spec)``
        As for :func:`omnibias.jax.laplacian.deep_field_value_grad_laplacian`:
        ``W`` has shape ``(out, in)``; ``spec=None`` is an affine layer.
    diffusion : optional factor ``S`` of the metric ``A = S S^T`` on the
        selected inputs, of shape ``(m, r)``, or its diagonal of shape
        ``(m,)``; ``None`` gives the Laplacian and the Euclidean gradient norm
        in those inputs.
    point_diffusion : optional state-dependent diagonal factor ``s(x_b)`` of
        shape ``(B, m)`` (``(m,)`` for an unbatched ``x``): the metric at
        point ``b`` is ``diag(s_b)^2`` on the selected inputs, e.g.
        ``s_b = sigma x_b / sqrt(2)`` for Black-Scholes. Its values enter as
        given; derivatives of ``s`` with respect to ``x`` are not taken.
        Exclusive with ``diffusion``.
    coordinates : optional concrete distinct integer indices of shape
        ``(m,)``: the operator acts on these inputs only, for example the
        space coordinates of a field that also takes time. ``None`` selects
        all ``D`` inputs.
    directions : optional drift directions ``(D, q)`` shared by all points or
        ``(B, D, q)`` per point (``(D, q)`` for an unbatched ``x``), over all
        ``D`` inputs.

    Returns
    -------
    value, grad_sq, laplacian : Arrays of shape ``(C,)`` or ``(B, C)``.
    directional : ``(q, C)`` or ``(B, q, C)``, or ``None`` without directions.
    """
    if form != "cholesky":
        from omnibias.jax.experimental.streams import second_order_streams

        r = second_order_streams(
            x, layers, diffusion=diffusion, point_diffusion=point_diffusion, coordinates=coordinates,
            directions=directions, form=form, allow_almost_everywhere=allow_almost_everywhere,
        )
        return r.value, r.grad_sq, r.laplacian, r.directional
    require_layers(layers)
    a, batched = points_2d(x)
    B, dim = a.shape
    W1 = jnp.asarray(layers[0][0])
    if W1.ndim != 2 or W1.shape[1] != dim:
        raise ValueError(f"the first layer weight must have shape (H, {dim})")
    if point_diffusion is None:
        F = input_factor(W1, diffusion, coordinates=coordinates)
        J = jnp.broadcast_to(F[None], (B, *F.shape))
        norms = jnp.sum(F * F, axis=-1)
    else:
        if diffusion is not None:
            raise ValueError("pass either diffusion or point_diffusion, not both")
        m = _first_layer_block(W1, None, coordinates).shape[1]
        s = jnp.asarray(point_diffusion)
        s = s if batched else s[None]
        if s.shape != (B, m):
            shape = f"({B}, {m})" if batched else f"({m},)"
            raise ValueError(f"point_diffusion must have shape {shape}")
        J, norms = _point_factor(W1, s, coordinates)
    L = jnp.zeros((B, W1.shape[0]), dtype=a.dtype)
    V: Array | None = None
    if directions is not None:
        v = jnp.asarray(directions)
        if v.ndim == 2 and v.shape[0] == dim:
            V = jnp.broadcast_to((W1 @ v)[None], (B, W1.shape[0], v.shape[1]))
        elif batched and v.ndim == 3 and v.shape[:2] == (B, dim):
            V = jnp.einsum("hd,bdm->bhm", W1, v)
        else:
            raise ValueError("directions must have shape (D, q) or (B, D, q)")
    for i, (W, b, spec) in enumerate(layers):
        W = jnp.asarray(W)
        if W.ndim != 2 or W.shape[1] != a.shape[1]:
            raise ValueError(f"layer {i} weight must have shape (out, {a.shape[1]})")
        u = a @ W.T
        if b is not None:
            u = u + jnp.asarray(b)
        if i:
            J = jnp.einsum("oh,bhk->bok", W, J)
            L = L @ W.T
            if V is not None:
                V = jnp.einsum("oh,bhm->bom", W, V)
        if spec is None:
            a = u
            if i == 0:
                L = jnp.zeros_like(u)
            continue
        resolved = fastpath_activation(spec, allow_almost_everywhere=allow_almost_everywhere)
        s1 = resolved.fastpath(u, 1)
        s2 = resolved.fastpath(u, 2)
        sq = norms if i == 0 else jnp.sum(J * J, axis=-1)
        L = s2 * sq + s1 * L
        J = s1[..., None] * J
        if V is not None:
            V = s1[..., None] * V
        a = resolved.forward(u)
    grad_sq = jnp.sum(J * J, axis=-1)
    directional = None if V is None else jnp.swapaxes(V, -1, -2)
    if not batched:
        a, grad_sq, L = a[0], grad_sq[0], L[0]
        directional = None if directional is None else directional[0]
    return a, grad_sq, L, directional
