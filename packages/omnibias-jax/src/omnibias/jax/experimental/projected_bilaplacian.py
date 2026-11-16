# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Bi-Laplacian of deep MLPs through the first-layer projection.

For ``u(x) = f(W_1 x + b_1)`` and a constant metric ``A = S S^T``, the
operator ``(tr(A hess))^2 u`` (the bi-Laplacian for ``A = I``) contracts
derivatives of ``f`` at ``z = W_1 x + b_1`` with products of
``G = W_1 A W_1^T`` only. It therefore equals the bi-Laplacian of
``g(y) = f(F y + z)`` at ``y = 0`` for any ``F`` with ``F F^T = G``;
:func:`~omnibias.jax.experimental.projected_laplacian.input_factor` gives one
with ``k = min(r, H)`` columns and exact parameter gradients.

The forward recursion carries, for every hidden unit, its gradient (``k``),
Hessian (``k x k``), Laplacian, the gradient of its Laplacian (``k``) and its
bi-Laplacian along ``y``, using closed-form activation derivatives up to
order four. Each layer costs ``O(B H^2 k^2)`` instead of the ``O(B H^2 D^2)``
of the same recursion on the input. From ``k = 8`` the symmetric Hessians are
stored packed (upper triangle, or lower block triangle for larger ``k``),
which nearly halves the dominant products and the memory traffic on them.

The carried gradients also give ``|S^T grad u|^2``, and drift directions
``v`` are carried as first-order tangents ``W_1 v`` returning ``v . grad u``.
Together with ``coordinates`` these assemble time-dependent fourth-order
residuals such as Kuramoto–Sivashinsky (``u_t + Delta^2 u + Delta u +
|grad u|^2 / 2``) or Cahn–Hilliard with ``Delta f(u) = f'(u) Delta u +
f''(u) |grad u|^2``. Unused outputs cost nothing under ``jax.jit``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
from omnibias.jax.experimental._chain import (
    fastpath_activation,
    packed_triangle,
    points_2d,
    require_layers,
)
from omnibias.jax.experimental.projected_laplacian import input_factor

import jax
import jax.numpy as jnp
from jax import Array

if TYPE_CHECKING:  # pragma: no cover
    from omnibias.jax.experimental._chain import LayerSpec

__all__ = ["projected_deep_bilaplacian"]


class _SymmetricFull:
    """Symmetric ``k x k`` matrices stored densely as ``k * k`` trailing entries."""

    def __init__(self, k: int) -> None:
        self.k, self.size = k, k * k

    def matvec(self, T: Array, g: Array) -> Array:
        full = T.reshape(*T.shape[:-1], self.k, self.k)
        return jnp.einsum("...ij,...j->...i", full, g)

    def outer(self, g: Array) -> Array:
        return (g[..., :, None] * g[..., None, :]).reshape(*g.shape[:-1], self.size)

    def frobenius2(self, T: Array) -> Array:
        return jnp.sum(T * T, axis=-1)


class _SymmetricPacked:
    """Symmetric ``k x k`` matrices as their ``k (k + 1) / 2`` upper-triangle entries.

    Matrix-vector products unpack with one constant gather fused into the
    reduction. The tangents of the matrix-vector product in the packed matrix
    and of the packed outer product ``g g^T`` are written as linear transposes
    of their adjoints, which are gathers, so reverse mode differentiates
    through gathers only and forward mode stays available. The tangent in the
    vector is rematerialized so that reverse mode keeps the packed matrix, not
    its unpacked copy.
    """

    def __init__(self, k: int, dtype: jnp.dtype) -> None:
        rows, cols, self.full_index = packed_triangle(k)
        self.k, self.size = k, rows.size
        self.upper_index = rows * k + cols
        self.diagonal_index = self.full_index.reshape(k, k)[np.arange(k), np.arange(k)]
        self.rows, self.cols = rows, cols
        self.off_diagonal = rows != cols
        self._dtype = dtype
        self._weight = np.where(self.off_diagonal, 2.0, 1.0)
        rematerialized = jax.checkpoint(self._unpacked_matvec)

        @jax.custom_jvp
        def matvec(T: Array, g: Array) -> Array:
            return self._unpacked_matvec(T, g)

        @matvec.defjvp
        def _matvec_jvp(
            primals: tuple[Array, Array], tangents: tuple[Array, Array]
        ) -> tuple[Array, Array]:
            T, g = primals
            dT, dg = tangents
            y = self._unpacked_matvec(T, g)
            (dy,) = jax.linear_transpose(lambda v: self._matvec_adjoint(v, g), y)(dT)
            return y, dy + rematerialized(T, dg)

        @jax.custom_jvp
        def outer(g: Array) -> Array:
            return self._packed_outer(g)

        @outer.defjvp
        def _outer_jvp(primals: tuple[Array], tangents: tuple[Array]) -> tuple[Array, Array]:
            (g,), (dg,) = primals, tangents
            o = self._packed_outer(g)
            (do,) = jax.linear_transpose(lambda v: self._outer_adjoint(v, g), o)(dg)
            return o, do

        self._matvec, self._outer = matvec, outer

    @property
    def weight(self) -> Array:
        # Built where it is used: an array made once inside a ``jit`` trace would be a tracer that a later trace (or a
        # ``custom_jvp`` rule) captures.
        return jnp.asarray(self._weight, self._dtype)

    def _unpacked_matvec(self, T: Array, g: Array) -> Array:
        full = jnp.take(T, self.full_index, axis=-1).reshape(*T.shape[:-1], self.k, self.k)
        return jnp.einsum("...ij,...j->...i", full, g)

    def _matvec_adjoint(self, v: Array, g: Array) -> Array:
        """Transpose of ``T -> unpack(T) g`` applied to ``v``."""
        take = jnp.take
        mirrored = take(v, self.cols, axis=-1) * take(g, self.rows, axis=-1)
        return take(v, self.rows, axis=-1) * take(g, self.cols, axis=-1) + jnp.where(
            self.off_diagonal, mirrored, 0.0
        )

    def _packed_outer(self, g: Array) -> Array:
        full = (g[..., :, None] * g[..., None, :]).reshape(*g.shape[:-1], self.k * self.k)
        return jnp.take(
            full, self.upper_index, axis=-1, unique_indices=True, indices_are_sorted=True
        )

    def _outer_adjoint(self, v: Array, g: Array) -> Array:
        """Transpose of ``dg -> pack(dg g^T + g dg^T)`` applied to ``v``."""
        diagonal = jnp.take(v, self.diagonal_index, axis=-1)
        return self._unpacked_matvec(v, g) + diagonal * g

    def matvec(self, T: Array, g: Array) -> Array:
        out: Array = self._matvec(T, g)
        return out

    def outer(self, g: Array) -> Array:
        out: Array = self._outer(g)
        return out

    def frobenius2(self, T: Array) -> Array:
        return jnp.sum(self.weight * T * T, axis=-1)


class _SymmetricBlocks:
    """Symmetric ``k x k`` matrices as the ``P = nb (nb + 1) / 2`` blocks of their lower block triangle.

    ``k`` is zero-padded to ``nb * 8``; diagonal blocks are stored in full and
    the ``P * 64`` entries are flattened on the trailing axis. Selecting and
    reassembling blocks are products with constant one-hot matrices, so the
    outer product and the tangents need neither gathers nor scatters.
    Matrix-vector products evaluate through one gather into ``k x k``, fused
    into the reduction, and differentiate through the block products.
    """

    block = 8

    def __init__(self, k: int, dtype: jnp.dtype) -> None:
        s = self.block
        self.k, self.nb = k, -(-k // s)
        rows, cols = np.tril_indices(self.nb)
        self.pairs = rows.size
        self.size = self.pairs * s * s
        eye = np.eye(self.nb)
        self._dtype = dtype
        self._row_onehot, self._col_onehot = eye[rows], eye[cols]
        self._off_diagonal = (rows != cols)[:, None].astype(np.float64)
        self._weight = np.repeat(np.where(rows == cols, 1.0, 2.0), s * s)
        pair = np.full((self.nb, self.nb), -1)
        pair[rows, cols] = np.arange(self.pairs)
        i, j = np.meshgrid(np.arange(k), np.arange(k), indexing="ij")
        hi, lo = np.maximum(i, j), np.minimum(i, j)
        self.full_index = (pair[hi // s, lo // s] * s * s + (hi % s) * s + lo % s).reshape(-1)

        @jax.custom_jvp
        def matvec(T: Array, g: Array) -> Array:
            return self._gather_matvec(T, g)

        @matvec.defjvp
        def _matvec_jvp(
            primals: tuple[Array, Array], tangents: tuple[Array, Array]
        ) -> tuple[Array, Array]:
            T, g = primals
            dT, dg = tangents
            return self._gather_matvec(T, g), self._block_matvec(dT, g) + self._block_matvec(T, dg)

        self._matvec = matvec

    # The constants below are made where they are used. An array built once inside a ``jit`` trace is a tracer, and the
    # ``custom_jvp`` rule that captures it fails to lower ("no constant handler") when the program is differentiated.
    @property
    def row_onehot(self) -> Array:
        return jnp.asarray(self._row_onehot, self._dtype)

    @property
    def col_onehot(self) -> Array:
        return jnp.asarray(self._col_onehot, self._dtype)

    @property
    def off_diagonal(self) -> Array:
        return jnp.asarray(self._off_diagonal, self._dtype)

    @property
    def weight(self) -> Array:
        return jnp.asarray(self._weight, self._dtype)

    def _split(self, g: Array) -> tuple[Array, Array]:
        s = self.block
        pad = self.nb * s - self.k
        if pad:
            g = jnp.pad(g, [(0, 0)] * (g.ndim - 1) + [(0, pad)])
        blocks = g.reshape(*g.shape[:-1], self.nb, s)
        return (
            jnp.einsum("pn,...ns->...ps", self.row_onehot, blocks),
            jnp.einsum("pn,...ns->...ps", self.col_onehot, blocks),
        )

    def _block_matvec(self, T: Array, g: Array) -> Array:
        s = self.block
        T = T.reshape(*T.shape[:-1], self.pairs, s, s)
        gi, gj = self._split(g)
        rows = jnp.einsum("...pab,...pb->...pa", T, gj)
        cols = self.off_diagonal * jnp.einsum("...pab,...pa->...pb", T, gi)
        out = jnp.einsum("pn,...pa->...na", self.row_onehot, rows) + jnp.einsum(
            "pn,...pa->...na", self.col_onehot, cols
        )
        return out.reshape(*out.shape[:-2], self.nb * s)[..., : self.k]

    def _gather_matvec(self, T: Array, g: Array) -> Array:
        full = jnp.take(T, self.full_index, axis=-1).reshape(*T.shape[:-1], self.k, self.k)
        return jnp.einsum("...ij,...j->...i", full, g)

    def matvec(self, T: Array, g: Array) -> Array:
        out: Array = self._matvec(T, g)
        return out

    def outer(self, g: Array) -> Array:
        gi, gj = self._split(g)
        return (gi[..., :, None] * gj[..., None, :]).reshape(*g.shape[:-1], self.size)

    def frobenius2(self, T: Array) -> Array:
        return jnp.sum(self.weight * T * T, axis=-1)


_Symmetric = _SymmetricFull | _SymmetricPacked | _SymmetricBlocks


def _symmetric_storage(k: int, dtype: jnp.dtype) -> _Symmetric:
    """Storage for the per-unit Hessians, chosen by ``k`` from A100 float64 calibration.

    Below ``k = 8`` packing saves too little to pay for its index arithmetic.
    The gathers of the triangle grow with ``k^2`` per point in reverse mode,
    while the block layout keeps dense contractions and pads ``k`` to a
    multiple of 8, so the triangle is used up to ``k = 31`` and blocks above.
    """
    if k < 8:
        return _SymmetricFull(k)
    if k < 32:
        return _SymmetricPacked(k, dtype)
    return _SymmetricBlocks(k, dtype)


def projected_deep_bilaplacian(
    x: Array,
    layers: Sequence[LayerSpec],
    *,
    diffusion: Array | None = None,
    coordinates: Sequence[int] | np.ndarray | None = None,
    directions: Array | None = None,
    allow_almost_everywhere: bool = False,
) -> tuple[Array, Array, Array, Array, Array | None]:
    r"""``(u, |S^T grad u|^2, tr(A hess u), tr(A hess)^2 u, v . grad u)`` for a deep MLP, ``A = S S^T``.

    Parameters
    ----------
    x : Array of shape ``(D,)`` or ``(B, D)``.
    layers : sequence of ``(W, b, spec)``
        As for :func:`omnibias.jax.laplacian.deep_field_value_grad_laplacian`:
        ``W`` has shape ``(out, in)``; ``spec=None`` is an affine layer.
    diffusion : optional factor ``S`` on the selected inputs, of shape
        ``(m, r)``, or its diagonal of shape ``(m,)``; ``None`` gives the
        Laplacian and the bi-Laplacian in those inputs.
    coordinates : optional concrete distinct integer indices of shape
        ``(m,)``: the operator acts on these inputs only (for example the
        space coordinates of a field that also takes time); ``None`` selects
        all ``D`` inputs.
    directions : optional drift directions ``(D, q)`` shared by all points or
        ``(B, D, q)`` per point (``(D, q)`` for an unbatched ``x``), over all
        ``D`` inputs.
    allow_almost_everywhere : accept activations whose derivatives up to
        order three jump (ReLU family, ELU family, Huber, softsign, ...),
        returning values away from their breakpoints.

    Returns
    -------
    value, grad_sq, laplacian, bilaplacian : Arrays of shape ``(C,)`` or ``(B, C)``.
    directional : ``(q, C)`` or ``(B, q, C)``, or ``None`` without directions.
    """
    require_layers(layers)
    a, batched = points_2d(x)
    B, dim = a.shape
    W1 = jnp.asarray(layers[0][0])
    if W1.ndim != 2 or W1.shape[1] != dim:
        raise ValueError(f"the first layer weight must have shape (H, {dim})")
    F = input_factor(W1, diffusion, coordinates=coordinates)
    k = F.shape[1]
    sym = _symmetric_storage(k, F.dtype)
    # Unit-major layout (unit, point, ...): every layer contracts the leading
    # axis in one GEMM without transposing the packed Hessians.
    V: Array | None = None
    if directions is not None:
        v = jnp.asarray(directions)
        if v.ndim == 2 and v.shape[0] == dim:
            V = (W1 @ v)[:, None, :]
        elif batched and v.ndim == 3 and v.shape[:2] == (B, dim):
            V = jnp.einsum("hd,bdq->hbq", W1, v)
        else:
            raise ValueError("directions must have shape (D, q) or (B, D, q)")
    a = a.T
    # (gradient, Hessian, Laplacian, gradient of the Laplacian, bi-Laplacian) along y
    state: tuple[Array, Array, Array, Array, Array] | None = None
    for i, (W, b, spec) in enumerate(layers):
        W = jnp.asarray(W)
        if W.ndim != 2 or W.shape[1] != a.shape[0]:
            raise ValueError(f"layer {i} weight must have shape (out, {a.shape[0]})")
        z = W @ a
        if b is not None:
            z = z + jnp.asarray(b)[:, None]
        lower: tuple[Array, Array, Array, Array] | None
        if state is None:
            gz = F[:, None, :]
            n = jnp.sum(F * F, axis=-1)[:, None]
            lower = None
        else:
            J, T, L, P, Q = state
            gz = jnp.tensordot(W, J, axes=1)
            n = jnp.sum(gz * gz, axis=-1)
            lower = (jnp.tensordot(W, T, axes=1), W @ L, jnp.tensordot(W, P, axes=1), W @ Q)
            if V is not None:
                V = jnp.tensordot(W, V, axes=1)
        if spec is None:
            if lower is None:
                zero = jnp.zeros_like(z)
                gz = jnp.broadcast_to(gz, (*z.shape, k))
                lower = (jnp.zeros((*z.shape, sym.size), z.dtype), zero, jnp.zeros_like(gz), zero)
            state = (gz, *lower)
            a = z
            continue
        resolved = fastpath_activation(
            spec, allow_almost_everywhere=allow_almost_everywhere, smooth=4
        )
        try:
            s1, s2, s3, s4 = (resolved.fastpath(z, order) for order in (1, 2, 3, 4))
        except NotImplementedError as exc:
            raise ValueError(
                f"activation {resolved.name!r} fastpath does not support order 4, required for the bi-Laplacian"
            ) from exc
        e = (..., None)
        outer = sym.outer(gz)
        if lower is None:
            T = s2[e] * outer
            L = s2 * n
            P = (s3 * n)[e] * gz
            Q = s4 * n * n
        else:
            Hz, Lz, Pz, Qz = lower
            Hg = sym.matvec(Hz, gz)
            T = s1[e] * Hz + s2[e] * outer
            L = s1 * Lz + s2 * n
            P = s1[e] * Pz + s2[e] * (Lz[e] * gz + 2.0 * Hg) + (s3 * n)[e] * gz
            Q = (
                s1 * Qz
                + s2 * (4.0 * jnp.sum(gz * Pz, axis=-1) + Lz * Lz + 2.0 * sym.frobenius2(Hz))
                + s3 * (2.0 * n * Lz + 4.0 * jnp.sum(gz * Hg, axis=-1))
                + s4 * n * n
            )
        state = (s1[e] * gz, T, L, P, Q)
        if V is not None:
            V = s1[e] * V
        a = resolved.forward(z)
    assert state is not None
    value, L, Q = a.T, state[2].T, state[4].T
    grad_sq = jnp.sum(state[0] * state[0], axis=-1).T
    directional = None
    if V is not None:
        directional = jnp.transpose(jnp.broadcast_to(V, (V.shape[0], B, V.shape[2])), (1, 2, 0))
    if not batched:
        return value[0], grad_sq[0], L[0], Q[0], None if directional is None else directional[0]
    return value, grad_sq, L, Q, directional
