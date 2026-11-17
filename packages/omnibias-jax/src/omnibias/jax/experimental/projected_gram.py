# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Gram matrices of PDE residual gradients for deep MLPs whose input dimension exceeds the width.

:func:`omnibias.jax.experimental.residual_gram.deep_residual_gram` carries ``T = D + 2`` forward-Laplacian streams
through the network and forms the first-layer gradient of every row explicitly (``H D`` entries per row). For
``u(x) = f(W_1 x + b_1)`` and a metric ``A = S S^T`` on selected inputs, the residual depends on the first layer through
``G = W_1 A W_1^T`` and the drift tangents ``W_1 V`` only
(:mod:`omnibias.jax.experimental.projected_laplacian`). With ``F = M Q`` (``M = W_1 S``, ``Q`` an orthonormal basis of
the row space of ``M`` when ``m > H``, the identity otherwise), ``k = min(m, H)`` tangent columns replace the ``D``
input directions, and the gradient of row ``b`` in ``W_1`` is

.. math::

    \partial_{W_1} r_b = q_b x_b^T + Q^F_b\, \Omega + Q^V_b V^T,\qquad \Omega = Q^T S^T \in \mathbb{R}^{k \times D},

with ``q_b`` the cotangent of the first-layer value stream, ``Q^F_b`` of the ``k`` projected tangents and ``Q^V_b`` of the
``q`` drift tangents, all from one reverse pass of ``sum_b r_b`` over the projected streams. Writing
``c_b = \Omega x_b``, ``M = \Omega \Omega^T`` and ``\hat N = V^T \Omega^T``, the first-layer block of the Gram matrix is

.. math::

    (q_b \cdot q_c)(x_b \cdot x_c) + q_b^T Q^F_c c_b + q_c^T Q^F_b c_c + \langle Q^F_b, Q^F_c M \rangle
    + \text{drift terms} + q_b \cdot q_c ,

so the ``H D`` rows are never formed: ``O(N^2 (D + H k))`` for the block against ``O(N^2 H D)``, and ``O(N D)`` memory
against ``O(N H D)``. The saving is confined to the first layer and grows with ``D / H``: in calibration the same streams
with explicit first-layer rows were 1.0-1.3 times slower at ``D / H = 4``, 1.4-1.9 times at 8 and more than 2 times from 16.
Hidden layers use the stream identity or explicit rows per layer exactly as
:func:`~omnibias.jax.experimental.residual_gram.deep_residual_gram`, with ``T = k + q + 2`` streams. The result equals
``J J^T`` up to rounding, including when ``W_1 A W_1^T`` is rank-deficient.

Fourth-order blocks (``order = 4``, e.g. Kuramoto-Sivashinsky or Cahn-Hilliard residuals) carry the streams of
:func:`omnibias.jax.experimental.projected_bilaplacian.projected_deep_bilaplacian`: value, ``k`` gradients, ``q`` drift
tangents, the packed Hessian ``k (k + 1) / 2``, the Laplacian, its ``k`` gradients and the bi-Laplacian, with
``T = 1 + k + q + k (k + 1) / 2 + 1 + k + 1`` streams. The first-layer block, the hidden-layer identity and the explicit
rows are unchanged; only the constant of the per-layer rule differs (``1.2`` instead of ``2.2``: the measured
crossover between the identity and explicit rows for one to three space coordinates and widths 96-768). In those cells
explicit rows from the same streams were 1.1-5 times slower than the rule, the gap growing with the width.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from omnibias.jax.experimental import residual_gram as _rg
from omnibias.jax.experimental._chain import fastpath_activation, packed_triangle
from omnibias.jax.experimental.conditioning import (
    ConditioningPolicy,
    damped_solve,
    floored_damping,
)
from omnibias.jax.experimental.residual_gram import _assemble, _layer_diag, _prepare, _streams

import jax
import jax.numpy as jnp
from jax import Array

if TYPE_CHECKING:
    from omnibias.jax.activations import JaxActivationSpec

    LayerSpec = tuple[Array, Array | None, JaxActivationSpec | str | None]
    Block = tuple[Array, Callable[..., Array]] | tuple[Array, Callable[..., Array], int]

__all__ = [
    "projected_residual_gauss_newton",
    "projected_residual_gram",
    "projected_residual_ntk_diagonal",
]

_EXPLICIT_FIRST = False  # test and ablation hook: first-layer rows ``q x^T + Q^F Omega + Q^V V^T`` formed explicitly
_STREAMS_COST_4 = 1.2  # constant of the per-layer rule c pairs (in + out) < in out for pairs involving a fourth-order block


class _Layout:
    """Stream layout of a fourth-order block: ``z | g (k) | v (q) | packed H (k (k + 1) / 2) | lap | grad lap (k) | bilap``."""

    def __init__(self, k: int, q: int) -> None:
        self.k, self.q = k, q
        rows, cols, self.full_index = packed_triangle(k)
        self.rows, self.cols, self.npk = rows, cols, rows.size
        self.weight = np.where(rows != cols, 2.0, 1.0)
        self.iH = 1 + k + q
        self.iL = self.iH + self.npk
        self.iP = self.iL + 1
        self.n_zero = self.npk + 1 + k + 1
        self.T = 1 + k + q + self.n_zero

    def activate(self, S: Array, spec: Any, allow_ae: bool) -> Array:
        """Closed-form activation on the streams (value, tangents, Hessian, Laplacian, grad Laplacian, bi-Laplacian)."""
        if spec is None:
            return S
        act = fastpath_activation(spec, allow_almost_everywhere=allow_ae, smooth=4)
        k, iH, iL, iP = self.k, self.iH, self.iL, self.iP
        z = S[:, 0]
        g = S[:, 1 : 1 + k]
        v = S[:, 1 + k : iH]
        Hp = S[:, iH:iL]
        Lz = S[:, iL]
        Pz = S[:, iP : iP + k]
        Qz = S[:, -1]
        s1, s2, s3, s4 = (act.fastpath(z, order) for order in (1, 2, 3, 4))
        n = jnp.sum(g * g, axis=1)
        Hfull = Hp[:, self.full_index].reshape(Hp.shape[0], k, k, -1)
        Hg = jnp.einsum("bjlh,blh->bjh", Hfull, g)
        outer = g[:, self.rows] * g[:, self.cols]
        frob = jnp.einsum("p,bph->bh", jnp.asarray(self.weight, S.dtype), Hp * Hp)
        gP = jnp.sum(g * Pz, axis=1)
        gHg = jnp.sum(g * Hg, axis=1)
        e = s1[:, None]
        Pn = e * Pz + s2[:, None] * (Lz[:, None] * g + 2.0 * Hg) + (s3 * n)[:, None] * g
        Qn = (
            s1 * Qz
            + s2 * (4.0 * gP + Lz * Lz + 2.0 * frob)
            + s3 * (2.0 * n * Lz + 4.0 * gHg)
            + s4 * n * n
        )
        parts = [
            act.forward(z)[:, None],
            e * g,
            e * v,
            e * Hp + s2[:, None] * outer,
            (s1 * Lz + s2 * n)[:, None],
            Pn,
            Qn[:, None],
        ]
        return jnp.concatenate(parts, axis=1)


class _First:
    """First-layer projection: ``F = M Q``, ``Omega = Q^T S^T`` pieces and the drift tangents."""

    def __init__(
        self,
        W1: Array,
        coordinates: Sequence[int] | np.ndarray | None,
        diffusion: Array | None,
        directions: Array | None,
    ) -> None:
        H, D = W1.shape
        idx = np.arange(D) if coordinates is None else np.asarray(coordinates)
        m = idx.size
        s = jnp.ones(m, W1.dtype) if diffusion is None else jnp.asarray(diffusion, W1.dtype)
        M = W1[:, idx] * s
        self.idx, self.s, self.D, self.H = idx, s, D, H
        if m <= H:
            self.Qb: Array | None = None
            self.F = M
        else:
            self.Qb, _ = jnp.linalg.qr(jax.lax.stop_gradient(M).T)
            self.F = M @ self.Qb
        self.k = self.F.shape[1]
        self.V: Array | None = None
        self.q = 0
        Vt = jnp.zeros((H, 0), W1.dtype)
        if directions is not None:
            V = jnp.asarray(directions, W1.dtype)
            if V.ndim != 2 or V.shape[0] != D:
                raise ValueError(f"directions must have shape ({D}, q)")
            self.V, self.q = V, V.shape[1]
            Vt = W1 @ V
        self.tangents = jnp.concatenate([self.F, Vt], axis=1).T
        self.w = jnp.concatenate([jnp.ones(self.k, W1.dtype), jnp.zeros(self.q, W1.dtype)])
        if self.Qb is None:
            self.Mmat = jnp.diag(s * s)
        else:
            self.Mmat = self.Qb.T @ (self.Qb * (s * s)[:, None])
        if self.V is None:
            self.Nhat = jnp.zeros((0, self.k), W1.dtype)
            self.VtV = jnp.zeros((0, 0), W1.dtype)
        else:
            Vs = self.V[idx].T * s
            self.Nhat = Vs if self.Qb is None else Vs @ self.Qb
            self.VtV = self.V.T @ self.V

    @property
    def directions(self) -> Array:
        """The extra directions ``V``; only read where ``q > 0``."""
        if self.V is None:
            raise ValueError("no extra directions were given")
        return self.V

    def coordinates_of(self, x: Array) -> Array:
        """``c_b = Omega x_b`` for all points."""
        xs = x[:, self.idx] * self.s
        return xs if self.Qb is None else xs @ self.Qb

    def omega(self) -> Array:
        """``Omega = Q^T S^T`` of shape ``(k, D)``."""
        Ss = (
            jnp.zeros((self.D, self.idx.size), self.s.dtype)
            .at[self.idx, jnp.arange(self.idx.size)]
            .set(self.s)
        )
        return Ss.T if self.Qb is None else self.Qb.T @ Ss.T


def _parts(
    x: Array, Q0: Array, order: int, f: _First
) -> tuple[Array, Array | None, Array | None, Array, Array, Array | None]:
    """Row-flattened ``q``, ``Q^F``, ``Q^V`` and the per-row ``x``, ``c = Omega x``, ``V^T x``."""
    B, R = Q0.shape[:2]
    q = Q0[:, :, 0].reshape(B * R, -1)
    rep = (lambda a: jnp.repeat(a, R, axis=0)) if R > 1 else (lambda a: a)
    xr, c = rep(x), rep(f.coordinates_of(x))
    xv = rep(x @ f.directions) if f.q else None
    if order == 0:
        return q, None, None, xr, c, xv
    k, nq = f.k, f.q
    QF = jnp.swapaxes(Q0[:, :, 1 : 1 + k], -1, -2).reshape(B * R, -1, k)
    QV = None
    if nq:
        QV = jnp.swapaxes(Q0[:, :, 1 + k : 1 + k + nq], -1, -2).reshape(B * R, -1, nq)
    return q, QF, QV, xr, c, xv


def _explicit_rows(p: Any, omega: Array, f: _First, bias: bool) -> Array:
    q, QF, QV, x, _, _ = p
    G = q[:, :, None] * x[:, None, :]
    if QF is not None:
        G = G + QF @ omega
    if QV is not None:
        G = G + QV @ f.directions.T
    G = G.reshape(G.shape[0], -1)
    return jnp.concatenate([G, q], axis=1) if bias else G


def _flat(a: Array) -> Array:
    return a.reshape(a.shape[0], -1)


def _first_block(pi: Any, pj: Any, f: _First, bias: bool) -> Array:
    qi, QFi, QVi, xi, ci, xvi = pi
    qj, QFj, QVj, xj, cj, xvj = pj
    out: Array = (qi @ qj.T) * (xi @ xj.T)
    if bias:
        out = out + qi @ qj.T

    def cross(a: Any, b: Any) -> Array:
        """``<a_b, f_c> + <v_b, ...>`` terms with the left factor of block ``a`` and the right of block ``b``."""
        qa, QFa, QVa, xa, ca, xva = a
        qb, QFb, QVb, *_ = b
        acc = jnp.zeros((qa.shape[0], qb.shape[0]), qa.dtype)
        if QFb is not None:
            acc = acc + _flat(qa[:, :, None] * ca[:, None, :]) @ _flat(QFb).T
        if QVb is not None:
            acc = acc + _flat(qa[:, :, None] * xva[:, None, :]) @ _flat(QVb).T
        return acc

    out = out + cross(pi, pj) + cross(pj, pi).T
    if QFi is not None and QFj is not None:
        out = out + _flat(QFi @ f.Mmat) @ _flat(QFj).T
    if QVi is not None and QVj is not None:
        out = out + _flat(QVi @ f.VtV) @ _flat(QVj).T
    if QFi is not None and QVj is not None:
        out = out + _flat(QFi) @ _flat(QVj @ f.Nhat).T
    if QVi is not None and QFj is not None:
        out = out + _flat(QVi @ f.Nhat) @ _flat(QFj).T
    return out


def _streams_all(
    blocks: Sequence[Block],
    layers: Sequence[LayerSpec],
    coordinates: Sequence[int] | np.ndarray | None,
    diffusion: Array | None,
    directions: Array | None,
    strategy: str,
    allow_ae: bool,
) -> tuple[list[Any], Any, _First, list[Any], list[Any]]:
    """Normalized blocks, dtype, the first-layer projection, per-block ``(r, P, Q)`` and the first-layer parts."""
    norm, _, dtype = _prepare(
        blocks, layers, coordinates, diffusion, strategy, (0, 1, 2, 4), engine="projected"
    )
    f = _First(jnp.asarray(layers[0][0]), coordinates, diffusion, directions)
    k, nq = f.k, f.q
    lay = _Layout(k, nq)

    def wrap(res: Callable[..., Array], order: int, x: Array) -> Callable[[Array], Array]:
        def unpack(out: Array) -> Array:
            u = out[:, 0]
            if order == 0:
                return jnp.asarray(res(u, None, None, None, x))
            gs = jnp.sum(out[:, 1 : 1 + k] ** 2, axis=1)
            dv = out[:, 1 + k : 1 + k + nq] if nq else None
            if order == 4:
                return jnp.asarray(res(u, gs, out[:, lay.iL], out[:, -1], dv, x))
            lap = out[:, 1 + k + nq] if order == 2 else None
            return jnp.asarray(res(u, gs, lap, dv, x))

        return unpack

    four: dict[str, Any] = {"activate": lay.activate, "n_streams": lay.T, "n_zero": lay.n_zero}
    data = [
        _streams(
            layers,
            x,
            res,
            order,
            f.w,
            allow_ae,
            first=f.tangents,
            unpack=wrap(res, order, x),
            **(four if order == 4 else {}),
        )
        for x, res, order in norm
    ]
    parts = [_parts(x, Q[0], order, f) for (x, _, order), (_, _, Q) in zip(norm, data, strict=True)]
    return norm, dtype, f, data, parts


def _gram(
    blocks: Sequence[Block],
    layers: Sequence[LayerSpec],
    coordinates: Sequence[int] | np.ndarray | None,
    diffusion: Array | None,
    directions: Array | None,
    strategy: str,
    allow_ae: bool,
) -> tuple[Array, Array, _First, list[Any], list[Any], list[Any]]:
    norm, dtype, f, data, parts = _streams_all(
        blocks, layers, coordinates, diffusion, directions, strategy, allow_ae
    )
    biased = layers[0][1] is not None

    if _EXPLICIT_FIRST:
        omega = f.omega()
        rows = [_explicit_rows(p, omega, f, biased) for p in parts]

        def first_gram(i: int, j: int) -> Array:
            return rows[i] @ rows[j].T

    else:

        def first_gram(i: int, j: int) -> Array:
            return _first_block(parts[i], parts[j], f, biased)

    def cost(i: int, j: int) -> float:
        return _STREAMS_COST_4 if 4 in (norm[i][2], norm[j][2]) else _rg._STREAMS_COST

    r, K, norm_, data_ = _assemble(norm, data, layers, strategy, first_gram, dtype, cost)
    return r, K, f, parts, norm_, data_


def _first_diag(p: Any, f: _First, bias: bool) -> Array:
    """``|dr_b / dW_1|^2`` (+ bias) per row from the first-layer parts, without forming the rows."""
    q, QF, QV, x, c, xv = p
    qq = jnp.sum(q * q, axis=1)
    out = qq * jnp.sum(x * x, axis=1)
    if bias:
        out = out + qq
    if QF is not None:
        out = out + 2.0 * jnp.einsum("nh,nhk,nk->n", q, QF, c)
        out = out + jnp.einsum("nhk,kl,nhl->n", QF, f.Mmat, QF)
    if QV is not None:
        out = out + 2.0 * jnp.einsum("nh,nhp,np->n", q, QV, xv)
        out = out + jnp.einsum("nhp,pr,nhr->n", QV, f.VtV, QV)
    if QF is not None and QV is not None:
        out = out + 2.0 * jnp.einsum("nhk,nhp,pk->n", QF, QV, f.Nhat)
    return out


def projected_residual_ntk_diagonal(
    blocks: Sequence[Block],
    layers: Sequence[LayerSpec],
    *,
    coordinates: Sequence[int] | np.ndarray | None = None,
    diffusion: Array | None = None,
    directions: Array | None = None,
    strategy: str = "auto",
    allow_almost_everywhere: bool = False,
) -> tuple[Array, Array]:
    r"""Residuals ``r`` and the diagonal of ``K = J J^T`` through the first-layer projection, without ``K`` or ``J``.

    Per row, the first-layer term follows from the cotangent parts as
    ``|q|^2 |x|^2 + 2 q^T Q^F c + <Q^F, Q^F M> + (drift terms)`` (plus ``|q|^2`` for the bias) at ``O(H (k + q)^2)`` and no ``H D`` row, and every
    hidden layer uses the stream identity or explicit rows as for
    :func:`omnibias.jax.experimental.residual_gram.deep_residual_ntk_diagonal`. Arguments and row order as for
    :func:`projected_residual_gram`; memory is ``O(N)`` instead of the ``N^2`` of ``K``.
    """
    norm, _, f, data, parts = _streams_all(
        blocks, layers, coordinates, diffusion, directions, strategy, allow_almost_everywhere
    )
    biased = layers[0][1] is not None
    rs, ds = [], []
    for (_, _, _), (r, P, Q), p in zip(norm, data, parts, strict=True):
        d = _first_diag(p, f, biased).reshape(r.shape)
        for li, (_, b, _) in enumerate(layers[1:], start=1):
            d = d + _layer_diag(P[li - 1], Q[li], b is not None, strategy, _rg._DIAG_CHUNK)
        rs.append(r.reshape(-1))
        ds.append(d.reshape(-1))
    return jnp.concatenate(rs), jnp.concatenate(ds)


def projected_residual_gram(
    blocks: Sequence[Block],
    layers: Sequence[LayerSpec],
    *,
    coordinates: Sequence[int] | np.ndarray | None = None,
    diffusion: Array | None = None,
    directions: Array | None = None,
    strategy: str = "auto",
    allow_almost_everywhere: bool = False,
) -> tuple[Array, Array]:
    r"""Residuals ``r`` and ``K = J J^T`` over every weight and bias, through the first-layer projection.

    Parameters
    ----------
    blocks : sequence of ``(x, residual)`` or ``(x, residual, order)``
        ``x`` of shape ``(B, D)``; ``residual(u, grad_sq, lap, directional, x)`` returns shape ``(B,)`` or
        ``(B, R)`` from ``u (B, C)``, ``grad_sq = |S^T grad u|^2 (B, C)``, ``lap = tr(S S^T hess u) (B, C)``,
        ``directional = V^T grad u (B, q, C)`` (``None`` without ``directions``) and ``x``, called once on the block
        and pointwise in its rows (as :func:`omnibias.jax.experimental.projected_laplacian.projected_deep_laplacian`).
        ``order`` (default 2): ``0`` passes only ``u``, ``1`` leaves ``lap`` as ``None``. ``order = 4`` calls
        ``residual(u, grad_sq, lap, bilap, directional, x)`` with the additional
        ``bilap = (tr(S S^T hess))^2 u (B, C)`` (``lap`` and ``grad_sq`` as above), for fourth-order operators.
    layers : sequence of ``(W, b, spec)`` as for ``deep_residual_gram``; ``W_1`` has shape ``(H, D)``.
    coordinates, diffusion : the metric acts on these inputs (all by default) with diagonal factor ``diffusion``
        of shape ``(m,)`` (ones by default).
    directions : optional constant drift directions ``(D, q)``.
    strategy, allow_almost_everywhere : as for ``deep_residual_gram`` (hidden layers).

    Returns
    -------
    r : ``(N,)`` and ``K : (N, N)``, rows ordered by block, point, residual component.
    """
    r, K, *_ = _gram(
        blocks, layers, coordinates, diffusion, directions, strategy, allow_almost_everywhere
    )
    return r, K


def _step_jvp(
    norm: Sequence[Any],
    data: Sequence[Any],
    f: _First,
    delta: Sequence[tuple[Array, Array | None]],
) -> Array:
    """``J d`` for a per-layer step ``d = [(dW, db)]``: the transpose of the step assembly in the Gauss-Newton step.

    Row ``(b, r)`` is ``sum_l <Q_l[b, r], stream inputs of layer l contracted with dW_l>`` over the same streams the step
    sums over (value, ``k`` projected tangents and ``q`` drift tangents in the first layer; every stream of a hidden layer).
    """
    omega = f.omega()
    out = []
    for (x, _, order), (rb, P, Q) in zip(norm, data, strict=True):
        acc = jnp.zeros(rb.shape, rb.dtype)
        for li, (dW, db) in enumerate(delta):
            Ql = Q[li]
            bias = None if db is None else jnp.asarray(db)
            if li == 0:
                value = x @ dW.T + (0.0 if bias is None else bias)
                acc = acc + jnp.einsum("bro,bo->br", Ql[:, :, 0], value)
                if order >= 1:
                    k, nq = f.k, f.q
                    acc = acc + jnp.einsum("brko,ko->br", Ql[:, :, 1 : 1 + k], omega @ dW.T)
                    if nq:
                        acc = acc + jnp.einsum(
                            "brqo,qo->br", Ql[:, :, 1 + k : 1 + k + nq], f.directions.T @ dW.T
                        )
            else:
                acc = acc + jnp.einsum("brto,bto->br", Ql, jnp.einsum("bti,oi->bto", P[li - 1], dW))
                if bias is not None:
                    acc = acc + jnp.einsum("bro,o->br", Ql[:, :, 0], bias)
        out.append(acc.reshape(-1))
    return jnp.concatenate(out)


def projected_residual_gauss_newton(
    blocks: Sequence[Block],
    layers: Sequence[LayerSpec],
    damping: float | Array,
    *,
    coordinates: Sequence[int] | np.ndarray | None = None,
    diffusion: Array | None = None,
    directions: Array | None = None,
    strategy: str = "auto",
    allow_almost_everywhere: bool = False,
    conditioning: ConditioningPolicy | None = None,
    momentum: float | Array = 0.0,
    prev_delta: Sequence[tuple[Array, Array | None]] | None = None,
) -> tuple[Array, Array, list[tuple[Array, Array | None]]]:
    r"""Dual Gauss-Newton step ``J^T (J J^T + damping I)^{-1} r`` through the first-layer projection.

    ``damping`` is a scalar, or a vector of ``n`` values solved from one eigendecomposition of ``K``; the step arrays
    then carry a leading axis of length ``n``. A ``conditioning`` policy with mode ``"floor"`` first raises ``damping`` to
    the value at which the step is reproducible to the policy's tolerance. Returns ``(r, K, step)`` with ``step`` a list of ``(dW, db)`` as for
    :func:`omnibias.jax.experimental.residual_gram.deep_residual_gauss_newton`.

    ``momentum`` ``mu`` with ``prev_delta`` (the previous returned step, subtracted from the weights last time) is the SPRING
    update of :func:`omnibias.jax.optim_pinn.gauss_newton_solve` in the same dual form,
    ``delta = J^T (J J^T + damping I)^{-1} (r - mu J d_prev) + mu d_prev``, the minimiser of
    ``||J delta - r||^2 + damping ||delta - mu d_prev||^2``. ``J d_prev`` is read off the stored cotangent streams, the
    transpose of the step assembly below, so no Jacobian row and no extra backward pass is formed. ``prev_delta`` is a list
    of ``(dW, db)`` per layer (``db`` is ``None`` for a layer without bias); ``momentum`` without ``prev_delta`` is refused.
    """
    if prev_delta is None and not (isinstance(momentum, (int, float)) and momentum == 0.0):
        raise ValueError("momentum needs prev_delta")
    if prev_delta is not None and len(prev_delta) != len(layers):
        raise ValueError(f"prev_delta needs one (dW, db) per layer ({len(layers)}), got {len(prev_delta)}")
    r, K, f, parts, norm, data = _gram(
        blocks, layers, coordinates, diffusion, directions, strategy, allow_almost_everywhere
    )
    d = floored_damping(K, damping, conditioning)
    rhs = r if prev_delta is None else r - momentum * _step_jvp(norm, data, f, prev_delta)
    if d.ndim == 0:
        alphas = damped_solve(K, rhs, d)[0][None]
    else:
        lam, U = jnp.linalg.eigh(K)
        alphas = jnp.einsum("ni,ti->tn", U, (U.T @ rhs)[None] / (lam[None] + d[:, None]))
    offsets = np.cumsum([0] + [rb.size for rb, _, _ in data])
    omega = f.omega()
    steps: list[list[tuple[Array, Array | None]]] = []
    for alpha in alphas:
        a = [
            alpha[o0:o1].reshape(rb.shape)
            for o0, o1, (rb, _, _) in zip(offsets[:-1], offsets[1:], data, strict=True)
        ]
        step: list[tuple[Array, Array | None]] = []
        for li, (_, b, _) in enumerate(layers):
            dW: Array | float = 0.0
            db: Array | float = 0.0
            for ab, (x, _, order), (_, P, Q) in zip(a, norm, data, strict=True):
                Qa = jnp.einsum("br,brto->bto", ab, Q[li])
                if li == 0:
                    dW = dW + Qa[:, 0].T @ x
                    if order >= 1:
                        k, nq = f.k, f.q
                        dW = dW + jnp.sum(Qa[:, 1 : 1 + k], axis=0).T @ omega
                        if nq:
                            dW = dW + jnp.sum(Qa[:, 1 + k : 1 + k + nq], axis=0).T @ f.directions.T
                else:
                    dW = dW + jnp.einsum("bto,bti->oi", Qa, P[li - 1])
                db = db + jnp.sum(Qa[:, 0], axis=0)
            dW, db_ = jnp.asarray(dW), None if b is None else jnp.asarray(db)
            if prev_delta is not None:  # SPRING: the minimiser is centred on mu d_prev
                dW = dW + momentum * prev_delta[li][0]
                if db_ is not None:
                    db_ = db_ + momentum * cast(Array, prev_delta[li][1])
            step.append((dW, db_))
        steps.append(step)
    if d.ndim == 0:
        return r, K, steps[0]
    stacked = [
        (
            jnp.stack([s[li][0] for s in steps]),
            None if steps[0][li][1] is None else jnp.stack([cast(Array, s[li][1]) for s in steps]),
        )
        for li in range(len(layers))
    ]
    return r, K, stacked
