# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Gram matrices of PDE residual gradients for deep MLP fields, without the Jacobian.

Gauss-Newton and energy natural gradient steps in dual form,
``J^T (J J^T + lambda I)^{-1} r``, and the empirical neural tangent kernel of a
residual need ``K = J J^T`` with ``J = dr / d theta``. For a pointwise residual
``r_b = rho(u(x_b), grad u(x_b), lap u(x_b), x_b)`` of a deep MLP, every dense
layer ``z = W a + b`` maps the ``T`` forward-Laplacian streams of its input at
a point (value, first derivatives, Laplacian) by the same ``W``. The gradient
of ``r_b`` in ``W`` is therefore ``Q_b P_b^T``, with ``P_b`` the ``(in, T)``
input streams and ``Q_b`` the ``(out, T)`` cotangents of the output streams,
all from one reverse pass of ``sum_b r_b``. Hence

.. math::

    \langle \partial_W r_b, \partial_W r_c \rangle
    = \sum_{t, s} (P_b^T P_c)_{ts} (Q_b^T Q_c)_{ts},

which costs ``T (T + 1) / 2 (in + out)`` per pair of rows of one block (both
``(t, s)`` orders at once) instead of the ``in * out`` of explicit per-row
gradients. Hidden layers use it where that is cheaper by a calibrated margin;
the first layer, the readout and narrow layers form their per-row gradients
from the same cotangents. ``J^T`` acts through the same cotangents. The result
equals ``J J^T`` up to rounding.

Pairs of streams of one block are formed one stream against all later streams
per product. A cyclic-offset enumeration (one batched product of identical
shape per offset, about ``T / 2 + 1`` GEMM shapes instead of ``T``, the same
arithmetic) is implemented and exact but not the default: held out, it lowered
the cold compile 1.3-1.8x at nine or more streams but did not meet the
pre-registered gate (see the compile-time section of the performance page).
The diagonal ``|dr_b / d theta|^2`` alone follows from the
same identity at ``O(B T^2 (in + out))`` per layer and ``O(N)`` memory.
:func:`pad_block` and :func:`masked_residual` pad a block to a fixed row bucket
with exactly zero rows and columns, so a changing batch size reuses one
compiled program.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from omnibias.jax.activations import JaxActivationSpec
from omnibias.jax.experimental._chain import distinct_indices, fastpath_activation
from omnibias.jax.experimental.conditioning import ConditioningPolicy, damped_solve
from omnibias.jax.experimental.demand import Residual, block_order, lower_full, lower_projected

import jax
import jax.numpy as jnp
from jax import Array

if TYPE_CHECKING:
    LayerSpec = tuple[Array, Array | None, JaxActivationSpec | str | None]
    Block = tuple[Array, Callable[..., Array]] | tuple[Array, Callable[..., Array], int]

__all__ = [
    "deep_residual_gauss_newton",
    "deep_residual_gram",
    "deep_residual_ntk_diagonal",
    "masked_residual",
    "pad_block",
]

_STREAMS_COST = 2.2
_DIAG_COST = 1.0
_DIAG_CHUNK = 1 << 25


def _activate(
    S: Array, spec: JaxActivationSpec | str | None, order: int, w: Array, allow_ae: bool
) -> Array:
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
        L = s1 * S[:, D + 1] + act.fastpath(z, 2) * jnp.einsum("bkh,k->bh", Jz * Jz, w)
        parts.append(L[:, None])
    return jnp.concatenate(parts, axis=1)


def _forward(
    layers: Sequence[LayerSpec],
    x: Array,
    order: int,
    w: Array,
    eps: list[Array],
    allow_ae: bool,
    first: Array | None = None,
    activate: Callable[[Array, Any, bool], Array] | None = None,
    n_zero: int | None = None,
) -> tuple[Array, list[Array]]:
    """Output streams ``(B, T, C)`` and the input streams of every layer after the first.

    ``first`` ``(K, H)`` replaces the tangent block ``W_1^T`` of the first layer by a projected one
    (``K = w.shape[0]`` tangent columns shared by all points); ``activate(S, spec, allow_ae)`` replaces the activation
    on streams and ``n_zero`` the number of parameter-independent zero streams after the tangents.
    """
    B, D = x.shape
    inputs: list[Array] = []
    A = x
    for i, (W, b, spec) in enumerate(layers):
        W = jnp.asarray(W)
        if i == 0:
            parts = [(x @ W.T)[:, None]]
            if order >= 1:
                tang = W.T if first is None else first
                parts.append(jnp.broadcast_to(tang[None], (B, tang.shape[0], W.shape[0])))
            nz = 1 if order == 2 else (n_zero or 0)
            if nz:
                parts.append(jnp.zeros((B, nz, W.shape[0]), x.dtype))
            S = jnp.concatenate(parts, axis=1)
        else:
            inputs.append(A)
            S = A @ W.T
        if b is not None:
            S = S.at[:, 0].add(jnp.asarray(b))
        A = (
            _activate(S + eps[i], spec, order, w, allow_ae)
            if activate is None
            else activate(S + eps[i], spec, allow_ae)
        )
    return A, inputs


def _streams(
    layers: Sequence[LayerSpec],
    x: Array,
    residual: Callable[..., Array],
    order: int,
    w: Array,
    allow_ae: bool,
    first: Array | None = None,
    unpack: Callable[[Array], Array] | None = None,
    activate: Callable[[Array, Any, bool], Array] | None = None,
    n_streams: int | None = None,
    n_zero: int | None = None,
) -> tuple[Array, list[Array], list[Array]]:
    """Residuals ``(B, R)``, layer input streams ``(B, T, in)`` and output-stream cotangents ``(B, R, T, out)``.

    ``w`` weights the ``w.shape[0]`` tangent columns of the Laplacian; ``first`` and ``unpack`` (output streams to the
    residual's value) serve the projected module.
    """
    B = x.shape[0]
    D = w.shape[0]
    T = n_streams if n_streams is not None else (1, 1 + D, 2 + D)[order]
    eps = [jnp.zeros((B, T, jnp.shape(W)[0]), x.dtype) for W, _, _ in layers]

    def f(e: list[Array]) -> tuple[Array, list[Array]]:
        out, inputs = _forward(layers, x, order, w, e, allow_ae, first, activate, n_zero)
        if unpack is not None:
            return jnp.asarray(unpack(out)), inputs
        g = out[:, 1 : D + 1] if order >= 1 else None
        lap = out[:, D + 1] if order == 2 else None
        return jnp.asarray(residual(out[:, 0], g, lap, x)), inputs

    r, vjp, inputs = jax.vjp(f, eps, has_aux=True)
    if r.ndim not in (1, 2) or r.shape[0] != B:
        raise ValueError(f"residual must return shape ({B},) or ({B}, R), got {r.shape}")
    R = 1 if r.ndim == 1 else r.shape[1]

    def cotangents(e: Array) -> list[Array]:
        ct = jnp.broadcast_to(e, (B, R))
        (Q,) = vjp(ct[:, 0] if r.ndim == 1 else ct)
        return list(Q)

    Qs = jax.vmap(cotangents)(jnp.eye(R, dtype=r.dtype))
    return jnp.reshape(r, (B, R)), inputs, [jnp.moveaxis(q, 0, 1) for q in Qs]


def _first_jacobian(x: Array, Q: Array, order: int, bias: bool) -> Array:
    B, R, _, H = Q.shape
    D = x.shape[1]
    qv = Q[:, :, 0]
    G = qv[..., None] * x[:, None, None, :]
    if order >= 1:
        G = G + jnp.swapaxes(Q[:, :, 1 : D + 1], -1, -2)
    J = G.reshape(B * R, H * D)
    return jnp.concatenate([J, qv.reshape(B * R, H)], axis=1) if bias else J


def _explicit_jacobian(P: Array, Q: Array, bias: bool) -> Array:
    B, R = Q.shape[:2]
    J = jnp.einsum("brto,bti->broi", Q, P).reshape(B * R, -1)
    return jnp.concatenate([J, Q[:, :, 0].reshape(B * R, -1)], axis=1) if bias else J


_PAIRING = (
    "rows"  # "rows" (default), "cyclic", or "auto" (cyclic from _CYCLIC_MIN_T streams per point)
)
_CYCLIC_MIN_T = 6


def _cyclic_gram(Pi: Array, Qi: Array) -> Array:
    """Within-block stream Gram by cyclic offsets: offset ``d`` pairs stream ``t`` with ``(t + d) mod T`` in one batched
    product of identical shape, so XLA compiles ``about T / 2 + 1`` GEMM shapes instead of ``T``. ``A_{T-d} = A_d^T``,
    so ``K = S + S^T`` with ``S = A_0 / 2 + sum_{d <= (T-1)/2} A_d`` (+ the half-batch ``d = T/2`` for even ``T``)."""
    B, R, T, H = Qi.shape
    Pt = jnp.moveaxis(Pi, 1, 0)
    Qt = jnp.moveaxis(Qi, 2, 0).reshape(T, B * R, H)

    def offset(Pa: Array, Qa: Array, Pb: Array, Qb: Array) -> Array:
        n = Pa.shape[0]
        gp = jnp.einsum("tbi,tci->tbc", Pa, Pb)
        gq = jnp.einsum("tmo,tno->tmn", Qa, Qb).reshape(n, B, R, B, R)
        return jnp.einsum("tbc,tbrcq->brcq", gp, gq).reshape(B * R, B * R)

    S = 0.5 * offset(Pt, Qt, Pt, Qt)
    for d in range(1, (T - 1) // 2 + 1):
        S = S + offset(Pt, Qt, jnp.roll(Pt, -d, 0), jnp.roll(Qt, -d, 0))
    if T % 2 == 0 and T > 1:
        h = T // 2
        S = S + offset(Pt[:h], Qt[:h], Pt[h:], Qt[h:])
    return S + S.T


def _streams_gram(Pi: Array, Qi: Array, Pj: Array, Qj: Array, same: bool) -> Array:
    """``sum_{t,s} (P_t P_s^T) o (Q_t Q_s^T)`` between two blocks, one stream ``t`` against all ``s`` at a time."""
    if same and (_PAIRING == "cyclic" or (_PAIRING == "auto" and Qi.shape[2] >= _CYCLIC_MIN_T)):
        return _cyclic_gram(Pi, Qi)
    Bi, Ri, Ti, H = Qi.shape
    Bj, Rj, Tj, _ = Qj.shape
    Pit, Pjt = jnp.moveaxis(Pi, 1, 0), jnp.moveaxis(Pj, 1, 0)
    Qit = jnp.moveaxis(Qi, 2, 0).reshape(Ti, Bi * Ri, H)
    Qjt = jnp.moveaxis(Qj, 2, 0).reshape(Tj, Bj * Rj, H)
    A = jnp.zeros((Bi * Ri, Bj * Rj), Qi.dtype)
    for t in range(Ti):
        s0 = t if same else 0
        n = Tj - s0
        gp = (Pit[t] @ Pjt[s0:].reshape(n * Bj, -1).T).reshape(Bi, n, Bj)
        gq = (Qit[t] @ Qjt[s0:].reshape(n * Bj * Rj, H).T).reshape(Bi, Ri, n, Bj, Rj)
        if same:
            gp = gp.at[:, 0].multiply(0.5)
        A = A + jnp.einsum("bsc,brscq->brcq", gp, gq).reshape(Bi * Ri, Bj * Rj)
    return A + A.T if same else A


def _metric_weights(
    D: int, coordinates: Sequence[int] | np.ndarray | None, diffusion: Array | None, dtype: Any
) -> Array:
    idx = distinct_indices(
        coordinates, D, message=f"coordinates must be distinct integers in [0, {D})"
    )
    s2 = jnp.ones(idx.size, dtype) if diffusion is None else jnp.asarray(diffusion, dtype) ** 2
    if s2.shape != (idx.size,):
        raise ValueError(f"diffusion must have shape ({idx.size},)")
    return jnp.zeros(D, dtype).at[idx].set(s2)


def pad_block(x: Array, *data: Array, bucket: int = 256) -> tuple[Array, ...]:
    """Pad a block to a multiple of ``bucket`` rows so a changing batch size does not recompile.

    Parameters
    ----------
    x : ``(n, D)`` points. ``data`` : per-point arrays with leading axis ``n`` (targets, weights) closed over by the
        residual. ``bucket`` : the row granularity; ``n`` is rounded up to the next multiple.

    Returns
    -------
    ``(x_pad, mask, *data_pad)`` with ``mask`` of shape ``(n_pad,)`` (``1`` for the ``n`` real rows, ``0`` for padding).
    Padded rows repeat the last real row of ``x`` and of every data array, so a residual that is finite on the real
    rows is finite on the padding. Wrap the block residual in :func:`masked_residual` with ``mask``: padded rows of ``r``
    and of every ``K`` row and column are then exactly zero, ``K[:n, :n]`` equals the unpadded Gram up to rounding, and
    with ``damping > 0`` the Gauss-Newton step is unchanged. Call this outside ``jax.jit`` (the shapes it produces are
    static), then pass the padded arrays into the jitted program.
    """
    if bucket < 1:
        raise ValueError("bucket must be a positive integer")
    n = x.shape[0]
    if n < 1:
        raise ValueError("x must contain at least one point")
    if any(jnp.shape(a)[:1] != (n,) for a in data):
        raise ValueError("every data array needs the leading axis of x")
    pad = -n % bucket

    def grow(a: Array) -> Array:
        a = jnp.asarray(a)
        return a if pad == 0 else jnp.concatenate([a, jnp.repeat(a[-1:], pad, axis=0)])

    mask = jnp.concatenate([jnp.ones(n, x.dtype), jnp.zeros(pad, x.dtype)])
    return (grow(x), mask, *(grow(a) for a in data))


def masked_residual(residual: Any, mask: Array) -> Any:
    """Multiply the rows of ``residual(...)`` by ``mask`` (see :func:`pad_block`); any residual signature.

    A :class:`~omnibias.jax.experimental.demand.Residual` keeps its declared demand.
    """
    if isinstance(residual, Residual):
        return residual.with_rows(
            lambda out: out * jnp.reshape(mask, mask.shape + (1,) * (out.ndim - 1))
        )

    def masked(*args: Any) -> Array:
        out = jnp.asarray(residual(*args))
        return out * jnp.reshape(mask, mask.shape + (1,) * (out.ndim - 1))

    return masked


def deep_residual_gram(
    blocks: Sequence[Block],
    layers: Sequence[LayerSpec],
    *,
    coordinates: Sequence[int] | np.ndarray | None = None,
    diffusion: Array | None = None,
    strategy: str = "auto",
    allow_almost_everywhere: bool = False,
) -> tuple[Array, Array]:
    r"""Residuals ``r`` and ``K = J J^T``, ``J = dr / d theta`` over every weight and bias in ``layers``.

    Parameters
    ----------
    blocks : sequence of ``(x, residual)`` or ``(x, residual, order)``
        ``x`` of shape ``(B, D)``; ``residual(u, grad, lap, x)`` returns shape
        ``(B,)`` or ``(B, R)`` from ``u (B, C)``, ``grad (B, D, C)`` and
        ``lap (B, C)``, called once on the whole block. It must act
        pointwise: row ``b`` may depend only on point ``b`` (per-point data
        such as targets can be closed over). ``order`` (default 2) is the
        highest derivative used: ``0`` passes ``grad = lap = None`` (boundary
        values), ``1`` passes ``lap = None``. Row weights belong in the
        residual.
    layers : sequence of ``(W, b, spec)``
        As for :func:`omnibias.jax.laplacian.deep_field_value_grad_laplacian`.
    coordinates, diffusion : the Laplacian is ``sum_k s_k^2 d^2 u / dx_k^2``
        over the listed inputs (all by default), ``s = diffusion`` (ones by
        default); ``grad`` always covers all ``D`` inputs.
    strategy : ``"auto"`` chooses per hidden layer and pair of blocks between
        the stream identity and explicit per-row gradients by a calibrated
        operation count; ``"streams"`` and ``"explicit"`` force one for every
        layer after the first. The first layer always uses explicit rows.
    allow_almost_everywhere : as for
        :func:`omnibias.jax.laplacian.deep_field_value_grad_laplacian`; only
        consulted for blocks of order 1 or 2.

    Returns
    -------
    r : Array of shape ``(N,)``, ``N = sum_i B_i R_i``, rows ordered by block,
        point, then residual component.
    K : Array of shape ``(N, N)``.

    Memory is ``O(N^2 T)`` for the identity's per-stream products, instead of
    the ``N x P`` Jacobian of a layer for explicit rows.
    """
    r, K, _, _ = _gram(blocks, layers, coordinates, diffusion, strategy, allow_almost_everywhere)
    return r, K


def deep_residual_gauss_newton(
    blocks: Sequence[Block],
    layers: Sequence[LayerSpec],
    damping: float | Array,
    *,
    coordinates: Sequence[int] | np.ndarray | None = None,
    diffusion: Array | None = None,
    strategy: str = "auto",
    allow_almost_everywhere: bool = False,
    conditioning: ConditioningPolicy | None = None,
) -> tuple[Array, Array, list[tuple[Array, Array | None]]]:
    r"""Gauss-Newton step in dual form, ``step = J^T (J J^T + damping I)^{-1} r``.

    The minimizer of ``|r + J delta|^2 + damping |delta|^2`` is
    ``delta = -step``, so a damped Gauss-Newton (energy natural gradient,
    Levenberg-Marquardt) update is ``theta - step``. ``K`` is factored by
    Cholesky and ``J^T`` acts through the same stream cotangents as ``K``.
    Arguments as for :func:`deep_residual_gram`; ``damping`` must make
    ``K + damping I`` positive definite. A ``conditioning`` policy with mode ``"floor"`` raises ``damping`` to the value
    at which the step is reproducible to the policy's tolerance (see :mod:`omnibias.jax.experimental.conditioning`).

    Returns
    -------
    r, K : as for :func:`deep_residual_gram`.
    step : list of ``(dW, db)`` with the shapes of ``layers``' weights and
        biases (``db`` is ``None`` where the bias is ``None``).
    """
    r, K, norm, data = _gram(
        blocks, layers, coordinates, diffusion, strategy, allow_almost_everywhere
    )
    alpha, _ = damped_solve(K, r, damping, conditioning)
    offsets = np.cumsum([0] + [rb.size for rb, _, _ in data])
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
                D = x.shape[1]
                dW = dW + Qa[:, 0].T @ x
                if order >= 1:
                    dW = dW + jnp.sum(Qa[:, 1 : D + 1], axis=0).T
            else:
                dW = dW + jnp.einsum("bto,bti->oi", Qa, P[li - 1])
            db = db + jnp.sum(Qa[:, 0], axis=0)
        step.append((jnp.asarray(dW), None if b is None else jnp.asarray(db)))
    return r, K, step


def _row_chunks(fn: Callable[..., Array], per_row: int, chunk: int, *arrays: Array) -> Array:
    """``fn`` over row chunks of at most ``chunk`` elements of intermediate (``per_row`` per row); zero rows pad."""
    B = arrays[0].shape[0]
    c = max(1, chunk // max(per_row, 1))
    if c >= B:
        return fn(*arrays)
    n = -(-B // c)
    pad = n * c - B
    padded = [
        jnp.concatenate([a, jnp.zeros((pad, *a.shape[1:]), a.dtype)]) if pad else a for a in arrays
    ]
    out: Array = jax.lax.map(lambda xs: fn(*xs), [a.reshape(n, c, *a.shape[1:]) for a in padded])
    return out.reshape(n * c, *out.shape[2:])[:B]


def _layer_diag(P: Array, Q: Array, bias: bool, strategy: str, chunk: int) -> Array:
    """``|dr_{br}/dW|^2`` (+ bias) of one dense layer from input streams ``P (B, T, in)`` and cotangents ``Q (B, R, T, out)``."""
    B, R, T, out = Q.shape
    inp = P.shape[2]
    use = strategy == "streams" or (
        strategy == "auto" and _DIAG_COST * T * (inp + R * out) < R * inp * out
    )
    if use:
        gp = jnp.einsum("bti,bsi->bts", P, P)
        gq = jnp.einsum("brto,brso->brts", Q, Q)
        d = jnp.einsum("bts,brts->br", gp, gq)
    else:

        def rows(Pc: Array, Qc: Array) -> Array:
            G = jnp.einsum("brto,bti->broi", Qc, Pc)
            return jnp.sum(G * G, axis=(2, 3))

        d = _row_chunks(rows, R * out * inp, chunk, P, Q)
    return d + jnp.sum(Q[:, :, 0] ** 2, axis=-1) if bias else d


def deep_residual_ntk_diagonal(
    blocks: Sequence[Block],
    layers: Sequence[LayerSpec],
    *,
    coordinates: Sequence[int] | np.ndarray | None = None,
    diffusion: Array | None = None,
    strategy: str = "auto",
    allow_almost_everywhere: bool = False,
) -> tuple[Array, Array]:
    r"""Residuals ``r`` and the diagonal of ``K = J J^T``, ``|dr_b / d theta|^2`` per row, without ``K`` or ``J``.

    The gradient of row ``b`` in a dense layer is ``Q_b P_b^T`` (see
    :func:`deep_residual_gram`), so its squared norm is
    ``sum_{t,s} (P_b^T P_b)_{ts} (Q_b^T Q_b)_{ts}``: ``O(B T^2 (in + out))``
    for a layer instead of ``O(B T in out)`` for explicit rows, and ``O(N)``
    memory instead of the ``N^2`` of ``K``. Block traces
    ``tr K_ii = sum diag`` give NTK-based loss weights; ``diag`` itself gives
    per-point gradient-norm importance weights. Arguments and row order as for
    :func:`deep_residual_gram`; ``strategy`` ``"auto"`` chooses per layer.
    """
    norm, w, _ = _prepare(blocks, layers, coordinates, diffusion, strategy)
    rs, ds = [], []
    for x, res, order in norm:
        r, P, Q = _streams(layers, x, res, order, w, allow_almost_everywhere)
        d: Any = 0.0
        for li, (_, b, _) in enumerate(layers):
            bias = b is not None
            if li == 0:
                H = Q[0].shape[-1]
                per_row = Q[0].shape[1] * H * (x.shape[1] + 1)

                def first(xc: Array, Qc: Array, order: int = order, bias: bool = bias) -> Array:
                    Bc, Rc = Qc.shape[:2]
                    J = _first_jacobian(xc, Qc, order, bias)
                    return jnp.sum(J * J, axis=-1).reshape(Bc, Rc)

                d = d + _row_chunks(first, per_row, _DIAG_CHUNK, x, Q[0])
            else:
                d = d + _layer_diag(P[li - 1], Q[li], bias, strategy, _DIAG_CHUNK)
        rs.append(r.reshape(-1))
        ds.append(jnp.asarray(d).reshape(-1))
    return jnp.concatenate(rs), jnp.concatenate(ds)


def _prepare(
    blocks: Sequence[Block],
    layers: Sequence[LayerSpec],
    coordinates: Sequence[int] | np.ndarray | None,
    diffusion: Array | None,
    strategy: str,
    orders: tuple[int, ...] = (0, 1, 2),
    engine: str = "full",
) -> tuple[list[tuple[Array, Callable[..., Array], int]], Array, Any]:
    if not layers:
        raise ValueError("layers must be non-empty")
    if strategy not in ("auto", "streams", "explicit"):
        raise ValueError("strategy must be 'auto', 'streams' or 'explicit'")
    norm: list[tuple[Array, Callable[..., Array], int]] = []
    for blk in blocks:
        x, res, *rest = blk
        given = int(rest[0]) if rest else None
        order = (
            block_order(res, given, engine)
            if isinstance(res, Residual)
            else (2 if given is None else given)
        )
        if order not in orders:
            raise ValueError(
                "order must be " + ", ".join(map(str, orders[:-1])) + f" or {orders[-1]}"
            )
        x = jnp.asarray(x)
        if x.ndim != 2:
            raise ValueError("each block's points must have shape (B, D)")
        norm.append((x, res, order))
    if not norm:
        raise ValueError("blocks must be non-empty")
    D = norm[0][0].shape[1]
    if any(x.shape[1] != D for x, _, _ in norm):
        raise ValueError("every block must have the same input dimension")
    W0 = jnp.asarray(layers[0][0])
    if W0.ndim != 2 or W0.shape[1] != D:
        raise ValueError(f"the first layer weight must have shape (out, {D})")
    dtype = jnp.result_type(norm[0][0], W0)
    w = _metric_weights(D, coordinates, diffusion, dtype)
    norm = [
        (x, _lower(res, order, w, engine), order) if isinstance(res, Residual) else (x, res, order)
        for x, res, order in norm
    ]
    return norm, w, dtype


def _lower(res: Residual, order: int, w: Array, engine: str) -> Callable[..., Array]:
    """The engine callback of a declared :class:`Residual` (``w``: the metric ``s^2`` per input)."""
    return lower_full(res, order, w) if engine == "full" else lower_projected(res, order)


def _gram(
    blocks: Sequence[Block],
    layers: Sequence[LayerSpec],
    coordinates: Sequence[int] | np.ndarray | None,
    diffusion: Array | None,
    strategy: str,
    allow_almost_everywhere: bool,
) -> tuple[
    Array,
    Array,
    list[tuple[Array, Callable[..., Array], int]],
    list[tuple[Array, list[Array], list[Array]]],
]:
    norm, w, dtype = _prepare(blocks, layers, coordinates, diffusion, strategy)
    data = [_streams(layers, x, res, order, w, allow_almost_everywhere) for x, res, order in norm]
    bias0 = layers[0][1] is not None
    J0 = [
        _first_jacobian(x, Q[0], order, bias0)
        for (x, _, order), (_, _, Q) in zip(norm, data, strict=True)
    ]

    def first_gram(i: int, j: int) -> Array:
        return J0[i] @ J0[j].T

    return _assemble(norm, data, layers, strategy, first_gram, dtype)


def _assemble(
    norm: list[tuple[Array, Callable[..., Array], int]],
    data: list[tuple[Array, list[Array], list[Array]]],
    layers: Sequence[LayerSpec],
    strategy: str,
    first_gram: Callable[[int, int], Array],
    dtype: Any,
    cost: Callable[[int, int], float] | None = None,
) -> tuple[
    Array,
    Array,
    list[tuple[Array, Callable[..., Array], int]],
    list[tuple[Array, list[Array], list[Array]]],
]:
    """``K`` from the first-layer blocks ``first_gram(i, j)`` and the stream identity or explicit rows beyond it.

    ``cost(i, j)`` gives the constant of the per-layer rule for block pair ``(i, j)`` (default ``_STREAMS_COST``).
    """
    sizes = [r.size for r, _, _ in data]
    K = [[jnp.zeros((ni, nj), dtype) for nj in sizes] for ni in sizes]
    nb = len(data)
    for li, (W, b, _) in enumerate(layers):
        bias = b is not None
        if li == 0:
            for i in range(nb):
                for j in range(i, nb):
                    K[i][j] = K[i][j] + first_gram(i, j)
            continue
        out, inp = jnp.shape(W)
        cache: dict[int, Array] = {}
        for i in range(nb):
            for j in range(i, nb):
                Pi, Qi = data[i][1][li - 1], data[i][2][li]
                Pj, Qj = data[j][1][li - 1], data[j][2][li]
                Ti, Tj = Qi.shape[2], Qj.shape[2]
                pairs = Ti * (Ti + 1) // 2 if i == j else Ti * Tj
                c = _STREAMS_COST if cost is None else cost(i, j)
                use = strategy == "streams" or (
                    strategy == "auto" and c * pairs * (inp + out) < inp * out
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
        jnp.concatenate([K[i][j] if j >= i else K[j][i].T for j in range(nb)], axis=1)
        for i in range(nb)
    ]
    return (
        jnp.concatenate([r.reshape(-1) for r, _, _ in data]),
        jnp.concatenate(rows, axis=0),
        norm,
        data,
    )
