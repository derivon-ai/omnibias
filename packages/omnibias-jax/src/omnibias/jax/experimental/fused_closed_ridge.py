# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Fused GPU value-and-gradient kernels for derivative-closed ridge residuals.

Same loss and algebra as :func:`~omnibias.jax.experimental.closed_ridge_residual_loss`:
``0.5 * mean((sum_t coeff_t L_t u(x) - target)^2)`` for
``u(x) = sum_h c_h sigma(w_h . x + b_h)`` with a derivative-closed ``sigma``.  With
``P_k = A_k c`` the state amplitudes and ``Q_j = sum_k P_k C_kj`` their exact
derivative rows, one row block computes ``z``, the states ``v_k(z)``, the row
residual ``r = sum_(h,k) P_k v_k - target``, ``dz = r_bar sum_j Q_j v_j`` and the
per-unit partials ``sum dz``, ``x^T dz`` and ``sum r_bar v_k`` without writing a
``(B, H)`` array.  The gradient through the amplitudes is an ordinary JAX
pullback of the ``(H,)`` moments.

Two kernel shapes: ``D <= 4`` uses one pass per row block with a column
projection; larger ``D`` sums rows in one kernel and reduces parameter partials
on a two-dimensional grid with a dot projection.  Block sizes were calibrated
for ``sin`` on one GPU model; other activations and devices are untuned.

Requires a GPU backend with Pallas Triton float64 lowering; ``interpret=True``
runs the same kernels in the Pallas interpreter (slow, any backend) for tests.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from functools import lru_cache
from typing import Any

from omnibias.jax.experimental._pallas_compat import triton_dot
from omnibias.jax.experimental.closed_ridge import (
    _STATE_FNS,
    _amplitudes,
    _rows,
    _validate,
    derivative_closed_system,
)
from omnibias.jax.experimental.tanh_ridge import RidgeTerm, _derivative_order

import jax
import jax.numpy as jnp
from jax import Array

__all__ = ["fused_closed_ridge_residual_loss", "fused_closed_ridge_value_and_grad"]

_ONE_PASS_MAX_D = 4
_MIN_TILE = 16
_ONE_PASS = {"BB": 16, "BH": 32, "warps": 4}
_SPLIT = {"BB": 64, "BH": 16, "warps": 4, "BB2": 64, "BH2": 16, "L2": 4, "warps2": 4}


def _p2(n: int) -> int:
    p = 1
    while p < n:
        p *= 2
    return p


def _pad(a: Array, n: int, axis: int = 0) -> Array:
    k = n - a.shape[axis]
    if k == 0:
        return a
    widths = [(0, 0)] * a.ndim
    widths[axis] = (0, k)
    return jnp.pad(a, widths)


def _f64(*shape: int) -> Any:
    return jax.ShapeDtypeStruct(shape, jnp.float64)  # type: ignore[no-untyped-call]


def _weighted(refs: dict[int, Any], v: dict[int, Array], hs: Any) -> Array | None:
    """``sum_k refs[k][hs] * v[k]`` over a unit block; ``None`` for an empty set."""
    acc: Array | None = None
    for k, ref in refs.items():
        t = ref[hs][None, :] * v[k]
        acc = t if acc is None else acc + t
    return acc


def _one_pass(
    x: Array,
    W: Array,
    b: Array,
    P: dict[int, Array],
    Q: dict[int, Array],
    f: Array,
    w: Array,
    names: tuple[str, ...],
    interpret: bool,
) -> tuple[Array, Array, Array, dict[int, Array]]:
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import triton as plt

    B, D = x.shape
    H = b.shape[0]
    BB = min(_ONE_PASS["BB"], _p2(B))
    BH = _ONE_PASS["BH"]
    Bp = -(-B // BB) * BB
    Hp = _p2(H) if BH >= H else -(-H // BH) * BH
    BH = min(BH, Hp)
    nB, nH = Bp // BB, Hp // BH
    kp, kq = sorted(P), sorted(Q)
    need = sorted(set(kp) | set(kq))
    xin = [_pad(x[:, d], Bp) for d in range(D)]
    win = [_pad(W[:, d], Hp) for d in range(D)]
    pin = [_pad(P[k], Hp) for k in kp]
    qin = [_pad(Q[k], Hp) for k in kq]
    n_in = 2 * D + 3 + len(kp) + len(kq)

    def kernel(*refs):  # type: ignore[no-untyped-def]
        x_refs, w_refs = refs[:D], refs[D : 2 * D]
        b_ref, f_ref, wt_ref = refs[2 * D : 2 * D + 3]
        p_refs = dict(zip(kp, refs[2 * D + 3 : 2 * D + 3 + len(kp)], strict=True))
        q_refs = dict(zip(kq, refs[2 * D + 3 + len(kp) : n_in], strict=True))
        loss_ref, gb_ref = refs[n_in], refs[n_in + 1]
        gw_refs = refs[n_in + 2 : n_in + 2 + D]
        col_refs = dict(zip(kp, refs[n_in + 2 + D :], strict=True))
        xv = [r[...] for r in x_refs]

        def states(hs: Any) -> dict[int, Array]:
            z = b_ref[hs][None, :] + xv[0][:, None] * w_refs[0][hs][None, :]
            for d in range(1, D):
                z = z + xv[d][:, None] * w_refs[d][hs][None, :]
            return {k: _STATE_FNS[names[k]](z) for k in need}

        def row_sum(v: dict[int, Array], hs: Any) -> Array:
            acc = _weighted(p_refs, v, hs)
            assert acc is not None
            return jnp.sum(acc, axis=1)

        def partials(v: dict[int, Array], hs: Any, rbar: Array) -> None:
            acc = _weighted(q_refs, v, hs)
            if acc is None:
                dz = jnp.zeros((BB, BH), jnp.float64)
            else:
                dz = rbar[:, None] * acc
            gb_ref[hs] = jnp.sum(dz, axis=0)
            for d in range(D):
                gw_refs[d][hs] = jnp.sum(dz * xv[d][:, None], axis=0)
            for k in kp:
                col_refs[k][hs] = jnp.sum(rbar[:, None] * v[k], axis=0)

        fv, wv = f_ref[...], wt_ref[...]
        if nH == 1:
            hs = pl.ds(0, BH)
            v = states(hs)
            r = row_sum(v, hs) - fv
            loss_ref[...] = 0.5 * wv * r * r
            partials(v, hs, wv * r)
        else:

            def sums(j, acc):  # type: ignore[no-untyped-def]
                hs = pl.ds(j * BH, BH)
                return acc + row_sum(states(hs), hs)

            r = jax.lax.fori_loop(0, nH, sums, jnp.zeros((BB,), jnp.float64)) - fv
            loss_ref[...] = 0.5 * wv * r * r
            rbar = wv * r

            def grads(j, carry):  # type: ignore[no-untyped-def]
                hs = pl.ds(j * BH, BH)
                partials(states(hs), hs, rbar)
                return carry

            jax.lax.fori_loop(0, nH, grads, 0)

    row = pl.BlockSpec((BB,), lambda i: (i,))
    full = pl.BlockSpec((Hp,), lambda i: (0,))
    part = pl.BlockSpec((None, Hp), lambda i: (i, 0))
    n_part = 1 + D + len(kp)
    outs = pl.pallas_call(
        kernel,
        out_shape=(_f64(Bp),) + (_f64(nB, Hp),) * n_part,
        grid=(nB,),
        in_specs=[row] * D + [full] * D + [full, row, row] + [full] * (len(kp) + len(kq)),
        out_specs=(row,) + (part,) * n_part,
        compiler_params=plt.CompilerParams(num_warps=_ONE_PASS["warps"], num_stages=1),
        interpret=interpret,
    )(*xin, *win, _pad(b, Hp), _pad(f, Bp), _pad(w, Bp), *pin, *qin)
    loss = jnp.sum(outs[0])
    bbar = jnp.sum(outs[1], axis=0)[:H]
    Wbar = jnp.stack([jnp.sum(p, axis=0)[:H] for p in outs[2 : 2 + D]], axis=1)
    col = {k: jnp.sum(p, axis=0)[:H] for k, p in zip(kp, outs[2 + D :], strict=True)}
    return loss, Wbar, bbar, col


def _split(
    x: Array,
    W: Array,
    b: Array,
    P: dict[int, Array],
    Q: dict[int, Array],
    f: Array,
    w: Array,
    names: tuple[str, ...],
    interpret: bool,
) -> tuple[Array, Array, Array, dict[int, Array]]:
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import triton as plt

    cfg = _SPLIT
    B, D = x.shape
    H = b.shape[0]
    # float64 dots on the Triton MMAv2 layout need M >= 16, N >= 8 and K >= 16, so tiny batches and widths are padded up
    BBa = max(_MIN_TILE, min(cfg["BB"], _p2(B)))
    BBb = max(_MIN_TILE, min(cfg["BB2"], _p2(B)))
    L2 = max(1, min(cfg["L2"], _p2(B) // BBb))
    RC = BBb * L2
    Bp = -(-B // math.lcm(BBa, RC)) * math.lcm(BBa, RC)
    BHa = max(_MIN_TILE, min(cfg["BH"], _p2(H)))
    BHb = max(_MIN_TILE, min(cfg["BH2"], _p2(H)))
    Hl = math.lcm(BHa, BHb)
    Hp = -(-H // Hl) * Hl
    Dp = max(16, _p2(D + 1))
    kp, kq = sorted(P), sorted(Q)
    need = sorted(set(kp) | set(kq))
    xin = _pad(_pad(x, Bp, 0), Dp, 1).at[:, D].set(1.0)
    win = _pad(_pad(W.T, Dp, 0), Hp, 1)
    bp = _pad(b, Hp)
    pin = [_pad(P[k], Hp) for k in kp]
    qin = [_pad(Q[k], Hp) for k in kq]
    nHa = Hp // BHa

    def k1(*refs):  # type: ignore[no-untyped-def]
        x_ref, w_ref, b_ref = refs[:3]
        p_refs = dict(zip(kp, refs[3 : 3 + len(kp)], strict=True))
        s_ref = refs[3 + len(kp)]
        xv = x_ref[...]

        def body(j, acc):  # type: ignore[no-untyped-def]
            hs = pl.ds(j * BHa, BHa)
            z = triton_dot(xv, w_ref[:, hs]) + b_ref[hs][None, :]
            t = _weighted(p_refs, {k: _STATE_FNS[names[k]](z) for k in kp}, hs)
            assert t is not None
            return acc + jnp.sum(t, axis=1)

        s_ref[...] = jax.lax.fori_loop(0, nHa, body, jnp.zeros((BBa,), jnp.float64))

    rowa = pl.BlockSpec((BBa,), lambda i: (i,))
    fulla = pl.BlockSpec((Hp,), lambda i: (0,))
    S = pl.pallas_call(
        k1,
        out_shape=_f64(Bp),
        grid=(Bp // BBa,),
        in_specs=[
            pl.BlockSpec((BBa, Dp), lambda i: (i, 0)),
            pl.BlockSpec((Dp, Hp), lambda i: (0, 0)),
            fulla,
        ]
        + [fulla] * len(kp),
        out_specs=rowa,
        compiler_params=plt.CompilerParams(num_warps=cfg["warps"], num_stages=1),
        interpret=interpret,
    )(xin, win, bp, *pin)
    wp = _pad(w, Bp)
    r = S - _pad(f, Bp)
    loss = jnp.sum(0.5 * wp * r * r)
    rbar = wp * r

    nB2, nH2 = Bp // RC, Hp // BHb

    def k2(*refs):  # type: ignore[no-untyped-def]
        x_ref, rb_ref, w_ref, b_ref = refs[:4]
        q_refs = dict(zip(kq, refs[4 : 4 + len(kq)], strict=True))
        gw_ref = refs[4 + len(kq)]
        col_refs = refs[5 + len(kq) :]
        hs = pl.ds(0, BHb)

        def block(i: Any, acc: tuple[Array, ...]) -> tuple[Array, ...]:
            rs = pl.ds(i * BBb, BBb)
            xv = x_ref[rs, :]
            rb = rb_ref[rs][:, None]
            z = triton_dot(xv, w_ref[:, hs]) + b_ref[hs][None, :]
            v = {k: _STATE_FNS[names[k]](z) for k in need}
            dq = _weighted(q_refs, v, hs)
            dz = jnp.zeros((BBb, BHb), jnp.float64) if dq is None else rb * dq
            gw = acc[0] + triton_dot(xv, dz, trans_a=True)
            cols = tuple(a + jnp.sum(rb * v[k], axis=0) for a, k in zip(acc[1:], kp, strict=True))
            return (gw, *cols)

        acc0 = (jnp.zeros((Dp, BHb), jnp.float64),) + tuple(
            jnp.zeros((BHb,), jnp.float64) for _ in kp
        )
        acc = block(0, acc0) if L2 == 1 else jax.lax.fori_loop(0, L2, block, acc0)
        gw_ref[...] = acc[0]
        for ref, a in zip(col_refs, acc[1:], strict=True):
            ref[...] = a

    colb = pl.BlockSpec((BHb,), lambda i, j: (j,))
    partb = pl.BlockSpec((None, BHb), lambda i, j: (i, j))
    outs = pl.pallas_call(
        k2,
        out_shape=(_f64(nB2, Dp, Hp),) + (_f64(nB2, Hp),) * len(kp),
        grid=(nB2, nH2),
        in_specs=[
            pl.BlockSpec((RC, Dp), lambda i, j: (i, 0)),
            pl.BlockSpec((RC,), lambda i, j: (i,)),
            pl.BlockSpec((Dp, BHb), lambda i, j: (0, j)),
            colb,
        ]
        + [colb] * len(kq),
        out_specs=(pl.BlockSpec((None, Dp, BHb), lambda i, j: (i, 0, j)),) + (partb,) * len(kp),
        compiler_params=plt.CompilerParams(num_warps=cfg["warps2"], num_stages=1),
        interpret=interpret,
    )(xin, rbar, win, bp, *qin)
    tot = jnp.sum(outs[0], axis=0)
    col = {k: jnp.sum(p, axis=0)[:H] for k, p in zip(kp, outs[1:], strict=True)}
    return loss, tot[:D, :H].T, tot[D, :H], col


@lru_cache(maxsize=128)
def _make(terms: tuple[RidgeTerm, ...], activation: str, interpret: bool):  # type: ignore[no-untyped-def]
    names, C, _ = derivative_closed_system(activation)
    rows = _rows(activation, max(_derivative_order(t) for t in terms))
    r = len(names)

    def amps(W: Array, c: Array) -> dict[int, Array]:
        return {k: a * c for k, a in _amplitudes(W, terms, rows, 0).items()}

    def value_and_grad(
        W: Array, b: Array, c: Array, x: Array, target: Array
    ) -> tuple[Array, tuple[Array, Array, Array]]:
        P, pull = jax.vjp(amps, W, c)
        Q: dict[int, Array] = {}
        for j in range(r):
            acc: Array | None = None
            for k, p in P.items():
                if C[k][j]:
                    term = float(C[k][j]) * p
                    acc = term if acc is None else acc + term
            if acc is not None:
                Q[j] = acc
        w = jnp.full(x.shape[:1], 1.0 / x.shape[0], x.dtype)
        if not P:
            return 0.5 * jnp.mean(target * target), (
                jnp.zeros_like(W),
                jnp.zeros_like(b),
                jnp.zeros_like(c),
            )
        call = _one_pass if W.shape[1] <= _ONE_PASS_MAX_D else _split
        loss, Wbar, bbar, col = call(x, W, b, P, Q, target, w, names, interpret)
        sW, sc = pull(col)
        return loss, (Wbar + sW, bbar, sc)

    @jax.custom_vjp
    def loss_fn(W: Array, b: Array, c: Array, x: Array, target: Array) -> Array:
        return value_and_grad(W, b, c, x, target)[0]

    def fwd(W: Array, b: Array, c: Array, x: Array, target: Array):  # type: ignore[no-untyped-def]
        v, g = value_and_grad(W, b, c, x, target)
        return v, (g, x, target)

    def bwd(res, gbar):  # type: ignore[no-untyped-def]
        (gW, gb, gc), x, target = res
        return gbar * gW, gbar * gb, gbar * gc, jnp.zeros_like(x), jnp.zeros_like(target)

    loss_fn.defvjp(fwd, bwd)
    return value_and_grad, loss_fn


def _prepare(
    W: Array, x: Array, terms: Sequence[RidgeTerm], activation: str, interpret: bool
) -> Any:
    terms_t = _validate(terms, W.shape[1])
    derivative_closed_system(activation)
    if not interpret and jax.default_backend() != "gpu":
        raise NotImplementedError(
            "fused closed-ridge kernels require a GPU backend; pass interpret=True to run them "
            "in the Pallas interpreter"
        )
    if jnp.result_type(W, x) != jnp.float64:
        raise NotImplementedError("fused closed-ridge kernels are implemented for float64 only")
    return _make(terms_t, activation, bool(interpret))


def fused_closed_ridge_value_and_grad(
    W: Array,
    b: Array,
    c: Array,
    x: Array,
    target: Array,
    terms: Sequence[RidgeTerm],
    activation: str = "sin",
    *,
    interpret: bool = False,
) -> tuple[Array, tuple[Array, Array, Array]]:
    """``(loss, (dW, db, dc))`` of the closed-ridge residual loss from one fused pass."""
    vg, _ = _prepare(W, x, terms, activation, interpret)
    return vg(W, b, c, x, target)  # type: ignore[no-any-return]


def fused_closed_ridge_residual_loss(
    W: Array,
    b: Array,
    c: Array,
    x: Array,
    target: Array,
    terms: Sequence[RidgeTerm],
    activation: str = "sin",
    *,
    interpret: bool = False,
) -> Array:
    """Differentiable closed-ridge residual loss whose gradient comes from the same fused pass.

    Gradients flow into ``W``, ``b`` and ``c``; ``x`` and ``target`` are constants.
    """
    _, loss = _prepare(W, x, terms, activation, interpret)
    return loss(W, b, c, x, target)  # type: ignore[no-any-return]
