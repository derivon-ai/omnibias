# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""One-kernel stream programs for the low-order regime (Pallas Triton).

:mod:`omnibias.jax.experimental.streams` evaluates a deep MLP's derivative streams with a handful of XLA launches per
layer. When the batch and widths are small (the PINN regime: ``H <= 128``, a few layers, ``D`` or the number of
tangent channels ``<= 16``) the program is *launch-latency bound*: every layer pays for a ``dot_general`` and several
elementwise fusions that each move a small array through HBM. This module fuses the whole network into **one forward
kernel** and **one backward kernel**: a tile of ``BB`` points stays in registers/shared memory through every layer, the
activation tower is evaluated once per layer, and parameter gradients are accumulated per tile and reduced once.

Layout. The state of a point is a list of ``S`` channel arrays ``(BB, P)`` (``P`` = padded common width):

``lap``  (order 2)
    ``[value | k metric tangents | second-order stream]`` as in the stream engine; the squared gradient norm is
    ``sum_c w_c T_c^2`` with optional signed weights ``w``.
``jet``  (orders 1..4)
    ``[value | order-major Taylor coefficients u_1..u_n of K directions]`` with the Faa di Bruno update of
    :func:`omnibias.jax.experimental.streams.directional_jet_streams`.

Every layer is one ``(BB, P) @ (P, P)^T`` dot per channel (rows are points, so the MMA tile is full) and one elementwise
update. The backward pass recomputes the forward tile, obtains the adjoint of each *elementwise* update with
``jax.vjp`` inside the kernel (so no hand-derived adjoint can drift from the forward program, for any order or
activation) and propagates linear adjoints through the dots by transposed dots.

The prologue (``x W_1^T + b_1`` and the metric/direction images ``W_1 S``, ``W_1 V``) and the epilogue (the squared
gradient norm of the readout) stay in XLA, so gradients with respect to the inputs, the first-layer weights, the
diffusion and the directions are exact; the kernel's ``custom_vjp`` supplies the gradients of everything else.

Limits (the planner and :func:`fused_unsupported_reason` enforce them; larger problems use the XLA stream engine):
float64 dots need ``M >= 16, N >= 8, K >= 16`` so tiles are padded up to ``16``; ``S <= 24`` channels are unrolled;
``P <= 128``; the backward kernel is first order only, so Hessians of a loss through this operator (Newton, Gauss-Newton
second products) use the stream engine, whose program is differentiable to every order.
"""

from __future__ import annotations

import os
import time
import weakref
from collections.abc import Callable, Sequence
from functools import partial
from typing import Any, NamedTuple

import numpy as np
from omnibias.core.plan_algorithms import FUSED_SHARED_MEMORY_LIMIT, fused_shared_memory_bytes
from omnibias.core.polynomials import sigmoid_polynomial_coeffs, tanh_polynomial_coeffs
from omnibias.jax.experimental._chain import (
    fastpath_activation,
    points_2d,
    require_layers,
)
from omnibias.jax.experimental._pallas_compat import triton_dot
from omnibias.jax.experimental.streams import DirectionalJet, StreamResult, metric_block

import jax
import jax.numpy as jnp
from jax import Array

__all__ = [
    "FUSED_ACTIVATIONS",
    "FUSED_MAX_CHANNELS",
    "FUSED_MAX_WIDTH",
    "fused_directional_jets",
    "fused_partial_derivatives",
    "fused_second_order",
    "fused_unsupported_reason",
    "tune_fused_block",
]

FUSED_ACTIVATIONS = ("tanh", "sigmoid", "sin", "cos", "softplus")
FUSED_MAX_CHANNELS = 24
FUSED_MAX_WIDTH = 128
_MIN_TILE = 16
_DEFAULT_BLOCK = 16
_DEFAULT_WARPS = 4
_TUNED: dict[tuple[Any, ...], tuple[int, int]] = {}


def _p2(n: int) -> int:
    p = 1
    while p < n:
        p *= 2
    return p


# ------------------------------------------------------------------------------------------------ activation towers
def _horner(coeffs: Sequence[float], x: Array) -> Array:
    out = jnp.zeros_like(x) + coeffs[-1]
    for c in coeffs[-2::-1]:
        out = out * x + c
    return out


def _kernel_tower(name: str, z: Array, n: int) -> list[Array]:
    """``[sigma(z), ..., sigma^(n)(z)]`` from one base evaluation, with primitives the Triton lowering supports."""
    if name == "tanh":
        t = jnp.tanh(z)
        return [t] + [_horner(tanh_polynomial_coeffs(m), t) for m in range(1, n + 1)]
    if name == "sigmoid" or name == "softplus":
        e = jnp.exp(-jnp.abs(z))
        s = jnp.where(z >= 0, 1.0 / (1.0 + e), e / (1.0 + e))
        deriv = [_horner(sigmoid_polynomial_coeffs(m), s) for m in range(1, n + 1)] if n else []
        if name == "sigmoid":
            return [s] + deriv
        sp = jnp.maximum(z, 0.0) + jnp.log(1.0 + e)
        return [sp] if n == 0 else [sp, s] + deriv[: n - 1]
    if name in ("sin", "cos"):
        s, c = jnp.sin(z), jnp.cos(z)
        cycle = [s, c, -s, -c] if name == "sin" else [c, -s, -c, s]
        return [cycle[m % 4] for m in range(n + 1)]
    raise ValueError(f"activation {name!r} has no fused kernel tower")


def _faa_di_bruno(s: list[Array], u: list[list[Array]]) -> list[list[Array]]:
    """Physical derivatives ``(sigma o u)^(p)`` per direction; ``u[p-1][j]`` is the p-th coefficient of direction j."""
    n = len(u)
    K = len(u[0])
    out: list[list[Array]] = [[s[1] * u[0][j] for j in range(K)]]
    if n >= 2:
        out.append([s[1] * u[1][j] + s[2] * u[0][j] * u[0][j] for j in range(K)])
    if n >= 3:
        out.append(
            [
                s[1] * u[2][j] + 3.0 * s[2] * u[0][j] * u[1][j] + s[3] * u[0][j] * u[0][j] * u[0][j]
                for j in range(K)
            ]
        )
    if n >= 4:
        out.append(
            [
                s[1] * u[3][j]
                + s[2] * (3.0 * u[1][j] * u[1][j] + 4.0 * u[0][j] * u[2][j])
                + 6.0 * s[3] * u[0][j] * u[0][j] * u[1][j]
                + s[4] * u[0][j] * u[0][j] * u[0][j] * u[0][j]
                for j in range(K)
            ]
        )
    return out


class _Cfg(NamedTuple):
    """Static kernel configuration (hashable: it keys the compiled kernels)."""

    kind: str  # "lap" | "jet"
    order: int  # jet order (lap: 2)
    nk: int  # number of tangent seeds (lap: k, jet: K directions)
    acts: tuple[str, ...]  # activation after each of the n_act nonlinear layers
    P: int
    Cp: int
    Kp: int
    BB: int
    warps: int
    interpret: bool
    highest: bool

    @property
    def S(self) -> int:
        return self.nk + 2 if self.kind == "lap" else 1 + self.order * self.nk


def _update_fn(cfg: _Cfg, name: str, wrows: list[Array] | None):  # type: ignore[no-untyped-def]
    """The elementwise program of one layer: list of ``S`` pre-activation channels -> list of ``S`` outputs."""
    nk = cfg.nk

    if cfg.kind == "lap":

        def f(Z: list[Array]) -> list[Array]:
            s0, s1, s2 = _kernel_tower(name, Z[0], 2)
            q2: Array | None = None
            for c in range(nk):
                t = Z[1 + c] * Z[1 + c]
                if wrows is not None:
                    t = t * wrows[c]
                q2 = t if q2 is None else q2 + t
            assert q2 is not None
            return [s0] + [s1 * Z[1 + c] for c in range(nk)] + [s2 * q2 + s1 * Z[nk + 1]]

        return f

    n = cfg.order

    def g(Z: list[Array]) -> list[Array]:
        s = _kernel_tower(name, Z[0], n)
        u = [[Z[1 + p * nk + j] for j in range(nk)] for p in range(n)]
        new = _faa_di_bruno(s, u)
        return [s[0]] + [a for p in range(n) for a in new[p]]

    return g


# ------------------------------------------------------------------------------------------------------------ kernels
def _unpack(cfg: _Cfg, refs: Sequence[Any], n_ct: int):  # type: ignore[no-untyped-def]
    n_hid = len(cfg.acts) - 1
    i = 0
    u1, M, w = refs[0], refs[1], refs[2]
    i = 3
    Ws = refs[i : i + n_hid]
    bs = refs[i + n_hid : i + 2 * n_hid]
    i += 2 * n_hid
    Wo, bo = refs[i], refs[i + 1]
    i += 2
    cts = refs[i : i + n_ct]
    rest = refs[i + n_ct :]
    return u1, M, w, Ws, bs, Wo, bo, cts, rest


def _dot(cfg: _Cfg, a: Array, b: Array, *, ta: bool = False, tb: bool = False) -> Array:
    out: Array = triton_dot(a, b, trans_a=ta, trans_b=tb, allow_tf32=False if cfg.highest else None)
    return out


def _first_layer(cfg: _Cfg, u1_ref: Any, M_ref: Any, dt: Any) -> list[Array]:
    from jax.experimental import pallas as pl

    BB, P, nk = cfg.BB, cfg.P, cfg.nk
    zeros = jnp.zeros((BB, P), dt)
    Z = [u1_ref[...]]
    for c in range(nk):
        Z.append(jnp.broadcast_to(M_ref[pl.ds(c, 1), :], (BB, P)))
    Z += [zeros] * (cfg.S - 1 - nk)
    return Z


def _weight_rows(cfg: _Cfg, w_ref: Any) -> list[Array] | None:
    from jax.experimental import pallas as pl

    if cfg.kind != "lap":
        return None
    return [w_ref[pl.ds(c, 1), :] for c in range(cfg.nk)]


def _forward_tile(cfg: _Cfg, refs: Any, *, keep: bool):  # type: ignore[no-untyped-def]
    u1_ref, M_ref, w_ref, W_refs, b_refs, Wo_ref, bo_ref, _, _ = refs
    dt = u1_ref.dtype
    wrows = _weight_rows(cfg, w_ref)
    X: list[Array] = []
    saved: list[tuple[list[Array], list[Array], Any]] = []
    for layer, name in enumerate(cfg.acts):
        f = _update_fn(cfg, name, wrows)
        if layer == 0:
            Z = _first_layer(cfg, u1_ref, M_ref, dt)
        else:
            W = W_refs[layer - 1][...]
            Z = [_dot(cfg, X[c], W, tb=True) for c in range(cfg.S)]
            Z[0] = Z[0] + b_refs[layer - 1][...]
        if keep:
            X_new, pull = jax.vjp(f, Z)
            saved.append((X, Z, pull))
            X = list(X_new)
        else:
            X = f(Z)
    return X, saved


def _fwd_kernel(cfg: _Cfg, *refs: Any) -> None:
    parts = _unpack(cfg, refs, 0)
    out_refs = parts[8]
    X, _ = _forward_tile(cfg, parts, keep=False)
    Wo = parts[5][...]
    for c in range(cfg.S):
        o = _dot(cfg, X[c], Wo, tb=True)
        if c == 0:
            o = o + parts[6][...]
        out_refs[c][...] = o


def _bwd_kernel(cfg: _Cfg, *refs: Any) -> None:
    from jax.experimental import pallas as pl

    parts = _unpack(cfg, refs, cfg.S)
    u1_ref, M_ref, w_ref, W_refs, b_refs, Wo_ref, bo_ref, ct_refs, rest = parts
    n_hid = len(cfg.acts) - 1
    ubar_ref, mbar_ref = rest[0], rest[1]
    # the per-layer partials share one buffer per kind (n_hid, P, P) and (n_hid, 1, P): one reduction, not one per layer
    dW_ref, db_ref = (rest[2], rest[3]) if n_hid else (None, None)
    dWo_ref, dbo_ref = rest[-2], rest[-1]
    X, saved = _forward_tile(cfg, parts, keep=True)
    Wo = Wo_ref[...]
    ct = [r[...] for r in ct_refs]
    Xbar = [_dot(cfg, ct[c], Wo) for c in range(cfg.S)]
    dWo = _dot(cfg, ct[0], X[0], ta=True)
    for c in range(1, cfg.S):
        dWo = dWo + _dot(cfg, ct[c], X[c], ta=True)
    dWo_ref[...] = dWo
    dbo_ref[...] = jnp.sum(ct[0], axis=0, keepdims=True)
    for layer in reversed(range(len(cfg.acts))):
        Xp, Z, pull = saved[layer]
        (Zbar,) = pull(list(Xbar))
        if layer == 0:
            ubar_ref[...] = Zbar[0]
            mbar_ref[...] = jnp.zeros((cfg.Kp, cfg.P), Zbar[0].dtype)
            for c in range(cfg.nk):
                mbar_ref[pl.ds(c, 1), :] = jnp.sum(Zbar[1 + c], axis=0, keepdims=True)
        else:
            W = W_refs[layer - 1][...]
            dW = _dot(cfg, Zbar[0], Xp[0], ta=True)
            for c in range(1, cfg.S):
                dW = dW + _dot(cfg, Zbar[c], Xp[c], ta=True)
            dW_ref[layer - 1] = dW
            db_ref[layer - 1] = jnp.sum(Zbar[0], axis=0, keepdims=True)
            Xbar = [_dot(cfg, Zbar[c], W) for c in range(cfg.S)]


def _sds(shape: tuple[int, ...], dtype: Any) -> Any:
    return jax.ShapeDtypeStruct(shape, dtype)  # type: ignore[no-untyped-call]


def _ready(tree: Any) -> Any:
    return jax.tree_util.tree_map(lambda a: a.block_until_ready(), tree)


def _specs(cfg: _Cfg, Bp: int, n_ct: int) -> tuple[list[Any], Callable[[int], Any]]:
    from jax.experimental import pallas as pl

    P, Cp, Kp, BB = cfg.P, cfg.Cp, cfg.Kp, cfg.BB
    n_hid = len(cfg.acts) - 1

    def full(*shape: int) -> Any:
        return pl.BlockSpec(shape, lambda i: (0,) * len(shape))

    def row(width: int) -> Any:
        return pl.BlockSpec((BB, width), lambda i: (i, 0))

    ins = [row(P), full(Kp, P), full(Kp, P)]
    ins += [full(P, P)] * n_hid + [full(1, P)] * n_hid + [full(Cp, P), full(1, Cp)]
    ins += [row(Cp)] * n_ct
    return ins, row


def _compiler_params(cfg: _Cfg) -> Any:
    from jax.experimental.pallas import triton as plt

    return plt.CompilerParams(num_warps=cfg.warps, num_stages=1)


def _fwd_call(
    cfg: _Cfg,
    u1: Array,
    M: Array,
    w: Array,
    Ws: tuple[Array, ...],
    bs: tuple[Array, ...],
    Wo: Array,
    bo: Array,
) -> Any:
    from jax.experimental import pallas as pl

    Bp = u1.shape[0]
    ins, row = _specs(cfg, Bp, 0)
    outs = tuple(_sds((Bp, cfg.Cp), u1.dtype) for _ in range(cfg.S))
    return pl.pallas_call(
        partial(_fwd_kernel, cfg),
        out_shape=outs,
        grid=(Bp // cfg.BB,),
        in_specs=ins,
        out_specs=tuple(row(cfg.Cp) for _ in range(cfg.S)),
        compiler_params=_compiler_params(cfg),
        interpret=cfg.interpret,
    )(u1, M, w, *Ws, *bs, Wo, bo)


def _bwd_call(cfg: _Cfg, u1, M, w, Ws, bs, Wo, bo, cts):  # type: ignore[no-untyped-def]
    from jax.experimental import pallas as pl

    Bp = u1.shape[0]
    P, Cp, Kp = cfg.P, cfg.Cp, cfg.Kp
    nT = Bp // cfg.BB
    n_hid = len(cfg.acts) - 1
    ins, row = _specs(cfg, Bp, cfg.S)
    dt = u1.dtype

    def part(*shape: int) -> Any:
        return pl.BlockSpec((None, *shape), lambda i: (i, *(0,) * len(shape)))

    def stacked(*shape: int) -> Any:
        return pl.BlockSpec((n_hid, None, *shape), lambda i: (0, i, *(0,) * len(shape)))

    shapes = [_sds((Bp, P), dt), _sds((nT, Kp, P), dt)]
    specs = [row(P), part(Kp, P)]
    if n_hid:
        shapes += [_sds((n_hid, nT, P, P), dt), _sds((n_hid, nT, 1, P), dt)]
        specs += [stacked(P, P), stacked(1, P)]
    shapes += [_sds((nT, Cp, P), dt), _sds((nT, 1, Cp), dt)]
    specs += [part(Cp, P), part(1, Cp)]
    out = pl.pallas_call(
        partial(_bwd_kernel, cfg),
        out_shape=tuple(shapes),
        grid=(nT,),
        in_specs=ins,
        out_specs=tuple(specs),
        compiler_params=_compiler_params(cfg),
        interpret=cfg.interpret,
    )(u1, M, w, *Ws, *bs, Wo, bo, *cts)
    ubar, mbar = out[0], jnp.sum(out[1], axis=0)
    dWs = tuple(jnp.sum(out[2], axis=1)) if n_hid else ()
    dbs = tuple(jnp.sum(out[3], axis=1)) if n_hid else ()
    dWo = jnp.sum(out[-2], axis=0)
    dbo = jnp.sum(out[-1], axis=0)
    return ubar, mbar, dWs, dbs, dWo, dbo


_FIRST_ORDER = (
    "the fused kernels are first order: a second derivative of a loss through the fused operator "
    "(Hessian, Gauss-Newton products through the operator, forward-over-reverse) needs the stream engine "
    "(omnibias.jax.experimental.streams), whose program is differentiable to every order"
)


@partial(jax.custom_jvp, nondiff_argnums=(0,))
def _bwd_guard(cfg, u1, M, w, Ws, bs, Wo, bo, cts):  # type: ignore[no-untyped-def]
    return _bwd_call(cfg, u1, M, w, Ws, bs, Wo, bo, cts)


@_bwd_guard.defjvp
def _bwd_guard_jvp(cfg, primals, tangents):  # type: ignore[no-untyped-def]
    raise NotImplementedError(_FIRST_ORDER)


@partial(jax.custom_jvp, nondiff_argnums=(0,))
def _fwd_guard(cfg, u1, M, w, Ws, bs, Wo, bo):  # type: ignore[no-untyped-def]
    return _fwd_call(cfg, u1, M, w, Ws, bs, Wo, bo)


@_fwd_guard.defjvp
def _fwd_guard_jvp(cfg, primals, tangents):  # type: ignore[no-untyped-def]
    raise NotImplementedError(_FIRST_ORDER)


@partial(jax.custom_vjp, nondiff_argnums=(0,))
def _fused(cfg: _Cfg, u1, M, w, Ws, bs, Wo, bo):  # type: ignore[no-untyped-def]
    return _fwd_guard(cfg, u1, M, w, Ws, bs, Wo, bo)


def _fused_fwd(cfg, u1, M, w, Ws, bs, Wo, bo):  # type: ignore[no-untyped-def]
    return _fwd_guard(cfg, u1, M, w, Ws, bs, Wo, bo), (u1, M, w, Ws, bs, Wo, bo)


def _fused_bwd(cfg, res, cts):  # type: ignore[no-untyped-def]
    u1, M, w, Ws, bs, Wo, bo = res
    if not cfg.interpret:
        need = fused_shared_memory_bytes(
            dtype=np.dtype(u1.dtype).name, width=cfg.P, channels=cfg.S, block=cfg.BB, backward=True
        )
        if need > FUSED_SHARED_MEMORY_LIMIT:
            raise NotImplementedError(
                f"the fused backward kernel needs about {need} B of shared memory (tile {cfg.BB}, width {cfg.P}, "
                f"{cfg.S} channels) against {FUSED_SHARED_MEMORY_LIMIT} B; differentiate through the stream engine"
            )
    ubar, mbar, dWs, dbs, dWo, dbo = _bwd_guard(cfg, u1, M, w, Ws, bs, Wo, bo, tuple(cts))
    return ubar, mbar, jnp.zeros_like(w), dWs, dbs, dWo, dbo


_fused.defvjp(_fused_fwd, _fused_bwd)


# ------------------------------------------------------------------------------------------------- front end
def _padded(a: Array, shape: tuple[int, ...]) -> Array:
    widths = [(0, t - s) for s, t in zip(a.shape, shape, strict=True)]
    return a if not any(w for _, w in widths) else jnp.pad(a, widths)


_PADDED: dict[int, tuple[Any, ...]] = {}


def _padded_param(src: Any, dt: Any, shape: tuple[int, ...]) -> Array:
    """``_padded`` of a parameter converted to ``dt``; a concrete array keeps its padded copy while it is alive.

    Arrays are immutable, so the copy is valid for as long as the array object is; an eager call that repeats with the same
    parameters (inference, a fixed evaluation set) pads once. A traced value, and a concrete one padded while a trace is
    open (the result is then a tracer), is padded inside its program as before and never cached.
    """
    if not isinstance(src, Array) or isinstance(src, jax.core.Tracer):
        return _padded(jnp.asarray(src, dt), shape)
    key = id(src)
    hit = _PADDED.get(key)
    if hit is not None and hit[0]() is src and hit[1] == (str(np.dtype(dt)), shape):
        return hit[2]  # type: ignore[no-any-return]
    value = _padded(jnp.asarray(src, dt), shape)
    if isinstance(value, jax.core.Tracer):  # a concrete array closed over by a trace pads to a tracer: never keep one
        return value
    _PADDED[key] = (
        weakref.ref(src, lambda _r, k=key: _PADDED.pop(k, None)),
        (str(np.dtype(dt)), shape),
        value,
    )
    return value


class _Prepared(NamedTuple):
    cfg: _Cfg
    u1: Array
    M: Array
    w: Array
    Ws: tuple[Array, ...]
    bs: tuple[Array, ...]
    Wo: Array
    bo: Array
    B: int
    C: int


def fused_unsupported_reason(
    layers: Sequence[Any],
    *,
    channels: int,
    order: int = 2,
    allow_almost_everywhere: bool = False,
    dtype: Any = None,
) -> str | None:
    """Why the fused kernels cannot run ``layers`` (``None`` when they can). Cheap and static: no tracing."""
    try:
        require_layers(layers)
    except ValueError as exc:
        return str(exc)
    if len(layers) < 2:
        return "the fused kernels need at least one hidden layer and an affine readout"
    *hidden, (_, _, last) = layers
    if last is not None:
        return "the last layer must be affine"
    widths = [int(np.shape(W)[0]) for W, _, _ in layers[:-1]]
    for i, (_, _, spec) in enumerate(hidden):
        if spec is None:
            return (
                f"hidden layer {i} is affine; the fused kernels fuse nonlinear hidden layers only"
            )
        try:
            name = fastpath_activation(
                spec, allow_almost_everywhere=allow_almost_everywhere, smooth=4 if order >= 3 else 2
            ).name
        except ValueError as exc:
            return str(exc)
        if name not in FUSED_ACTIVATIONS:
            return f"activation {name!r} has no fused kernel tower (supported: {', '.join(FUSED_ACTIVATIONS)})"
    if max(widths) > FUSED_MAX_WIDTH:
        return f"hidden width {max(widths)} exceeds the fused limit {FUSED_MAX_WIDTH}"
    if channels > FUSED_MAX_CHANNELS:
        return f"{channels} stream channels exceed the fused limit {FUSED_MAX_CHANNELS}"
    if dtype is not None and np.dtype(dtype) not in (np.dtype("float32"), np.dtype("float64")):
        return f"dtype {np.dtype(dtype)} is not float32/float64"
    return None


def _fit_block(dtype: str, width: int, channels: int, block: int, *, explicit: bool) -> int:
    """The tile that fits the shared memory: an unnamed block shrinks toward the minimum tile, the forward must fit."""

    def need(bb: int, backward: bool) -> int:
        return fused_shared_memory_bytes(
            dtype=dtype, width=width, channels=channels, block=bb, backward=backward
        )

    if need(_MIN_TILE, False) > FUSED_SHARED_MEMORY_LIMIT:
        raise NotImplementedError(
            f"fused kernel unavailable: the forward kernel needs about {need(_MIN_TILE, False)} B of shared memory at "
            f"the smallest tile against {FUSED_SHARED_MEMORY_LIMIT} B (width {width}, {channels} channels, {dtype}); "
            "use the stream engine"
        )
    if not explicit:
        while block > _MIN_TILE and need(block, True) > FUSED_SHARED_MEMORY_LIMIT:
            block //= 2
    return block


def _prepare(
    a: Array,
    layers: Sequence[Any],
    kind: str,
    order: int,
    seeds: Array,
    weights: Array | None,
    *,
    block: int | None,
    warps: int | None,
    interpret: bool,
    precision: str,
    allow_almost_everywhere: bool,
) -> _Prepared:
    B = a.shape[0]
    dt = jnp.result_type(a, *[jnp.asarray(W) for W, _, _ in layers])
    nk = seeds.shape[0]
    channels = nk + 2 if kind == "lap" else 1 + order * nk
    reason = fused_unsupported_reason(
        layers,
        channels=channels,
        order=order,
        allow_almost_everywhere=allow_almost_everywhere,
        dtype=dt,
    )
    if reason is not None:
        raise NotImplementedError(f"fused kernel unavailable: {reason}")
    acts = tuple(
        fastpath_activation(
            spec, allow_almost_everywhere=allow_almost_everywhere, smooth=4 if order >= 3 else 2
        ).name
        for _, _, spec in layers[:-1]
    )
    W1, b1, _ = layers[0]
    W1 = jnp.asarray(W1, dt)
    full = _full_precision(precision)
    prec = jax.lax.Precision.HIGHEST if full else None
    u1 = jnp.matmul(a.astype(dt), W1.T, precision=prec)
    if b1 is not None:
        u1 = u1 + jnp.asarray(b1, dt)
    widths = [int(jnp.shape(W)[0]) for W, _, _ in layers[:-1]]
    C = int(jnp.shape(layers[-1][0])[0])
    P = max(_MIN_TILE, _p2(max(widths)))
    Cp = max(_MIN_TILE, _p2(C))
    Kp = max(_MIN_TILE, _p2(nk))
    BB0, W0 = (
        (block, warps)
        if block is not None
        else _TUNED.get(_tune_key(B, channels, P, dt), (_DEFAULT_BLOCK, _DEFAULT_WARPS))
    )
    BB = max(_MIN_TILE, _p2(BB0))
    if not interpret:
        BB = _fit_block(np.dtype(dt).name, max(widths), channels, BB, explicit=block is not None)
    cfg = _Cfg(
        kind,
        order,
        nk,
        acts,
        P,
        Cp,
        Kp,
        BB,
        warps or W0 or _DEFAULT_WARPS,
        bool(interpret),
        full,
    )
    Bp = -(-B // BB) * BB
    Ws, bs = [], []
    for i in range(1, len(layers) - 1):
        W, b, _ = layers[i]
        Ws.append(_padded_param(W, dt, (P, P)))
        bs.append(
            _padded(
                (jnp.zeros((jnp.shape(W)[0],), dt) if b is None else jnp.asarray(b, dt))[None],
                (1, P),
            )
            if b is None or not isinstance(b, Array) or isinstance(b, jax.core.Tracer)
            else _padded_param(b, dt, (P,))[None]
        )
    bo_raw = layers[-1][1]
    bo = _padded(
        (jnp.zeros((C,), dt) if bo_raw is None else jnp.asarray(bo_raw, dt))[None], (1, Cp)
    )
    wmat = jnp.ones((nk,), dt) if weights is None else jnp.asarray(weights, dt)
    w = _padded(jnp.broadcast_to(wmat[:, None], (nk, P)), (Kp, P))
    return _Prepared(
        cfg,
        _padded(u1, (Bp, P)),
        _padded(seeds.astype(dt), (Kp, P)),
        w,
        tuple(Ws),
        tuple(bs),
        _padded_param(layers[-1][0], dt, (Cp, P)),
        bo,
        B,
        C,
    )


def _resolve_interpret(interpret: bool | None) -> bool:
    if interpret is not None:
        return interpret
    if jax.default_backend() != "gpu":
        if os.environ.get("OMNIBIAS_PALLAS_INTERPRET", "") == "1":  # tests and debugging on a CPU
            return True
        raise NotImplementedError(
            "the fused kernels need a GPU backend; pass interpret=True for the Pallas interpreter, set "
            "OMNIBIAS_PALLAS_INTERPRET=1, or use the stream engine"
        )
    return False


_PRECISIONS = ("highest", "tf32", "default")


def _full_precision(precision: str) -> bool:
    """Whether the kernels' ``dot`` runs in full float32 precision.

    ``"highest"`` is full precision, ``"tf32"`` is TF32, ``"default"`` follows ``jax.default_matmul_precision`` (unset: the
    backend default, which is TF32 on GPUs that have it), the same meaning as ``PrecisionPolicy(matmul="default")``.
    """
    if precision not in _PRECISIONS:
        raise ValueError(f"precision must be one of {_PRECISIONS}, got {precision!r}")
    if precision != "default":
        return precision == "highest"
    return str(jax.config.jax_default_matmul_precision) in ("highest", "float32")


def _tune_key(B: int, channels: int, P: int, dt: Any) -> tuple[Any, ...]:
    return (jax.default_backend(), _p2(B), channels, P, str(np.dtype(dt)))


def fused_second_order(
    x: Array,
    layers: Sequence[Any],
    *,
    diffusion: Array | None = None,
    coordinates: Sequence[int] | np.ndarray | None = None,
    weights: Array | None = None,
    allow_almost_everywhere: bool = False,
    interpret: bool | None = None,
    block: int | None = None,
    warps: int | None = None,
    precision: str = "highest",
) -> StreamResult:
    """Value, squared gradient norm and metric Laplacian of a deep MLP in one fused forward kernel.

    Same contract as :func:`omnibias.jax.experimental.streams.second_order_streams` for the diagonal form
    (``diffusion``, ``coordinates`` and signed ``weights``); the Jacobian and drift outputs are not produced.
    Differentiable once (``jax.grad``/``jax.vjp``) with respect to ``x``, every layer and the metric; a second derivative
    of a loss through this function raises (use the stream engine, which differentiates to every order).
    """
    require_layers(layers)
    a, batched = points_2d(x)
    W1 = jnp.asarray(layers[0][0])
    if W1.ndim != 2 or W1.shape[1] != a.shape[1]:
        raise ValueError(f"the first layer weight must have shape (H, {a.shape[1]})")
    M = metric_block(W1, diffusion, coordinates)  # (H1, r)
    seeds = M.T
    prep = _prepare(
        a,
        layers,
        "lap",
        2,
        seeds,
        weights,
        block=block,
        warps=warps,
        interpret=_resolve_interpret(interpret),
        precision=precision,
        allow_almost_everywhere=allow_almost_everywhere,
    )
    outs = _fused(prep.cfg, prep.u1, prep.M, prep.w, prep.Ws, prep.bs, prep.Wo, prep.bo)
    B, C, k = prep.B, prep.C, prep.cfg.nk
    value = outs[0][:B, :C]
    lap = outs[k + 1][:B, :C]
    grad_sq = jnp.zeros_like(value)
    for c in range(k):
        t = outs[1 + c][:B, :C]
        grad_sq = grad_sq + (t * t if weights is None else jnp.asarray(weights, t.dtype)[c] * t * t)
    if not batched:
        return StreamResult(value[0], grad_sq[0], lap[0], None, None)
    return StreamResult(value, grad_sq, lap, None, None)


def fused_directional_jets(
    x: Array,
    layers: Sequence[Any],
    directions: Array,
    order: int,
    *,
    allow_almost_everywhere: bool = False,
    interpret: bool | None = None,
    block: int | None = None,
    warps: int | None = None,
    precision: str = "highest",
) -> DirectionalJet:
    """Directional derivatives ``D^p_v u``, ``p = 1..order <= 4``, in one fused kernel (see ``directional_jet_streams``)."""
    require_layers(layers)
    if not 1 <= order <= 4:
        raise ValueError("order must be in 1..4")
    a, batched = points_2d(x)
    W1 = jnp.asarray(layers[0][0])
    if W1.ndim != 2 or W1.shape[1] != a.shape[1]:
        raise ValueError(f"the first layer weight must have shape (H, {a.shape[1]})")
    V = jnp.asarray(directions, a.dtype)
    if V.ndim != 2 or V.shape[0] != a.shape[1]:
        raise ValueError(f"directions must have shape ({a.shape[1]}, K)")
    seeds = (W1 @ V).T  # (K, H1)
    prep = _prepare(
        a,
        layers,
        "jet",
        order,
        seeds,
        None,
        block=block,
        warps=warps,
        interpret=_resolve_interpret(interpret),
        precision=precision,
        allow_almost_everywhere=allow_almost_everywhere,
    )
    outs = _fused(prep.cfg, prep.u1, prep.M, prep.w, prep.Ws, prep.bs, prep.Wo, prep.bo)
    B, C, K = prep.B, prep.C, prep.cfg.nk
    value = outs[0][:B, :C]
    derivs = jnp.stack([o[:B, :C] for o in outs[1:]], axis=1).reshape(B, order, K, C)
    derivs = jnp.swapaxes(derivs, 1, 2)
    if not batched:
        return DirectionalJet(value[0], derivs[0])
    return DirectionalJet(value, derivs)


def fused_partial_derivatives(
    x: Array,
    layers: Sequence[Any],
    multi_indices: Sequence[Sequence[int]],
    *,
    allow_almost_everywhere: bool = False,
    interpret: bool | None = None,
    block: int | None = None,
    warps: int | None = None,
    precision: str = "highest",
) -> tuple[Array, Array]:
    """Mixed partials of order at most four by polarization over fused directional jets (``partial_derivatives``)."""
    from omnibias.core.polarization import polarization_plan

    a, _ = points_2d(x)
    plan = polarization_plan(multi_indices, a.shape[1])
    V = jnp.asarray(plan.direction_matrix().astype(np.float64).T, a.dtype)
    jet = fused_directional_jets(
        x,
        layers,
        V,
        plan.order,
        allow_almost_everywhere=allow_almost_everywhere,
        interpret=interpret,
        block=block,
        warps=warps,
        precision=precision,
    )
    from omnibias.jax.experimental.streams import _contract_plan

    return jet.value, _contract_plan(plan, jet.derivatives)


# ---------------------------------------------------------------------------------------------------- autotune
def tune_fused_block(
    x: Array,
    layers: Sequence[Any],
    *,
    kind: str = "lap",
    directions: Array | None = None,
    order: int = 2,
    candidates: Sequence[tuple[int, int]] = ((16, 4), (16, 2), (32, 4), (32, 8), (64, 8)),
    repeats: int = 20,
    **kwargs: Any,
) -> tuple[int, int]:
    """Time the forward kernel for each ``(block, warps)`` candidate and remember the fastest per shape/device.

    The result is cached in-process (keyed on device, padded batch, channels, width, dtype) and used by every later fused
    call that does not pass ``block``/``warps``. Candidates the lowering rejects are skipped.
    """
    a, _ = points_2d(x)
    best: tuple[float, tuple[int, int]] | None = None
    probe = None
    for blk, wp in candidates:
        try:
            if kind == "lap":
                fn = jax.jit(
                    lambda x_, b=blk, w=wp: fused_second_order(
                        x_, layers, block=b, warps=w, **kwargs
                    ).laplacian
                )
            else:
                if directions is None:
                    raise ValueError("directions are required for kind='jet'")
                fn = jax.jit(
                    lambda x_, b=blk, w=wp: fused_directional_jets(
                        x_, layers, directions, order, block=b, warps=w, **kwargs
                    ).value
                )
            _ready(fn(a))
            t0 = time.perf_counter()
            for _ in range(repeats):
                probe = fn(a)
            _ready(probe)
            dt_s = (time.perf_counter() - t0) / repeats
        except NotImplementedError:
            raise
        except Exception:  # noqa: BLE001  # a tile configuration the lowering rejects is skipped
            continue
        if best is None or dt_s < best[0]:
            best = (dt_s, (blk, wp))
    if best is None:
        raise RuntimeError("no fused kernel configuration compiled")
    W1 = jnp.asarray(layers[0][0])
    nk = (
        metric_block(W1, kwargs.get("diffusion"), kwargs.get("coordinates")).shape[1]
        if kind == "lap"
        else int(jnp.shape(directions)[1])  # type: ignore[arg-type]
    )
    channels = nk + 2 if kind == "lap" else 1 + order * nk
    P = max(_MIN_TILE, _p2(max(int(jnp.shape(W)[0]) for W, _, _ in layers[:-1])))
    _TUNED[_tune_key(a.shape[0], channels, P, jnp.result_type(a, W1))] = best[1]
    return best[1]
