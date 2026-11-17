# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""One-kernel stream programs for the low-order regime (Triton).

The PyTorch twin of :mod:`omnibias.jax.experimental.fused_streams`: the whole network (every hidden layer's per-channel
GEMM, the activation tower and the stream update) runs in **one forward kernel** and **one backward kernel**; a tile of
``BB`` points stays in registers through every layer and the parameter gradients are accumulated per tile and reduced once.
The kernels are generated from a traced graph of the layer update and its mechanically derived adjoint
(:mod:`omnibias.torch.experimental._triton_program`), so orders 1-4 and every supported activation share one code path.

The operators are registered with :mod:`torch.library` (``omnibias::fused_stream_forward`` / ``_backward``) with fake
implementations and an autograd formula, so they compose with ``torch.compile`` and can be captured in a CUDA graph (the
launch is asynchronous, allocates only its outputs and never synchronizes with the host). The backward kernel is first order:
a second derivative of a loss through the fused operator raises, and the stream engine
(:mod:`omnibias.torch.experimental.streams`) serves Hessians and Gauss-Newton second products.

Limits: float32/float64; ``S <= 24`` channels; width ``<= 128``; ``tanh``, ``sigmoid``, ``sin``, ``cos``, ``softplus``.
A CUDA device is required unless ``TRITON_INTERPRET=1`` (the Triton interpreter, used by the CPU tests).
"""

from __future__ import annotations

import functools
import os
import time
import weakref
from collections.abc import Sequence
from typing import Any, cast

import numpy as np
from omnibias.core.plan_algorithms import FUSED_SHARED_MEMORY_LIMIT, fused_shared_memory_bytes
from omnibias.torch.experimental._chain import fastpath_activation, points_2d, require_layers
from omnibias.torch.experimental._triton_program import Cfg, compile_kernels
from omnibias.torch.experimental.streams import (
    DirectionalJet,
    StreamResult,
    metric_block,
    plan_directions,
)

import torch
from torch import Tensor

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
_ONES: dict[tuple[Any, ...], Tensor] = {}
_FIRST_ORDER = (
    "the fused kernels are first order: a second derivative of a loss through the fused operator "
    "(Hessian, Gauss-Newton products through the operator) needs the stream engine "
    "(omnibias.torch.experimental.streams), whose program is differentiable to every order"
)


def _p2(n: int) -> int:
    p = 1
    while p < n:
        p *= 2
    return p


def _interpreting() -> bool:
    return os.environ.get("TRITON_INTERPRET", "") == "1"


_PRECISIONS = ("highest", "tf32", "default")


def _process_matmul_is_ieee() -> bool:
    """Whether this process runs float32 matrix products in full precision (the PyTorch default)."""
    matmul = torch.backends.cuda.matmul
    if hasattr(matmul, "fp32_precision"):  # PyTorch 2.9 and later; "none" defers to the global backend setting
        mode = str(matmul.fp32_precision)
        if mode == "none":
            mode = str(getattr(torch.backends, "fp32_precision", "none"))
        return mode in ("ieee", "none")
    return torch.get_float32_matmul_precision() == "highest"


def _full_precision(precision: str, dtype: torch.dtype) -> bool:
    """Whether the kernels' float32 ``tl.dot`` runs in IEEE precision.

    ``"highest"`` is IEEE, ``"tf32"`` is TF32, ``"default"`` follows the process setting of
    ``torch.set_float32_matmul_precision`` (the same meaning as ``PrecisionPolicy(matmul="default")``). A float64 product
    is always full precision, so the three modes name one kernel there.
    """
    if precision not in _PRECISIONS:
        raise ValueError(f"precision must be one of {_PRECISIONS}, got {precision!r}")
    if dtype == torch.float64 or precision == "highest":
        return True
    return precision == "default" and _process_matmul_is_ieee()


def _fit_block(dtype: str, width: int, channels: int, block: int, *, explicit: bool) -> int:
    """The tile that fits the shared memory: an unnamed block shrinks toward the minimum tile, the forward must fit."""

    def need(bb: int, backward: bool) -> int:
        return fused_shared_memory_bytes(
            dtype=dtype,
            width=width,
            channels=channels,
            block=bb,
            backward=backward,
            backend="torch",
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


# ----------------------------------------------------------------------------------------------- launchers
@functools.cache  # a handful of distinct configurations per process; parsed once, not on every launch
def _cfg_from(text: str) -> tuple[Cfg, int]:
    kind, order, nk, acts, P, Cp, Kp, BB, dtype, highest, warps = text.split("|")
    cfg = Cfg(
        kind,
        int(order),
        int(nk),
        tuple(acts.split(",")),
        int(P),
        int(Cp),
        int(Kp),
        int(BB),
        dtype,
        highest == "1",
    )
    return cfg, int(warps)


def _cfg_format(cfg: Cfg, warps: int) -> str:
    return (
        f"{cfg.kind}|{cfg.order}|{cfg.nk}|{','.join(cfg.acts)}|{cfg.P}|{cfg.Cp}|{cfg.Kp}|{cfg.BB}|{cfg.dtype}|"
        f"{1 if cfg.highest else 0}|{warps}"
    )


@functools.cache
def _cfg_format_cached(cfg: Cfg, warps: int) -> str:
    return _cfg_format(cfg, warps)


def _cfg_text(cfg: Cfg, warps: int) -> str:
    """The configuration as a plain string (the custom ops take it as an argument; no tracing-hostile serializer).

    Eager calls reuse the formatted string; a compiler trace formats it directly (it does not trace through a cache wrapper).
    """
    return _cfg_format(cfg, warps) if torch.compiler.is_compiling() else _cfg_format_cached(cfg, warps)


def _tl_dtype(cfg: Cfg) -> Any:
    import triton.language as tl  # type: ignore[import-untyped]

    return tl.float64 if cfg.dtype == "float64" else tl.float32


def _launch_forward(
    cfg: Cfg,
    warps: int,
    u1: Tensor,
    M: Tensor,
    w: Tensor,
    Ws: Sequence[Tensor],
    bs: Sequence[Tensor],
    Wo: Tensor,
    bo: Tensor,
) -> list[Tensor]:
    Bp = u1.shape[0]
    outs = [torch.empty((Bp, cfg.Cp), dtype=u1.dtype, device=u1.device) for _ in range(cfg.S)]
    mod = compile_kernels(cfg)
    mod.fused_forward[(Bp // cfg.BB,)](
        u1,
        M,
        w,
        *Ws,
        *bs,
        Wo,
        bo,
        *outs,
        P=cfg.P,
        CP=cfg.Cp,
        KP=cfg.Kp,
        BB=cfg.BB,
        DT=_tl_dtype(cfg),
        num_warps=warps,
        num_stages=1,
    )
    return outs


def _launch_backward(
    cfg: Cfg,
    warps: int,
    u1: Tensor,
    M: Tensor,
    w: Tensor,
    Ws: Sequence[Tensor],
    bs: Sequence[Tensor],
    Wo: Tensor,
    bo: Tensor,
    cts: Sequence[Tensor],
) -> list[Tensor]:
    Bp, P, Cp, Kp = u1.shape[0], cfg.P, cfg.Cp, cfg.Kp
    nT, n_hid = Bp // cfg.BB, len(cfg.acts) - 1
    dev, dt = u1.device, u1.dtype
    ubar = torch.empty_like(u1)
    mbar = torch.zeros((nT, Kp, P), dtype=dt, device=dev)
    dW_all = torch.empty((n_hid, nT, P, P), dtype=dt, device=dev)  # one buffer per kind: one reduction instead of one per layer
    db_all = torch.empty((n_hid, nT, P), dtype=dt, device=dev)
    dWo = torch.empty((nT, Cp, P), dtype=dt, device=dev)
    dbo = torch.empty((nT, Cp), dtype=dt, device=dev)
    mod = compile_kernels(cfg)
    mod.fused_backward[(nT,)](
        u1,
        M,
        w,
        *Ws,
        *bs,
        Wo,
        bo,
        *[c.contiguous() for c in cts],
        ubar,
        mbar,
        *dW_all.unbind(0),
        *db_all.unbind(0),
        dWo,
        dbo,
        P=P,
        CP=Cp,
        KP=Kp,
        BB=cfg.BB,
        DT=_tl_dtype(cfg),
        num_warps=warps,
        num_stages=1,
    )
    # [ubar, M-bar, dW (n_hid, P, P), db (n_hid, P), dWo, dbo]
    return [ubar, mbar.sum(0), dW_all.sum(1), db_all.sum(1), dWo.sum(0), dbo.sum(0)]


@torch.library.custom_op("omnibias::fused_stream_forward", mutates_args=())
def _forward_op(
    u1: Tensor,
    M: Tensor,
    w: Tensor,
    Ws: list[Tensor],
    bs: list[Tensor],
    Wo: Tensor,
    bo: Tensor,
    cfg: str,
) -> list[Tensor]:
    c, warps = _cfg_from(cfg)
    return _launch_forward(c, warps, u1, M, w, Ws, bs, Wo, bo)


@_forward_op.register_fake
def _(u1, M, w, Ws, bs, Wo, bo, cfg):  # type: ignore[no-untyped-def]
    c, _ = _cfg_from(cfg)
    return [u1.new_empty((u1.shape[0], c.Cp)) for _ in range(c.S)]


@torch.library.custom_op("omnibias::fused_stream_backward", mutates_args=())
def _backward_op(
    u1: Tensor,
    M: Tensor,
    w: Tensor,
    Ws: list[Tensor],
    bs: list[Tensor],
    Wo: Tensor,
    bo: Tensor,
    cts: list[Tensor],
    cfg: str,
) -> list[Tensor]:
    c, warps = _cfg_from(cfg)
    return _launch_backward(c, warps, u1, M, w, Ws, bs, Wo, bo, cts)


@_backward_op.register_fake
def _(u1, M, w, Ws, bs, Wo, bo, cts, cfg):  # type: ignore[no-untyped-def]
    c, _ = _cfg_from(cfg)
    n_hid = len(c.acts) - 1
    P = c.P
    return [
        torch.empty_like(u1),
        u1.new_empty((c.Kp, P)),
        u1.new_empty((n_hid, P, P)),
        u1.new_empty((n_hid, P)),
        u1.new_empty((c.Cp, P)),
        u1.new_empty((c.Cp,)),
    ]


def _backward_of_backward(ctx, *grads):  # type: ignore[no-untyped-def]
    raise NotImplementedError(_FIRST_ORDER)


_backward_op.register_autograd(_backward_of_backward)


def _setup_context(ctx, inputs, output):  # type: ignore[no-untyped-def]
    u1, M, w, Ws, bs, Wo, bo, cfg = inputs
    ctx.cfg = cfg
    ctx.n_hid = len(Ws)
    ctx.save_for_backward(u1, M, w, *Ws, *bs, Wo, bo)


def _autograd_backward(ctx, grads):  # type: ignore[no-untyped-def]
    n = ctx.n_hid
    saved = ctx.saved_tensors
    u1, M, w = saved[0], saved[1], saved[2]
    c, _ = _cfg_from(ctx.cfg)
    if u1.device.type == "cuda" and not _interpreting():
        need = fused_shared_memory_bytes(
            dtype=c.dtype, width=c.P, channels=c.S, block=c.BB, backward=True, backend="torch"
        )
        if need > FUSED_SHARED_MEMORY_LIMIT:
            raise NotImplementedError(
                f"the fused backward kernel needs about {need} B of shared memory (tile {c.BB}, width {c.P}, "
                f"{c.S} channels) against {FUSED_SHARED_MEMORY_LIMIT} B; differentiate through the stream engine"
            )
    Ws, bs = list(saved[3 : 3 + n]), list(saved[3 + n : 3 + 2 * n])
    Wo, bo = saved[3 + 2 * n], saved[4 + 2 * n]
    out = _backward_op(u1, M, w, Ws, bs, Wo, bo, list(grads), ctx.cfg)
    ubar, mbar, dWs, dbs, dWo, dbo = out
    return ubar, mbar, None, list(dWs.unbind(0)), list(dbs.unbind(0)), dWo, dbo, None


_forward_op.register_autograd(_autograd_backward, setup_context=_setup_context)


# ------------------------------------------------------------------------------------------------ front end
def fused_unsupported_reason(
    layers: Sequence[Any],
    *,
    channels: int,
    order: int = 2,
    allow_almost_everywhere: bool = False,
    dtype: Any = None,
    device: Any = None,
) -> str | None:
    """Why the fused kernels cannot run ``layers`` (``None`` when they can)."""
    try:
        require_layers(layers)
    except ValueError as exc:
        return str(exc)
    if len(layers) < 2:
        return "the fused kernels need at least one hidden layer and an affine readout"
    *hidden, (_, _, last) = layers
    if last is not None:
        return "the last layer must be affine"
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
    widths = [int(torch.as_tensor(W).shape[0]) for W, _, _ in layers[:-1]]
    if max(widths) > FUSED_MAX_WIDTH:
        return f"hidden width {max(widths)} exceeds the fused limit {FUSED_MAX_WIDTH}"
    if channels > FUSED_MAX_CHANNELS:
        return f"{channels} stream channels exceed the fused limit {FUSED_MAX_CHANNELS}"
    if dtype is not None and dtype not in (torch.float32, torch.float64):
        return f"dtype {dtype} is not float32/float64"
    if device is not None and torch.device(device).type != "cuda" and not _interpreting():
        return "the fused kernels need a CUDA device (or TRITON_INTERPRET=1)"
    return None


def _padded(a: Tensor, shape: tuple[int, ...]) -> Tensor:
    if tuple(a.shape) == shape:
        return a.contiguous()
    pad: list[int] = []
    for s, t in zip(reversed(a.shape), reversed(shape), strict=True):
        pad += [0, t - s]
    return torch.nn.functional.pad(a, pad).contiguous()


_PADDED: dict[int, tuple[Any, ...]] = {}
_ZEROS: dict[tuple[Any, ...], Tensor] = {}


def _capturing(device: torch.device) -> bool:
    """Whether a CUDA graph is being captured: an allocation made now belongs to the graph's pool and must not be kept."""
    return device.type == "cuda" and torch.cuda.is_current_stream_capturing()


def _zeros(n: int, dt: torch.dtype, device: torch.device) -> Tensor:
    """A read-only zero vector (the bias of a layer without one), formed once per length, dtype and device."""
    if torch.compiler.is_compiling():  # a traced program forms its constants itself; a module-level cache is not traceable
        return torch.zeros(n, dtype=dt, device=device)
    key = (n, dt, str(device))
    z = _ZEROS.get(key)
    if z is None:
        z = torch.zeros(n, dtype=dt, device=device)
        if not _capturing(device):
            _ZEROS[key] = z
    return z


def _padded_param(src: Any, dt: torch.dtype, device: torch.device, shape: tuple[int, ...]) -> Tensor:
    """``_padded`` of a parameter; a call that records no gradient reuses the padded copy while the parameter is unchanged.

    The copy is keyed by the parameter's identity (a weak reference), its in-place version counter and its address, so an
    optimizer step, ``copy_`` or a replaced tensor forms a new one. A call that records gradients always pads afresh,
    because the padded tensor must be part of that call's graph; a CUDA-graph capture pads inside the graph.
    """
    if not isinstance(src, Tensor):
        return _padded(torch.as_tensor(src, dtype=dt, device=device), shape)
    if tuple(src.shape) == shape and src.dtype == dt and src.device == device:
        return src.contiguous()
    if torch.compiler.is_compiling() or (torch.is_grad_enabled() and src.requires_grad) or _capturing(device):
        # a captured graph must pad inside the graph, so that an in-place update of the parameter reaches every replay
        return _padded(torch.as_tensor(src, dtype=dt, device=device), shape)
    key = id(src)
    hit = _PADDED.get(key)
    if (
        hit is not None
        and hit[0]() is src
        and hit[1] == (src._version, src.data_ptr(), dt, device, shape)
    ):
        return cast(Tensor, hit[2])
    value = _padded(torch.as_tensor(src, dtype=dt, device=device), shape)
    ref = weakref.ref(src, lambda _r, k=key: _PADDED.pop(k, None))
    _PADDED[key] = (ref, (src._version, src.data_ptr(), dt, device, shape), value)
    return value


def _tune_key(
    B: int, channels: int, P: int, dt: torch.dtype, device: torch.device
) -> tuple[Any, ...]:
    return (device.type, _p2(B), channels, P, str(dt))


def _prepare(
    a: Tensor,
    layers: Sequence[Any],
    kind: str,
    order: int,
    seeds: Tensor,
    weights: Tensor | None,
    *,
    block: int | None,
    warps: int | None,
    precision: str,
    allow_almost_everywhere: bool,
) -> tuple[Any, ...]:
    B = a.shape[0]
    dt = a.dtype
    nk = seeds.shape[0]
    channels = nk + 2 if kind == "lap" else 1 + order * nk
    reason = fused_unsupported_reason(
        layers,
        channels=channels,
        order=order,
        allow_almost_everywhere=allow_almost_everywhere,
        dtype=dt,
        device=a.device,
    )
    if reason is not None:
        raise NotImplementedError(f"fused kernel unavailable: {reason}")
    acts = tuple(
        fastpath_activation(
            spec, allow_almost_everywhere=allow_almost_everywhere, smooth=4 if order >= 3 else 2
        ).name
        for _, _, spec in layers[:-1]
    )
    W1 = torch.as_tensor(layers[0][0], dtype=dt, device=a.device)
    u1 = torch.matmul(a, W1.T)
    if layers[0][1] is not None:
        u1 = u1 + torch.as_tensor(layers[0][1], dtype=dt, device=a.device)
    widths = [int(torch.as_tensor(W).shape[0]) for W, _, _ in layers[:-1]]
    C = int(torch.as_tensor(layers[-1][0]).shape[0])
    P = max(_MIN_TILE, _p2(max(widths)))
    Cp = max(_MIN_TILE, _p2(C))
    Kp = max(_MIN_TILE, _p2(nk))
    BB0, W0 = (
        (block, warps)
        if block is not None
        else _TUNED.get(_tune_key(B, channels, P, dt, a.device), (_DEFAULT_BLOCK, _DEFAULT_WARPS))
    )
    BB = max(_MIN_TILE, _p2(BB0))
    if a.device.type == "cuda" and not _interpreting():
        BB = _fit_block(
            "float64" if dt == torch.float64 else "float32",
            max(widths),
            channels,
            BB,
            explicit=block is not None,
        )
    cfg = Cfg(
        kind,
        order,
        nk,
        acts,
        P,
        Cp,
        Kp,
        BB,
        "float64" if dt == torch.float64 else "float32",
        _full_precision(precision, dt),
    )
    Bp = -(-B // BB) * BB
    Ws, bs = [], []
    for i in range(1, len(layers) - 1):
        W, b, _ = layers[i]
        Ws.append(_padded_param(W, dt, a.device, (P, P)))
        bs.append(_zeros(P, dt, a.device) if b is None else _padded_param(b, dt, a.device, (P,)))
    bo_raw = layers[-1][1]
    bo = _zeros(Cp, dt, a.device) if bo_raw is None else _padded_param(bo_raw, dt, a.device, (Cp,))
    if weights is None:
        key = (nk, Kp, P, dt, str(a.device))
        w = _ONES.get(key)
        if w is None:
            w = _padded(torch.ones((nk, P), dtype=dt, device=a.device), (Kp, P))
            _ONES[key] = w
    else:
        w = _padded(
            torch.as_tensor(weights, dtype=dt, device=a.device)[:, None].expand(nk, P), (Kp, P)
        )
    return (
        cfg,
        warps or W0,
        _padded(u1, (Bp, P)),
        _padded(seeds.to(dt), (Kp, P)),
        w,
        tuple(Ws),
        tuple(bs),
        _padded_param(layers[-1][0], dt, a.device, (Cp, P)),
        bo,
        B,
        C,
    )


def _apply(
    cfg: Cfg,
    warps: int,
    u1: Tensor,
    M: Tensor,
    w: Tensor,
    Ws: Sequence[Tensor],
    bs: Sequence[Tensor],
    Wo: Tensor,
    bo: Tensor,
) -> list[Tensor]:
    return list(_forward_op(u1, M, w, list(Ws), list(bs), Wo, bo, _cfg_text(cfg, warps)))


def fused_second_order(
    x: Tensor,
    layers: Sequence[Any],
    *,
    diffusion: Tensor | None = None,
    coordinates: Sequence[int] | np.ndarray | None = None,
    weights: Tensor | None = None,
    allow_almost_everywhere: bool = False,
    block: int | None = None,
    warps: int | None = None,
    precision: str = "highest",
) -> StreamResult:
    """Value, squared gradient norm and metric Laplacian of a deep MLP in one fused forward kernel (diagonal form)."""
    require_layers(layers)
    a, batched = points_2d(x)
    W1 = torch.as_tensor(layers[0][0])
    if W1.ndim != 2 or W1.shape[1] != a.shape[1]:
        raise ValueError(f"the first layer weight must have shape (H, {a.shape[1]})")
    seeds = metric_block(W1.to(a.dtype), diffusion, coordinates).T
    cfg, wp, u1, M, w, Ws, bs, Wo, bo, B, C = _prepare(
        a,
        layers,
        "lap",
        2,
        seeds,
        weights,
        block=block,
        warps=warps,
        precision=precision,
        allow_almost_everywhere=allow_almost_everywhere,
    )
    outs = _apply(cfg, wp, u1, M, w, Ws, bs, Wo, bo)
    k = cfg.nk
    value, lap = outs[0][:B, :C], outs[k + 1][:B, :C]
    t = torch.stack([o[:B, :C] for o in outs[1 : 1 + k]])  # (k, B, C): one product and one reduction, not 3 per channel
    if weights is None:
        grad_sq = (t * t).sum(0)
    else:
        w_c = torch.as_tensor(weights, dtype=t.dtype, device=t.device)[:k, None, None]
        grad_sq = (w_c * t * t).sum(0)
    if not batched:
        return StreamResult(value[0], grad_sq[0], lap[0], None, None)
    return StreamResult(value, grad_sq, lap, None, None)


def fused_directional_jets(
    x: Tensor,
    layers: Sequence[Any],
    directions: Tensor,
    order: int,
    *,
    allow_almost_everywhere: bool = False,
    block: int | None = None,
    warps: int | None = None,
    precision: str = "highest",
) -> DirectionalJet:
    """Directional derivatives ``D^p_v u``, ``p = 1..order <= 4``, in one fused kernel."""
    require_layers(layers)
    if not 1 <= order <= 4:
        raise ValueError("order must be in 1..4")
    a, batched = points_2d(x)
    W1 = torch.as_tensor(layers[0][0])
    if W1.ndim != 2 or W1.shape[1] != a.shape[1]:
        raise ValueError(f"the first layer weight must have shape (H, {a.shape[1]})")
    V = torch.as_tensor(directions, dtype=a.dtype, device=a.device)
    if V.ndim != 2 or V.shape[0] != a.shape[1]:
        raise ValueError(f"directions must have shape ({a.shape[1]}, K)")
    seeds = (W1.to(a.dtype) @ V).T
    cfg, wp, u1, M, w, Ws, bs, Wo, bo, B, C = _prepare(
        a,
        layers,
        "jet",
        order,
        seeds,
        None,
        block=block,
        warps=warps,
        precision=precision,
        allow_almost_everywhere=allow_almost_everywhere,
    )
    outs = _apply(cfg, wp, u1, M, w, Ws, bs, Wo, bo)
    K = cfg.nk
    value = outs[0][:B, :C]
    derivs = (
        torch.stack([o[:B, :C] for o in outs[1:]], dim=1).reshape(B, order, K, C).transpose(1, 2)
    )
    if not batched:
        return DirectionalJet(value[0], derivs[0])
    return DirectionalJet(value, derivs)


def fused_partial_derivatives(
    x: Tensor,
    layers: Sequence[Any],
    multi_indices: Sequence[Sequence[int]],
    *,
    allow_almost_everywhere: bool = False,
    block: int | None = None,
    warps: int | None = None,
    precision: str = "highest",
) -> tuple[Tensor, Tensor]:
    """Mixed partials of order at most four by polarization over fused directional jets."""
    from omnibias.core.polarization import polarization_plan

    a, _ = points_2d(x)
    plan = polarization_plan(multi_indices, a.shape[1])
    V = plan_directions(plan, a)
    jet = fused_directional_jets(
        x,
        layers,
        V,
        plan.order,
        allow_almost_everywhere=allow_almost_everywhere,
        block=block,
        warps=warps,
        precision=precision,
    )
    from omnibias.torch.experimental.streams import _contract_plan

    return jet.value, _contract_plan(plan, jet.derivatives)


def tune_fused_block(
    x: Tensor,
    layers: Sequence[Any],
    *,
    kind: str = "lap",
    directions: Tensor | None = None,
    order: int = 2,
    candidates: Sequence[tuple[int, int]] = ((16, 4), (16, 2), (32, 4), (32, 8), (64, 8)),
    repeats: int = 20,
    **kwargs: Any,
) -> tuple[int, int]:
    """Time the forward kernel per ``(block, warps)`` candidate and remember the fastest per shape and device."""
    a, _ = points_2d(x)
    best: tuple[float, tuple[int, int]] | None = None
    for blk, wp in candidates:
        try:

            def call(b: int = blk, w: int = wp) -> Tensor:
                if kind == "lap":
                    return fused_second_order(a, layers, block=b, warps=w, **kwargs).laplacian
                assert directions is not None
                return fused_directional_jets(
                    a, layers, directions, order, block=b, warps=w, **kwargs
                ).value

            with torch.no_grad():
                call()
                if a.is_cuda:
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(repeats):
                    call()
                if a.is_cuda:
                    torch.cuda.synchronize()
                dt_s = (time.perf_counter() - t0) / repeats
        except NotImplementedError:
            raise
        except Exception:  # noqa: BLE001  # a configuration the compiler rejects is skipped
            continue
        if best is None or dt_s < best[0]:
            best = (dt_s, (blk, wp))
    if best is None:
        raise RuntimeError("no fused kernel configuration compiled")
    W1 = torch.as_tensor(layers[0][0])
    nk = (
        metric_block(W1.to(a.dtype), kwargs.get("diffusion"), kwargs.get("coordinates")).shape[1]
        if kind == "lap"
        else int(cast(Tensor, directions).shape[1])
    )
    channels = nk + 2 if kind == "lap" else 1 + order * nk
    P = max(_MIN_TILE, _p2(max(int(torch.as_tensor(W).shape[0]) for W, _, _ in layers[:-1])))
    _TUNED[_tune_key(a.shape[0], channels, P, a.dtype, a.device)] = best[1]
    return best[1]
