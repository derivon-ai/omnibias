# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Stacked second-order stream engine for deep MLPs.

A forward Laplacian carries, per layer and per point, the value ``a``, the tangents ``T = d a / d s_c`` along the columns of
a metric factor, optional drift tangents ``Q = d a / d v_j``, and the second-order stream ``L = sum_c d^2 a / d s_c^2``.
This module keeps all of them in one array ``X`` of shape ``(B, S, H)``::

    channel 0                value a
    channels 1 .. k          tangents T            (metric channels)
    channels k+1 .. k+kp     companions P          (Gram form only)
    next q channels          drift tangents Q
    last channel             second-order stream L

so that each hidden layer is **one** ``dot_general`` (the affine map is linear in every stream; the bias enters the value
channel only) followed by **one** elementwise update that evaluates the activation tower once and writes the next stack.
There is no ``lax.cond``, no factorization and no host synchronization anywhere in the program; every operation is a plain
differentiable array expression, so parameter derivatives of every order are exact.

Two ways to carry the metric ``A = S S^T`` through the first layer
(``M = W_1 S`` has shape ``(H, r)``):

``diag`` -- the ``r`` columns of ``M`` are the tangent channels. The squared gradient norm of a later unit is
    ``sum_c w_c T_c^2`` (signed weights ``w`` give an indefinite diagonal metric). Cost per layer ``~ B (r + 2) H^2``.
``gram`` -- no factorization of ``G = M M^T``. The tangents are taken with respect to the ``H`` first-layer
    pre-activations (``T`` starts as the identity) and a companion ``P = T G`` is carried with them, so the squared gradient
    norm is ``rowsum(T * P)``. This is exact for any symmetric ``G``, including indefinite ones, and costs
    ``~ 2 B H^3`` per layer independent of ``r`` and ``D``; only the first layer ever sees ``D``.

``auto`` picks ``diag`` while ``r <= 2 H``.
"""

from __future__ import annotations

import contextlib
import contextvars
from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np
from omnibias.jax.experimental._chain import (
    Skip,
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

__all__ = [
    "AUTO_SPLIT_MAX_HIDDEN",
    "DEFAULT_LAYOUT",
    "DirectionalJet",
    "JET_LAYOUT",
    "JET_SPLIT_MIN_POINTS",
    "StreamLayout",
    "StreamResult",
    "TRAINING_LAYOUT",
    "activation_tower",
    "bilaplacian_by_jets",
    "directional_jet_streams",
    "laplacian_gradient",
    "metric_block",
    "partial_derivatives",
    "resolve_layout",
    "second_order_streams",
    "training_layout",
]


#: Under ``jit`` a ``hess_form`` of size ``m`` is carried as ``m`` coordinate tangents with the metric applied in-graph (``B m^2 H``
#: per layer for the quadratic form) instead of the Gram stack (``2 H + 2`` channels, an elementwise partner product) when
#: ``m <= ratio * H``. On a CPU it was no slower in either pass at every size measured (``m / H`` from 0.25 to 0.83: 0.15 to 1.0 of
#: the forward time and 0.10 to 0.73 of the gradient time of the Gram stack); on an A100 it is faster in both passes where
#: ``m / H <= 1/4``, no faster in the gradient at 0.5, and 1.4 times slower in the forward pass from 0.67, so an accelerator keeps
#: the Gram stack above that ratio.
HESS_FORM_COORDINATES_MAX_RATIO = 2.0
HESS_FORM_COORDINATES_MAX_RATIO_ACCELERATOR = 0.25


#: The layout ``second_order_streams`` uses when it is not given one: ``"stack"``, ``"split"`` or ``"auto"`` (``TRAINING_LAYOUT`` on a
#: CPU and while a training program is traced under :func:`training_layout`, which :func:`compile_training_step` enters; ``"stack"``
#: otherwise). On one CPU thread (seven narrow deep-network cells, two timing modes, nine paired fresh-process rounds) the split
#: layout took 0.48 to 0.72 of the stack layout's time in the value-and-gradient pass in 14 of 14 cases and 0.63 to 0.88 in the
#: forward pass in 11 of 14 (one forward case, `D = 16`, four layers of width 128, took 1.09). On an A100 (17 cells, 4 to 6 rounds)
#: it took 0.50 to 0.88 in the value-and-gradient pass in 25 of 34 cases and was slower in none; in the forward pass it took 0.56 to
#: 0.92 in 12 cases and 1.06 to 1.62 in 7 (the four-layer cells and one `D = 2` cell). A forward-only call on an accelerator
#: therefore takes the split layout for float64 when the network has at most ``AUTO_SPLIT_MAX_HIDDEN`` hidden layers and keeps the
#: stack for deeper ones (40 cells, four fresh-process rounds against the stack, float64: two hidden layers, device time 13 of 28
#: cells faster and none slower, eager time 3 faster and none slower; four hidden layers, 2 of 12 cells slower at `H = 128`). In
#: float32 the same comparison found 14 of 28 faster and none slower with TF32 matrix products and 11 of 28 faster and one slower
#: (5 percent, `D = 4,096`, `H = 64`, 32 points) with the highest precision, so float32 keeps the stack and
#: ``layout="split"`` selects the other side. A call traced while its weights or points are being differentiated (``jax.grad``,
#: ``jax.value_and_grad``, ``jax.jvp`` around the call) takes the split layout as well.
DEFAULT_LAYOUT = "auto"
#: Most hidden layers for which ``"auto"`` takes the split layout in a forward-only float64 call on an accelerator (a CPU and
#: training programs take it at every depth and dtype). Set 0 to keep the stack for every forward-only call on an accelerator.
AUTO_SPLIT_MAX_HIDDEN = 2
#: The layout ``"auto"`` takes on a CPU and for programs traced for parameter gradients.
TRAINING_LAYOUT = "split"
#: The layout :func:`directional_jet_streams` uses when it is not given one: ``"stack"``, ``"split"`` or ``"auto"`` (the split layout
#: on a CPU at ``JET_SPLIT_MIN_POINTS`` points or more, the stack otherwise). On one CPU thread (third order, one direction, six
#: cells, three fresh processes) the split layout took 0.59 to 0.91 of the stack's time in the forward pass in 6 of 6 cells and
#: 0.53 to 0.89 in the value-and-gradient pass in the 4 cells of 256 or 2,048 points; it was ahead of both nested ``jax.jvp`` and
#: ``jax.experimental.jet`` in all 24 CPU comparisons (1.02 to 1.61 and 1.28 to 2.41 times). At 32 points the gradient pass took
#: 1.07 to 1.25 of the stack's time, so the stack stays below the threshold. On an A100 the split layout's gradient pass took 1.06
#: to 1.20 of the stack's in every cell and its forward pass 0.81 to 1.07 (mixed), so an accelerator keeps the stack.
JET_LAYOUT = "auto"
#: Fewest points at which ``JET_LAYOUT="auto"`` takes the split layout on a CPU (measured at 32, 256 and 2,048 points).
JET_SPLIT_MIN_POINTS = 128
_TRAINING: contextvars.ContextVar[bool] = contextvars.ContextVar("omnibias_stream_training", default=False)


@contextlib.contextmanager
def training_layout() -> Iterator[None]:
    """Trace the programs built inside the block for parameter gradients: ``layout="auto"`` takes ``TRAINING_LAYOUT``."""
    token = _TRAINING.set(True)
    try:
        yield
    finally:
        _TRAINING.reset(token)


_DIFFERENTIATION_TRACERS = frozenset({"JVPTracer", "LinearizeTracer"})


def _differentiated(arrays: Sequence[Any]) -> bool:
    """Whether one of ``arrays`` is being differentiated by ``jax.grad``, ``jax.value_and_grad`` or ``jax.jvp`` while it is traced.

    The tracer classes are told by name (``JVPTracer``; ``LinearizeTracer`` where ``jax.grad`` linearizes directly), as neither is
    public API for ``LinearizeTracer``. A function that is traced first and differentiated afterwards (``jax.grad(jax.jit(f))``) is
    not seen: its trace carries no differentiation, and ``training_layout`` or an explicit ``layout`` is how such a program asks
    for the split layout.
    """
    return any(
        _DIFFERENTIATION_TRACERS.intersection(c.__name__ for c in type(a).__mro__) for a in arrays if a is not None
    )


def resolve_layout(layout: str | None = None, *arrays: Any, hidden: int | None = None, dtype: Any = None) -> str:
    """``"stack"`` or ``"split"``: an explicit layout, else ``DEFAULT_LAYOUT``.

    ``"auto"`` is the split layout on a CPU, in a training context, when one of ``arrays`` (the weights, biases and points of the
    call) is being differentiated as the program is traced, and in a forward-only float64 call (``dtype`` given) on an accelerator
    whose network has ``hidden`` (given) at most ``AUTO_SPLIT_MAX_HIDDEN`` hidden layers; the stack otherwise.
    """
    layout = DEFAULT_LAYOUT if layout is None else layout
    if layout == "auto":
        training = _TRAINING.get() or jax.default_backend() == "cpu" or _differentiated(arrays)
        shallow = hidden is not None and hidden <= AUTO_SPLIT_MAX_HIDDEN and dtype is not None and jnp.dtype(dtype) == jnp.float64
        layout = TRAINING_LAYOUT if training or shallow else "stack"
    if layout not in ("stack", "split"):
        raise ValueError(f"layout must be 'stack', 'split' or 'auto', not {layout!r}")
    return layout


def _coordinate_metric(m: int, H: int, form: str) -> bool:
    """Whether a traced ``hess_form`` of size ``m`` on ``H`` channels is carried as ``m`` coordinate tangents."""
    if form == "diag":
        return True
    ratio = (
        HESS_FORM_COORDINATES_MAX_RATIO
        if jax.default_backend() == "cpu"
        else HESS_FORM_COORDINATES_MAX_RATIO_ACCELERATOR
    )
    return m <= ratio * H


class StreamLayout(NamedTuple):
    """Static channel layout of a stack ``(B, S, H)``."""

    k: int  # metric tangent channels
    kp: int  # companion channels (Gram form), else 0
    q: int  # drift channels
    gram: bool

    @property
    def size(self) -> int:
        return 2 + self.k + self.kp + self.q


class StreamResult(NamedTuple):
    value: Array  # (B, C)
    grad_sq: Array  # (B, C): |S^T grad u|^2 (signed weights / Gram: the A-form)
    laplacian: Array  # (B, C): tr(A hess u)
    directional: Array | None  # (B, q, C): v . grad u
    jacobian: Array | None  # (B, k, C): the metric tangents (full input Jacobian when requested)


def _check_skip(i: int, U: Array, X: Array) -> None:
    if i == 0:
        raise ValueError("the first layer cannot be a Skip block (the width changes)")
    if U.shape != X.shape:
        raise ValueError(
            f"layer {i} is a Skip block and needs a square weight, got width {U.shape[-1]} for {X.shape[-1]}"
        )


def activation_tower(spec: Any, u: Array, order: int) -> list[Array]:
    """``[sigma(u), sigma'(u), ..., sigma^(order)(u)]`` from one base evaluation when the activation has a tower."""
    tower = getattr(spec, "tower", None)
    if tower is not None:
        stacked = tower(u, order)
        return [stacked[n] for n in range(order + 1)]
    return [spec.forward(u)] + [spec.fastpath(u, n) for n in range(1, order + 1)]


def metric_block(
    W1: Array, diffusion: Array | None, coordinates: Sequence[int] | np.ndarray | None
) -> Array:
    """``W1 S`` with ``S`` restricted to ``coordinates`` (no selection matrix is formed)."""
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


def _q2(U: Array, lay: StreamLayout, weights: Array | None) -> Array:
    """``|grad|^2`` of every unit from the metric channels of a stack: ``(B, H)``."""
    T = U[:, 1 : 1 + lay.k]
    if lay.gram:
        return jnp.sum(T * U[:, 1 + lay.k : 1 + lay.k + lay.kp], axis=1)
    if weights is None:
        return jnp.sum(T * T, axis=1)
    if weights.ndim == 2:  # a metric matrix ``A`` on coordinate tangents: ``sum_cd A_cd T_c T_d``
        return jnp.sum(jnp.einsum("cd,bdh->bch", weights, T) * T, axis=1)
    return jnp.einsum("c,bch->bh", weights, T * T)


def _update(
    U: Array,
    bias: Array | None,
    spec: Any,
    lay: StreamLayout,
    weights: Array | None,
    q2: Array | None,
) -> tuple[Array, Array]:
    """One elementwise pass: activation tower once, then the next stack ``(B, S, H)`` and ``s1 = sigma'(u)``."""
    u = U[:, 0] if bias is None else U[:, 0] + bias
    s0, s1, s2 = activation_tower(spec, u, 2)
    if q2 is None:
        q2 = _q2(U, lay, weights)
    lap = s2 * q2 + s1 * U[:, -1]
    mid = s1[:, None, :] * U[:, 1:-1]
    return jnp.concatenate([s0[:, None, :], mid, lap[:, None, :]], axis=1), s1


def _after_identity_block(X: Array, W: Array, s1: Array, k: int) -> Array:
    """``einsum("bsh,oh->bso", X, W)`` when channels ``1..k`` of ``X`` are ``diag(s1)``: the first layer's identity block.

    Each such channel multiplies ``W`` by one nonzero entry, so the ``k`` channels cost ``B k H`` multiplications instead of a
    ``(B k, H)`` by ``(H, H')`` product; the other channels keep their products, split around the block.
    """
    head = jnp.einsum("bh,oh->bo", X[:, 0], W)[:, None]
    ident = s1[:, :, None] * W.T[None]
    tail = jnp.einsum("bsh,oh->bso", X[:, 1 + k :], W)
    return jnp.concatenate([head, ident, tail], axis=1)


def _first_stack(
    x: Array,
    W1: Array,
    b1: Array | None,
    *,
    diffusion: Array | None,
    point_diffusion: Array | None,
    coordinates: Sequence[int] | np.ndarray | None,
    directions: Array | None,
    batched: bool,
    form: str,
    full_jacobian: bool,
    weights: Array | None,
    hess_form: Array | None = None,
) -> tuple[Array, StreamLayout, Array, Array | None]:
    """Pre-activation stack of layer 1, its layout, the unit squared-norm seed ``q2`` and the channel weights."""
    B, dim = x.shape
    H = W1.shape[0]
    u1 = x @ W1.T
    if b1 is not None:
        u1 = u1 + jnp.asarray(b1)
    dtype = u1.dtype
    q = 0
    Q = None
    if directions is not None:
        v = jnp.asarray(directions, dtype)
        if v.ndim == 2 and v.shape[0] == dim:
            Q = jnp.broadcast_to((W1 @ v).T[None], (B, v.shape[1], H))
        elif batched and v.ndim == 3 and v.shape[:2] == (B, dim):
            Q = jnp.einsum("hd,bdm->bmh", W1, v)
        else:
            raise ValueError("directions must have shape (D, q) or (B, D, q)")
        q = Q.shape[1]
    if full_jacobian:
        if diffusion is not None or point_diffusion is not None or coordinates is not None:
            raise ValueError("the full input Jacobian excludes a metric or coordinate selection")
        T = jnp.broadcast_to(W1.T[None], (B, dim, H))
        lay = StreamLayout(dim, 0, q, False)
        q2 = jnp.broadcast_to(jnp.sum(W1 * W1, axis=1), (B, H))
        U = jnp.concatenate(
            [u1[:, None], T] + ([] if Q is None else [Q]) + [jnp.zeros_like(u1)[:, None]], axis=1
        )
        return U, lay, q2, None
    if point_diffusion is not None:
        if diffusion is not None:
            raise ValueError("pass either diffusion or point_diffusion, not both")
        M0 = metric_block(W1, None, coordinates)
        m = M0.shape[1]
        s = jnp.asarray(point_diffusion, dtype)
        s = s if batched else s[None]
        if s.shape != (B, m):
            shape = f"({B}, {m})" if batched else f"({m},)"
            raise ValueError(f"point_diffusion must have shape {shape}")
        r = m
        use_diag = form == "diag" or (form == "auto" and r <= 2 * H)
        if use_diag:
            T = M0.T[None] * s[:, :, None]  # (B, m, H)
            q2 = (s * s) @ (M0 * M0).T
            lay = StreamLayout(r, 0, q, False)
            U = jnp.concatenate(
                [u1[:, None], T] + ([] if Q is None else [Q]) + [jnp.zeros_like(u1)[:, None]],
                axis=1,
            )
            return U, lay, q2, None
        s2 = s * s
        if 2 * B >= H:
            iu, ju = np.triu_indices(H)
            index = np.zeros((H, H), dtype=np.int64)
            index[iu, ju] = index[ju, iu] = np.arange(iu.size)
            G = (s2 @ (M0[iu] * M0[ju]).T)[:, index]
        else:
            G = jnp.einsum("bhm,gm->bhg", M0[None] * s2[:, None, :], M0)
        q2 = jnp.diagonal(G, axis1=-2, axis2=-1)
        return _gram_stack(u1, G, Q, B, H, dtype), StreamLayout(H, H, q, True), q2, None
    if hess_form is not None:
        if diffusion is not None or weights is not None:
            raise ValueError("hess_form excludes diffusion and weights")
        M0 = metric_block(W1, None, coordinates)
        m = M0.shape[1]
        A = jnp.asarray(hess_form, dtype)
        if A.shape != (m, m):
            raise ValueError(f"hess_form must have shape ({m}, {m})")
        A = 0.5 * (A + A.T)
        try:
            A_np: np.ndarray | None = np.asarray(A)
        except jax.errors.TracerArrayConversionError:
            A_np = None
        if form != "gram" and A_np is not None:
            lam, V = np.linalg.eigh(A_np)
            keep = np.abs(lam) > 1e-13 * max(float(np.max(np.abs(lam))), 1e-300)
            r = int(keep.sum())
            if form == "diag" or r <= 2 * H:
                S = jnp.asarray(V[:, keep] * np.sqrt(np.abs(lam[keep])), dtype)
                signs = jnp.asarray(np.sign(lam[keep]), dtype)
                M = M0 @ S
                T = jnp.broadcast_to(M.T[None], (B, r, H))
                q2 = (M * M) @ signs
                U = jnp.concatenate(
                    [u1[:, None], T] + ([] if Q is None else [Q]) + [jnp.zeros_like(u1)[:, None]],
                    axis=1,
                )
                return U, StreamLayout(r, 0, q, False), jnp.broadcast_to(q2, (B, H)), signs
        if A_np is None and form != "gram" and _coordinate_metric(m, H, form):
            # traced metric (under jit): the m coordinate tangents with the metric applied in-graph, no eigendecomposition
            T = jnp.broadcast_to(M0.T[None], (B, m, H))
            q2 = jnp.broadcast_to(jnp.sum((M0 @ A) * M0, axis=1), (B, H))
            U = jnp.concatenate(
                [u1[:, None], T] + ([] if Q is None else [Q]) + [jnp.zeros_like(u1)[:, None]],
                axis=1,
            )
            return U, StreamLayout(m, 0, q, False), q2, A
        G = M0 @ A @ M0.T
        q2 = jnp.broadcast_to(jnp.diagonal(G), (B, H))
        return _gram_stack(u1, G, Q, B, H, dtype), StreamLayout(H, H, q, True), q2, None
    M = metric_block(W1, diffusion, coordinates)
    r = M.shape[1]
    use_diag = form == "diag" or (form == "auto" and r <= 2 * H)
    if use_diag:
        T = jnp.broadcast_to(M.T[None], (B, r, H))
        lay = StreamLayout(r, 0, q, False)
        w = None if weights is None else jnp.asarray(weights, dtype)
        if w is not None and w.shape != (r,):
            raise ValueError(f"weights must have shape ({r},)")
        q2 = jnp.sum(M * M, axis=1) if w is None else (M * M) @ w
        U = jnp.concatenate(
            [u1[:, None], T] + ([] if Q is None else [Q]) + [jnp.zeros_like(u1)[:, None]], axis=1
        )
        return U, lay, jnp.broadcast_to(q2, (B, H)), w
    if weights is not None:
        raise ValueError(
            "channel weights apply to the diag form only; pass the signed metric as `diffusion`/Gram"
        )
    G = M @ M.T
    q2 = jnp.broadcast_to(jnp.diagonal(G), (B, H))
    return _gram_stack(u1, G, Q, B, H, dtype), StreamLayout(H, H, q, True), q2, None


def _gram_stack(u1: Array, G: Array, Q: Array | None, B: int, H: int, dtype: Any) -> Array:
    eye = jnp.broadcast_to(jnp.eye(H, dtype=dtype)[None], (B, H, H))
    P = jnp.broadcast_to(G[None], (B, H, H)) if G.ndim == 2 else G
    parts = [u1[:, None], eye, P] + ([] if Q is None else [Q]) + [jnp.zeros_like(u1)[:, None]]
    return jnp.concatenate(parts, axis=1)


def _q2_split(T: Array, P: Array | None, weights: Array | None) -> Array:
    """``|grad|^2`` of every unit from the metric tangents ``T`` ``(B, k, H)`` (and the companions ``P`` of the Gram form)."""
    if P is not None:
        return jnp.sum(T * P, axis=1)
    if weights is None:
        return jnp.sum(T * T, axis=1)
    if weights.ndim == 2:
        return jnp.sum(jnp.einsum("cd,bdh->bch", weights, T) * T, axis=1)
    return jnp.einsum("c,bch->bh", weights, T * T)


def _split_layers(
    U: Array,
    lay: StreamLayout,
    q2_seed: Array | None,
    weights: Array | None,
    layers: Sequence[LayerSpec],
    allow_almost_everywhere: bool,
    remat: bool,
) -> tuple[Array, Array, Array | None, Array | None, Array]:
    """The recursion of the stack with every stream in its own array: ``(value, T, P, Q, L)``.

    ``value`` and ``L`` (the second-order stream) are ``(B, H)``; ``T`` holds the ``k`` metric tangents ``(B, k, H)``, ``P`` the
    ``kp`` companions of the Gram form and ``Q`` the ``q`` drift tangents (``None`` when absent). Each layer is one product per
    stream; the tangents are never sliced out of, or concatenated into, a larger array.
    """
    k, kp, q = lay.k, lay.kp, lay.q
    val, L = U[:, 0], U[:, -1]
    T = U[:, 1 : 1 + k]
    P = U[:, 1 + k : 1 + k + kp] if kp else None
    Q = U[:, 1 + k + kp : 1 + k + kp + q] if q else None
    for i, (W, b, spec) in enumerate(layers):
        weight = bias = None
        if i:
            weight = jnp.asarray(W)
            if weight.ndim != 2 or weight.shape[1] != val.shape[-1]:
                raise ValueError(f"layer {i} weight must have shape (out, {val.shape[-1]})")
            bias = None if b is None else jnp.asarray(b)
        skip = isinstance(spec, Skip)
        inner = spec.spec if isinstance(spec, Skip) else spec
        resolved = (
            None
            if inner is None
            else fastpath_activation(inner, allow_almost_everywhere=allow_almost_everywhere)
        )

        def layer(
            val: Array,
            T: Array,
            P: Array | None,
            Q: Array | None,
            L: Array,
            weight: Array | None,
            bias: Array | None,
            *,
            i: int = i,
            skip: bool = skip,
            resolved: Any = resolved,
        ) -> tuple[Array, Array, Array | None, Array | None, Array]:
            if i:
                assert weight is not None
                u = jnp.einsum("bh,oh->bo", val, weight)
                if bias is not None:
                    u = u + bias
                Tp = jnp.einsum("bch,oh->bco", T, weight)
                Pp = None if P is None else jnp.einsum("bch,oh->bco", P, weight)
                Qp = None if Q is None else jnp.einsum("bch,oh->bco", Q, weight)
                Lp = jnp.einsum("bh,oh->bo", L, weight)
                q2 = None
            else:
                u, Tp, Pp, Qp, Lp, q2 = val, T, P, Q, L, q2_seed  # layer 1's bias is already in the value
            if skip:
                if i == 0:
                    raise ValueError("the first layer cannot be a Skip block (the width changes)")
                if u.shape != val.shape:
                    raise ValueError(
                        f"layer {i} is a Skip block and needs a square weight, got width {u.shape[-1]} for {val.shape[-1]}"
                    )
            if resolved is None:
                return u, Tp, Pp, Qp, Lp
            s0, s1, s2 = activation_tower(resolved, u, 2)
            if q2 is None:
                q2 = _q2_split(Tp, Pp, weights)
            new = (
                s0,
                s1[:, None, :] * Tp,
                None if Pp is None else s1[:, None, :] * Pp,
                None if Qp is None else s1[:, None, :] * Qp,
                s2 * q2 + s1 * Lp,
            )
            if skip:
                return (
                    new[0] + val,
                    new[1] + T,
                    None if Pp is None else new[2] + P,
                    None if Qp is None else new[3] + Q,
                    new[4] + L,
                )
            return new

        val, T, P, Q, L = (jax.checkpoint(layer) if remat else layer)(val, T, P, Q, L, weight, bias)
    return val, T, P, Q, L


def second_order_streams(
    x: Array,
    layers: Sequence[LayerSpec],
    *,
    diffusion: Array | None = None,
    point_diffusion: Array | None = None,
    coordinates: Sequence[int] | np.ndarray | None = None,
    directions: Array | None = None,
    weights: Array | None = None,
    form: str = "auto",
    full_jacobian: bool = False,
    allow_almost_everywhere: bool = False,
    hess_form: Array | None = None,
    remat: bool = False,
    identity_block: bool = False,
    layout: str | None = None,
) -> StreamResult:
    """Value, squared gradient norm, Laplacian (metric trace of the Hessian) and drift derivatives of a deep MLP.

    ``x`` has shape ``(D,)`` or ``(B, D)``; ``layers`` are ``(W, b, spec)`` with ``W`` of shape ``(out, in)`` and
    ``spec=None`` an affine layer. ``diffusion`` is ``S`` of the metric ``A = S S^T`` (``(m, r)`` or diagonal ``(m,)``) on
    the inputs in ``coordinates``; ``point_diffusion`` a per-point diagonal factor ``(B, m)``; ``weights`` signed channel
    weights ``(r,)`` (diag form), making the metric ``S diag(w) S^T``; ``directions`` drift directions ``(D, q)`` or
    ``(B, D, q)``. ``form`` is ``"auto"``, ``"diag"`` or ``"gram"``. ``full_jacobian=True`` carries the identity metric
    on all inputs and returns the Jacobian ``(B, D, C)`` with the Laplacian (the classic forward Laplacian).
    ``remat=True`` wraps each layer in ``jax.checkpoint``: a gradient through the streams then stores the stack at every layer
    boundary and recomputes the inside of each layer in the backward pass, trading arithmetic for activation memory (default
    off; the values are unchanged). Under ``hess_form`` traced by ``jit``, a matrix of size ``m <= 2H`` (``m <= H/4`` on an accelerator, where the
    Gram stack is faster above that) is carried as ``m`` coordinate tangents with the metric applied in-graph
    (``form="gram"`` keeps the Gram stack).
    ``identity_block=True`` (Gram form, default off) multiplies the first layer's identity channels, which are
    ``diag(sigma'(u))``, by the next weight as ``B k H`` products instead of one ``(B k, H)`` by ``(H, H')`` matrix product. In
    the audit's forward pass it took 0.83 and 0.75 of the time on two medium and large Gram cells and was not faster on the
    small ones; a gradient through it held up to 4.6 percent more temporary memory, which is why it is off by default.
    ``layout`` is ``"stack"`` (one array per layer, one product per layer) or ``"split"`` (value, tangents and second-order
    stream in separate arrays: no slice or concatenate copy of the tangents, and a readout that forms only the channels that
    are used); the values agree to rounding. ``"split"`` applies to both forms and is not combined with ``identity_block``
    (which keeps the stack). ``"auto"`` is the split layout on a CPU, under :func:`training_layout`, while the weights or
    points are being differentiated as the call is traced, and in a forward-only float64 call on an accelerator with at most
    ``AUTO_SPLIT_MAX_HIDDEN`` (2) hidden layers; it is the stack otherwise; ``None`` takes ``DEFAULT_LAYOUT``.
    """
    require_layers(layers)
    if form not in ("auto", "diag", "gram"):
        raise ValueError(f"form must be 'auto', 'diag' or 'gram', not {form!r}")
    layout = resolve_layout(
        layout,
        x,
        *(leaf for layer in layers for leaf in layer[:2]),
        hidden=sum(1 for layer in layers if layer[2] is not None),
        dtype=getattr(x, "dtype", None),
    )
    a, batched = points_2d(x)
    B, dim = a.shape
    W1 = jnp.asarray(layers[0][0])
    if W1.ndim != 2 or W1.shape[1] != dim:
        raise ValueError(f"the first layer weight must have shape (H, {dim})")
    U, lay, q2_seed, w = _first_stack(
        a,
        W1,
        layers[0][1],
        diffusion=diffusion,
        point_diffusion=point_diffusion,
        coordinates=coordinates,
        directions=directions,
        batched=batched,
        form=form,
        full_jacobian=full_jacobian,
        weights=weights,
        hess_form=hess_form,
    )
    matrix_metric = (
        w is not None and w.ndim == 2
    )  # coordinate tangents of a traced hess_form: not metric tangents
    if layout == "split" and not identity_block:
        value, T, P, Q, lap = _split_layers(U, lay, q2_seed, w, layers, allow_almost_everywhere, remat)
        grad_sq = _q2_split(T, P, w)
        directional = Q
        jacobian = T if (full_jacobian or not (lay.gram or matrix_metric)) else None
        if not batched:
            value, grad_sq, lap = value[0], grad_sq[0], lap[0]
            directional = None if directional is None else directional[0]
            jacobian = None if jacobian is None else jacobian[0]
        return StreamResult(value, grad_sq, lap, directional, jacobian)
    q2: Array | None = q2_seed
    X = U
    identity_scale: Array | None = (
        None  # s1 of layer 0 on the Gram form: channels 1..H of X are then diag(s1)
    )
    for i, (W, b, spec) in enumerate(layers):
        weight = bias = None
        if i:
            weight = jnp.asarray(W)
            if weight.ndim != 2 or weight.shape[1] != X.shape[-1]:
                raise ValueError(f"layer {i} weight must have shape (out, {X.shape[-1]})")
            bias = None if b is None else jnp.asarray(b)
        skip = isinstance(spec, Skip)
        inner = spec.spec if isinstance(spec, Skip) else spec
        resolved = (
            None
            if inner is None
            else fastpath_activation(inner, allow_almost_everywhere=allow_almost_everywhere)
        )

        def layer(
            X: Array,
            weight: Array | None,
            bias: Array | None,
            w: Array | None,
            q2: Array | None,
            carried: Array | None,
            *,
            i: int = i,
            skip: bool = skip,
            resolved: Any = resolved,
        ) -> tuple[Array, Array | None]:
            if i:
                assert weight is not None
                U = (
                    jnp.einsum("bsh,oh->bso", X, weight)
                    if carried is None
                    else _after_identity_block(X, weight, carried, lay.k)
                )
                q2 = None
            else:
                U, bias = X, None  # layer 1's bias is already in channel 0
            if skip:
                _check_skip(i, U, X)
            if resolved is None:
                return (U if bias is None else U.at[:, 0].add(bias)), None
            new, s1 = _update(U, bias, resolved, lay, w, q2)
            return (new + X if skip else new), (
                s1 if i == 0 and lay.gram and identity_block else None
            )

        X, identity_scale = (jax.checkpoint(layer) if remat else layer)(
            X, weight, bias, w, q2 if i == 0 else None, identity_scale
        )
    value = X[:, 0]
    lap = X[:, -1]
    grad_sq = _q2(X, lay, w)
    q0 = 1 + lay.k + lay.kp
    directional = None if lay.q == 0 else X[:, q0 : q0 + lay.q]
    jacobian = X[:, 1 : 1 + lay.k] if (full_jacobian or not (lay.gram or matrix_metric)) else None
    if not batched:
        value, grad_sq, lap = value[0], grad_sq[0], lap[0]
        directional = None if directional is None else directional[0]
        jacobian = None if jacobian is None else jacobian[0]
    return StreamResult(value, grad_sq, lap, directional, jacobian)


# --------------------------------------------------------------------------------------------- directional Taylor streams
class DirectionalJet(NamedTuple):
    value: Array  # (B, C)
    derivatives: (
        Array  # (B, K, n, C): the p-th directional derivative D^p_v u along direction v_j, p = 1..n
    )


def _faa_di_bruno(s: list[Array], u: list[Array]) -> list[Array]:
    """Physical derivatives ``(sigma o u)^(p)``, ``p = 1..n``, from the tower ``s[0..n]`` and ``u[1..n]`` (index ``p-1``)."""
    n = len(u)
    s1 = s[1][:, None, :]
    out = [s1 * u[0]]
    if n >= 2:
        s2 = s[2][:, None, :]
        out.append(s1 * u[1] + s2 * u[0] ** 2)
    if n >= 3:
        s3 = s[3][:, None, :]
        out.append(s1 * u[2] + 3.0 * s2 * u[0] * u[1] + s3 * u[0] ** 3)
    if n >= 4:
        s4 = s[4][:, None, :]
        out.append(
            s1 * u[3]
            + s2 * (3.0 * u[1] ** 2 + 4.0 * u[0] * u[2])
            + 6.0 * s3 * u[0] ** 2 * u[1]
            + s4 * u[0] ** 4
        )
    return out


def _split_jets(
    u1: Array,
    first: Array,
    layers: Sequence[LayerSpec],
    order: int,
    B: int,
    K: int,
    batched: bool,
    *,
    allow_almost_everywhere: bool,
    remat: bool,
) -> DirectionalJet:
    """The split layout of :func:`directional_jet_streams`: the value ``(B, H)`` and the coefficients ``(B, K, H)`` apart.

    The first layer is linear, so its coefficients of order two and higher are zero: the Faa di Bruno terms that contain
    them are not formed (the stack carries them as zero channels). Every later layer is ``order + 1`` products with ``W``.
    """
    smooth = 4 if order >= 3 else 2
    state: tuple[Array, list[Array]] | None = None
    for i, (W, b, spec) in enumerate(layers):
        skip = isinstance(spec, Skip)
        inner = spec.spec if isinstance(spec, Skip) else spec
        resolved = (
            None
            if inner is None
            else fastpath_activation(inner, allow_almost_everywhere=allow_almost_everywhere, smooth=smooth)
        )
        weight = bias = None
        if i:
            weight = jnp.asarray(W)
            assert state is not None
            if weight.ndim != 2 or weight.shape[1] != state[0].shape[-1]:
                raise ValueError(f"layer {i} weight must have shape (out, {state[0].shape[-1]})")
            bias = None if b is None else jnp.asarray(b)

        def layer(
            value: Array,
            coeffs: list[Array],
            weight: Array | None,
            bias: Array | None,
            *,
            i: int = i,
            skip: bool = skip,
            resolved: Any = resolved,
        ) -> tuple[Array, list[Array]]:
            if i:
                U0 = value @ weight.T  # type: ignore[union-attr]
                if bias is not None:
                    U0 = U0 + bias
                U = [c @ weight.T for c in coeffs]  # type: ignore[union-attr]
            else:
                U0, U = value, coeffs
            if skip:
                _check_skip(i, U0, value)
            if resolved is None:
                return U0, U
            tower = activation_tower(resolved, U0, order)
            if i:
                new = _faa_di_bruno(tower, U)
            else:  # only the first coefficient is nonzero: (sigma o u)^(p) = sigma^(p) u1^p
                s = [tower[p][:, None, :] for p in range(1, order + 1)]
                new = [s[p] * U[0] ** (p + 1) for p in range(order)]
            if skip:
                return tower[0] + value, [n + c for n, c in zip(new, coeffs, strict=True)]
            return tower[0], new

        if i == 0:
            state = (u1, [first] + [jnp.zeros_like(first)] * (order - 1))
        assert state is not None
        value, coeffs = state
        state = (jax.checkpoint(layer) if remat else layer)(value, coeffs, weight, bias)
    assert state is not None
    value, coeffs = state
    derivatives = jnp.stack(coeffs, axis=2)
    if not batched:
        return DirectionalJet(value[0], derivatives[0])
    return DirectionalJet(value, derivatives)


def _resolve_jet_layout(layout: str, points: int) -> str:
    if layout == "auto":
        return "split" if points >= JET_SPLIT_MIN_POINTS and jax.default_backend() == "cpu" else "stack"
    if layout not in ("stack", "split"):
        raise ValueError(f"layout must be 'stack', 'split' or 'auto', not {layout!r}")
    return layout


def directional_jet_streams(
    x: Array,
    layers: Sequence[LayerSpec],
    directions: Array,
    order: int,
    *,
    allow_almost_everywhere: bool = False,
    remat: bool = False,
    layout: str | None = None,
) -> DirectionalJet:
    """Directional derivatives ``D^p_v u``, ``p = 1..order <= 4``, along the columns ``v`` of ``directions`` ``(D, K)``.

    ``layout="stack"`` carries one stack ``(B, 1 + order K, H)`` with the value and the ``order`` Taylor coefficients of every
    direction, so each layer is one ``dot_general`` and one elementwise Faa di Bruno update from a single activation tower.
    ``layout="split"`` keeps the value and each coefficient in its own array (``order + 1`` smaller products per layer, no
    concatenate or slice, and the first hidden layer reads its coefficients from ``W`` without the zero channels of the stack);
    the values agree to rounding. ``layout=None`` takes ``JET_LAYOUT`` (``"auto"``: the split layout on a CPU from
    ``JET_SPLIT_MIN_POINTS`` points, the stack otherwise). Any constant
    coefficient mixed partial of order at most four follows by polarization (:func:`partial_derivatives`). ``remat=True``
    wraps each layer in ``jax.checkpoint`` (a gradient stores the state at the layer boundaries and recomputes the inside of
    each layer; default off, the values are unchanged).
    """
    require_layers(layers)
    if not 1 <= order <= 4:
        raise ValueError("order must be in 1..4")
    a, batched = points_2d(x)
    B, dim = a.shape
    W1 = jnp.asarray(layers[0][0])
    if W1.ndim != 2 or W1.shape[1] != dim:
        raise ValueError(f"the first layer weight must have shape (H, {dim})")
    V = jnp.asarray(directions, a.dtype)
    if V.ndim != 2 or V.shape[0] != dim:
        raise ValueError(f"directions must have shape ({dim}, K)")
    K = V.shape[1]
    u1 = a @ W1.T
    if layers[0][1] is not None:
        u1 = u1 + jnp.asarray(layers[0][1])
    H = W1.shape[0]
    first = jnp.broadcast_to((W1 @ V).T[None], (B, K, H))
    if _resolve_jet_layout(JET_LAYOUT if layout is None else layout, B) == "split":
        return _split_jets(
            u1, first, layers, order, B, K, batched, allow_almost_everywhere=allow_almost_everywhere, remat=remat
        )
    zeros = jnp.zeros((B, (order - 1) * K, H), u1.dtype)
    X = jnp.concatenate([u1[:, None], first, zeros], axis=1)
    smooth = 4 if order >= 3 else 2
    for i, (W, b, spec) in enumerate(layers):
        weight = bias = None
        if i:
            weight = jnp.asarray(W)
            if weight.ndim != 2 or weight.shape[1] != X.shape[-1]:
                raise ValueError(f"layer {i} weight must have shape (out, {X.shape[-1]})")
            bias = None if b is None else jnp.asarray(b)
        skip = isinstance(spec, Skip)
        inner = spec.spec if isinstance(spec, Skip) else spec
        resolved = (
            None
            if inner is None
            else fastpath_activation(
                inner, allow_almost_everywhere=allow_almost_everywhere, smooth=smooth
            )
        )

        def layer(
            X: Array,
            weight: Array | None,
            bias: Array | None,
            *,
            i: int = i,
            skip: bool = skip,
            resolved: Any = resolved,
        ) -> Array:
            U = jnp.einsum("bsh,oh->bso", X, weight) if i else X
            if bias is not None:
                U = U.at[:, 0].add(bias)
            if skip:
                _check_skip(i, U, X)
            if resolved is None:
                return U
            tower = activation_tower(resolved, U[:, 0], order)
            u = [U[:, 1 + p * K : 1 + (p + 1) * K] for p in range(order)]
            new = _faa_di_bruno(tower, u)
            nxt = jnp.concatenate([tower[0][:, None], *new], axis=1)
            return nxt + X if skip else nxt

        X = (jax.checkpoint(layer) if remat else layer)(X, weight, bias)
    value = X[:, 0]
    C = X.shape[-1]
    derivatives = jnp.swapaxes(X[:, 1:].reshape(B, order, K, C), 1, 2)
    if not batched:
        return DirectionalJet(value[0], derivatives[0])
    return DirectionalJet(value, derivatives)


def partial_derivatives(
    x: Array,
    layers: Sequence[LayerSpec],
    multi_indices: Sequence[Sequence[int]],
    *,
    allow_almost_everywhere: bool = False,
) -> tuple[Array, Array]:
    """``(u, [d^alpha u for alpha in multi_indices])`` for mixed partials of total order 1..4.

    ``multi_indices`` are length-``D`` tuples of per-axis orders, e.g. ``(3, 0)`` for ``u_xxx`` or ``(2, 2)`` for
    ``u_xxyy``. The directions come from the polarization identity (:mod:`omnibias.core.polarization`); shared directions are
    carried once. Returns ``u`` of shape ``(B, C)`` and the derivatives ``(B, R, C)`` (unbatched ``x``: ``(C,)``, ``(R, C)``).
    """
    from omnibias.core.polarization import polarization_plan

    a, _ = points_2d(x)
    return _partials_by_plan(
        x, layers, polarization_plan(multi_indices, a.shape[1]), allow_almost_everywhere
    )


def _contract_plan(plan: Any, derivatives: Array) -> Array:
    """``sum_p w[r, j, p] D^p_j u`` for every request ``r`` from ``derivatives`` of shape ``(..., K, n, C)``.

    A small plan contracts with its dense table in one ``einsum``; a large one gathers its nonzero entries and adds them into
    their requests, which is the same sum over the entries that exist (a dense table grows as ``R K n``, the entries as ``R``).
    """
    if plan.dense_is_cheap:
        coef = jnp.asarray(plan.dense_floats(), derivatives.dtype)
        return jnp.einsum("rkp,...kpc->...rc", coef, derivatives)
    gathered = (
        derivatives[..., plan.cols, plan.slots, :]
        * jnp.asarray(plan.values, derivatives.dtype)[:, None]
    )
    out = jnp.zeros(
        (*derivatives.shape[:-3], plan.num_requests, derivatives.shape[-1]), derivatives.dtype
    )
    return out.at[..., plan.rows, :].add(gathered, indices_are_sorted=True)


def _partials_by_plan(
    x: Array, layers: Sequence[LayerSpec], plan: Any, allow_almost_everywhere: bool
) -> tuple[Array, Array]:
    a, _ = points_2d(x)
    V = jnp.asarray(plan.direction_matrix().astype(np.float64).T, a.dtype)
    jet = directional_jet_streams(
        x, layers, V, plan.order, allow_almost_everywhere=allow_almost_everywhere
    )
    return jet.value, _contract_plan(plan, jet.derivatives)


def laplacian_gradient(
    x: Array, layers: Sequence[LayerSpec], *, allow_almost_everywhere: bool = False
) -> tuple[Array, Array]:
    """``(u, grad_x (Delta u))``: the gradient of the Laplacian, ``(B, C)`` and ``(B, D, C)``.

    Component ``i`` is ``sum_j d_j^2 d_i u``; all ``D^2`` third derivatives come from the shared directional streams
    (``e_i`` and ``e_i +- e_j``), so the cost grows with ``D^2`` streams: this is the low-dimensional (PINN) form.
    """
    from omnibias.core.polarization import laplacian_gradient_plan

    a, batched = points_2d(x)
    dim = a.shape[1]
    u, d = _partials_by_plan(x, layers, laplacian_gradient_plan(dim), allow_almost_everywhere)
    d = d.reshape(*d.shape[:-2], dim, dim, d.shape[-1]).sum(axis=-2)
    return u, d


def bilaplacian_by_jets(
    x: Array,
    layers: Sequence[LayerSpec],
    *,
    coordinates: Sequence[int] | np.ndarray | None = None,
    allow_almost_everywhere: bool = False,
) -> tuple[Array, Array]:
    """``(u, Delta^2 u)`` as ``sum_i u_iiii + 2 sum_{i<j} u_iijj`` from shared directional streams (low-dimensional form).

    ``coordinates`` restricts the Laplacian to those inputs (a time coordinate, say, is left out).
    """
    from omnibias.core.polarization import bilaplacian_plan

    a, _ = points_2d(x)
    dim = a.shape[1]
    idx = list(range(dim)) if coordinates is None else [int(c) for c in np.asarray(coordinates)]
    weights = [1.0] * len(idx) + [2.0] * (len(idx) * (len(idx) - 1) // 2)
    u, d = _partials_by_plan(x, layers, bilaplacian_plan(dim, idx), allow_almost_everywhere)
    w = jnp.asarray(weights, d.dtype)
    return u, jnp.einsum("r,...rc->...c", w, d)
