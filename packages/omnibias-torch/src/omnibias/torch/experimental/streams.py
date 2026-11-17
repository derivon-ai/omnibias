# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Stacked second-order stream engine for deep MLPs (PyTorch).

The twin of :mod:`omnibias.jax.experimental.streams`: value, metric tangents, optional Gram companions, drift tangents and
the second-order stream live in one tensor of shape ``(B, S, H)``; each hidden layer is one matmul over the flattened
``(B*S, H)`` rows followed by one elementwise update that evaluates the activation tower once. No factorization, no
conditional and no host synchronization (the CUDA stream is never drained, so the program can be captured in a CUDA graph);
everything is plain autograd, so parameter derivatives of every order are exact. See the JAX module for the channel layout
and for the ``diag`` and ``gram`` ways of carrying a metric through the first layer.
"""

from __future__ import annotations

import weakref
from collections.abc import Sequence
from typing import Any, NamedTuple, cast

import numpy as np
from omnibias.torch.experimental._chain import (
    Skip,
    columns,
    distinct_indices,
    fastpath_activation,
    points_2d,
    require_layers,
)
from omnibias.torch.fastpath.eulerian import (
    sigmoid_tower,
    sigmoid_tower_rows,
    softplus_tower,
    softplus_tower_rows,
)
from omnibias.torch.fastpath.legendre import tanh_tower, tanh_tower_rows

import torch
from torch import Tensor

__all__ = [
    "DirectionalJet",
    "STREAM_LAYOUT",
    "StreamLayout",
    "StreamResult",
    "activation_tower",
    "bilaplacian_by_jets",
    "directional_jet_streams",
    "laplacian_gradient",
    "metric_block",
    "partial_derivatives",
    "second_order_streams",
]


class StreamLayout(NamedTuple):
    k: int
    kp: int
    q: int
    gram: bool

    @property
    def size(self) -> int:
        return 2 + self.k + self.kp + self.q


class StreamResult(NamedTuple):
    value: Tensor
    grad_sq: Tensor
    laplacian: Tensor
    directional: Tensor | None
    jacobian: Tensor | None


def _check_skip(i: int, U: Tensor, X: Tensor) -> None:
    if i == 0:
        raise ValueError("the first layer cannot be a Skip block (the width changes)")
    if U.shape != X.shape:
        raise ValueError(
            f"layer {i} is a Skip block and needs a square weight, got width {U.shape[-1]} for {X.shape[-1]}"
        )


# The towers that also have a list form: the stream engine indexes the rows, so a stacked copy of them is not needed.
_TOWER_ROWS = {
    tanh_tower: tanh_tower_rows,
    sigmoid_tower: sigmoid_tower_rows,
    softplus_tower: softplus_tower_rows,
}


def has_row_tower(spec: Any) -> bool:
    """Whether :func:`activation_tower` evaluates ``spec`` as one base evaluation plus Horner polynomials, the same values
    that ``spec.forward`` and ``spec.fastpath`` return one at a time."""
    return getattr(spec, "tower", None) in _TOWER_ROWS


def activation_tower(spec: Any, u: Tensor, order: int) -> list[Tensor]:
    """``[sigma(u), ..., sigma^(order)(u)]`` from one base evaluation when the activation has a tower."""
    tower = getattr(spec, "tower", None)
    if tower is not None:
        rows = _TOWER_ROWS.get(tower)
        if rows is not None:
            return rows(u, order)
        stacked = tower(u, order)
        return [stacked[n] for n in range(order + 1)]
    return [spec.forward(u)] + [spec.fastpath(u, n) for n in range(1, order + 1)]


def metric_block(
    W1: Tensor, diffusion: Tensor | None, coordinates: Sequence[int] | np.ndarray | None
) -> Tensor:
    """``W1 S`` with ``S`` restricted to ``coordinates`` (no selection matrix is formed)."""
    dim = W1.shape[1]
    M = W1
    if coordinates is not None:
        M = columns(W1, distinct_indices(coordinates, dim))
    if diffusion is None:
        return M
    S = torch.as_tensor(diffusion, dtype=W1.dtype, device=W1.device)
    m = M.shape[1]
    if S.ndim == 1 and S.shape[0] == m:
        return M * S
    if S.ndim == 2 and S.shape[0] == m and S.shape[1] >= 1:
        return M @ S
    raise ValueError(f"diffusion must have shape ({m},) or ({m}, r) with r >= 1")


def _q2(U: Tensor, lay: StreamLayout, weights: Tensor | None) -> Tensor:
    T = U[:, 1 : 1 + lay.k]
    if lay.gram:
        return torch.sum(T * U[:, 1 + lay.k : 1 + lay.k + lay.kp], dim=1)
    if weights is None:
        return torch.sum(T * T, dim=1)
    return torch.einsum("c,bch->bh", weights, T * T)


def _update(
    U: Tensor,
    bias: Tensor | None,
    spec: Any,
    lay: StreamLayout,
    weights: Tensor | None,
    q2: Tensor | None,
    first: bool = False,
) -> Tensor:
    """The stack after the activation."""
    u = U[:, 0] if bias is None else U[:, 0] + bias
    s0, s1, s2 = activation_tower(spec, u, 2)
    if q2 is None:
        q2 = _q2(U, lay, weights)
    # ``first``: the stack was just built from the input, so its Laplacian channel is zero and ``s1 * U[:, -1]`` adds zeros
    lap = s2 * q2 if first else s2 * q2 + s1 * U[:, -1]
    mid = s1.unsqueeze(1) * U[:, 1:-1]
    return torch.cat([s0.unsqueeze(1), mid, lap.unsqueeze(1)], dim=1)


def _gram_stack(u1: Tensor, G: Tensor, Q: Tensor | None, B: int, H: int) -> Tensor:
    eye = torch.eye(H, dtype=u1.dtype, device=u1.device).unsqueeze(0).expand(B, H, H)
    P = G.unsqueeze(0).expand(B, H, H) if G.ndim == 2 else G
    parts = (
        [u1.unsqueeze(1), eye, P] + ([] if Q is None else [Q]) + [torch.zeros_like(u1).unsqueeze(1)]
    )
    return torch.cat(parts, dim=1)


_HESS_SPLITS: dict[tuple[int, torch.dtype], tuple[Any, int, int, Any, dict[Any, tuple[Tensor, Tensor]]]] = {}


def _hess_split(hess_form: Any, A: Tensor) -> tuple[int, Any]:
    """Rank ``r`` of the symmetric ``A`` and a function giving its signed split ``S`` and ``signs`` (``S diag(signs) S^T = A``).

    The eigendecomposition reads ``A`` back to the host and factors it, so it is kept while the same ``hess_form`` tensor is
    unchanged (identity, version counter and the dtype it was decomposed in). Only a tensor that does not require grad is
    kept; the split is a constant.
    """
    cacheable = isinstance(hess_form, Tensor) and not hess_form.requires_grad
    key = (id(hess_form), A.dtype)
    entry = _HESS_SPLITS.get(key) if cacheable else None
    if entry is None or entry[0]() is not hess_form or entry[1] != hess_form._version:
        lam, V = np.linalg.eigh(A.detach().cpu().numpy())
        keep = np.abs(lam) > 1e-13 * max(float(np.max(np.abs(lam))), 1e-300)
        entry = (
            weakref.ref(hess_form) if cacheable else (lambda: None),
            hess_form._version if cacheable else -1,
            int(keep.sum()),
            (V[:, keep] * np.sqrt(np.abs(lam[keep])), np.sign(lam[keep])),
            {},
        )
        if cacheable:
            if len(_HESS_SPLITS) >= 8:
                _HESS_SPLITS.pop(next(iter(_HESS_SPLITS)))
            _HESS_SPLITS[key] = entry
    _, _, r, (S_np, signs_np), tensors = entry

    def split(dtype: torch.dtype, device: torch.device) -> tuple[Tensor, Tensor]:
        if (dtype, device) not in tensors:
            tensors[(dtype, device)] = (
                torch.as_tensor(S_np, dtype=dtype, device=device),
                torch.as_tensor(signs_np, dtype=dtype, device=device),
            )
        return tensors[(dtype, device)]

    return r, split


_PACKED_INDEX: dict[tuple[int, torch.device], tuple[Tensor, Tensor, Tensor]] = {}


def _packed_symmetric_index(H: int, device: torch.device) -> tuple[Tensor, Tensor, Tensor]:
    """Positions of a symmetric ``(H, H)`` matrix in its packed upper triangle, and that triangle's row and column indices,
    as tensors on ``device`` (built once per size)."""
    key = (H, device)
    held = _PACKED_INDEX.get(key)
    if held is None:
        if len(_PACKED_INDEX) >= 16:
            _PACKED_INDEX.pop(next(iter(_PACKED_INDEX)))
        iu, ju = np.triu_indices(H)
        index = np.zeros((H, H), dtype=np.int64)
        index[iu, ju] = index[ju, iu] = np.arange(iu.size)
        held = _PACKED_INDEX[key] = (
            torch.as_tensor(index, device=device),
            torch.as_tensor(iu, device=device),
            torch.as_tensor(ju, device=device),
        )
    return held


def _pack(u1: Tensor, T: Tensor, tail: list[Tensor], split: bool) -> Tensor | tuple[Tensor, Tensor]:
    """The first stack, or (split layout, no drift directions) the value and the tangents as two tensors."""
    if split and len(tail) == 1:
        return u1, T
    return torch.cat([u1.unsqueeze(1), T, *tail], dim=1)


def _first_stack(
    x: Tensor,
    W1: Tensor,
    b1: Tensor | None,
    *,
    diffusion: Tensor | None,
    point_diffusion: Tensor | None,
    coordinates: Sequence[int] | np.ndarray | None,
    directions: Tensor | None,
    batched: bool,
    form: str,
    full_jacobian: bool,
    weights: Tensor | None,
    hess_form: Tensor | None = None,
    split: bool = False,
) -> tuple[Tensor | tuple[Tensor, Tensor], StreamLayout, Tensor, Tensor | None]:
    B, dim = x.shape
    H = W1.shape[0]
    u1 = x @ W1.T
    if b1 is not None:
        u1 = u1 + torch.as_tensor(b1)
    dtype = u1.dtype
    zeros = torch.zeros_like(u1).unsqueeze(1)
    q = 0
    Q = None
    if directions is not None:
        v = torch.as_tensor(directions, dtype=dtype, device=u1.device)
        if v.ndim == 2 and v.shape[0] == dim:
            Q = (W1 @ v).T.unsqueeze(0).expand(B, v.shape[1], H)
        elif batched and v.ndim == 3 and tuple(v.shape[:2]) == (B, dim):
            Q = torch.einsum("hd,bdm->bmh", W1, v)
        else:
            raise ValueError("directions must have shape (D, q) or (B, D, q)")
        q = Q.shape[1]
    tail = ([] if Q is None else [Q]) + [zeros]
    if full_jacobian:
        if diffusion is not None or point_diffusion is not None or coordinates is not None:
            raise ValueError("the full input Jacobian excludes a metric or coordinate selection")
        T = W1.T.unsqueeze(0).expand(B, dim, H)
        q2 = torch.sum(W1 * W1, dim=1).unsqueeze(0).expand(B, H)
        return (
            _pack(u1, T, tail, split),
            StreamLayout(dim, 0, q, False),
            q2,
            None,
        )
    if point_diffusion is not None:
        if diffusion is not None:
            raise ValueError("pass either diffusion or point_diffusion, not both")
        M0 = metric_block(W1, None, coordinates)
        m = M0.shape[1]
        s = torch.as_tensor(point_diffusion, dtype=dtype, device=u1.device)
        s = s if batched else s[None]
        if tuple(s.shape) != (B, m):
            shape = f"({B}, {m})" if batched else f"({m},)"
            raise ValueError(f"point_diffusion must have shape {shape}")
        if form == "diag" or (form == "auto" and m <= 2 * H):
            T = M0.T.unsqueeze(0) * s.unsqueeze(2)
            q2 = (s * s) @ (M0 * M0).T
            return (
                _pack(u1, T, tail, split),
                StreamLayout(m, 0, q, False),
                q2,
                None,
            )
        s2 = s * s
        if 2 * B >= H:
            idx, iu, ju = _packed_symmetric_index(H, u1.device)
            G = (s2 @ (M0[iu] * M0[ju]).T)[:, idx]
        else:
            G = torch.einsum("bhm,gm->bhg", M0.unsqueeze(0) * s2.unsqueeze(1), M0)
        q2 = torch.diagonal(G, dim1=-2, dim2=-1)
        return _gram_stack(u1, G, Q, B, H), StreamLayout(H, H, q, True), q2, None
    if hess_form is not None:
        if diffusion is not None or weights is not None:
            raise ValueError("hess_form excludes diffusion and weights")
        M0 = metric_block(W1, None, coordinates)
        m = M0.shape[1]
        A = torch.as_tensor(hess_form, dtype=dtype, device=u1.device)
        if tuple(A.shape) != (m, m):
            raise ValueError(f"hess_form must have shape ({m}, {m})")
        A = 0.5 * (A + A.T)
        if form != "gram" and not A.requires_grad:
            r, split = _hess_split(hess_form, A)
            if form == "diag" or r <= 2 * H:
                S, signs = split(dtype, u1.device)
                M = M0 @ S
                T = M.T.unsqueeze(0).expand(B, r, H)
                q2 = (M * M) @ signs
                return (
                    _pack(u1, T, tail, split),
                    StreamLayout(r, 0, q, False),
                    q2.unsqueeze(0).expand(B, H),
                    signs,
                )
        G = M0 @ A @ M0.T
        q2 = torch.diagonal(G).unsqueeze(0).expand(B, H)
        return _gram_stack(u1, G, Q, B, H), StreamLayout(H, H, q, True), q2, None
    M = metric_block(W1, diffusion, coordinates)
    r = M.shape[1]
    if form == "diag" or (form == "auto" and r <= 2 * H):
        w = None if weights is None else torch.as_tensor(weights, dtype=dtype, device=u1.device)
        if w is not None and tuple(w.shape) != (r,):
            raise ValueError(f"weights must have shape ({r},)")
        q2 = torch.sum(M * M, dim=1) if w is None else (M * M) @ w
        T = M.T.unsqueeze(0).expand(B, r, H)
        return (
            _pack(u1, T, tail, split),
            StreamLayout(r, 0, q, False),
            q2.unsqueeze(0).expand(B, H),
            w,
        )
    if weights is not None:
        raise ValueError(
            "channel weights apply to the diag form only; pass the signed metric as `diffusion`/Gram"
        )
    G = M @ M.T
    q2 = torch.diagonal(G).unsqueeze(0).expand(B, H)
    return _gram_stack(u1, G, Q, B, H), StreamLayout(H, H, q, True), q2, None


STREAM_LAYOUT = "auto"
"""Layout of :func:`second_order_streams` when ``layout`` is not given: ``"stack"``, ``"split"`` or ``"auto"`` (the split layout,
except while a CUDA graph is being captured or ``torch.compile`` traces a call that no gradient flows through, which keep the stack;
a captured call whose weights or points require grad takes the split layout).

Called eagerly, one forward Laplacian is dominated by the host's per-operation dispatch, and the split layout issues fewer
operations: on an A100 (21 cells, two precisions) it took 0.5 to 0.9 of the stack's time in the host-timed call in 13 to 17
cells and was slower in none, and 0.4 to 0.8 in the value-and-gradient call. Replayed from a CUDA graph the host is out of the
picture and the kernel count decides: the split layout launches more kernels and took 1.04 to 1.19 of the stack's time in the
forward replay of the small-batch cells, so a captured forward-only call keeps the stack. The backward pass of the stack's
concatenations and slices costs more kernel time than it saves, and a captured call that gradients flow through (grad mode on, a
weight, a bias or the points require grad) takes the split layout: its value-and-gradient replay took 0.51 to 0.94 of the stack's
time in the 38 of 48 cells (three precisions, 16 cells, three paired rounds) where the interval decided, and was slower in none. On one CPU thread the split layout
was faster or level in every one of 30 cells."""


def _capturing(x: Tensor) -> bool:
    return bool(x.is_cuda and torch.cuda.is_current_stream_capturing())


def _differentiated(x: Tensor, layers: Sequence[Any]) -> bool:
    """Whether a gradient flows through the call: grad mode is on and the points or one weight or bias requires grad."""
    if not torch.is_grad_enabled():
        return False
    return x.requires_grad or any(isinstance(t, Tensor) and t.requires_grad for layer in layers for t in layer[:2])


def _resolve_layout(layout: str, x: Tensor, differentiated: bool = False) -> str:
    if layout == "auto":
        if torch.compiler.is_compiling():
            return "stack"
        return "stack" if _capturing(x) and not differentiated else "split"
    if layout not in ("stack", "split"):
        raise ValueError(f"layout must be 'stack', 'split' or 'auto', not {layout!r}")
    return layout


def _q2_split(T: Tensor, weights: Tensor | None) -> Tensor:
    if weights is None:
        return torch.sum(T * T, dim=1)
    return torch.einsum("c,bch->bh", weights, T * T)


def _split_loop(
    first: tuple[Tensor, Tensor],
    q2_seed: Tensor,
    weights: Tensor | None,
    layers: Sequence[Any],
    allow_almost_everywhere: bool,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """The layer loop of :func:`second_order_streams` with the value, the tangents ``T`` ``(B, k, H)`` and the second-order
    stream ``lap`` in separate tensors: the same arithmetic as the stack, without its per-layer concatenation."""
    value, T = first
    lap = value  # the second-order stream of the first layer is zero and is never read there
    for i, (W, b, spec) in enumerate(layers):
        prev = (value, T, lap)
        bias = None
        q2: Tensor | None = q2_seed if i == 0 else None
        if i:
            W = torch.as_tensor(W)
            if W.ndim != 2 or W.shape[1] != value.shape[-1]:
                raise ValueError(f"layer {i} weight must have shape (out, {value.shape[-1]})")
            Wt = W.T
            value, T, lap = value @ Wt, T @ Wt, lap @ Wt
            bias = None if b is None else torch.as_tensor(b)
        skip = isinstance(spec, Skip)
        if isinstance(spec, Skip):
            if i == 0:
                raise ValueError("the first layer cannot be a Skip block (the width changes)")
            if value.shape != prev[0].shape:
                raise ValueError(
                    f"layer {i} is a Skip block and needs a square weight, got width {value.shape[-1]} for {prev[0].shape[-1]}"
                )
            spec = spec.spec
        if spec is None:
            if i == 0:
                # An affine first layer has a zero second-order stream; the placeholder above is only valid when an
                # activation follows it (its update reads no incoming stream).
                lap = torch.zeros_like(value)
                if value.requires_grad:
                    # The stack's concatenation keeps the first layer's weight and bias in the graph of every stream.
                    tie = value.sum() * 0.0
                    lap, T = lap + tie, T + tie
            if bias is not None:
                value = value + bias
                if bias.requires_grad:
                    # The stack's concatenation keeps an affine layer's bias in the graph of every stream, so differentiating
                    # the Laplacian reaches it with a zero gradient; separate tensors would leave it out and
                    # ``torch.autograd.grad`` would raise for it. One scalar ties it back.
                    tie = bias.sum() * 0.0
                    lap, T = lap + tie, T + tie
            continue
        resolved = fastpath_activation(spec, allow_almost_everywhere=allow_almost_everywhere)
        u = value if bias is None else value + bias
        s0, s1, s2 = activation_tower(resolved, u, 2)
        if q2 is None:
            q2 = _q2_split(T, weights)
        new_lap = s2 * q2 if i == 0 else s2 * q2 + s1 * lap
        new_T = s1.unsqueeze(1) * T
        if skip:
            s0, new_T, new_lap = s0 + prev[0], new_T + prev[1], new_lap + prev[2]
        value, T, lap = s0, new_T, new_lap
    return value, _q2_split(T, weights), lap, T


def second_order_streams(
    x: Tensor,
    layers: Sequence[Any],
    *,
    diffusion: Tensor | None = None,
    point_diffusion: Tensor | None = None,
    coordinates: Sequence[int] | np.ndarray | None = None,
    directions: Tensor | None = None,
    weights: Tensor | None = None,
    form: str = "auto",
    full_jacobian: bool = False,
    allow_almost_everywhere: bool = False,
    hess_form: Tensor | None = None,
    layout: str | None = None,
) -> StreamResult:
    """Value, squared gradient norm, Laplacian and drift derivatives of a deep MLP (see the JAX function of the same name).

    ``layout`` is ``"stack"`` (one ``(B, S, H)`` tensor, one matmul per layer, one concatenation per layer), ``"split"`` (the value,
    the tangents and the second-order stream in their own tensors: three matmuls and no concatenation per layer; the diag form
    without drift directions, other requests use the stack) or ``"auto"`` (split, except under CUDA graph capture or ``torch.compile`` of a call that no gradient flows through); ``None`` takes
    :data:`STREAM_LAYOUT`.

    ``hess_form`` is a symmetric ``(m, m)`` matrix ``A`` (indefinite allowed): the Laplacian is ``tr(A hess u)`` and
    ``grad_sq`` is ``grad u^T A grad u``. A constant ``A`` of rank ``r <= 2 H`` is split into signed tangent channels; an
    ``A`` that requires grad (or ``form="gram"``) goes through the Gram form and stays differentiable in ``A``.
    """
    require_layers(layers)
    if form not in ("auto", "diag", "gram"):
        raise ValueError(f"form must be 'auto', 'diag' or 'gram', not {form!r}")
    a, batched = points_2d(x)
    B, dim = a.shape
    W1 = torch.as_tensor(layers[0][0])
    if W1.ndim != 2 or W1.shape[1] != dim:
        raise ValueError(f"the first layer weight must have shape (H, {dim})")
    split = (
        directions is None
        and _resolve_layout(STREAM_LAYOUT if layout is None else layout, a, _differentiated(a, layers)) == "split"
    )
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
        split=split,
    )
    if isinstance(U, tuple):
        value, grad_sq, lap, jacobian = _split_loop(U, q2_seed, w, layers, allow_almost_everywhere)
        if not batched:
            value, grad_sq, lap, jacobian = value[0], grad_sq[0], lap[0], jacobian[0]
        return StreamResult(value, grad_sq, lap, None, jacobian)
    q2: Tensor | None = q2_seed
    X = U
    for i, (W, b, spec) in enumerate(layers):
        bias = None
        if i:
            W = torch.as_tensor(W)
            if W.ndim != 2 or W.shape[1] != X.shape[-1]:
                raise ValueError(f"layer {i} weight must have shape (out, {X.shape[-1]})")
            U = X @ W.T
            bias = None if b is None else torch.as_tensor(b)
            q2 = None
        skip = isinstance(spec, Skip)
        if isinstance(spec, Skip):
            _check_skip(i, U, X)
            spec = spec.spec
        if spec is None:
            if bias is not None:
                head = U[:, :1] + bias.reshape(1, 1, -1)
                X = torch.cat([head, U[:, 1:]], dim=1)
            else:
                X = U
            continue
        resolved = fastpath_activation(spec, allow_almost_everywhere=allow_almost_everywhere)
        new = _update(U, bias, resolved, lay, w, q2, first=i == 0)
        X = new + X if skip else new
    value = X[:, 0]
    lap = X[:, -1]
    grad_sq = _q2(X, lay, w)
    q0 = 1 + lay.k + lay.kp
    directional = None if lay.q == 0 else X[:, q0 : q0 + lay.q]
    jacobian = X[:, 1 : 1 + lay.k] if (full_jacobian or not lay.gram) else None
    if not batched:
        value, grad_sq, lap = value[0], grad_sq[0], lap[0]
        directional = None if directional is None else directional[0]
        jacobian = None if jacobian is None else jacobian[0]
    return StreamResult(value, grad_sq, lap, directional, jacobian)


# --------------------------------------------------------------------------------------------- directional Taylor streams
class DirectionalJet(NamedTuple):
    value: Tensor  # (B, C)
    derivatives: Tensor  # (B, K, n, C): the p-th directional derivative D^p_v u along direction v_j, p = 1..n


def _faa_di_bruno(s: list[Tensor], u: list[Tensor]) -> list[Tensor]:
    """Physical derivatives ``(sigma o u)^(p)``, ``p = 1..n``, from the tower ``s[0..n]`` and ``u[1..n]`` (index ``p-1``)."""
    n = len(u)
    s1 = s[1].unsqueeze(1)
    out = [s1 * u[0]]
    if n >= 2:
        s2 = s[2].unsqueeze(1)
        out.append(s1 * u[1] + s2 * u[0] ** 2)
    if n >= 3:
        s3 = s[3].unsqueeze(1)
        out.append(s1 * u[2] + 3.0 * s2 * u[0] * u[1] + s3 * u[0] ** 3)
    if n >= 4:
        s4 = s[4].unsqueeze(1)
        out.append(
            s1 * u[3]
            + s2 * (3.0 * u[1] ** 2 + 4.0 * u[0] * u[2])
            + 6.0 * s3 * u[0] ** 2 * u[1]
            + s4 * u[0] ** 4
        )
    return out


def _faa_di_bruno_first(s: list[Tensor], u0: Tensor, n: int) -> list[Tensor]:
    """:func:`_faa_di_bruno` when only the first-order tangent is nonzero (the first layer): ``s_p * u0^p``."""
    out = [s[1].unsqueeze(1) * u0]
    for p in range(2, n + 1):
        out.append(s[p].unsqueeze(1) * u0**p)
    return out


def directional_jet_streams(
    x: Tensor,
    layers: Sequence[Any],
    directions: Tensor,
    order: int,
    *,
    allow_almost_everywhere: bool = False,
) -> DirectionalJet:
    """Directional derivatives ``D^p_v u``, ``p = 1..order <= 4``, along the columns ``v`` of ``directions`` ``(D, K)``."""
    require_layers(layers)
    if not 1 <= order <= 4:
        raise ValueError("order must be in 1..4")
    a, batched = points_2d(x)
    B, dim = a.shape
    W1 = torch.as_tensor(layers[0][0])
    if W1.ndim != 2 or W1.shape[1] != dim:
        raise ValueError(f"the first layer weight must have shape (H, {dim})")
    V = torch.as_tensor(directions, dtype=a.dtype, device=a.device)
    if V.ndim != 2 or V.shape[0] != dim:
        raise ValueError(f"directions must have shape ({dim}, K)")
    K = V.shape[1]
    H = W1.shape[0]
    u1 = a @ W1.T
    if layers[0][1] is not None:
        u1 = u1 + torch.as_tensor(layers[0][1])
    first = (W1 @ V).T.unsqueeze(0).expand(B, K, H)
    zeros = torch.zeros(B, (order - 1) * K, H, dtype=u1.dtype, device=u1.device)
    X = torch.cat([u1.unsqueeze(1), first, zeros], dim=1)
    smooth = 4 if order >= 3 else 2
    for i, (W, b, spec) in enumerate(layers):
        bias = None
        if i:
            W = torch.as_tensor(W)
            if W.ndim != 2 or W.shape[1] != X.shape[-1]:
                raise ValueError(f"layer {i} weight must have shape (out, {X.shape[-1]})")
            U = X @ W.T
            bias = None if b is None else torch.as_tensor(b)
        else:
            U = X
        skip = isinstance(spec, Skip)
        if isinstance(spec, Skip):
            _check_skip(i, U, X)
            spec = spec.spec
        if spec is None:
            X = U if bias is None else torch.cat([U[:, :1] + bias.reshape(1, 1, -1), U[:, 1:]], dim=1)
            continue
        resolved = fastpath_activation(
            spec, allow_almost_everywhere=allow_almost_everywhere, smooth=smooth
        )
        tower = activation_tower(resolved, U[:, 0] if bias is None else U[:, 0] + bias, order)
        if i == 0:  # the higher-order channels of the freshly built stack are zero
            new = _faa_di_bruno_first(tower, U[:, 1 : 1 + K], order)
        else:
            u = [U[:, 1 + p * K : 1 + (p + 1) * K] for p in range(order)]
            new = _faa_di_bruno(tower, u)
        nxt = torch.cat([tower[0].unsqueeze(1), *new], dim=1)
        X = nxt + X if skip else nxt
    value = X[:, 0]
    C = X.shape[-1]
    derivatives = X[:, 1:].reshape(B, order, K, C).transpose(1, 2)
    if not batched:
        return DirectionalJet(value[0], derivatives[0])
    return DirectionalJet(value, derivatives)


def partial_derivatives(
    x: Tensor,
    layers: Sequence[Any],
    multi_indices: Sequence[Sequence[int]],
    *,
    allow_almost_everywhere: bool = False,
) -> tuple[Tensor, Tensor]:
    """``(u, [d^alpha u for alpha in multi_indices])`` for mixed partials of total order 1..4 (polarization)."""
    from omnibias.core.polarization import polarization_plan

    a, _ = points_2d(x)
    return _partials_by_plan(x, layers, polarization_plan(multi_indices, a.shape[1]), allow_almost_everywhere)


def plan_constant(plan: Any, name: str, dtype: torch.dtype | None, device: torch.device, build: Any) -> Tensor:
    """``build()`` (a numpy array) as a tensor of ``dtype`` on ``device``, kept on the plan so a call does not upload it again.

    A plan is memoized and never changes, so its arrays are constants. Without this every call converted them and, on a GPU,
    copied them from pageable host memory (a copy that waits for the stream and cannot be captured in a CUDA graph).
    """
    key = ("torch", name, dtype, device)
    value = plan.memo.get(key)
    if value is None:
        value = plan.memo[key] = torch.as_tensor(build(), dtype=dtype, device=device)
    return cast(Tensor, value)


def _contract_plan(plan: Any, derivatives: Tensor) -> Tensor:
    """``sum_p w[r, j, p] D^p_j u`` for every request ``r`` from ``derivatives`` of shape ``(..., K, n, C)``.

    A small plan contracts with its dense table in one ``einsum``; a large one gathers its nonzero entries and adds them into
    their requests, which is the same sum over the entries that exist (a dense table grows as ``R K n``, the entries as ``R``).
    """
    dev, dt = derivatives.device, derivatives.dtype
    if plan.dense_is_cheap:
        dense = plan_constant(plan, "dense", dt, dev, plan.dense_floats)
        return torch.einsum("rkp,...kpc->...rc", dense, derivatives)
    cols = plan_constant(plan, "cols", None, dev, plan.cols.copy)
    slots = plan_constant(plan, "slots", None, dev, plan.slots.copy)
    rows = plan_constant(plan, "rows", None, dev, plan.rows.copy)
    w = plan_constant(plan, "values", dt, dev, plan.values.copy)
    gathered = derivatives[..., cols, slots, :] * w[:, None]
    out = derivatives.new_zeros((*derivatives.shape[:-3], plan.num_requests, derivatives.shape[-1]))
    return out.index_add(-2, rows, gathered)


def plan_directions(plan: Any, like: Tensor) -> Tensor:
    """The plan's directions as ``(dim, K)`` columns in ``like``'s dtype and device (kept on the plan)."""
    return plan_constant(plan, "directions", like.dtype, like.device, lambda: plan.direction_matrix().astype(np.float64).T)


def _partials_by_plan(
    x: Tensor, layers: Sequence[Any], plan: Any, allow_almost_everywhere: bool
) -> tuple[Tensor, Tensor]:
    a, _ = points_2d(x)
    V = plan_directions(plan, a)
    jet = directional_jet_streams(
        x, layers, V, plan.order, allow_almost_everywhere=allow_almost_everywhere
    )
    return jet.value, _contract_plan(plan, jet.derivatives)


def laplacian_gradient(
    x: Tensor, layers: Sequence[Any], *, allow_almost_everywhere: bool = False
) -> tuple[Tensor, Tensor]:
    """``(u, grad_x (Delta u))``: the gradient of the Laplacian, ``(B, C)`` and ``(B, D, C)`` (low-dimensional form)."""
    from omnibias.core.polarization import laplacian_gradient_plan

    a, batched = points_2d(x)
    dim = a.shape[1]
    u, d = _partials_by_plan(x, layers, laplacian_gradient_plan(dim), allow_almost_everywhere)
    d = d.reshape(*d.shape[:-2], dim, dim, d.shape[-1]).sum(dim=-2)
    return u, d


def bilaplacian_by_jets(
    x: Tensor,
    layers: Sequence[Any],
    *,
    coordinates: Sequence[int] | None = None,
    allow_almost_everywhere: bool = False,
) -> tuple[Tensor, Tensor]:
    """``(u, Delta^2 u)`` as ``sum_i u_iiii + 2 sum_{i<j} u_iijj`` from shared directional streams (low-dimensional form).

    ``coordinates`` restricts the Laplacian to those inputs (a time coordinate, say, is left out).
    """
    from omnibias.core.polarization import bilaplacian_plan

    a, _ = points_2d(x)
    dim = a.shape[1]
    idx = list(range(dim)) if coordinates is None else [int(c) for c in coordinates]
    plan = bilaplacian_plan(dim, idx)
    u, d = _partials_by_plan(x, layers, plan, allow_almost_everywhere)
    w = plan_constant(
        plan,
        "bilaplacian_weights",
        d.dtype,
        d.device,
        lambda: np.array([1.0] * len(idx) + [2.0] * (len(idx) * (len(idx) - 1) // 2)),
    )
    return u, torch.einsum("r,...rc->...c", w, d)
