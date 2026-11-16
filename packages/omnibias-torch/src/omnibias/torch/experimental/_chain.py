# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Pieces shared by the structured deep-MLP programs of the PyTorch backend.

The same layer convention as :mod:`omnibias.jax.experimental`: ``layers = [(W, b, spec), ...]`` with ``W`` of shape
``(out, in)``, ``spec=None`` for an affine layer, and points of shape ``(D,)`` or ``(B, D)``. The helpers are the
PyTorch counterparts of the JAX ``_chain`` module and compute the same quantities.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from omnibias.core.chain import ChainDescriptor, LayerDescriptor
from omnibias.torch.activations.registry import get_activation
from omnibias.torch.laplacian import _check_piecewise_second_order

import torch
from torch import Tensor

__all__ = [
    "NOT_C3",
    "Skip",
    "affine_readout",
    "carry_columns",
    "columns",
    "describe_layers",
    "distinct_indices",
    "fastpath_activation",
    "fourier_feature_layer",
    "hidden_slopes",
    "pairing_index",
    "points_2d",
    "readout_value",
    "require_layers",
]

_DTYPE_NAMES = {
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.float32: "float32",
    torch.float64: "float64",
}

#: Activations whose derivatives up to order three jump; the bi-Laplacian would drop a singular part.
NOT_C3 = frozenset(
    {
        "abs",
        "celu",
        "elu",
        "hardshrink",
        "hardsigmoid",
        "hardswish",
        "hardtanh",
        "huber",
        "leaky_relu",
        "prelu",
        "relu",
        "relu6",
        "selu",
        "sign",
        "softshrink",
        "softsign",
        "step",
        "threshold",
    }
)


class Skip:
    """A hidden layer ``h -> h + sigma(W h + b)`` (a residual block): wrap its activation, ``(W, b, Skip("tanh"))``.

    ``W`` must be square and the layer cannot be the first one. The stream engines
    (:func:`~omnibias.torch.experimental.streams.second_order_streams` and the directional jets) carry the identity branch
    through every stream, so a block costs one extra addition and the derivatives stay exact.
    """

    __slots__ = ("spec",)

    def __init__(self, spec: Any) -> None:
        if spec is None or isinstance(spec, Skip):
            raise ValueError("Skip wraps an activation")
        self.spec = spec

    def __repr__(self) -> str:
        return f"Skip({self.spec!r})"


def fourier_feature_layer(frequencies: Any, phase: Any = None) -> tuple[Tensor, Tensor, str]:
    """The Fourier-feature embedding ``[sin(B x + c), cos(B x + c)]`` as one exact ``(W, b, "sin")`` layer.

    ``frequencies`` is ``B`` of shape ``(m, D)`` (fold any ``2 pi`` into it) and ``phase`` an optional ``(m,)`` offset
    ``c``. Since ``cos z = sin(z + pi / 2)`` the embedding is a single layer of width ``2 m`` with a sine activation, so
    every stream program carries it with no special case, and a trainable ``B`` receives exact derivatives.
    """
    B = torch.as_tensor(frequencies)
    c = (
        torch.zeros(B.shape[0], dtype=B.dtype, device=B.device)
        if phase is None
        else torch.as_tensor(phase, dtype=B.dtype, device=B.device)
    )
    return (
        torch.cat([B, B], dim=0),
        torch.cat([c, c + (torch.pi / 2)], dim=0),
        "sin",
    )


def require_layers(layers: Sequence[Any]) -> None:
    if not layers:
        raise ValueError("layers must be non-empty")


def points_2d(x: Tensor) -> tuple[Tensor, bool]:
    """``(a, batched)``: ``a`` has shape ``(B, D)`` (a single point gains a leading axis)."""
    x = torch.as_tensor(x)
    batched = x.ndim > 1
    a = x if batched else x[None, :]
    if a.ndim != 2:
        raise ValueError("x must have shape (D,) or (B, D)")
    return a, batched


_INDEX_TENSORS: dict[tuple[bytes, str, torch.device], Tensor] = {}


def index_tensor(index: np.ndarray, device: torch.device) -> Tensor:
    """``index`` as an integer tensor on ``device``, uploaded once per distinct index (a small bounded cache).

    A column selection used to copy its indices from host memory on every call, a copy that waits for the stream.
    """
    key = (index.tobytes(), index.dtype.str + str(index.shape), device)
    held = _INDEX_TENSORS.get(key)
    if held is None:
        if len(_INDEX_TENSORS) >= 64:
            _INDEX_TENSORS.pop(next(iter(_INDEX_TENSORS)))
        held = _INDEX_TENSORS[key] = torch.as_tensor(index, device=device)
    return held


def columns(W: Tensor, index: np.ndarray | None) -> Tensor:
    """``W[:, index]``: no copy for ``None``, a view for consecutive indices."""
    if index is None:
        return W
    if np.all(np.diff(index) == 1):
        return W[:, int(index[0]) : int(index[0]) + index.size]
    return W[:, index_tensor(np.asarray(index), W.device)]


def pairing_index(
    coordinates: Sequence[int] | np.ndarray | None, outputs: int, dim: int
) -> np.ndarray | None:
    """Validated pairing of output ``j`` with input ``coordinates[j]`` (``None``: output ``j`` with input ``j``)."""
    if coordinates is None:
        if outputs != dim:
            raise ValueError(
                f"the field has {outputs} outputs for {dim} inputs; pass coordinates to pair them"
            )
        return None
    index = np.asarray(coordinates)
    if (
        index.shape != (outputs,)
        or not np.issubdtype(index.dtype, np.integer)
        or index.min() < 0
        or index.max() >= dim
    ):
        raise ValueError(f"coordinates must be {outputs} integer indices into the {dim} inputs")
    return index


def distinct_indices(
    coordinates: Sequence[int] | np.ndarray | None, dim: int, *, message: str | None = None
) -> np.ndarray:
    """Distinct integer indices into ``dim`` inputs (all inputs for ``None``)."""
    if coordinates is None:
        return np.arange(dim)
    index = np.asarray(coordinates)
    if (
        index.ndim != 1
        or index.size == 0
        or not np.issubdtype(index.dtype, np.integer)
        or index.min() < 0
        or index.max() >= dim
        or np.unique(index).size != index.size
    ):
        raise ValueError(
            message or f"coordinates must be distinct integer indices into the {dim} inputs"
        )
    return index


def fastpath_activation(spec: Any, *, allow_almost_everywhere: bool, smooth: int = 2) -> Any:
    """Resolve ``spec`` and refuse activations a closed-form stream program cannot differentiate faithfully.

    ``smooth`` is the derivative order the program uses: ``2`` refuses an a.e.-zero second derivative, ``4`` refuses
    derivatives that jump below order four, ``0`` only requires a fast path.
    """
    resolved = get_activation(spec)
    if resolved.fastpath is None:
        raise ValueError(f"activation {resolved.name!r} has no fast-path kernel")
    if smooth == 2:
        _check_piecewise_second_order(
            resolved.name, allow_almost_everywhere=allow_almost_everywhere
        )
    elif smooth == 4 and resolved.name in NOT_C3 and not allow_almost_everywhere:
        raise ValueError(
            f"activation {resolved.name!r} is not three times continuously "
            "differentiable; the bi-Laplacian would drop its singular part. "
            "Pass allow_almost_everywhere=True to opt in explicitly."
        )
    return resolved


def affine_readout(
    layers: Sequence[Any], *, require_hidden: bool = False
) -> tuple[list[tuple[Tensor, Tensor | None, Any]], Tensor, Tensor | None]:
    """``(hidden, W_L, b_L)`` of a vector field whose last layer must be affine."""
    require_layers(layers)
    *hidden, (WL, bL, last) = layers
    if last is not None:
        raise ValueError("the last layer must be affine (spec=None)")
    if require_hidden and not hidden:
        raise ValueError("at least one hidden layer is required")
    WL = torch.as_tensor(WL)
    if WL.ndim != 2:
        raise ValueError("the last layer weight must have shape (m, H)")
    return hidden, WL, bL


def _slope(resolved: Any, z: Tensor) -> Tensor:
    if resolved.derivative is not None:
        out: Tensor = resolved.derivative(z)
        return out
    if resolved.fastpath is not None:
        out = resolved.fastpath(z, 1)
        return out
    _, tangent = torch.func.jvp(resolved.forward, (z,), (torch.ones_like(z),))
    return tangent


def hidden_slopes(
    a: Tensor, hidden: Sequence[tuple[Tensor, Tensor | None, Any]]
) -> tuple[list[Tensor], list[Tensor], Tensor]:
    """Weights, activation slopes ``sigma'(z_l)`` and the last hidden output of one pass.

    ``a`` is unit-major ``(D, B)``; every layer is one GEMM over the leading axis and the slopes are ``(H_l, B)``. An
    affine hidden layer has unit slopes.
    """
    weights: list[Tensor] = []
    slopes: list[Tensor] = []
    for i, (W, b, spec) in enumerate(hidden):
        W = torch.as_tensor(W)
        if W.ndim != 2 or W.shape[1] != a.shape[0]:
            raise ValueError(f"layer {i} weight must have shape (out, {a.shape[0]})")
        z = W @ a
        if b is not None:
            z = z + torch.as_tensor(b)[:, None]
        weights.append(W)
        if spec is None:
            slopes.append(torch.ones_like(z))
            a = z
            continue
        resolved = get_activation(spec)
        slopes.append(_slope(resolved, z))
        a = resolved.forward(z)
    return weights, slopes, a


def readout_value(WL: Tensor, bL: Tensor | None, a: Tensor, n_hidden: int) -> Tensor:
    """``W_L a + b_L`` of the unit-major last hidden output ``a``, shape ``(m, B)``."""
    if WL.shape[1] != a.shape[0]:
        raise ValueError(f"layer {n_hidden} weight must have shape (out, {a.shape[0]})")
    value = WL @ a
    if bL is not None:
        value = value + torch.as_tensor(bL)[:, None]
    return value


def carry_columns(
    weights: Sequence[Tensor], slopes: Sequence[Tensor], seeds: Tensor | Callable[[], Tensor]
) -> Tensor:
    """The columns ``seeds`` ``(H_1, k)`` carried through ``S_n W_n ... W_2 S_1``: ``(H_n, B, k)``."""
    lead = slopes[0][:, :, None]
    T = lead * (seeds() if callable(seeds) else seeds)[:, None, :]
    for W, s in zip(weights[1:], slopes[1:], strict=True):
        T = s[:, :, None] * torch.tensordot(W, T, dims=1)
    return T


def describe_layers(layers: Sequence[Any], *, dtype: str | None = None) -> ChainDescriptor:
    """The :class:`~omnibias.core.chain.ChainDescriptor` of a layer list (names, widths, biases, dtype)."""
    require_layers(layers)
    described = []
    for W, b, spec in layers:
        W = torch.as_tensor(W)
        name = None if spec is None else get_activation(spec).name
        described.append(LayerDescriptor(int(W.shape[1]), int(W.shape[0]), name, b is not None))
    if dtype is None:
        dtype = _DTYPE_NAMES[torch.as_tensor(layers[0][0]).dtype]
    return ChainDescriptor(tuple(described), dtype)
