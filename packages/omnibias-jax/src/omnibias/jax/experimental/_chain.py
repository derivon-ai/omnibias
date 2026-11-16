# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Pieces shared by the deep-MLP structured programs.

Every structured program of this package reads the same layer list ``[(W, b, spec), ...]`` with ``W`` of shape
``(out, in)``, ``spec=None`` for an affine layer, and a point array of shape ``(D,)`` or ``(B, D)``. The point
normalization, the coordinate checks, the activation gates and the slope pass of the vector-field programs
(divergence, log-determinant, Woodbury solves) used to be copied into each module; they live here once. Nothing
in this file changes what a program computes: the operations a caller emits are the ones it emitted before.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Protocol, cast

import numpy as np
from omnibias.jax.activations import JaxActivationSpec, get_activation
from omnibias.jax.laplacian import _check_piecewise_second_order

import jax
import jax.numpy as jnp
from jax import Array

__all__ = [
    "FastpathActivation",
    "LayerSpec",
    "NOT_C3",
    "Skip",
    "affine_readout",
    "carry_columns",
    "columns",
    "distinct_indices",
    "fastpath_activation",
    "fourier_feature_layer",
    "hidden_slopes",
    "packed_triangle",
    "pairing_index",
    "points_2d",
    "readout_value",
    "require_layers",
]

LayerSpec = tuple[Array, Array | None, JaxActivationSpec | str | None]

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
    (:func:`~omnibias.jax.experimental.streams.second_order_streams` and the directional jets) carry the identity branch
    through every stream, so a block costs one extra addition and the derivatives stay exact.
    """

    __slots__ = ("spec",)

    def __init__(self, spec: Any) -> None:
        if spec is None or isinstance(spec, Skip):
            raise ValueError("Skip wraps an activation")
        self.spec = spec

    def __repr__(self) -> str:
        return f"Skip({self.spec!r})"


def fourier_feature_layer(frequencies: Any, phase: Any = None) -> Any:
    """The Fourier-feature embedding ``[sin(B x + c), cos(B x + c)]`` as one exact ``(W, b, "sin")`` layer.

    ``frequencies`` is ``B`` of shape ``(m, D)`` (fold any ``2 pi`` into it) and ``phase`` an optional ``(m,)`` offset
    ``c``. Since ``cos z = sin(z + pi / 2)`` the embedding is a single layer of width ``2 m`` with a sine activation, so
    every stream program carries it with no special case, and a trainable ``B`` receives exact derivatives.
    """
    B = jnp.asarray(frequencies)
    c = jnp.zeros(B.shape[0], B.dtype) if phase is None else jnp.asarray(phase, B.dtype)
    return (
        jnp.concatenate([B, B], axis=0),
        jnp.concatenate([c, c + (jnp.pi / 2)], axis=0),
        "sin",
    )


def require_layers(layers: Sequence[Any]) -> None:
    if not layers:
        raise ValueError("layers must be non-empty")


def points_2d(x: Array) -> tuple[Array, bool]:
    """``(a, batched)``: ``a`` has shape ``(B, D)`` (a single point gains a leading axis)."""
    x = jnp.asarray(x)
    batched = x.ndim > 1
    a = x if batched else x[None, :]
    if a.ndim != 2:
        raise ValueError("x must have shape (D,) or (B, D)")
    return a, batched


def columns(W: Array, index: np.ndarray | None) -> Array:
    """``W[:, index]``: no copy for ``None``, a static slice for consecutive indices."""
    if index is None:
        return W
    if np.all(np.diff(index) == 1):
        return W[:, int(index[0]) : int(index[0]) + index.size]
    return W[:, index]


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


class FastpathActivation(Protocol):
    """An activation whose closed-form derivative kernel is known to exist (``fastpath`` is not ``None``)."""

    @property
    def name(self) -> str: ...

    @property
    def forward(self) -> Callable[[Array], Array]: ...

    @property
    def fastpath(self) -> Callable[[Array, int], Array]: ...


def fastpath_activation(
    spec: JaxActivationSpec | str, *, allow_almost_everywhere: bool, smooth: int = 2
) -> FastpathActivation:
    """Resolve ``spec`` and refuse activations a closed-form stream program cannot differentiate faithfully.

    ``smooth`` is the derivative order the program uses: ``2`` refuses an a.e.-zero second derivative, ``4``
    (the bi-Laplacian family) refuses derivatives that jump below order four, ``0`` only requires a fast path.
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
    return cast(FastpathActivation, resolved)


def packed_triangle(k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Upper-triangle ``rows, cols`` of a symmetric ``k x k`` matrix and the ``(k * k,)`` index that unpacks it."""
    rows, cols = np.triu_indices(k)
    index = np.empty((k, k), np.int64)
    index[rows, cols] = np.arange(rows.size)
    index[cols, rows] = np.arange(rows.size)
    return rows, cols, index.reshape(-1)


def affine_readout(
    layers: Sequence[LayerSpec], *, require_hidden: bool = False
) -> tuple[list[tuple[Array, Array | None, Any]], Array, Array | None]:
    """``(hidden, W_L, b_L)`` of a vector field whose last layer must be affine."""
    require_layers(layers)
    *hidden, (WL, bL, last) = layers
    if last is not None:
        raise ValueError("the last layer must be affine (spec=None)")
    if require_hidden and not hidden:
        raise ValueError("at least one hidden layer is required")
    WL = jnp.asarray(WL)
    if WL.ndim != 2:
        raise ValueError("the last layer weight must have shape (m, H)")
    return hidden, WL, bL


def hidden_slopes(
    a: Array, hidden: Sequence[tuple[Array, Array | None, Any]]
) -> tuple[list[Array], list[Array], Array]:
    """Weights, activation slopes ``sigma'(z_l)`` and the last hidden output of one pass.

    ``a`` is unit-major ``(D, B)``; every layer is one GEMM over the leading axis and the slopes are
    ``(H_l, B)``. An affine hidden layer has unit slopes. Activations without a closed-form derivative use one
    forward-mode tangent.
    """
    weights: list[Array] = []
    slopes: list[Array] = []
    for i, (W, b, spec) in enumerate(hidden):
        W = jnp.asarray(W)
        if W.ndim != 2 or W.shape[1] != a.shape[0]:
            raise ValueError(f"layer {i} weight must have shape (out, {a.shape[0]})")
        z = W @ a
        if b is not None:
            z = z + jnp.asarray(b)[:, None]
        weights.append(W)
        if spec is None:
            slopes.append(jnp.ones_like(z))
            a = z
            continue
        resolved = get_activation(spec)
        if resolved.derivative is not None:
            slopes.append(resolved.derivative(z))
        else:
            slopes.append(jax.jvp(resolved.forward, (z,), (jnp.ones_like(z),))[1])
        a = resolved.forward(z)
    return weights, slopes, a


def readout_value(WL: Array, bL: Array | None, a: Array, n_hidden: int) -> Array:
    """``W_L a + b_L`` of the unit-major last hidden output ``a``, shape ``(m, B)``."""
    if WL.shape[1] != a.shape[0]:
        raise ValueError(f"layer {n_hidden} weight must have shape (out, {a.shape[0]})")
    value = WL @ a
    if bL is not None:
        value = value + jnp.asarray(bL)[:, None]
    return value


def carry_columns(
    weights: Sequence[Array], slopes: Sequence[Array], seeds: Array | Callable[[], Array]
) -> Array:
    """The columns ``seeds`` ``(H_1, k)`` carried through ``S_n W_n ... W_2 S_1``: ``(H_n, B, k)``.

    A callable ``seeds`` is evaluated after the leading slopes are broadcast, which keeps the operation order of a
    caller that formed the seeds inline.
    """
    lead = slopes[0][:, :, None]
    T = lead * (seeds() if callable(seeds) else seeds)[:, None, :]
    for W, s in zip(weights[1:], slopes[1:], strict=True):
        T = s[:, :, None] * jnp.tensordot(W, T, axes=1)
    return T
