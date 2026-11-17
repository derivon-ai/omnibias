# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""One layer convention for the structured programs.

The deep-MLP programs read ``layers = [(W, b, spec), ...]`` with ``W`` of shape ``(out, in)``, ``b`` of shape
``(out,)`` or ``None``, and ``spec`` an activation (name or spec object) or ``None`` for an affine layer. The older
one-layer programs read ``(W, beta, c)`` or ``(W, b, c)`` in two different argument orders, and the separable-grid
tables stack per-axis ``(in, out)`` weights. The adapters below turn each of them into the layer list, and
:func:`describe_layers` turns a layer list into the backend-neutral
:class:`~omnibias.core.chain.ChainDescriptor` a planner keys on.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from omnibias.core.chain import ChainDescriptor, LayerDescriptor
from omnibias.jax.activations import get_activation
from omnibias.jax.experimental._chain import LayerSpec, points_2d

import jax.numpy as jnp
from jax import Array

__all__ = [
    "as_batch",
    "describe_layers",
    "layers_of",
    "ridge_layers",
    "separable_axis_layers",
]

_DTYPE_NAMES = {"float16", "bfloat16", "float32", "float64"}


def as_batch(x: Array) -> tuple[Array, bool]:
    """``(a, batched)``: ``x`` as an array of shape ``(B, D)``; ``batched`` is ``False`` for a single point ``(D,)``."""
    return points_2d(x)


def ridge_layers(
    W: Array,
    b: Array,
    c: Array,
    activation: Any = "tanh",
    *,
    readout_bias: Array | float | None = None,
) -> list[LayerSpec]:
    """The layer list of ``u(x) = readout_bias + sum_h c_h activation(W_h . x + b_h)``.

    ``W`` is ``(H, D)``, ``b`` and ``c`` are ``(H,)``; ``c`` may also be ``(C, H)`` for ``C`` outputs. The argument
    order is the one of :func:`omnibias.jax.experimental.closed_ridge_residual_loss`.
    """
    W, b, c = jnp.asarray(W), jnp.asarray(b), jnp.asarray(c)
    if W.ndim != 2 or b.shape != (W.shape[0],):
        raise ValueError("W must have shape (H, D) and b shape (H,)")
    readout = c[None, :] if c.ndim == 1 else c
    if readout.ndim != 2 or readout.shape[1] != W.shape[0]:
        raise ValueError("c must have shape (H,) or (C, H)")
    rb = (
        None
        if readout_bias is None
        else jnp.broadcast_to(jnp.asarray(readout_bias), (readout.shape[0],))
    )
    return [(W, b, activation), (readout, rb, None)]


def separable_axis_layers(
    layers: Sequence[tuple[Array, Array]], axis: int, activation: Any = "tanh"
) -> list[LayerSpec]:
    """The layer list of one axis network of :func:`~omnibias.jax.experimental.mlp_axis_tables`.

    ``layers[l] = (W (D, n_l, n_{l+1}), b (D, n_{l+1}))`` stacks the per-axis networks with ``(in, out)`` weights;
    the result transposes axis ``axis`` to the ``(out, in)`` convention and leaves the last layer affine.
    """
    if not layers:
        raise ValueError("layers must be non-empty")
    out: list[LayerSpec] = []
    last = len(layers) - 1
    for i, (W, b) in enumerate(layers):
        W, b = jnp.asarray(W), jnp.asarray(b)
        if W.ndim != 3 or b.ndim != 2 or b.shape != (W.shape[0], W.shape[2]):
            raise ValueError("layer weights must have shape (D, in, out) and biases (D, out)")
        if not 0 <= axis < W.shape[0]:
            raise ValueError(f"axis must be in [0, {W.shape[0]})")
        out.append((W[axis].T, b[axis], None if i == last else activation))
    return out


def layers_of(model: Any) -> list[LayerSpec]:
    """The layer list of ``model``.

    Accepts a sequence of ``(W, b, spec)`` (validated and returned as a list) or an object that provides its layers
    through ``layer_specs()`` (the architectures of :mod:`omnibias.jax.architectures` expose ``_layer_specs()``).
    """
    if isinstance(model, Sequence) and not isinstance(model, str):
        layers = list(model)
    else:
        provider = getattr(model, "layer_specs", None) or getattr(model, "_layer_specs", None)
        if provider is None:
            raise TypeError(
                "expected a sequence of (W, b, spec) or an object with layer_specs(); "
                f"got {type(model).__name__}"
            )
        layers = list(provider())
    if not layers:
        raise ValueError("layers must be non-empty")
    previous: int | None = None
    for i, layer in enumerate(layers):
        if len(layer) != 3:
            raise ValueError(f"layer {i} must be a (W, b, spec) triple")
        W = jnp.asarray(layer[0])
        if W.ndim != 2 or (previous is not None and W.shape[1] != previous):
            raise ValueError(
                f"layer {i} weight must have shape (out, {previous if previous is not None else 'in'})"
            )
        previous = W.shape[0]
    return layers


def describe_layers(layers: Sequence[LayerSpec], *, dtype: str | None = None) -> ChainDescriptor:
    """The :class:`~omnibias.core.chain.ChainDescriptor` of a layer list (names, widths, biases, dtype)."""
    layers = layers_of(layers)
    described = []
    for W, b, spec in layers:
        W = jnp.asarray(W)
        name = None if spec is None else get_activation(spec).name
        described.append(LayerDescriptor(W.shape[1], W.shape[0], name, b is not None))
    if dtype is None:
        dtype = jnp.asarray(layers[0][0]).dtype.name
    if dtype not in _DTYPE_NAMES:
        raise ValueError(f"dtype must be one of {sorted(_DTYPE_NAMES)}, got {dtype!r}")
    return ChainDescriptor(tuple(described), dtype)
