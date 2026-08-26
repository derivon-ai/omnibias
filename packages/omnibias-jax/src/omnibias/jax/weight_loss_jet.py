# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Closed-form one-layer weight-space loss jets, JAX twin.

Host-side conversion into :mod:`omnibias.core.weight_loss_jet`. Bit-identical
to the torch twin on the same floats. Do not ``jit`` the driver.
"""

from __future__ import annotations

from collections.abc import Sequence

from omnibias.core import weight_loss_jet as core

import jax.numpy as jnp
from jax import Array

WeightLossJetSpec = core.WeightLossJetSpec
honesty_payload = core.honesty_payload
one_layer_param_count = core.one_layer_param_count
pack_one_layer_params = core.pack_one_layer_params
unpack_one_layer_params = core.unpack_one_layer_params

__all__ = [
    "WeightLossJetSpec",
    "honesty_payload",
    "one_layer_loss",
    "one_layer_loss_grad",
    "one_layer_loss_hessian",
    "one_layer_loss_jet",
    "one_layer_newton_direction",
    "one_layer_param_count",
    "pack_one_layer_params",
    "unpack_one_layer_params",
]


def _rows(matrix: Array) -> list[list[float]]:
    data = jnp.asarray(matrix).reshape(-1, matrix.shape[-1])
    return [[float(v) for v in row] for row in data]


def _vec(vector: Array | Sequence[float]) -> list[float]:
    if isinstance(vector, Array):
        return [float(v) for v in jnp.asarray(vector).reshape(-1)]
    return [float(v) for v in vector]


def one_layer_loss_jet(
    features: Array,
    targets: Array,
    params: Array,
    direction: Array,
    *,
    order: int,
    hidden: int,
    dim: int,
    activation: str = "tanh",
) -> list[float]:
    """``φ^(k)(0)`` via the shared core assembler."""
    return core.one_layer_loss_jet(
        _rows(features),
        _vec(targets),
        _vec(params),
        _vec(direction),
        order=order,
        hidden=hidden,
        dim=dim,
        activation=activation,
    )


def one_layer_loss(
    features: Array,
    targets: Array,
    params: Array,
    hidden: int,
    dim: int,
    activation: str = "tanh",
) -> float:
    return core.one_layer_loss(
        _rows(features), _vec(targets), _vec(params), hidden, dim, activation
    )


def one_layer_loss_grad(
    features: Array,
    targets: Array,
    params: Array,
    hidden: int,
    dim: int,
    activation: str = "tanh",
) -> list[float]:
    return core.one_layer_loss_grad(
        _rows(features), _vec(targets), _vec(params), hidden, dim, activation
    )


def one_layer_loss_hessian(
    features: Array,
    targets: Array,
    params: Array,
    hidden: int,
    dim: int,
    activation: str = "tanh",
    *,
    max_params: int = 256,
) -> list[list[float]]:
    return core.one_layer_loss_hessian(
        _rows(features),
        _vec(targets),
        _vec(params),
        hidden,
        dim,
        activation,
        max_params=max_params,
    )


def one_layer_newton_direction(
    grad: Array | Sequence[float],
    hess: Array | Sequence[Sequence[float]],
    *,
    damping: float = 1e-6,
) -> list[float]:
    if isinstance(hess, Array):
        rows = [[float(v) for v in row] for row in hess]
    else:
        rows = [[float(v) for v in row] for row in hess]
    return core.one_layer_newton_direction(_vec(grad), rows, damping=damping)
