# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Closed-form one-layer weight-space loss jets, PyTorch twin.

Tensors are converted to host floats and evaluated by
:mod:`omnibias.core.weight_loss_jet`. The jax twin is bit-identical on
the same floats. Do not wrap the driver in ``torch.compile``.
"""

from __future__ import annotations

from collections.abc import Sequence

from omnibias.core import weight_loss_jet as core

import torch
from torch import Tensor

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


def _rows(matrix: Tensor) -> list[list[float]]:
    data = torch.as_tensor(matrix).detach().reshape(-1, matrix.shape[-1])
    return [[float(v) for v in row] for row in data]


def _vec(vector: Tensor | Sequence[float]) -> list[float]:
    if isinstance(vector, Tensor):
        return [float(v) for v in torch.as_tensor(vector).detach().reshape(-1)]
    return [float(v) for v in vector]


def one_layer_loss_jet(
    features: Tensor,
    targets: Tensor,
    params: Tensor,
    direction: Tensor,
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
    features: Tensor,
    targets: Tensor,
    params: Tensor,
    hidden: int,
    dim: int,
    activation: str = "tanh",
) -> float:
    return core.one_layer_loss(
        _rows(features), _vec(targets), _vec(params), hidden, dim, activation
    )


def one_layer_loss_grad(
    features: Tensor,
    targets: Tensor,
    params: Tensor,
    hidden: int,
    dim: int,
    activation: str = "tanh",
) -> list[float]:
    return core.one_layer_loss_grad(
        _rows(features), _vec(targets), _vec(params), hidden, dim, activation
    )


def one_layer_loss_hessian(
    features: Tensor,
    targets: Tensor,
    params: Tensor,
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
    grad: Tensor | Sequence[float],
    hess: Tensor | Sequence[Sequence[float]],
    *,
    damping: float = 1e-6,
) -> list[float]:
    if isinstance(hess, Tensor):
        rows = [[float(v) for v in row] for row in hess.detach()]
    else:
        rows = [[float(v) for v in row] for row in hess]
    return core.one_layer_newton_direction(_vec(grad), rows, damping=damping)
