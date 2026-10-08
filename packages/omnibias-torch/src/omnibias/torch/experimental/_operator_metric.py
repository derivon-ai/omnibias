# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Private weighted Laplacian states for affine operator transport.

The spatially constant symmetric metric ``gram`` remains a live tensor. The
compiler proves this condition and checks state budgets before calling these
kernels. All derivatives are raw; Hessians retain only their upper triangle.
Dense Hessian temporaries are O(d²), never third/fourth derivative tensors.
"""

from __future__ import annotations

from math import factorial
from types import SimpleNamespace
from typing import NamedTuple

from omnibias.core.jet_algebra import JetSpec
from omnibias.core.jet_series import activation

import torch
import torch.nn.functional as F
from torch import Tensor

_OPS = SimpleNamespace(
    exp=torch.exp, log=torch.log, sqrt=torch.sqrt, tanh=torch.tanh,
    sigmoid=torch.sigmoid, sin=torch.sin, cos=torch.cos, erf=torch.erf,
    erfc=torch.erfc, zeros_like=torch.zeros_like, ones_like=torch.ones_like,
    logaddexp=torch.logaddexp, softplus=F.softplus,
)
_UNIT2 = JetSpec.directional(2)
_UNIT4 = JetSpec.directional(4)


def _tower(value: Tensor, name: str, order: int) -> list[Tensor]:
    spec = _UNIT2 if order == 2 else _UNIT4
    rows = [value, torch.ones_like(value)] + [torch.zeros_like(value)] * (order - 1)
    normalized = activation(rows, spec, name, _OPS)
    return [row * float(factorial(k)) for k, row in enumerate(normalized)]


class LaplacianState(NamedTuple):
    value: Tensor
    gradient: Tensor
    laplacian: Tensor


class MetricState(NamedTuple):
    value: Tensor
    gradient: Tensor
    hessian: Tensor
    grad_laplacian: Tensor
    biharmonic: Tensor


def laplacian_seed(x: Tensor, dim: int) -> LaplacianState:
    gradient = torch.zeros_like(x).unsqueeze(-1) + x.new_ones((dim,)).diag()
    return LaplacianState(x, gradient, torch.zeros_like(x))


def laplacian_affine(state: LaplacianState, weight: Tensor,
                     bias: Tensor | None = None) -> LaplacianState:
    # One matrix kernel applies the same affine weight to every retained
    # coefficient. Only the primal row receives the source bias.
    packed = torch.cat((state.value.unsqueeze(-2), state.gradient.transpose(-1, -2),
                        state.laplacian.unsqueeze(-2)), dim=-2)
    result = F.linear(packed, weight)
    value = result[..., 0, :]
    if bias is not None:
        value = value + bias
    return LaplacianState(value, result[..., 1:-1, :].transpose(-1, -2), result[..., -1, :])


def laplacian_add(a: LaplacianState, b: LaplacianState) -> LaplacianState:
    return LaplacianState(*(x + y for x, y in zip(a, b, strict=True)))


def laplacian_multiply(a: LaplacianState, b: LaplacianState,
                       gram: Tensor) -> LaplacianState:
    return LaplacianState(
        a.value * b.value,
        b.value.unsqueeze(-1) * a.gradient + a.value.unsqueeze(-1) * b.gradient,
        b.value * a.laplacian + a.value * b.laplacian
        + 2 * ((a.gradient @ gram) * b.gradient).sum(-1),
    )


def laplacian_activation(state: LaplacianState, gram: Tensor,
                         name: str = "tanh") -> LaplacianState:
    value, s1, s2 = _tower(state.value, name, 2)
    square = ((state.gradient @ gram) * state.gradient).sum(-1)
    return LaplacianState(value, s1.unsqueeze(-1) * state.gradient,
                          s1 * state.laplacian + s2 * square)


def _triangle(dim: int) -> tuple[list[int], list[int]]:
    # Host-only immutable-shape metadata. No parameter values enter this plan.
    return ([i for i in range(dim) for _ in range(i, dim)],
            [j for i in range(dim) for j in range(i, dim)])


def _dense(packed: Tensor, dim: int) -> Tensor:
    indices = [min(i, j) * dim - min(i, j) * (min(i, j) - 1) // 2
               + abs(j - i) for i in range(dim) for j in range(dim)]
    return packed[..., indices].reshape(packed.shape[:-1] + (dim, dim))


def _outer(a: Tensor, b: Tensor, dim: int) -> Tensor:
    i, j = _triangle(dim)
    return a[..., i] * b[..., j]


def _mv(matrix: Tensor, vector: Tensor) -> Tensor:
    return (matrix @ vector.unsqueeze(-1)).squeeze(-1)


def _trace_product(a: Tensor, b: Tensor) -> Tensor:
    return (a * b.transpose(-1, -2)).sum(dim=(-2, -1))


def metric_seed(x: Tensor, dim: int) -> MetricState:
    zero = torch.zeros_like(x)
    gradient = zero.unsqueeze(-1) + x.new_ones((dim,)).diag()
    hessian = zero.unsqueeze(-1) * x.new_zeros((dim * (dim + 1) // 2,))
    return MetricState(x, gradient, hessian, torch.zeros_like(gradient), zero)


def metric_laplacian(state: MetricState, gram: Tensor, dim: int) -> Tensor:
    return _trace_product(_dense(state.hessian, dim), gram)


def metric_affine(state: MetricState, weight: Tensor,
                  bias: Tensor | None = None) -> MetricState:
    dim, triangle = state.gradient.shape[-1], state.hessian.shape[-1]
    packed = torch.cat((state.value.unsqueeze(-2), state.gradient.transpose(-1, -2),
                        state.hessian.transpose(-1, -2), state.grad_laplacian.transpose(-1, -2),
                        state.biharmonic.unsqueeze(-2)), dim=-2)
    result = F.linear(packed, weight)
    value = result[..., 0, :]
    if bias is not None:
        value = value + bias
    return MetricState(value, result[..., 1:1 + dim, :].transpose(-1, -2),
                       result[..., 1 + dim:1 + dim + triangle, :].transpose(-1, -2),
                       result[..., 1 + dim + triangle:-1, :].transpose(-1, -2), result[..., -1, :])


def metric_add(a: MetricState, b: MetricState) -> MetricState:
    return MetricState(*(x + y for x, y in zip(a, b, strict=True)))


def metric_multiply(a: MetricState, b: MetricState,
                    gram: Tensor, dim: int) -> MetricState:
    ha, hb = _dense(a.hessian, dim), _dense(b.hessian, dim)
    la, lb = _trace_product(ha, gram), _trace_product(hb, gram)
    ga, gb = a.gradient @ gram, b.gradient @ gram
    av, bv = a.value.unsqueeze(-1), b.value.unsqueeze(-1)
    return MetricState(
        a.value * b.value,
        bv * a.gradient + av * b.gradient,
        bv * a.hessian + av * b.hessian
        + _outer(a.gradient, b.gradient, dim) + _outer(b.gradient, a.gradient, dim),
        bv * a.grad_laplacian + av * b.grad_laplacian
        + la.unsqueeze(-1) * b.gradient + lb.unsqueeze(-1) * a.gradient
        + 2 * (_mv(ha, gb) + _mv(hb, ga)),
        b.value * a.biharmonic + a.value * b.biharmonic + 2 * la * lb
        + 4 * (ga * b.grad_laplacian).sum(-1)
        + 4 * (gb * a.grad_laplacian).sum(-1)
        + 4 * _trace_product(ha @ gram, hb @ gram),
    )


def metric_activation(state: MetricState, gram: Tensor, dim: int,
                      name: str = "tanh") -> MetricState:
    value, s1, s2, s3, s4 = _tower(state.value, name, 4)
    hessian = _dense(state.hessian, dim)
    hg_matrix = hessian @ gram
    lap = _trace_product(hessian, gram)
    metric_gradient = state.gradient @ gram
    square = (metric_gradient * state.gradient).sum(-1)
    hg = _mv(hessian, metric_gradient)
    return MetricState(
        value,
        s1.unsqueeze(-1) * state.gradient,
        s1.unsqueeze(-1) * state.hessian
        + s2.unsqueeze(-1) * _outer(state.gradient, state.gradient, dim),
        s1.unsqueeze(-1) * state.grad_laplacian
        + s2.unsqueeze(-1) * (lap.unsqueeze(-1) * state.gradient + 2 * hg)
        + (s3 * square).unsqueeze(-1) * state.gradient,
        s1 * state.biharmonic
        + s2 * (lap.square() + 2 * _trace_product(hg_matrix, hg_matrix)
                + 4 * (metric_gradient * state.grad_laplacian).sum(-1))
        + s3 * (2 * lap * square + 4 * (metric_gradient * hg).sum(-1))
        + s4 * square.square(),
    )
