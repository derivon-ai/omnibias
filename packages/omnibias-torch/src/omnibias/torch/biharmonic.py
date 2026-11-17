# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Experimental exact fourth-order contractions with packed Hessian state.

Each scalar carries value, gradient, symmetric Hessian, gradient of Laplacian,
and biharmonic value: 2 + 2D + D(D+1)/2 real components. No third/fourth
derivative tensor is constructed. State minimality does not imply optimal runtime.
All kernels are ordinary differentiable tensor expressions.
"""

from __future__ import annotations

from collections.abc import Sequence
from math import prod
from typing import NamedTuple

from omnibias.torch.activations import get_activation

import torch
from torch import Tensor


class BiharmonicState(NamedTuple):
    """Raw derivatives, with derivative coordinates on the final axis.

``value`` and ``biharmonic`` have the scalar-value shape; ``gradient`` and
``grad_laplacian`` append D; ``hessian`` appends packed upper-triangle entries
in row-major order. An MLP's value shape is (..., output_features).
    """

    value: Tensor
    gradient: Tensor
    hessian: Tensor
    grad_laplacian: Tensor
    biharmonic: Tensor


def _indices(state: BiharmonicState) -> tuple[Tensor, Tensor, Tensor]:
    dim = state.gradient.shape[-1]
    device = state.value.device
    # Generate compact integer tensors directly, avoiding O(D²) Python tuples
    # and a cache that could retain large dimension-specific index tables.
    triangle = torch.triu_indices(dim, dim, device=device)
    row = torch.arange(dim, device=device)
    diagonal = row * dim - row * (row - 1) // 2
    return triangle[0], triangle[1], diagonal


def _outer(a: Tensor, b: Tensor, i: Tensor, j: Tensor) -> Tensor:
    return a.index_select(-1, i) * b.index_select(-1, j)


def _matvec(hessian: Tensor, v: Tensor, i: Tensor, j: Tensor) -> Tensor:
    # Upper triangle contributes both orientations, except on the diagonal.
    out = torch.zeros_like(v).index_add(-1, i, hessian * v.index_select(-1, j))
    off_diagonal = (i != j).to(hessian.dtype)
    return out.index_add(-1, j, hessian * v.index_select(-1, i) * off_diagonal)


def _laplacian(state: BiharmonicState, diagonal: Tensor) -> Tensor:
    return state.hessian.index_select(-1, diagonal).sum(dim=-1)


def _frobenius(a: Tensor, b: Tensor, i: Tensor, j: Tensor) -> Tensor:
    return (a * b * (1 + (i != j).to(a.dtype))).sum(dim=-1)


def _check_dtype(x: Tensor) -> None:
    if x.dtype not in (torch.float32, torch.float64):
        raise TypeError("biharmonic propagation requires float32 or float64")


def _check_size(value_shape: Sequence[int], dim: int, itemsize: int, max_state_bytes: int) -> None:
    if isinstance(max_state_bytes, bool) or not isinstance(max_state_bytes, int) or max_state_bytes < 1:
        raise ValueError("max_state_bytes must be a positive integer")
    count = 2 + 2 * dim + dim * (dim + 1) // 2
    size = prod(value_shape) * count * itemsize
    if size > max_state_bytes:
        raise ValueError(f"biharmonic state needs {size} bytes, above budget {max_state_bytes}; reduce batch, width or directions")


def biharmonic_seed(x: Tensor, *, max_state_bytes: int = 256 * 1024**2) -> BiharmonicState:
    """Lift coordinate inputs (..., D); use mlp_biharmonic to skip zero input Hessians."""
    _check_dtype(x)
    if x.ndim < 1 or x.shape[-1] < 1:
        raise ValueError("expected a nonempty final coordinate axis")
    d = x.shape[-1]
    _check_size(x.shape, d, x.element_size(), max_state_bytes)
    gradient = torch.eye(d, dtype=x.dtype, device=x.device).expand(*x.shape[:-1], d, d)
    return BiharmonicState(x, gradient, x.new_zeros((*x.shape, d * (d + 1) // 2)),
                          torch.zeros_like(gradient), torch.zeros_like(x))


def biharmonic_add(a: BiharmonicState, b: BiharmonicState) -> BiharmonicState:
    """Add equally shaped derivative states."""
    return BiharmonicState(*(x + y for x, y in zip(a, b, strict=True)))


def biharmonic_multiply(a: BiharmonicState, b: BiharmonicState) -> BiharmonicState:
    """Exact product rule, including all gradient/Hessian cross contractions."""
    i, j, diagonal = _indices(a)
    la, lb = _laplacian(a, diagonal), _laplacian(b, diagonal)
    av, bv = a.value.unsqueeze(-1), b.value.unsqueeze(-1)
    gradient = bv * a.gradient + av * b.gradient
    hessian = (bv * a.hessian + av * b.hessian
               + _outer(a.gradient, b.gradient, i, j) + _outer(b.gradient, a.gradient, i, j))
    q = (bv * a.grad_laplacian + av * b.grad_laplacian
         + la.unsqueeze(-1) * b.gradient + lb.unsqueeze(-1) * a.gradient
         + 2 * (_matvec(a.hessian, b.gradient, i, j) + _matvec(b.hessian, a.gradient, i, j)))
    r = (b.value * a.biharmonic + a.value * b.biharmonic + 2 * la * lb
         + 4 * (a.gradient * b.grad_laplacian).sum(-1)
         + 4 * (b.gradient * a.grad_laplacian).sum(-1)
         + 4 * _frobenius(a.hessian, b.hessian, i, j))
    return BiharmonicState(a.value * b.value, gradient, hessian, q, r)


def biharmonic_activation(state: BiharmonicState, activation: str = "tanh") -> BiharmonicState:
    """Lift a registered smooth elementwise activation using derivatives 1..4."""
    spec = get_activation(activation)
    if spec.fastpath is None:
        raise NotImplementedError(f"activation {activation!r} has no derivative tower")
    tower = (spec.tower(state.value, 4) if spec.tower is not None
             else tuple(spec.fastpath(state.value, k) for k in range(5)))
    s1, s2, s3, s4 = (tower[k] for k in range(1, 5))
    i, j, diagonal = _indices(state)
    lap = _laplacian(state, diagonal)
    norm2 = (state.gradient**2).sum(-1)
    hg = _matvec(state.hessian, state.gradient, i, j)
    gradient = s1.unsqueeze(-1) * state.gradient
    hessian = s1.unsqueeze(-1) * state.hessian + s2.unsqueeze(-1) * _outer(state.gradient, state.gradient, i, j)
    q = (s1.unsqueeze(-1) * state.grad_laplacian
         + s2.unsqueeze(-1) * (lap.unsqueeze(-1) * state.gradient + 2 * hg)
         + (s3 * norm2).unsqueeze(-1) * state.gradient)
    r = (s1 * state.biharmonic
         + s2 * (lap**2 + 2 * _frobenius(state.hessian, state.hessian, i, j)
                 + 4 * (state.gradient * state.grad_laplacian).sum(-1))
         + s3 * (2 * lap * norm2 + 4 * (state.gradient * hg).sum(-1))
         + s4 * norm2**2)
    return BiharmonicState(tower[0], gradient, hessian, q, r)


def biharmonic_affine(state: BiharmonicState, weight: Tensor, bias: Tensor | None = None) -> BiharmonicState:
    """Apply a spatially constant, trainable affine map to the feature axis."""
    return BiharmonicState(
        torch.nn.functional.linear(state.value, weight, bias),
        torch.einsum("oi,...id->...od", weight, state.gradient),
        torch.einsum("oi,...ip->...op", weight, state.hessian),
        torch.einsum("oi,...id->...od", weight, state.grad_laplacian),
        torch.nn.functional.linear(state.biharmonic, weight),
    )


def mlp_biharmonic(
    x: Tensor, layers: Sequence[tuple[Tensor, Tensor | None, str | None]], *,
    max_state_bytes: int = 256 * 1024**2,
) -> BiharmonicState:
    """Exact value/gradient/Hessian/gradient-Laplacian/biharmonic of an MLP.

Layers are (weight, bias, activation), with ``None`` for an affine readout.
The first affine map seeds its Jacobian directly from weights. The byte budget
is per layer's explicit forward state, excluding backend workspaces and tapes.
    """
    _check_dtype(x)
    if x.ndim < 1 or x.shape[-1] < 1:
        raise ValueError("expected a nonempty coordinate axis")
    if not layers:
        return biharmonic_seed(x, max_state_bytes=max_state_bytes)
    d = x.shape[-1]
    for weight, bias, _ in layers:
        _check_dtype(weight)
        if weight.ndim != 2 or weight.dtype != x.dtype or weight.device != x.device:
            raise ValueError("weights must be matrices matching input dtype and device")
        if bias is not None and (bias.dtype != x.dtype or bias.device != x.device):
            raise ValueError("bias must match input dtype and device")
        _check_size((*x.shape[:-1], weight.shape[0]), d, x.element_size(), max_state_bytes)
    w, b, activation = layers[0]
    value = torch.nn.functional.linear(x, w, b)
    g = w.expand(*x.shape[:-1], *w.shape)
    state = BiharmonicState(value, g, value.new_zeros((*value.shape, d * (d + 1) // 2)),
                            torch.zeros_like(g), torch.zeros_like(value))
    if activation is not None:
        state = biharmonic_activation(state, activation)
    for w, b, activation in layers[1:]:
        state = biharmonic_affine(state, w, b)
        if activation is not None:
            state = biharmonic_activation(state, activation)
    return state


__all__ = ["BiharmonicState", "biharmonic_activation", "biharmonic_add",
           "biharmonic_affine", "biharmonic_multiply", "biharmonic_seed", "mlp_biharmonic"]
