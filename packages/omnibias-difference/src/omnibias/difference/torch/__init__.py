# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""PyTorch twin of the founding finite-difference stencil operator.

Batched, differentiable ``finite_difference_tower`` / ``collapse_to_derivative``
built on the **shared pure-Python** stencil numbers in
:mod:`omnibias.difference._core.stencil`, so this operator and its JAX twin
(:mod:`omnibias.difference.jax`) are bit-identical by construction (only the base
activation's libm evaluation differs across backends, by a few ulp).

``finite_difference_tower`` is the literal multi-bias sum ``sum_k s_k sigma(z + b_k)``
(numerical); ``collapse_to_derivative`` is the closed-form ``sigma^(order)(z)`` from
the backend fastpath. Their residual ``-> 0`` as ``delta -> 0`` -- the founding
bias collapse of many biases into the derivative ``sigma^(K-1)``.
"""

from __future__ import annotations

import torch
from omnibias.difference._core.stencil import Stencil, stencil_offsets, stencil_signs
from omnibias.torch.activations.registry import get_activation
from torch import Tensor


def finite_difference_tower(
    name: str, z: Tensor, order: int, delta: float, stencil: Stencil = "central"
) -> Tensor:
    """Numerical multi-bias finite-difference stencil ``sum_k s_k sigma(z + b_k)``.

    The signs and offsets come from the shared pure-Python
    :mod:`omnibias.difference._core.stencil`, materialised into tensors matching
    ``z``'s dtype / device, so the value is bit-identical to the JAX twin up to the
    base activation's per-backend rounding.
    """
    spec = get_activation(name)
    signs = torch.tensor(stencil_signs(order, delta, stencil), dtype=z.dtype, device=z.device)
    offsets = torch.tensor(stencil_offsets(order, delta, stencil), dtype=z.dtype, device=z.device)
    values = spec.forward(z.unsqueeze(-1) + offsets)  # (..., K)
    return (values * signs).sum(dim=-1)


def collapse_to_derivative(name: str, z: Tensor, order: int) -> Tensor:
    """Closed-form ``sigma^(order)(z)`` via the backend fastpath (the exact limit)."""
    if order < 0:
        raise ValueError(f"order must be >= 0, got {order}")
    spec = get_activation(name)
    if spec.fastpath is None:
        raise NotImplementedError(f"activation {name!r} has no closed-form derivative kernel")
    out: Tensor = spec.fastpath(z, order)
    return out


def finite_difference_residual(
    name: str, z: Tensor, order: int, delta: float, stencil: Stencil = "central"
) -> Tensor:
    """``finite_difference_tower - collapse_to_derivative``; ``-> 0`` as ``delta -> 0``."""
    return finite_difference_tower(name, z, order, delta, stencil) - collapse_to_derivative(
        name, z, order
    )


__all__ = [
    "collapse_to_derivative",
    "finite_difference_residual",
    "finite_difference_tower",
]
