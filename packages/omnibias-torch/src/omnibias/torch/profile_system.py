# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Evaluate carrier-modulated profiles through their annihilator companion ODE."""

from __future__ import annotations

from collections.abc import Sequence

from omnibias.core.profile_system import ProfilePack

import torch
from torch import Tensor


def profile_solution(
    x: Tensor,
    pack: ProfilePack,
    initial_jet: Tensor | Sequence[float],
    bias: float | Tensor = 0.0,
) -> Tensor:
    r"""Evaluate ``exp(a @ x) * sum_j q[j] * phi^(j)(w @ x + bias)``.

    ``pack`` fixes rational carrier ``a``, direction ``w``, coefficients ``q``
    and a monic annihilator ``R``. The profile solves ``R(D) phi = 0`` with
    trainable RAW initial derivatives ``initial_jet[j] = phi^(j)(0)``. The
    companion state is evaluated as ``exp(z * C) @ initial_jet``; repeated
    roots, trigonometric and exponential profiles use the same computation.

    ``x`` is float32 or float64 with shape ``(..., dimension)``; the return
    shape is ``(..., outputs)``. ``initial_jet`` has shape ``(profile_order,)``
    and ``bias`` is scalar. Their conversion preserves parameter gradients
    while matching ``x``'s dtype/device. The pack is fixed metadata.
    Initial derivatives, bias and mixture weights must be spatially constant;
    computing them from ``x`` generally invalidates the PDE identity.

    This evaluator does not verify PDE membership of manually created packs.
    For compiler-produced homogeneous packs, changing initial derivatives,
    scalar biases and scalar mixture weights preserves the PDE. For a
    particular inhomogeneous pack, changing the initial derivatives or bias
    also changes its forcing; arbitrary output mixing is not invariant.

    Dense matrix exponentials need quadratic state storage per input point.
    Exact rational algebra does not certify floating-point evaluation:
    large arguments, poorly conditioned companions and exponential growth
    can overflow or lose accuracy. This does not solve boundary conditions.
    """
    if not isinstance(pack, ProfilePack):
        raise TypeError("pack must be a ProfilePack")
    if x.ndim < 1 or x.shape[-1] != len(pack.direction):
        raise ValueError("x must have shape (..., len(pack.direction))")
    if x.dtype not in (torch.float32, torch.float64):
        raise TypeError("x must have float32 or float64 dtype")
    initial = torch.as_tensor(initial_jet, dtype=x.dtype, device=x.device)
    if initial.shape != (pack.profile_order,):
        raise ValueError("initial_jet must have shape (pack.profile_order,)")
    scalar_bias = torch.as_tensor(bias, dtype=x.dtype, device=x.device)
    if scalar_bias.ndim != 0:
        raise ValueError("bias must be scalar to preserve the profile direction")
    direction = x.new_tensor([float(value) for value in pack.direction])
    carrier = x.new_tensor([float(value) for value in pack.carrier])
    coefficients = x.new_tensor([[float(value) for value in row] for row in pack.coefficients])
    order = pack.profile_order
    companion = x.new_tensor(
        [[float(j == i + 1) for j in range(order)] for i in range(order - 1)]
        + [[-float(value) for value in pack.annihilator[:-1]]]
    )
    z = x @ direction + scalar_bias
    state = torch.linalg.matrix_exp(z[..., None, None] * companion) @ initial
    result: Tensor = torch.exp(x @ carrier)[..., None] * (state @ coefficients)
    return result


__all__ = ["profile_solution"]
