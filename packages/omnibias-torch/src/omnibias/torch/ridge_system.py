# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Forward evaluation of exact, activation-independent PDE ridge syzygies."""

from __future__ import annotations

from omnibias.core.ridge_system import RidgeSyzygy
from omnibias.torch.activations import get_activation
from omnibias.torch.jet import _sigma_tower

import torch
from torch import Tensor

_SMOOTH_PROFILES = frozenset(
    ("sigmoid", "tanh", "softplus", "gaussian", "exp", "sin", "cos", "sinh", "cosh")
)


def ridge_solution(
    x: Tensor,
    pack: RidgeSyzygy,
    bias: float | Tensor = 0.0,
    *,
    activation: str = "tanh",
) -> Tensor:
    r"""Evaluate ``sum_j pack.coefficients[j] * phi^(j)(w @ x + bias)``.

    ``x`` has shape ``(..., dimension)`` and the result ``(..., outputs)``.
    Derivatives are RAW derivatives, not Taylor coefficients. The immutable
    rational direction and coefficients use ``x``'s floating dtype and device.
    Packs obtained from ``RidgeSystem.kernel_basis`` satisfy that system;
    this evaluator does not verify PDE membership of user-created packs.
    Algebraic cancellation is exact over the rationals, not necessarily in
    floating-point arithmetic.

    The scalar ``bias`` remains trainable. Sum multiple calls with scalar
    trainable readout weights to fit data while preserving the homogeneous
    PDE. Coordinate-dependent biases/readouts or arbitrary output mixing
    generally destroy that property. Directions are deliberately fixed.

    Supported profiles are sigmoid, tanh, softplus, gaussian, exp, sin, cos,
    sinh and cosh (including registered aliases). They are globally smooth;
    piecewise activation towers that omit singular derivatives are refused.
    This evaluates a solution family, not boundary conditions or a complete
    PDE solver. Large arguments/orders can still overflow or lose accuracy.
    """
    if not isinstance(pack, RidgeSyzygy):
        raise TypeError("pack must be a RidgeSyzygy")
    if x.ndim < 1 or x.shape[-1] != len(pack.direction):
        raise ValueError("x must have shape (..., len(pack.direction))")
    if not x.is_floating_point():
        raise TypeError("x must have a real floating dtype")
    if not isinstance(activation, str):
        raise TypeError("activation must be a supported smooth profile name")
    spec = get_activation(activation)
    if spec.name not in _SMOOTH_PROFILES:
        raise ValueError(f"activation {activation!r} is not a supported globally smooth profile")
    scalar_bias = torch.as_tensor(bias, dtype=x.dtype, device=x.device)
    if scalar_bias.ndim != 0:
        raise ValueError("bias must be scalar to preserve the ridge direction")
    direction = x.new_tensor([float(value) for value in pack.direction])
    coefficients = x.new_tensor(
        [[float(value) for value in row] for row in pack.coefficients]
    )
    z = x @ direction + scalar_bias
    tower = _sigma_tower(spec, z, pack.degree)
    result: Tensor = torch.tensordot(tower, coefficients, dims=([0], [0]))
    return result


__all__ = ["ridge_solution"]
