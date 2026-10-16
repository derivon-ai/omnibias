# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""JIT-compatible evaluation of exact PDE ridge syzygies with runtime biases."""

from __future__ import annotations

from omnibias.core.ridge_system import RidgeSyzygy
from omnibias.jax.activations import get_activation
from omnibias.jax.jet import _sigma_tower

import jax.numpy as jnp
from jax import Array

_SMOOTH_PROFILES = frozenset(
    ("sigmoid", "tanh", "softplus", "gaussian", "exp", "sin", "cos", "sinh", "cosh")
)


def ridge_solution(
    x: Array,
    pack: RidgeSyzygy,
    bias: float | Array = 0.0,
    *,
    activation: str = "tanh",
) -> Array:
    r"""Evaluate ``sum_j pack.coefficients[j] * phi^(j)(w @ x + bias)``.

    ``x`` has shape ``(..., dimension)`` and the result ``(..., outputs)``.
    Derivatives are RAW derivatives, not Taylor coefficients. Direction and
    coefficients are immutable rational metadata, converted to ``x.dtype``.
    Packs obtained from ``RidgeSystem.kernel_basis`` satisfy that system;
    this evaluator does not verify PDE membership of user-created packs.
    Close over ``pack`` and ``activation`` in a JIT-compiled function, or mark
    them static; ``x`` and scalar ``bias`` remain runtime arguments supporting
    ``jit``, ``vmap`` and parameter gradients.

    Sum multiple calls with scalar trainable readout weights to fit data while
    preserving the homogeneous PDE. Coordinate-dependent biases/readouts or
    arbitrary output mixing generally destroy that property. Directions are
    deliberately fixed. Algebraic cancellation is exact over the rationals,
    not necessarily in floating-point arithmetic.

    Supported globally smooth profiles are sigmoid, tanh, softplus, gaussian,
    exp, sin, cos, sinh and cosh (including registered aliases). Piecewise
    towers omitting singular derivatives are refused. Boundary conditions,
    completeness and floating-point stability are separate responsibilities;
    large arguments/orders can still overflow or lose accuracy.
    """
    if not isinstance(pack, RidgeSyzygy):
        raise TypeError("pack must be a RidgeSyzygy")
    if x.ndim < 1 or x.shape[-1] != len(pack.direction):
        raise ValueError("x must have shape (..., len(pack.direction))")
    if not jnp.issubdtype(x.dtype, jnp.floating):
        raise TypeError("x must have a real floating dtype")
    if not isinstance(activation, str):
        raise TypeError("activation must be a supported smooth profile name")
    spec = get_activation(activation)
    if spec.name not in _SMOOTH_PROFILES:
        raise ValueError(f"activation {activation!r} is not a supported globally smooth profile")
    scalar_bias = jnp.asarray(bias, dtype=x.dtype)
    if scalar_bias.ndim != 0:
        raise ValueError("bias must be scalar to preserve the ridge direction")
    direction = jnp.asarray([float(value) for value in pack.direction], dtype=x.dtype)
    coefficients = jnp.asarray(
        [[float(value) for value in row] for row in pack.coefficients], dtype=x.dtype
    )
    z = x @ direction + scalar_bias
    tower = _sigma_tower(spec, z, pack.degree)
    return jnp.tensordot(tower, coefficients, axes=([0], [0]))


__all__ = ["ridge_solution"]
