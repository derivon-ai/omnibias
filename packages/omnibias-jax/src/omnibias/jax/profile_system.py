# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""JIT-compatible carrier-modulated profiles from an annihilator companion ODE."""

from __future__ import annotations

from collections.abc import Sequence

from omnibias.core.profile_system import ProfilePack

import jax
import jax.numpy as jnp
from jax import Array
from jax.scipy.linalg import expm


def profile_solution(
    x: Array,
    pack: ProfilePack,
    initial_jet: Array | Sequence[float],
    bias: float | Array = 0.0,
) -> Array:
    r"""Evaluate ``exp(a @ x) * sum_j q[j] * phi^(j)(w @ x + bias)``.

    ``pack`` fixes rational carrier ``a``, direction ``w``, coefficients ``q``
    and a monic annihilator ``R``. The profile solves ``R(D) phi = 0`` with
    trainable RAW initial derivatives ``initial_jet[j] = phi^(j)(0)``. The
    companion state ``exp(z * C) @ initial_jet`` includes repeated roots,
    trigonometric and exponential profiles without special cases.

    ``x`` is float32 or float64 with shape ``(..., dimension)``; the return
    shape is ``(..., outputs)``. ``initial_jet`` has shape ``(profile_order,)``
    and ``bias`` is scalar. Close over ``pack`` when using ``jit``; coordinates,
    initial derivatives and bias remain runtime arguments supporting ``vmap``
    and parameter gradients. Computation uses ``x.dtype``.

    This evaluator does not verify PDE membership of manually created packs.
    For compiler-produced homogeneous packs, changing initial derivatives,
    scalar biases and scalar mixture weights preserves the PDE. For a
    particular inhomogeneous pack, changing initial derivatives or bias also
    changes its forcing; arbitrary output mixing is not invariant.

    Dense matrix exponentials need quadratic state storage per input point.
    Exact rational algebra does not certify floating-point evaluation:
    large arguments, poorly conditioned companions and exponential growth
    can overflow or lose accuracy. JAX's default 16-squaring limit can return
    NaNs for large matrix norms. This does not solve boundary conditions.
    Initial derivatives, bias and mixture weights must be spatially constant;
    computing them from ``x`` generally invalidates the PDE identity.
    """
    if not isinstance(pack, ProfilePack):
        raise TypeError("pack must be a ProfilePack")
    if x.ndim < 1 or x.shape[-1] != len(pack.direction):
        raise ValueError("x must have shape (..., len(pack.direction))")
    if x.dtype not in (jnp.float32, jnp.float64):
        raise TypeError("x must have float32 or float64 dtype")
    initial = jnp.asarray(initial_jet, dtype=x.dtype)
    if initial.shape != (pack.profile_order,):
        raise ValueError("initial_jet must have shape (pack.profile_order,)")
    scalar_bias = jnp.asarray(bias, dtype=x.dtype)
    if scalar_bias.ndim != 0:
        raise ValueError("bias must be scalar to preserve the profile direction")
    direction = jnp.asarray([float(value) for value in pack.direction], dtype=x.dtype)
    carrier = jnp.asarray([float(value) for value in pack.carrier], dtype=x.dtype)
    coefficients = jnp.asarray([[float(value) for value in row] for row in pack.coefficients], dtype=x.dtype)
    order = pack.profile_order
    companion = jnp.asarray(
        [[float(j == i + 1) for j in range(order)] for i in range(order - 1)]
        + [[-float(value) for value in pack.annihilator[:-1]]], dtype=x.dtype
    )
    z = x @ direction + scalar_bias
    # Explicit vmap also supports JAX versions whose expm accepts one matrix.
    matrices = z.reshape((-1, 1, 1)) * companion
    state = (jax.vmap(expm)(matrices) @ initial).reshape((*z.shape, order))
    result: Array = jnp.exp(x @ carrier)[..., None] * (state @ coefficients)
    return result


__all__ = ["profile_solution"]
