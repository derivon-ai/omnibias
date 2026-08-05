# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""JAX twins of the beta-tempered smooth-surrogate activation family.

Bit-identical mirror of :mod:`omnibias.torch.activations.tempered`. Built on the
backend-neutral :func:`omnibias.core.spec.tempered` combinator, so every order
matches the torch tower by construction. Imported for side-effect registration
by :mod:`omnibias.jax.activations`. ``beta`` may be a float or a traced
``Array`` (learnable temperature); no host-side branching on ``beta``.
"""

from __future__ import annotations

from omnibias.core.spec import tempered
from omnibias.jax import _fastpath
from omnibias.jax.activations import (
    SIGMOID,
    SOFTPLUS,
    JaxActivationSpec,
    get_activation,
    register_activation,
)

import jax.numpy as jnp
from jax import Array

_DEFAULT_BETA = 1.0


def tempered_activation(
    base: str | JaxActivationSpec,
    beta: float | Array = _DEFAULT_BETA,
    *,
    scale: str = "one_over_beta",
    name: str | None = None,
) -> JaxActivationSpec:
    """Functional learnable-temperature surrogate (JAX twin of the module blocks).

    Resolve ``base`` (registry name or spec) and return a tempered
    :class:`JaxActivationSpec` whose whole closed-form tower is scaled by
    ``beta`` -- which may be a traced ``Array`` (e.g. a learnable temperature
    threaded through ``jax.grad``). Mirrors the torch/keras
    ``TemperedActivation`` layer without holding state.
    """
    spec = get_activation(base)
    return tempered(
        spec,
        beta,
        scale=scale,
        name=name if name is not None else f"tempered_{spec.name}",
    )


def make_soft_relu_spec(
    beta: float = _DEFAULT_BETA, *, name: str = "soft_relu"
) -> JaxActivationSpec:
    """Smooth ReLU surrogate ``softplus(beta z) / beta`` (all-orders tower)."""
    return tempered(
        SOFTPLUS,
        beta,
        scale="one_over_beta",
        name=name,
        operator_role=(
            "Smooth ReLU: softplus(beta z)/beta -> relu as beta -> inf; "
            "the n=2 bump -> delta."
        ),
        limit_neg_inf=0.0,
    )


SOFT_RELU = register_activation(make_soft_relu_spec())


def make_soft_step_spec(
    beta: float = _DEFAULT_BETA, *, name: str = "soft_step"
) -> JaxActivationSpec:
    """Smooth step surrogate ``sigmoid(beta z)`` (all-orders tower)."""
    return tempered(
        SIGMOID,
        beta,
        scale="unit",
        name=name,
        aliases=("soft_heaviside",) if name == "soft_step" else (),
        operator_role="Smooth Heaviside: sigmoid(beta z) -> step as beta -> inf.",
        limit_pos_inf=1.0,
        limit_neg_inf=0.0,
    )


SOFT_STEP = register_activation(make_soft_step_spec())


def make_swish_spec(beta: float = _DEFAULT_BETA, *, name: str = "swish") -> JaxActivationSpec:
    """Swish ``z * sigmoid(beta z)`` with tunable ``beta`` (``silu`` is beta=1)."""

    def fwd(z: Array) -> Array:
        return z * _fastpath.jax_sigmoid(beta * z)

    def fp(z: Array, n: int) -> Array:
        if n < 0:
            raise ValueError(f"order n must be >= 0, got {n}.")
        if n == 0:
            return fwd(z)
        bz = beta * z
        term_a = z * (beta**n) * _fastpath.sigmoid_nth_derivative(bz, n)
        term_b = n * (beta ** (n - 1)) * _fastpath.sigmoid_nth_derivative(bz, n - 1)
        return term_a + term_b

    def deriv(z: Array) -> Array:
        return fp(z, 1)

    return JaxActivationSpec(
        name=name,
        forward=fwd,
        derivative=deriv,
        fastpath=fp,
        riccati_polynomial=None,
        noise_model="none",
        operator_role="Swish: z * sigmoid(beta z); exact all-orders (silu is beta=1).",
    )


def make_soft_leaky_relu_spec(
    negative_slope: float = 0.01,
    beta: float = _DEFAULT_BETA,
    *,
    name: str = "soft_leaky_relu",
) -> JaxActivationSpec:
    """Smooth leaky ReLU ``alpha*z + (1-alpha)*soft_relu(z; beta)`` (all orders)."""
    alpha = negative_slope
    one_minus_alpha = 1.0 - alpha

    def _soft_relu_nth(z: Array, n: int) -> Array:
        return (beta ** (n - 1)) * _fastpath.softplus_nth_derivative(beta * z, n)

    def fwd(z: Array) -> Array:
        return alpha * z + one_minus_alpha * (
            jnp.logaddexp(jnp.zeros_like(z), beta * z) / beta
        )

    def fp(z: Array, n: int) -> Array:
        if n < 0:
            raise ValueError(f"order n must be >= 0, got {n}.")
        if n == 0:
            return fwd(z)
        soft = one_minus_alpha * _soft_relu_nth(z, n)
        if n == 1:
            return jnp.full_like(z, alpha) + soft
        return soft

    def deriv(z: Array) -> Array:
        return fp(z, 1)

    return JaxActivationSpec(
        name=name,
        forward=fwd,
        derivative=deriv,
        fastpath=fp,
        riccati_polynomial=None,
        noise_model="none",
        operator_role=(
            "Smooth leaky ReLU: alpha*z + (1-alpha)*softplus(beta z)/beta "
            "-> leaky_relu as beta -> inf."
        ),
    )


__all__ = [
    "SOFT_RELU",
    "SOFT_STEP",
    "make_soft_leaky_relu_spec",
    "make_soft_relu_spec",
    "make_soft_step_spec",
    "make_swish_spec",
    "tempered_activation",
]
