# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Beta-tempered smooth surrogates of the hard non-smooth activations.

Keras (``keras.ops``) twin of :mod:`omnibias.torch.activations.tempered`. Built
on the backend-neutral :func:`omnibias.core.spec.tempered` combinator, so every
order matches the torch / jax towers by construction. ``beta -> inf`` recovers
the hard activation (``softplus -> relu``, ``sigmoid -> Heaviside``,
``tanh -> sign``); ``beta`` may be a float or a tensor for a learnable
temperature.
"""

from __future__ import annotations

from typing import Any

from omnibias.core.spec import tempered
from omnibias.keras.activations.registry import ActivationSpec, register_activation
from omnibias.keras.activations.smooth import SIGMOID, SOFTPLUS
from omnibias.keras.fastpath.eulerian import (
    sigmoid_nth_derivative,
    softplus_nth_derivative,
)

from keras import ops

_DEFAULT_BETA = 1.0


def make_soft_relu_spec(
    beta: float = _DEFAULT_BETA, *, name: str = "soft_relu"
) -> ActivationSpec[Any]:
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
) -> ActivationSpec[Any]:
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


def make_swish_spec(beta: float = _DEFAULT_BETA, *, name: str = "swish") -> ActivationSpec[Any]:
    """Swish ``z * sigmoid(beta z)`` with tunable ``beta`` (``silu`` is beta=1)."""

    def fwd(z: Any) -> Any:
        return z * ops.sigmoid(beta * z)

    def fp(z: Any, n: int) -> Any:
        if n < 0:
            raise ValueError(f"order n must be >= 0, got {n}.")
        if n == 0:
            return fwd(z)
        bz = beta * z
        term_a = z * (beta**n) * sigmoid_nth_derivative(bz, n)
        term_b = n * (beta ** (n - 1)) * sigmoid_nth_derivative(bz, n - 1)
        return term_a + term_b

    def deriv(z: Any) -> Any:
        return fp(z, 1)

    return ActivationSpec(
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
) -> ActivationSpec[Any]:
    """Smooth leaky ReLU ``alpha*z + (1-alpha)*soft_relu(z; beta)`` (all orders)."""
    alpha = negative_slope
    one_minus_alpha = 1.0 - alpha

    def _soft_relu_nth(z: Any, n: int) -> Any:
        return (beta ** (n - 1)) * softplus_nth_derivative(beta * z, n)

    def fwd(z: Any) -> Any:
        return alpha * z + one_minus_alpha * (ops.softplus(beta * z) / beta)

    def fp(z: Any, n: int) -> Any:
        if n < 0:
            raise ValueError(f"order n must be >= 0, got {n}.")
        if n == 0:
            return fwd(z)
        soft = one_minus_alpha * _soft_relu_nth(z, n)
        if n == 1:
            return alpha * ops.ones_like(z) + soft
        return soft

    def deriv(z: Any) -> Any:
        return fp(z, 1)

    return ActivationSpec(
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
]
