# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Beta-tempered smooth surrogates of the hard non-smooth activations.

Where :mod:`omnibias.torch.activations.piecewise` gives the *hard* activations
their almost-everywhere tower (dropping the singular part), this module gives
the complementary *smooth* family: differentiable surrogates with a temperature
``beta`` that sharpen to the hard activation as ``beta -> inf``, and whose
higher-order bumps *become* the dropped Dirac deltas in that limit.

Each surrogate reuses an existing closed-form tower through the backend-neutral
:func:`omnibias.core.spec.tempered` combinator (or a small hand-written Leibniz
product), so every order is exact and bit-identical across backends:

============  ==================================  =========================
Surrogate     definition                          hard limit (beta -> inf)
============  ==================================  =========================
``soft_relu`` ``softplus(beta z) / beta``         ``relu``
``soft_step`` ``sigmoid(beta z)``                 ``Heaviside``
``soft_sign`` ``tanh(beta z)`` (= ``smooth_sign``) ``sign``
``soft_abs``  ``sqrt(z^2 + eps^2) - eps`` (softabs) ``abs``
============  ==================================  =========================

``beta`` may be a Python float **or** a tensor / ``nn.Parameter`` for a
learnable, differentiable temperature (see
:mod:`omnibias.torch.tempered_blocks`).
"""

from __future__ import annotations

from omnibias.core.spec import tempered
from omnibias.torch.activations.registry import ActivationSpec, register_activation
from omnibias.torch.activations.smooth import SIGMOID, SOFTPLUS
from omnibias.torch.fastpath.eulerian import (
    sigmoid_nth_derivative,
    softplus_nth_derivative,
)

import torch
import torch.nn.functional as F
from torch import Tensor

_DEFAULT_BETA = 1.0


# --- soft_relu = softplus(beta z) / beta  ->  relu -------------------------


def make_soft_relu_spec(
    beta: float | Tensor = _DEFAULT_BETA, *, name: str = "soft_relu"
) -> ActivationSpec[Tensor]:
    """Smooth ReLU surrogate ``softplus(beta z) / beta`` (all-orders tower).

    ``soft_relu^(n)(z) = beta**(n-1) * softplus^(n)(beta z)``; ``n = 2`` is the
    beta-scaled logistic bump that concentrates into ``relu``'s Dirac delta as
    ``beta -> inf``.
    """
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


# --- soft_step = sigmoid(beta z)  ->  Heaviside ----------------------------


def make_soft_step_spec(
    beta: float | Tensor = _DEFAULT_BETA, *, name: str = "soft_step"
) -> ActivationSpec[Tensor]:
    """Smooth step surrogate ``sigmoid(beta z)`` (all-orders tower).

    ``soft_step^(n)(z) = beta**n * sigma^(n)(beta z) -> Heaviside`` as
    ``beta -> inf``.
    """
    return tempered(
        SIGMOID,
        beta,
        scale="unit",
        name=name,
        aliases=("soft_heaviside",) if name == "soft_step" else (),
        operator_role=(
            "Smooth Heaviside: sigmoid(beta z) -> step as beta -> inf."
        ),
        limit_pos_inf=1.0,
        limit_neg_inf=0.0,
    )


SOFT_STEP = register_activation(make_soft_step_spec())


# --- factories reusing towers (not auto-registered) ------------------------


def make_swish_spec(
    beta: float | Tensor = _DEFAULT_BETA, *, name: str = "swish"
) -> ActivationSpec[Tensor]:
    """Swish ``z * sigmoid(beta z)`` with tunable ``beta`` (``silu`` is beta=1).

    Exact all-orders via Leibniz on ``z * sigmoid(beta z)``:
    ``swish^(n)(z) = z * beta**n * sigma^(n)(beta z) + n * beta**(n-1) *
    sigma^(n-1)(beta z)``.
    """

    def fwd(z: Tensor) -> Tensor:
        return z * torch.sigmoid(beta * z)

    def fp(z: Tensor, n: int) -> Tensor:
        if n < 0:
            raise ValueError(f"order n must be >= 0, got {n}.")
        if n == 0:
            return fwd(z)
        bz = beta * z
        term_a = z * (beta**n) * sigmoid_nth_derivative(bz, n)
        term_b = n * (beta ** (n - 1)) * sigmoid_nth_derivative(bz, n - 1)
        return term_a + term_b

    def deriv(z: Tensor) -> Tensor:
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
    beta: float | Tensor = _DEFAULT_BETA,
    *,
    name: str = "soft_leaky_relu",
) -> ActivationSpec[Tensor]:
    """Smooth leaky ReLU ``alpha*z + (1-alpha)*soft_relu(z; beta)`` (all orders).

    Recovers ``leaky_relu`` with the given ``negative_slope`` as ``beta -> inf``.
    """
    alpha = negative_slope
    one_minus_alpha = 1.0 - alpha

    def _soft_relu_nth(z: Tensor, n: int) -> Tensor:
        return (beta ** (n - 1)) * softplus_nth_derivative(beta * z, n)

    def fwd(z: Tensor) -> Tensor:
        return alpha * z + one_minus_alpha * (F.softplus(beta * z) / beta)

    def fp(z: Tensor, n: int) -> Tensor:
        if n < 0:
            raise ValueError(f"order n must be >= 0, got {n}.")
        if n == 0:
            return fwd(z)
        soft = one_minus_alpha * _soft_relu_nth(z, n)
        if n == 1:
            return torch.full_like(z, alpha) + soft
        return soft

    def deriv(z: Tensor) -> Tensor:
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
