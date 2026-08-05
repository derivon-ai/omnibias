# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Smooth Riccati-class activations: sigmoid, tanh, softplus, gaussian.

Each has a closed-form derivative tower, so its :class:`ActivationSpec`
carries a fast-path kernel that evaluates ``sigma^(n)(z)`` with one base
activation call regardless of order. Mirrors
:mod:`omnibias.torch.activations.smooth` on ``keras.ops``.
"""

from __future__ import annotations

import math
from typing import Any

from omnibias.keras.activations.registry import ActivationSpec, register_activation
from omnibias.keras.fastpath.eulerian import (
    sigmoid_nth_derivative,
    softplus_nth_derivative,
)
from omnibias.keras.fastpath.hermite import gaussian_forward, gaussian_nth_derivative
from omnibias.keras.fastpath.legendre import tanh_nth_derivative

from keras import ops

_LOG_TWO = math.log(2.0)


# --- sigmoid ---------------------------------------------------------------


def _sigmoid(z: Any) -> Any:
    return ops.sigmoid(z)


def _sigmoid_derivative(z: Any) -> Any:
    s = ops.sigmoid(z)
    return s * (1.0 - s)


def _sigmoid_integral(z: Any) -> Any:
    return ops.softplus(z)


SIGMOID = register_activation(
    ActivationSpec(
        name="sigmoid",
        forward=_sigmoid,
        derivative=_sigmoid_derivative,
        fastpath=sigmoid_nth_derivative,
        integral=_sigmoid_integral,
        riccati_polynomial=(0.0, 1.0, -1.0),
        noise_model="bernoulli",
        operator_role=(
            "K=2 collapse -> Bernoulli variance s(1-s); Newton/IRLS step for logistic regression."
        ),
        aliases=("logistic",),
    )
)


# --- tanh ------------------------------------------------------------------


def _tanh(z: Any) -> Any:
    return ops.tanh(z)


def _tanh_derivative(z: Any) -> Any:
    t = ops.tanh(z)
    return 1.0 - t * t


def _tanh_integral(z: Any) -> Any:
    return z + ops.softplus(-2.0 * z) - _LOG_TWO


TANH = register_activation(
    ActivationSpec(
        name="tanh",
        forward=_tanh,
        derivative=_tanh_derivative,
        fastpath=tanh_nth_derivative,
        integral=_tanh_integral,
        riccati_polynomial=(1.0, 0.0, -1.0),
        noise_model="symmetric_bernoulli",
        operator_role=(
            "K=2 collapse -> 1 - tanh^2; symmetric IRLS bell, "
            "useful as a saturation-aware feature gate."
        ),
    )
)


# --- softplus --------------------------------------------------------------


def _softplus(z: Any) -> Any:
    return ops.softplus(z)


def _softplus_derivative(z: Any) -> Any:
    return ops.sigmoid(z)


SOFTPLUS = register_activation(
    ActivationSpec(
        name="softplus",
        forward=_softplus,
        derivative=_softplus_derivative,
        fastpath=softplus_nth_derivative,
        riccati_polynomial=None,
        noise_model="bernoulli",
        operator_role=(
            "K=2 collapse -> sigmoid (Bernoulli mean); "
            "log-link Newton step / canonical PINN base for diffusion-class PDEs."
        ),
    )
)


# --- gaussian --------------------------------------------------------------


def _gaussian(z: Any) -> Any:
    return gaussian_forward(z)


def _gaussian_derivative(z: Any) -> Any:
    return -z * gaussian_forward(z)


def _gaussian_integral(z: Any) -> Any:
    return math.sqrt(math.pi / 2.0) * ops.erf(z / math.sqrt(2.0))


GAUSSIAN = register_activation(
    ActivationSpec(
        name="gaussian",
        forward=_gaussian,
        derivative=_gaussian_derivative,
        fastpath=gaussian_nth_derivative,
        integral=_gaussian_integral,
        riccati_polynomial=None,
        noise_model="gaussian_kernel",
        operator_role=(
            "K=2 collapse -> -z * exp(-z^2/2); RBF/Hermite spectral basis, "
            "Laplacian-of-Gaussian and Difference-of-Gaussian conv kernels."
        ),
        aliases=("rbf",),
    )
)


__all__ = ["GAUSSIAN", "SIGMOID", "SOFTPLUS", "TANH"]
