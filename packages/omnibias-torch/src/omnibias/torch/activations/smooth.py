# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Smooth Riccati-class activations: sigmoid, tanh, softplus, gaussian.

Each one has a closed-form derivative tower, so its :class:`ActivationSpec`
carries a fast-path kernel that evaluates ``sigma^(n)(z)`` with one base
activation call regardless of order.
"""

from __future__ import annotations

import math

from omnibias.torch.activations.registry import ActivationSpec, register_activation
from omnibias.torch.fastpath.eulerian import (
    sigmoid_nth_derivative,
    softplus_nth_derivative,
)
from omnibias.torch.fastpath.hermite import gaussian_forward, gaussian_nth_derivative
from omnibias.torch.fastpath.legendre import tanh_nth_derivative
from omnibias.torch.transforms import GAUSSIAN_TRANSFORMS, SIGMOID_TRANSFORMS, TANH_TRANSFORMS

import torch
import torch.nn.functional as F
from torch import Tensor

# --- sigmoid ---------------------------------------------------------------


def _sigmoid(z: Tensor) -> Tensor:
    return torch.sigmoid(z)


def _sigmoid_derivative(z: Tensor) -> Tensor:
    s = torch.sigmoid(z)
    return s * (1.0 - s)


def _sigmoid_integral(z: Tensor) -> Tensor:
    return F.softplus(z)


SIGMOID = register_activation(
    ActivationSpec(
        name="sigmoid",
        transforms=SIGMOID_TRANSFORMS,
        forward=_sigmoid,
        derivative=_sigmoid_derivative,
        fastpath=sigmoid_nth_derivative,
        integral=_sigmoid_integral,
        riccati_polynomial=(0.0, 1.0, -1.0),  # P(s) = s - s^2
        noise_model="bernoulli",
        operator_role=(
            "K=2 collapse -> Bernoulli variance s(1-s); Newton/IRLS step for logistic regression."
        ),
        aliases=("logistic",),
        limit_pos_inf=1.0,
        limit_neg_inf=0.0,
    )
)


# --- tanh ------------------------------------------------------------------


def _tanh(z: Tensor) -> Tensor:
    return torch.tanh(z)


def _tanh_derivative(z: Tensor) -> Tensor:
    t = torch.tanh(z)
    return 1.0 - t * t


def _tanh_integral(z: Tensor) -> Tensor:
    log_two = z.new_tensor(math.log(2.0))
    return z + F.softplus(-2.0 * z) - log_two


TANH = register_activation(
    ActivationSpec(
        name="tanh",
        transforms=TANH_TRANSFORMS,
        forward=_tanh,
        derivative=_tanh_derivative,
        fastpath=tanh_nth_derivative,
        integral=_tanh_integral,
        riccati_polynomial=(1.0, 0.0, -1.0),  # P(t) = 1 - t^2
        noise_model="symmetric_bernoulli",
        operator_role=(
            "K=2 collapse -> 1 - tanh^2; symmetric IRLS bell, "
            "useful as a saturation-aware feature gate."
        ),
        limit_pos_inf=1.0,
        limit_neg_inf=-1.0,
    )
)


# --- softplus --------------------------------------------------------------


def _softplus(z: Tensor) -> Tensor:
    return F.softplus(z)


def _softplus_derivative(z: Tensor) -> Tensor:
    return torch.sigmoid(z)


SOFTPLUS = register_activation(
    ActivationSpec(
        name="softplus",
        forward=_softplus,
        derivative=_softplus_derivative,
        fastpath=softplus_nth_derivative,
        riccati_polynomial=None,  # softplus itself is not in Riccati form,
        # but its derivative tower from order >= 1 is (sigmoid's tower).
        noise_model="bernoulli",
        operator_role=(
            "K=2 collapse -> sigmoid (Bernoulli mean); "
            "log-link Newton step / canonical PINN base for diffusion-class PDEs."
        ),
        limit_neg_inf=0.0,  # softplus -> 0 as z -> -inf; diverges as z -> +inf
    )
)


# --- gaussian --------------------------------------------------------------


def _gaussian(z: Tensor) -> Tensor:
    return gaussian_forward(z)


def _gaussian_derivative(z: Tensor) -> Tensor:
    return -z * gaussian_forward(z)


def _gaussian_integral(z: Tensor) -> Tensor:
    return math.sqrt(math.pi / 2.0) * torch.erf(z / math.sqrt(2.0))


GAUSSIAN = register_activation(
    ActivationSpec(
        name="gaussian",
        transforms=GAUSSIAN_TRANSFORMS,
        forward=_gaussian,
        derivative=_gaussian_derivative,
        fastpath=gaussian_nth_derivative,
        integral=_gaussian_integral,
        riccati_polynomial=None,  # not Riccati in g itself; uses Hermite tower instead.
        noise_model="gaussian_kernel",
        operator_role=(
            "K=2 collapse -> -z * exp(-z^2/2); RBF/Hermite spectral basis, "
            "Laplacian-of-Gaussian and Difference-of-Gaussian conv kernels."
        ),
        aliases=("rbf",),
        limit_pos_inf=0.0,
        limit_neg_inf=0.0,
    )
)


__all__ = ["GAUSSIAN", "SIGMOID", "SOFTPLUS", "TANH"]
