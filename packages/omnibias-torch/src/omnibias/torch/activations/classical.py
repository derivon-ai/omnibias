# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Classical activations: relu, gelu, silu, exp.

All four now carry closed-form all-orders fast paths:

* ``exp`` -- every derivative equals ``exp(z)``.
* ``silu = z * sigmoid(z)`` and ``gelu = z * Phi(z)`` -- **exact** all orders
  via Leibniz on the analytic ``z * f(z)`` product, reusing the sigmoid /
  Gaussian-Hermite towers.
* ``relu`` -- closed-form on the **almost-everywhere / regular-part**
  convention: ``n = 0`` value, ``n = 1`` Heaviside, ``n >= 2`` zero away from
  the kink (the singular delta at ``z = 0`` is dropped). Its smooth
  beta-tempered twin ``soft_relu`` lives in
  :mod:`omnibias.torch.activations.tempered`.
"""

from __future__ import annotations

import math

from omnibias.torch.activations.registry import ActivationSpec, register_activation
from omnibias.torch.fastpath.eulerian import sigmoid_nth_derivative
from omnibias.torch.fastpath.hermite import gaussian_nth_derivative
from omnibias.torch.transforms import EXP_TRANSFORMS, RELU_TRANSFORMS

import torch
import torch.nn.functional as F
from torch import Tensor

# --- exp ------------------------------------------------------------------


def _exp_forward(z: Tensor) -> Tensor:
    return torch.exp(z)


def _exp_derivative(z: Tensor) -> Tensor:
    return torch.exp(z)


def _exp_fastpath(z: Tensor, n: int) -> Tensor:
    """``exp^(n)(z) = exp(z)`` for every order ``n``."""
    if n < 0:
        raise ValueError(f"order n must be >= 0, got {n}.")
    return torch.exp(z)


EXP = register_activation(
    ActivationSpec(
        name="exp",
        transforms=EXP_TRANSFORMS,
        forward=_exp_forward,
        derivative=_exp_derivative,
        fastpath=_exp_fastpath,
        integral=_exp_forward,
        riccati_polynomial=(0.0, 1.0),  # dE/dz = E, so P(E) = E
        noise_model="poisson",
        operator_role=(
            "K=2 collapse -> exp(z) (eigenfunction of d/dz); "
            "log-link Newton step for Poisson regression."
        ),
        limit_neg_inf=0.0,  # exp -> 0 as z -> -inf; diverges as z -> +inf
    )
)


# --- relu -----------------------------------------------------------------


def _relu_forward(z: Tensor) -> Tensor:
    return F.relu(z)


def _relu_derivative(z: Tensor) -> Tensor:
    """Heaviside step at zero; we adopt the standard PyTorch convention
    ``H(0) = 0``."""
    return (z > 0).to(z.dtype)


def _relu_integral(z: Tensor) -> Tensor:
    r = F.relu(z)
    return 0.5 * r * r


def _relu_fastpath(z: Tensor, n: int) -> Tensor:
    """Closed-form ``relu^(n)`` on the almost-everywhere / regular-part convention.

    ``n = 0`` is ``max(0, z)``, ``n = 1`` is the Heaviside step (``H(0) = 0``),
    and every higher order is ``0`` away from the kink. The distributional delta
    (and its derivatives) at ``z = 0`` -- a measure-zero singular set -- is
    dropped; for the smooth surrogate whose bump converges to that delta as
    ``beta -> inf`` use ``soft_relu`` from :mod:`omnibias.torch.activations.tempered`.
    """
    if n < 0:
        raise ValueError(f"order n must be >= 0, got {n}.")
    if n == 0:
        return _relu_forward(z)
    if n == 1:
        return _relu_derivative(z)
    return torch.zeros_like(z)


RELU = register_activation(
    ActivationSpec(
        name="relu",
        transforms=RELU_TRANSFORMS,
        forward=_relu_forward,
        derivative=_relu_derivative,
        fastpath=_relu_fastpath,
        integral=_relu_integral,
        riccati_polynomial=None,
        noise_model="none",
        operator_role=(
            "K=2 collapse -> Heaviside step; equality-constraint indicator, "
            "Moore-Penrose pseudoinverse limit."
        ),
    )
)


# --- silu (a.k.a. swish) --------------------------------------------------


def _silu_forward(z: Tensor) -> Tensor:
    return F.silu(z)


def _silu_derivative(z: Tensor) -> Tensor:
    s = torch.sigmoid(z)
    return s + z * s * (1.0 - s)


def _silu_fastpath(z: Tensor, n: int) -> Tensor:
    """Exact closed-form ``silu^(n)`` (all orders).

    ``silu(z) = z * sigmoid(z)`` is analytic, so Leibniz on the product with the
    all-orders sigmoid tower gives ``silu^(n)(z) = z * sigma^(n)(z) + n *
    sigma^(n-1)(z)`` (only the ``k in {0, 1}`` Leibniz terms survive because
    ``(z)'' = 0``). No truncation, no delta.
    """
    if n < 0:
        raise ValueError(f"order n must be >= 0, got {n}.")
    if n == 0:
        return _silu_forward(z)
    return z * sigmoid_nth_derivative(z, n) + n * sigmoid_nth_derivative(z, n - 1)


SILU = register_activation(
    ActivationSpec(
        name="silu",
        forward=_silu_forward,
        derivative=_silu_derivative,
        fastpath=_silu_fastpath,
        riccati_polynomial=None,
        noise_model="none",
        operator_role=(
            "K=2 collapse -> sigmoid + z * sigmoid * (1 - sigmoid); "
            "smoothed gate for transformer FFN compatibility."
        ),
        aliases=("swish",),
    )
)


# --- gelu (exact, Phi-based) ----------------------------------------------


_INV_SQRT_2 = 1.0 / math.sqrt(2.0)
_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


def _normal_cdf(z: Tensor) -> Tensor:
    return 0.5 * (1.0 + torch.erf(z * _INV_SQRT_2))


def _normal_pdf(z: Tensor) -> Tensor:
    return _INV_SQRT_2PI * torch.exp(-0.5 * z * z)


def _gelu_forward(z: Tensor) -> Tensor:
    return z * _normal_cdf(z)


def _gelu_derivative(z: Tensor) -> Tensor:
    return _normal_cdf(z) + z * _normal_pdf(z)


def _gelu_integral(z: Tensor) -> Tensor:
    cdf = _normal_cdf(z)
    pdf = _normal_pdf(z)
    return 0.5 * ((z * z - 1.0) * cdf + z * pdf)


def _normal_cdf_nth(z: Tensor, k: int) -> Tensor:
    """``Phi^(k)(z)``: the CDF for ``k = 0``, else ``phi^(k-1)(z)`` via the
    Gaussian/Hermite tower (``phi = gaussian / sqrt(2 pi)``)."""
    if k == 0:
        return _normal_cdf(z)
    return gaussian_nth_derivative(z, k - 1) * _INV_SQRT_2PI


def _gelu_fastpath(z: Tensor, n: int) -> Tensor:
    """Exact closed-form ``gelu^(n)`` (all orders).

    ``gelu(z) = z * Phi(z)`` is analytic; Leibniz gives ``gelu^(n)(z) = z *
    Phi^(n)(z) + n * Phi^(n-1)(z)`` with ``Phi^(k) = phi^(k-1)`` (``k >= 1``)
    from the closed-form Gaussian/Hermite tower. No truncation, no delta.
    """
    if n < 0:
        raise ValueError(f"order n must be >= 0, got {n}.")
    if n == 0:
        return _gelu_forward(z)
    return z * _normal_cdf_nth(z, n) + n * _normal_cdf_nth(z, n - 1)


GELU = register_activation(
    ActivationSpec(
        name="gelu",
        forward=_gelu_forward,
        derivative=_gelu_derivative,
        fastpath=_gelu_fastpath,
        integral=_gelu_integral,
        riccati_polynomial=None,
        noise_model="none",
        operator_role=(
            "K=2 collapse -> Phi(z) + z * phi(z); "
            "smoothed half-space gate, transformer FFN compatibility."
        ),
    )
)


__all__ = ["EXP", "GELU", "RELU", "SILU"]
