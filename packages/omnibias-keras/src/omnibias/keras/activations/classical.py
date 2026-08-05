# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Classical activations: exp, relu, silu, gelu.

Mirrors :mod:`omnibias.torch.activations.classical` on ``keras.ops``.
``exp`` / ``silu`` / ``gelu`` have exact all-orders fast paths (``silu`` /
``gelu`` via Leibniz on the analytic ``z*f(z)`` product, reusing the sigmoid /
Gaussian-Hermite towers); ``relu`` is closed form to all orders on the
almost-everywhere / regular-part convention (``n >= 2 -> 0``; singular delta at
the kink dropped).
"""

from __future__ import annotations

import math
from typing import Any

from omnibias.keras.activations.registry import ActivationSpec, register_activation
from omnibias.keras.fastpath.eulerian import sigmoid_nth_derivative
from omnibias.keras.fastpath.hermite import gaussian_nth_derivative

from keras import ops

# --- exp ------------------------------------------------------------------


def _exp_forward(z: Any) -> Any:
    return ops.exp(z)


def _exp_fastpath(z: Any, n: int) -> Any:
    if n < 0:
        raise ValueError(f"order n must be >= 0, got {n}.")
    return ops.exp(z)


EXP = register_activation(
    ActivationSpec(
        name="exp",
        forward=_exp_forward,
        derivative=_exp_forward,
        fastpath=_exp_fastpath,
        integral=_exp_forward,
        riccati_polynomial=(0.0, 1.0),
        noise_model="poisson",
        operator_role=(
            "K=2 collapse -> exp(z) (eigenfunction of d/dz); "
            "log-link Newton step for Poisson regression."
        ),
    )
)


# --- relu -----------------------------------------------------------------


def _relu_forward(z: Any) -> Any:
    return ops.relu(z)


def _relu_derivative(z: Any) -> Any:
    return ops.cast(z > 0, dtype=z.dtype)


def _relu_integral(z: Any) -> Any:
    r = ops.relu(z)
    return 0.5 * r * r


def _relu_fastpath(z: Any, n: int) -> Any:
    """Closed-form ``relu^(n)`` on the almost-everywhere / regular-part convention.

    ``n = 0`` value, ``n = 1`` Heaviside (``H(0) = 0``), ``n >= 2`` zero away
    from the kink (singular delta dropped). Smooth twin: ``soft_relu``.
    """
    if n < 0:
        raise ValueError(f"order n must be >= 0, got {n}.")
    if n == 0:
        return _relu_forward(z)
    if n == 1:
        return _relu_derivative(z)
    return ops.zeros_like(z)


RELU = register_activation(
    ActivationSpec(
        name="relu",
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


# --- silu (swish) ---------------------------------------------------------


def _silu_forward(z: Any) -> Any:
    return ops.silu(z)


def _silu_derivative(z: Any) -> Any:
    s = ops.sigmoid(z)
    return s + z * s * (1.0 - s)


def _silu_fastpath(z: Any, n: int) -> Any:
    """Exact closed-form ``silu^(n)`` (all orders) via Leibniz on ``z * sigmoid(z)``."""
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


def _normal_cdf(z: Any) -> Any:
    return 0.5 * (1.0 + ops.erf(z * _INV_SQRT_2))


def _normal_pdf(z: Any) -> Any:
    return _INV_SQRT_2PI * ops.exp(-0.5 * z * z)


def _gelu_forward(z: Any) -> Any:
    return z * _normal_cdf(z)


def _gelu_derivative(z: Any) -> Any:
    return _normal_cdf(z) + z * _normal_pdf(z)


def _gelu_integral(z: Any) -> Any:
    cdf = _normal_cdf(z)
    pdf = _normal_pdf(z)
    return 0.5 * ((z * z - 1.0) * cdf + z * pdf)


def _normal_cdf_nth(z: Any, k: int) -> Any:
    """``Phi^(k)(z)``: the CDF for ``k = 0``, else ``phi^(k-1)`` via the Hermite tower."""
    if k == 0:
        return _normal_cdf(z)
    return gaussian_nth_derivative(z, k - 1) * _INV_SQRT_2PI


def _gelu_fastpath(z: Any, n: int) -> Any:
    """Exact closed-form ``gelu^(n)`` (all orders) via Leibniz on ``z * Phi(z)``."""
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
