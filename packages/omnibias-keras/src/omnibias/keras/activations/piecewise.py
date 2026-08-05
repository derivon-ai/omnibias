# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Piecewise activations with almost-everywhere (regular-part) derivative towers.

Keras (``keras.ops``) twin of :mod:`omnibias.torch.activations.piecewise`; see
that module for the derivation of every tower. Registered into the Keras
registry so all three backends expose the same names. Higher-order towers use
the a.e. / regular-part convention (singular delta dropped; ``H(0) = 0``,
``sign(0) = 0``). Smooth surrogates live in
:mod:`omnibias.keras.activations.tempered`.
"""

from __future__ import annotations

import math
from typing import Any

from omnibias.keras.activations.registry import ActivationSpec, register_activation

from keras import ops


def _ae_linear_fastpath(z: Any, n: int, forward: Any, derivative: Any) -> Any:
    """``n = 0`` forward, ``n = 1`` derivative, ``n >= 2`` zero (a.e.)."""
    if n < 0:
        raise ValueError(f"order n must be >= 0, got {n}.")
    if n == 0:
        return forward(z)
    if n == 1:
        return derivative(z)
    return ops.zeros_like(z)


# --- leaky_relu / prelu ----------------------------------------------------


_DEFAULT_LEAKY_SLOPE = 0.01
_DEFAULT_PRELU_INIT = 0.25


def _leaky_relu_forward(z: Any, slope: float) -> Any:
    return ops.where(z > 0, z, slope * z)


def _leaky_relu_derivative(z: Any, slope: float) -> Any:
    return ops.where(z > 0, ops.ones_like(z), slope * ops.ones_like(z))


def _leaky_relu_integral(z: Any, slope: float) -> Any:
    return ops.where(z > 0, 0.5 * z * z, 0.5 * slope * z * z)


def make_leaky_relu_spec(
    slope: float = _DEFAULT_LEAKY_SLOPE, *, name: str = "leaky_relu"
) -> ActivationSpec[Any]:
    """Leaky-ReLU spec with fixed negative ``slope`` (a.e. all-orders tower)."""

    def fwd(z: Any) -> Any:
        return _leaky_relu_forward(z, slope)

    def deriv(z: Any) -> Any:
        return _leaky_relu_derivative(z, slope)

    def integ(z: Any) -> Any:
        return _leaky_relu_integral(z, slope)

    def fp(z: Any, n: int) -> Any:
        return _ae_linear_fastpath(z, n, fwd, deriv)

    return ActivationSpec(
        name=name,
        forward=fwd,
        derivative=deriv,
        fastpath=fp,
        integral=integ,
        riccati_polynomial=None,
        noise_model="none",
        operator_role=(
            "Leaky ReLU: z for z>0, slope*z otherwise; a.e. regular-part tower "
            "(n>=2 -> 0)."
        ),
    )


LEAKY_RELU = register_activation(make_leaky_relu_spec(_DEFAULT_LEAKY_SLOPE))
PRELU = register_activation(make_leaky_relu_spec(_DEFAULT_PRELU_INIT, name="prelu"))


# --- relu6 -----------------------------------------------------------------


def _relu6_forward(z: Any) -> Any:
    return ops.clip(z, 0.0, 6.0)


def _relu6_derivative(z: Any) -> Any:
    return ops.cast(ops.logical_and(z > 0, z < 6.0), dtype=z.dtype)


def _relu6_fastpath(z: Any, n: int) -> Any:
    return _ae_linear_fastpath(z, n, _relu6_forward, _relu6_derivative)


RELU6 = register_activation(
    ActivationSpec(
        name="relu6",
        forward=_relu6_forward,
        derivative=_relu6_derivative,
        fastpath=_relu6_fastpath,
        riccati_polynomial=None,
        noise_model="none",
        operator_role="ReLU6: clamp(z, 0, 6); a.e. regular-part tower (n>=2 -> 0).",
        limit_pos_inf=6.0,
        limit_neg_inf=0.0,
    )
)


# --- hardtanh --------------------------------------------------------------


_DEFAULT_HARDTANH_MIN = -1.0
_DEFAULT_HARDTANH_MAX = 1.0


def _hardtanh_forward(z: Any, lo: float, hi: float) -> Any:
    return ops.clip(z, lo, hi)


def _hardtanh_derivative(z: Any, lo: float, hi: float) -> Any:
    return ops.cast(ops.logical_and(z > lo, z < hi), dtype=z.dtype)


def make_hardtanh_spec(
    min_val: float = _DEFAULT_HARDTANH_MIN,
    max_val: float = _DEFAULT_HARDTANH_MAX,
    *,
    name: str = "hardtanh",
) -> ActivationSpec[Any]:
    """Hardtanh spec clamping to ``[min_val, max_val]`` (a.e. all-orders tower)."""

    def fwd(z: Any) -> Any:
        return _hardtanh_forward(z, min_val, max_val)

    def deriv(z: Any) -> Any:
        return _hardtanh_derivative(z, min_val, max_val)

    def fp(z: Any, n: int) -> Any:
        return _ae_linear_fastpath(z, n, fwd, deriv)

    return ActivationSpec(
        name=name,
        forward=fwd,
        derivative=deriv,
        fastpath=fp,
        riccati_polynomial=None,
        noise_model="none",
        operator_role="Hardtanh: clamp(z, min, max); a.e. regular-part tower (n>=2 -> 0).",
        limit_pos_inf=max_val,
        limit_neg_inf=min_val,
    )


HARDTANH = register_activation(make_hardtanh_spec())


# --- hardsigmoid -----------------------------------------------------------


def _hardsigmoid_forward(z: Any) -> Any:
    return ops.clip(z / 6.0 + 0.5, 0.0, 1.0)


def _hardsigmoid_derivative(z: Any) -> Any:
    return ops.where(
        ops.logical_and(z > -3.0, z < 3.0),
        (1.0 / 6.0) * ops.ones_like(z),
        ops.zeros_like(z),
    )


def _hardsigmoid_fastpath(z: Any, n: int) -> Any:
    return _ae_linear_fastpath(z, n, _hardsigmoid_forward, _hardsigmoid_derivative)


HARDSIGMOID = register_activation(
    ActivationSpec(
        name="hardsigmoid",
        forward=_hardsigmoid_forward,
        derivative=_hardsigmoid_derivative,
        fastpath=_hardsigmoid_fastpath,
        riccati_polynomial=None,
        noise_model="none",
        operator_role=(
            "Hard sigmoid: clamp(z/6 + 1/2, 0, 1); a.e. regular-part tower (n>=2 -> 0)."
        ),
        limit_pos_inf=1.0,
        limit_neg_inf=0.0,
    )
)


# --- hardswish -------------------------------------------------------------


def _hardswish_forward(z: Any) -> Any:
    return z * ops.clip(z / 6.0 + 0.5, 0.0, 1.0)


def _hardswish_derivative(z: Any) -> Any:
    mid = (2.0 * z + 3.0) / 6.0
    out = ops.where(z >= 3.0, ops.ones_like(z), mid)
    return ops.where(z <= -3.0, ops.zeros_like(z), out)


def _hardswish_fastpath(z: Any, n: int) -> Any:
    if n < 0:
        raise ValueError(f"order n must be >= 0, got {n}.")
    if n == 0:
        return _hardswish_forward(z)
    if n == 1:
        return _hardswish_derivative(z)
    if n == 2:
        return ops.where(
            ops.logical_and(z > -3.0, z < 3.0),
            (1.0 / 3.0) * ops.ones_like(z),
            ops.zeros_like(z),
        )
    return ops.zeros_like(z)


HARDSWISH = register_activation(
    ActivationSpec(
        name="hardswish",
        forward=_hardswish_forward,
        derivative=_hardswish_derivative,
        fastpath=_hardswish_fastpath,
        riccati_polynomial=None,
        noise_model="none",
        operator_role=(
            "Hard swish: z * hardsigmoid(z); a.e. regular-part tower "
            "(quadratic middle gives n=2 -> 1/3, n>=3 -> 0)."
        ),
        limit_neg_inf=0.0,
    )
)


# --- elu / selu / celu -----------------------------------------------------


_DEFAULT_ELU_ALPHA = 1.0
_SELU_ALPHA = 1.6732632423543772848170429916717
_SELU_SCALE = 1.0507009873554804934193349852946
_DEFAULT_CELU_ALPHA = 1.0


def _elu_forward(z: Any, alpha: float) -> Any:
    return ops.where(z > 0, z, alpha * (ops.exp(z) - 1.0))


def _elu_derivative(z: Any, alpha: float) -> Any:
    return ops.where(z > 0, ops.ones_like(z), alpha * ops.exp(z))


def _elu_integral(z: Any, alpha: float) -> Any:
    return ops.where(z > 0, 0.5 * z * z, alpha * (ops.exp(z) - 1.0 - z))


def _elu_fastpath(z: Any, n: int, alpha: float) -> Any:
    if n < 0:
        raise ValueError(f"order n must be >= 0, got {n}.")
    if n == 0:
        return _elu_forward(z, alpha)
    if n == 1:
        return _elu_derivative(z, alpha)
    return ops.where(z > 0, ops.zeros_like(z), alpha * ops.exp(z))


def make_elu_spec(alpha: float = _DEFAULT_ELU_ALPHA, *, name: str = "elu") -> ActivationSpec[Any]:
    """ELU spec with fixed ``alpha`` (exact on each open piece, all orders)."""

    def fwd(z: Any) -> Any:
        return _elu_forward(z, alpha)

    def deriv(z: Any) -> Any:
        return _elu_derivative(z, alpha)

    def integ(z: Any) -> Any:
        return _elu_integral(z, alpha)

    def fp(z: Any, n: int) -> Any:
        return _elu_fastpath(z, n, alpha)

    return ActivationSpec(
        name=name,
        forward=fwd,
        derivative=deriv,
        fastpath=fp,
        integral=integ,
        riccati_polynomial=None,
        noise_model="none",
        operator_role="ELU: z for z>0, alpha*(exp(z)-1) otherwise; exact per-piece tower.",
        limit_neg_inf=-alpha,
    )


ELU = register_activation(make_elu_spec(_DEFAULT_ELU_ALPHA))


def _selu_forward(z: Any) -> Any:
    return _SELU_SCALE * ops.where(z > 0, z, _SELU_ALPHA * (ops.exp(z) - 1.0))


def _selu_derivative(z: Any) -> Any:
    return _SELU_SCALE * ops.where(z > 0, ops.ones_like(z), _SELU_ALPHA * ops.exp(z))


def _selu_fastpath(z: Any, n: int) -> Any:
    if n < 0:
        raise ValueError(f"order n must be >= 0, got {n}.")
    if n == 0:
        return _selu_forward(z)
    if n == 1:
        return _selu_derivative(z)
    return _SELU_SCALE * ops.where(z > 0, ops.zeros_like(z), _SELU_ALPHA * ops.exp(z))


SELU = register_activation(
    ActivationSpec(
        name="selu",
        forward=_selu_forward,
        derivative=_selu_derivative,
        fastpath=_selu_fastpath,
        riccati_polynomial=None,
        noise_model="none",
        operator_role="SELU: scale * ELU_{alpha}; self-normalizing; exact per-piece tower.",
        limit_neg_inf=-_SELU_SCALE * _SELU_ALPHA,
    )
)


def _celu_forward(z: Any, alpha: float) -> Any:
    return ops.where(z > 0, z, alpha * (ops.exp(z / alpha) - 1.0))


def _celu_derivative(z: Any, alpha: float) -> Any:
    return ops.where(z > 0, ops.ones_like(z), ops.exp(z / alpha))


def _celu_fastpath(z: Any, n: int, alpha: float) -> Any:
    if n < 0:
        raise ValueError(f"order n must be >= 0, got {n}.")
    if n == 0:
        return _celu_forward(z, alpha)
    if n == 1:
        return _celu_derivative(z, alpha)
    coeff = alpha ** (1 - n)
    return ops.where(z > 0, ops.zeros_like(z), coeff * ops.exp(z / alpha))


def make_celu_spec(alpha: float = _DEFAULT_CELU_ALPHA, *, name: str = "celu") -> ActivationSpec[Any]:
    """CELU spec with fixed ``alpha`` (exact on each open piece, all orders)."""
    if alpha == 0:
        raise ValueError("CELU alpha must be nonzero.")

    def fwd(z: Any) -> Any:
        return _celu_forward(z, alpha)

    def deriv(z: Any) -> Any:
        return _celu_derivative(z, alpha)

    def fp(z: Any, n: int) -> Any:
        return _celu_fastpath(z, n, alpha)

    return ActivationSpec(
        name=name,
        forward=fwd,
        derivative=deriv,
        fastpath=fp,
        riccati_polynomial=None,
        noise_model="none",
        operator_role="CELU: z for z>0, alpha*(exp(z/alpha)-1) otherwise; exact per-piece tower.",
        limit_neg_inf=-alpha,
    )


CELU = register_activation(make_celu_spec(_DEFAULT_CELU_ALPHA))


# --- softshrink / hardshrink -----------------------------------------------


_DEFAULT_SHRINK_LAMBD = 0.5


def _softshrink_forward(z: Any, lambd: float) -> Any:
    return ops.where(
        z > lambd, z - lambd, ops.where(z < -lambd, z + lambd, ops.zeros_like(z))
    )


def _softshrink_derivative(z: Any, lambd: float) -> Any:
    return ops.cast(ops.abs(z) > lambd, dtype=z.dtype)


def make_softshrink_spec(
    lambd: float = _DEFAULT_SHRINK_LAMBD, *, name: str = "softshrink"
) -> ActivationSpec[Any]:
    """Soft-shrinkage spec (proximal of L1) with threshold ``lambd``."""

    def fwd(z: Any) -> Any:
        return _softshrink_forward(z, lambd)

    def deriv(z: Any) -> Any:
        return _softshrink_derivative(z, lambd)

    def fp(z: Any, n: int) -> Any:
        return _ae_linear_fastpath(z, n, fwd, deriv)

    return ActivationSpec(
        name=name,
        forward=fwd,
        derivative=deriv,
        fastpath=fp,
        riccati_polynomial=None,
        noise_model="none",
        operator_role="Soft-shrink (L1 proximal); a.e. regular-part tower (n>=2 -> 0).",
    )


SOFTSHRINK = register_activation(make_softshrink_spec())


def _hardshrink_forward(z: Any, lambd: float) -> Any:
    return ops.where(ops.abs(z) > lambd, z, ops.zeros_like(z))


def _hardshrink_derivative(z: Any, lambd: float) -> Any:
    return ops.cast(ops.abs(z) > lambd, dtype=z.dtype)


def make_hardshrink_spec(
    lambd: float = _DEFAULT_SHRINK_LAMBD, *, name: str = "hardshrink"
) -> ActivationSpec[Any]:
    """Hard-shrinkage spec with threshold ``lambd`` (discontinuous at +/-lambd)."""

    def fwd(z: Any) -> Any:
        return _hardshrink_forward(z, lambd)

    def deriv(z: Any) -> Any:
        return _hardshrink_derivative(z, lambd)

    def fp(z: Any, n: int) -> Any:
        return _ae_linear_fastpath(z, n, fwd, deriv)

    return ActivationSpec(
        name=name,
        forward=fwd,
        derivative=deriv,
        fastpath=fp,
        riccati_polynomial=None,
        noise_model="none",
        operator_role="Hard-shrink; a.e. regular-part tower (n>=2 -> 0).",
    )


HARDSHRINK = register_activation(make_hardshrink_spec())


# --- threshold -------------------------------------------------------------


_DEFAULT_THRESHOLD = 0.0
_DEFAULT_THRESHOLD_VALUE = 0.0


def _threshold_forward(z: Any, threshold: float, value: float) -> Any:
    return ops.where(z > threshold, z, value * ops.ones_like(z))


def _threshold_derivative(z: Any, threshold: float) -> Any:
    return ops.cast(z > threshold, dtype=z.dtype)


def make_threshold_spec(
    threshold: float = _DEFAULT_THRESHOLD,
    value: float = _DEFAULT_THRESHOLD_VALUE,
    *,
    name: str = "threshold",
) -> ActivationSpec[Any]:
    """Threshold spec: identity above ``threshold``, constant ``value`` below."""

    def fwd(z: Any) -> Any:
        return _threshold_forward(z, threshold, value)

    def deriv(z: Any) -> Any:
        return _threshold_derivative(z, threshold)

    def fp(z: Any, n: int) -> Any:
        return _ae_linear_fastpath(z, n, fwd, deriv)

    return ActivationSpec(
        name=name,
        forward=fwd,
        derivative=deriv,
        fastpath=fp,
        riccati_polynomial=None,
        noise_model="none",
        operator_role="Threshold gate; a.e. regular-part tower (n>=2 -> 0).",
        limit_neg_inf=value,
    )


THRESHOLD = register_activation(make_threshold_spec())


# --- abs -------------------------------------------------------------------


def _abs_forward(z: Any) -> Any:
    return ops.abs(z)


def _abs_derivative(z: Any) -> Any:
    return ops.sign(z)


def _abs_integral(z: Any) -> Any:
    return 0.5 * z * ops.abs(z)


def _abs_fastpath(z: Any, n: int) -> Any:
    return _ae_linear_fastpath(z, n, _abs_forward, _abs_derivative)


ABS = register_activation(
    ActivationSpec(
        name="abs",
        forward=_abs_forward,
        derivative=_abs_derivative,
        fastpath=_abs_fastpath,
        integral=_abs_integral,
        riccati_polynomial=None,
        noise_model="none",
        operator_role="Absolute value; derivative sign(z) (sign(0)=0), n>=2 -> 0 (a.e.).",
    )
)


# --- sign ------------------------------------------------------------------


def _sign_forward(z: Any) -> Any:
    return ops.sign(z)


def _sign_derivative(z: Any) -> Any:
    return ops.zeros_like(z)


def _sign_fastpath(z: Any, n: int) -> Any:
    if n < 0:
        raise ValueError(f"order n must be >= 0, got {n}.")
    if n == 0:
        return _sign_forward(z)
    return ops.zeros_like(z)


SIGN = register_activation(
    ActivationSpec(
        name="sign",
        forward=_sign_forward,
        derivative=_sign_derivative,
        fastpath=_sign_fastpath,
        riccati_polynomial=None,
        noise_model="none",
        operator_role=(
            "Sign (sign(0)=0); a.e. tower is zero for all n>=1 (the singular "
            "delta at 0 is dropped). Smooth twin: soft_sign (tanh(beta z))."
        ),
        limit_pos_inf=1.0,
        limit_neg_inf=-1.0,
    )
)


# --- step / heaviside ------------------------------------------------------


def _step_forward(z: Any) -> Any:
    return ops.cast(z > 0, dtype=z.dtype)


def _step_fastpath(z: Any, n: int) -> Any:
    if n < 0:
        raise ValueError(f"order n must be >= 0, got {n}.")
    if n == 0:
        return _step_forward(z)
    return ops.zeros_like(z)


STEP = register_activation(
    ActivationSpec(
        name="step",
        forward=_step_forward,
        derivative=_sign_derivative,
        fastpath=_step_fastpath,
        riccati_polynomial=None,
        noise_model="none",
        operator_role=(
            "Heaviside step (H(0)=0); a.e. tower is zero for all n>=1. Smooth "
            "twin: soft_step (sigmoid(beta z))."
        ),
        aliases=("heaviside",),
        limit_pos_inf=1.0,
        limit_neg_inf=0.0,
    )
)


# --- softsign --------------------------------------------------------------


def _softsign_forward(z: Any) -> Any:
    return z / (1.0 + ops.abs(z))


def _softsign_derivative(z: Any) -> Any:
    d = 1.0 + ops.abs(z)
    return 1.0 / (d * d)


def _softsign_fastpath(z: Any, n: int) -> Any:
    """Exact closed-form ``softsign^(n)`` (all orders); shared denominator (1+|z|)^(n+1)."""
    if n < 0:
        raise ValueError(f"order n must be >= 0, got {n}.")
    if n == 0:
        return _softsign_forward(z)
    fact = float(math.factorial(n))
    denom = (1.0 + ops.abs(z)) ** (n + 1)
    sign_pos = fact if (n % 2 == 1) else -fact
    num = ops.where(z >= 0, sign_pos * ops.ones_like(z), fact * ops.ones_like(z))
    return num / denom


SOFTSIGN = register_activation(
    ActivationSpec(
        name="softsign",
        forward=_softsign_forward,
        derivative=_softsign_derivative,
        fastpath=_softsign_fastpath,
        riccati_polynomial=None,
        noise_model="none",
        operator_role="Softsign: z/(1+|z|); exact rational per-arm tower.",
        limit_pos_inf=1.0,
        limit_neg_inf=-1.0,
    )
)


__all__ = [
    "ABS",
    "CELU",
    "ELU",
    "HARDSHRINK",
    "HARDSIGMOID",
    "HARDSWISH",
    "HARDTANH",
    "LEAKY_RELU",
    "PRELU",
    "RELU6",
    "SELU",
    "SIGN",
    "SOFTSHRINK",
    "SOFTSIGN",
    "STEP",
    "THRESHOLD",
    "make_celu_spec",
    "make_elu_spec",
    "make_hardshrink_spec",
    "make_hardtanh_spec",
    "make_leaky_relu_spec",
    "make_softshrink_spec",
    "make_threshold_spec",
]
