# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""JAX twins of the piecewise (almost-everywhere) activation family.

Bit-identical mirror of :mod:`omnibias.torch.activations.piecewise`; see that
module for the derivation of every tower. Imported for side-effect registration
by :mod:`omnibias.jax.activations`. Every kernel is ``jit`` / ``vmap`` safe:
branching is only on the static Python order ``n`` and on the traced input via
``jnp.where`` (never on host-side ``beta``).
"""

from __future__ import annotations

import math
from collections.abc import Callable

from omnibias.jax.activations import JaxActivationSpec, register_activation

import jax.numpy as jnp
from jax import Array

ArrayFn = Callable[[Array], Array]


def _ae_linear_fastpath(z: Array, n: int, forward: ArrayFn, derivative: ArrayFn) -> Array:
    """``n = 0`` forward, ``n = 1`` derivative, ``n >= 2`` zero (a.e.)."""
    if n < 0:
        raise ValueError(f"order n must be >= 0, got {n}.")
    if n == 0:
        return forward(z)
    if n == 1:
        return derivative(z)
    return jnp.zeros_like(z)


# --- leaky_relu / prelu ----------------------------------------------------


_DEFAULT_LEAKY_SLOPE = 0.01
_DEFAULT_PRELU_INIT = 0.25


def _leaky_relu_forward(z: Array, slope: float) -> Array:
    return jnp.where(z > 0, z, slope * z)


def _leaky_relu_derivative(z: Array, slope: float) -> Array:
    return jnp.where(z > 0, jnp.ones_like(z), jnp.full_like(z, slope))


def _leaky_relu_integral(z: Array, slope: float) -> Array:
    return jnp.where(z > 0, 0.5 * z * z, 0.5 * slope * z * z)


def make_leaky_relu_spec(
    slope: float = _DEFAULT_LEAKY_SLOPE, *, name: str = "leaky_relu"
) -> JaxActivationSpec:
    """Leaky-ReLU spec with fixed negative ``slope`` (a.e. all-orders tower)."""

    def fwd(z: Array) -> Array:
        return _leaky_relu_forward(z, slope)

    def deriv(z: Array) -> Array:
        return _leaky_relu_derivative(z, slope)

    def integ(z: Array) -> Array:
        return _leaky_relu_integral(z, slope)

    def fp(z: Array, n: int) -> Array:
        return _ae_linear_fastpath(z, n, fwd, deriv)

    return JaxActivationSpec(
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


def _relu6_forward(z: Array) -> Array:
    return jnp.clip(z, 0.0, 6.0)


def _relu6_derivative(z: Array) -> Array:
    return ((z > 0) & (z < 6.0)).astype(z.dtype)


def _relu6_fastpath(z: Array, n: int) -> Array:
    return _ae_linear_fastpath(z, n, _relu6_forward, _relu6_derivative)


RELU6 = register_activation(
    JaxActivationSpec(
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


def _hardtanh_forward(z: Array, lo: float, hi: float) -> Array:
    return jnp.clip(z, lo, hi)


def _hardtanh_derivative(z: Array, lo: float, hi: float) -> Array:
    return ((z > lo) & (z < hi)).astype(z.dtype)


def make_hardtanh_spec(
    min_val: float = _DEFAULT_HARDTANH_MIN,
    max_val: float = _DEFAULT_HARDTANH_MAX,
    *,
    name: str = "hardtanh",
) -> JaxActivationSpec:
    """Hardtanh spec clamping to ``[min_val, max_val]`` (a.e. all-orders tower)."""

    def fwd(z: Array) -> Array:
        return _hardtanh_forward(z, min_val, max_val)

    def deriv(z: Array) -> Array:
        return _hardtanh_derivative(z, min_val, max_val)

    def fp(z: Array, n: int) -> Array:
        return _ae_linear_fastpath(z, n, fwd, deriv)

    return JaxActivationSpec(
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


def _hardsigmoid_forward(z: Array) -> Array:
    return jnp.clip(z / 6.0 + 0.5, 0.0, 1.0)


def _hardsigmoid_derivative(z: Array) -> Array:
    return jnp.where(
        (z > -3.0) & (z < 3.0), jnp.full_like(z, 1.0 / 6.0), jnp.zeros_like(z)
    )


def _hardsigmoid_fastpath(z: Array, n: int) -> Array:
    return _ae_linear_fastpath(z, n, _hardsigmoid_forward, _hardsigmoid_derivative)


HARDSIGMOID = register_activation(
    JaxActivationSpec(
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


def _hardswish_forward(z: Array) -> Array:
    return z * jnp.clip(z / 6.0 + 0.5, 0.0, 1.0)


def _hardswish_derivative(z: Array) -> Array:
    mid = (2.0 * z + 3.0) / 6.0
    out = jnp.where(z >= 3.0, jnp.ones_like(z), mid)
    return jnp.where(z <= -3.0, jnp.zeros_like(z), out)


def _hardswish_fastpath(z: Array, n: int) -> Array:
    if n < 0:
        raise ValueError(f"order n must be >= 0, got {n}.")
    if n == 0:
        return _hardswish_forward(z)
    if n == 1:
        return _hardswish_derivative(z)
    if n == 2:
        return jnp.where(
            (z > -3.0) & (z < 3.0), jnp.full_like(z, 1.0 / 3.0), jnp.zeros_like(z)
        )
    return jnp.zeros_like(z)


HARDSWISH = register_activation(
    JaxActivationSpec(
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


def _elu_forward(z: Array, alpha: float) -> Array:
    return jnp.where(z > 0, z, alpha * (jnp.exp(z) - 1.0))


def _elu_derivative(z: Array, alpha: float) -> Array:
    return jnp.where(z > 0, jnp.ones_like(z), alpha * jnp.exp(z))


def _elu_integral(z: Array, alpha: float) -> Array:
    return jnp.where(z > 0, 0.5 * z * z, alpha * (jnp.exp(z) - 1.0 - z))


def _elu_fastpath(z: Array, n: int, alpha: float) -> Array:
    if n < 0:
        raise ValueError(f"order n must be >= 0, got {n}.")
    if n == 0:
        return _elu_forward(z, alpha)
    if n == 1:
        return _elu_derivative(z, alpha)
    return jnp.where(z > 0, jnp.zeros_like(z), alpha * jnp.exp(z))


def make_elu_spec(alpha: float = _DEFAULT_ELU_ALPHA, *, name: str = "elu") -> JaxActivationSpec:
    """ELU spec with fixed ``alpha`` (exact on each open piece, all orders)."""

    def fwd(z: Array) -> Array:
        return _elu_forward(z, alpha)

    def deriv(z: Array) -> Array:
        return _elu_derivative(z, alpha)

    def integ(z: Array) -> Array:
        return _elu_integral(z, alpha)

    def fp(z: Array, n: int) -> Array:
        return _elu_fastpath(z, n, alpha)

    return JaxActivationSpec(
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


def _selu_forward(z: Array) -> Array:
    return _SELU_SCALE * jnp.where(z > 0, z, _SELU_ALPHA * (jnp.exp(z) - 1.0))


def _selu_derivative(z: Array) -> Array:
    return _SELU_SCALE * jnp.where(z > 0, jnp.ones_like(z), _SELU_ALPHA * jnp.exp(z))


def _selu_fastpath(z: Array, n: int) -> Array:
    if n < 0:
        raise ValueError(f"order n must be >= 0, got {n}.")
    if n == 0:
        return _selu_forward(z)
    if n == 1:
        return _selu_derivative(z)
    return _SELU_SCALE * jnp.where(z > 0, jnp.zeros_like(z), _SELU_ALPHA * jnp.exp(z))


SELU = register_activation(
    JaxActivationSpec(
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


def _celu_forward(z: Array, alpha: float) -> Array:
    return jnp.where(z > 0, z, alpha * (jnp.exp(z / alpha) - 1.0))


def _celu_derivative(z: Array, alpha: float) -> Array:
    return jnp.where(z > 0, jnp.ones_like(z), jnp.exp(z / alpha))


def _celu_fastpath(z: Array, n: int, alpha: float) -> Array:
    if n < 0:
        raise ValueError(f"order n must be >= 0, got {n}.")
    if n == 0:
        return _celu_forward(z, alpha)
    if n == 1:
        return _celu_derivative(z, alpha)
    coeff = alpha ** (1 - n)
    return jnp.where(z > 0, jnp.zeros_like(z), coeff * jnp.exp(z / alpha))


def make_celu_spec(alpha: float = _DEFAULT_CELU_ALPHA, *, name: str = "celu") -> JaxActivationSpec:
    """CELU spec with fixed ``alpha`` (exact on each open piece, all orders)."""
    if alpha == 0:
        raise ValueError("CELU alpha must be nonzero.")

    def fwd(z: Array) -> Array:
        return _celu_forward(z, alpha)

    def deriv(z: Array) -> Array:
        return _celu_derivative(z, alpha)

    def fp(z: Array, n: int) -> Array:
        return _celu_fastpath(z, n, alpha)

    return JaxActivationSpec(
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


def _softshrink_forward(z: Array, lambd: float) -> Array:
    return jnp.where(
        z > lambd, z - lambd, jnp.where(z < -lambd, z + lambd, jnp.zeros_like(z))
    )


def _softshrink_derivative(z: Array, lambd: float) -> Array:
    return (jnp.abs(z) > lambd).astype(z.dtype)


def make_softshrink_spec(
    lambd: float = _DEFAULT_SHRINK_LAMBD, *, name: str = "softshrink"
) -> JaxActivationSpec:
    """Soft-shrinkage spec (proximal of L1) with threshold ``lambd``."""

    def fwd(z: Array) -> Array:
        return _softshrink_forward(z, lambd)

    def deriv(z: Array) -> Array:
        return _softshrink_derivative(z, lambd)

    def fp(z: Array, n: int) -> Array:
        return _ae_linear_fastpath(z, n, fwd, deriv)

    return JaxActivationSpec(
        name=name,
        forward=fwd,
        derivative=deriv,
        fastpath=fp,
        riccati_polynomial=None,
        noise_model="none",
        operator_role="Soft-shrink (L1 proximal); a.e. regular-part tower (n>=2 -> 0).",
    )


SOFTSHRINK = register_activation(make_softshrink_spec())


def _hardshrink_forward(z: Array, lambd: float) -> Array:
    return jnp.where(jnp.abs(z) > lambd, z, jnp.zeros_like(z))


def _hardshrink_derivative(z: Array, lambd: float) -> Array:
    return (jnp.abs(z) > lambd).astype(z.dtype)


def make_hardshrink_spec(
    lambd: float = _DEFAULT_SHRINK_LAMBD, *, name: str = "hardshrink"
) -> JaxActivationSpec:
    """Hard-shrinkage spec with threshold ``lambd`` (discontinuous at +/-lambd)."""

    def fwd(z: Array) -> Array:
        return _hardshrink_forward(z, lambd)

    def deriv(z: Array) -> Array:
        return _hardshrink_derivative(z, lambd)

    def fp(z: Array, n: int) -> Array:
        return _ae_linear_fastpath(z, n, fwd, deriv)

    return JaxActivationSpec(
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


def _threshold_forward(z: Array, threshold: float, value: float) -> Array:
    return jnp.where(z > threshold, z, jnp.full_like(z, value))


def _threshold_derivative(z: Array, threshold: float) -> Array:
    return (z > threshold).astype(z.dtype)


def make_threshold_spec(
    threshold: float = _DEFAULT_THRESHOLD,
    value: float = _DEFAULT_THRESHOLD_VALUE,
    *,
    name: str = "threshold",
) -> JaxActivationSpec:
    """Threshold spec: identity above ``threshold``, constant ``value`` below."""

    def fwd(z: Array) -> Array:
        return _threshold_forward(z, threshold, value)

    def deriv(z: Array) -> Array:
        return _threshold_derivative(z, threshold)

    def fp(z: Array, n: int) -> Array:
        return _ae_linear_fastpath(z, n, fwd, deriv)

    return JaxActivationSpec(
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


def _abs_forward(z: Array) -> Array:
    return jnp.abs(z)


def _abs_derivative(z: Array) -> Array:
    return jnp.sign(z)


def _abs_integral(z: Array) -> Array:
    return 0.5 * z * jnp.abs(z)


def _abs_fastpath(z: Array, n: int) -> Array:
    return _ae_linear_fastpath(z, n, _abs_forward, _abs_derivative)


ABS = register_activation(
    JaxActivationSpec(
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


def _sign_forward(z: Array) -> Array:
    return jnp.sign(z)


def _sign_derivative(z: Array) -> Array:
    return jnp.zeros_like(z)


def _sign_fastpath(z: Array, n: int) -> Array:
    if n < 0:
        raise ValueError(f"order n must be >= 0, got {n}.")
    if n == 0:
        return _sign_forward(z)
    return jnp.zeros_like(z)


SIGN = register_activation(
    JaxActivationSpec(
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


def _step_forward(z: Array) -> Array:
    return (z > 0).astype(z.dtype)


def _step_fastpath(z: Array, n: int) -> Array:
    if n < 0:
        raise ValueError(f"order n must be >= 0, got {n}.")
    if n == 0:
        return _step_forward(z)
    return jnp.zeros_like(z)


STEP = register_activation(
    JaxActivationSpec(
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


def _softsign_forward(z: Array) -> Array:
    return z / (1.0 + jnp.abs(z))


def _softsign_derivative(z: Array) -> Array:
    d = 1.0 + jnp.abs(z)
    return 1.0 / (d * d)


def _softsign_fastpath(z: Array, n: int) -> Array:
    """Exact closed-form ``softsign^(n)`` (all orders); shared denominator (1+|z|)^(n+1)."""
    if n < 0:
        raise ValueError(f"order n must be >= 0, got {n}.")
    if n == 0:
        return _softsign_forward(z)
    fact = float(math.factorial(n))
    denom = (1.0 + jnp.abs(z)) ** (n + 1)
    sign_pos = fact if (n % 2 == 1) else -fact
    num = jnp.where(z >= 0, jnp.full_like(z, sign_pos), jnp.full_like(z, fact))
    return num / denom


SOFTSIGN = register_activation(
    JaxActivationSpec(
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
