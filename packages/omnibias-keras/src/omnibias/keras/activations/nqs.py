# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""NQS-friendly activations: log_cosh, softabs, smooth_sign, mish.

Canonical building blocks of neural-quantum-state ansatze. Mirrors
:mod:`omnibias.torch.activations.nqs` (and
:mod:`omnibias.jax.activations`) on ``keras.ops`` so all backends
register the same names.
"""

from __future__ import annotations

import math
from typing import Any

from omnibias.core.polynomials import mish_inner_coeffs
from omnibias.keras.activations.registry import ActivationSpec, register_activation
from omnibias.keras.fastpath.legendre import tanh_nth_derivative

from keras import ops

_LOG_TWO = math.log(2.0)


# --- log_cosh -------------------------------------------------------------


def _stable_log_cosh(z: Any) -> Any:
    absz = ops.abs(z)
    return absz + ops.log1p(ops.exp(-2.0 * absz)) - _LOG_TWO


def _log_cosh_forward(z: Any) -> Any:
    return _stable_log_cosh(z)


def _log_cosh_fastpath(z: Any, n: int) -> Any:
    if n < 0:
        raise ValueError(f"order n must be >= 0, got {n}.")
    if n == 0:
        return _log_cosh_forward(z)
    if n == 1:
        return ops.tanh(z)
    if n == 2:
        t = ops.tanh(z)
        return 1.0 - t * t
    if n == 3:
        t = ops.tanh(z)
        return -2.0 * t * (1.0 - t * t)
    raise NotImplementedError(f"log_cosh fast path only implements n in {{0, 1, 2, 3}}, got {n}.")


LOG_COSH = register_activation(
    ActivationSpec(
        name="log_cosh",
        forward=_log_cosh_forward,
        derivative=lambda z: ops.tanh(z),
        fastpath=_log_cosh_fastpath,
        riccati_polynomial=None,
        noise_model="laplace_smoothed",
        operator_role=(
            "Smooth-|z| activation: log(cosh(z)) ~ |z| for large |z|, "
            "~ z^2/2 for small. Standard log-amplitude in spin-NQS; "
            "first derivative is tanh, second is 1 - tanh^2."
        ),
        aliases=("logcosh",),
    )
)


# --- softabs --------------------------------------------------------------


_DEFAULT_SOFTABS_EPS = 1e-3


def _softabs_forward(z: Any, eps: float = _DEFAULT_SOFTABS_EPS) -> Any:
    return ops.sqrt(z * z + eps * eps) - eps


def _softabs_derivative(z: Any, eps: float = _DEFAULT_SOFTABS_EPS) -> Any:
    return z / ops.sqrt(z * z + eps * eps)


def _softabs_integral(z: Any, eps: float = _DEFAULT_SOFTABS_EPS) -> Any:
    root = ops.sqrt(z * z + eps * eps)
    return 0.5 * (z * root + eps * eps * ops.arcsinh(z / eps)) - eps * z


def _softabs_fastpath(z: Any, n: int, eps: float = _DEFAULT_SOFTABS_EPS) -> Any:
    if n < 0:
        raise ValueError(f"order n must be >= 0, got {n}.")
    if n == 0:
        return _softabs_forward(z, eps)
    if n == 1:
        return _softabs_derivative(z, eps)
    if n == 2:
        denom = (z * z + eps * eps) ** 1.5
        return (eps * eps) / denom
    raise NotImplementedError(f"softabs fast path only implements n in {{0, 1, 2}}, got {n}.")


SOFTABS = register_activation(
    ActivationSpec(
        name="softabs",
        forward=_softabs_forward,
        derivative=_softabs_derivative,
        fastpath=_softabs_fastpath,
        integral=_softabs_integral,
        riccati_polynomial=None,
        noise_model="huber_smoothed",
        operator_role=(
            "Smooth absolute value: sqrt(z^2 + eps^2) - eps. C^2 everywhere; "
            "the eps-tempered abs surrogate (-> |z| as eps -> 0); useful in "
            "Jastrow factors and complex-amplitude magnitudes."
        ),
        aliases=("soft_abs",),
    )
)


# --- smooth_sign ----------------------------------------------------------


_DEFAULT_SMOOTH_SIGN_T = 1.0


def _smooth_sign_forward(z: Any, T: float = _DEFAULT_SMOOTH_SIGN_T) -> Any:
    return ops.tanh(z / T)


def _smooth_sign_derivative(z: Any, T: float = _DEFAULT_SMOOTH_SIGN_T) -> Any:
    t = ops.tanh(z / T)
    return (1.0 - t * t) / T


def _smooth_sign_integral(z: Any, T: float = _DEFAULT_SMOOTH_SIGN_T) -> Any:
    return T * _stable_log_cosh(z / T)


def _smooth_sign_fastpath(z: Any, n: int, T: float = _DEFAULT_SMOOTH_SIGN_T) -> Any:
    """Closed-form all-orders ``d^n/dz^n tanh(z / T) = tanh^(n)(z/T) / T^n``.

    The beta-tempered ``sign`` surrogate (``beta = 1 / T``).
    """
    if n < 0:
        raise ValueError(f"order n must be >= 0, got {n}.")
    if n == 0:
        return _smooth_sign_forward(z, T)
    return tanh_nth_derivative(z / T, n) / (T**n)


SMOOTH_SIGN = register_activation(
    ActivationSpec(
        name="smooth_sign",
        forward=_smooth_sign_forward,
        derivative=_smooth_sign_derivative,
        fastpath=_smooth_sign_fastpath,
        integral=_smooth_sign_integral,
        riccati_polynomial=None,
        noise_model="symmetric_bernoulli_tempered",
        operator_role=(
            "Temperature-controlled smooth sign: tanh(z/T) -> sign(z) as "
            "T -> 0. The beta-tempered sign surrogate; used in variational "
            "annealing schedules."
        ),
        aliases=("soft_sign",),
        limit_pos_inf=1.0,
        limit_neg_inf=-1.0,
    )
)


# --- mish -----------------------------------------------------------------


def _mish_forward(z: Any) -> Any:
    return z * ops.tanh(ops.softplus(z))


def _mish_inner_nth(z: Any, n: int) -> Any:
    """``g^(n)(z)`` for ``g(z) = tanh(softplus(z))`` via the shared ``(t, s)`` tower."""
    t = ops.tanh(ops.softplus(z))
    if n == 0:
        return t
    s = ops.sigmoid(z)
    acc = ops.zeros_like(z)
    for i, j, c in mish_inner_coeffs(n):
        acc = acc + c * (t**i) * (s**j)
    return acc


def _mish_derivative(z: Any) -> Any:
    return _mish_fastpath(z, 1)


def _mish_fastpath(z: Any, n: int) -> Any:
    """Exact closed-form ``mish^(n)`` (all orders); ``mish = z * g``, Leibniz + ``(t, s)`` tower."""
    if n < 0:
        raise ValueError(f"order n must be >= 0, got {n}.")
    if n == 0:
        return _mish_forward(z)
    return z * _mish_inner_nth(z, n) + n * _mish_inner_nth(z, n - 1)


MISH = register_activation(
    ActivationSpec(
        name="mish",
        forward=_mish_forward,
        derivative=_mish_derivative,
        fastpath=_mish_fastpath,
        riccati_polynomial=None,
        noise_model="none",
        operator_role=("Mish: z * tanh(softplus(z)). Self-gated residual activation."),
    )
)


__all__ = ["LOG_COSH", "MISH", "SMOOTH_SIGN", "SOFTABS"]
