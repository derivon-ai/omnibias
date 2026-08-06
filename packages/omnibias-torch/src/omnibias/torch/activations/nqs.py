# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""NQS-friendly activations (lattice work, mirrored from JAX).

These activations are the canonical building blocks of neural-quantum-state
ansatze on lattice systems:

==================  =========================================  ===================
Activation          K=2 collapse / role                        Max fastpath order
==================  =========================================  ===================
``log_cosh``        ``log(cosh(z))``: smooth |z|; the standard  n in {0, 1, 2, 3}
                    log-amplitude in spin-NQS (Carleo-Troyer
                    RBM family). 1st derivative is ``tanh``.
``softabs``         ``sqrt(z^2 + eps^2) - eps``: smooth |z|     n in {0, 1, 2}
                    used in Jastrow factors and complex
                    amplitude magnitudes.
``smooth_sign``     ``tanh(z / T)``: temperature-controlled     all orders
                    smooth sign; used in variational
                    annealing schedules.
``mish``            ``z * tanh(softplus(z))``: self-gated       all orders
                    residual activation, transformer FFN.
==================  =========================================  ===================

``smooth_sign`` is ``tanh`` rescaled, so it inherits the whole Riccati tower;
``mish`` is the analytic product ``z * g(z)``, so Leibniz over the ``(t, s)``
recurrence gives every order. Both were once capped and are no longer.

The PyTorch implementations here mirror :mod:`omnibias.jax.activations` so
both backends register the same set of names; the parity test in
``tests/test_jax_parity.py`` then verifies bit-stable agreement.
"""

from __future__ import annotations

from omnibias.core.polynomials import mish_inner_coeffs
from omnibias.torch.activations.registry import ActivationSpec, register_activation
from omnibias.torch.fastpath.legendre import tanh_nth_derivative

import torch
from torch import Tensor

# ---------------------------------------------------------------------------
# log_cosh: log(cosh(z))   (smooth |z|; standard NQS log-amplitude)
# ---------------------------------------------------------------------------


def _stable_log_cosh(z: Tensor) -> Tensor:
    log_two = z.new_tensor(0.6931471805599453)
    absz = z.abs()
    return absz + torch.log1p(torch.exp(-2.0 * absz)) - log_two


def _log_cosh_forward(z: Tensor) -> Tensor:
    return _stable_log_cosh(z)


def _log_cosh_derivative(z: Tensor) -> Tensor:
    return torch.tanh(z)


def _log_cosh_fastpath(z: Tensor, n: int) -> Tensor:
    if n < 0:
        raise ValueError(f"order n must be >= 0, got {n}.")
    if n == 0:
        return _log_cosh_forward(z)
    if n == 1:
        return _log_cosh_derivative(z)
    if n == 2:
        t = torch.tanh(z)
        return 1.0 - t * t
    if n == 3:
        t = torch.tanh(z)
        return -2.0 * t * (1.0 - t * t)
    raise NotImplementedError(f"log_cosh fast path only implements n in {{0, 1, 2, 3}}, got {n}.")


LOG_COSH = register_activation(
    ActivationSpec(
        name="log_cosh",
        forward=_log_cosh_forward,
        derivative=_log_cosh_derivative,
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


# ---------------------------------------------------------------------------
# softabs: sqrt(z^2 + eps^2) - eps   (smooth |z|, C^2 everywhere)
# ---------------------------------------------------------------------------


_DEFAULT_SOFTABS_EPS = 1e-3


def _softabs_forward(z: Tensor, eps: float = _DEFAULT_SOFTABS_EPS) -> Tensor:
    return torch.sqrt(z * z + eps * eps) - eps


def _softabs_derivative(z: Tensor, eps: float = _DEFAULT_SOFTABS_EPS) -> Tensor:
    return z / torch.sqrt(z * z + eps * eps)


def _softabs_integral(z: Tensor, eps: float = _DEFAULT_SOFTABS_EPS) -> Tensor:
    root = torch.sqrt(z * z + eps * eps)
    return 0.5 * (z * root + eps * eps * torch.asinh(z / eps)) - eps * z


def _softabs_fastpath(z: Tensor, n: int, eps: float = _DEFAULT_SOFTABS_EPS) -> Tensor:
    if n < 0:
        raise ValueError(f"order n must be >= 0, got {n}.")
    if n == 0:
        return _softabs_forward(z, eps)
    if n == 1:
        return _softabs_derivative(z, eps)
    if n == 2:
        denom = (z * z + eps * eps) ** 1.5
        return torch.full_like(z, eps * eps) / denom
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


# ---------------------------------------------------------------------------
# smooth_sign: tanh(z / T)   (temperature-annealed smooth sign)
# ---------------------------------------------------------------------------


_DEFAULT_SMOOTH_SIGN_T = 1.0


def _smooth_sign_forward(z: Tensor, T: float = _DEFAULT_SMOOTH_SIGN_T) -> Tensor:
    return torch.tanh(z / T)


def _smooth_sign_derivative(z: Tensor, T: float = _DEFAULT_SMOOTH_SIGN_T) -> Tensor:
    t = torch.tanh(z / T)
    return (1.0 - t * t) / T


def _smooth_sign_integral(z: Tensor, T: float = _DEFAULT_SMOOTH_SIGN_T) -> Tensor:
    return T * _stable_log_cosh(z / T)


def _smooth_sign_fastpath(z: Tensor, n: int, T: float = _DEFAULT_SMOOTH_SIGN_T) -> Tensor:
    """Closed-form all-orders ``d^n/dz^n tanh(z / T) = tanh^(n)(z/T) / T^n``.

    The beta-tempered ``sign`` surrogate (``beta = 1 / T``): as ``T -> 0`` it
    approaches ``sign(z)`` and the higher-order bumps concentrate into the
    distributional derivatives of ``sign``.
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


# ---------------------------------------------------------------------------
# mish: z * tanh(softplus(z))   (self-gated residual)
# ---------------------------------------------------------------------------


def _mish_forward(z: Tensor) -> Tensor:
    sp = torch.nn.functional.softplus(z)
    return z * torch.tanh(sp)


def _mish_inner_nth(z: Tensor, n: int) -> Tensor:
    """``g^(n)(z)`` for ``g(z) = tanh(softplus(z))`` via the ``(t, s)`` tower."""
    t = torch.tanh(torch.nn.functional.softplus(z))
    if n == 0:
        return t
    s = torch.sigmoid(z)
    acc = torch.zeros_like(z)
    for i, j, c in mish_inner_coeffs(n):
        acc = acc + c * t**i * s**j
    return acc


def _mish_derivative(z: Tensor) -> Tensor:
    return _mish_fastpath(z, 1)


def _mish_fastpath(z: Tensor, n: int) -> Tensor:
    """Exact closed-form ``mish^(n)`` (all orders).

    ``mish(z) = z * g(z)`` with ``g = tanh(softplus(z))``; Leibniz gives
    ``mish^(n) = z * g^(n) + n * g^(n-1)`` and the inner tower ``g^(n)`` comes
    from the closed-form ``(t, s)`` recurrence in
    :func:`omnibias.core.polynomials.mish_inner_coeffs`.
    """
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
