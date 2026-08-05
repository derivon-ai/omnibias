# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Torch closed-form quantization-gradient kernels."""

from __future__ import annotations

import torch
from omnibias.core.polynomials import sigmoid_polynomial_coeffs, tanh_polynomial_coeffs
from torch import Tensor

__all__ = [
    "binarize",
    "binarize01",
    "heaviside",
    "kbit_quantize",
    "riccati_sigmoid_derivative",
    "riccati_tanh_derivative",
    "ternarize",
]


def _horner(coeffs: tuple[float, ...], x: Tensor) -> Tensor:
    deg = len(coeffs) - 1
    result = torch.full_like(x, coeffs[deg])
    for k in range(deg - 1, -1, -1):
        result = result * x + coeffs[k]
    return result


def riccati_tanh_derivative(t: Tensor, order: int = 1) -> Tensor:
    """Evaluate ``T_order(t) = tanh^(order)(z)`` as a polynomial in ``t = tanh(z)``."""
    if order < 0:
        raise ValueError(f"order must be >= 0, got {order}")
    if order == 0:
        return t
    coeffs = tanh_polynomial_coeffs(order)
    return _horner(coeffs, t)


def _tanh_beta_prime(z: Tensor, beta: Tensor | float) -> Tensor:
    """``beta * tanh'(beta z)`` via the Riccati polynomial at ``t = tanh(beta z)``."""
    t = torch.tanh(beta * z)
    return beta * riccati_tanh_derivative(t, order=1)


def riccati_sigmoid_derivative(s: Tensor, order: int = 1) -> Tensor:
    """Evaluate ``P_order(s) = sigmoid^(order)(z)`` as a polynomial in ``s = sigmoid(z)``."""
    if order < 0:
        raise ValueError(f"order must be >= 0, got {order}")
    if order == 0:
        return s
    coeffs = sigmoid_polynomial_coeffs(order)
    return _horner(coeffs, s)


def _sigmoid_beta_prime(z: Tensor, beta: Tensor | float) -> Tensor:
    """``beta * sigmoid'(beta z)`` via the Riccati polynomial at ``s = sigmoid(beta z)``."""
    s = torch.sigmoid(beta * z)
    return beta * riccati_sigmoid_derivative(s, order=1)


def _kbit_thresholds(bits: int, lo: float, hi: float) -> tuple[float, float, tuple[float, ...]]:
    n_levels = 2**bits
    level_step = (hi - lo) / (n_levels - 1)
    thresholds = tuple(lo + k * level_step for k in range(1, n_levels))
    return level_step, float(n_levels), thresholds


def _kbit_forward(z: Tensor, bits: int, lo: float, hi: float) -> Tensor:
    level_step, n_levels, _ = _kbit_thresholds(bits, lo, hi)
    idx = torch.round((z - lo) / level_step)
    idx = torch.clamp(idx, 0, int(n_levels) - 1)
    return lo + idx * level_step


def _kbit_surrogate_grad(z: Tensor, bits: int, lo: float, hi: float, beta: Tensor | float) -> Tensor:
    level_step, _, thresholds = _kbit_thresholds(bits, lo, hi)
    grad = torch.zeros_like(z)
    for t_k in thresholds:
        grad = grad + level_step * _tanh_beta_prime(z - t_k, beta)
    return grad


class _BinarizeFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z: Tensor, beta: Tensor | float) -> Tensor:  # type: ignore[no-untyped-def]
        ctx.save_for_backward(z)
        ctx.beta = beta
        return torch.where(z >= 0, torch.ones_like(z), -torch.ones_like(z))

    @staticmethod
    def backward(ctx, grad_out: Tensor) -> tuple[Tensor | None, Tensor | None]:  # type: ignore[no-untyped-def]
        (z,) = ctx.saved_tensors
        beta = ctx.beta
        grad_z = grad_out * _tanh_beta_prime(z, beta)
        grad_beta: Tensor | None = None
        if isinstance(beta, Tensor) and beta.requires_grad:
            t = torch.tanh(beta * z)
            grad_beta = (grad_out * z * riccati_tanh_derivative(t, order=1)).sum()
        return grad_z, grad_beta


class _Binarize01Fn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z: Tensor, beta: Tensor | float) -> Tensor:  # type: ignore[no-untyped-def]
        ctx.save_for_backward(z)
        ctx.beta = beta
        return torch.where(z >= 0, torch.ones_like(z), torch.zeros_like(z))

    @staticmethod
    def backward(ctx, grad_out: Tensor) -> tuple[Tensor | None, Tensor | None]:  # type: ignore[no-untyped-def]
        (z,) = ctx.saved_tensors
        beta = ctx.beta
        grad_z = grad_out * _sigmoid_beta_prime(z, beta)
        grad_beta: Tensor | None = None
        if isinstance(beta, Tensor) and beta.requires_grad:
            s = torch.sigmoid(beta * z)
            grad_beta = (grad_out * z * riccati_sigmoid_derivative(s, order=1)).sum()
        return grad_z, grad_beta


class _TernarizeFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z: Tensor, beta: Tensor | float, delta: float) -> Tensor:  # type: ignore[no-untyped-def]
        ctx.save_for_backward(z)
        ctx.beta = beta
        ctx.delta = delta
        pos = torch.ones_like(z)
        neg = -torch.ones_like(z)
        out = torch.where(z > delta, pos, torch.where(z < -delta, neg, torch.zeros_like(z)))
        return out

    @staticmethod
    def backward(ctx, grad_out: Tensor) -> tuple[Tensor | None, Tensor | None, None]:  # type: ignore[no-untyped-def]
        (z,) = ctx.saved_tensors
        beta = ctx.beta
        delta = ctx.delta
        half = 0.5
        grad_z = grad_out * half * (
            _tanh_beta_prime(z - delta, beta) + _tanh_beta_prime(z + delta, beta)
        )
        grad_beta: Tensor | None = None
        if isinstance(beta, Tensor) and beta.requires_grad:
            t_lo = torch.tanh(beta * (z - delta))
            t_hi = torch.tanh(beta * (z + delta))
            grad_beta = (
                grad_out
                * half
                * (
                    (z - delta) * riccati_tanh_derivative(t_lo, order=1)
                    + (z + delta) * riccati_tanh_derivative(t_hi, order=1)
                )
            ).sum()
        return grad_z, grad_beta, None


class _KBitQuantizeFn(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[no-untyped-def]
        ctx,
        z: Tensor,
        bits: int,
        lo: float,
        hi: float,
        beta: Tensor | float,
    ) -> Tensor:
        ctx.save_for_backward(z)
        ctx.bits = bits
        ctx.lo = lo
        ctx.hi = hi
        ctx.beta = beta
        return _kbit_forward(z, bits, lo, hi)

    @staticmethod
    def backward(  # type: ignore[no-untyped-def]
        ctx, grad_out: Tensor,
    ) -> tuple[Tensor | None, None, None, None, Tensor | None]:
        (z,) = ctx.saved_tensors
        grad_z = grad_out * _kbit_surrogate_grad(z, ctx.bits, ctx.lo, ctx.hi, ctx.beta)
        grad_beta: Tensor | None = None
        if isinstance(ctx.beta, Tensor) and ctx.beta.requires_grad:
            level_step, _, thresholds = _kbit_thresholds(ctx.bits, ctx.lo, ctx.hi)
            acc = torch.zeros_like(z)
            for t_k in thresholds:
                u = ctx.beta * (z - t_k)
                t = torch.tanh(u)
                acc = acc + level_step * (z - t_k) * riccati_tanh_derivative(t, order=1)
            grad_beta = (grad_out * acc).sum()
        return grad_z, None, None, None, grad_beta


def binarize(z: Tensor, beta: float = 10.0) -> Tensor:
    """Hard ``sign(z)`` in ``{-1, +1}`` (``sign(0)=+1``); Riccati surrogate backward."""
    return _BinarizeFn.apply(z, beta)


def binarize01(z: Tensor, beta: float = 10.0) -> Tensor:
    """Hard Heaviside step in ``{0, 1}`` (``H(0)=1``); ``sigmoid(beta z)`` surrogate backward.

    The ``{0, 1}`` codomain twin of :func:`binarize`, using the Eulerian
    ``sigmoid_polynomial_coeffs`` Riccati tower instead of the ``tanh`` one. It is
    affinely conjugate to :func:`binarize`: ``binarize01(z, beta)`` equals
    ``(binarize(z, beta / 2) + 1) / 2`` in both forward and backward.
    """
    return _Binarize01Fn.apply(z, beta)


def heaviside(z: Tensor, beta: float = 10.0) -> Tensor:
    """Alias of :func:`binarize01` (the hard ``{0, 1}`` Heaviside step)."""
    return _Binarize01Fn.apply(z, beta)


def ternarize(z: Tensor, beta: float = 10.0, delta: float = 0.5) -> Tensor:
    """Hard ternary ``{-1, 0, +1}``; smooth ``tanh`` dead-zone surrogate backward."""
    return _TernarizeFn.apply(z, beta, delta)


def kbit_quantize(
    z: Tensor,
    bits: int = 2,
    lo: float = -1.0,
    hi: float = 1.0,
    beta: float = 10.0,
) -> Tensor:
    """Uniform k-bit quantize on ``[lo, hi]``; tanh-step surrogate backward."""
    if bits < 1:
        raise ValueError(f"bits must be >= 1, got {bits}")
    return _KBitQuantizeFn.apply(z, bits, lo, hi, beta)
