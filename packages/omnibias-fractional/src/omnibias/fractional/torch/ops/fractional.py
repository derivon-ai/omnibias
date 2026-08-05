# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Grid-based fractional derivatives (torch).

.. warning::

   These are non-local numerical approximations on a grid, **not** closed-form
   sigma-tower derivatives. Inputs are sampled function values; outputs are the
   fractional derivative sampled on the same grid. For the package's
   *closed-form* operator (on the analytic-function class) see
   :mod:`omnibias.fractional.torch.ops.analytic`.
"""

from __future__ import annotations

import math

import torch
from omnibias.fractional._core.kernels import gl_matrix, spectral_multiplier
from torch import Tensor


def _gl_weights_backend(alpha: Tensor, n: int) -> Tensor:
    r"""Differentiable Grunwald-Letnikov weights ``w_k = (-1)^k binom(alpha, k)``.

    In-backend twin of :func:`omnibias.fractional._core.kernels.gl_weights`, built
    with the same stable recurrence ``w_0 = 1``, ``w_k = w_{k-1} (1 - (alpha+1)/k)``
    -- a ``cumprod`` that is smooth in ``alpha``, so autograd flows to the order.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    ones = torch.ones(1, dtype=alpha.dtype, device=alpha.device)
    if n == 1:
        return ones
    k = torch.arange(1, n, dtype=alpha.dtype, device=alpha.device)
    factors = 1.0 - (alpha + 1.0) / k
    return torch.cat([ones, torch.cumprod(factors, dim=0)])


def _gl_matmul(f: Tensor, alpha: Tensor, h: float) -> Tensor:
    r"""Grunwald-Letnikov operator applied to ``f`` with a tensor-valued ``alpha``.

    Builds the same lower-triangular Toeplitz operator as
    :func:`omnibias.fractional._core.kernels.gl_matrix` (``mat[i, j] = w_{i-j}``
    for ``j <= i``, else 0) entirely in-backend, so it is differentiable w.r.t.
    ``alpha`` through both the weights and the ``h^{-alpha}`` scaling.
    """
    if h <= 0.0:
        raise ValueError(f"grid spacing h must be > 0, got {h}")
    n = f.shape[0]
    a = alpha.to(dtype=f.dtype, device=f.device)
    w = _gl_weights_backend(a, n)
    idx = torch.arange(n, device=f.device)
    diff = idx[:, None] - idx[None, :]
    zero = torch.zeros((), dtype=f.dtype, device=f.device)
    mat = torch.where(diff >= 0, w[diff.clamp(min=0)], zero)
    return (mat * (h ** (-a))) @ f


def grunwald_letnikov(f: Tensor, *, alpha: float | Tensor, h: float) -> Tensor:
    r"""Grunwald-Letnikov fractional derivative of order ``alpha`` on a grid.

    Parameters
    ----------
    f
        Sampled values ``f(x_0), ..., f(x_{N-1})`` on a uniform grid, shape
        ``(N,)``.
    alpha
        Derivative order (may be fractional; ``alpha >= 0``). A Python ``float``
        uses the fast numpy kernel (unchanged); a **tensor** (e.g. an
        ``nn.Parameter``) takes the in-backend path so the order itself is
        *learnable* -- gradients flow to ``alpha``.
    h
        Uniform grid spacing.
    """
    if isinstance(alpha, Tensor):
        return _gl_matmul(f, alpha, h)
    n = f.shape[0]
    mat = torch.as_tensor(gl_matrix(alpha, n, h), dtype=f.dtype, device=f.device)
    return mat @ f


def riemann_liouville(f: Tensor, *, alpha: float | Tensor, h: float) -> Tensor:
    r"""Riemann-Liouville fractional derivative (Grunwald-Letnikov discretisation)."""
    return grunwald_letnikov(f, alpha=alpha, h=h)


def caputo(f: Tensor, *, alpha: float | Tensor, h: float) -> Tensor:
    r"""Caputo fractional derivative for ``0 < alpha < 1``.

    For ``0 < alpha < 1`` the Caputo derivative equals the Riemann-Liouville
    derivative of ``f(x) - f(0)``; this subtraction removes the boundary term.
    ``alpha`` may be a tensor, in which case the order is learnable.
    """
    a = float(alpha) if isinstance(alpha, Tensor) else alpha
    if not (0.0 < a < 1.0):
        raise ValueError(f"caputo here supports 0 < alpha < 1, got {a}")
    return grunwald_letnikov(f - f[0], alpha=alpha, h=h)


def _spectral_multiplier_backend(alpha: Tensor, n: int, length: float) -> Tensor:
    r"""Differentiable Fourier multiplier ``(i k)^alpha`` (zero mode dropped).

    In-backend twin of
    :func:`omnibias.fractional._core.kernels.spectral_multiplier`, evaluated as
    ``exp(alpha * log(i k))`` so the gradient w.r.t. ``alpha`` is well defined and
    ``nan``-free (the zero mode is masked before the ``log``).
    """
    if length <= 0.0:
        raise ValueError(f"length must be > 0, got {length}")
    kk = torch.fft.fftfreq(n, d=length / n, device=alpha.device, dtype=torch.float64)
    kk = kk * (2.0 * math.pi)
    z = 1j * kk.to(torch.complex128)
    z_safe = torch.where(kk == 0, torch.ones_like(z), z)
    mult = torch.exp(alpha.to(torch.float64) * torch.log(z_safe))
    return torch.where(kk == 0, torch.zeros_like(mult), mult)


def spectral_fractional(f: Tensor, *, alpha: float | Tensor, length: float) -> Tensor:
    r"""Spectral fractional derivative on a periodic domain via the FFT.

    Returns a complex tensor ``ifft((i k)^alpha * fft(f))``. For integer
    ``alpha`` the real part recovers the ordinary derivative; applying order
    ``alpha`` twice equals order ``2 alpha`` (exact for band-limited inputs). A
    tensor ``alpha`` takes the differentiable in-backend path (learnable order).
    """
    n = f.shape[0]
    if isinstance(alpha, Tensor):
        mult = _spectral_multiplier_backend(alpha, n, length)
    else:
        mult = torch.as_tensor(
            spectral_multiplier(alpha, n, length), dtype=torch.complex128, device=f.device,
        )
    fhat = torch.fft.fft(f.to(torch.complex128))
    return torch.fft.ifft(mult * fhat)


__all__ = ["caputo", "grunwald_letnikov", "riemann_liouville", "spectral_fractional"]
