# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Non-periodic spectral fractional operators (torch).

The FFT operator in :mod:`omnibias.fractional.torch.ops.fractional`
(:func:`spectral_fractional`) assumes a **periodic** domain. This module adds the
two-sided (symmetric, Riesz-type) spectral fractional Laplacian ``(-Delta)^{alpha/2}``
on a **bounded, non-periodic** interval ``[0, L]``, diagonalised in the sine
(Dirichlet) or cosine (Neumann) basis, plus a windowed-FFT convenience for signals
that merely decay at the ends:

* :func:`spectral_fractional_laplacian` -- orthonormal DST-I (Dirichlet BC) or
  DCT-II (Neumann BC) transform, multiply by ``xi_k^{alpha}`` with
  ``xi_k = k pi / L``, inverse transform. Exact on the basis modes; for
  ``alpha = 2`` it reproduces ``-u''`` on the corresponding grid. Differentiable in
  the order ``alpha`` (through ``xi_k^{alpha} = exp(alpha log xi_k)``).
* :func:`windowed_spectral_fractional` -- Tukey-taper the signal to kill the
  boundary jump, then apply the periodic ``(i k)^{alpha}`` operator; for compactly
  supported / boundary-decaying ``f``.

All ``alpha`` arguments accept a tensor (e.g. an ``nn.Parameter`` or
:class:`~omnibias.fractional.torch.order.LearnableOrder`) so the order is learnable.
Bit-identical to the JAX twin (:mod:`omnibias.fractional.jax.ops.spectral`).
"""

from __future__ import annotations

import math

import torch
from torch import Tensor


def _dst1_matrix(n: int, dtype: torch.dtype, device: torch.device) -> Tensor:
    r"""Orthonormal DST-I matrix ``Q[k,j] = sqrt(2/(n+1)) sin(pi (k+1)(j+1)/(n+1))``.

    Symmetric and orthogonal (``Q = Q^T``, ``Q Q = I``), so it is its own inverse.
    """
    idx = torch.arange(1, n + 1, dtype=dtype, device=device)
    ang = math.pi / (n + 1) * idx.reshape(n, 1) * idx.reshape(1, n)
    return math.sqrt(2.0 / (n + 1)) * torch.sin(ang)


def _dct2_matrix(n: int, dtype: torch.dtype, device: torch.device) -> Tensor:
    r"""Orthonormal DCT-II matrix ``C`` (``C C^T = I``); inverse is ``C^T`` (DCT-III)."""
    k = torch.arange(n, dtype=dtype, device=device).reshape(n, 1)
    j = torch.arange(n, dtype=dtype, device=device).reshape(1, n)
    c = math.sqrt(2.0 / n) * torch.cos(math.pi * (2.0 * j + 1.0) * k / (2.0 * n))
    c[0, :] = c[0, :] / math.sqrt(2.0)
    return c


def _symbol(xi: Tensor, alpha: float | Tensor, *, mask_zero: bool) -> Tensor:
    r"""Spectral symbol ``xi^alpha`` (differentiable in ``alpha``; zero mode masked)."""
    a = alpha if isinstance(alpha, Tensor) else torch.tensor(float(alpha), dtype=xi.dtype)
    a = a.to(dtype=xi.dtype, device=xi.device)
    safe = torch.where(xi > 0, xi, torch.ones_like(xi))
    mult = torch.exp(a * torch.log(safe))
    if mask_zero:
        return torch.where(xi > 0, mult, torch.zeros_like(mult))
    return mult


def spectral_fractional_laplacian(
    f: Tensor,
    *,
    alpha: float | Tensor,
    length: float,
    bc: str = "dirichlet",
) -> Tensor:
    r"""Two-sided spectral fractional Laplacian ``(-Delta)^{alpha/2} f`` on ``[0, L]``.

    Parameters
    ----------
    f
        Samples of ``f`` on the basis grid, shape ``(N,)``. For ``bc="dirichlet"``
        the grid is the interior ``x_j = (j+1) L/(N+1)`` (homogeneous Dirichlet
        ends); for ``bc="neumann"`` it is the midpoint grid ``x_j = (j+1/2) L/N``.
    alpha
        Fractional order (the operator is ``(-Delta)^{alpha/2}``, i.e. spectral
        symbol ``xi_k^{alpha}``). ``alpha = 2`` gives ``-u''``. A tensor ``alpha``
        is differentiable (learnable order).
    length
        Interval length ``L > 0``.
    bc
        ``"dirichlet"`` (DST-I) or ``"neumann"`` (DCT-II).

    Returns
    -------
    Tensor
        A real tensor of shape ``(N,)``. Exact on the basis modes; the result is
        the analytic operator applied to the band-limited interpolant of ``f``.
    """
    if length <= 0.0:
        raise ValueError(f"length must be > 0, got {length}")
    if f.ndim != 1:
        raise ValueError(f"f must be 1-D (samples on a grid), got shape {tuple(f.shape)}")
    n = f.shape[0]
    if bc == "dirichlet":
        q = _dst1_matrix(n, f.dtype, f.device)
        xi = torch.arange(1, n + 1, dtype=f.dtype, device=f.device) * (math.pi / length)
        mult = _symbol(xi, alpha, mask_zero=False)
        return q @ (mult * (q @ f))
    if bc == "neumann":
        c = _dct2_matrix(n, f.dtype, f.device)
        xi = torch.arange(n, dtype=f.dtype, device=f.device) * (math.pi / length)
        mult = _symbol(xi, alpha, mask_zero=True)
        return c.transpose(0, 1) @ (mult * (c @ f))
    raise ValueError(f"bc must be 'dirichlet' or 'neumann', got {bc!r}")


def tukey_window(n: int, taper: float, dtype: torch.dtype, device: torch.device) -> Tensor:
    r"""Tukey (tapered-cosine) window of length ``n`` with edge fraction ``taper``.

    ``taper = 0`` is the rectangular window; ``taper = 1`` is a Hann window. The
    flat middle is ``1`` and each end ramps up/down with a raised cosine.
    """
    if not (0.0 <= taper <= 1.0):
        raise ValueError(f"taper must be in [0, 1], got {taper}")
    if taper == 0.0 or n <= 1:
        return torch.ones(n, dtype=dtype, device=device)
    x = torch.arange(n, dtype=dtype, device=device) / (n - 1)
    edge = taper / 2.0
    w = torch.ones(n, dtype=dtype, device=device)
    lo = x < edge
    hi = x > 1.0 - edge
    w = torch.where(lo, 0.5 * (1.0 + torch.cos(math.pi * (2.0 * x / taper - 1.0))), w)
    w = torch.where(hi, 0.5 * (1.0 + torch.cos(math.pi * (2.0 * x / taper - 2.0 / taper + 1.0))), w)
    return w


def windowed_spectral_fractional(
    f: Tensor,
    *,
    alpha: float | Tensor,
    length: float,
    taper: float = 0.1,
) -> Tensor:
    r"""Windowed-FFT spectral fractional derivative for a non-periodic ``f``.

    Applies a Tukey taper (fraction ``taper``) to suppress the boundary
    discontinuity, then the periodic ``(i k)^{alpha}`` operator
    (:func:`~omnibias.fractional.torch.ops.fractional.spectral_fractional`). Returns
    a complex tensor. Intended for signals that decay toward the interval ends; the
    interior is accurate, the tapered margins are damped by construction.
    """
    from omnibias.fractional.torch.ops.fractional import spectral_fractional

    if f.ndim != 1:
        raise ValueError(f"f must be 1-D (samples on a grid), got shape {tuple(f.shape)}")
    w = tukey_window(f.shape[0], taper, f.dtype, f.device)
    out: Tensor = spectral_fractional(f * w, alpha=alpha, length=length)
    return out


__all__ = [
    "spectral_fractional_laplacian",
    "tukey_window",
    "windowed_spectral_fractional",
]
