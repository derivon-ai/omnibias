# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Differentiable special functions of fractional calculus (torch).

The closed-form fractional derivatives of specific activations are expressed
through the classical special functions of the field:

* :func:`mittag_leffler` -- ``E_{alpha,beta}(z) = sum_k z^k / Gamma(alpha k + beta)``,
  the fractional exponential (the eigenfunction of the Caputo derivative);
* :func:`polylog` -- ``Li_s(z) = sum_{k>=1} z^k / k^s``;
* :func:`lerch` -- the Lerch transcendent ``Phi(z, s, a) = sum_{k>=0} z^k / (k+a)^s``;
* :func:`lower_incomplete_gamma` -- ``gamma(s, x) = x^s sum_k (-x)^k / (k! (s+k))``.

Each is a **differentiable truncated series** (autograd flows to ``z`` and to the
orders ``alpha`` / ``s`` / ``beta`` / ``a`` through ``lgamma`` / ``log``), honest
about its truncation: it is the exact special function up to the ``terms`` tail,
accurate inside the series' radius of convergence. The JAX twin
(:mod:`omnibias.fractional.jax.ops.special`) is bit-identical.
"""

from __future__ import annotations

import torch
from torch import Tensor


def _recip_gamma(y: Tensor) -> Tensor:
    r"""Reciprocal gamma ``1 / Gamma(y)`` (entire; zero at non-positive integers).

    ``exp(-lgamma(y))`` with an explicit sign restoring ``sign(Gamma(y))`` on the
    negative axis; at a pole ``lgamma = +inf`` so the value collapses to ``0`` --
    exactly ``1/Gamma`` at the non-positive integers. Shared verbatim with the JAX
    twin (torch has no ``torch.special.gamma``).
    """
    log_mag = -torch.lgamma(y)
    parity = torch.remainder(torch.ceil(-y), 2.0)
    sign = torch.where(y > 0, torch.ones_like(y), 1.0 - 2.0 * parity)
    return sign * torch.exp(log_mag)


def _as_scalar(value: float | Tensor, ref: Tensor) -> Tensor:
    if isinstance(value, Tensor):
        return value.to(dtype=ref.dtype, device=ref.device)
    return torch.tensor(float(value), dtype=ref.dtype, device=ref.device)


def _int_powers(z: Tensor, terms: int, *, start_one: bool) -> Tensor:
    """Stack ``z^0..z^{terms-1}`` (``start_one``) or ``z^1..z^{terms}`` along a new axis 0.

    Built by ``cumprod`` (never ``pow`` with a possibly-negative base), so it is
    exact for negative ``z`` and differentiable.
    """
    zr = z.unsqueeze(0).expand((terms if not start_one else terms - 1, *z.shape))
    cp = torch.cumprod(zr, dim=0) if zr.shape[0] > 0 else zr
    if start_one:
        ones = torch.ones((1, *z.shape), dtype=z.dtype, device=z.device)
        return torch.cat([ones, cp], dim=0)
    return cp


def mittag_leffler(
    z: Tensor | float,
    alpha: float | Tensor,
    beta: float | Tensor = 1.0,
    *,
    terms: int = 64,
) -> Tensor:
    r"""Two-parameter Mittag-Leffler ``E_{alpha,beta}(z) = sum_k z^k / Gamma(alpha k + beta)``.

    ``E_{1,1}(z) = e^z``, ``E_{2,1}(z^2) = cosh(z)``, ``E_{1,2}(z) = (e^z - 1)/z``.
    Differentiable in ``z``, ``alpha`` and ``beta``.
    """
    if terms < 1:
        raise ValueError("terms must be >= 1")
    z_t = torch.as_tensor(z, dtype=torch.get_default_dtype()) if not isinstance(z, Tensor) else z
    a = _as_scalar(alpha, z_t)
    b = _as_scalar(beta, z_t)
    k = torch.arange(terms, dtype=z_t.dtype, device=z_t.device)
    args = a * k + b
    rg = _recip_gamma(args).reshape((terms,) + (1,) * z_t.ndim)
    zk = _int_powers(z_t, terms, start_one=True)
    out: Tensor = (zk * rg).sum(dim=0)
    return out


def polylog(s: float | Tensor, z: Tensor | float, *, terms: int = 64) -> Tensor:
    r"""Polylogarithm ``Li_s(z) = sum_{k>=1} z^k / k^s`` (``|z| < 1``).

    ``Li_1(z) = -log(1 - z)``. Differentiable in ``s`` (via ``exp(-s log k)``) and ``z``.
    """
    if terms < 1:
        raise ValueError("terms must be >= 1")
    z_t = torch.as_tensor(z, dtype=torch.get_default_dtype()) if not isinstance(z, Tensor) else z
    s_t = _as_scalar(s, z_t)
    k = torch.arange(1, terms + 1, dtype=z_t.dtype, device=z_t.device)
    ks = torch.exp(-s_t * torch.log(k)).reshape((terms,) + (1,) * z_t.ndim)
    zk = _int_powers(z_t, terms, start_one=False)  # z^1..z^terms
    out: Tensor = (zk * ks).sum(dim=0)
    return out


def lerch(
    z: Tensor | float,
    s: float | Tensor,
    a: float | Tensor,
    *,
    terms: int = 64,
) -> Tensor:
    r"""Lerch transcendent ``Phi(z, s, a) = sum_{k>=0} z^k / (k + a)^s`` (``a > 0``).

    ``Phi(z, 1, 1) = -log(1 - z) / z``; ``Phi(1, s, 1) = zeta(s)``. Differentiable in
    ``z``, ``s`` and ``a``.
    """
    if terms < 1:
        raise ValueError("terms must be >= 1")
    z_t = torch.as_tensor(z, dtype=torch.get_default_dtype()) if not isinstance(z, Tensor) else z
    s_t = _as_scalar(s, z_t)
    a_t = _as_scalar(a, z_t)
    k = torch.arange(terms, dtype=z_t.dtype, device=z_t.device)
    ka = k + a_t
    denom = torch.exp(s_t * torch.log(ka)).reshape((terms,) + (1,) * z_t.ndim)
    zk = _int_powers(z_t, terms, start_one=True)
    out: Tensor = (zk / denom).sum(dim=0)
    return out


def lower_incomplete_gamma(
    s: float | Tensor,
    x: Tensor | float,
    *,
    terms: int = 64,
) -> Tensor:
    r"""Lower incomplete gamma ``gamma(s, x) = x^s sum_k (-x)^k / (k! (s + k))`` (``x >= 0``).

    ``gamma(1, x) = 1 - e^{-x}``. The regularised ``P(s, x) = gamma(s, x) / Gamma(s)``
    is the CDF of a Gamma(s) law. Differentiable in ``s`` and ``x``.
    """
    if terms < 1:
        raise ValueError("terms must be >= 1")
    x_t = torch.as_tensor(x, dtype=torch.get_default_dtype()) if not isinstance(x, Tensor) else x
    s_t = _as_scalar(s, x_t)
    k = torch.arange(terms, dtype=x_t.dtype, device=x_t.device)
    kfact = torch.lgamma(k + 1.0)
    denom = (torch.exp(kfact).reshape((terms,) + (1,) * x_t.ndim)) * (s_t + k).reshape(
        (terms,) + (1,) * x_t.ndim
    )
    mxk = _int_powers(-x_t, terms, start_one=True)  # (-x)^0..(-x)^{terms-1}
    series = (mxk / denom).sum(dim=0)
    out: Tensor = torch.exp(s_t * torch.log(x_t)) * series
    return out


__all__ = [
    "lerch",
    "lower_incomplete_gamma",
    "mittag_leffler",
    "polylog",
]
