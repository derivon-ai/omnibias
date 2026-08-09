# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Dissipative CCF residual (torch twin)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from omnibias.pinn.torch.equations.ccf_compactified import (
    ccf_compactified_residual_samples,
)
from omnibias.pinn.torch.equations.cordoba_cordoba_fontelos import ccf_residual_samples
from torch import Tensor


def spectral_fractional_laplacian_1d(
    values: Tensor,
    *,
    alpha: Tensor | float,
    length: float,
) -> Tensor:
    x = torch.as_tensor(values)
    a = torch.as_tensor(alpha, dtype=x.dtype, device=x.device)
    if float(torch.abs(a)) < 1e-14:
        return torch.zeros_like(x)
    n = x.shape[-1]
    k = torch.fft.fftfreq(n, d=1.0) * n
    omega = torch.abs(2.0 * torch.pi * k / float(length))
    mult = torch.pow(omega + 0.0, a)
    mult = mult.clone()
    mult[0] = 0.0
    return torch.fft.ifft(torch.fft.fft(x) * mult).real


def ccf_dissipative_residual_samples(
    y: Tensor,
    theta: Tensor,
    theta_y: Tensor,
    lam: Tensor | float,
    alpha: Tensor | float,
    *,
    form: str = "transport",
    velocity_sign: float = 1.0,
    domain: str = "periodic",
    length: float | None = None,
) -> Tensor:
    y = torch.as_tensor(y)
    theta = torch.as_tensor(theta)
    if domain == "periodic":
        base = ccf_residual_samples(
            y, theta, theta_y, lam, form=form, velocity_sign=velocity_sign
        )
        L = float(length) if length is not None else float(y[-1] - y[0] + (y[1] - y[0]))
        return base + spectral_fractional_laplacian_1d(theta, alpha=alpha, length=L)
    if domain == "line_compactified":
        eq, _, _ = ccf_compactified_residual_samples(
            y, theta, theta_y, lam, form=form, velocity_sign=velocity_sign
        )
        order = torch.argsort(y)
        y_s = y[order]
        th_s = theta[order]
        n = int(y_s.numel())
        y_u = torch.linspace(float(y_s[0]), float(y_s[-1]), n, dtype=y.dtype, device=y.device)
        # linear interp
        idx = torch.searchsorted(y_s, y_u, right=True).clamp(1, n - 1)
        x0, x1 = y_s[idx - 1], y_s[idx]
        t0, t1 = th_s[idx - 1], th_s[idx]
        w = (y_u - x0) / torch.clamp(x1 - x0, min=1e-30)
        th_u = t0 + w * (t1 - t0)
        L = float(y_u[-1] - y_u[0])
        diss_u = spectral_fractional_laplacian_1d(th_u, alpha=alpha, length=max(L, 1e-12))
        idx2 = torch.searchsorted(y_u, y_s, right=True).clamp(1, n - 1)
        u0, u1 = y_u[idx2 - 1], y_u[idx2]
        d0, d1 = diss_u[idx2 - 1], diss_u[idx2]
        w2 = (y_s - u0) / torch.clamp(u1 - u0, min=1e-30)
        diss_s = d0 + w2 * (d1 - d0)
        diss = torch.empty_like(theta)
        diss[order] = diss_s
        return eq + diss
    raise ValueError(f"unknown domain {domain!r}")


@dataclass
class CordobaCordobaFontelosDissipative:
    lam: float = 0.6057
    alpha: float = 0.0
    form: str = "transport"
    velocity_sign: float = 1.0
    domain: str = "periodic"

    def residual(
        self,
        y: Tensor,
        theta: Tensor,
        theta_y: Tensor,
        *,
        alpha: Tensor | float | None = None,
    ) -> Tensor:
        a = self.alpha if alpha is None else alpha
        return ccf_dissipative_residual_samples(
            y, theta, theta_y, self.lam, a,
            form=self.form, velocity_sign=self.velocity_sign, domain=self.domain,
        )


__all__ = [
    "CordobaCordobaFontelosDissipative",
    "ccf_dissipative_residual_samples",
    "spectral_fractional_laplacian_1d",
]
