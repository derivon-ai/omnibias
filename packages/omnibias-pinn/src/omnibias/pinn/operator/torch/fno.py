# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Fourier Neural Operator (torch) -- in-repo named baseline.

A compact spectral-convolution stack with omnibias activations. Periodicity is
a property of the FFT; derivatives of an FNO output are **FFT-based and
periodic-grid-bound**, so the closed-form trunk-jet claim of the DeepONet path
does **not** transfer here. This module exists both as a capability and as the
named baseline that makes ``docs/benchmarks.md`` §3a reproducible from this
repository.
"""

from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn
from omnibias.torch.activations.registry import ActivationSpec, get_activation
from torch import Tensor


class SpectralConv1d(nn.Module):
    """Complex spectral convolution on the leading ``modes`` Fourier modes.

    Input / output shape ``(batch, channels, grid)``. The weight is a complex
    tensor of shape ``(in_channels, out_channels, modes)``; higher modes are
    zeroed (the truncated FNO spectral multiplier).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes: int,
        *,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        if modes < 1:
            raise ValueError(f"modes must be >= 1, got {modes}")
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.modes = int(modes)
        # Complex weights stored as real pair for dtype portability.
        scale = 1.0 / (in_channels * out_channels)
        self.weight_real = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes, dtype=dtype)
        )
        self.weight_imag = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes, dtype=dtype)
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError(f"SpectralConv1d expects (B, C, N); got {tuple(x.shape)}")
        n = x.shape[-1]
        x_ft = torch.fft.rfft(x, dim=-1)  # (B, C, n//2+1)
        out_ft = torch.zeros(
            x.shape[0],
            self.out_channels,
            x_ft.shape[-1],
            dtype=torch.complex128 if x.dtype == torch.float64 else torch.complex64,
            device=x.device,
        )
        m = min(self.modes, x_ft.shape[-1])
        w = torch.complex(self.weight_real[..., :m], self.weight_imag[..., :m])
        # (B, in, m) x (in, out, m) -> (B, out, m)
        out_ft[:, :, :m] = torch.einsum("bim,iom->bom", x_ft[:, :, :m], w)
        return cast(Tensor, torch.fft.irfft(out_ft, n=n, dim=-1))


class FNO1d(nn.Module):
    """1-D Fourier Neural Operator ``u(·, 0) ↦ u(·, T)`` on a periodic grid.

    Parameters
    ----------
    modes
        Number of Fourier modes retained in each spectral convolution.
    width
        Channel width of the lifted representation.
    n_layers
        Number of spectral-conv + pointwise blocks.
    in_channels, out_channels
        Input / output channel counts (default 1).
    base
        Pointwise activation (omnibias registry name or spec).
    """

    def __init__(
        self,
        *,
        modes: int = 8,
        width: int = 32,
        n_layers: int = 4,
        in_channels: int = 1,
        out_channels: int = 1,
        base: str | ActivationSpec[Tensor] = "gelu",
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        if n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {n_layers}")
        self.modes = int(modes)
        self.width = int(width)
        self.spec = base if isinstance(base, ActivationSpec) else get_activation(base)
        self.lift = nn.Linear(in_channels, width, dtype=dtype)
        self.proj = nn.Linear(width, out_channels, dtype=dtype)
        self.spectral = nn.ModuleList(
            [SpectralConv1d(width, width, modes, dtype=dtype) for _ in range(n_layers)]
        )
        self.pointwise = nn.ModuleList(
            [nn.Linear(width, width, dtype=dtype) for _ in range(n_layers)]
        )

    def forward(self, u0: Tensor) -> Tensor:
        """Map initial condition ``(B, N)`` or ``(B, N, C_in)`` to ``(B, N, C_out)``."""
        if u0.ndim == 2:
            u0 = u0.unsqueeze(-1)
        if u0.ndim != 3:
            raise ValueError(f"u0 must be (B, N) or (B, N, C); got {tuple(u0.shape)}")
        # Lift: (B, N, C) -> (B, N, width) -> (B, width, N) for spectral conv.
        h = self.lift(u0)
        h = h.transpose(1, 2)
        for spec_conv, pw in zip(self.spectral, self.pointwise, strict=True):
            h1 = spec_conv(h)
            h2 = pw(h.transpose(1, 2)).transpose(1, 2)
            h = self.spec.forward(h1 + h2)
        out = self.proj(h.transpose(1, 2))
        return cast(Tensor, out)


def build_fno1d(
    *,
    modes: int = 8,
    width: int = 32,
    n_layers: int = 4,
    in_channels: int = 1,
    out_channels: int = 1,
    base: str | ActivationSpec[Tensor] = "gelu",
    dtype: torch.dtype = torch.float64,
) -> FNO1d:
    """Build a :class:`FNO1d`."""
    return FNO1d(
        modes=modes,
        width=width,
        n_layers=n_layers,
        in_channels=in_channels,
        out_channels=out_channels,
        base=base,
        dtype=dtype,
    )


__all__ = [
    "FNO1d",
    "SpectralConv1d",
    "build_fno1d",
]
