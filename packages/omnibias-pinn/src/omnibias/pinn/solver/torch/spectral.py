# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""1-D periodic spectral grid for the method-of-lines spatial discretisation.

Spatial derivatives are FFT-based (**spectral**, exact for band-limited data on
a periodic domain) -- this is the *spatial* discretisation for the MOL drivers
and is labelled ``SPECTRAL``, distinct from the closed-form ``sigma``-tower used
by the mesh-free collocation ansatz. It is a standard, minimal Fourier grid, not
a re-implementation of any ``omnibias-fields`` operator.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor


class SpectralGrid1D:
    """A periodic 1-D grid with FFT differentiation on ``[0, length)``."""

    def __init__(
        self,
        n: int,
        length: float = 2.0 * math.pi,
        *,
        dtype: torch.dtype = torch.float64,
        device: torch.device | None = None,
    ) -> None:
        if n % 2 != 0:
            raise ValueError(f"spectral grid size must be even, got {n}")
        self.n = int(n)
        self.length = float(length)
        self.dtype = dtype
        self.device = device
        modes = torch.fft.fftfreq(self.n, d=1.0 / self.n).to(dtype=dtype, device=device)
        self.k = (2.0 * math.pi / self.length) * modes  # angular wavenumbers
        self.x = (self.length / self.n) * torch.arange(
            self.n, dtype=dtype, device=device
        )

    def points(self) -> Tensor:
        return self.x

    def deriv(self, u: Tensor, order: int) -> Tensor:
        """``d^order u / dx^order`` along the last axis (periodic / spectral)."""
        uh = torch.fft.fft(u)
        factor = (1j * self.k.to(u.device)) ** order
        if order % 2 == 1:
            # zero the (unpaired) Nyquist mode for odd-order derivatives
            factor = factor.clone()
            factor[self.n // 2] = 0
        return torch.fft.ifft(factor * uh).real

    def dx(self, u: Tensor) -> Tensor:
        return self.deriv(u, 1)

    def dxx(self, u: Tensor) -> Tensor:
        return self.deriv(u, 2)


__all__ = ["SpectralGrid1D"]
