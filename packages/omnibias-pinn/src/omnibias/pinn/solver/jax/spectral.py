# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""1-D periodic spectral grid for the jax method-of-lines (twin of the torch one)."""

from __future__ import annotations

import math
from typing import Any

import jax.numpy as jnp


class SpectralGrid1D:
    """A periodic 1-D grid with FFT differentiation on ``[0, length)``."""

    def __init__(
        self,
        n: int,
        length: float = 2.0 * math.pi,
        *,
        dtype: Any = jnp.float64,
    ) -> None:
        if n % 2 != 0:
            raise ValueError(f"spectral grid size must be even, got {n}")
        self.n = int(n)
        self.length = float(length)
        self.dtype = dtype
        modes = jnp.fft.fftfreq(self.n, d=1.0 / self.n).astype(dtype)
        self.k = (2.0 * math.pi / self.length) * modes
        self.x = (self.length / self.n) * jnp.arange(self.n, dtype=dtype)

    def points(self) -> Any:
        return self.x

    def deriv(self, u: Any, order: int) -> Any:
        uh = jnp.fft.fft(u)
        factor = (1j * self.k) ** order
        if order % 2 == 1:
            factor = factor.at[self.n // 2].set(0)
        return jnp.real(jnp.fft.ifft(factor * uh))

    def dx(self, u: Any) -> Any:
        return self.deriv(u, 1)

    def dxx(self, u: Any) -> Any:
        return self.deriv(u, 2)


__all__ = ["SpectralGrid1D"]
