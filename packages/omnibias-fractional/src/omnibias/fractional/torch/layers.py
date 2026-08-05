# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Trainable ``nn.Module`` fractional layers (torch).

Thin wrappers that turn the operator functions in
:mod:`omnibias.fractional.torch.ops` into network layers with a (optionally
learnable) order. Each layer maps sampled function values ``f`` (shape ``(N,)``)
to the fractional derivative sampled on the same grid, and -- when
``learnable_order=True`` -- carries a :class:`~omnibias.fractional.torch.order.LearnableOrder`
so the order ``alpha`` trains end-to-end (gradients flow through the operator's
differentiable in-backend path).

* :class:`GrunwaldLetnikovLayer` -- grid GL / Riemann-Liouville operator.
* :class:`SpectralFractionalLayer` -- periodic FFT operator (real part by default).
* :class:`SpectralFractionalLaplacianLayer` -- two-sided ``(-Delta)^{alpha/2}`` on a
  bounded interval (DST-I / DCT-II).
"""

from __future__ import annotations

import torch
from omnibias.fractional.torch.ops.fractional import grunwald_letnikov, spectral_fractional
from omnibias.fractional.torch.ops.spectral import spectral_fractional_laplacian
from omnibias.fractional.torch.order import LearnableOrder
from torch import Tensor, nn


class _FractionalLayer(nn.Module):
    r"""Shared (optionally learnable) order handling for the fractional layers."""

    _order: Tensor

    def __init__(
        self,
        *,
        order: float = 0.5,
        learnable_order: bool = True,
        lo: float = 0.0,
        hi: float = 2.0,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self._learnable = bool(learnable_order)
        if learnable_order:
            self.order_module = LearnableOrder(init=order, lo=lo, hi=hi)
        else:
            dt = dtype if dtype is not None else torch.get_default_dtype()
            self.register_buffer("_order", torch.tensor(float(order), dtype=dt))

    @property
    def alpha(self) -> Tensor:
        """The current fractional order as a tensor (differentiable if learnable)."""
        a: Tensor = self.order_module() if self._learnable else self._order
        return a


class GrunwaldLetnikovLayer(_FractionalLayer):
    r"""Grid Grunwald-Letnikov / Riemann-Liouville fractional-derivative layer.

    ``forward(f)`` returns ``grunwald_letnikov(f, alpha, h)`` on the uniform grid of
    spacing ``h``.
    """

    def __init__(
        self,
        *,
        h: float,
        order: float = 0.5,
        learnable_order: bool = True,
        lo: float = 0.0,
        hi: float = 2.0,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(order=order, learnable_order=learnable_order, lo=lo, hi=hi, dtype=dtype)
        if h <= 0.0:
            raise ValueError(f"grid spacing h must be > 0, got {h}")
        self.h = float(h)

    def forward(self, f: Tensor) -> Tensor:
        out: Tensor = grunwald_letnikov(f, alpha=self.alpha, h=self.h)
        return out

    def extra_repr(self) -> str:
        return f"h={self.h}, learnable_order={self._learnable}"


class SpectralFractionalLayer(_FractionalLayer):
    r"""Periodic FFT fractional-derivative layer (returns the real part by default)."""

    def __init__(
        self,
        *,
        length: float,
        order: float = 1.0,
        learnable_order: bool = True,
        lo: float = 0.0,
        hi: float = 2.0,
        real: bool = True,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(order=order, learnable_order=learnable_order, lo=lo, hi=hi, dtype=dtype)
        if length <= 0.0:
            raise ValueError(f"length must be > 0, got {length}")
        self.length = float(length)
        self.real = bool(real)

    def forward(self, f: Tensor) -> Tensor:
        out: Tensor = spectral_fractional(f, alpha=self.alpha, length=self.length)
        return out.real if self.real else out

    def extra_repr(self) -> str:
        return f"length={self.length}, real={self.real}, learnable_order={self._learnable}"


class SpectralFractionalLaplacianLayer(_FractionalLayer):
    r"""Two-sided spectral fractional Laplacian ``(-Delta)^{alpha/2}`` layer (DST/DCT)."""

    def __init__(
        self,
        *,
        length: float,
        bc: str = "dirichlet",
        order: float = 1.0,
        learnable_order: bool = True,
        lo: float = 0.0,
        hi: float = 2.0,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(order=order, learnable_order=learnable_order, lo=lo, hi=hi, dtype=dtype)
        if length <= 0.0:
            raise ValueError(f"length must be > 0, got {length}")
        if bc not in ("dirichlet", "neumann"):
            raise ValueError(f"bc must be 'dirichlet' or 'neumann', got {bc!r}")
        self.length = float(length)
        self.bc = str(bc)

    def forward(self, f: Tensor) -> Tensor:
        out: Tensor = spectral_fractional_laplacian(
            f, alpha=self.alpha, length=self.length, bc=self.bc
        )
        return out

    def extra_repr(self) -> str:
        return f"length={self.length}, bc={self.bc}, learnable_order={self._learnable}"


__all__ = [
    "GrunwaldLetnikovLayer",
    "SpectralFractionalLaplacianLayer",
    "SpectralFractionalLayer",
]
